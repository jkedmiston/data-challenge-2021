"""
Perform sequential backward selection on the initial feature list 
"""
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sys
import pandas as pd
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
import warnings
import os
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses


def logistic_adjust_threshold(model, x, decision_threshold=0.5):
    """
    Quick and dirty adjustment of the decision threshold for binary classification
    """
    arr = model.predict_proba(x)[:, 1]
    y = np.zeros(len(x))
    y[arr > decision_threshold] = 1
    return y


lr = LogisticRegression()
# lr = RandomForestClassifier()
# lr = CalibratedClassifierCV(base_estimator=lr, cv=3)
df = pd.read_csv("out1.csv", index_col=0)
# cross product terms
excluded_columns = ["id", "label"]
columns = list(set(df.columns) - set(excluded_columns))
original_columns = columns[:]
columns = ['UpperTailFraction_sort_True_cutoff_0.10__TheilSenRegressorData',
           'LowerTailFraction_sort_False_cutoff_0.05__WeightedFFTData',
           'UpperTailFraction_sort_True_cutoff_0.25__RANSACResidualData',
           'LowerTailFraction_sort_False_cutoff_0.20__WeightedFFTData',
           'LowerTailFraction_sort_True_cutoff_0.05__TheilSenRegressorData',
           'LowerTailFraction_sort_True_cutoff_0.25__TheilSenRegressorData',
           'LowerTailFraction_sort_True_cutoff_0.01__RegressionResidualData',
           'UpperTailFraction_sort_True_cutoff_0.01__TheilSenRegressorData']

for i, c in enumerate(columns):
    for j, c2 in enumerate(columns[i+1:]):
        df["%s*%s" % (c, c2)] = df[c] * df[c2]
        df["%s/%s" % (c, c2)] = df[c] / df[c2]

columns = list(set(df.columns) - set(excluded_columns))
X = df[columns]
s = StandardScaler()
X = s.fit_transform(X)
y = df["label"]


def get_cv_scores(model, X_train, y_train):
    scores = cross_val_score(model,
                             X_train,
                             y_train,
                             cv=5,
                             scoring='roc_auc')

    print('CV Mean: ', np.mean(scores))
    print('STD: ', np.std(scores))
    print('\n')
    return {'mean': np.mean(scores), 'std': np.std(scores)}


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
# ['LowerTailFraction_sort_True_cutoff_0.05__TheilSenRegressorData', 'UpperTailFraction_sort_True_cutoff_0.01__TheilSenRegressorData', 'LowerTailFraction_sort_False_cutoff_0.20__WeightedFFTData']
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
sfs1 = SFS(lr, k_features=1, forward=False, floating=False,
           verbose=2, scoring='roc_auc', cv=10, n_jobs=-1)

sfs1.fit(X_train, y_train)
Xcols = [columns[c] for c in sfs1.k_feature_idx_]
print(Xcols, len(Xcols))
X_train = X_train[:, sfs1.k_feature_idx_]
X_test = X_test[:, sfs1.k_feature_idx_]
lr.fit(X_train, y_train)


cv_score_info = get_cv_scores(lr, X_train, y_train)
print(cv_score_info)
sc_train = lr.score(X_train, y_train)
sc_test = lr.score(X_test, y_test)


pipe_lr = make_pipeline(lr)
train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr,
                                                        X=X_train,
                                                        y=y_train,
                                                        train_sizes=np.linspace(
                                                            0.3, 1.0, 10),
                                                        cv=6,
                                                        scoring='roc_auc',
                                                        n_jobs=1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
fig, ax = plt.subplots(1, 1)
ax.plot(train_sizes, train_mean,
        color='blue', marker='o',
        markersize=5, label='Training')

ax.plot(train_sizes, test_mean,
        color='green', linestyle='--',
        marker='s', markersize=5,
        label='Validation')

ax.set_xlabel('Number of training examples')
ax.set_ylabel('Accuracy')
ax.legend(loc='lower right')
fig.show()

y_pred = lr.predict(X_test)
cm0 = confusion_matrix(y_test, y_pred)
# ConfusionMatrixDisplay(cm).plot()
# https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_display_object_visualization.html#sphx-glr-auto-examples-miscellaneous-plot-display-object-visualization-py
# y_score = lr.decision_function(X_test)
y_score = lr.predict_proba(X_test)[:, 1]
print(sc_train, sc_test)

roc_score = roc_auc_score(y_test, y_score)
print(roc_score)


ds = np.linspace(0, 1, 300)
vals = []
for d in ds:
    y = logistic_adjust_threshold(lr, X_test, decision_threshold=d)
    cm = confusion_matrix(y_test, y)
    vals.append([d] + [*cm.flatten()])

vals = pd.DataFrame.from_records(
    vals, columns=["d", "c00", "c01", "c10", "c11"])
f, a = plt.subplots(1, 1)
a.plot(vals["d"], vals["c11"] / (vals["c11"] + vals["c01"]),
       label="precision")  # false positive
a.plot(vals["d"], vals["c11"] / (vals["c11"] + vals["c10"]),
       label="recall")  # false negative
a.plot(vals["d"], 1 - vals["c01"] / (vals["c01"] + vals["c00"]),
       label="specificity")  # false negative
a.plot(vals["d"], .5 * ((1 - vals["c01"] / (vals["c01"] + vals["c00"])) + (vals["c11"] / (vals["c11"] + vals["c10"]))),
       label="balanced accuracy")  # false negative

vals["specificity"] = 1 - vals["c01"] / (vals["c01"] + vals["c00"])
vals["recall"] = vals["c11"] / (vals["c11"] + vals["c10"])
vals["balanced_accuracy"] = (vals["specificity"] + vals["recall"])*.5
vals["fpr"] = (vals["c01"] / (vals["c01"] + vals["c00"]))
vals["tpr"] = (vals["c11"] / (vals["c11"] + vals["c10"]))


result = vals[vals["balanced_accuracy"] ==
              np.max(vals["balanced_accuracy"])].iloc[0]


class RandomChance:
    def __init__(self, threshold):
        self.threshold = threshold

    def predict(self, X):
        arr = np.random.random(len(X))
        threshold = self.threshold
        z = np.zeros(len(arr))
        z[arr <= threshold] = 1
        z[arr > threshold] = 0
        return z


class LRAdjust:
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def predict(self, x):
        return logistic_adjust_threshold(self.model, x, self.threshold)


# a.plot(vals[:, 0], vals[:, 3], label="c10")
# a.plot(vals[:, 0], vals[:, 4], label="c11")
# https://www.datascienceblog.net/post/machine-learning/specificity-vs-precision/
a.set_xlabel("decision threshold")
a.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

f.tight_layout()
f.show()
#  0.387960
l = LRAdjust(lr, 0.387960)
y_pred_adjust = l.predict(X_test)

np.random.seed(1)
rc = RandomChance(774.0 / 2000)
y_random = rc.predict(X_test)

cm = confusion_matrix(y_test, y_pred_adjust)
tn, fp, fn, tp = cm.flatten()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
print(cm, precision, recall)


cm = confusion_matrix(y_test, y_random)
tn, fp, fn, tp = cm.flatten()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
print(cm, precision, recall)


fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)

# fpr, tpr, _ = roc_curve(y_test, y_pred)
f, a = plt.subplots(1, 1)
a.plot(fpr, tpr)
a.scatter([result["fpr"]], [result["tpr"]], s=40,
          facecolors='none', edgecolors='r')
a.set_xlabel("FPR")
a.set_ylabel("TPR")
f.show()


ks = list(sfs1.subsets_.keys())
ks.sort()
marks = []
for k in ks:
    marks.append([k, sfs1.subsets_[k]["avg_score"]])

marks = np.asarray(marks)
f, a = plt.subplots(1, 2)
a[0].plot(marks[:, 0], marks[:, 1], marker='o')
a[1].plot(marks[:, 0][:20], marks[:, 1][:20], marker='o', markersize=4)
a[0].set_ylabel("Avg. Score")
a[1].set_ylabel("Avg. Score")
a[0].set_xlabel("N features [backward selection]")
a[1].set_xlabel("N features [backward selection]")
a[1].scatter([8], [marks[marks[:, 0] == 8][0][1]],
             s=50, facecolors='none', edgecolors='r')
f.tight_layout()
f.savefig("avg_score_products.png")
f.show()
sfs1.subsets_[25]
[columns[i] for i in sfs1.subsets_[25]["feature_idx"]]
columns = ['UpperTailFraction_sort_False_cutoff_0.10__FFTDataAboveIndex_idx1', 'LowerTailFraction_sort_False_cutoff_0.05__WeightedFFTData/LowerTailFraction_sort_False_cutoff_0.20__WeightedFFTData', 'UpperTailFraction_sort_False_cutoff_0.20__WeightedFFTData', 'UpperTailFraction_sort_True_cutoff_0.25__RegressionResidualData', 'std__RegressionResidualData', 'LowerTailFraction_sort_False_cutoff_0.05__WeightedFFTData*UpperTailFraction_sort_True_cutoff_0.25__RANSACResidualData', 'LowerTailFraction_sort_True_cutoff_0.25__RegressionResidualData', 'LowerTailFraction_sort_False_cutoff_0.20__WeightedFFTData/LowerTailFraction_sort_True_cutoff_0.01__RegressionResidualData', 'LowerTailFraction_sort_True_cutoff_0.05__TheilSenRegressorData*LowerTailFraction_sort_True_cutoff_0.25__TheilSenRegressorData', 'UpperTailFraction_sort_False_cutoff_0.01__FFTDataAboveIndex_idx1', 'UpperTailFraction_sort_False_cutoff_0.01__WeightedFFTData', 'LowerTailFraction_sort_False_cutoff_0.20__WeightedFFTData*LowerTailFraction_sort_True_cutoff_0.01__RegressionResidualData',
           'skew__RANSACResidualDataSquared', 'skew__RegressionResidualData', 'skew__TheilSenRegressorData', 'kurtosis__RANSACResidualData', 'UpperTailFraction_sort_True_cutoff_0.10__TheilSenRegressorData/UpperTailFraction_sort_True_cutoff_0.25__RANSACResidualData', 'UpperTailFraction_sort_True_cutoff_0.25__RANSACResidualData/LowerTailFraction_sort_False_cutoff_0.20__WeightedFFTData', 'UpperTailFraction_sort_True_cutoff_0.10__TheilSenRegressorData*LowerTailFraction_sort_True_cutoff_0.05__TheilSenRegressorData', 'LowerTailFraction_sort_True_cutoff_0.01__RegressionResidualData', 'UpperTailFraction_sort_True_cutoff_0.10__TheilSenRegressorData*UpperTailFraction_sort_True_cutoff_0.01__TheilSenRegressorData', 'median__TheilSenRegressorData', 'UpperTailFraction_sort_True_cutoff_0.25__RANSACResidualData*LowerTailFraction_sort_True_cutoff_0.01__RegressionResidualData', 'UpperTailFraction_sort_False_cutoff_0.20__FFTDataAboveIndex_idx1', 'LowerTailFraction_sort_False_cutoff_0.20__WeightedFFTData*LowerTailFraction_sort_True_cutoff_0.05__TheilSenRegressorData']
# 7
col7 = ['LowerTailFraction_sort_False_cutoff_0.20__WeightedFFTData/LowerTailFraction_sort_True_cutoff_0.01__RegressionResidualData', 'LowerTailFraction_sort_True_cutoff_0.05__TheilSenRegressorData*LowerTailFraction_sort_True_cutoff_0.25__TheilSenRegressorData', 'skew__RegressionResidualData', 'UpperTailFraction_sort_True_cutoff_0.10__TheilSenRegressorData/UpperTailFraction_sort_True_cutoff_0.25__RANSACResidualData',
        'LowerTailFraction_sort_True_cutoff_0.01__RegressionResidualData', 'UpperTailFraction_sort_True_cutoff_0.10__TheilSenRegressorData*UpperTailFraction_sort_True_cutoff_0.01__TheilSenRegressorData', 'LowerTailFraction_sort_False_cutoff_0.20__WeightedFFTData*LowerTailFraction_sort_True_cutoff_0.05__TheilSenRegressorData']  # 0.872
