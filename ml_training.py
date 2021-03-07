import pdb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import cross_val_score
import numpy as np


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


def logistic_adjust_threshold(model, x, decision_threshold=0.5):
    """
    Quick and dirty adjustment of the decision threshold for binary classification
    """
    arr = model.predict_proba(x)[:, 1]
    y = np.zeros(len(x))
    y[arr > decision_threshold] = 1
    return y


lr = LogisticRegression()
df = pd.read_csv("features.csv", index_col=0)


mult_columns = ['UpperTailFraction_sort_True_cutoff_0.10__TheilSenRegressorData',
                'LowerTailFraction_sort_False_cutoff_0.05__WeightedFFTData',
                'UpperTailFraction_sort_True_cutoff_0.25__RANSACResidualData',
                'LowerTailFraction_sort_False_cutoff_0.20__WeightedFFTData',
                'LowerTailFraction_sort_True_cutoff_0.05__TheilSenRegressorData',
                'LowerTailFraction_sort_True_cutoff_0.25__TheilSenRegressorData',
                'LowerTailFraction_sort_True_cutoff_0.01__RegressionResidualData',
                'UpperTailFraction_sort_True_cutoff_0.01__TheilSenRegressorData']

# apply new features, then trim
for i, c in enumerate(mult_columns):
    for j, c2 in enumerate(mult_columns[i+1:]):
        df["%s*%s" % (c, c2)] = df[c] * df[c2]
        df["%s/%s" % (c, c2)] = df[c] / df[c2]

# final featurelist, from ml_sbs.py
final_columns = ['LowerTailFraction_sort_False_cutoff_0.20__WeightedFFTData/LowerTailFraction_sort_True_cutoff_0.01__RegressionResidualData',
                 'LowerTailFraction_sort_True_cutoff_0.05__TheilSenRegressorData*LowerTailFraction_sort_True_cutoff_0.25__TheilSenRegressorData',
                 'skew__RegressionResidualData',
                 'UpperTailFraction_sort_True_cutoff_0.10__TheilSenRegressorData/UpperTailFraction_sort_True_cutoff_0.25__RANSACResidualData',
                 'LowerTailFraction_sort_True_cutoff_0.01__RegressionResidualData',
                 'UpperTailFraction_sort_True_cutoff_0.10__TheilSenRegressorData*UpperTailFraction_sort_True_cutoff_0.01__TheilSenRegressorData',
                 'LowerTailFraction_sort_False_cutoff_0.20__WeightedFFTData*LowerTailFraction_sort_True_cutoff_0.05__TheilSenRegressorData']
dk = pd.read_csv("final_columns.csv", index_col=0)
final_columns = dk["final_columns"].values
# .872 auc

X = df[final_columns]
s = StandardScaler()
X = s.fit_transform(X)
y = df["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
lr.fit(X_train, y_train)

# Learning curve
pipe_lr = make_pipeline(lr)
train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr,
                                                        X=X_train,
                                                        y=y_train,
                                                        train_sizes=np.linspace(
                                                            0.1, 1.0, 20),
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
ax.set_ylabel('ROC AUC')
ax.legend(loc='lower right')
fig.savefig("learning.png")
fig.show()

# evaluate base performance
y_pred = lr.predict(X_test)
cm0 = confusion_matrix(y_test, y_pred)
obj = ConfusionMatrixDisplay(cm0).plot()
obj.figure_.savefig("cm0.png")

# https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_display_object_visualization.html#sphx-glr-auto-examples-miscellaneous-plot-display-object-visualization-py
y_score = lr.predict_proba(X_test)[:, 1]
roc_score = roc_auc_score(y_test, y_score)
print("ROC score %s" % roc_score)
tmp = pd.DataFrame.from_dict({"roc_auc": [roc_score]})
tmp.to_csv("best_roc_auc_score.csv")

# find decision threshold
ds = np.linspace(0, 1, 300)
vals = []
for d in ds:
    y = logistic_adjust_threshold(lr, X_test, decision_threshold=d)
    cm = confusion_matrix(y_test, y)
    vals.append([d] + [*cm.flatten()])

vals = pd.DataFrame.from_records(
    vals, columns=["d", "c00", "c01", "c10", "c11"])
vals["precision"] = vals["c11"] / (vals["c11"] + vals["c01"])
vals["specificity"] = 1 - vals["c01"] / (vals["c01"] + vals["c00"])
vals["recall"] = vals["c11"] / (vals["c11"] + vals["c10"])
vals["balanced_accuracy"] = (vals["specificity"] + vals["recall"])*.5
vals["fpr"] = (vals["c01"] / (vals["c01"] + vals["c00"]))
vals["tpr"] = (vals["c11"] / (vals["c11"] + vals["c10"]))

fig, ax = plt.subplots(1, 1)
ax.plot(vals["d"], vals["precision"],
        label="precision")
ax.plot(vals["d"], vals["recall"],
        label="recall")
ax.plot(vals["d"], vals["specificity"],
        label="specificity")
ax.plot(vals["d"], vals["balanced_accuracy"],
        label="balanced accuracy")
best_result = vals[vals["balanced_accuracy"] ==
                   np.max(vals["balanced_accuracy"])].iloc[0]
best_result.to_csv("best_decision_threshold.csv")
ax.scatter([best_result["d"]],
           [best_result["balanced_accuracy"]],
           s=50,
           facecolors='none',
           edgecolors='r')
ax.set_xlabel("Decision threshold")
ax.set_ylabel("Score")
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
fig.tight_layout()
fig.savefig("performance.png")
fig.show()

# now that we have the decision threshold, we get the final model


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
    """
    Wraps around model with decision threshold adjusted
    """

    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def predict(self, x):
        return logistic_adjust_threshold(self.model, x, self.threshold)


lr_adjust = LRAdjust(lr, best_result["d"])
y_pred_adjust = lr_adjust.predict(X_test)

np.random.seed(1)
rc = RandomChance(774.0 / 2000)
y_random = rc.predict(X_test)

cm = confusion_matrix(y_test, y_pred_adjust)
tn, fp, fn, tp = cm.flatten()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
print("Adjusted model", cm, precision, recall)
obj = ConfusionMatrixDisplay(cm).plot()
obj.figure_.savefig("cm_tuned.png")

# pdb.set_trace()

cm = confusion_matrix(y_test, y_random)
tn, fp, fn, tp = cm.flatten()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
obj = ConfusionMatrixDisplay(cm).plot()
obj.figure_.savefig("cm_random.png")
print("Random model", cm, precision, recall)

fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)

fig, ax = plt.subplots(1, 1)
ax.plot(fpr, tpr)
ax.scatter([best_result["fpr"]],
           [best_result["tpr"]],
           s=40,
           facecolors='none',
           edgecolors='r')
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
fig.tight_layout()
fig.savefig("ROC.png")
fig.show()
