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

import pandas as pd
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.calibration import CalibratedClassifierCV


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
df = pd.read_csv("features.csv", index_col=0)
excluded_columns = ["id", "label"]
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
ConfusionMatrixDisplay(cm0).plot()
raise
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
