"""
Perform sequential backward selection on the initial feature list to find a list of minimal features. 
"""
import sys
import os
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

lr = LogisticRegression()
df = pd.read_csv("features.csv", index_col=0)

# cross product terms on a reduced subset
excluded_columns = ["id", "label"]
columns = list(set(df.columns) - set(excluded_columns))
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
sfs1 = SFS(lr, k_features=1, forward=False, floating=False,
           verbose=2, scoring='roc_auc', cv=10, n_jobs=-1)
sfs1.fit(X_train, y_train)

# assemble backward selection results for plotting
ks = list(sfs1.subsets_.keys())
ks.sort()
marks = []
for k in ks:
    marks.append([k, sfs1.subsets_[k]["avg_score"]])

marks = np.asarray(marks)
# produce net trend plot and the zoom in on the final feature list.
fig, ax = plt.subplots(1, 2)
ax[0].plot(marks[:, 0], marks[:, 1], marker='o')
ax[1].plot(marks[:, 0][:20], marks[:, 1][:20], marker='o', markersize=4)
ax[0].set_ylabel("Avg. Score")
ax[1].set_ylabel("Avg. Score")
ax[0].set_xlabel("N features [backward selection]")
ax[1].set_xlabel("N features [backward selection]")
best_selection = 8  # by manual inspection of the plot, and DS choice.
ax[1].scatter([best_selection], [marks[marks[:, 0] == best_selection][0][1]],
              s=50, facecolors='none', edgecolors='r')
fig.tight_layout()
fig.savefig("ml_sbs_avg_score_products.png")

fig.show()
final_columns = [columns[i]
                 for i in sfs1.subsets_[best_selection]["feature_idx"]]
# save final columns for training
dg = pd.DataFrame.from_dict({'final_columns': final_columns})
dg.to_csv("final_columns.csv")
