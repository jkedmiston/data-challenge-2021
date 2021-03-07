"""
Prepares features for the ML application from the raw time series. 
* Also produces some plots of the raw time series for feature hypothesis generation
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats
from features import (RegressionResidualData,
                      RegressionResidualDataSquared,
                      TheilSenRegressorData,
                      RANSACResidualData,
                      RANSACResidualDataSquared,
                      WeightedFFTData,
                      UpperTailFraction,
                      LowerTailFraction,
                      FFTData,
                      FFTDataAboveIndex,
                      LabeledFeature,
                      maxrange)


def date_increment_calc(dg):
    """
    renormalize timestamps to the start of the date range
    """
    t0 = dg.iloc[0]["date"]
    out = (dg["date"] - t0).apply(lambda x: x.total_seconds())
    return pd.DataFrame.from_dict({'date_increment': out, 'index': dg.index})


def data_pull():
    """
    Creates raw data frame
    """
    df = pd.read_csv("challenge-data.csv")
    labels = pd.read_csv("challenge-labels.csv")
    df = pd.merge(df, labels, "left", left_on="id", right_on="id")
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values(by=["id", "date"], ascending=[
        True, True]).reset_index(drop=True)
    tmp = df.groupby("id", as_index=False).apply(date_increment_calc)
    df = pd.concat([tmp, df], axis=1)
    return df


df = data_pull()

# exploration plot, to get an idea of the time series of the different
# datasets.
fig, ax = plt.subplots(1, 2)
grps = df.groupby(["id", "label"])
ct = [0, 0]  # instance counter for the different groups
for i, g in enumerate(grps.groups.keys()):
    dg = grps.get_group(g)
    if g[1] == True:
        if ct[0] > 20:
            continue
        ax[1].plot(dg["date_increment"], -2 * ct[0] + (dg["value"] -
                                                       dg["value"].median()) / (dg["value"].max() - dg["value"].min()))
        ct[0] += 1
    else:
        if ct[1] > 20:
            continue

        ax[0].plot(dg["date_increment"], -2 * ct[1] + (dg["value"] -
                                                       dg["value"].median()) / (dg["value"].max() - dg["value"].min()))
        ct[1] += 1
    if ct[0] > 20 and ct[1] > 20:
        break

ax[0].set_title('label=False')
ax[1].set_title('label=True')
ax[0].axes.get_xaxis().set_ticks([])
ax[0].axes.get_yaxis().set_ticks([])
ax[1].axes.get_xaxis().set_ticks([])
ax[1].axes.get_yaxis().set_ticks([])
fig.tight_layout()
fig.savefig("exploration.png")
fig.show()


def fft_based_features(dg):
    print("working on FFT based feature for %s" % dg.name)
    f = FFTData(dg["value"].values)
    fft_index = FFTDataAboveIndex(f, 1)  # exclude f0
    weighted_fft = WeightedFFTData(dg["value"].values)
    functions_to_apply = [np.std,
                          np.mean,
                          scipy.stats.kurtosis,
                          maxrange,
                          np.median,
                          UpperTailFraction(0.2),
                          UpperTailFraction(0.1),
                          UpperTailFraction(0.05),
                          UpperTailFraction(0.01),
                          LowerTailFraction(0.01),
                          LowerTailFraction(0.05),
                          LowerTailFraction(0.1),
                          LowerTailFraction(0.2)]

    datasets = [weighted_fft,
                fft_index]
    infos = []
    for func in functions_to_apply:
        for dataset in datasets:
            tmp = LabeledFeature(func)
            infos.append(tmp.apply_as_info(dataset))

    retval = []
    labels = []
    for label, value in infos:
        retval.append(value)
        labels.append(label)

    return pd.Series(retval, index=labels)


def regression_based_features(dg):
    """
    Features based on finding a linear fit to the curve with 
    varying sensitivity to outliers, then operating on the residual as
    a data distribution with respect to the fit. 
    """
    print("working on regression based feature for %s" % dg.name)
    x = dg["date_increment"].values
    y = dg["value"].values
    reg = RegressionResidualData(x, y)
    ransac_reg = RANSACResidualData(x, y)
    ransac_reg_squared = RANSACResidualDataSquared(x, y)
    theil_sen_reg = TheilSenRegressorData(x, y)
    functions_to_apply = [maxrange,
                          np.std,
                          np.mean,
                          np.median,
                          scipy.stats.skew,
                          scipy.stats.kurtosis,
                          UpperTailFraction(0.1, True),
                          UpperTailFraction(0.05, True),
                          UpperTailFraction(0.01, True),
                          UpperTailFraction(0.25, True),
                          LowerTailFraction(0.1, True),
                          LowerTailFraction(0.25, True),
                          LowerTailFraction(0.05, True),
                          LowerTailFraction(0.01, True)]
    datasets = [reg,
                ransac_reg,
                ransac_reg_squared,
                theil_sen_reg]

    infos = []
    for func in functions_to_apply:
        for dataset in datasets:
            tmp = LabeledFeature(func)
            infos.append(tmp.apply_as_info(dataset))

    retval = []
    labels = []
    for label, value in infos:
        retval.append(value)
        labels.append(label)

    return pd.Series(retval, index=labels)


# apply features in turn
fft_out = df.groupby("id", as_index=False).apply(fft_based_features)
regression_out = df.groupby("id", as_index=False).apply(
    regression_based_features)

# add labels, join all results, save to file for next stage.
labels = pd.read_csv("challenge-labels.csv")
all_features = pd.merge(fft_out, labels, "left", left_on="id", right_on="id")
all_features = pd.merge(all_features, regression_out,
                        "left", left_on="id", right_on="id")
all_features["label"] = all_features["label"].astype(int)
filename = "features.csv"
all_features.to_csv(filename)
print(f"saved features in {filename}")
