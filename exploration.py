import math
import matplotlib
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats


def date_increment_calc(dg):
    t0 = dg.iloc[0]["date"]
    out = (dg["date"] - t0).apply(lambda x: x.total_seconds())
    return pd.DataFrame.from_dict({'date_increment': out, 'index': dg.index})

# data munge


def data_pull():
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


class DataReducer:
    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__


class FFTData:
    def __init__(self, values):
        N = len(values)
        yf = abs(np.fft.fft(values))
        yf2 = (2./N) * np.abs(yf[0:N//2])
        self.data = yf2

    def __repr__(self):
        return self.__class__.__name__


class FFTDataAboveIndex:
    def __init__(self, fftdataobj, index=1):
        assert isinstance(fftdataobj, FFTData)
        self.data = fftdataobj.data[index:]
        self.index = index

    def __repr__(self):
        return self.__class__.__name__ + f'_idx{self.index}'


class RegressionResidualData(DataReducer):
    def __init__(self, x, y):
        DataReducer.__init__(self)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
            x, y)
        residual = y - (slope * x + intercept)
        self.data = residual
        self.slope = slope
        self.intercept = intercept
        self.r_value = r_value
        self.p_value = p_value
        self.std_err = std_err

    def __repr__(self):
        return self.__class__.__name__


class RegressionResidualDataSquared(DataReducer):
    def __init__(self, x, y):
        DataReducer.__init__(self)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
            x, y)
        residual = y - (slope * x + intercept)
        self.data = residual**2

    def __repr__(self):
        return self.__class__.__name__


class TheilSenRegressorData(DataReducer):
    def __init__(self, x, y):
        DataReducer.__init__(self)
        from sklearn.linear_model import TheilSenRegressor
        X = np.zeros((len(x), 1))
        X[:, 0] = x
        reg = TheilSenRegressor(random_state=0)
        reg.fit(X, y)
        slope, intercept = reg.coef_[0], reg.intercept_
        residual = y - (slope * x + intercept)
        self.data = residual
        self.slope = slope
        self.intercept = intercept

    def __repr__(self):
        return self.__class__.__name__


class RANSACResidualData(DataReducer):
    def __init__(self, x, y):
        DataReducer.__init__(self)
        from sklearn.linear_model import RANSACRegressor, LinearRegression
        X = np.zeros((len(x), 1))
        X[:, 0] = x
        try:
            reg = RANSACRegressor(random_state=0).fit(X, y)
        except ValueError:
            reg = LinearRegression().fit(X, y)
            setattr(reg, "estimator_", reg)

        slope, intercept = reg.estimator_.coef_[0], reg.estimator_.intercept_
        residual = y - (slope * x + intercept)
        self.data = residual
        self.slope = slope
        self.intercept = intercept

    def __repr__(self):
        return self.__class__.__name__


class RANSACResidualDataSquared(DataReducer):
    def __init__(self, x, y):
        DataReducer.__init__(self)
        from sklearn.linear_model import RANSACRegressor, LinearRegression
        X = np.zeros((len(x), 1))
        X[:, 0] = x
        try:
            reg = RANSACRegressor(random_state=0).fit(X, y)
        except ValueError:
            reg = LinearRegression().fit(X, y)
            setattr(reg, "estimator_", reg)

        slope, intercept = reg.estimator_.coef_[0], reg.estimator_.intercept_
        residual = y - (slope * x + intercept)
        self.data = residual ** 2
        self.slope = slope
        self.intercept = intercept

    def __repr__(self):
        return self.__class__.__name__


class WeightedFFTData(FFTData):
    def __init__(self, values):
        FFTData.__init__(self, values)
        self.data = self.data[1:] * np.arange(1, len(self.data[1:])+1, 1)

    def __repr__(self):
        return self.__class__.__name__


class LabeledFeature:
    def __init__(self, func):
        if hasattr(func, "name"):
            self.funcname = func.name
        else:
            self.funcname = func.__name__

        self.func = func

    def apply_as_info(self, feature):
        return self.funcname + f'__{str(feature)}', self.func(feature.data)


def maxrange(x):
    return np.max(x) - np.min(x)


def fft_based_features(dg):
    f = FFTData(dg["value"].values)
    g = FFTDataAboveIndex(f, 1)
    wf = WeightedFFTData(dg["value"].values)

    functions_to_apply = [np.std, np.mean,
                          scipy.stats.kurtosis, maxrange, np.median,
                          UpperTailFraction(0.2),
                          UpperTailFraction(0.1),
                          UpperTailFraction(0.05),
                          UpperTailFraction(0.01),
                          LowerTailFraction(0.01),
                          LowerTailFraction(0.05),
                          LowerTailFraction(0.1),
                          LowerTailFraction(0.2)]

    datasets = [wf, g]

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


class UpperTailFraction:
    def __init__(self, fraction, sort=False):
        self.name = f"{self.__class__.__name__}_sort_{sort}_cutoff_{fraction:.2f}"
        self.fraction = fraction
        self.sort = sort

    def __call__(self, data):
        if self.sort:
            data = np.sort(data)

        x = np.abs(data)
        idx = math.ceil(len(x) * self.fraction)
        return x[-idx:].sum() / x.sum()


class LowerTailFraction:
    def __init__(self, fraction, sort=False):
        self.name = f"{self.__class__.__name__}_sort_{sort}_cutoff_{fraction:.2f}"
        self.fraction = fraction
        self.sort = sort

    def __call__(self, data):
        if self.sort:
            data = np.sort(data)

        x = np.abs(data)
        idx = math.ceil(len(x) * self.fraction)
        return x[:idx].sum() / x.sum()


def regression_based_features(dg):
    x = dg["date_increment"].values
    y = dg["value"].values
    reg = RegressionResidualData(x, y)
    reg2 = RANSACResidualData(x, y)
    reg3 = RANSACResidualDataSquared(x, y)
    reg4 = TheilSenRegressorData(x, y)
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
    datasets = [reg, reg2, reg3, reg4]
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


# second derivative
# fit regression line and pull out anything outlier
out = df.groupby("id", as_index=False).apply(fft_based_features)
rout = df.groupby("id", as_index=False).apply(regression_based_features)
out = pd.merge(out, labels, "left", left_on="id", right_on="id")
out = pd.merge(out, rout, "left", left_on="id", right_on="id")
out["label"] = out["label"].astype(int)
out.to_csv("out1.csv")

"""
X = df[global_features]
y = df["ground_truth_label"]

X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X, y, df, test_size=0.2, random_state=1)
X_train_ = poly.fit_transform(X_train)
lr.fit(X_train_, y_train)


# "mean__FFTDataAboveIndex_idx1"
fig, ax = plt.subplots(1, 1)
ax.scatter(out["label"], out["maxrange__RANSACResidualData"])
fig.show()
"""
