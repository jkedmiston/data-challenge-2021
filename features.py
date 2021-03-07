import math
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TheilSenRegressor
import scipy.stats
import numpy as np


class DataReducer:
    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__


class FFTData(DataReducer):
    def __init__(self, values):
        DataReducer.__init__(self)
        N = len(values)
        yf = abs(np.fft.fft(values))
        yf2 = (2./N) * np.abs(yf[0:N//2])
        self.data = yf2


class FFTDataAboveIndex(DataReducer):
    def __init__(self, fftdataobj, index=1):
        DataReducer.__init__(self)
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

    def __repr__(self):
        return self.__class__.__name__


class RegressionResidualDataSquared(DataReducer):
    def __init__(self, x, y):
        DataReducer.__init__(self)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
            x, y)
        residual = y - (slope * x + intercept)
        self.data = residual**2


class TheilSenRegressorData(DataReducer):
    def __init__(self, x, y):
        DataReducer.__init__(self)
        X = np.zeros((len(x), 1))
        X[:, 0] = x
        reg = TheilSenRegressor(random_state=0)
        reg.fit(X, y)
        slope, intercept = reg.coef_[0], reg.intercept_
        residual = y - (slope * x + intercept)
        self.data = residual


class RANSACResidualData(DataReducer):
    def __init__(self, x, y):
        DataReducer.__init__(self)

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


class RANSACResidualDataSquared(DataReducer):
    def __init__(self, x, y):
        DataReducer.__init__(self)

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


class WeightedFFTData(FFTData):
    def __init__(self, values):
        FFTData.__init__(self, values)
        self.data = self.data[1:] * np.arange(1, len(self.data[1:])+1, 1)


class LabeledFeature:
    def __init__(self, func):
        if hasattr(func, "name"):
            self.funcname = func.name
        else:
            self.funcname = func.__name__

        self.func = func

    def apply_as_info(self, feature):
        return self.funcname + f'__{str(feature)}', self.func(feature.data)


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


def maxrange(x):
    return np.max(x) - np.min(x)
