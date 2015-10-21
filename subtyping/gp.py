import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from numpy import dot
from scipy.linalg import inv, solve

from ..covariance import DiagonalCovariance


class GaussianProcess:
    def __init__(self, mean, covariance):
        self.mean = mean
        self.covariance = covariance

    def sample(self, x, size=1):
        m = self.mean(x)
        V = self.covariance(x)
        f = stats.multivariate_normal.rvs(m, V, size=size)
        return f


class GaussianProcessRegression:
    def __init__(self, prior, noise=DiagonalCovariance(1.0)):
        self.prior = prior
        self.noise = noise
        self._is_fit = False

    @property
    def mean(self):
        if self._is_fit:
            return self.posterior.mean
        else:
            return self.prior.mean

    @property
    def covariance(self):
        if self._is_fit:
            return self.posterior.covariance
        else:
            return self.prior.covariance

    def fit(self, x, y):
        r = y - self.prior.mean(x)
        K = self.prior.covariance(x) + self.noise(x)
        
        def mean(x_new):
            C = self.prior.covariance(x_new, x)
            m = self.prior.mean(x_new)
            m += dot(C, solve(K, r))
            return m

        def covariance(x_new):
            C = self.prior.covariance(x_new, x)
            V = self.prior.covariance(x_new)
            V -= dot(C, solve(K, C.T))
            return V

        self.posterior = GaussianProcess(mean, covariance)
        self._is_fit = True
        self._observed = x, y        
        return self

    def plot(self, x_new, ax=None):
        if ax is None:
            ax = plt.gca()

        if self._is_fit:
            x, y = self._observed
            ax.plot(x, y, 'x', label='Observed')

        y_new = self.mean(x_new)
        y_var = np.diag(self.covariance(x_new))
        y_std = np.sqrt(y_var)
        ax.plot(x_new, y_new, label='Predicted')
        ax.plot(x_new, y_new - 2 * y_std, '-', label=r'Predicted - $2 \sigma$')
        ax.plot(x_new, y_new + 2 * y_std, '-', label=r'Predicted + $2 \sigma$')

