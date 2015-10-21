import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.linalg as linalg

from . import bsplines
from . import covariance

from numpy import dot
from scipy.linalg import inv, solve

from .prob import MultivariateNormal
from .prob import GaussianProcess


class LinearRegression:
    def __init__(self, prior):
        self.prior = prior
        self._is_fit = False

    def fit(self, X, y, C=None):
        n, _ = X.shape
        if C is None:
            C = np.eye(n)

        m0 = self.prior.mean
        V0 = self.prior.covariance
        P0 = inv(V0)
        
        A = dot(X.T, solve(C, X)) + P0
        b = dot(X.T, solve(C, y)) + solve(V0, m0)
        w = solve(A, b)

        self._coef = w
        self._is_fit = True

        return self

    def predict(self, X):
        w = self._coef
        y = dot(X, w)
        return y


class GaussianProcessRegression:
    def __init__(self, prior, noise=covariance.DiagonalCovariance(1.0)):
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
        self._observed = x, y
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
        return self

    def plot(self, x_new, ax=None):
        if ax is None:
            ax = plt.gca()

        x, y = self._observed
        ax.plot(x, y, 'x', label='Observed')

        y_new = self.mean(x_new)
        y_var = np.diag(self.covariance(x_new))
        y_std = np.sqrt(y_var)
        ax.plot(x_new, y_new, label='Predicted')
        ax.plot(x_new, y_new - y_std, '-', label='Predicted - Sigma')
        ax.plot(x_new, y_new + y_std, '-', label='Predicted + Sigma')

