import numpy as np
import scipy.linalg as linalg

from numpy import dot
from scipy.linalg import inv, solve

from .util import rowvec, colvec


class AbstractCovariance:
    def __call__(self, x1, x2=None):
        return self._auto(x1) if x2 is None else self._cross(x1, x2)


class DiagonalCovariance(AbstractCovariance):
    def __init__(self, v):
        self.v = v

    def _auto(self, x):
        n = x.size
        C = self.v * np.eye(n)
        return C

    def _cross(self, x1, x2):
        C = np.zeros((x1.size, x2.size))
        return C


class ConstantCovariance(AbstractCovariance):
    def __init__(self, v):
        self.v = v

    def _auto(self, x):
        n = x.size
        C = self.v * np.ones((n, n))
        return C

    def _cross(self, x1, x2):
        C = self.v * np.ones((x1.size, x2.size))
        return C


class BasisCovariance(AbstractCovariance):
    def __init__(self, basis, covariance):
        self.basis = basis
        self.covariance = covariance

    def _auto(self, x):
        X = self.basis(x)
        V = self.covariance
        C = dot(X, dot(V, X.T))
        return C

    def _cross(self, x1, x2):
        X1 = self.basis(x1)
        X2 = self.basis(x2)
        V = self.covariance
        C = dot(X1, dot(V, X2.T))
        return C


class BSplineCovariance(BasisCovariance):
    def __init__(self, basis, v, der=1):
        self.basis = basis
        precision = basis.smoothness_penalty(der=der, dt=1e-2)
        l, U = linalg.eigh(precision)
        L = np.diag(l[1:])
        U = U[:, 1:]
        self.covariance = v * dot(U, solve(L, U.T))


class OUCovariance(AbstractCovariance):
    def __init__(self, v, l):
        self.v = v
        self.l = l

    def _auto(self, x):
        d = self._abs_diff(x, x)
        C = self.v * np.exp(- d / self.l)
        return C

    def _cross(self, x1, x2):
        d = self._abs_diff(x1, x2)
        C = self.v * np.exp(- d / self.l)
        return C

    def _abs_diff(self, x1, x2):
        x1 = colvec(x1)
        x2 = rowvec(x2)
        return np.abs(x1 - x2)
