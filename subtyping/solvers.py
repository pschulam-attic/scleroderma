import numpy as np

from numpy import dot
from scipy.linalg import inv, solve


class Solver:
    def __init__(self, basis, covariance, penalty=None):
        self.basis = basis
        self.covariance = covariance
        self.penalty = 0.0 if penalty is None else penalty

    def solve_coef(self, xs, ys, ws=None):
        Xs = [self.basis(x) for x in xs]
        Cs = [self.covariance(x) for x in xs]
        ss1, ss2 = self._compute_suffstats(Xs, Cs, ys, ws)
        return solve(ss1 + self.penalty, ss2)

    def _compute_suffstats(self, Xs, Cs, ys, ws):
        ss1 = [dot(X.T, solve(C, X)) for X, C in zip(Xs, Cs)]
        ss1 = sum(ss1) if ws is None else _weighted_sum(ss1, ws)
        ss2 = [dot(X.T, solve(C, y)) for X, C, y in zip(Xs, Cs, ys)]
        ss2 = sum(ss2) if ws is None else _weighted_sum(ss2, ws)
        return ss1, ss2


class OrthogonalSolver(Solver):
    def __init__(self, basis, covariance, orth_basis, penalty=None):
        self.basis = basis
        self.covariance = covariance
        self.orth_basis = orth_basis
        self.penalty = 0.0 if penalty is None else penalty

    def solve_coef(self, xs, ys, ws=None):
        Xs = [self.basis(x) for x in xs]
        Cs = [self.covariance(x) for x in xs]

        Xs = self._orthogonalize(xs, Xs)
        ys = self._orthogonalize(xs, ys)

        ss1, ss2 = self._compute_suffstats(Xs, Cs, ys, ws)
        return solve(ss1 + self.penalty, ss2)

    def _orthogonalize(self, xs, ys):
        Xs = [self.orth_basis(x) for x in xs]
        Cs = [self.covariance(x) for x in xs]
        A = sum(dot(X.T, solve(C, X)) for X, C in zip(Xs, Cs))
        A += 1e-4 * np.eye(len(A))
        b = sum(dot(X.T, solve(C, y)) for X, C, y in zip(Xs, Cs, ys))
        w = solve(A, b)
        ys_proj = [dot(X, w) for X in Xs]
        return [(y - yhat) for y, yhat in zip(ys, ys_proj)]


def _weighted_sum(xs, ws):
    return sum(w * x for w, x in zip(ws, xs))
