import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import splev


class BSplineBasis:
    """An encoder mapping from a scalar to a basis representation.

    """
    def __init__(self, knots, degree):
        """Initializes the parameters of the B-spline basis.

        Arguments
        ---------
        knots: knots of the basis, first and last elements should be
            the boundaries.
        degree: the degree of the polynomial pieces

        """
        self.low = knots[0]
        self.high = knots[-1]
        self._iknots = knots[1:-1]
        self.degree = degree

        lo = [self.low] * self.order
        hi = [self.high] * self.order
        self._knots = np.r_[lo, self._iknots, hi]

    @property
    def order(self):
        """One plus the degree of the polynomial pieces."""
        return self.degree + 1

    @property
    def interior_knots(self):
        """The interior knots of the basis."""
        return self._iknots.copy()

    @property
    def knots(self):
        """The complete knot sequence (boundary and interior)."""
        return self._knots.copy()

    @property
    def dimension(self):
        """The number of basis functions in the representation."""
        return len(self.knots) - self.order

    @property
    def tck(self):
        """The `tck` representation of the basis.

        t: the complete sequence of knots
        c: the basis coefficients
        k: the degree of the polynomial pieces

        """
        t = self.knots
        c = np.eye(self.dimension)
        k = self.degree
        return (t, c, k)

    @classmethod
    def with_dimension(cls, dimension, low, high, degree):
        """Creates a B-spline basis with a fixed number of bases.

        Arguments
        ---------
        dimension: number of bases
        low/high: lower/upper bound on scalar values that can be encoded
        degree: degree of the polynomial pieces

        """
        num_interior_knots = dimension - degree - 1
        knots = np.linspace(low, high, num_interior_knots + 2)
        return cls(knots, degree)

    def __call__(self, x, der=0):
        """Encode values in the basis representation.

        Returns
        -------
        A matrix wherein each row contains the encoded corresponding
        value in `x`.

        """
        f = splev(x, self.tck, der=der)
        B = np.asarray(f).T
        return B

    def smoothness_penalty(self, der=2, dt=1e-2):
        lo = self.low
        hi = self.high

        N = np.ceil((hi - lo) / dt)
        dt = (hi -lo) / N
        x = np.linspace(lo, hi, N)
        X = self(x, der)
        
        W = dt * np.eye(N)
        P = np.dot(X.T, np.dot(W, X))
        return P

    def fused_penalty(self, der=1):
        D = np.eye(self.dimension)
        D = np.diff(D, der)
        P = np.dot(D, D.T)
        return P

    def plot(self, ndt=200, ax=None):
        ax = plt.gca() if ax is None else ax
        x = np.linspace(self.low, self.high, ndt)
        B = self(x).T
        for b in B:
            ax.plot(x, b)
