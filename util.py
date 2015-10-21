import numpy as np
import collections


def rowvec(x):
    """Casts an array as a row vector.

    Notes
    -----
    The data is not copied and is shared with the original array.

    """
    x = x.ravel()
    return x[None, :]


def colvec(x):
    """Casts an array as a column vector.

    Notes
    -----
    The data is not copied and is shared with the original array.

    """
    x = x.ravel()
    return x[:, None]


def random_distribution(k, state=None):
    """Randomly initializes a discrete distribution.

    Arguments
    ---------
    k: the number of outcomes

    """
    rnd = np.random if state is None else state
    p = rnd.uniform(size=k)
    p /= p.sum()
    return p


def memoize_callable(cls):
    """Cache results of a callable class.

    This function is designed to work as a decorator for class
    definitions. The call signature for the class can only contain
    positional arguments, and each must be hashable in order for the
    look-up to work.

    """
    class Memoized(cls):
        def __init__(self, *args, **kwargs):
            self._cache = {}
            super().__init__(*args, **kwargs)

        def __call__(self, *args):
            if any(not hashable(arg) for arg in args):
                return super().__call__(*args)
            
            key = tuple(args)
            if key not in self._cache:
                self._cache[key] = super().__call__(*args)
            return self._cache[key]

    return Memoized


def hashable(x):
    return isinstance(x, collections.Hashable)
