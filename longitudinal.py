import numpy as np


class LongitudinalSequence:
    """A sequence of measurements recorded over time for a single
    individual.

    identifier: a unique key for the individual (should be hashable)
    times: a 1D array of measurement times
    values: a 1D array of measurement values
    features: a 1D array of individual-specific characteristics encoded as a
        feature vector
   
    """
    __slots__ = ['identifier', 'times', 'values', 'features']

    def __init__(self, identifier, times, values, features):
        self.identifier = identifier
        self.times = times
        self.values = values
        self.features = features

    def __len__(self):
        return len(self.times)

    def __hash__(self):
        return hash(self.identifier)

    def __repr__(self):
        s = 'LongitudinalSequence(id={}, num_obs={})'
        return s.format(self.identifier, self.times.size)

    def make_dummy(self, new_times):
        """Create a dummy sequence of empty observation times."""
        return LongitudinalSequence(
            self.identifier, new_times, None, self.features)

    def prefix(self, upper):
        ix = self.times <= upper
        return self._subseq(ix)

    def suffix(self, lower):
        ix = self.times > lower
        return self._subseq(ix)

    def _subseq(self, ix):
        i = self.identifier
        x = self.times[ix]
        y = self.values[ix]
        f = self.features
        return LongitudinalSequence(i, x, y, f)        

    @classmethod
    def from_tables(cls, measurements, characteristics=None):
        has_char = characteristics is not None
        m = dict(list(measurements.groupby(level=0)))
        if has_char:
            c = dict(list(characteristics.groupby(level=0)))
            assert set(m).issubset(set(c))

        sequences = []
        for key in m:
            identifier = key
            times = m[key].iloc[:, 0].values
            values = m[key].iloc[:, 1].values
            if has_char:
                features = c[key].values.ravel()
            else:
                features = np.zeros(1)
            seq = cls(identifier, times, values, features)
            sequences.append(seq)

        return sequences
