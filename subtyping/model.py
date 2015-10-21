"""Learn subtype trajectories from longitudinal data."""

import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.stats as stats

from scipy.misc import logsumexp

from .. import covariance
from ..bsplines import BSplineBasis
from ..longitudinal import LongitudinalSequence
from ..util import random_distribution, memoize_callable

from .gp import GaussianProcess, GaussianProcessRegression


class SubtypeMixtureModel:
    def __init__(self, log_likelihood, prior, pop_coef, sub_coef):
        self._log_likelihood = log_likelihood        
        self._prior = np.array(prior)
        self._pop_coef = np.array(pop_coef)
        self._sub_coef = np.array(sub_coef)

    @property
    def num_subtypes(self):
        return len(self._prior)

    @property
    def prior(self):
        return self._prior.copy()

    @prior.setter
    def prior(self, v):
        self._prior[:] = v

    @property
    def pop_basis(self):
        return self._log_likelihood.pop_basis

    @property
    def pop_coef(self):
        return self._pop_coef.copy()

    @pop_coef.setter
    def pop_coef(self, v):
        self._pop_coef[:] = v

    @property
    def sub_basis(self):
        return self._log_likelihood.sub_basis

    @property
    def sub_coef(self):
        return self._sub_coef.copy()

    @sub_coef.setter
    def sub_coef(self, v):
        self._sub_coef[:, :] = v

    @property
    def covariance(self):
        return self._log_likelihood.covariance

    def log_prior(self, seq):
        return np.log(self.prior)

    def log_likelihood(self, seq):
        wpop = self.pop_coef
        wsub = self.sub_coef        
        if len(seq) > 0:
            return [self._log_likelihood(seq, wpop, w) for w in wsub]
        else:
            return 0.0

    def log_prob(self, seq):
        lp = self.log_prior(seq)
        ll = self.log_likelihood(seq)
        return lp + ll

    def inference(self, seq):
        log_joint = self.log_prob(seq)
        log_evidence = logsumexp(log_joint)
        posterior = np.exp(log_joint - log_evidence)
        return posterior, log_evidence

    def bic(self, sequences):
        inferences = [self.inference(s) for s in sequences]
        LL = sum(inf[1] for inf in inferences)
        #n = sum(len(s) for s in sequences)
        n = len(sequences)
        k = self.prior.size + self.pop_coef.size + self.sub_coef.size
        return float(-2 * LL + k * np.log(n))

    def predictive(self, seq, levels='const ou', subtype=None):
        if subtype is None:
            posterior, _ = self.inference(seq)
            subtype = np.argmax(posterior)

        def mean(x):
            s = seq.make_dummy(x)
            X1 = self.pop_basis(s)
            X2 = self.sub_basis(s)
            m = np.dot(X1, self.pop_coef)
            m += np.dot(X2, self.sub_coef[subtype])
            return m

        levels = set(levels.split())
        prior_levels = levels & self.covariance.levels
        if not prior_levels:
            raise ValueError('No specified levels found in covariance.')
        
        noise_levels = self.covariance.levels - levels
        if not noise_levels:
            raise ValueError('No levels in covariance left for noise.')
        
        prior_cov = [getattr(self.covariance, n) for n in prior_levels]
        noise_cov = [getattr(self.covariance, n) for n in noise_levels]

        def covariance(x1, x2=None):
            return sum(cov(x1, x2) for cov in prior_cov)

        def noise(x1, x2=None):
            return sum(cov(x1, x2) for cov in noise_cov)

        gp = GaussianProcess(mean, covariance)
        gpr = GaussianProcessRegression(gp, noise)
        if len(seq) > 0:
            gpr.fit(seq.times, seq.values)

        return gpr

    def plot_subtypes(self, ax=None):
        if ax is None:
            ax = plt.gca()

        lo = self.sub_basis.low
        hi = self.sub_basis.high
        x = np.linspace(lo, hi, 200)
        X = self.sub_basis(x)
        for i, w in enumerate(self.sub_coef):
            ax.plot(x, np.dot(X, w), label='Subtype {}'.format(i + 1))

    def to_dict(self):
        d = {}
        d['sub_basis'] = self.sub_basis.to_dict()
        d['covariance'] = self.covariance.to_dict()
        d['prior'] = self.prior.tolist()
        d['pop_coef'] = self.pop_coef.tolist()
        d['sub_coef'] = self.sub_coef.tolist()
        return d

    def to_json(self, filename):
        d = self.to_dict()
        with open(filename, 'w') as f:
            json.dump(d, f, indent=4)

    @classmethod
    def from_dict(cls, d):
        pop_basis = PopulationBasis()
        sub_basis = SubpopulationBasis.with_dimension(**d['sub_basis'])
        covariance = Covariance(**d['covariance'])
        log_likelihood = LogLikelihood(pop_basis, sub_basis, covariance)
        
        prior = np.array(d['prior'])
        pop_coef = np.array(d['pop_coef'])
        sub_coef = np.array(d['sub_coef'])
        return cls(log_likelihood, prior, pop_coef, sub_coef)

    @classmethod
    def from_json(cls, filename):
        with open(filename, 'r') as f:
            model = cls.from_dict(json.load(f))
        return model

    
class LogLikelihood:
    def __init__(self, pop_basis, sub_basis, covariance):
        self.pop_basis = pop_basis        
        self.sub_basis = sub_basis
        self.covariance = covariance

    def __call__(self, seq, pop_coef, sub_coef):
        Xpop = self.pop_basis(seq)
        Xsub = self.sub_basis(seq)
        m = Xpop.dot(pop_coef) + Xsub.dot(sub_coef)
        C = self.covariance(seq)
        return stats.multivariate_normal.logpdf(seq.values, m, C)


@memoize_callable
class PopulationBasis:
    def __call__(self, seq):
        x = seq.times
        f = seq.features
        X = np.zeros((len(x), len(f)))
        X[:, :] = f
        return X


@memoize_callable
class SubpopulationBasis(BSplineBasis):
    def __call__(self, seq, *args):
        if isinstance(seq, LongitudinalSequence):
            x = seq.times
        else:
            x = seq
        return super().__call__(x, *args)

    def to_dict(self):
        return {'dimension': int(self.dimension),
                'low': float(self.low),
                'high': float(self.high),
                'degree': int(self.degree)}


@memoize_callable
class Covariance:
    def __init__(self, v_const, v_ou, l_ou, v_noise=1.0):
        self.const = covariance.ConstantCovariance(v_const)
        self.ou = covariance.OUCovariance(v_ou, l_ou)
        self.noise = covariance.DiagonalCovariance(v_noise)
        self.levels = {'const', 'ou', 'noise'}

    def __call__(self, seq):
        x = seq.times
        C = self.const(x) + self.ou(x) + self.noise(x)
        return C

    def to_dict(self):
        return {'v_const': float(self.const.v),
                'v_ou': float(self.ou.v),
                'l_ou': float(self.ou.l),
                'v_noise': float(self.noise.v)}
