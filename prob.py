import numpy as np
import scipy.stats as stats


class GaussianProcess:
    def __init__(self, mean, covariance):
        self.mean = mean
        self.covariance = covariance

    def sample(self, x, size=1):
        m = self.mean(x)
        V = self.covariance(x)
        f = stats.multivariate_normal.rvs(m, V, size=size)
        return f


class MultivariateNormal:
    def __init__(self, m, V):
        self._mean = m
        self._covariance = V
        self._freeze()

    def _freeze(self):
        m = self.mean
        V = self.covariance
        self._frozen = stats.multivariate_normal(m, V)

    def sample(self, size=1):
        return self._frozen.rvs(size=size)


# class Student:
#     def __init__(self, m, V, df):
#         self.mean = m
#         self.covariance = V
#         self.deg_freedom = df

#     def sample(self, size=1):
#         m = self.mean
#         V = self.covariance
#         v = self.deg_freedom

#         samples = []
#         for i in range(size):
#             y = stats.multivariate_normal.rvs(cov=V)
#             u = stats.gamma.rvs(v / 2.0, scale=2.0)
#             x = m + y * np.sqrt(v / u)
#             samples.append(x)

#         samples = np.array(samples)
#         return samples
        

# class NormalInverseGamma:
#     def __init__(self, m, K, a, b):
#         self.mean = m
#         self.covariance = K
#         self.a = a
#         self.b = b

#     @property
#     def dimension(self):
#         return self.mean.size

#     def sample(self, size=1):
#         a = self.a
#         b = self.b
#         v = stats.invgamma.rvs(a, scale=b, size=size)

#         m = self.mean
#         K = self.covariance

#         w = []
#         for i, vi in enumerate(v):
#             Vi = vi * K
#             wi = stats.multivariate_normal.rvs(m, Vi)
#             w.append(wi)
#         w = np.array(w)

#         return w, v
