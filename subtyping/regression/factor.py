import numpy as np


def LinearFactor(size, num_feat, n=1):
    size = list(size)
    size = [n] + size + [num_feat]
    factor = np.zeros(size)
    return factor


def num_obs(factor):
    n, *_ = factor.shape
    return n


def scope_shape(factor):
    _, *size, _ = factor.shape
    return tuple(size)


def num_weights(factor):
    *_, n = factor.shape
    return n


def num_features(factor):
    return num_weights(factor)


def random_weights(factor, rnd=np.random):
    n = num_weights(factor)
    return rnd.normal(size=n)


def score_factor(factor, weights):
    size = scope_shape(factor)
    num_feat = num_features(factor)

    assert num_feat == weights.size
    scores = (factor * weights).sum(axis=-1)
    
    return scores


def reduce_scores(scores, keep, agg_fn=np.sum):
    keep = set(keep)
    assert all(1 <= axis < scores.ndim for axis in keep)
    keep.add(0)

    all_axes = set(range(scores.ndim))
    remove = all_axes - keep
    remove= tuple(sorted(list(remove)))

    reduced = agg_fn(scores, axis=remove)
    return reduced
