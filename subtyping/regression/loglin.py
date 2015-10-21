import numpy as np
from longloglin import factor
from scipy.misc import logsumexp

import pdb


def LogLinearDataset(future_loglik, loglik_factor, single_factor, pair_factor):
    return (future_loglik, loglik_factor, single_factor, pair_factor)


def num_obs(dataset):
    future, *_ = dataset
    return int(future.shape[0])


def sub_dataset(dataset, select):
    future, loglik, single, pair = dataset
    future = future[select]
    loglik = [l[select] for l in loglik]
    single = [s[select] for s in single]
    pair   = [p[select] for p in pair]
    return future, loglik, single, pair


def split_dataset(dataset, train, test):
    return sub_dataset(dataset, train), sub_dataset(dataset, test)


def remove_marker(dataset, idx):
    future, loglik, single, pair = dataset
    loglik = [ll for i, ll in enumerate(loglik) if not i == idx]
    single = [s for i, s in enumerate(single) if not i == idx]
    pair = [p for i, p in enumerate(pair, 1) if not i == idx]
    return future, loglik, single, pair


def total_weights(dataset):
    *_, single, pair = dataset
    n = 0
    n += sum(factor.num_weights(f) for f in single)
    n += sum(factor.num_weights(f) for f in pair)
    return n


def random_weights(dataset, seed):
    *_, single, pair = dataset
    rnd = np.random.RandomState(seed)
    single_weights = [factor.random_weights(f, rnd) for f in single]
    pair_weights = [factor.random_weights(f, rnd) for f in pair]
    return single_weights, pair_weights


def flatten_weights(weights):
    single, pair = weights
    flattened = [w.ravel() for w in single] + [w.ravel() for w in pair]
    return np.concatenate(flattened)


def structure_weights(dataset, w):
    *_, single, pair = dataset
    num_single = [factor.num_weights(f) for f in single]
    num_pair = [factor.num_weights(f) for f in pair]

    start = 0
    single_weights = []
    for n in num_single:
        end = start + n
        w_new = w[start:end]
        single_weights.append(w_new)
        start += n

    pair_weights = []
    for n in num_pair:
        end = start + n
        w_new = w[start:end]
        pair_weights.append(w_new)
        start += n

    return single_weights, pair_weights


def objective(weights, dataset):
    _, marg, _ = inference(weights, dataset)
    log_likel = dataset[0]
    log_prior = np.log(marg[0])
    log_joint = log_likel + log_prior
    log_obs = logsumexp(log_joint, axis=1)
    return log_obs.mean()


def gradient(weights, dataset):
    future_loglik, loglik, single_factors, pair_factors = dataset
    _, marginals, joints = inference(weights, dataset)

    log_joint_targ = future_loglik + np.log(marginals[0])
    log_postr_targ = log_joint_targ - logsumexp(log_joint_targ, axis=1)[:, np.newaxis]
    postr_targ = np.exp(log_postr_targ)

    single_weights, pair_weights = weights
    single_grads = [np.zeros_like(w) for w in single_weights]
    pair_grads = [np.zeros_like(w) for w in pair_weights]

    num_groups = postr_targ.shape[1]
    for k in range(num_groups):
        wt = postr_targ[:, k]
        
        # Compute conditional inferences
        _, cond_marginals, cond_joints = inference(weights, dataset, {0:k})

        # Compute the single gradients
        for i, f in enumerate(single_factors):
            f_obs = cond_marginals[i][..., np.newaxis] * f
            f_exp = marginals[i][..., np.newaxis] * f
            f_dif = f_obs - f_exp
            f_dif = (f_dif.T * wt).T
            single_grads[i] += f_dif.sum(axis=1).mean(axis=0)

        # Compute the pair gradients
        for i, f in enumerate(pair_factors):
            f_obs = cond_joints[i][..., np.newaxis] * f
            f_exp = joints[i][..., np.newaxis] * f
            f_dif = f_obs - f_exp
            f_dif = (f_dif.T * wt).T
            pair_grads[i] += f_dif.sum(axis=(1, 2)).mean(axis=0)

    return single_grads, pair_grads


def inference(weights, dataset, observed={}):
    s_weights, p_weights = weights
    _, logliks, s_factors, p_factors = dataset

    # Initialize singleton beliefs
    s_beliefs = []
    for i, w in enumerate(s_weights):
        if i in observed:
            b = -np.inf * np.ones_like(logliks[i])
            z = observed[i]
            if isinstance(z, np.ndarray):
                r = np.arange(b.shape[0])
                b[r, z] = 0.0
            else:
                b[:, z] = 0.0

        else:
            b = logliks[i].copy()
            b += factor.score_factor(s_factors[i], w)
            
        s_beliefs.append(b)

    # Initialize pair beliefs
    p_beliefs = []
    for i, w in enumerate(p_weights):
        b = factor.score_factor(p_factors[i], w)
        p_beliefs.append(b)
        
    # Do upward pass (towards target singleton)
    for i, s in enumerate(s_beliefs[1:]):
        p = p_beliefs[i]
        p += s[:, np.newaxis, :]
        s_beliefs[0] += factor.reduce_scores(p, [1], logsumexp)

    # Compute log partition and renormalize target.
    log_partition = logsumexp(s_beliefs[0], axis=1)
    s_beliefs[0] -= log_partition[:, np.newaxis]

    # Do downward pass (from target singleton)
    s0 = s_beliefs[0]
    for i, s in enumerate(s_beliefs[1:]):
        p = p_beliefs[i]
        m = s0 - factor.reduce_scores(p, [1], logsumexp)
        p += m[:, :, np.newaxis]
        s += factor.reduce_scores(p, [2], logsumexp) - s

    # Normalize beliefs
    for i, s in enumerate(s_beliefs[1:]):
        z_s = logsumexp(s, axis=1)
        s -= z_s[:, np.newaxis]
        z_p = logsumexp(p_beliefs[i], axis=(1, 2))
        p_beliefs[i] -= z_p[:, np.newaxis, np.newaxis]

    # Move to probability space
    for s in s_beliefs:
        s[:] = np.exp(s)

    for p in p_beliefs:
        p[:] = np.exp(p)

    return log_partition, s_beliefs, p_beliefs
