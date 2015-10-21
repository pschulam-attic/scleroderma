import numpy as np
import logging
from lbfgs import LBFGS, LBFGSError
from longloglin import loglin
from sklearn.cross_validation import KFold


def learn_weights(dataset, l1_penalty, l2_penalty=0.0, seed=0, warm_start=None):
    objective = new_objective(dataset, l2_penalty=l2_penalty)

    if warm_start is None:
        weights = loglin.random_weights(dataset, seed)
        weights = loglin.flatten_weights(weights)
    else:
        weights = loglin.flatten_weights(warm_start)

    optimizer = LBFGS()
    optimizer.orthantwise_c = l1_penalty
    optimizer.linesearch = 'wolfe'
    optimizer.delta = 1e-6
    optimizer.past = 1

    iterations = np.zeros(1)
    learned_weights = np.zeros_like(weights)
    def progress(x, g, y, xnorm, gnorm, step, k, ls):
        l0_norm = (np.abs(x) > 0).sum()
        learned_weights[:] = x
        m = 'f(x)={:10.6f} ||x||={:10.6f} ||x||_0={:04d} ||f\'(x)||={:10.6f}'


        if iterations[0] % 5 == 0:
            logging.info(m.format(y, xnorm, l0_norm, gnorm))

        iterations[0] += 1

    try:
        optimizer.minimize(objective, weights, progress)
    except KeyError as e:
        pass

    return loglin.structure_weights(dataset, learned_weights)


def learn_weights_cv(dataset, penalties, n_folds, seed=0):
    penalties = list(penalties)
    scores = np.zeros((n_folds, len(penalties)))
    cv = KFold(loglin.num_obs(dataset), n_folds, shuffle=True, random_state=seed)

    for i, (train, test) in enumerate(cv):

        logging.info('Starting CV fold {}'.format(i))

        train_data, test_data = loglin.split_dataset(dataset, train, test)

        for j, penalty in enumerate(penalties):
            logging.info('Testing penalty {:.03e}'.format(penalty))
            weights = learn_weights(train_data, penalty)
            scores[i, j] = loglin.objective(weights, test_data)

    return scores


def learn_time_regularized_weights(datasets, times, time_penalty, l1_penalty, seed=0, warm_start=None):
    objective = new_tied_objective(datasets, times, time_penalty)

    if warm_start is None:
        weights = [loglin.random_weights(d, seed) for d in datasets]
    else:
        weights = warm_start
        
    weights = [loglin.flatten_weights(w) for w in weights]
    weights = np.concatenate(weights)
    
    optimizer = LBFGS()
    optimizer.orthantwise_c = l1_penalty
    optimizer.linesearch = 'wolfe'
    optimizer.delta = 1e-6
    optimizer.past = 1

    iterations = np.zeros(1)
    learned_weights = np.zeros_like(weights)
    def progress(x, g, y, xnorm, gnorm, step, k, ls):
        l0_norm = (np.abs(x) > 0).sum()
        learned_weights[:] = x
        m = 'f(x)={:10.6f} ||x||={:10.6f} ||x||_0={:04d} ||f\'(x)||={:10.6f}'


        if iterations[0] % 5 == 0:
            logging.info(m.format(y, xnorm, l0_norm, gnorm))

        iterations[0] += 1

    try:
        optimizer.minimize(objective, weights, progress)
    except KeyError as e:
        pass

    w_pieces = tied_weight_pieces(learned_weights, datasets)
    w_struct = [loglin.structure_weights(d, w) for d, w in zip(datasets, w_pieces)]
    return w_struct


def new_objective(dataset, l2_penalty=0.0):
    def objective(w, g):
        weights = loglin.structure_weights(dataset, w)
        y = -loglin.objective(weights, dataset)
        y += l2_penalty / 2.0 * np.dot(w, w)

        gradient = loglin.gradient(weights, dataset)
        g[:] = -loglin.flatten_weights(gradient)
        g[:] += l2_penalty * w

        return y

    return objective


def new_centered_objective(dataset, center):
    center = loglin.flatten_weights(center)

    def ojective(d, g):
        w = d + center
        y = -loglin.objective(w, dataset)

        gradient = loglin.gradient(w, dataset)
        g[:] = -loglin.flatten_weights(gradient)

        return y

    return objective


def new_tied_objective(datasets, times, time_penalty):
    num_contexts = len(datasets)
    num_observations = [loglin.num_obs(d) for d in datasets]
    loss_weights = [n / max(num_observations) for n in num_observations]

    def objective(w, g):
        y = 0.0
        w_pieces = tied_weight_pieces(w, datasets)
        g_pieces = tied_weight_pieces(g, datasets)

        for d_i, w_i, g_i, wt_i in zip(datasets, w_pieces, g_pieces, loss_weights):
            weights = loglin.structure_weights(d_i, w_i)
            y += - wt_i * loglin.objective(weights, d_i)

            gradient = loglin.gradient(weights, d_i)
            g_i[:] = - wt_i * loglin.flatten_weights(gradient)

        for i in range(num_contexts - 1):
            w1 = w_pieces[i]
            w2 = w_pieces[i + 1]

            dt = abs(times[i] - times[i + 1])
            # dt = 1.0
            dw = w1 - w2
            y += time_penalty / dt / 2.0 * np.dot(dw, dw)

            g_pieces[i][:] += time_penalty / dt * dw
            g_pieces[i + 1][:] -= time_penalty / dt * dw

        return y

    return objective


def tied_weight_pieces(w, datasets):
    pieces = []
    offset = 0
    for d in datasets:
        n = loglin.total_weights(d)
        end = offset + n
        pieces.append(w[offset:end])
        offset += n

    return pieces
