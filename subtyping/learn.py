import logging
import numpy as np

from ..longitudinal import LongitudinalSequence
from ..util import random_distribution

from .model import PopulationBasis
from .model import SubpopulationBasis
from .model import Covariance
from .model import LogLikelihood
from .model import SubtypeMixtureModel
from .solvers import Solver, OrthogonalSolver


def default_options():
    options = {
        'num_subtypes': 4,
        'bs_min_obs_time': 0.0,
        'bs_max_obs_time': 5.0,
        'bs_dimension': 8,
        'bs_degree': 2,
        'bs_penalty': 'curvature',
        'bs_smoothness': 1.0,
        'cov_v_const': 16.0,
        'cov_v_bspline': 1.0,
        'cov_v_ou': 4.0,
        'cov_l_ou': 2.0,
        'em_max_iter': 100,
        'em_rel_tol': 1e-4,
        'seed': 0,
        'init_model': None,
        'init_subtypes': None
    }
    return options


def learn_subtypes(sequences, options):
    all_values = [seq.values for seq in sequences]
    num_features = sequences[0].features.size
    
    num_subtypes  = options['num_subtypes']
    low           = options['bs_min_obs_time']
    high          = options['bs_max_obs_time']
    dimension     = options['bs_dimension']
    degree        = options['bs_degree']
    penalty_type  = options['bs_penalty']   
    smoothness    = options['bs_smoothness']
    v_const       = options['cov_v_const']
    v_ou          = options['cov_v_ou']
    l_ou          = options['cov_l_ou']
    seed          = options['seed']
    init_model    = options['init_model']
    init_subtypes = options['init_subtypes']

    rnd = np.random.RandomState(seed)    

    if init_model is not None:
        model = SubtypeMixtureModel.from_json(init_model)

    else:
        pop_basis = PopulationBasis()
        sub_basis = SubpopulationBasis.with_dimension(
            dimension, low, high, degree)
        covariance = Covariance(v_const, v_ou, l_ou)
        log_likelihood = LogLikelihood(pop_basis, sub_basis, covariance)
        prior = random_distribution(num_subtypes, rnd)
        pop_coef = np.zeros(num_features)
        sub_coef = _init_sub_coef(
            dimension, num_subtypes, np.concatenate(all_values))
        model = SubtypeMixtureModel(log_likelihood, prior, pop_coef, sub_coef)

    pop_solver = OrthogonalSolver(
        model.pop_basis, model.covariance, model.sub_basis)

    if penalty_type == 'curvature':
        penalty = model.sub_basis.smoothness_penalty(der=2, dt=1e-2)
    elif penalty_type == 'fused':
        penalty = model.sub_basis.fused_penalty(der=1)
    else:
        raise RuntimeError('Invalid penalty type: {}'.format(penalty_type))

    penalty *= smoothness
    sub_solver = Solver(model.sub_basis, model.covariance, penalty)

    ## Initial E-step...

    posteriors = np.zeros((len(sequences), num_subtypes))
    log_evidence = np.zeros(len(sequences))    

    if init_subtypes is not None:
        for i, z in enumerate(init_subtypes):
            posteriors[i, z] = 1.0
            log_evidence[i] = -np.inf
        
    else:
        for i, seq in enumerate(sequences):
            posteriors[i, :], log_evidence[i] = model.inference(seq)

    iterations = 0
    while True:

        ## M-step...
        
        iterations += 1
        model.prior = posteriors.sum(axis=0) / posteriors.sum()
        model.pop_coef = pop_solver.solve_coef(sequences, all_values)

        pop_residuals = []
        for i, seq in enumerate(sequences):
            yhat = np.dot(model.pop_basis(seq), model.pop_coef)
            pop_residuals.append(seq.values - yhat)

        coef = model.sub_coef
        for k in range(num_subtypes):
            ws = posteriors[:, k]
            coef[k, :] = sub_solver.solve_coef(sequences, pop_residuals, ws)
        model.sub_coef = coef

        ## E-step...
        
        old_logl = log_evidence.mean()
        for i, seq in enumerate(sequences):
            posteriors[i, :], log_evidence[i] = model.inference(seq)
        logl = log_evidence.mean()
        
        delta = (logl - old_logl) / np.abs(old_logl)
        logging.info('Iter={:04d}, LL={:12.6f}, Delta={:12.10f}'.format(
            iterations, logl, delta))

        reached_max = iterations >= options['em_max_iter']
        converged = delta < options['em_rel_tol']
        if reached_max or converged:
            break

    return model, posteriors, log_evidence


def _init_sub_coef(dimension, num_subtypes, all_values):
    q = np.linspace(0, 100, num_subtypes + 2)[1:-1]
    p = np.percentile(all_values, q)
    coef = np.ones((num_subtypes, dimension))
    coef *= p[::-1, None]
    return coef
