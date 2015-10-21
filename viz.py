import numpy as np
import matplotlib.pyplot as plt

from math import sqrt, ceil


def visualize_posterior(sequence, model, unobs=None, **kwargs):
    num_subtypes = model.num_subtypes
    w, h = _guess_grid_size(num_subtypes)
    fig, axes = plt.subplots(w, h, **kwargs)

    lo = model.sub_basis.low
    hi = model.sub_basis.high
    xgrid = np.linspace(lo, hi, 100)

    posterior, _ = model.inference(sequence)
    rank = np.argsort(np.argsort(posterior)[::-1])
    zmap = np.argmax(posterior)

    for z in range(num_subtypes):
        pz = posterior[z]
        i, j = np.unravel_index(rank[z], (w, h))
        ax = axes[i, j]
        gp1 = model.predictive(sequence, 'const', subtype=z)
        gp2 = model.predictive(sequence, 'const ou', subtype=z)

        if z == zmap:
            ax.set_title('**Subtype {} (Pr. = {:.02f})**'.format(z, pz))
        else:
            ax.set_title('Subtype {} (Pr. = {:.02f})'.format(z, pz))
            
        ax.plot(sequence.times, sequence.values, 'ko', label='Observed')
        if unobs is not None:
            ax.plot(unobs.times, unobs.values, 'ro', label='Unobserved')
        
        ax.plot(xgrid, gp1.prior.mean(xgrid), 'b', label='Pop. + Sub.')
        ax.plot(xgrid, gp1.mean(xgrid), 'g', label='Pop. + Sub. + Const.')
        ax.plot(xgrid, gp2.mean(xgrid), 'r', label='Pop. + Sub. + Const. + OU')

        cov = gp2.covariance(xgrid)
        var = np.diag(cov)
        ax.plot(xgrid, gp2.mean(xgrid) + 1.96 * np.sqrt(var), 'k-')
        ax.plot(xgrid, gp2.mean(xgrid) - 1.96 * np.sqrt(var), 'k-')
        
        ax.legend(loc=0)

    return fig, axes


def _guess_grid_size(n):
    sq = ceil(sqrt(n))
    return sq, sq
