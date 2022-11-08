# This file is part of MovementAnalysis.
#
# [1] Ferreira, M. D., Campbell, J. N., & Matwin, S. (2022).
# A novel machine learning approach to analyzing geospatial vessel patterns using AIS data.
# GIScience & Remote Sensing, 59(1), 1473-1490.
#
# Piece of code from: https://github.com/jwergieluk/ou_noise/tree/c5eee685c8a80a079dd32c759df3b97e05ef51ef
#
import numpy as np
import math
import scipy


def ou_process(t, x, start=None):
    """
    OU (Ornstein-Uhlenbeck) process using Maximum-likelihood estimator:
    dX = -A(X-alpha)dt + v dB
    Piece of code from: https://github.com/jwergieluk/ou_noise/tree/c5eee685c8a80a079dd32c759df3b97e05ef51ef
    """
    if start is None:
        v = est_v_quadratic_variation(t, x)
        start = (0.5, np.mean(x), v)

    def error_fuc(theta):
        return -loglik(t, x, theta[0], theta[1], theta[2])

    start = np.array(start)
    result = scipy.optimize.minimize(error_fuc, start, method='L-BFGS-B',
                                     bounds=[(1e-6, None), (None, None), (1e-8, None)],
                                     options={'maxiter': 500, 'disp': False})
    return result.x


def est_v_quadratic_variation(t, x, weights=None):
    """ Estimate v using quadratic variation"""
    assert len(t) == x.shape[1]
    q = quadratic_variation(x, weights)
    return math.sqrt(q/(t[-1] - t[0]))


def quadratic_variation(x, weights=None):
    """ Realized quadratic variation of a path. The weights must sum up to one. """
    assert x.shape[1] > 1
    dx = np.diff(x)
    if weights is None:
        return np.sum(dx*dx)
    return x.shape[1]*np.sum(dx * dx * weights)


def loglik(t, x, mean_rev_speed, mean_rev_level, vola):
    """Calculates log likelihood of a path"""
    dt = np.diff(t)
    mu = mean(x[:, :-1], dt, mean_rev_speed, mean_rev_level)
    sigma = std(dt, mean_rev_speed, vola)
    return np.sum(scipy.stats.norm.logpdf(x[:, 1:], loc=mu, scale=sigma))


def mean(x0, t, mean_rev_speed, mean_rev_level):
    assert mean_rev_speed >= 0
    return x0 * np.exp(-mean_rev_speed * t) + (1.0 - np.exp(- mean_rev_speed * t)) * mean_rev_level


def std(t, mean_rev_speed, vola):
    return np.sqrt(variance(t, mean_rev_speed, vola))


def variance(t, mean_rev_speed, vola):
    assert mean_rev_speed >= 0
    assert vola >= 0
    return vola * vola * (1.0 - np.exp(- 2.0 * mean_rev_speed * t)) / (2 * mean_rev_speed)

