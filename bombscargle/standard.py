"""
Standard Lomb-Scargle Periodogram
"""
from __future__ import division, print_function

import numpy as np
from scipy import optimize


def construct_X(t, dy, omega, Nterms=1, compute_offset=False):
    cols = []

    if compute_offset:
        cols.append(np.ones(len(t), dtype=float))

    for i in range(Nterms):
        cols.append(np.sin((i + 1) * omega * t))
        cols.append(np.cos((i + 1) * omega * t))

    return np.transpose(np.vstack(cols) / dy)


def best_params(t, y, dy, omega, Nterms=1, compute_offset=False):
    t, y, dy, omega = map(np.asarray, (t, y, dy, omega))

    Xw = construct_X(t, dy, omega, Nterms, compute_offset)
    yw = y / dy
    return np.linalg.solve(np.dot(Xw.T, Xw),
                           np.dot(Xw.T, yw))


def lomb_scargle(t, y, dy, omegas, Nterms=1, compute_offset=False):
    t, y, dy, omegas = map(np.asarray, (t, y, dy, omegas))

    if compute_offset:
        # TODO: look at the zechmeister & kurster paper to figure this out
        import warnings
        warnings.warn("Something is currently broken with compute_offset=True")

    # if offset is not being computed, then center the y data
    if compute_offset:
        yfit = y
    else:
        w = 1. / dy ** 2
        yfit = y - np.dot(y, w) / w.sum()

    P_LS = np.zeros_like(omegas)
    yw = yfit / dy  

    for i, omega in enumerate(omegas):
        Xw = construct_X(t, dy, omega, Nterms, compute_offset)
        XTy = np.dot(Xw.T, yw)
        XTX = np.dot(Xw.T, Xw)
        P_LS[i] = np.dot(XTy.T, np.linalg.solve(XTX, XTy)) / np.dot(yw.T, yw)
    return P_LS


def huber_loss(params, y, X, dy=1, c=3):
    y_model = np.dot(X, params)
    t = (y - y_model) / dy
    mask = t > c
    loss = 0.5 * t ** 2
    loss[mask] = c * abs(t[mask]) - 0.5 * c ** 2
    return loss.sum()


def best_params_huber(t, y, dy, omega, Nterms=1, compute_offset=False,
                      c=3, return_fmin=False):
    theta_guess = best_params(t, y, dy, omega, Nterms, compute_offset)
    X = construct_X(t, 1, omega, Nterms, compute_offset)
    res = optimize.fmin_bfgs(huber_loss, theta_guess, full_output=True,
                             disp=False, args=(y, X, dy, c))

    if return_fmin:
        return res[:2]
    else:
        return res[0]


def lomb_scargle_huber(t, y, dy, omegas, c=3, Nterms=1, compute_offset=False):
    t, y, dy, omegas = map(np.asarray, (t, y, dy, omegas))

    if compute_offset:
        # TODO: look at the zechmeister & kurster paper to figure this out
        import warnings
        warnings.warn("Something is currently broken with compute_offset=True")

    # if offset is not being computed, then center the y data
    if compute_offset:
        yfit = y
    else:
        w = 1. / dy ** 2
        yfit = y - np.dot(y, w) / w.sum()

    # use non-robust version as the opening guess
    P_LS = np.zeros_like(omegas)

    X = np.ones((t.shape[0], 2 * Nterms + 1))
    loss_reference = huber_loss(np.zeros(2 * Nterms + 1), yfit, X, dy, c)

    for i, omega in enumerate(omegas):
        theta0, f0 = best_params_huber(t, yfit, dy, omega, Nterms,
                                       compute_offset, c, True)
        P_LS[i] = 1 - (f0 / loss_reference)
    return P_LS
