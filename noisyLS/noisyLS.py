from __future__ import print_function
from __future__ import division

import numpy as np
import emcee

from astroML.time_series import MultiTermFit, multiterm_periodogram


def multiterm_model(t, omega, theta, broadcast=True):
    """
    Compute the multi-term model at t given Omega and theta

    Parameters
    ----------
    t : array_like
        times of observation
    omega : float
        frequency of fit
    theta : array_like
        length (2n + 1) array, where n is the number of harmonics

    Returns
    -------
    y : ndarray
        the model evaluated at t
    """
    t = np.asarray(t)
    theta = np.asarray(theta)
    assert len(theta) % 2 == 1
    nterms = (len(theta) - 1) // 2
    An = theta[1:nterms + 1]
    Bn = theta[nterms + 1:]

    if broadcast:
        args = (np.arange(1, len(An) + 1)[:, None] * omega * t)
        return theta[0] + np.dot(An, np.sin(args)) + np.dot(Bn, np.cos(args))
    else:
        return theta[0] + sum(A * np.sin(omega * (m + 1) * t) +
                              B * np.cos(omega * (m + 1) * t)
                              for m, (A, B) in enumerate(zip(An, Bn)))


def log_likelihood_pure(t, y, dy, omega, theta):
    y_model = multiterm_model(t, omega, theta)
    Vy = dy ** 2
    return -0.5 * np.sum(np.log(2 * np.pi * Vy) + (y - y_model) ** 2 / Vy)


def log_likelihood_mixture(t, y, dy, omega, Pb, Vb, Yb, theta):
    y_model = multiterm_model(t, omega, theta)
    Vy = dy ** 2
    VyVb = Vy + Vb

    loglike_model = -0.5 * (np.log(2 * np.pi * Vy) + (y - y_model) ** 2 / Vy)
    loglike_bg = -0.5 * (np.log(2 * np.pi * VyVb) + (y - Yb) ** 2 / VyVb)

    return np.sum(np.logaddexp(np.log(1 - Pb) + loglike_model,
                               np.log(Pb) + loglike_bg))


class MultiTermMixtureFit(object):
    def __init__(self, omega, n_terms, mixture=True,
                 nwalkers = 50, nburn = 1000, nsteps = 2000):
        self.omega = omega
        self.n_terms = n_terms
        self.mixture = mixture
        self.nwalkers = nwalkers
        self.nburn = nburn
        self.nsteps = nsteps

    @staticmethod
    def log_prior_mixture(params):
        # params = [Pb, Vb, Yb] + theta
        if params[0] < 0 or params[0] > 1 or params[1] <= 0:
            return -np.inf
        else:
            Pb = params[0]
            return -0.5 * np.log(2 * np.pi * 0.1) - 0.5 * (Pb - 0.1) ** 2 / (0.1 ** 2)
            #return 1

    @staticmethod
    def log_prior_pure(params):
        # params = theta
        return 1

    @staticmethod
    def log_likelihood_mixture(params, t, y, dy, omega):
        return log_likelihood_mixture(t, y, dy, omega,
                                      params[0], params[1], params[2],
                                      params[3:])

    @staticmethod
    def log_likelihood_pure(params, t, y, dy, omega):
        return log_likelihood_pure(t, y, dy, omega, params)
    
    def fit(self, t, y, dy):
        # Do a simple fit to find the starting guess
        simple_fit = MultiTermFit(self.omega, self.n_terms)
        simple_fit.fit(t, y, dy)
        w_best = list(simple_fit.w_)

        if self.mixture:
            starting_guess = [0.5, 10 * np.var(y), np.mean(y)] + w_best
            log_prior = self.log_prior_mixture
            log_likelihood = self.log_likelihood_mixture
        else:
            starting_guess = w_best
            log_prior = self.log_prior_pure
            log_likelihood = self.log_likelihood_pure

        def log_posterior(params, t, y, dy, omega):
            lnprior = log_prior(params)
            if np.isneginf(lnprior):
                return lnprior
            else:
                return lnprior + log_likelihood(params, t, y, dy, omega)


        ndim = len(starting_guess)
        starting_guesses = np.random.normal(starting_guess, 0.1,
                                            size=(self.nwalkers, ndim))

        sampler = emcee.EnsembleSampler(self.nwalkers, ndim, log_posterior,
                                        args=[t, y, dy, self.omega])
        sampler.run_mcmc(starting_guesses, self.nsteps)
        self.emcee_trace_ = sampler.chain[:, self.nburn:, :].reshape(-1,ndim).T
        self.w_ = self.emcee_trace_.mean(1)

        return self

# 1004849 6 terms
