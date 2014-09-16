from __future__ import print_function
from __future__ import division

import numpy as np
import emcee


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


class MultiTermFit(object):
    """Multi-term Fourier fit to a light curve

    Parameters
    ----------
    omega : float
        angular frequency of the fundamental mode
    n_terms : int
        the number of Fourier modes to use in the fit
    """
    def __init__(self, omega, n_terms):
        self.omega = omega
        self.n_terms = n_terms

    def _make_X(self, t):
        t = np.asarray(t)
        k = np.arange(1, self.n_terms + 1)
        X = np.hstack([np.ones(t[:, None].shape),
                       np.sin(k * self.omega * t[:, None]),
                       np.cos(k * self.omega * t[:, None])])
        return X

    def log_likelihood(self, t, y, dy):
        y_model = self.compute_model(t)
        Vy = dy ** 2
        return -0.5 * np.sum(np.log(2 * np.pi * Vy) + (y - y_model) ** 2 / Vy)

    def fit(self, t, y, dy):
        """Fit multiple Fourier terms to the data

        Parameters
        ----------
        t: array_like
            observed times
        y: array_like
            observed fluxes or magnitudes
        dy: array_like
            observed errors on y

        Returns
        -------
        self :
            The MultiTermFit object is  returned
        """
        t = np.asarray(t)
        y = np.asarray(y)
        dy = np.asarray(dy)

        X_scaled = self._make_X(t) / dy[:, None]
        y_scaled = y / dy

        self.t_ = t
        self.w_ = np.linalg.solve(np.dot(X_scaled.T, X_scaled),
                                  np.dot(X_scaled.T, y_scaled))
        return self

    def compute_model(self, t):
        return np.dot(self._make_X(t), self.w_)

    def predict(self, Nphase, return_phased_times=False, adjust_offset=True):
        """Compute the phased fit, and optionally return phased times

        Parameters
        ----------
        Nphase : int
            Number of terms to use in the phased fit
        return_phased_times : bool
            If True, then return a phased version of the input times
        adjust_offset : bool
            If true, then shift results so that the minimum value is at phase 0

        Returns
        -------
        phase, y_fit : ndarrays
            The phase and y value of the best-fit light curve
        phased_times : ndarray
            The phased version of the training times.  Returned if
            return_phased_times is set to  True.
        """
        phase_fit = np.linspace(0, 1, Nphase + 1)[:-1]
        y_fit = self.compute_model(2 * np.pi * phase_fit / self.omega)

        if adjust_offset:
            i_offset = np.argmin(y_fit)
            y_fit = np.concatenate([y_fit[i_offset:], y_fit[:i_offset]])

        if return_phased_times:
            if adjust_offset:
                offset = phase_fit[i_offset]
            else:
                offset = 0
            phased_times = (self.t_ * self.omega * 0.5 / np.pi - offset) % 1

            return phase_fit, y_fit, phased_times

        else:
            return phase_fit, y_fit


class MultiTermFitMCMC(MultiTermFit):
    def __init__(self, omega, n_terms,
                 nwalkers = 50, nburn = 1000, nsteps = 2000):
        self.omega = omega
        self.n_terms = n_terms
        self.nwalkers = nwalkers
        self.nburn = nburn
        self.nsteps = nsteps

    def log_prior(self, params):
        return 1

    def log_likelihood(self, params, t, y, dy):
        self.w_ = params
        return MultiTermFit.log_likelihood(self, t, y, dy)

    def log_posterior(self, params, t, y, dy):
        lnprior = self.log_prior(params)
        if np.isneginf(lnprior):
            return lnprior
        else:
            return lnprior + self.log_likelihood(params, t, y, dy)

    def fit(self, t, y, dy):
        t, y, dy = map(np.asarray, (t, y, dy))

        # initialize with a non-robust fit
        MultiTermFit.fit(self, t, y, dy)
        w0 = self.w_
        
        # vary starting guesses around the closed-form guess
        ndim = len(w0)
        starting_guesses = w0 * (0.99 + 0.02
                                 * np.random.rand(self.nwalkers, ndim))

        sampler = emcee.EnsembleSampler(self.nwalkers, ndim,
                                        self.log_posterior,
                                        args=[t, y, dy])
        sampler.run_mcmc(starting_guesses, self.nsteps)
        self.emcee_trace_ = sampler.chain[:, self.nburn:, :].reshape(-1,ndim).T
        self.w_ = self.emcee_trace_.mean(1)

        return self


class MultiTermMixtureFitMCMC(MultiTermFitMCMC):
    def __init__(self, omega, n_terms, Pb=0.1,
                 nwalkers = 50, nburn = 1000, nsteps = 2000):
        self.Pb = Pb
        MultiTermFitMCMC.__init__(self, omega, n_terms,
                                  nwalkers=nwalkers, nburn=nburn,
                                  nsteps=nsteps)

    def fit(self, t, y, dy):
        self.Vb = 25 * np.var(y)
        self.Yb = np.mean(y)
        return MultiTermFitMCMC.fit(self, t, y, dy)

    def log_likelihood(self, params, t, y, dy):
        self.w_ = params
        y_model = self.compute_model(t)
        Vy = dy ** 2
        VyVb = Vy + self.Vb
        loglike_model = -0.5 * (np.log(2 * np.pi * Vy)
                                + (y - y_model) ** 2 / Vy)
        loglike_bg = -0.5 * (np.log(2 * np.pi * VyVb)
                             + (y - self.Yb) ** 2 / VyVb)
        return np.sum(np.logaddexp(np.log(1 - self.Pb) + loglike_model,
                                   np.log(self.Pb) + loglike_bg))
