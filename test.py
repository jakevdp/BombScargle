import numpy as np
from astroML.datasets import fetch_LINEAR_sample
from astroML.time_series import multiterm_periodogram, MultiTermFit

from bombscargle.bombscargle import MultiTermMixtureFit

import matplotlib.pyplot as plt


data = fetch_LINEAR_sample()
t, y, dy = data[1004849].T

omega0 = 17.217
width = 0.03
omega = np.linspace(omega0 - width - 0.01, omega0 + width - 0.01, 1000)

PSD = multiterm_periodogram(t, y, dy, omega, 2)

omega_best = omega[np.argmax(PSD)]
print omega_best


fit = MultiTermFit(omega_best, 2).fit(t, y, dy)
print fit.w_

fit = MultiTermMixtureFit(omega_best, 2, mixture=False).fit(t, y, dy)
print fit.w_

fit = MultiTermMixtureFit(omega_best, 2, mixture=True).fit(t, y, dy)
print fit.w_
