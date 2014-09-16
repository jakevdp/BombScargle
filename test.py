import numpy as np
from astroML.datasets import fetch_LINEAR_sample
from astroML.time_series import multiterm_periodogram, MultiTermFit

from bombscargle.bombscargle import MultiTermMixtureFit

import matplotlib.pyplot as plt


data = fetch_LINEAR_sample()
t, y, dy = data[1004849].T

omega = np.linspace(13.4, 14, 10000)
PSD = multiterm_periodogram(t, y, dy, omega, 6)
#plt.plot(omega, PSD)
#plt.show()
#exit()
omega_best = omega[np.argmax(PSD)]
print omega_best

models = [MultiTermFit(omega_best, 6),
          MultiTermMixtureFit(omega_best, 6, Pb=0),
          MultiTermMixtureFit(omega_best, 6, Pb=0.01)]

for model in models:
    model = model.fit(t, y, dy)
    print model.w_
    phase_fit, y_fit, phased_t = model.predict(1000, return_phased_times=True)
    plt.plot(phase_fit, y_fit)
    
plt.errorbar(phased_t, y, dy, fmt='.k', ecolor='gray', alpha=0.2)
plt.gca().invert_yaxis()
plt.show()
