import sys

import matplotlib.pyplot as plt
import numpy as np
from pyrap.tables import table


def freq_profile(
    nu,
    a=-1.2369319597991164,
    b=-7.995603882017982,
    c=11.605973123430397,
    d=-15.787559501497967,
    e=-3.928824456855068,
    ref_flux=15.088731791006047,
    ref_freq=1283791015.625,
):
    """
    Returns spectrum of a point source parametrised as

    I(nu) = I(nu0) (nu/nu0) ** (a + b * log10(nu/nu0)) +
             c * log10(nu/nu0)**2 + d * log10(nu/nu0)**3) +
             e * log10(nu/nu0)**4)

    for v specified in Hz.
    """
    w = nu / ref_freq
    logw = np.log10(w)
    expon = a + b * logw + c * logw**2 + d * logw**3 + e * logw**4
    return ref_flux * w ** (expon)


def freq_profile_tim(
    nu,
    a=2.694,
    b=0.2478,
    c=-0.7137,
    d=0.1129,
    ref_freq=1e9,
):
    """
    Returns spectrum of a point source parametrised as

    I(nu) = exp(a + b * log(nu/nu0)) + c * log(nu/nu0)**2 + d * log(nu/nu0)**3))

    for v specified in Hz.
    """
    w = nu / ref_freq
    logw = np.log(w)
    expon = a + b * logw + c * logw**2 + d * logw**3
    return np.exp(expon)


try:
    ms_name = sys.argv[1]
    freq = table(ms_name + "::SPECTRAL_WINDOW").getcol("CHAN_FREQ").squeeze()
except Exception:
    freq = np.linspace(0.8e9, 1.67e9, 1000)
fluxes = freq_profile(freq)
fluxes_tim = freq_profile_tim(freq)
plt.plot(freq, fluxes, label="PKS1934-638")
plt.plot(freq, fluxes_tim, label="TIM")
plt.show()
