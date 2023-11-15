import numpy as np
from .ftheta_hat import ftheta_hat
from scipy.optimize import fsolve

# normalized flux as function of non-dimensional time \
# (that_vals in scale-heights in observer plane)
# for isothermal atmosphere with optional random band model opacity
# tau_hl = half-light slant-path optical depth
# tau_gamma = random band model strength parameter

def get_phi_tau_vals(that_vals,tau_hl,tau_gamma,IsothermalOnly=False,xtol=1e-10):
    phi_tau_vals = np.zeros(len(that_vals))
    for i,xval in enumerate(that_vals):
        if xval <= 0:
            theta_hat0 = np.exp(xval)
        elif xval <= 4:
            theta_hat0 = 2.
        elif xval <= 20:
            theta_hat0 = 20.
        else:
            theta_hat0 = 1/xval
        theta_hat_val = fsolve(ftheta_hat,theta_hat0,xval,xtol=xtol)
        phi_iso = 1./(1.+theta_hat_val)
        if IsothermalOnly:
            phi_tau_vals[i] = phi_iso
        else:
  # Random band model
            tau_gamma = max([0,tau_gamma]) # to avoid negative square root
            tau_exponent = -tau_hl * np.sqrt((1+tau_gamma)/(1+tau_gamma*theta_hat_val)) * theta_hat_val
            # suppress exponent overflow message
            tau_exponent = np.clip(tau_exponent, -709.78, 709.78)
            tau_factor = np.exp(tau_exponent)
            refrac_factor = 1
            phi_tau_vals[i] = phi_iso * tau_factor * refrac_factor
    return phi_tau_vals

