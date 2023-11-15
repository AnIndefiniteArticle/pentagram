import numpy as np
from .ftheta_hat import ftheta_hat
from scipy.optimize import fsolve

# normalized flux as function of non-dimensional time \
# (that_vals in scale-heights in observer plane) for isothermal atmosphere
def get_phi_vals(that_vals,xtol=1e-10): 
    phi_vals = np.zeros(len(that_vals))
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
        phi_vals[i] = phi_iso
    return phi_vals

