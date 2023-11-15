import numpy as np
import astropy.constants as const
import astropy.units as u

def lnbarometric2(lnP, h, method_ToflogPmbar,mu,g):

    T = method_ToflogPmbar(lnP)*u.K
    H = (const.k_B * const.N_A *T/(mu*g)).to('km').value
    if T<0:
        P = np.exp(lnP)
        print(P,T)
    deriv = -1/H # dlnP/dh
    return deriv

