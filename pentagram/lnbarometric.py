def lnbarometric(lnP, h, method_TofPmbar,mu,g):
    P = np.exp(lnP)
    T = method_TofPmbar(P)*u.K
    H = (const.k_B * const.N_A *T/(mu*g)).to('km').value
    if T<0:
        print(P,T)
    deriv = -1/H # dlnP/dh
    return deriv

