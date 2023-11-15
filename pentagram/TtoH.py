import astropy.constants as const
import astropy.units as u

def TtoH(G,mu,T):
    Avogadro =const.N_A # 6.022141e23
    kBoltzCGS = const.k_B.to(u.erg/u.K) #1.380658e-16
    Gcgs    = G.to(u.cm/u.s**2)
    Tfac = (mu *Gcgs /(const.N_A * const.k_B)).to(u.K/u.km)
    H = (T/Tfac).to(u.km)
    return H

