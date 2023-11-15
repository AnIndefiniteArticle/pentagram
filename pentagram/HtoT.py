import astropy.constants as const
import astropy.units as u

def HtoT(G,mu,H):
    Avogadro =const.N_A # 6.022141e23
    kBoltzCGS = const.k_B.to(u.erg/u.K) #1.380658e-16
    Gcgs    = G.to(u.cm/u.s**2)
    Tfac = (mu *Gcgs /(const.N_A * const.k_B)).to(u.K/u.km)
    T = (Tfac * H).to(u.K)
    return T # in K

