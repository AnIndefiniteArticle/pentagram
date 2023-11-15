# get physical units profile from refractivity

def pro_refrac2profile(R,nu,Ttop,G,mu,refrac):
#     ; input
# ;       nu(R)   radial refractivity profile  (R in km, radius from center of planet)
# ;       Ttop    Temperature (K) at top of profile
# ;       Gref    gravitational acceleration (m/s^2) 
# ;       mu      scalar mean molecular wt
# ;       refrac  refractivity

# ; output
# ;       T(R)    Kelvin
# ;       P(R)    Pressure (Pascals)
# ;       den     density  (kg/m^3)
# ;       H(R)    scale height (km)
#     print('R=',R[0:4])
#     print('nu=',nu[0:4])
#     print('Ttop',Ttop)
#     print('G',G)
#     print('mu',mu)
#     print('refrac',refrac)
    

    Avogadro =const.N_A # 6.022141e23
    kBoltzCGS = const.k_B.to(u.erg/u.K) #1.380658e-16
    mAMU    = 1*u.g/const.N_A # 1.66053886e-24 # gm /mol 
    Loschmidt = 2.6867811e+19 /u.cm**3 # const.amagat # 2.68684e19
    nR      = len(nu)
    ncm3    = nu*Loschmidt/refrac
    Pmbar    = np.zeros(nR,dtype=float)*u.mbar
    Gcgs    = G.to(u.cm/u.s**2)
    dencgs  = ncm3 * mu / Avogadro # gm/cm^3
    den     = dencgs * 1000.0 # kg/m^3

    Pmbar[0] = (ncm3[0] * kBoltzCGS * Ttop).to(u.mbar) # need pressure at top of top shell
#     print('ncm3[0],kBoltzCGS,Ttop,Pcgs[0],refrac',ncm3[0],kBoltzCGS,Ttop,Pmbar[0],refrac)

    # integrate hydrostatic equation to get dP

    ivals = np.linspace(1,nR-1,num=nR-1,dtype=int)
    for i in ivals:
#         if i < 5:
#             print(i)
#             print('R[i]',R[i])
#             print('nu[i]',nu[i])
#             print('Pmbar[i-1]',Pmbar[i-1])
#             print('ncm3[i-1]',ncm3[i-1])
#             print('dencgs[i-1]',dencgs[i-1])
#        Pmbar[i] = Pmbar[i-1] - (dencgs[i-1] * Gcgs * (R[i] - R[i-1])).to(u.mbar)
#  this reduces the error considerably
        Pmbar[i] = Pmbar[i-1] - ((dencgs[i]+dencgs[i-1])/2. * Gcgs * (R[i] - R[i-1])).to(u.mbar)

    ncm3[-1]  = ncm3[-2] # fix off-scale endpints
    Pmbar[-1]  = Pmbar[-2] 
    den[-1]  = den[-2]

    T       = (Pmbar/(kBoltzCGS * ncm3)).to(u.K)
    Tfac = (mu *Gcgs /(const.N_A * const.k_B)).to(u.K/u.km)
    H       = T/Tfac

    return ncm3,den,Pmbar,T,H

