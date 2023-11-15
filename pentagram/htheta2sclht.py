import astropy.units as u

# altitude and bending angle to pressure, density, scale height, and temperature
# using summations, but no isothermal cap to infinity

def htheta2sclht(hvals,theta,D,denfac=1,prfac=1,Tfac=1*u.K/u.km):  
    imax = np.size(theta)
    dh = np.abs(hvals[0] - hvals[1])
    hfac = 0.4 * dh
    dthet = D * np.gradient(-theta,hvals) # factor of D from Wasserman and Ververka and C code quick_invert.c
    znum = np.zeros(imax)
    zden = np.zeros(imax)
    index = np.linspace(0,imax-1,imax) 
    sqrt_index = np.sqrt(index)
    for i in range(1,imax+1):        
        j_index = np.array(index[1:i],dtype=int)
        fimj_vals = i - j_index
        fimj1_vals= fimj_vals - 1
        fac1_vals = sqrt_index[fimj_vals]*fimj_vals
        fac2_vals = sqrt_index[fimj1_vals]*fimj1_vals
        fac3_vals = fimj_vals  * fac1_vals
        fac4_vals = fimj1_vals * fac2_vals
        znum[i-1]= sum(dthet[j_index-1].value*(fac3_vals-fac4_vals))
        zden[i-1]= sum(dthet[j_index-1].value*(fac1_vals-fac2_vals))
    pr  = znum * prfac
    den = zden * denfac
    zden[0] = zden[1] # to avoid divide by zero
    H = hfac*znum/zden
    T = H * Tfac
    return dthet,den,pr,H,T,znum,zden# invert the isothermal model curve


