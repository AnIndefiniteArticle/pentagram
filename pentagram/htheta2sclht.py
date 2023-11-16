import astropy.units as u

# altitude and bending angle to pressure, density, scale height, and temperature
# using summations, but no isothermal cap to infinity

def htheta2sclht(hvals,theta,D,denfac=1,prfac=1,Tfac=1*u.K/u.km):  
    """
    From altitude and bending angle to scale height (pressure, density, temperature).
    This code takes in optical and geometry data, and converts it to a thermal profile

    FINDME: Does fac mean derivative somehow?
    
    Parameters
    ----------

    hvals  : 
    theta  :
    D      :
    denfac :
    prfac  :
    Tfac   :

    Returns
    ----------
    dthet  :
    den    :
    pr     :
    H      :
    T      :
    znum   :
    zden   :

    Examples
    ----------

    """
    # maximum index
    imax = np.size(theta)
    # change in h
    dh = np.abs(hvals[0] - hvals[1])
    # FINDME: what is this?
    hfac = 0.4 * dh
    # change in bending angle
    dthet = D * np.gradient(-theta,hvals) # factor of D from Wasserman and Ververka and C code quick_invert.c
    # allocating zeros
    znum = np.zeros(imax)
    zden = np.zeros(imax)
    # array of indices (atmospheric layers)
    index = np.linspace(0,imax-1,imax) 
    # sqrt of index
    sqrt_index = np.sqrt(index)
    # for each atmospheric layer, skipping zero
    # this entire for loop is just index manipulation. Find a way to rewrite this!
    for i in range(1,imax+1):        
        #select j indices, everything of smaller index, skipping zero again
        j_index = np.array(index[1:i],dtype=int)
        # distance in index space between current layer and j comparison layer
        fimj_vals = i - j_index
        # now shift by 1
        fimj1_vals= fimj_vals - 1
        # index * sqrt index for fimj
        fac1_vals = sqrt_index[fimj_vals]*fimj_vals
        # index * sqrt index for fimj1
        fac2_vals = sqrt_index[fimj1_vals]*fimj1_vals
        # fac 1*fimj
        fac3_vals = fimj_vals  * fac1_vals
        # fac 2*fimj1
        fac4_vals = fimj1_vals * fac2_vals
        # FINDME do some algebra
        znum[i-1]= sum(dthet[j_index-1].value*(fac3_vals-fac4_vals))
        zden[i-1]= sum(dthet[j_index-1].value*(fac1_vals-fac2_vals))
    # pressure?
    pr  = znum * prfac
    # density?
    den = zden * denfac
    # skip the outer layer
    zden[0] = zden[1] # to avoid divide by zero
    # from hfac to H FINDME
    H = hfac*znum/zden
    # from tfac to T FINDME
    T = H * Tfac
    # return it all
    return dthet,den,pr,H,T,znum,zden# invert the isothermal model curve
