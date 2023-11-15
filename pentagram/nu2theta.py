# Use refractivity profile to determine bending angle

def nu2theta(dh_km,nu_in):

# integrate along line of sight through successively deeper rays...
    if nu_in[10]< nu_in[9]:
        print('Refractivity should increase with depth. Flipping...')
        nu=np.flip(nu_in)
    else:
        nu = nu_in
    imax = np.size(nu) 
    theta = np.zeros(imax) 
    index = np.linspace(0,imax-1,imax) 
    fac2_all = np.sqrt(index)
    for i in range(imax):
        i_index= np.array(index[0:i+1],dtype=int) 
        i_vals = i - i_index
        fac1_vals = -2*fac2_all[(i_vals-1).clip(min=0)]
        fac2_vals = fac2_all[i_vals]
        fac3_vals = fac2_all[(i_vals-2).clip(min=0)]
        facs = fac1_vals + fac2_vals + fac3_vals
        theta[i-1] = sum(nu[i_index] * facs)
        print(i,end='\r')
    # scale theta by R_planet/dh
    theta *= (VIMS['r_curv']/abs(dh_km)).value
    return theta


