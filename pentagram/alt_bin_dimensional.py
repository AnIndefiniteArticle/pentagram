def alt_bin_dimensional(that_vals_sec,flux,dh_bin_km,hakm,vperp_kms,Dkm,tau_hl=0,Verbose=False):
        
    dt_av = that_vals_sec[1] - that_vals_sec[0]
    i = 0
    j = 0
    i_tot = 0
    t_start = dt_av/2.

    MAXSTORE = 500000 # modify this at some point to use append()
    profile_flux = np.zeros(MAXSTORE)
    profile_time = np.zeros(MAXSTORE)*u.s
    profile_dtheta = np.zeros(MAXSTORE)
    
    t_skip = that_vals_sec[0] - dt_av # confirmed that this and t_start give proper
                                  # alignment of profile_flux and profile_tim
    t_j = t_start + dt_av

    while(j < np.size(flux)):
        if Verbose == True:
            print('j,np.size(flux)=',j,np.size(flux))
        dt_phi = (t_j - t_start) * flux[j]
        d_sum = dt_phi * abs(vperp_kms)
        first = True
        if Verbose== True:
                  print('TP1: d_sum,dt_phi,dh_bin=',d_sum,dt_phi,dh_bin)
        if d_sum <= dh_bin_km: # altitude bin not yet full
            first = False
            while d_sum < dh_bin_km:
                t_j += dt_av
                j+= 1
                #print('TP2: t_j,j=',t_j,j)
                if j >= np.size(flux):
                    break
                dt_phi = flux[j] * dt_av
                d_sum += dt_phi * abs(vperp_kms)
        if j >= np.size(flux):
            break
        dxs = (d_sum - dh_bin_km)/abs(vperp_kms)
        ddt = dxs/flux[j]
#        print('ddt should have units of time:',ddt)
        if first == True:
            ddt = (t_j - t_start)*dxs/dt_phi
        t_last = t_j - ddt
        delta_t = np.abs(t_last - t_start)
        profile_flux[i] = dh_bin_km/(abs(vperp_kms)*delta_t)
    # apply tau_hl if non-zero
        this_alt = hakm-i*dh_bin_km - dh_bin_km/2 # to agree with below
#         if tau_hl != 0:
#             this_tau = tau_hl * np.exp(-this_alt)
#             profile_flux[i] *= np.exp(-this_tau)
        profile_dtheta[i] = 1/profile_flux[i] - 1
        t_start = t_last
        t_now = t_skip + t_last
        profile_time[i] = t_now
        i_tot = i
        i+=1
    profile_flux = profile_flux[0:i_tot]
    profile_time = profile_time[0:i_tot]
    profile_dtheta = profile_dtheta[0:i_tot]/Dkm # note the 1/D factor!
    profile_theta = np.cumsum(profile_dtheta)*dh_bin_km
#   print('check units of profile_theta',profile_theta)
    profile_alt = hakm - np.linspace(0,i_tot-1,i_tot)*dh_bin_km - dh_bin_km/2
# ??? not sure what this was supposed to be...
    profile_alt_theta = hakm - np.linspace(0,i_tot-1,i_tot)*dh_bin_km - dh_bin_km
    return profile_flux,profile_time,profile_alt,profile_alt_theta,\
        profile_dtheta,profile_theta,i_tot


