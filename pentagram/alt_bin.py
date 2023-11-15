import numpy as np
import astropy.units as u

#rebin normalized input lightcurve into equal altitude bins
def alt_bin(that_vals,flux,dh_bin,ha,D,tau_hl=0,Verbose=False):
    
    dt_av = that_vals[1] - that_vals[0]
    if Verbose == True:
        print('alt_bin debug:')
        print('that_vals[0:5]=',that_vals[0:5])
        print('in alt_bin: dt_av=',dt_av)
        print('that_vals=',that_vals)
        print('flux:',flux)
        print('dt_av:',dt_av)
    i = 0
    j = 0
    i_tot = 0
    t_start = dt_av/2.

    MAXSTORE = 500000 # modify this at some point to use append()
    profile_flux = np.zeros(MAXSTORE)
    profile_time = np.zeros(MAXSTORE)
    profile_dtheta = np.zeros(MAXSTORE)
    
    t_skip = that_vals[0] - dt_av # confirmed that this and t_start give proper
                                  # alignment of profile_flux and profile_time
    t_j = t_start + dt_av

    while(j < np.size(flux)):
        if Verbose == True:
            print('\nj,np.size(flux)=',j,np.size(flux))
        dt_phi = (t_j - t_start) * flux[j]
        d_sum = dt_phi
        first = True
        if Verbose== True:
            print('first,TP1: d_sum,dt_phi,dh_bin=',first,d_sum,dt_phi,dh_bin)
        if d_sum <= dh_bin: # altitude bin not yet full
            first = False
            while d_sum < dh_bin:
                t_j += dt_av
                j+= 1
                if Verbose == True:
                    print('TP2: t_j,j=',t_j,j)
                if j >= np.size(flux):
                    break
                dt_phi = flux[j] * dt_av
                d_sum += dt_phi
        if j >= np.size(flux):
            break
        dxs = d_sum - dh_bin
        ddt = dxs/flux[j]
        if first == True:
            ddt = (t_j - t_start)*dxs/dt_phi
        t_last = t_j - ddt
        delta_t = np.abs(t_last - t_start)
        profile_flux[i] = dh_bin/delta_t
    # apply tau_hl if non-zero
        this_alt = ha-i*dh_bin - dh_bin/2 # to agree with below
        if tau_hl != 0:
            this_tau = tau_hl * np.exp(-this_alt)
            profile_flux[i] *= np.exp(-this_tau)
        profile_dtheta[i] = 1/profile_flux[i] - 1
        t_start = t_last
        t_now = t_skip + t_last
        profile_time[i] = t_now
        i_tot = i
        if Verbose == True:# and flux[j] < 0.99:
            print('output i,profile_flux[i],profile_time[i]',i,profile_flux[i],profile_time[i])
            print('input  j,flux[j]        ',j,flux[j])
        i+=1
        if i > MAXSTORE-1:
            break
    profile_flux = profile_flux[0:i_tot]
    profile_time = profile_time[0:i_tot]
    profile_dtheta = profile_dtheta[0:i_tot]/D # note the 1/D factor!
    profile_theta = np.cumsum(profile_dtheta)*dh_bin
    profile_alt = ha - np.linspace(0,i_tot-1,i_tot)*dh_bin - dh_bin/2
# ??? not sure what this was supposed to be...
    profile_alt_theta = ha - np.linspace(0,i_tot-1,i_tot)*dh_bin - dh_bin
    return profile_flux,profile_time,profile_alt,profile_alt_theta,\
        profile_dtheta,profile_theta,i_tot

