__all__ = []#"ftheta_hat",
           #"get_phi_vals",
           #"get_phi_tau_vals",
           #"fit_iso_lmfit",
           #"fit_iso_tau_lmfit",
           #"fit_iso_to_observations",
           #"fit_iso_tau_to_observations",
           #"plot_VIMS_data",
           #"plot_isofits",
           #"plot_isofits_CIRS",
           #"alt_bin",
           #"alt_bin_dimensional",
           #"lnbarometric",
           #"lnbarometric2",
           #"htheta2sclht",
           #"HtoT",
           #"TtoH",
           #"pro_refrac2profile",
           #"hvalstheta2tsecflux",
           #"lightcurve_new",
           #"nu2theta",
           #"pro_lcgen_v2",
           #"pro_lcinvert"
          #]

"""
This file contains all functions from Dick in their original form, copied from the ipynb.
My task is to split each function out into its own file, in the process documenting and determining interdependencies.
The name of each function I move out will need to be added to the __all__ string which is the only code that should be left here when the process is complete, leaving this file to act as a simple directory
"""

from astropy import units as u

# non-dimensional bending angle (theta_hat = 1 at half-light)
def ftheta_hat(theta_hat,x):
    return np.log(theta_hat)+theta_hat - 1. - x

# normalized flux as function of non-dimensional time \
# (that_vals in scale-heights in observer plane) for isothermal atmosphere
def get_phi_vals(that_vals,xtol=1e-10): 
    phi_vals = np.zeros(len(that_vals))
    for i,xval in enumerate(that_vals):
        if xval <= 0:
            theta_hat0 = np.exp(xval)
        elif xval <= 4:
            theta_hat0 = 2.
        elif xval <= 20:
            theta_hat0 = 20.
        else:
            theta_hat0 = 1/xval
        theta_hat_val = fsolve(ftheta_hat,theta_hat0,xval,xtol=xtol)
        phi_iso = 1./(1.+theta_hat_val)
        phi_vals[i] = phi_iso
    return phi_vals

# normalized flux as function of non-dimensional time \
# (that_vals in scale-heights in observer plane)
# for isothermal atmosphere with optional random band model opacity
# tau_hl = half-light slant-path optical depth
# tau_gamma = random band model strength parameter

def get_phi_tau_vals(that_vals,tau_hl,tau_gamma,IsothermalOnly=False,xtol=1e-10):
    phi_tau_vals = np.zeros(len(that_vals))
    for i,xval in enumerate(that_vals):
        if xval <= 0:
            theta_hat0 = np.exp(xval)
        elif xval <= 4:
            theta_hat0 = 2.
        elif xval <= 20:
            theta_hat0 = 20.
        else:
            theta_hat0 = 1/xval
        theta_hat_val = fsolve(ftheta_hat,theta_hat0,xval,xtol=xtol)
        phi_iso = 1./(1.+theta_hat_val)
        if IsothermalOnly:
            phi_tau_vals[i] = phi_iso
        else:
  # Random band model
            tau_gamma = max([0,tau_gamma]) # to avoid negative square root
            tau_exponent = -tau_hl * np.sqrt((1+tau_gamma)/(1+tau_gamma*theta_hat_val)) * theta_hat_val
            # suppress exponent overflow message
            tau_exponent = np.clip(tau_exponent, -709.78, 709.78)
            tau_factor = np.exp(tau_exponent)
            refrac_factor = 1
            phi_tau_vals[i] = phi_iso * tau_factor * refrac_factor
    return phi_tau_vals

# compute isothermal model and residuals to observations, used by lmfit

def fit_iso_lmfit(params,tsec,v_perp_kms,data,\
    returnModel=False,Verbose=False):
    Hkm = params['H_km']
    t_hl = params['t_hl']
    that_vals = -v_perp_kms * (tsec - t_hl)/Hkm
    neg_resid = []
    y_bkg = params['y_bkg']
    y_scale = params['y_scale']
    phi_vals = get_phi_vals(that_vals,xtol=1e-10)
    model = y_bkg + y_scale*phi_vals
    if returnModel:
        this_neg_resid = model 
    else:
        this_neg_resid = model - data
    if len(neg_resid) == 0:
        neg_resid = this_neg_resid
    else:
        neg_resid = np.concatenate((neg_resid,this_neg_resid))
    return neg_resid

# compute isothermal + opacity model and residuals to observations, used by lmfit

def fit_iso_tau_lmfit(params,tsec,v_perp_kms,tau_hl,tau_gamma,data,\
    returnModel=False,Verbose=False,IsothermalOnly=False):
    Hkm = params['H_km']
    t_hl = params['t_hl']
    that_vals = -v_perp_kms * (tsec - t_hl)/Hkm

    y_bkg = params['y_bkg']
    y_scale = params['y_scale']     
    tau_hl = params['tau_hl']
    tau_gamma = params['tau_gamma']
    
    phi_vals = get_phi_tau_vals(that_vals,tau_hl,tau_gamma,\
            IsothermalOnly=IsothermalOnly,xtol=1e-10)
            
    neg_resid = []
    model = y_bkg + y_scale*phi_vals
    if returnModel:
        this_neg_resid = model 
    else:
        this_neg_resid = model - data
    if len(neg_resid) == 0:
        neg_resid = this_neg_resid
    else:
        neg_resid = np.concatenate((neg_resid,this_neg_resid))
    return neg_resid

# perform isothermal fit to raw lightcurve, using minimize()

def fit_iso_to_observations(tsec,v_perp_kms,data,params,\
    max_nfev=50,fit_xtol=1.e-8,Verbose=False):
    keywords_dict = {"returnModel":False}
    fitted_params = minimize(fit_iso_lmfit,params,\
            args=(tsec,v_perp_kms,data),kws=keywords_dict,\
            method='least_squares',xtol=fit_xtol,max_nfev = max_nfev,nan_policy='omit')
    yfit = fit_iso_lmfit(fitted_params.params,tsec,v_perp_kms,data,returnModel=True,Verbose=False)
    return yfit,fitted_params

# perform isothermal fit + opacity to raw lightcurve, using minimize()

def fit_iso_tau_to_observations(tsec,v_perp_kms,tau_hl,tau_gamma,data,params,\
    max_nfev=50,fit_xtol=1.e-8,Verbose=False):
    keywords_dict = {"returnModel":False}
    fitted_params = minimize(fit_iso_tau_lmfit,params,\
            args=(tsec,v_perp_kms,tau_hl,tau_gamma,data),kws=keywords_dict,\
            method='least_squares',xtol=fit_xtol,max_nfev = max_nfev,nan_policy='omit')
    yfit = fit_iso_tau_lmfit(fitted_params.params,tsec,v_perp_kms,tau_hl,tau_gamma,data,\
            returnModel=True,Verbose=False)
    return yfit,fitted_params

# Read and plot the data...

def plot_VIMS_data(infile,figfile_rawdata):
    with open(infile,newline='') as csvfile:
        reader = csv.reader(csvfile)
        counts = np.array(list(reader),dtype='float')[:,0]

# three-panel plot of raw data, saved as figfile_rawdata
# log scale
# linear scale
# zoomed linear cale

    fig1=plt.figure(1,figsize=(12,14)) # open up figure 
    plt.rcParams.update({'font.size': 18})

    plt.subplot(3,1,1)
    plt.plot(counts)
    #plt.xlabel('Frame number')
    plt.ylabel('DN')
    plt.ylim(0.1,max(counts)*1.5)
    #plt.ylim(max(min([counts,.1]),max(counts)*1.5))
    plt.yscale('log')
    plt.title(VIMS['event']+' raw observations')

    plt.subplot(3,1,2)

    plt.plot(counts)
    #plt.xlabel('Frame number')
    plt.ylabel('DN')
    plt.ylim(0.0,200)
    plt.yscale('linear')

    plt.subplot(3,1,3)

    plt.plot(counts)
    plt.xlabel('Frame number')
    plt.ylabel('DN')
    plt.xlim(800,1450)
    plt.ylim(-10,200)
    plt.yscale('linear')

    plt.show()
    plt.savefig(figfile_rawdata)
    print("Saved",figfile_rawdata,"\n")
    
    return counts # for later use

#isothermal fits to the data
def plot_isofits(VIMS,counts,figfile_isofits):
    params = Parameters()
# specify initial guesses to fitted parameters
    Hkm = VIMS['H_km'].value
    t_hl = VIMS['half-light']*VIMS['t_cube'].value
    y_bkg = 0.
    y_scale = max(counts)

    vary_H_km = True
    vary_t_hl = True
    vary_y_bkg = False
    vary_y_scale = True

    params.add('H_km',value=Hkm ,vary=vary_H_km)
    params.add('t_hl',value=t_hl,vary=vary_t_hl)
    params.add('y_bkg',value=y_bkg,vary=vary_y_bkg)
    params.add('y_scale',value=y_scale,vary=vary_y_scale)

    frame = np.linspace(0,len(counts)-1,len(counts))

    L = np.where((frame >= 800) & (frame <= 1200)) # zoom specification

    tsec = frame[L]*VIMS['t_cube'] # time in sec from start of data
    data = counts[L] # the observed lightcurve

    yfit,fitted_params = fit_iso_to_observations(tsec.value,VIMS['vperp'].value,data,params,\
        max_nfev=50,fit_xtol=1.e-6,Verbose=True)

    print('--------------Isothermal model------------------')
    print('Parameter                  Value          Stderr')

    for name, param in fitted_params.params.items():
        try:
              print('{:18s} {:14.4f} {:14.4f}'.format(name, param.value, param.stderr))
        except:
              pass

    model_iso = fit_iso_lmfit(fitted_params.params,tsec.value,VIMS['vperp'].value,data,\
        returnModel=True,Verbose=False)

    # Plot isothermal fit, isothermal + absorption fits, and PDN model

    fig2=plt.figure(2,figsize=(12,8)); # open up figure 
    plt.rcParams.update({'font.size': 18})

    plt.plot(tsec,data,label='Obs')

    # overplot PDN model

    result = np.genfromtxt( VIMS['path2inputfiles'] + VIMS['VIMS_isothermal_model'],
                           dtype='float',skip_header = 12,usecols=[6,7],names=['tsec','phi'])
    dt = result['tsec']+fitted_params.params['t_hl']
    phi = result['phi']
    label='H = '+f'{VIMS["H_km"].value:.2f}'+' km (PDN)'
    Lfit = np.where((dt >0 ) & (dt < max(tsec.value)))
    plt.plot(dt[Lfit],fitted_params.params['y_scale']*phi[Lfit],label=label)

    # plot isothermal fit

    label='H = '+f'{fitted_params.params["H_km"].value:.2f}'+' km (fit)'
    plt.plot(tsec,model_iso,label=label)

    # now perform fit with band model

    params = Parameters()

    # specify initial guesses

    Hkm = VIMS['H_km'].value
    t_hl = VIMS['half-light']*VIMS['t_cube'].value
    y_bkg = 0.
    y_scale = max(counts)
    tau_hl = 0.2

    # specify set of fixed values of tau_gamma
    tau_gammas = [0,100]

    vary_t_hl = True
    vary_y_bkg = False
    vary_H_km = True
    vary_y_scale = True
    vary_tau_hl = True
    vary_tau_gamma = False

    for imodel,tau_gamma in enumerate(tau_gammas):
        params.add('H_km',value=Hkm ,vary=vary_H_km)
        params.add('t_hl',value=t_hl,vary=vary_t_hl)
        params.add('y_bkg',value=y_bkg,vary=vary_y_bkg)
        params.add('y_scale',value=y_scale,vary=vary_y_scale)
        params.add('tau_hl',value=tau_hl,vary=vary_tau_hl)
        params.add('tau_gamma',value=tau_gamma,vary=vary_tau_gamma)

        yfit,fitted_params_tau = fit_iso_tau_to_observations(tsec.value,VIMS['vperp'].value,
            tau_hl,tau_gamma,data,params,max_nfev=50,fit_xtol=1.e-6,Verbose=True)

        print('-------------------Band model '+str(imodel+1)+'-----------------')
        print('Parameter                  Value          Stderr')

        for name, param in fitted_params_tau.params.items():
            try:
                  print('{:18s} {:14.4f} {:14.4f}'.format(name, param.value, param.stderr))
            except:
                  pass

        model_tau = fit_iso_tau_lmfit(fitted_params_tau.params,tsec.value,VIMS['vperp'].value,
            tau_hl,tau_gamma,data,returnModel=True,Verbose=False)

        label='H = '+f'{fitted_params_tau.params["H_km"].value:.2f}'+' km '+\
            r'$\tau_{1/2}=$' +f'{fitted_params_tau.params["tau_hl"].value:.2f}' +\
            ', '+r"$\gamma=$" + str(tau_gamma)
        plt.plot(tsec.value,model_tau,label=label)

    plt.xlabel('t (sec)')
    plt.ylabel('DN')
    plt.legend()
    plt.title(VIMS['event'])
    plt.show();
    plt.savefig(figfile_isofits);
    print("Saved",figfile_isofits,"\n")
    
    return tsec,data,model_iso, fitted_params, model_tau, fitted_params_tau

#isothermal fits with CIRS synthetic lightcurve overplotted
def plot_isofits_CIRS(VIMS,tsec_CIRS,flux_CIRS,counts,figfile_isofits_CIRS):
    params = Parameters()
# specify initial guesses to fitted parameters
    Hkm = VIMS['H_km'].value
    t_hl = VIMS['half-light']*VIMS['t_cube'].value
    y_bkg = 0.
    y_scale = max(counts)

    vary_H_km = True
    vary_t_hl = True
    vary_y_bkg = False
    vary_y_scale = True

    params.add('H_km',value=Hkm ,vary=vary_H_km)
    params.add('t_hl',value=t_hl,vary=vary_t_hl)
    params.add('y_bkg',value=y_bkg,vary=vary_y_bkg)
    params.add('y_scale',value=y_scale,vary=vary_y_scale)

    frame = np.linspace(0,len(counts)-1,len(counts))

    L = np.where((frame >= 800) & (frame <= 1200)) # zoom specification

    tsec = frame[L]*VIMS['t_cube'] # time in sec from start of data
    data = counts[L] # the observed lightcurve

    yfit,fitted_params = fit_iso_to_observations(tsec.value,VIMS['vperp'].value,data,params,\
        max_nfev=50,fit_xtol=1.e-6,Verbose=True)

    print('--------------Isothermal model------------------')
    print('Parameter                  Value          Stderr')

    for name, param in fitted_params.params.items():
        try:
              print('{:18s} {:14.4f} {:14.4f}'.format(name, param.value, param.stderr))
        except:
              pass

    model_iso = fit_iso_lmfit(fitted_params.params,tsec.value,VIMS['vperp'].value,data,\
        returnModel=True,Verbose=False)

    # Plot isothermal fit, isothermal + absorption fits, and PDN model

    fig2=plt.figure(2,figsize=(12,8)); # open up figure 
    plt.rcParams.update({'font.size': 18})

    plt.plot(tsec,data,label='Obs')

    # overplot PDN model

    result = np.genfromtxt( VIMS['path2inputfiles'] + VIMS['VIMS_isothermal_model'],
                           dtype='float',skip_header = 12,usecols=[6,7],names=['tsec','phi'])
    dt = result['tsec']+fitted_params.params['t_hl']
    phi = result['phi']
    label='H = '+f'{VIMS["H_km"].value:.2f}'+' km (PDN)'
    Lfit = np.where((dt >0 ) & (dt < max(tsec.value)))
    plt.plot(dt[Lfit],fitted_params.params['y_scale']*phi[Lfit],label=label)

    # plot isothermal fit

    label='H = '+f'{fitted_params.params["H_km"].value:.2f}'+' km (fit)'
    plt.plot(tsec,model_iso,label=label)

    # now perform fit with band model

    params = Parameters()

    # specify initial guesses

    Hkm = VIMS['H_km'].value
    t_hl = VIMS['half-light']*VIMS['t_cube'].value
    y_bkg = 0.
    y_scale = max(counts)
    tau_hl = 0.2

    # specify set of fixed values of tau_gamma
    tau_gammas = [0,100]

    vary_t_hl = True
    vary_y_bkg = False
    vary_H_km = True
    vary_y_scale = True
    vary_tau_hl = True
    vary_tau_gamma = False

    for imodel,tau_gamma in enumerate(tau_gammas):
        params.add('H_km',value=Hkm ,vary=vary_H_km)
        params.add('t_hl',value=t_hl,vary=vary_t_hl)
        params.add('y_bkg',value=y_bkg,vary=vary_y_bkg)
        params.add('y_scale',value=y_scale,vary=vary_y_scale)
        params.add('tau_hl',value=tau_hl,vary=vary_tau_hl)
        params.add('tau_gamma',value=tau_gamma,vary=vary_tau_gamma)

        yfit,fitted_params_tau = fit_iso_tau_to_observations(tsec.value,VIMS['vperp'].value,
            tau_hl,tau_gamma,data,params,max_nfev=50,fit_xtol=1.e-6,Verbose=True)

        print('-------------------Band model '+str(imodel+1)+'-----------------')
        print('Parameter                  Value          Stderr')

        for name, param in fitted_params_tau.params.items():
            try:
                  print('{:18s} {:14.4f} {:14.4f}'.format(name, param.value, param.stderr))
            except:
                  pass

        model_tau = fit_iso_tau_lmfit(fitted_params_tau.params,tsec.value,VIMS['vperp'].value,
            tau_hl,tau_gamma,data,returnModel=True,Verbose=False)

        label='H = '+f'{fitted_params_tau.params["H_km"].value:.2f}'+' km '+\
            r'$\tau_{1/2}=$' +f'{fitted_params_tau.params["tau_hl"].value:.2f}' +\
            ', '+r"$\gamma=$" + str(tau_gamma)
        plt.plot(tsec.value,model_tau,label=label)
    t_CIRS = tsec_CIRS.value+fitted_params.params['t_hl']
    model_CIRS = fitted_params.params['y_scale']*flux_CIRS + fitted_params.params['y_bkg']
    plt.plot(t_CIRS,model_CIRS,label='CIRS T(P)',linewidth=4)
    plt.xlabel('t (sec)')
    plt.ylabel('DN')
    plt.legend()
    plt.xlim(np.min(tsec.value),np.max(tsec.value))
    plt.title(VIMS['event'])
    plt.show();
    plt.savefig(figfile_isofits_CIRS);
    print("Saved",figfile_isofits_CIRS,"\n")
    return #tsec,model_iso, fitted_params, model_tau, fitted_params_tau

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

def lnbarometric(lnP, h, method_TofPmbar,mu,g):
    P = np.exp(lnP)
    T = method_TofPmbar(P)*u.K
    H = (const.k_B * const.N_A *T/(mu*g)).to('km').value
    if T<0:
        print(P,T)
    deriv = -1/H # dlnP/dh
    return deriv

def lnbarometric2(lnP, h, method_ToflogPmbar,mu,g):

    T = method_ToflogPmbar(lnP)*u.K
    H = (const.k_B * const.N_A *T/(mu*g)).to('km').value
    if T<0:
        P = np.exp(lnP)
        print(P,T)
    deriv = -1/H # dlnP/dh
    return deriv

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

def HtoT(G,mu,H):
    Avogadro =const.N_A # 6.022141e23
    kBoltzCGS = const.k_B.to(u.erg/u.K) #1.380658e-16
    Gcgs    = G.to(u.cm/u.s**2)
    Tfac = (mu *Gcgs /(const.N_A * const.k_B)).to(u.K/u.km)
    T = (Tfac * H).to(u.K)
    return T # in K

def TtoH(G,mu,T):
    Avogadro =const.N_A # 6.022141e23
    kBoltzCGS = const.k_B.to(u.erg/u.K) #1.380658e-16
    Gcgs    = G.to(u.cm/u.s**2)
    Tfac = (mu *Gcgs /(const.N_A * const.k_B)).to(u.K/u.km)
    H = (T/Tfac).to(u.km)
    return H

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

# convert altitude/bending angle arrays to lightcurve in time, accommodating ray crossing
def hvalstheta2tsecflux(hvals,theta,vperp,tsec_start,tsec_stop,dt_sec,D_km):
    dh_km = abs(hvals[0] - hvals[1]) # input altitude spacing
    dlc_km = dt_sec * abs(vperp) # conversion between time and distance on lightcurve
    dh_dlc = dh_km/dlc_km # non-dimensional
    ijmax = int((tsec_stop - tsec_start)/dt_sec+0.5)
    flux = np.zeros(ijmax)
    tsec_vals = tsec_start + np.linspace(0,ijmax-1,ijmax)*dt_sec
    thetaD = theta * D_km / dlc_km
#    print('theta',theta)
#    print('thetaD -  should be dimensionless!',thetaD)
    imax = theta.size
    for i in range(1,imax):
        zji=(i-1)*dh_dlc + thetaD[i-1] - 1.5 # empirical factor
        zji1=i*dh_dlc + thetaD[i] - 1.5
        zjmin=min([zji,zji1])
        zjmax=max([zji,zji1])
        jmin=int(1+zjmin)
        jmax=int(1+zjmax)
        if jmin <= ijmax and jmax >= 1:
            jmin1=max([jmin,1])
            jmax1=min([jmax,ijmax])
            fphi=dh_dlc/(zjmax-zjmin)
            phifac=fphi
            for j in range(jmin1,jmax1+1):
                add=phifac*(min([zjmax,j])-max([zjmin,(j-1)]))
                flux[j-1] += add
    flux[0:2] = flux[2] # since first bins not filled
    return tsec_vals,flux

def lightcurve_new(hvals,theta,tau_hl=0,pts_per_H=1000):
    dh = hvals[0] - hvals[1]
    print('in lightcurve_new: dh = ',dh)
    hhat = 1./dh
    that = pts_per_H
    dj = 1./that
    at = dj
 
# compute duration of lightcurve

    ha = max(hvals)
    hb = -min(hvals)
    du = ha + np.exp(hb) + hb
    
   # print("in lightcurve: pts_per_H,hhat,ha,hb,du = ",pts_per_H,hhat,ha,hb,du)
    
    ijmax = int(du/at+0.5)
    du = at*ijmax

    flux = np.zeros(ijmax)
    #print("in lightcurve: pts_per_H,du,ijmax,at = ",\
    #      pts_per_H,du,ijmax,at)

# This is the best I could come up with for alignment with variety of choices of dh and dj
# not ideal but good enough.
# using golc() run 5 for a variety of choices
   # that_vals = -max(hvals) + np.linspace(0,ijmax-1,ijmax)*dj - 1 +dh - 2*dj# so zero at half-light
   # that_vals = -max(hvals) + np.linspace(0,ijmax-1,ijmax)*dj - 1 +2*dh - 1*dj
    that_vals = -max(hvals) + np.linspace(0,ijmax-1,ijmax)*dj - 1 +dh# -2*dh #- 1*dj
    #print('in lightcurve: dj,that_vals[0:4]=',dj,that_vals[0:4])
# step through entire theta array and compute contribution to this part of lightcurve

    imax = theta.size
    for i in range(1,imax):
        zji=((i-1)*dh+theta[i-1])/dj
        zji1=(i*dh+theta[i])/dj
        zjmin=min([zji,zji1])
        zjmax=max([zji,zji1])
        jmin=int(1+zjmin)
        jmax=int(1+zjmax)
        if jmin <= ijmax  and jmax >= 1:
            jmin1=max([jmin,1])
            jmax1=min([jmax,ijmax])
            fphi=dh/(dj*(zjmax-zjmin))
            phifac=fphi
            for j in range(jmin1,jmax1+1):
                add=phifac*(min([zjmax,j])-max([zjmin,(j-1)]))
                flux[j-1] += add
    if tau_hl>0:
        flux *= np.exp(-tau_hl * (1/flux-1))    
    return that_vals,flux

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

def pro_lcgen_v2(R_in,nu_in,D_in):
    R = R_in.value
    nu = nu_in
    D = D_in.value
    nRp1 = len(R)
    nR = nRp1 - 1
    nRm1 = nR - 1
# determine if arrays are increasing outward or not

    if R[0] < R[-1]:
        print('flipping input R array')
        R       = np.flip(R)
    if nu[0] > nu[-1]:
        print('flipping input nu array')
        nu      = np.flip(nu)
    print('First ten elements of R, nu: R should decrease, nu should increase')
    print(R[0:10])
    print(nu[0:10])
    print('len(R),len(nu)',len(R),len(nu))
    if R[-1] > R[100]:
        print('R array is not in the correct order!')
    else:
        print('R array is in the correct order')
    if nu[-1] < nu[100]:
        print('nu array is not in the correct order!')
    else:
        print('nu array is in the correct order')
#  compute coefficients without using 2-D arrays
    dr      = np.abs(R[0:nRm1+1] - R[1:nR+1]) # note that dr is always positive
    Rsq     = (R*R)

# ; compute theta array
    theta = np.zeros(nRp1,dtype=object)
    rho = np.zeros(nRp1,dtype=object)
    dnudr = np.gradient(nu,R)
    dnudr = abs(np.diff(nu)/np.diff(R))
    ivals = np.linspace(1,nR,num=nR,dtype=int)
    
    for i in ivals:
        print(i,end='\r')
#        Xji = np.array([np.sqrt(Rsq[0:i-1+1] - Rsq[i]),0.0],dtype=object)
        Xji = np.sqrt(Rsq[0:i-1+1] - Rsq[i])
        Xji=np.append(Xji,0.0)
        dXji = -np.diff(Xji)
#        print('i,Xji,dXji',i,Xji,dXji)
        vals = dXji[0:i] * dnudr[0:i]/R[0:i] * 2.0 * R[i]

#        print('vals[0],np.sum(vals[0])',vals[0],np.sum(vals[0],keepdims=False))       
#THIS IS VERY SLOW
#        theta[i] = np.sum(vals[0],keepdims=False)
# try this instead
        theta[i] = sum(vals)
#        print('i,theta[i],theta_compare',i,theta[i],theta_compare)
        rho[i]  = R[i] - D*theta[i]
#        print('theta[0:i+1]:',theta[0:i+1])
#         if i <= 4:
#             print('\nR[i],nu[i],dnudr[i]',R[i],nu[i],dnudr[i])
#             print('Xji',Xji)
#             print('dXji',dXji)
#             print('vals',vals)
#             print('theta[i]',theta[i])
#             print('rho[i]',rho[i])
    phi_cyl = np.zeros(nRp1)
    drho = np.abs(np.diff(rho))
#    print('drho',drho)
#    phi_cyl = dr/drho
    for i in ivals:
        phi_cyl[i] = dr[i-1]/abs(rho[i-1] - rho[i])
    phi_cyl[0]      = 1.0 # top of light curve
    phi_cyl[1] = (phi_cyl[0] + phi_cyl[2])/2
#    print('phi_cyl[0:4]',phi_cyl[0:5])
    rho[0]=rho[1]
    return rho,theta,phi_cyl

# determine refractivity and bending angle from numerical inversion of lightcurve
def pro_lcinvert(rho_in,phi_cyl_in,nu0_in,D_in):
#    rho_in: distance in observer plane of ray from projected center of planet in km
#    phi_cyl_in: normalized flux (cyl means no spherical focussing)

    phi_cyl = np.zeros(len(phi_cyl_in),dtype=float) + phi_cyl_in
#     print(phi_cyl.flags.writeable)
    D = D_in
    
    nRp1 = len(rho_in)
    rho = np.zeros(nRp1,dtype=float) + rho_in
    rho[0] = rho[1] + rho[1] - rho[2]

    nR = nRp1-1
    nRm1 = nR -1
    
    if rho_in[10] < rho_in[-1]:
        rho = np.flip(rho)
        print('reversed order of rho')
    if phi_cyl[5]<phi_cyl[-1]:
        phi_cyl = np.flip(phi_cyl)
        print('reversed order of phi_cyl')
    if phi_cyl[1] < .1:
        phi_cyl[1]  = 0.99999643  
    R = np.zeros(nRp1, dtype = float)
    dr = np.zeros(nR, dtype = float)

    R[0] = rho[0] # align at top of 
    
#     print('rho[0:4]',rho[0:5])
#     print('phi_cyl[0:4]',phi_cyl[0:5])
        
    ivals = np.linspace(1,nR-1,num=nR-1,dtype=int)
    for i in ivals:
        dr[i] = abs(rho[i-1]- rho[i]) * phi_cyl[i]
        if i == 1:
            dr[0] = dr[1]
        R[i] = R[i-1] - dr[i-1]
#         if i <=5:
#             print('i,dr[i-1]',i,dr[i-1])
#             print('R[i-1],R[i]',R[i-1],R[i])
#    if R[0] == R[1]:
 #       R[0] = R[1]-(R[2]-R[1])
    Rsq     = R*R
    if dr[0] == 0:
        dr[0] = dr[1]
    theta   = (R - rho)/D
    dnu     = np.zeros(nRp1,dtype=float)
    nu      = np.zeros(nRp1,dtype=float)
    nu0     = nu0_in
    Aji_01  = 2.0*np.sqrt(Rsq[0] - Rsq[1])/dr[0]
    if Aji_01>0:
        dnu[0]  = theta[1]/Aji_01
    nu[1]   = nu[0] + dnu[0]
#     print('rho[0:4]',rho[0:4])
#     print('theta[0:4]',theta[0:4])
#     print('phi_cyl[0:4]',phi_cyl[0:4])
#     print('R[0:4]',R[0:4])
#     print('dr[0:4]',dr[0:4])    
#     print('Aji_01=',Aji_01)
#     print('dr[0]',dr[0])
#     print('R[0:4]-rho[0:4]',R[0:4]-rho[0:4])
#     print('D',D)
    for i in ivals:
        Xji1    = np.sqrt(Rsq[0:i+2] - Rsq[i+1])
        Aji1    = 2. * R[i+1]/R[0:i+1] * (Xji1[0:i+1] - Xji1[1:i+2])/dr[0:i+1]
        if Aji1[i] >0:
            dnu[i]  = (theta[i+1] - sum(Aji1[0:i-1+1]*dnu[0:i-1+1]))/Aji1[i]
        nu[i+1] = nu[i] + dnu[i]
#         if i <= 4:
#             print(i)
#             print('Xji1',Xji1)
#             print('Aji1',Aji1)
#             print('dnu[i]',dnu[i])
#             print('nu[i]',nu[i])
        print(i,end='\\r')
    nu[0] = nu[1]
    nu[-1]=nu[-2]
    return nu,theta,R


