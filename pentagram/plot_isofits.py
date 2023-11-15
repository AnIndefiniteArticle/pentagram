from lmfit import Parameters,minimize,fit_report
import numpy as np
from .fit_iso_to_observations import fit_iso_to_observations
from .fit_iso_tau_to_observations import fit_iso_tau_to_observations
from .fit_iso_lmfit import fit_iso_lmfit
from .fit_iso_tau_lmfit import fit_iso_tau_lmfit
import matplotlib.pyplot as plt

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

