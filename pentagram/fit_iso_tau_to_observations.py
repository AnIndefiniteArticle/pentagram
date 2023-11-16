from lmfit import Parameters,minimize,fit_report
from .fit_iso_tau_lmfit import fit_iso_tau_lmfit
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
