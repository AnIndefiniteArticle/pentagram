# perform isothermal fit to raw lightcurve, using minimize()

def fit_iso_to_observations(tsec,v_perp_kms,data,params,\
    max_nfev=50,fit_xtol=1.e-8,Verbose=False):
    keywords_dict = {"returnModel":False}
    fitted_params = minimize(fit_iso_lmfit,params,\
            args=(tsec,v_perp_kms,data),kws=keywords_dict,\
            method='least_squares',xtol=fit_xtol,max_nfev = max_nfev,nan_policy='omit')
    yfit = fit_iso_lmfit(fitted_params.params,tsec,v_perp_kms,data,returnModel=True,Verbose=False)
    return yfit,fitted_params

