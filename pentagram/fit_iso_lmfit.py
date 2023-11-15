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

