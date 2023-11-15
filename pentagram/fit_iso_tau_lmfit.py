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

