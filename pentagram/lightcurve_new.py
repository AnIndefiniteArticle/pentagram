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

