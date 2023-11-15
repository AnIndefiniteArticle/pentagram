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

