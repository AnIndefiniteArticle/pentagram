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


