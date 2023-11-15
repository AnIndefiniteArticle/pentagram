import numpy as np

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
        print(i,end='\r')
    nu[0] = nu[1]
    nu[-1]=nu[-2]
    return nu,theta,R


