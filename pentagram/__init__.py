__all__ = ["ftheta_hat",
           "get_phi_vals",
           "get_phi_tau_vals",
           "fit_iso_lmfit",
           "fit_iso_tau_lmfit",
           "fit_iso_to_observations",
           "fit_iso_tau_to_observations",
           "plot_VIMS_data",
           "plot_isofits",
           "plot_isofits_CIRS",
           "alt_bin",
           "alt_bin_dimensional",
           "lnbarometric",
           "lnbarometric2",
           "htheta2sclht",
           "HtoT",
           "TtoH",
           "pro_refrac2profile",
           "hvalstheta2tsecflux",
           "lightcurve_new",
           "nu2theta",
           "pro_lcgen_v2",
           "pro_lcinvert"
          ]

from .ftheta_hat import ftheta_hat
from .get_phi_vals import get_phi_vals
from .get_phi_tau_vals import get_phi_tau_vals
from .fit_iso_lmfit import fit_iso_lmfit
from .fit_iso_tau_lmfit import fit_iso_tau_lmfit
from .fit_iso_to_observations import fit_iso_to_observations
from .fit_iso_tau_to_observations import fit_iso_tau_to_observations
from .plot_VIMS_data import plot_VIMS_data
from .plot_isofits import plot_isofits
from .plot_isofits_CIRS import plot_isofits_CIRS
from .alt_bin import alt_bin
from .alt_bin_dimensional import alt_bin_dimensional
from .lnbarometric import lnbarometric
from .lnbarometric2 import lnbarometric2
from .htheta2sclht import htheta2sclht
from .HtoT import HtoT
from .TtoH import TtoH
from .pro_refrac2profile import pro_refrac2profile
from .hvalstheta2tsecflux import hvalstheta2tsecflux
from .lightcurve_new import lightcurve_new
from .nu2theta import nu2theta
from .pro_lcgen_v2 import pro_lcgen_v2
from .pro_lcinvert import pro_lcinvert

""" hard coded stuff that the functions need in their current state """

import os
import astropy.units as u

# specify paths to required data files, and define VIMS dictionary with event information

path2inputfiles = './' # modify this to point to directory containing datafiles
path2kernels = './' # modify this to point to directory containing leapseconds kernel file
path2outputfiles = './output/' # modify this to point to directory containing datafiles
if not os.path.exists(path2outputfiles): # create the output directory if it doesn't exist
   os.makedirs(path2outputfiles)

VIMS_observed_lightcurve = 'lightcurvewithgoodbaseline.csv' # from An Foster
VIMS_isothermal_model = 'nicholso-isothermal-alpori271.out' # Phil Nicholson isothermal model
VIMS_isothermal_metadata = 'nicholso-isothermal-metadata-alpori271.out' # ... and metadata for documentation
tlsfile = 'naif0012.tls'

CIRS_TPprofiles = 'globaltemp.sav' # IDL savefile containing CIRS T(P) profiles as function of JD and latitude
# (Obtained from PDS Atmos node? not sure of origin)

# define VIMS dictionary containing essential event geometry and information

VIMS = {
    'path2inputfiles':path2inputfiles,
    'path2outputfiles':path2outputfiles,
    'path2kernels':path2kernels,
    'VIMS_observed_lightcurve':VIMS_observed_lightcurve,
    'dtsec':1.68, # from An Foster
    'VIMS_isothermal_model':VIMS_isothermal_model,
    'VIMS_isothermal_metadata':VIMS_isothermal_metadata,
    'tlsfile':tlsfile, 
    'CIRS_TPprofiles':CIRS_TPprofiles,
    'event':'VIMS alpOri271I',
    'UTC':"2017-116T21:20", #approximate only - used only to determine JD of event for CIRS T(P)
    'g_ms2':11.90*u.m/u.s**2, # from PDN
    'H_km':44.*u.km, # from An 
    'half-light':930, # frame number, from metadata file...
    't_cube':1.68*u.s, 
    't_pixel':0.021*u.s, 
    'rc':54914.3*u.km, 
    'vperp':-2.311*u.km/u.s, 
    'lat_c':-74.44*u.deg,
    'lat_g':-77.23*u.deg,
    'time':1566.6*u.s,
    'range':6.8726e+05*u.km, 
    'alpha':-66.36*u.deg, 
    'r_curv':66043.5*u.km,
    'mu':2.2* u.g/u.mol, # from PDN 
    'RSTP': 129.e-6 # refractivity at STP
}
