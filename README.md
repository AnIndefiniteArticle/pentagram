# Initial code and functions from Dick French
This repository has a goal of becoming a library!
The code will have documentation, tests, etc.
This will make these functions accessible to other researchers! (including me!)

The name pentagram is because the functions perform inversions of occultation lightcurves.
A pentagram is an inverted star, and is associated with the occult.
It's a thermometer that uses refraction to recover the thermal profile of an atmosphere.

# How to install

Currently, `pip freeze` output is stored in the `requirements.txt` file.
To reinstantiate a python environment with the latest set of known-to-be-working libraries, do the following in the root directory of this repository:

    $ python -m venv venv # creates virtual environment
    $ source venv/bin/activate # enters virtual environment
    $ pip install -r requirements.txt # installs requirements to virtual environment

Once built the first time, you will only need to use the second command to return to the environment.

Alternatively, you can just use the third command to install the requirements in your existing python environment, and hope for the best.

TODO: Once we have the library importable, show how to import and use it

# Overview of contents

Initial state: all code was in an `.ipynb`, with functions dumped at the top.
These functions will be turned into a library to be imported, with documentation.

#### Description of initial ipynb from Dick French

## VIMS_alpOri271I_analysis_v2.ipynb

Richard G. French, Wellesley College

This Jupyter notebook contains an analysis of the Cassini VIMS observations of the alpOri271 occultation, using data provided by An Foster and Phil Nicholson.

The following steps are performed:

1) Perform model fits to the observations:
    isothermal fits
    isothermal + absorption model based on Goody random band model
2) Overplot Phil Nicholson's model predictions
3) Perform numerical inversion of observations to derive T(P) profile
4) Compare retrieved T(P) with CIRS observations
5) Construct synthetic lightcurve based on CIRS T(P) and compare with observations
6) Perform end-to-end test of inverting the synthetic lightcurve to show that it matches input T(P)

This is a non-interactive notebook - simply run all cells
All required Python packages and data files are defined in the first code cell
All individual functions and procedures are in separate cells, with a description of the purpose and method of each.
The final cells perform the numbered steps above.

Revisions:

v2:

    2023 Oct 10 - rfrench - Move all routines to top, retain tests for documentation, but start new section

Liens:

1. g(r) not implemented - assumed to be constant over range of inversion (easy to fix)


