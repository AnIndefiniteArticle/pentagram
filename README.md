# Initial code and functions from Dick French
This repository has a goal of becoming a library!
The code will have documentation, tests, etc.
This will make these functions accessible to other researchers! (including me!)

# How to install

Currently, `pip freeze` output is stored in the `requirements.txt` file.
To reinstantiate a python environment with the latest set of known-to-be-working libraries, do the following in the root directory of this repository:

    $ python -m venv venv # creates virtual environment
    $ source venv/bin/activate # enters virtual environment
    $ pip install -r requirements.txt # installs requirements to virtual environment

Once built the first time, you will only need to use the second command to return to the environment.

TODO: Once we have the library importable, show how to import and use it

# Overview of contents

Initial state: all code was in an `.ipynb`, with functions dumped at the top.
These functions will be turned into a library to be imported, with documentation.
