# BAGLE: Bayesian Analysis of Gravitational Lensing Events

BAGLE allows modeling of gravitational microlensing events both photometrically and astrometrically. Supported microlensing models include:

- PSPL: point-source, point-lens with (and without) parallax
- BSPL: binary-point-source, point-lens
- PSBL: point-source, binary-point-lens
- FSPL: finite-source, point-lens (minimal support... not well tested yet)
     
All models support fitting data with single or multi-band photometry only, astrometry only, or joint fitting of photometry and astrometry (recommended).

## Documentation
The documentation to the BAGLE code can be found [here](https://bagle.readthedocs.io/en/latest/)

## Installation Instructions

### Install required modules:
Before you can use BAGLE, you will need to install the following modules:

    pip install numpy
    pip install astropy
    pip install matplotlib 
    pip install celerite 
    pip install ephem
    pip install pymultinest
    pip install Pyerfa
    pip install pytest 
    pip install joblib
      
### Install BAGLE from pip or conda (users) or GitHub (developers)
Preferred (on conda-forge):

    conda install BAGLE

or 

    pip install BAGLE

or 

    git clone https://github.com/ninjab3381/BAGLE_Microlensing.git

Test your install by opening python and running:

    import bagle

## Tutorial
A Jupyter Notebook tutorial to see some examples of how to use the code can be found [here](./BAGLE_TUTORIAL.ipynb)

## Developers
After installation of BAGLE source, navigate to the BAGLE_Microlensing folder:

    cd BAGLE_Microlensing/      
      
Then run the tests using the following commands in the BAGLE_Microlensing folder:

    python3 -m pytest tests
      
or you can run the testing scripts individually:
      
    python3 -m pytest tests/test_model.py
    python3 -m pytest tests/test_model_fitter.py
    python3 -m pytest tests/testingmodels.py

