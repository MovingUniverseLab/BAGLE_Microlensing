============
Installation
============

Conda - Recommended
-------------------

BAGLE is available via conda (source-forge) and pip. We recommend installing with conda or mamba::

    conda install bagle

PIP
---
BAGLE can also be installed from pip::

    pip install bagle

From source
-----------

Clone the repository::

    git clone https://github.com/MovingUniverseLab/BAGLE_Microlensing.git bagle

Install required modules::

    pip install numpy
    pip install matplotlib
    pip install astropy
    pip install celerite
    pip install ephem
    pip install pymultinest

Build and install using the command line inside your cloned repo::

    python setupy.py install

Or you can use the source in place (for development) by adding to your
$.bash_profile$ or $.zshenv$::

    export PYTHONPATH=$PYTHONPATH:$HOME/code/python/bagle/src


