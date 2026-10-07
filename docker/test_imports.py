print("Testing BAGLE Docker environment...")

import astropy
import jax
import numpy
import pymc
import pymultinest
from bagle import model_fitter_jax, model_jax

# The runner loads BAGLE/JAX before lazy-loading optional backends. Exercise
# that order here, especially for pocoMC's PyTorch dependency on Linux.
import pocomc
import nautilus
import blackjax
import numpyro
import jaxns
import psutil
import tensorflow_probability

print("Core, optional sampler, and BAGLE imports passed.")
print("Environment test PASSED.")
