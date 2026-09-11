"""
    Loads runtime configuration for lya2pcf from a YAML file and computes
    the values that are derived from it.

    The config file loaded is ./parameters.yml by default; override with
    the LYA2PCF_CONFIG environment variable.
"""
import os
import numpy as np
import yaml

_config_path = os.environ.get('LYA2PCF_CONFIG', 'parameters.yml')

with open(_config_path) as _f:
    _cfg = yaml.safe_load(_f)

# IO parammeters
data_dir = _cfg['data_dir']
corr_dir = _cfg['corr_dir']

# Name of the keyword for the deltas in the .fit.gz files
delta_key = _cfg['delta_key']

# Size and number of pixels of correlation outputs
bin_size_r = _cfg['bin_size_r']
rmax = _cfg['rmax']

# For the two point
rpmax = rmax  # Mpc/h
rtmax = rmax  # Mpc/h
numpix_rp = rpmax // bin_size_r
numpix_rt = rtmax // bin_size_r

# For the three point
numpix_mu = _cfg['numpix_mu']
numpix_theta = _cfg['numpix_theta']
numpix_r = rmax // bin_size_r

# Constants
lambdaa = _cfg['lambdaa']  # Anstrongs
la = np.log10(lambdaa)
c = _cfg['c']  # km/s
halfpi = np.pi / 2

# Forest parammeters
gamma = _cfg['gamma']
gammaovertwo = gamma / 2
z_ref = _cfg['z_ref']
nside = _cfg['nside']  # Healpix parammeter
chiquito = _cfg['chiquito_arcsec'] / 3600. * 3.14159 / 180.

# Parammeters for the cosmology module
Omm = _cfg['Omm']
OmDE = 1. - Omm
d_H0 = c / 100  # Mpc/h
# Interpolation parammeters for d_c
zmin = _cfg['zmin']
zmax = _cfg['zmax']
nz = _cfg['nz']

distortion_threads_per_block = tuple(_cfg['distortion_threads_per_block'])
distortion_threads_per_block_2 = tuple(_cfg['distortion_threads_per_block_2'])
max_threads = _cfg['max_threads']

number_of_neighs = _cfg['number_of_neighs']

# If using a machine with several cuda devices
number_of_cuda_devices = _cfg['number_of_cuda_devices']
cuda_device_first_number = _cfg['cuda_device_first_number']
