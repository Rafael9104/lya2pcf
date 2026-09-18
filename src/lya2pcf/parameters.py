"""
    Loads runtime configuration for lya2pcf from a YAML file and computes
    the values that are derived from it.

    The config file loaded is ./parameters.yml by default; override with
    the LYA2PCF_CONFIG environment variable.
"""
import os
import numpy as np
import yaml

_CONFIG_ENV_VAR = 'LYA2PCF_CONFIG'
_config_path = os.environ.get(_CONFIG_ENV_VAR, 'parameters.yml')

try:
    with open(_config_path) as _f:
        _cfg = yaml.safe_load(_f)
except FileNotFoundError:
    _from = ('the %s environment variable' % _CONFIG_ENV_VAR
              if _CONFIG_ENV_VAR in os.environ else
              'the default filename (%s is not set)' % _CONFIG_ENV_VAR)
    raise FileNotFoundError(
        "lya2pcf could not find its config file.\n"
        "\n"
        "  looked for : %r\n"
        "  resolved to: %s\n"
        "  from       : %s\n"
        "  cwd        : %s\n"
        "\n"
        "parameters.yml is read from the current directory by default, "
        "not from wherever lya2pcf itself is installed -- so this usually "
        "means the command was run from the wrong directory. Either run it "
        "from the directory holding your parameters.yml, or point %s at "
        "one explicitly:\n"
        "\n"
        "  %s=/path/to/parameters.yml <command>\n"
        % (_config_path, os.path.abspath(_config_path), _from, os.getcwd(),
           _CONFIG_ENV_VAR, _CONFIG_ENV_VAR)
    ) from None

# IO parammeters
data_dir = _cfg['data_dir']
corr_dir = _cfg['corr_dir']

# Names of the HDUs and metadata columns in the delta files; see parameters.yml.
# Split by where they are looked up, since the two need different error reports.
delta_hdu_keys = ('delta', 'lambda', 'weight', 'metadata')
delta_column_keys = ('los_id', 'targetid', 'ra', 'dec')
delta_keys = _cfg['delta_keys']
_missing_keys = [k for k in delta_hdu_keys + delta_column_keys if k not in delta_keys]
if _missing_keys:
    raise ValueError("delta_keys in %s is missing: %s"
                     % (_config_path, ', '.join(_missing_keys)))

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

# Precision of the two-point correlation kernel: the numpy dtype for the device
# buffers and the matching C type, compiled into the kernel via -DMYFLOAT.
_gpu_precisions = {'float32': (np.float32, 'float'), 'float64': (np.float64, 'double')}
gpu_precision = _cfg['gpu_precision']
if gpu_precision not in _gpu_precisions:
    raise ValueError("gpu_precision must be one of %s, got %r"
                     % (sorted(_gpu_precisions), gpu_precision))
gpu_dtype, gpu_ctype = _gpu_precisions[gpu_precision]

distortion_threads_per_block = tuple(_cfg['distortion_threads_per_block'])
distortion_threads_per_block_2 = tuple(_cfg['distortion_threads_per_block_2'])
max_threads = _cfg['max_threads']

number_of_neighs = _cfg['number_of_neighs']

# If using a machine with several cuda devices
number_of_cuda_devices = _cfg['number_of_cuda_devices']
cuda_device_first_number = _cfg['cuda_device_first_number']
