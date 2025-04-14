"""
    Constants and parammeters go here
"""
import numpy as np

# IO parammeters
data_dir = './deltas_lya2pcf/'
corr_dir = './outputs/test/'



# Size and number of pixels of correlation outputs
bin_size_r = 4 #Mpc/h
rmax = 200  #Mpc/h

# Name of the keyword for the deltas in the .fit.gz files
#delta_key = "DELTA" #Use this for eBOSS, DESI EDR and DESIY5 mocks
delta_key = "DELTA_BLIND" #Use this for DESI DR1 onwards

# For the two point
rpmax = rmax #Mpc/h
rtmax = rmax #Mpc/h
numpix_rp = rpmax // bin_size_r
numpix_rt = rtmax // bin_size_r

# For the three point
numpix_mu = 10 # cos of the angle between r12 and r13 (-1 < mu < 1)
numpix_theta = 10 # angle between the sides of the triangles and the line of sight
numpix_r = rmax // bin_size_r



# Constants
lambdaa = 1215.67 #Anstrongs
la = np.log10(lambdaa)
c = 299792.458 #km/s
halfpi = np.pi / 2

# Forest parammeters
gamma = 3.8
gammaovertwo = gamma / 2
z_ref = 2.25
nside=32 # Healpix parammeter
chiquito = 2./3600.*3.14159/180. # Minimum angle to approximate sin(x)~x 2 arcsec

# Parammeters for the cosmology module
Omm  = 0.3153
OmDE = 1. - Omm
d_H0 = c / 100 # Mpc/h
# Interpolation parammeters for d_c
zmin = 0
zmax = 10
nz   = 10000


# threads_per_block = (32, 4, 4)
threads_per_block = (8, 32, 4)
# threads_per_block_2 = (16, 16)
threads_per_block_2 = (32, 32, 1)
max_threads = 1024

max_lenght = np.int32(208)

number_of_neighs = 80

# Set to true to fix broadcast errors
distributed_memory = False
# If using a machine with several cuda devices
number_of_cuda_devices = 1
cuda_device_first_number = 0
