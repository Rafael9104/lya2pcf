# lya2pcf
This program computes the correlation functions of the Lyman alpha forest.

## Download and configuration

If you are intended to use it for testing download the git repository with
```
$ git clone https://github.com/Rafael9104/lya2pcf
```
then move to the downloaded directory 
```
$ cd lya2pcf/
```

Install the needed libraries that include: `numpy`, `scipy`, `astropy`, `numba`, `healpy`, `mpy4pi`, `fitsio`. If you will be using a GPU you will
need `pycuda`. You can install these packages from requiriments.txt file as
```
$ conda install --yes --file requirements.txt
```
Currently, latest version of healpy is not in conda, therefore you need to install it with pip
```
pip install --force-reinstall healpy numpy==1.24
```
 
## Usage

To use it, first produce the delta files with PICCA or from another source that could produce the same format as PICCA.

Once you have the files in `DELTA_DIR/*.fits.gz` run:
```
$ python delta_reader.py --delta-dir DELTA_DIR --split-number NUMBER_FILES
```
this will produce the files `data#.npy` that will be used to compute the correlation functions. If you are using data from eBOSS extract the delta files with the following command instead:
```
$ python delta_reader_eboss.py --delta-dir DELTA_DIR
```

Next we need to compute the histograms of w and wdelta which is the more computationally expensive part. Edit `parameters.py`
to the appropiate rmax, and number of bins that you want to compute your correlation function, as well as the location of
your prefered output directory. Then execute:
```
$ mpirun -np NUMBER_OF_CORES python 2pla.py (--cpu | --gpu)
```
In the case that you are using a GPU, you need specify the number NUMBER_OF_CORES equal to the number of GPUs available.
If you are using more than one `data#.npy` file where you stored the deltas, you need to compute the correlation with the following command instead:
```
$ mpirun -np NUMBER_OF_CORES python 2pla_multiple_data.py (--cpu | --gpu)
```
To compute the distortion matrix you need to run
```
$ mpirun -np NUMBER_OF_CORES python distortion.py
```
it will produce the file `distortion.npy` in the directory `corr_dir`. CPU version is not implemented. In the case that you are using more than one `data#.npy` file where you stored the deltas, you need to compute the distortion with the following command instead:

```
$ mpirun -np NUMBER_OF_CORES python distortion_multiple_data.py
```

Now that the hardest part has finished, you just need to compute the correlation function and its error with
```
$ python post_processing.py
```
It will produce the files `correlation.npy`, `error.npy`, and for the two point correlation the `covariance.npy`. These files together with `distortion.npy` are stored in the file `correlation.out.gz` file in the directory `corr_dir` with the same format that produce PICCA.

Finally, to plot the results you can use the Jupyter notebook `two_point_analysis.ipynb`

## Machine configuration

If you have several GPU's set `number_of_cuda_devices` equal to the number of devices per node. Also edit `cuda_device_first_number`
in case you need to left free the first cuda devices in your machine.


## Things to do

1. To use multiprocessing for the cases of shared memory machines, this will ponentially reduce the memory usage.

3. Do not call for mpi4py when only used with 1 cpu

## Code Contributors

Josue De Santiago

Rafael Gutiérrez Balboa

Alma Gonzalez (Science Advisor)

Gustavo Niz (Algorithm and Science Advisor)
