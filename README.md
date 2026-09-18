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

Install lya2pcf into a virtual environment of your choice (conda, venv, or
otherwise -- pip resolves every dependency itself, including `mpi4py`,
`healpy` and `fitsio`, so no separate conda install step is needed):
```
$ pip install -e .
```
This is enough for the CPU correlation path and the extraction/post-processing
steps. Anywhere you will use a GPU -- the correlation's `--gpu` path, and the
distortion matrix, which is GPU-only -- pull in `pycuda` too:
```
$ pip install -e ".[gpu]"
```
`pycuda` has no prebuilt wheel: it compiles against whatever CUDA toolkit
(`nvcc` plus headers) is already on the machine, so that needs to be in
place first (a module load on a cluster, or the CUDA toolkit installed
locally) -- `pip` cannot supply it.

On an MPI cluster, `mpi4py`'s wheel links whatever MPI implementation it
finds on the library search path at import time, which is not necessarily
the cluster's own Slurm/fabric-aware MPI. To force it to build against the
`mpicc` actually on `PATH` (matching whatever `mpirun`/`srun` will launch
with) instead of using the prebuilt wheel:
```
$ pip install --no-binary mpi4py -e ".[gpu]"
```
Either way this puts the `lya2pcf-*` commands used below on your `PATH`,
and makes `lya2pcf` importable as a library from any directory.

## Usage

To use it, first produce the delta files with PICCA or from another source that could produce the same format as PICCA.

Once you have the files in `DELTA_DIR/*.fits.gz` run:
```
$ lya2pcf-extract --delta-dir DELTA_DIR --split-number NUMBER_FILES
```
this will produce the files `data#.npy` that will be used to compute the correlation functions. If you are using data from eBOSS extract the delta files with the following command instead:
```
$ lya2pcf-extract-eboss --delta-dir DELTA_DIR
```

Next we need to compute the histograms of w and wdelta which is the more computationally expensive part. Edit `parameters.yml`
to the appropiate rmax, and number of bins that you want to compute your correlation function, as well as the location of
your prefered output directory. (`parameters.yml` is read from the current directory by default; set the `LYA2PCF_CONFIG`
environment variable to point somewhere else.) Then execute:
```
$ mpirun -np NUMBER_OF_CORES lya2pcf-correlate (--cpu | --gpu)
```
In the case that you are using a GPU, you need specify the number NUMBER_OF_CORES equal to the number of GPUs available.
If you are using more than one `data#.npy` file where you stored the deltas, you need to compute the correlation with the following command instead:
```
$ mpirun -np NUMBER_OF_CORES lya2pcf-correlate-multi (--cpu | --gpu)
```
To compute the distortion matrix you need to run
```
$ mpirun -np NUMBER_OF_CORES lya2pcf-distort
```
it will produce the file `distortion.npy` in the directory `corr_dir`. The distortion matrix is GPU-only by design; there is no CPU path. In the case that you are using more than one `data#.npy` file where you stored the deltas, you need to compute the distortion with the following command instead:

```
$ mpirun -np NUMBER_OF_CORES lya2pcf-distort-multi
```

Now that the hardest part has finished, you just need to compute the correlation function and its error with
```
$ lya2pcf-post
```
It will produce the files `correlation_2d.npy`, `error_2d.npy`, and for the two point correlation the `covariance.npy`. These files together with `distortion.npy` are stored in the file `correlation.out.gz` file in the directory `corr_dir` that can be used with the fitter [Vega](https://github.com/andreicuceu/vega).

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
