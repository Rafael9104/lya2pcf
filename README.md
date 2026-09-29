# lya2pcf
This program computes the correlation functions of the Lyman alpha forest.

If you are on NERSC (Perlmutter), the generic install steps below need a few
extra fixes -- see the [dedicated section](#nersc-perlmutter) before you hit
them the hard way.

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
$ pip install -e ".[gpu]"
```
`pycuda` has no prebuilt wheel: it compiles against whatever CUDA toolkit
(`nvcc` plus headers) is already on the machine, so that needs to be in
place first (a module load on a cluster, or the CUDA toolkit installed
locally) -- `pip` cannot supply it.

If you only need the CPU correlation path and the extraction/post-processing
steps -- no GPU available, or just testing that part -- drop the extra:
```
$ pip install -e .
```

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

### NERSC (Perlmutter)

Installing and running on Perlmutter needs a few extra steps beyond the
generic instructions above, all specific to its Cray/HPC-SDK software stack.

**Environment**, once (a plain venv rather than conda works fine):
```
$ module load python
$ python -m venv $HOME/venvs/lya2pcf   # or somewhere in /global/common/software
```
and every session after that:
```
$ source $HOME/venvs/lya2pcf/bin/activate
```

**Building mpi4py** needs the Cray compiler wrapper, not a generic `mpicc`,
plus a clean rebuild so it actually picks up whatever `cray-mpich` module you
have loaded (a cached/prebuilt version can otherwise silently link against
the wrong one):
```
$ MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
```

**Building pycuda** fails at the final link step (`cannot find -lcurand`)
unless the linker is also pointed at NVIDIA HPC SDK's `math_libs` tree: the
HPC SDK splits the math libraries (`libcurand`, `libcublas`, ...) into a
directory separate from the core CUDA toolkit (`cuda/<ver>/lib64`), which is
the only one `pycuda`'s own build looks at. `LDFLAGS`'s `-rpath` makes the
fix permanent (baked into the compiled `.so`, not needed again after this):
```
$ find /opt/nvidia/hpc_sdk/Linux_x86_64/*/math_libs -name "libcurand.so*"  # confirm the path below still matches
$ export MATH_LIBS=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/math_libs/13.2/lib64
$ export LIBRARY_PATH=$MATH_LIBS:$LIBRARY_PATH
$ export LDFLAGS="-Wl,-rpath,$MATH_LIBS $LDFLAGS"
$ pip install -e ".[gpu]"
```

**Running anything that imports mpi4py** (`lya2pcf-correlate`,
`lya2pcf-distort`) needs two things set *every session* (unlike the two
build-time fixes above, these aren't baked into anything): mpi4py's newer
"MPI ABI" loader resolves the actual MPI library at runtime via `dlopen`,
and Cray's generic-named MPICH ABI shim
(`libmpi.so.12`, for programs that don't expect Cray's own compiler-tagged
name) lives in a `lib-abi-mpich` directory next to `cray-mpich`'s normal
`lib/`, which isn't on the linker path by default. Separately, Cray MPICH
aborts on init if GPU-aware MPI support is requested but not linked in
(`MPIDI_CRAY_init: GPU_SUPPORT_ENABLED is requested, but GTL library is not
linked`) -- lya2pcf's own MPI use is plain host-memory `bcast`/`reduce`, so
just turn that off rather than rebuild mpi4py against the GTL library:
```
$ export MPICH_GPU_SUPPORT_ENABLED=0
$ export LD_LIBRARY_PATH=/opt/cray/pe/mpich/9.1.0/ofi/gnu/12.3/lib-abi-mpich:$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
```
The `cray-mpich` path is tied to the module version actually loaded
(`module show cray-mpich` to check, or `find /opt/cray/pe/mpich -name
"libmpi*.so*"` to search more broadly) -- the one above matched
`cray-mpich/9.1.0` with `PrgEnv-gnu`; it will differ on another module set.

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

Next we need to compute the histograms of w and wdelta (plus w*rp, w*rt and w*z, which give the weighted-average coordinates of every bin) which is the more computationally expensive part. Edit `parameters.yml`
to the appropiate rmax, and number of bins that you want to compute your correlation function, as well as the location of
your prefered output directory. (`parameters.yml` is read from the current directory by default; set the `LYA2PCF_CONFIG`
environment variable to point somewhere else.) Then execute:
```
$ mpirun -np NUMBER_OF_CORES lya2pcf-correlate (--cpu | --gpu)
```
In the case that you are using a GPU, you need specify the number NUMBER_OF_CORES equal to the number of GPUs available.
This works the same way whether the extraction wrote one `data#.npy` file or many (`--split-number`): pixels are always
split evenly across the MPI ranks, and each rank loads only the files its own pixels -- plus a buffer of neighbouring
pixels from other files, so no pair is missed at a file boundary -- actually need.

To compute the distortion matrix you need to run
```
$ mpirun -np NUMBER_OF_CORES lya2pcf-distort
```
it will produce the file `distortion.npy` in the directory `corr_dir`. The distortion matrix is GPU-only by design; there is no CPU path.

Now that the hardest part has finished, you just need to compute the correlation function and its error with
```
$ lya2pcf-post
```
It will produce the files `correlation_2d.npy`, `error_2d.npy`, and for the two point correlation the `covariance.npy`. These files together with `distortion.npy` are stored in the file `correlation.out.gz` file in the directory `corr_dir` that can be used with the fitter [Vega](https://github.com/andreicuceu/vega).

Finally, to plot the results you can use the Jupyter notebook `two_point_analysis.ipynb`

## Machine configuration

The GPUs per node are detected, so there is nothing to configure for them. Launch one MPI rank per GPU:
```
$ mpirun -np <number of nodes x GPUs per node> lya2pcf-correlate --gpu
```
The run stops with a message naming both numbers if a node gets a different number of ranks than it has GPUs (use `cuda_device_first_number` to leave some out on purpose).
Set `cuda_device_first_number` in `parameters.yml` if you need to leave the first cuda devices in your machine free.
If a scheduler gives each task its own GPU through `CUDA_VISIBLE_DEVICES`, that is used as is and no check is made.


## Code Contributors

Josue De Santiago

Rafael Gutiérrez Balboa

Alma Gonzalez (Science Advisor)

Gustavo Niz (Algorithm and Science Advisor)
