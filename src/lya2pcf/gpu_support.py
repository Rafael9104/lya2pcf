"""
    Device capability and memory checks shared by the GPU backends, plus the
    kernel compile step itself.

    correlation_procedures_pycuda and distortion_procedures_pycuda compile the
    same kernels and fail in the same two ways — a device that cannot do
    float64 atomics, and buffers that do not fit in device memory — so the
    checks and their messages live here. Public API: any GPU code built on
    lya2pcf's forests (a three-point correlation, say) needs the same device
    checks and the same precision-aware compile step, so both are exported
    rather than kept private to the two-point path.
"""
import importlib.resources as _resources

import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from . import parameters as params

# atomicAdd(double*, double) was introduced in Pascal.
_MIN_FLOAT64_CAPABILITY = (6, 0)

# The kernels shipped with lya2pcf itself; a caller compiling its own source
# passes a different `path` to compile_kernels() instead.
_DEFAULT_KERNELS = _resources.files(__package__) / 'cuda_kernels.cpp'


def _gb(nbytes):
    return nbytes / 1024.**3


def check_precision_supported():
    """Fail before compiling if gpu_precision cannot work on this device.

    Every kernel in cuda_kernels.cpp is compiled together, so an unsupported
    atomicAdd stops the whole module from loading, including kernels that do
    not use it. The raw nvcc error does not mention the GPU or the setting
    that caused it, hence this check.
    """
    if params.gpu_precision != 'float64':
        return

    device = cuda.Context.get_device()
    capability = device.compute_capability()
    if capability >= _MIN_FLOAT64_CAPABILITY:
        return

    raise RuntimeError(
        "gpu_precision is 'float64', which this GPU cannot run.\n"
        "\n"
        "  device             : %s\n"
        "  compute capability : %d.%d\n"
        "  required           : %d.%d or newer\n"
        "\n"
        "atomicAdd on double only exists from compute capability %d.%d "
        "(Pascal onwards), so the CUDA kernels cannot be compiled on this "
        "device at all.\n"
        "\n"
        "Set 'gpu_precision: float32' in parameters.yml to run here. Note "
        "that float32 accumulates the histograms in single precision and "
        "loses accuracy as the bin sums grow, so it suits development and "
        "testing rather than production runs (see IMPROVEMENTS.md #7)."
        % ((device.name(),) + capability + _MIN_FLOAT64_CAPABILITY
           + _MIN_FLOAT64_CAPABILITY))


def memory_message(required, what, hints):
    """Build the out-of-memory report for `required` bytes of `what`."""
    free, total = cuda.mem_get_info()
    device = cuda.Context.get_device()
    if params.gpu_precision == 'float64':
        hints = list(hints) + [
            "set 'gpu_precision: float32' in parameters.yml, which halves "
            "every floating point buffer (at a cost in accuracy, see "
            "IMPROVEMENTS.md #7)"]
    return (
        "Not enough GPU memory for the %s.\n"
        "\n"
        "  required : %.2f GB\n"
        "  free     : %.2f GB\n"
        "  device   : %s (%.2f GB total)\n"
        "\n"
        "Options, roughly in order of how much they save:\n"
        "%s"
        % (what, _gb(required), _gb(free), device.name(), _gb(total),
           "".join("  - %s\n" % hint for hint in hints)))


def compile_kernels(path=None):
    """Compile a CUDA/C++ source file at the configured gpu_precision.

    Applies `-DMYFLOAT=<float|double>` from `parameters.gpu_precision`, so a
    kernel file compiled here always agrees with the numpy dtype the rest of
    lya2pcf uploads to the device (see `parameters.gpu_dtype`).

    `path` defaults to lya2pcf's own cuda_kernels.cpp; pass a different
    path (str, or anything with a .read_text(), such as an
    importlib.resources.Traversable) to compile another package's kernels
    at the same precision instead of hardcoding a type that could silently
    disagree with this library's buffers.

    Checks device support before compiling, since an unsupported
    atomicAdd(double*) fails every kernel in the file, not just the one
    that uses it, with an nvcc error that does not say why.
    """
    check_precision_supported()
    source = path if path is not None else _DEFAULT_KERNELS
    if hasattr(source, 'read_text'):
        text = source.read_text()
    else:
        with open(source) as f:
            text = f.read()
    return SourceModule(text, options=['-DMYFLOAT=' + params.gpu_ctype])


def require_memory(required, what, hints):
    """Raise before allocating if `required` bytes will not fit.

    Allocating first and reporting afterwards leaves partially allocated
    buffers behind and reports only the allocation that happened to fail,
    rather than the total the run actually needs.
    """
    free, _ = cuda.mem_get_info()
    if required > free:
        raise MemoryError(memory_message(required, what, hints))


def free_module_buffers(module_globals):
    """Free every device buffer held in a module's globals and drop the names.

    The pycuda modules keep their device buffers in module-level globals
    (see init()). Calling init() again for the next chunk of pixels would
    allocate the new buffers while the old ones are still alive, so the
    peak would be two chunks' worth of memory -- and require_memory()
    would refuse the second chunk. Freeing explicitly first keeps the
    peak at one chunk.
    """
    import pycuda.gpuarray as gpuarray
    freed = set()
    for name, value in list(module_globals.items()):
        if isinstance(value, gpuarray.GPUArray):
            value = value.gpudata
        if isinstance(value, cuda.DeviceAllocation):
            # The same allocation can sit behind several names
            if id(value) not in freed:
                freed.add(id(value))
                value.free()
            module_globals[name] = None
