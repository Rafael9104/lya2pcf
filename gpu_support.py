"""
    Device capability and memory checks shared by the GPU backends.

    correlation_procedures_pycuda and distortion_procedures_pycuda compile the
    same kernels and fail in the same two ways — a device that cannot do
    float64 atomics, and buffers that do not fit in device memory — so the
    checks and their messages live here.
"""
import pycuda.driver as cuda

import parameters as params

# atomicAdd(double*, double) was introduced in Pascal.
_MIN_FLOAT64_CAPABILITY = (6, 0)


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


def require_memory(required, what, hints):
    """Raise before allocating if `required` bytes will not fit.

    Allocating first and reporting afterwards leaves partially allocated
    buffers behind and reports only the allocation that happened to fail,
    rather than the total the run actually needs.
    """
    free, _ = cuda.mem_get_info()
    if required > free:
        raise MemoryError(memory_message(required, what, hints))
