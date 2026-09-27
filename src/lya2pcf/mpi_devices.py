"""
    Assigns each MPI rank a GPU on its own node, and checks that the number of
    ranks launched (mpirun -np N) matches the GPUs that are there.

    The device count is not configured any more: it is what the machine reports
    (pycuda.driver.Device.count()), and the number of ranks sharing a node is
    what MPI reports (a communicator split by shared memory). The right N for
    mpirun is therefore nodes x GPUs per node, and a mistake in it is reported
    with both numbers instead of surfacing as a CUDA error, or as two ranks
    silently sharing one GPU. See IMPROVEMENTS.md #17.

    Must run before pycuda.autoinit is imported: the device is chosen through
    the CUDA_DEVICE environment variable, which autoinit reads when it creates
    its context.
"""

import os
import socket

from mpi4py import MPI

from . import parameters as params


class DeviceMismatch(RuntimeError):
    pass


def node_layout(comm):
    """(local_rank, local_size): this rank's index among the ranks on its
    node, and how many ranks share the node."""
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    local = node_comm.Get_rank(), node_comm.Get_size()
    node_comm.Free()
    return local


def visible_gpu_count():
    """GPUs this process can see (honours CUDA_VISIBLE_DEVICES). Initialises
    the driver but creates no context, so it is safe before autoinit."""
    import pycuda.driver as cuda
    cuda.init()
    return cuda.Device.count()


def check_layout(layouts, first_device, bound_by_launcher=False):
    """Compares, for every node, the ranks placed on it with the GPUs it has.

    layouts   one (host, ranks_on_node, gpus_visible) per rank.
    Returns the list of error messages, one per node that does not match.
    bound_by_launcher: CUDA_VISIBLE_DEVICES is set and every rank sees exactly
    one GPU -- a scheduler that gives each task its own GPU -- so there is
    nothing to compare.
    """
    nodes = {}
    for host, ranks, gpus in layouts:
        nodes[host] = (ranks, gpus)
    errors = []
    for host, (ranks, gpus) in sorted(nodes.items()):
        usable = gpus - first_device
        if bound_by_launcher and gpus == 1 and first_device == 0:
            continue
        if usable < 1:
            errors.append("%s: cuda_device_first_number is %d but the node has %d GPU(s)."
                          % (host, first_device, gpus))
        elif ranks > usable:
            errors.append("%s: %d ranks share the node but only %d GPU(s) are usable, so several "
                          "ranks would run on the same GPU. Run mpirun with -np <nodes> x <GPUs per node>."
                          % (host, ranks, usable))
        elif ranks < usable:
            errors.append("%s: %d ranks for %d usable GPU(s), so %d GPU(s) would sit idle. "
                          "Run mpirun with -np <nodes> x <GPUs per node>."
                          % (host, ranks, usable, usable - ranks))
    return errors


def assign_gpu(comm):
    """Sets CUDA_DEVICE for this rank and validates the -np against the GPUs.
    Collective: call it on every rank. Raises DeviceMismatch on all ranks if a
    node has a different number of ranks than usable GPUs.
    Returns the device index assigned to this rank."""
    local_rank, local_size = node_layout(comm)
    gpus = visible_gpu_count()
    layouts = comm.gather((socket.gethostname(), local_size, gpus), root=0)

    first = params.cuda_device_first_number
    bound = 'CUDA_VISIBLE_DEVICES' in os.environ and gpus == 1
    if comm.Get_rank() == 0:
        errors = check_layout(layouts, first, bound_by_launcher=bound)
    else:
        errors = None
    errors = comm.bcast(errors, root=0)

    if errors:
        raise DeviceMismatch("mpirun -np %d does not match the GPUs:\n  %s"
                             % (comm.Get_size(), "\n  ".join(errors)))

    device = 0 if bound else first + local_rank
    os.environ['CUDA_DEVICE'] = str(device)
    return device
