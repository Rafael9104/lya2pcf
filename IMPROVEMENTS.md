# lya2pcf — planned improvements

Working notes from 2026-09-10. Not committed to git by default (personal
planning doc, not project documentation) — delete, commit, or move it
wherever you like.

## Still to do (2026-09-18)

Every other item on this list is done. Remaining:

- **#4** — In-memory pipeline (skip `data.npy` round-trip)
- **#6** — Single host-memory copy shared across multiple GPUs (one node)
- **#11** — Add a rebinning (coadding) procedure to lya2pcf
- **#13** — Forests are padded to `max_lenght`, wasting a large share of
  GPU memory
- **#14** — `np.save` of the forest objects is slow and spikes RAM at
  scale (partially addressed — see the item for what's left)
- **#16** — `number_of_neighs` should be derived from the data, not a
  config guess ([open issue #1](https://github.com/Rafael9104/lya2pcf/issues/1))
- **#17** — Multi-GPU runs split "how many GPUs" across two unrelated
  places
- **#18** — mpi4py is a hard dependency even for a single process

## 1. `src/lya2pcf/` layout + pip-installable package

Currently everything is a flat collection of scripts at the repo root
(`delta_reader.py`, `2pla.py`, `forest_class.py`, ...) that talk to each
other via `from parameters import *` and relative paths. Move to:

```
src/lya2pcf/
    __init__.py
    forest_class.py
    cosmology.py
    delta_reader.py
    delta_reader_eboss.py
    correlation_procedures_{cpu,pycuda}.py
    distortion*.py
    post_processing.py
    cuda_kernels.cpp   (as package data)
pyproject.toml
```

with console-script entry points (`lya2pcf-extract`, `lya2pcf-correlate`,
`lya2pcf-distort`, `lya2pcf-post`, or a single `lya2pcf` CLI with
subcommands) so `process_all.sh` becomes a thin wrapper around installed
commands instead of `python delta_reader.py`.

**Why:** makes the code usable outside this one directory (other students,
clusters, notebooks) via `pip install lya2pcf` / `pip install -e .`, and
gives a natural place to put the config loader from item 2.

### Design it as a library, not just an application

lya2pcf should be usable as a dependency by other correlation code —
a three-point correlation built on the same forests and the same GPU
machinery is the obvious case, and it is worth designing for now rather
than reopening the packaging later.

The test is whether a separate program can reuse the expensive parts
without copying them. That means exporting, as public API:

- **`upload_forests(data, pixel_list) -> handles`** — the host-to-device
  upload currently living inside `correlation_procedures_pycuda.init()`:
  packing the forests into the flat `gran_*` arrays, allocating, copying,
  and computing `max_lenght` from the data. This is ~45 lines that any
  GPU code over the same forests needs, and it is identical whatever
  correlation is being computed afterwards.
- **`compile_kernels(path)`** — the `SourceModule` wrapper that applies
  `gpu_precision` via `-DMYFLOAT` (#7). Exporting it lets another
  package compile *its own* `.cu` file at the same precision, instead of
  hardcoding a type and silently disagreeing with the library it is
  built on.
- **the `gpu_support` checks** — device capability and the
  pre-allocation memory report, which are useful to any caller and not
  specific to the two-point path.
- the obvious data pieces: `quasar`, `cosmology`, the config loader, and
  delta extraction returning a `data` dict (which is #4).

None of this is speculative structure for its own sake: it is the code
that already exists, exported rather than kept private. The consumer
then writes only its own kernels and its own per-pixel routine.

A corollary worth stating: **do not delete apparently-unused code during
the move without checking.** The three-point hooks already in this repo
(`numpix_mu`, `numpix_theta`, `numpix_r`, the `numpix_d` upload in
`correlation_procedures_pycuda.init()`, and the
`precompute_distance_and_angles` kernel, which no script here calls)
are leftovers from an earlier split. They are dead *here*, so they can
go — but check before assuming that of anything else.

**Depends on:** nothing strictly, but do it *after* item 2 (YAML config) —
otherwise you restructure imports once for packaging and again for config
loading.

**Done 2026-09-17** on branch `feat/src-layout`, PR #15, merged into
`main` 2026-09-18. Everything moved into
`src/lya2pcf/`, `import X` became `from . import X` throughout, and
`pyproject.toml` gives console scripts `lya2pcf-extract[-eboss]`,
`lya2pcf-correlate[-multi]`, `lya2pcf-distort[-multi]`, `lya2pcf-post`
(the "several entry points" option, since the drivers are still
separate — see #15). `2pla.py` / `2pla_multiple_data.py` are renamed to
`two_point.py` / `two_point_multiple_data.py`, since `2pla` cannot be a
Python module name (`import 2pla` is a syntax error) and so cannot back
a `module:function` entry point. `process_all.sh` and the README now use
the installed commands; the analysis notebook's import cell points at
`lya2pcf.parameters` / `lya2pcf.plot_auxiliars` instead of the flat
modules.

`delta_reader.py` and `delta_reader_eboss.py` ran their argparse/extraction
logic at module import time, which an installed entry point cannot call —
wrapped in `main()`, same for the `if __name__ == '__main__':` bodies of
the other drivers. `record_from_deltas` stays a top-level function (needed
as-is for `multiprocessing.Pool.map` to pickle it by qualified name).

**The library API from the "design it as a library" section above is
done, except the part deferred to #4.** `correlation_procedures_pycuda.py`
now has a standalone `upload_forests(data, pixel_list) -> ForestBuffers`
(a dataclass of the device pointers and `max_lenght`), with `init()`
calling it and unpacking the result into its existing globals so
`two_point_per_pixel` is untouched. `gpu_support.compile_kernels(path)`
wraps the `SourceModule` + `-DMYFLOAT` step, defaulting `path` to the
package's own `cuda_kernels.cpp` via `importlib.resources`; both
`correlation_procedures_pycuda.py` and `distortion_procedures_pycuda.py`
now call it instead of each separately opening the file and calling
`check_precision_supported()`. `distortion_procedures_pycuda.init()` was
deliberately *not* split into an `upload_forests`-style function: its
buffers (`etas12/21/...`, `x12/y12/z12/r12`) are distortion-specific, not
the generic per-forest upload a different correlation would reuse.
`quasar`, `cosmology` and `parameters` (the config loader) are exposed at
`lya2pcf.__init__`, deliberately excluding `gpu_support` and the two
`*_pycuda` modules — they import `pycuda` at module level, so re-exporting
them from `__init__.py` would make `import lya2pcf` fail on CPU-only
nodes; a caller that needs them imports the submodule directly, same as
today's scripts already do conditionally on `--cpu`/`--gpu`. Delta
extraction returning a `data` dict in memory is explicitly left for #4;
`delta_reader.py` still only writes `data*.npy`.

The dead three-point hooks noted above (`numpix_mu`, `numpix_theta`,
`numpix_d`, `precompute_distance_and_angles`) were moved as-is, not
removed — this PR is the file move, not a cleanup pass.

**Breaking change for existing `data*.npy` files.** These pickle `quasar`
objects tagged with whatever module path `forest_class` had at extraction
time. A file written before this move references the flat `forest_class`
module, which no longer exists as a top-level import. Old files still load
because `forest_class.py` now does
`sys.modules.setdefault('forest_class', sys.modules[__name__])` on import
— pickle resolves a module name against `sys.modules` before touching
`sys.path`, and every driver already imports `forest_class` before calling
`np.load(..., allow_pickle=True)` (this is, in fact, why `two_point.py`
imports `quasar` without otherwise using it — the existing code already
relied on this same trick for a different reason). **Verified**: loaded
the pre-existing `deltas_lya2pcf/data1.npy` (committed before this move)
successfully after the change; `forest.__class__.__module__` correctly
came back as `lya2pcf.forest_class`. New extractions naturally pickle
under the new path and need no shim.

**Verified against the real DR1 set** (same 4 files as #15, `deltas_dr1/`,
`gpu_precision: float32`, GTX 970): extraction reproduces the exact
figures already on record (1446 forests, longest 967) unchanged.
Comparing the packaged GPU correlation (`lya2pcf-correlate --gpu`) against
the pre-move code (`git worktree` of the previous commit, same input,
same config) over all 16 pixels: total `w_hist` differs by `2.8e-8`
relative, and the largest single-bin difference is consistent with the
float32 `atomicAdd` reordering non-determinism already measured in #7 (not
a regression from the refactor — GPU sums were never expected to be
bit-identical between runs). Also compiled and ran the distortion kernels
(`distortion_procedures_pycuda`) through the new shared `compile_kernels()`
path with no errors.

**CPU correlation path, verified after the fact:** the same-input
comparison against the pre-move code (`correlation_procedures_cpu.py`,
only import statements changed) finished after this was first written —
numba plus the O(forests²) pair search took about 8 minutes on the
16-pixel DR1 set, run in the background while the rest of this item was
wrapped up. All 16 pixel histograms (`w_hist` and `dw_hist`) came back
**bit-identical** to the pre-move code, as expected for a path with no
logic changes and no floating-point non-determinism (unlike the GPU
path above, the CPU kernel does an ordinary in-order sum, not atomics).

**Still not executed end-to-end:** `two_point_multiple_data.py`,
`distortion.py`, `distortion_multiple_data.py`, `delta_reader_eboss.py`
and `post_processing.py` were checked by reading the diff (import-only
changes plus the `main()` wrap) and, for `post_processing.py`, by
actually running `lya2pcf-post` on real correlation output (see above) —
none of them changed logic, all of them changed only imports and the
top-level-code-to-function wrapping.

## 2. `parameters.py` → `parameters.yml`

Replace the star-import global-constants module with a YAML config file
plus a loader (`lya2pcf.config.load(path)` returning a dataclass or
`SimpleNamespace`). Things to watch for since they don't map 1:1 to YAML:

- Derived values (`numpix_rp = rpmax // bin_size_r`, `OmDE = 1 - Omm`,
  `d_H0 = c / 100`, `gammaovertwo = gamma / 2`, `la = np.log10(lambdaa)`)
  should be computed by the loader after reading the raw YAML values, not
  stored twice.
- `threads_per_block` tuples and `np.int32(...)` wrapping are Python/CUDA
  concerns, not config — keep those as code constants or cast on load.
- `delta_reader.py`'s `substitute_parameter()` currently rewrites
  `parameters.py` in place via `fileinput` to persist the measured
  `max_lenght` back into the params file. This **cannot** work once params
  live outside an installed package (or even now, it's fragile — editing
  live source with text substitution). Replace with read-modify-write of
  the YAML file, or better, stop round-tripping `max_lenght` through a
  config file at all and pass it explicitly between the extract and
  correlate steps (natural fit with item 4, in-memory pipeline).

**Why:** current setup means every script implicitly depends on a
`parameters.py` sitting in the CWD or importable path — awkward once the
package is pip-installed and used from arbitrary directories. YAML also
makes per-run configs easy to version and diff.

**Depends on:** do before item 1's import restructuring settles, since
every file currently does `from parameters import *`.

**Done 2026-09-10** on branch `feat/parameters-yaml`, issue #4, PR #5,
merged. `parameters.yml` plus `parameters.py` as the loader, read once
at import time, with every derived value (`numpix_rp`, `OmDE`, `d_H0`,
`gammaovertwo`, `la`, `chiquito`, and later `gpu_dtype`/`gpu_ctype` from
#7) computed from the raw YAML values rather than stored twice.
`delta_reader.py`'s `max_lenght` self-rewrite via `fileinput` is gone
entirely — `max_lenght` isn't in the config at all any more, computed
fresh from whichever data is actually loaded (see #1's `upload_forests`
and `distortion_procedures_pycuda.init()`), which is what #12 leans on.

One deviation from the plan as originally written: instead of a
`lya2pcf.config.load(path)` function returning a dataclass/
`SimpleNamespace`, `parameters.py` itself *is* the loaded config,
accessed as `params.X` after `import parameters as params` (later
`from . import parameters as params` once #8 and #1 landed) — simpler,
and it's what #1's packaging move built on rather than reopening.

## 3. Generalize hardcoded metadata/column keys (blinding-safe extraction)

`delta_key = "DELTA"` in parameters is already configurable, but it's read
from a fixed metadata column set (`LOS_ID`, `TARGETID`, `RA`, `DEC`,
`LAMBDA`, `WEIGHT`) in `delta_reader.py`. You mentioned some delta files
carry redshift-like info under a different key when blinding is applied
(e.g. `Z` vs `Z_UNBLINDED` or similar — worth checking the current DESI/
Picca blinding convention when you get to this, since you weren't sure of
the exact name). Fix: make every metadata column name configurable in
the YAML config (with sane defaults), or try a small list of known
aliases per field and fall back gracefully with a clear error if none
match, instead of a hard `KeyError` deep in `record_from_deltas`.

**Confirmed instance (2026-09-10):** ran `delta_reader.py` against a real
blinded DESI dataset (`deltas_dr1/`) as a smoke test and hit exactly this
class of bug — but on the delta values themselves, not a redshift field.
The FITS extension holding the flux deltas is named `DELTA_BLIND`, not
`DELTA`:

```
extnum hdutype         hduname[v]
0      IMAGE_HDU
1      IMAGE_HDU       LAMBDA
2      BINARY_TBL      METADATA
3      IMAGE_HDU       DELTA_BLIND
4      IMAGE_HDU       WEIGHT
5      IMAGE_HDU       CONT
```

`record_from_deltas` (`delta_reader.py`) does
`deltafile[delta_key].get_dims()` with the default `delta_key = "DELTA"`,
so this raised `OSError: extension not found: delta (case insensitive)`
— a multiprocessing-wrapped traceback that obscures the real cause.
Worked once `delta_key` was manually set to `"DELTA_BLIND"`. For contrast,
`METADATA`'s columns on this same dataset are
`LOS_ID, RA, DEC, Z, MEANSNR, TARGETID, NIGHT, PETAL, TILE` — `Z` is
present and *not* renamed here, so the redshift-column case you
remembered may be dataset/blinding-scheme-specific and worth re-checking
against whatever produced it. Either way, both are instances of the same
underlying problem: any single hardcoded column/extension name can be
wrong for a given dataset, and today that fails deep inside a worker
process with a confusing error instead of a clear one up front.

**Why:** avoids a silent/obscure crash the next time you're handed a
delta set produced under a different blinding scheme.

**Depends on:** item 2 (config), for a clean place to put the extra keys.

**Done 2026-09-11** on branch `feat/metadata-keys`. The flat `delta_key`
became a `delta_keys` block in `parameters.yml` covering all eight names
`delta_reader.py` reads — the four HDUs (`delta`, `lambda`, `weight`,
`metadata`) and the four metadata columns (`los_id`, `targetid`, `ra`,
`dec`).

Rather than alias lists, a wrong name now produces a message that shows
what the file actually holds, which answers the question in one step:

```
ValueError: These names from delta_keys in the configuration are not in
the delta file:

  file: ./deltas_dr1/delta-94.fits.gz

  delta_keys.delta     = 'DELTA'  (not found)

The extensions in this file are:
  LAMBDA, METADATA, DELTA_BLIND, WEIGHT, CONT
```

The check runs once up front on the first file as well as inside each
worker, because an exception raised in a `multiprocessing` worker comes
back wrapped in a `RemoteTraceback` — which is exactly how the original
`DELTA` failure presented, and part of why it was unclear.

Two things left undone, deliberately:

- **`delta_reader_eboss.py` still hardcodes its names** (`DELTA`,
  `WEIGHT`, `LOGLAM`, and the header keys `FIBERID`, `PLATE`,
  `THING_ID`, `RA`, `DEC`). It reads the older per-HDU eBOSS layout with
  a different set of names — `LOGLAM` rather than `LAMBDA`, per-forest
  headers rather than a metadata table — so it needs its own block
  rather than sharing this one. It also ignores `delta_key` today, so
  this is not a regression.
- While changing the key lookups, `metadata` and `lambd_list` were
  hoisted out of the per-forest loop; the old code re-read the whole
  metadata table once per forest, which is quadratic in forests per
  file. Measured only ~2-4% on the DR1 files (~360 forests each), so it
  is not the win it first looks like at this size, but it would matter
  on files with many more forests.

## 4. In-memory pipeline (skip `data.npy` round-trip)

`delta_reader.py` always writes `data_dir/data*.npy`, and `2pla.py` /
`distortion.py` always `np.load(...)` it back. You want the option to
build the `data` dict in memory and immediately hand it to the
correlation/distortion step in the same process, no disk round-trip.

Concretely: factor `record_from_deltas` + the pixel-splitting/neighbor
logic in `delta_reader.py` into a function that *returns* the `data`
dict (already almost the case — the script just also serializes it),
and give `2pla.py`/`distortion.py` an entry point that accepts a `data`
dict directly rather than only a path. The CLI scripts become thin
wrappers: `extract → save` and `extract → correlate` respectively.

**Why:** for quick/small runs (e.g. testing, `--verbose` runs) the
save/load of potentially large `.npy` files is pure overhead.

**Depends on:** pairs naturally with item 1 (package layout) since this
is exactly the shape a library API should have; also depends on item 2
for the `max_lenght` handling mentioned above (currently threaded through
`parameters.py`, needs to become an explicit return value/argument when
extraction and correlation share a process).

**Caveat:** this only applies to the single-process (`--cpu` /
single-GPU, no MPI split across files) path. `2pla.py`'s multi-rank
MPI flow loads independently per rank, which is where item 6 also
matters.

## 5. Fix `data_dir`/`corr_dir` trailing-slash bug

Every path is built with plain string concatenation:
`corr_dir + 'thread_' + ...`, `data_dir + 'data1.npy'`,
`corr_dir + name_partials + str(pixel)`, etc. (see `2pla.py`,
`2pla_multiple_data.py`, `distortion*.py`, `post_processing.py`,
`delta_reader*.py`). If the configured directory doesn't end in `/`,
these silently become filename prefixes instead of directory paths
(e.g. `outputs/testdata1.npy`).

Fix: switch to `pathlib.Path` (or `os.path.join`) everywhere paths are
built, or normalize once at config-load time by appending `os.sep` if
missing. The latter is a 2-line fix; the former is more correct and a
good match for item 2's config loader (store `Path` objects, not
strings).

**Why:** silent, hard-to-notice failure mode — no error, just wrong
output location or worse, files mixed into the wrong directory.

**Depends on:** nothing; cheapest, most isolated fix on this list. Good
first PR to build confidence in the new structure, or fine to do
standalone today regardless of the rest.

**Done 2026-09-10** on branch `fix/path-joining`, issue #2, PR #3,
merged. Every `data_dir + ...` / `corr_dir + ...` concatenation site
replaced with `os.path.join(params.data_dir, ...)` /
`os.path.join(params.corr_dir, ...)`. Reconfirmed with a grep across
`src/lya2pcf/` after #1's move: no `data_dir +`/`corr_dir +`
concatenation remains anywhere.

## 6. Single host-memory copy shared across multiple GPUs (one node)

Current architecture: `2pla.py` is launched under `mpirun -np N`, one MPI
*process* per GPU. Each rank independently does
`np.load(data_dir + 'data1.npy')` (`2pla.py:52`) and then
`correlation_procedures_pycuda.init()` uploads its own full copy to its
own GPU (`os.environ['CUDA_DEVICE']` is set per-rank before that module's
`pycuda.autoinit` runs). Since MPI ranks are separate OS processes with
separate address spaces, "upload once, share across GPUs" can't be done
just by refactoring Python-level code — the CPU-side copies really are in
different processes today.

Two realistic paths, in increasing order of invasiveness:

- **a. Shared host memory across ranks on the same node** (smaller
  change): keep the MPI-process-per-GPU model, but have rank 0 build the
  `data` arrays once into a POSIX shared-memory segment
  (`multiprocessing.shared_memory.SharedMemory` or a `numpy.memmap` on
  `/dev/shm`), and have the other ranks on the same node attach to it
  read-only instead of loading their own `.npy` copy. Saves host RAM
  (the thing you said is currently duplicated N times) without touching
  the GPU-upload code at all — each rank still does its own
  `cuda.memcpy_htod` from the shared host buffer to its own device.
- **b. One process, one thread per GPU** (bigger rewrite): drop MPI
  (or keep it only for multi-*node* runs) and instead have a single
  process spawn one Python thread per local GPU, each creating its own
  pycuda context. Since threads share the process heap, the `data` numpy
  arrays are naturally shared with zero duplication. This requires
  turning `correlation_procedures_pycuda.py`'s module-level globals
  (`gran_dc_d`, `mod`, `pair_correlation`, etc.) into an instance/class
  per thread, since right now a second `init()` call in the same process
  would clobber the first thread's GPU handles.

**Recommendation:** start with (a) — it directly solves the RAM
duplication you described, is a localized change (touches only how
`data` is loaded in `2pla.py`/`2pla_multiple_data.py`, not the CUDA
upload/kernel code), and keeps the existing MPI multi-node story intact.
Only reach for (b) if you outgrow shared-memory host RAM too, or want to
simplify the process model for other reasons.

**Depends on:** logically independent from 1–5, but it's the riskiest
and most architecture-level change on this list — do it last, once the
rest of the codebase is settled, so you're not restructuring packaging
*and* the process/memory model at the same time.

(This is the same idea the README's old "Things to do" list had as
"use multiprocessing for shared memory machines to reduce memory
usage" — folded in here rather than kept as a separate note, since #6
already covers it in more depth. See #18 for the README's other
"Things to do" entry, about not requiring mpi4py for a single process.)

## 7. float32 vs float64: the CUDA kernel won't build on older GPUs

**Discovered 2026-09-10**, while smoke-testing the GPU correlation path
on this machine (GTX 970). Importing `correlation_procedures_pycuda`
fails at `SourceModule(f.read())` — i.e. at kernel compile time, before
any of our Python runs:

```
pycuda.driver.CompileError: nvcc compilation of kernel.cu failed
[command: nvcc --cubin -arch sm_52 ...]
kernel.cu(155): error: no instance of overloaded function "atomicAdd"
                       matches the argument list
                       argument types are: (myfloat *, double)
```

**Cause:** `cuda_kernels.cpp` has `using myfloat = double`, and the
histogram accumulation uses `atomicAdd` on it. A native
`atomicAdd(double*, double)` requires compute capability ≥ 6.0
(Pascal). The GTX 970 is sm_52 (Maxwell), so the overload doesn't
exist and compilation fails. `atomicAdd(float*, float)` has been
supported since CC 2.0, which is why the single-precision path builds
fine anywhere.

**This explains the uncommitted diff that was sitting in the working
tree at the start of this session** (reverted on 2026-09-10 as "not
important"): it flipped `myfloat` from `double` to `float` in
`cuda_kernels.cpp`, `correlation_procedures_pycuda.py`, and the array
dtypes in `forest_class.py` — almost certainly a local workaround for
exactly this error, not a deliberate precision experiment.

Note `distortion_procedures_pycuda.py` already allocates everything as
`np.float32` unconditionally, so the two GPU modules don't currently
agree on precision.

**Options:**

- **a. Make precision a config value** (now that #2 exists): one
  setting driving both the numpy dtype and the `using myfloat = ...`
  line, injected into the kernel source at `SourceModule()` time
  (string substitution or an `-D` compiler define) instead of being
  hand-edited in two files that must be kept in sync. This is the
  cleanest fix and also kills the "change this line as well as the
  appropriate lines in cuda_kernels.cpp" comments in both files.
- **b. Ship a software `atomicAdd(double*)` fallback** — the standard
  `atomicCAS`-loop implementation, guarded by
  `#if __CUDA_ARCH__ < 600`. Keeps double precision working on old
  cards, at a significant speed cost on those cards.
- **c. Just require CC ≥ 6.0** and document it — fine for the
  production cluster, but then this repo can't be smoke-tested on this
  workstation at all, which is inconvenient for development.

**Recommendation:** (a), with (b) added only if double precision on
pre-Pascal hardware turns out to actually matter.

**Done 2026-09-11** on branch `feat/gpu-precision` via option (a):
`gpu_precision: float32|float64` in `parameters.yml` drives both the
numpy dtype of the device buffers and `-DMYFLOAT=<float|double>` passed
to `nvcc`, so the two can no longer disagree. Default stays float64.
CPU-side forest data is untouched (always float64, converted on
upload), and the distortion kernels remain float32 — they are written
with fixed `float` types, and only `pair_correlation` was ever written
to be precision-agnostic.

**Measured cost of float32 — this matters scientifically.** Comparing
GPU float32 against the CPU float64 reference on one healpix pixel of
the 4-file DR1 set, for ξ = dw/w (what `post_processing.py` consumes):

```
spread of xi itself (std)        1.627e-02
rms |xi_gpu - xi_cpu|            2.936e-04   (1.8% of that spread)
max |xi_gpu - xi_cpu|            6.657e-03   (41% of that spread)
```

Binning is unaffected — both agree on exactly which 2200 bins are
populated — so this is purely accumulation/storage precision, not a
geometry difference. Note float32 eps is 1.2e-7 while the observed
error is ~1e-3 relative: the loss comes from summing many small
contributions into large per-bin totals (mean w per bin ≈ 1.4e5 here),
and it grows with the number of pairs, so a full dataset will be worse
than this measurement, not better.

**float32 GPU results are also not bit-reproducible.** Running the same
pixel four times in one process on the same data gave two distinct
sums, spread 1.0e-07 relative. `atomicAdd` completes in whatever order
the scheduler produces, and in single precision that reordering changes
the rounding. The same non-determinism exists at float64, but ~1e-16
makes it invisible. Worth knowing before chasing a "regression" between
two runs, and another reason not to use float32 for results anyone will
try to reproduce.

**Conclusion:** treat float32 as a development/compatibility mode for
running on pre-Pascal cards, not as a production setting. If float32
throughput ever becomes desirable in production, the fix is not a
blanket precision switch but Kahan/pairwise summation, or float32
storage with float64 accumulators for the histogram bins only.

**The distortion path is deliberately not covered by this setting** —
see #10, which records why and what it would take.

The float64 GPU path could not be run on this workstation at all (Pascal
or newer only), so these changes were originally exercised in float32
only, with the distortion path verified as compiling but never executed.

**Resolved 2026-09-18:** Josue confirmed on production Pascal-or-newer
hardware that both the two-point correlation and the distortion matrix
run successfully at `gpu_precision: float64`. That was a run-completes
check, not a numeric comparison against a reference (unlike the float32
vs CPU comparison already measured above) — worth keeping in mind if a
result from this path is ever questioned, since "runs" and "produces the
expected numbers" are still two different claims. See #10 for the
distortion side of the same confirmation.

**Depends on:** #2 for the config plumbing, otherwise independent.

## 8. Drop `from parameters import *`

Every module does `from parameters import *`, so at a use site there's
nothing distinguishing a config value from a local: in
`angmax = 2*np.arcsin(0.5*rtmax/dmin)`, `rtmax` is config and `dmin`
was computed three lines up, and nothing says so. It also means any key
added to `parameters.yml` silently appears in every module's namespace,
and a typo'd local can shadow a config value with no error.

Deliberately left alone during #2 (keeping `parameters` as the import
name is what kept that change to 7 files instead of 13). Target:
`import parameters as params` with `params.rtmax` at use sites.

**Caveat — this is not a pure find-and-replace.**
`correlation_procedures_cpu.py:47` `pair_correlation` is
`@jit(nopython=True)` and closes over `chiquito`, `rpmax`, `rtmax`,
`numpix_rp`, `numpix_rt` and `shape_hist` as globals. Numba freezes
globals as compile-time constants on first call, and module-attribute
access inside nopython code doesn't always resolve the same way. Those
values need to be passed in as arguments (or bound to locals outside
the jitted scope). Note `shape_hist` is a *runtime* global there, set
by `init()` — it only works today because the first call happens after
`init()`, which is its own latent fragility.

**Depends on:** best done as part of #1, since packaging restructures
imports anyway — doing it separately means touching every use site
twice.

**Done 2026-09-10** on branch `feat/parameters-yaml`, ahead of #1 after
all. Numba 0.67 turned out to resolve `params.X` fine in `nopython`
mode (tested before committing to the approach), so no special-casing
was needed and the CPU correlation results came out bit-identical.
Three things it flushed out, recorded as #9 below.

## 9. Things the star-import removal exposed

Found 2026-09-10 while doing #8, recorded here so they weren't lost.
Status per sub-item: (a) done, (b) resolved with a residual noted below,
(c) promoted to its own item, #16.

**a. `delta_reader.py` had an undefined name — done**, fixed as part of
#8, it was a one-word change: the "no delta files found" branch printed
`delta_dir`, which never existed in any namespace — it would have
raised `NameError` instead of the intended error message. Never caught
because `import *` makes static checking impossible; `pyflakes` found
it in seconds once the star imports were gone.
`delta_reader_eboss.py` already had `args.delta_dir` correctly.

**b. Config `threads_per_block` is dead for the two-point
correlation.** `correlation_procedures_pycuda.two_point_per_pixel()`
assigns its own local `threads_per_block = (1, 16, 16)` (with a comment
that it can't change unless the kernel changes), which shadows the
config value entirely. So the `parameters.yml` setting only affects
`distortion_procedures_pycuda.py`. **Resolved 2026-09-10** by renaming
the keys to `distortion_threads_per_block` /
`distortion_threads_per_block_2`, so the narrower scope is explicit;
the correlation kernel's block size stays hardcoded, since changing it
requires changing the kernel. Still open: `max_threads` is referenced
by no module at all (confirmed by an AST sweep of every `params.*`
access) and looks like dead config.

`distortion_threads_per_block_2` was simplified further on
2026-09-18, in `simplify/2d-threads-per-block`: it only ever configured
`order_active`, the one kernel in this file that is genuinely
2D-indexed (confirmed by reading it -- only `.x`/`.y` are read, never
`.z`), so carrying a trailing `1` in config for a dimension the kernel
never reads was pure noise. Renamed to `2d_threads_per_block: [32, 32]`
(a real 2-tuple) in `parameters.yml`; `distortion_procedures_pycuda.py`
appends the unused `z=1` itself when building the `block=` argument,
since pycuda's `block=` requires exactly 3 ints regardless (confirmed
empirically: a 2-tuple raises `ArgumentError`, CUDA block dims are
always 3D at the driver level). Verified end to end: `order_active`
launches correctly and `distortion_per_pixel` produces sane, non-zero
output with the new config.

`pair_correlation`'s hardcoded `threads_per_block = (1, 16, 16)`
(noted above as staying hardcoded, "since changing it requires changing
the kernel") turned out to share `2d_threads_per_block` too, same day:
its `y`/`z` are genuine strided-loop parameters (same shape as
`order_active`'s), so `threads_per_block = (1,) + params.threads_per_block_2d`
reuses the same config value for them, changing the launch from
`(1, 16, 16)` to `(1, 32, 32)`. `x` stays a hardcoded `1` in code, not
part of the shared config -- it isn't a spare dimension like
`order_active`'s `z`, it's load-bearing: the kernel reads its
pixel-in-forest1 index from `blockIdx.x`, never from `threadIdx.x`, so
`blockDim.x > 1` would re-run the same accumulation redundantly and
double-count into the histogram. **Verified against the real DR1 set**:
compared the new `(1, 32, 32)` block against the old `(1, 16, 16)` in
the same session (`git stash` to isolate the one-line change) -- total
`w_hist` differs by `1.2e-7` relative, `dw_hist` by `4.2e-6`, both
consistent with the float32 `atomicAdd` reordering noise already
measured in #7, not a regression from the larger block.

**c. Possible out-of-bounds GPU writes in
`distortion_procedures_pycuda.py`.** `init()` sizes `le1`, `le2`,
`le3`, `le4`, `activeBs`, `activeBs_index` and `size_auxiliars_*` using
the *config* `number_of_neighs` (80). But `distortion_per_pixel()`
computes its own local

```python
number_of_neighs = int(np.ceil(number_of_neighs_full*(1.-reject_fraction)))
```

from the real neighbour count, and uses *that* for `total_blocks_z`,
`total_blocks_y2` and the `base` array passed to the kernels. If a
forest's retained-neighbour count exceeds 80, the kernels index past
arrays allocated for 80 — silent corruption or a crash. With the
default `reject_fraction = 0.95` that needs a forest with >1600
neighbours, which may well be reachable on dense fields or at large
`angmax`. Worth either clamping the local value, asserting on it, or
sizing the arrays from the actual maximum (the same argument as
`max_lenght` in #2). Before the rename both were spelled
`number_of_neighs`, which is presumably how it went unnoticed.

**Promoted to its own item — see #16**, which covers deriving
`number_of_neighs` from the data instead of configuring it (the actual
fix), rather than just detecting the overrun after the fact.

## 10. Distortion matrix precision is not configurable (and is memory-bound)

`gpu_precision` (#7) covers the two-point correlation only. The
distortion matrix is always float32: of the six kernels in
`cuda_kernels.cpp`, only `pair_correlation` declares its arguments as
`myfloat` — `precompute_distances`, `compute_etas`, `order_active` and
`compute_d` hardcode `float` — and `distortion_procedures_pycuda.py`
has 21 hardcoded `np.float32` allocations. (It still has to pass
`-DMYFLOAT`, purely because it compiles the same file and
`pair_correlation`'s double `atomicAdd` would otherwise break a build
that never uses that kernel.)

This looks like a deliberate constraint rather than an oversight,
because the buffers are big. With `max_lenght` 967, `number_of_neighs`
80 and the default 50x50 binning:

| buffer | scales as | float32 | float64 |
|---|---|---|---|
| `etas12/21/13/31` (x4) | `max_lenght` | 774 MB each | 1547 MB each |
| `x12/y12/z12/r12` (x4) | **`max_lenght²`** | 299 MB each | 598 MB each |
| **total** | | **~4.9 GB** | **~9.2 GB** |

So float64 distortion needs an A100-class card, and even float32 does
not fit this workstation's 4 GB GTX 970 — the module compiles here but
`init()` dies with `cuMemAlloc failed: out of memory`.

**The delta binning is the biggest lever here, and it interacts with
the above.** The DR1 files used for testing are on the *native* DESI
0.8 Angstrom grid — measured, not assumed: 2716 points from 3600 to
5772 Angstrom, spacing exactly 0.8, i.e. 1.00x native, so no coadding
was applied upstream. If the deltas were coadded by 3 (2.4 Angstrom,
`max_lenght` ~322), the footprint drops from ~4.9 GB to ~1.26 GB — a
**3.9x** reduction rather than 3x, because the `x12/y12/z12/r12`
buffers scale quadratically. At that size float64 distortion would need
only ~2.5 GB and become practical on ordinary hardware.

Nothing in lya2pcf needs changing to *read* rebinned deltas — it just
uses whatever `LAMBDA` grid the files carry — so this is a choice made
upstream when producing them. But it is worth knowing that GPU memory
is superlinear in the delta sampling, and that testing against native
0.8 Angstrom deltas is close to the worst case.

**Done 2026-09-11**, same branch: all six kernels now take `myfloat`,
and `distortion_procedures_pycuda.py` uses `params.gpu_dtype`
throughout, so `gpu_precision` covers the distortion too. One shared
setting rather than two, on the grounds that mixing precisions between
the correlation and its distortion matrix is more likely to be a
mistake than an intention; split it later if a real case appears.

Two things that had to change with it, both silent-corruption risks
rather than obvious failures:

- The `cuda.memset_d32_async` calls zero *32-bit words* but were being
  passed an **element** count, which is only correct for 4-byte dtypes.
  At float64 they would have zeroed exactly half of each buffer,
  leaving stale values in the rest. Now `memset_d8_async` with
  `.nbytes`, which is dtype-independent.
- `index_j` is an `int*` in the kernel, but was sized and zeroed from
  `le4`, a float array. Once `le4` followed `gpu_precision` that sizing
  became wrong, so `le4` is now explicitly `np.int32`.

Verified by running the distortion end to end at float32 on the GTX 970
(with `number_of_neighs` and `rmax` reduced to fit 4 GB) — the first
time that path has run on this machine. float64 remained untested here
for the same compute-capability reason as #7.

**Resolved 2026-09-18**, same as #7: Josue confirmed the distortion
matrix also runs successfully at `gpu_precision: float64` on production
Pascal-or-newer hardware. As with the two-point side, this confirms the
run completes, not that its output has been checked against a reference.

## 11. Add a rebinning (coadding) procedure to lya2pcf

Deltas are sometimes undersampled — coadded by averaging over groups of
pixels — to cut their size. Right now lya2pcf can only consume whatever
sampling the delta files already have, so that decision is made
upstream when the deltas are produced, and re-running at a different
sampling means regenerating them.

Doing it here instead would mean averaging groups of N pixels
(weighted by `we`, presumably) in `delta_reader.py` as the forests are
built, exposed as something like `rebin: 3` in `parameters.yml`.

**Why it is worth having:** it is the single biggest lever on GPU
memory in this code, and the effect is superlinear. Measured on the DR1
set (native 0.8 Angstrom, `max_lenght` 967) at the default binning, the
distortion buffers need ~4.9 GB; coadding by 3 (`max_lenght` ~322)
brings that to ~1.26 GB — a **3.9x** reduction, because
`x12/y12/z12/r12` scale as `max_lenght²`. It also decides whether
float64 distortion is feasible at all: ~9.2 GB unrebinned versus
~2.5 GB rebinned, i.e. the difference between needing an A100 and
running on an ordinary card. It cuts the correlation cost too, since
pair counting is quadratic in forest length.

**Things to get right:** weights must be combined consistently with
`fill_dw`'s projection (the `delta_lambda` moments are computed on the
sampling being used), and rebinning interacts with `bin_size_r` — there
is no point sampling far finer than the radial bin width, which is the
physical argument for why coadding costs little accuracy.

**Depends on:** nothing structural, but it lands naturally with #4
(in-memory pipeline), since both are changes to how `delta_reader.py`
builds forests.

## 12. Detect max_lenght / neighbour overruns instead of corrupting memory

The kernels index flat buffers as `index * max_lenght + offset`, with
nothing checking that the offsets stay inside what was allocated. If
`max_lenght` does not match the data the buffers were sized from, or
the neighbour count exceeds the allocation (#9c), the kernels write
outside their allocations. On a GPU that does not reliably fault — it
silently corrupts whatever is adjacent, so the run completes and
produces wrong numbers.

**Partially addressed by #2.** The `max_lenght` half of this is no
longer really live: both `correlation_procedures_pycuda.upload_forests`
and `distortion_procedures_pycuda.init()` compute `max_lenght` from
whichever data they are about to pack, in the same function that then
sizes the buffers from it — so the buffers and the data are sized from
one measurement by construction, not merely "less likely" to disagree.
No host-side assertion is needed for that half; there is no longer a
separate `max_lenght` value that could drift from the data.

The `number_of_neighs` half is unaddressed and is where the remaining
work is — **see #16**, which covers deriving it from the data the same
way `max_lenght` already is (rather than adding an assertion around a
config guess that shouldn't exist).

**What would still help, independent of #16, cheapest first:**

- Bounds-check inside the kernels behind a debug flag, with a
  `printf` naming the index and the limit. Too costly for the inner
  loop in production, but a build-time switch makes it usable when
  something looks wrong.
- Run the test suite under `compute-sanitizer` (`cuda-memcheck`'s
  successor), which reports out-of-bounds device writes directly and
  would have caught #9c without anyone suspecting it.

**Depends on:** nothing for the two bullets above; the `number_of_neighs`
piece depends on #16.

## 13. Forests are padded to max_lenght, wasting a large share of GPU memory

Every forest is stored in a slot of exactly `max_lenght` pixels —
`gran_dc[count * max_lenght : count * max_lenght + len_forest]` — and
the kernels index as `index * max_lenght + offset`. Forests are not
close to uniform in length, so a lot of each buffer is zeros.

Measured on the DR1 set (1446 forests): lengths run from 150 to 967
with a median of 660 and mean 591, so the buffers hold 1,398,282 slots
for 854,755 real pixels — **38.9% padding**. On a 40 GB dataset that is
roughly 15.5 GB of zeros uploaded and iterated over. It is worse in the
distortion, where `x12/y12/z12/r12` are `max_lenght²` per neighbour and
so pay the padding twice.

**The fix is a standard CSR-style layout:** replace the fixed stride
with a per-forest offset array (the prefix sum of the actual lengths),
pass that to the kernels instead of `max_lenght`, and index as
`offsets[index] + i`. Buffers then hold exactly the real pixels.

Two things make this more attractive than a pure memory saving:

- It removes the hazard in #12 entirely. With explicit per-forest
  offsets and lengths there is no global `max_lenght` that can
  disagree with the data, which is the thing that currently lets the
  kernels write out of bounds.
- It should help the inner loops too, since the padded regions are
  currently still walked in some access patterns.

The cost is touching every kernel's indexing, so it wants doing when
the GPU path can actually be run and compared — i.e. on a machine where
float64 compiles, or accepting float32 for the comparison (bearing in
mind the non-determinism noted in #7, so compare within ~1e-7 rather
than for equality).

**Worth prioritising if the plan is to process tens of GB**, where a
39% saving is the difference between fitting on a given card and not.

## 14. `np.save` of the forest objects is slow and spikes RAM at scale

Reported from a real run: at the ~40 GB scale the `np.save` in
`delta_reader.py` takes around 15 minutes and spikes memory badly.

**Cause:** `data` is a dict of lists of `quasar` *objects*, which numpy
cannot store natively, so `np.save` falls back to **pickle**. That walks
the entire object graph — one Python object per forest, each with ~20
attributes — and pickle's memo table holds a reference to everything it
has seen, which is where the memory goes. Numpy arrays inside pickle
reasonably efficiently as raw buffers; Python lists of ints do not.

**The unused neighbour lists make it materially worse.**
`delta_reader.py` sets `forest.neigh_names` and `forest.neigh_pixels`
from `neighborhood_names()`, and **nothing ever reads them** — the
correlation and distortion both call `forest.neighborhood()`, which
recomputes from scratch. They are pure dead weight that gets pickled
into every file. Measured on the DR1 set: 1446 forests carrying 309,425
neighbour entries between them; dropping the lists made the save
**1.6x faster** for 6% less size. The entry count grows with forest
density, so this gets worse on a full dataset, not better.

The surrounding loop is also ~36% of extraction time and exists only to
produce the `neighbors` diagnostic file, so making it optional would
recover that too.

**Fixes, in order of value:**

1. **Do not save at all** — #4. If extraction hands the dict straight to
   the correlation in one process, the 15 minutes disappear entirely.
   This is the real answer for a single-machine workflow.
2. **Stop storing Python objects** — #13's flat layout. Concatenated
   arrays plus an offsets array save as raw buffers with no pickle
   involved: fast, and no memo table to blow up memory. #13 and this
   item want doing together; the CSR layout solves the padding, the
   bounds hazard *and* this.
3. ~~**Drop the unused `neigh_*` lists**, and make the neighbour
   statistics pass optional.~~ **Done 2026-09-11** on branch
   `feat/faster-delta-read`. The lists are no longer attached, and the
   neighbour search now runs only under a new `--statistics` flag, which
   also writes `sizes`/`neighbors` into the data directory rather than
   the working directory. On the DR1 set extraction went from 2.44 s to
   1.94 s (~20%) with files 6% smaller; the saving is larger on denser
   data, where neighbour counts per forest are higher.
4. `--split-number` already exists and lowers the peak *per `np.save`
   call*, at the cost of more files -- but **not** the peak for
   extraction as a whole. Checked while working on #15 (2026-09-18):
   `pool.map(record_from_deltas, directory)` is synchronous, so it
   doesn't return until every worker's forests are collected into one
   list in the main process -- at that point the whole dataset is
   resident in memory no matter what `--split-number` is; splitting
   only controls what happens afterwards, as `data` gets chopped up and
   saved/popped piece by piece. A real memory measurement on the DR1 set
   (`/usr/bin/time -v`, `--split-number 1` vs `4`) showed only an 11%
   difference in peak RSS (350.7 MB vs 311.7 MB) -- consistent with this,
   though the set is too small (~50 MB) to be a scale-representative
   test on its own; the code-flow argument above doesn't depend on scale
   to hold, `Pool.map()`'s semantics are what they are regardless of
   dataset size.

**Depends on:** (1) and (2) are #4 and #13, and are where the real
15-minute saving is. (3) is done and helps, but does not change the
fundamental problem: as long as the file holds pickled Python objects,
saving is slow and memory-hungry.

**(4), the actual fix, done 2026-09-18** on branch
`refactor/pixel-buffer-zones` (same branch as #15, since it grew out of
reviewing that work): `delta_reader.py` now extracts in two passes
instead of one `pool.map` over everything.

- **Pass 1**, `record_pixel_only`: reads only each file's RA/DEC (no
  delta/weight/lambda arrays, no `quasar` objects) to learn which output
  pixels it contributes to and how many forests each gets. Every file is
  therefore opened and decompressed *twice* -- once here, once again in
  pass 2 -- which is **not as cheap as it sounds**, corrected after
  Josue asked directly: measured metadata-only vs. a full read on the
  same real file and got 0.042s vs. 0.046s -- almost identical. `fitsio`
  pays nearly the same cost to open and decompress a `.fits.gz` file
  regardless of how many columns you read from it; the expensive part is
  the gzip decompression on open, not the volume subsequently read.
  Checked the FITS headers too, in case a coarser pixel/region keyword
  could avoid touching per-forest RA/DEC at all -- none present. So the
  doubled I/O is real and unavoidable with `fitsio` used this way, not
  negligible the way "reads only RA/DEC" implies on its own. What kept
  the measured *total* wall-clock overhead modest (~15-20%, see the
  timing note below) despite this is that per-file decompression is
  small next to the *other* cost pass 1 skips -- the per-forest Python
  work in pass 2 (cosmology interpolation, building `quasar` objects) --
  not that pass 1 itself is cheap. Whether that ratio holds at the real
  ~40 GB scale is a reasonable extrapolation (both terms scale with file
  size) rather than something measured here. **One input file does not
  necessarily map to one output pixel** -- checked on real data before
  assuming otherwise: one DR1 file spans 4 distinct pixels at the
  default `nside=32`. Also checked whether a pixel could be split across
  *different* input files (it would break the one-pixel-one-file
  invariant `data_index.npy`/`pixel_partition.py` depend on) -- not
  observed on the real DR1 set (every pixel came from exactly one file,
  consistent with DESI's file-level pixelization being coarser than this
  `nside`), but the design does not actually depend on that holding:
  the census is a true many-to-many map, built the same way regardless.
- **The first version of this split the pixel list into contiguous
  sorted ranges** (`np.array_split`, matching item #15's own chunking),
  which meant a file whose pixels landed on both sides of a range
  boundary got re-extracted once per chunk it fed. Measured, not
  assumed: 3 timed trials each at `--split-number` 1/4/16 on the real
  DR1 set showed old (single-pass) code flat at ~2.1s regardless of
  split-number, but this first version growing with it -- 2.2s / 2.6s
  (+24%) / 4.3s (+~100%) -- from redundant I/O scaling with how often a
  chunk boundary happened to cut through a file. Reported to Josue, who
  judged that not worth the memory fix and asked for the alternative
  this item originally dismissed.
- **Replaced with `group_files_by_shared_pixels`**: unions files
  transitively whenever they share a pixel (so a chunk boundary can
  never cut through a file), then greedily packs the resulting groups
  into `--split-number` chunks, largest (by forest count) first into
  whichever chunk is currently smallest. Every file is read exactly
  once, by construction, at any `--split-number` -- re-verified on the
  real DR1 set with the same 3-trial timings: now flat at ~2.0-2.4s
  regardless of split-number (1, 4, 16), matching the old code's
  flatness rather than its scaling growth; the remaining ~15-20% above
  the old code's ~2.1s is the one-time census pass itself, not a
  growing cost. **Trade-off knowingly accepted**: chunks are no longer
  contiguous pixel-ID ranges, so the property #15 already flagged as
  unreliable for buffer locality (RING ordering) is now not even
  attempted -- traded away on purpose for zero redundant I/O, not lost
  by accident.
- **Pass 2** extracts and saves one output chunk at a time: only the
  files that chunk actually needs are read, only that chunk's forests
  are ever resident, and `data*.npy` is written and freed
  (`del subdata`) before the next chunk starts. This is genuinely
  sequential -- chunk 2 cannot start until chunk 1 is on disk and gone.
- **`--statistics`'s neighbour count needed the same buffer-zone
  treatment `two_point.py`/`distortion.py` got in #15**, for the same
  reason: computed one chunk at a time now (to stay out of the
  "everything in memory" business this item is about), a forest's
  `neighborhood()` call would otherwise only see its own chunk, which is
  exactly the missing-neighbour bug #15 fixed for the correlation
  drivers, reintroduced here for the diagnostic pass. Fixed the same
  way, reusing `pixel_partition.find_buffer_pixels`/`load_rank_data`
  directly rather than a second implementation.

**Verified against the real DR1 set**, in two stages matching the two
versions above:

- The first (sorted-pixel-range) version was checked against a `git
  worktree` of the pre-change commit, same data, `--split-number 4`
  both ways: `data_index.npy` byte-identical, every output file's
  pixels/forest-counts/forest-identities identical.
- The `group_files_by_shared_pixels` replacement necessarily produces a
  *different* `pixel_file` mapping (chunks follow file connectivity now,
  not sorted pixel ranges), so a byte-identical comparison against the
  old code no longer applies. Verified instead against a `--split-number
  1` extraction of the same data as ground truth: reconstructed the full
  dataset from whichever chunk each pixel actually landed in and
  compared every pixel's forest set (by `forest.name`) to the ground
  truth -- exact match, and confirmed each of the DR1 set's 4 chunks
  ended up holding exactly one input file's worth of pixels (the
  degenerate, zero-sharing case already established under #15).
- Re-ran #15's own per-forest neighbour-set verification (see #15)
  against this version's output too: still an exact match against the
  1-file ground truth, 320,330 pair-links, 0 missing, 0 spurious --
  confirms neither extraction change disturbed what #15 already fixed.
- `--statistics`'s new buffer-aware neighbour count: `sum(neighbors) =
  309425` on the real DR1 set -- an exact match to the figure already on
  record above ("1446 forests carrying 309,425 neighbour entries"),
  confirming the chunk-by-chunk buffered computation reproduces the old
  whole-dataset computation exactly, not just approximately.
- **Not demonstrated empirically:** the actual memory saving. The DR1
  test set (~50 MB of delta files) is dominated by Python/numpy/fitsio
  import overhead (~300 MB) either way -- measured peak RSS was, if
  anything, marginally *higher* for the new code on this tiny set
  (315.0 MB vs 311.7 MB old), which is expected noise at a scale far
  below where the fix matters, not a regression. The claim that peak
  memory is now bounded by one chunk's data rather than the whole
  dataset rests on reading `pool.map`/`del` semantics (deterministic
  language behaviour, not scale-dependent), the same way the original
  problem was found -- not on a large-scale measurement, which nothing
  available here could produce.
- **The census pass itself was checked directly**, since "does pass 1
  load much into memory" is a fair question on its own: ran
  `record_pixel_only` via `pool.map` in isolation and compared its peak
  RSS (229 MB) against the full extraction's (350.7 MB) on the same DR1
  set. The ~121 MB difference is attributable to pass 2's actual forest
  data (flux/weight arrays, `quasar` objects), which pass 1 never
  touches -- it returns only `(unique_pix, counts)` per file, so its own
  footprint scales with the number of distinct pixels in the dataset,
  not with forest count or delta-array size.

## 15. Merge the `*_multiple_data` drivers, and add a buffer zone so no pair is missed

There are four near-duplicate drivers — `2pla.py` /
`2pla_multiple_data.py` and `distortion.py` /
`distortion_multiple_data.py`, 526 lines between them — and the
multiple-data variants are the ones that will be used going forward,
since a full dataset does not fit in one file.

They are not just copies. They parallelise on different axes:

| | `2pla.py` | `2pla_multiple_data.py` |
|---|---|---|
| MPI splits | **pixels** across ranks | **files** across ranks |
| data per rank | the whole dataset | only that rank's files |
| neighbours visible | all of them | only those in the same file |

**That last row is a correctness difference, not an implementation
detail.** `two_point_per_pixel` finds neighbours with
`forest1.neighborhood(data, angmax)`, which searches only the `data`
dict it was given. In the multiple-data driver that is one file, so any
pair whose two forests landed in different `data*.npy` files is never
counted. `delta_reader.py --split-number` splits a sorted pixel list
into contiguous chunks, so the loss happens along the boundaries between
chunks. Confirmed as known and intended for a later fix.

**Measured 2026-09-11**, DR1 set, the same 16 healpix pixels either way:

```
one file   (2pla.py style)            w_hist total  16,756,279,121
four files (multiple_data style)      w_hist total  12,721,548,931
                                      -> 24.08% of pair weight lost
```

Read that as an order of magnitude rather than a universal figure: with
only 16 pixels in 4 chunks the boundaries are a large share of the
volume, and a real dataset has far more pixels per file, so the fraction
falls. But it rises again with `--split-number`, which a 40 GB run needs
a lot of — the loss tracks the surface-to-volume ratio of the chunks.

Note also that the lost pairs are **not a random subset**: pairs
straddling a boundary are preferentially the widely separated ones, so
the effect concentrates in the large-separation bins. Whether that
biases xi = dw/w or mainly inflates its errors was not tested — a
uniform loss would largely cancel in the ratio, a separation-dependent
one need not. Worth measuring on xi directly before drawing any
conclusion about the science impact.

**The planned fix is a buffer (halo) zone**: each file also carries the
neighbouring forests just outside its own pixels, so every forest it
owns can see all of its neighbours.

**The double-counting problem this raises is already solved by the
existing ordering rule.** `forest_class.py:96` counts a pair only when

```python
mu > mumin and self.name != forest2.name and self.ra < forest2.ra
```

so an unordered pair {A, B} has exactly one valid base forest: whichever
has the smaller RA. Given that, a buffer scheme is consistent provided:

- the outer loop uses only the file's **own** forests as `forest1`,
  never buffer forests; and
- the buffer contains every neighbour of the file's own forests (it need
  not be symmetric).

Then pair {A, B} is counted exactly once, by whichever file owns the
smaller-RA forest, with the other supplied from its buffer. No
cross-rank communication or de-duplication pass is needed — which is
what makes the ordering idea the right one.

Two details for whoever implements it:

- The rule is strict `<`, so two forests with *exactly* equal RA are
  dropped rather than double counted. Harmless in practice with floats,
  but it means the rule is "drop ties", not "count once".
- `correlation_procedures_pycuda.init()` already takes a `pixel_list`
  argument whose docstring anticipates exactly this — "the gpu needs
  more data than the pixels that it is computing, it also needs the
  neigboring pixels". The hook exists; nothing passes it yet.
- `delta_reader.py` used to store `neigh_pixels` per forest, which is
  precisely the per-forest map of which healpix pixels its neighbours
  live in that a buffer builder needs. It was removed in #14 because
  nothing read it and it slowed `np.save` significantly. That is not a
  blocker: it is cheap to recompute when building the buffer, and
  computing it at that point avoids carrying it through every save.

**Worth doing together with the merge**, since a single driver that
takes a list of files and a buffer policy replaces all four scripts, and
the `2pla.py` "everything in one file" case becomes just the special
case of one file with an empty buffer.

**Done 2026-09-18** on branch `refactor/pixel-buffer-zones` (issue #20,
PR #21). `two_point_multiple_data.py` and `distortion_multiple_data.py`
are deleted; `two_point.py` and `distortion.py` now handle one file or
many identically. The "single-file" case really is just the general
case with an empty buffer, as predicted -- no special-casing needed
anywhere in the drivers.

One deviation from the plan above: the buffer is built from **pixel**
centers via `query_disc`, not by recomputing each forest's own
`neigh_pixels` as the "`delta_reader.py` used to store `neigh_pixels`
per forest..." note suggested. Coarser (one `query_disc` call per owned
pixel rather than per forest) and cheaper, and still exact given the
`max_pixrad` margin described below -- a forest-level recomputation
would only buy tighter buffers, not correctness, so wasn't worth the
extra cost here.

**How work is split, settled as: pixels across ranks, files loaded
per-rank on demand.** Considered splitting files across ranks instead
(closer to what `*_multiple_data.py` already did) — rejected because it
leaves ranks idle whenever there are fewer files than ranks, including
the single-file case, which is common (small test runs, `--split-number`
left at its default of 1). Pixels split evenly across every rank
(`np.array_split` on the sorted global pixel list, same convention
`delta_reader.py` already uses to build files) works for any files/ranks
ratio, including more ranks than files; each rank then loads only
the `data*.npy` files its own pixels and their buffer actually touch,
not the whole dataset. New shared module `pixel_partition.py` holds
this (`assign_pixels`, `find_buffer_pixels`, `load_rank_data`), used by
both drivers.

**The buffer**, per pixel, is `query_disc` centered on that pixel with
radius `angmax + healpy.max_pixrad(nside)` — the margin matters:
`query_disc` is evaluated from the pixel's *center*, but a forest can
sit anywhere within it, so the search has to reach `max_pixrad` further
than `angmax` alone to guarantee no real neighbour outside the pixel is
missed. `max_pixrad(32)` is small (~2 arcmin) next to a typical `angmax`
(~3 degrees) but there is no reason to accept an inexact margin when an
exact one is one function call away.

**A second thing this needed that didn't exist yet: a globally
consistent `angmax`.** Each of the old `*_multiple_data.py` ranks
computed its own `angmax` from the minimum comoving distance in *its own*
loaded files only — a latent inconsistency (not previously called out in
this item, found while implementing it): a rank whose local minimum
happened to be larger than the true dataset-wide minimum would compute
too small an `angmax`, searching a narrower disc than the data actually
requires and silently missing legitimate neighbours near the edge of
that narrower disc — the same failure shape as the missing-pairs bug
this item is about, from a different cause. Fixed by having
`delta_reader.py`/`delta_reader_eboss.py` persist the dataset-wide
`min_distance` they already compute (previously printed and discarded)
into a new `data_index.npy`, alongside a `pixel -> file number` map
built for free while writing `data*.npy`. Every rank derives the same
`angmax` from it. `min_distance` is saved rather than `angmax` itself,
so a driver run with a different `rtmax` later still gets the `angmax`
that setting actually implies, rather than one baked in at extraction
time.

**Verified two ways on the real DR1 set** (4 files, `--split-number 4`,
against a 1-file extraction of the same data as ground truth):

- Directly, bypassing the drivers: for every one of the 1446 forests,
  compared its neighbour set (`forest.neighborhood(data, angmax)`,
  matched by `forest.name`) between the 1-file dataset and the 4-file
  dataset loaded through `pixel_partition` for `mpi_size=4`. Exact match
  for all 1446 forests -- 320,330 total pair-links either way, 0 missing,
  0 spurious. This is the same measurement #15 originally made as an
  aggregate `w_hist` sum (24.08% lost); checking every forest's
  neighbour *set* individually is a strictly stronger test than an
  aggregate total, which can hide errors that cancel.
- End to end through the real CLI: `mpirun -np 3 lya2pcf-correlate --cpu`
  and `mpirun -np 3 lya2pcf-distort` against the 4-file split, both
  completing correctly (6+5+5 pixels split exhaustively across the 3
  ranks, sane non-empty output, no NaNs in the distortion matrix).
- Also checked pixel ownership is exhaustive and non-overlapping across
  every rank count tried (1, 2, 3, 4, 5, 8 -- including more ranks than
  the 4 files, the case that would have starved ranks under a
  files-across-ranks split).

**Known limitation, not fixed here:** `assign_pixels` splits the
globally *sorted* pixel list into contiguous chunks, same as
`delta_reader.py` does for files, on the assumption that this keeps a
rank's buffer needs local to a handful of nearby files. That assumption
depends on healpix pixel *index* proximity tracking angular proximity,
which is only true for `nest=True` ordering — `forest_class.py` calls
`healpy.ang2pix`/`query_disc` without `nest=True`, i.e. RING ordering,
where pixels in the same latitude ring have nearby indices but adjacent
rings do not. So a rank's contiguous pixel range is not guaranteed to be
a spatially compact patch of sky, and in the worst case its buffer could
span many more files than the "just the neighbouring one or two" case
this design is optimized for. Not a correctness problem (verified
above), only a possible efficiency one at real scale (the ~40 GB
production case) that switching to NESTED ordering would fix -- a
bigger, separate change (touches every `pix = healpy.ang2pix(...)` call
site and anything that assumes RING) that is not part of this item.

## 16. `number_of_neighs` should be derived from the data, not a config guess

Tracked as GitHub issue #1 ("number_of_neighs causes an error"), open
since before this list existed. `number_of_neighs` (default 80 in
`parameters.yml`) is a fixed guess at the largest neighbour count any
forest will have, used only to size the distortion buffers in
`distortion_procedures_pycuda.init()` (`le1`-`le4`, `activeBs`,
`activeBs_index`, `x12/y12/z12/r12`, etc.). Nothing checks that the guess
holds. `distortion_per_pixel()` computes the *real* per-forest count from
`forest1.neighborhood(data, angmax)` and uses that directly against
buffers sized from the config value — already flagged as a silent
out-of-bounds write in #9c, with mitigation options (assert, debug-mode
bounds checking, `compute-sanitizer`) listed in #12.

This item is the fix `#9c`/`#12` point at but don't commit to: **don't
make `number_of_neighs` a better-documented config guess, stop it being
config at all.** Compute the actual maximum neighbour count from the
loaded data at the start of the run and size the buffers from that,
exactly the fix already applied to `max_lenght` in #2 (a static
`parameters.py` value that turned out to depend on whichever data was
actually loaded, and is now computed from `data` instead of configured).
Same argument here: no dataset-dependent quantity should be a fixed
number in `parameters.yml` when the loaded data can simply be measured
before the buffers it sizes are allocated.

**The real cost is not the fix, it's the extra pass it requires.**
Finding the true maximum means running `forest.neighborhood()` — the full
`query_disc` + pairwise `dot_product` search — over every forest before
`init()` allocates anything, which is exactly the search `#14` made
optional behind `--statistics` because it is a large fraction of
extraction runtime (see `#14`'s measurement: ~36% of extraction time on
the DR1 set, and it grows with forest density). Options, roughly in order
of how much they avoid repeating that cost:

- **Reuse `--statistics` output if it's already on disk.** `#14`'s
  `sizes`/`neighbors` diagnostic files in `data_dir` already contain the
  neighbour count per forest (unweighted by `reject_fraction`, but
  `ceil(count * (1 - reject_fraction))` recovers the sizing bound). Read
  it if present, fall back to computing fresh if not.
- **Compute it once in `distortion.py`/`distortion_multiple_data.py`
  before calling `init()`**, over exactly the `data` that run will use
  (which may be a subset of pixels for the multi-file drivers), rather
  than requiring a prior `--statistics` extraction. Pays the neighbour
  search cost once per distortion run instead of guessing it up front,
  which is strictly better than today's silent-corruption risk, but is
  the same cost `#14` opted out of paying by default during extraction.
- **A generous, checked upper bound instead of the exact maximum** — e.g.
  size from `angmax` and typical forest density rather than a full
  per-forest scan, with the `#12` host-side assertion catching the rare
  case it's still wrong. Cheaper, but reintroduces a guess, just a better
  one; the assertion is what makes that acceptable instead of silent.

Whichever approach, this should replace `number_of_neighs` as a required
`parameters.yml` key with, at most, an optional override (a ceiling to
guard against a pathological outlier blowing up memory) — the same
relationship `parameters.yml` now has with `max_lenght`, which is no
longer a key there at all.

**Depends on:** conceptually independent, but shares the "measure the
real neighbour count before allocating" cost with #12 and #15's buffer
zone (which needs per-forest neighbour-to-pixel info too — see #15's
notes on `neigh_pixels`), so worth doing alongside either rather than as
a third separate pass over the same neighbour search.

## 17. Multi-GPU runs split "how many GPUs" across two unrelated places

To run on several GPUs today you set two things that both amount to
"how many GPUs am I using," in two different places, with nothing
checking they agree:

- `mpirun -np N` on the command line — the total number of MPI
  processes/ranks for the whole job.
- `number_of_cuda_devices` in `parameters.yml` — described in the README
  as "the number of devices per node," used only to turn an MPI rank into
  a local CUDA device index:

  ```python
  cuda_device = str(int(mpi_rank % params.number_of_cuda_devices + params.cuda_device_first_number))
  os.environ['CUDA_DEVICE'] = cuda_device
  ```

  duplicated identically in `two_point.py`, `two_point_multiple_data.py`,
  `distortion.py` and `distortion_multiple_data.py` (the same
  four-way duplication #15 is about merging).

On one node these should be the same number, but nothing enforces that,
and nothing validates either one against the GPUs actually present. The
two use cases this actually has to cover: local single-GPU tests on this
workstation (`-np 1`, one device), and production runs across several
nodes with 4 GPUs each (`number_of_cuda_devices: 4`, `-np` a multiple of
4) — the default in `parameters.yml` already reflects that production
topology, not an arbitrary number. It is precisely the switch between
these two cases (or a typo in `-np` on the production cluster) that has
no safeguard today. Get
them out of sync and the failure is not a clear error naming the
mismatch — it is whatever `CUDA_DEVICE=<out-of-range index>` does to
`pycuda.autoinit`, which is either an opaque device-ordinal error or,
worse, two ranks silently landing on the *same* device (e.g. `-np 2`
against the default `number_of_cuda_devices: 4` on this single-GPU GTX
970 workstation: rank 0 gets device 0, which exists; the same command
with a device that happens to exist but is shared silently corrupts
nothing per se but duplicates work on one GPU while the run *looks*
like it used two). That silent-wrong-answer shape is the same category
of bug as #9c and #16, just for device assignment instead of buffer
sizing.

**The fix direction is the same as #2 and #16: stop asking the user to
state a machine fact that the machine can report itself.** The number of
CUDA devices on a node is `pycuda.driver.Device.count()` (or
`nvidia-smi -L`), not something that belongs in a project config file at
all — `number_of_cuda_devices` can be queried at startup instead of
configured, the same argument already applied to `max_lenght` (#2) and
proposed for `number_of_neighs` (#16). That removes one of the two
places, but doesn't by itself resolve the real multi-node case: MPI does
not tell a rank "how many ranks share my node" without extra work
(`MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)` gives a per-node
communicator whose local rank/size is the right thing to modulo against,
*instead of* the global `mpi_rank`/a configured per-node count). That
Split_type call is the actual fix for "assign ranks to local GPUs
correctly on multi-node jobs" — `number_of_cuda_devices` as a config
value is a manual, unchecked stand-in for information MPI can provide
directly.

**Short of that rewrite, a cheap improvement:** validate at startup, in
each driver, that `mpi_size` is consistent with the queried device count
(e.g. `mpi_size <= device_count` for a single-node run, or divides evenly
by it for multi-node with one rank per GPU) and fail with a message
naming both numbers, instead of leaving a silent mismatch to surface as
either a CUDA error several layers down or, worse, no error at all.

**Depends on:** nothing structural; the validation-only version is a
small, independent, low-risk change and could be done first.

**#15's driver merge (done) halved this duplication, not eliminated
it.** Four copies of the device-assignment line became two
(`two_point.py`, `distortion.py`) — the merge only combined each
driver with its own `*_multiple_data` twin, not the two-point and
distortion drivers with each other, so the `Split_type`-based fix still
has two call sites to replace, not one.

## 18. mpi4py is a hard dependency even for a single process

Moved here from the README's old "Things to do" list ("do not call for
mpi4py when only used with 1 cpu"). Every driver (`two_point.py`,
`two_point_multiple_data.py`, `distortion.py`,
`distortion_multiple_data.py`) unconditionally does `from mpi4py import
MPI` and builds `MPI.COMM_WORLD`, even for a single local process with
no `mpirun` in sight — that works fine (`COMM_WORLD` is a valid
size-1 communicator on its own, confirmed while checking mpi4py's pip
packaging for the install-instructions rewrite), so this is not a
correctness bug. It is dependency weight: `mpi4py` needs a real MPI
implementation resolvable on the machine to import at all (see the
README's install section, and the `--no-binary mpi4py` note for making
sure it links the right one), which is a real thing to have installed
just to run one process on one machine with no MPI job in sight — e.g.
a laptop with no MPI implementation could not `pip install mpi4py`
usably even for a `--gpu` single-device run that has nothing to
coordinate.

**Fix direction:** guard the `mpi4py` import and `COMM_WORLD` setup
behind an actual multi-process check — e.g. only import it when an MPI
rank/size environment variable set by the launcher (`OMPI_COMM_WORLD_SIZE`,
`PMI_SIZE`, etc.) is present, and fall back to a trivial rank-0-of-1
stand-in object otherwise that the rest of each driver already treats
`comm/mpi_rank/mpi_size` as (so the drivers themselves would not need to
branch — only the setup at the top of each would).

**#15's driver merge (done) halved this duplication too** -- the same
setup block now exists in two drivers (`two_point.py`, `distortion.py`)
instead of four, for the same reason as #17's note above.

**Depends on:** nothing structural; independent of #17, though both are
about the same "how many processes/GPUs" question from different
angles — #17 is about the count itself being split across two places,
this is about the cost of requiring MPI at all for the trivial case of
one process.

---

## Suggested order

1. ~~**#5 trailing-slash fix**~~ — **done 2026-09-10**, branch
   `fix/path-joining`, issue #2, PR #3, merged.
2. ~~**#2 `parameters.yml`**~~ — **done 2026-09-10**, branch
   `feat/parameters-yaml` (stacked on `fix/path-joining`), issue #4,
   PR #5, merged. Also removed the `max_lenght` self-rewrite.
3. ~~**#1 `src/lya2pcf/` package layout**~~ — **done 2026-09-17**, branch
   `feat/src-layout`, issue #14, PR #15, merged 2026-09-18. (**#8** was
   done ahead of it, so imports were already explicit — one less thing
   for the move to untangle.)
4. ~~**#3 configurable metadata keys**~~ — **done 2026-09-11**, branch
   `feat/metadata-keys`. `delta_reader_eboss.py` still has its own
   hardcoded names; see #3 for why it was left.
5. ~~**#7 float32/float64 precision**~~ — **done 2026-09-11**, branch
   `feat/gpu-precision` (stacked on `feat/parameters-yaml`). Being able
   to compile the kernel locally immediately caught a GPU-breaking
   regression in #2 (`max_lenght` passed as a plain Python int), which
   is the argument for having done it early.
6. **#4 in-memory pipeline** — builds on the library-style API that #1
   gives you.
7. **#6 shared host memory across GPUs** — biggest architectural change
   and the one most likely to introduce subtle bugs (shared-memory
   lifetime/cleanup, race conditions between ranks). Do it last, on a
   stable base, and test carefully on a real multi-GPU node before
   trusting results from it.
