# SYCL_CPU_GEMM_Bench

GEMM benchmark (C = A·B, double, row-major) for comparing SYCL implementations on CPUs
(AdaptiveCpp, DPC++, PoCL) with a sequence of optimization stages, plus a plain C++/OpenMP twin.
One run = one stage, one problem size, one result JSON.

| Stage | Name | Kernel |
|---|---|---|
| – | `ref` | single_task triple loop (harness testing only) |
| S1 | `naive` | one C element per work-item |
| S2 | `interchange` | one C row per work-item, k outer / j inner |
| S3 | `regblock` | MR × NR register block per work-item |
| S4 | `packed` | packed panels + KC/MC cache blocking around the S3 micro-kernel |

## Repository layout

```
CMakeLists.txt, cmake/     build (SYCL_IMPL / SYCL_TARGETS / ARCH_FLAGS, dependencies, build info)
include/bench/, src/       harness: CLI, data, verification, timing, metadata, JSON, drivers
src/backend_sycl|omp/      arena, device query, stage table per backend
kernels/                   GemmArgs, parameters, the stages (sycl_stages/, omp_stages/)
tests/                     GoogleTest tests (-DBENCH_BUILD_TESTS=ON)
scripts/local/             run on the local PC: sync to the server, pull results
scripts/server/            run on the server: machine profiles, flows, build, run, campaigns
docs/                      building.md, json_schema.md
```

## Running the binary

```
gemm_bench -s <stage> [--size N | -m M -n N -k K] [--ld-pad P] [--seed S]
           [--warmup W] [--reps R] [--no-verify] [-o result.json] [--label TEXT] [-q]
gemm_bench --help | --version | --list-stages | --list-devices
```

Sequence: allocation (one page-aligned arena), parallel initialization, 1 cold + W warmup + R timed
iterations (`std::chrono::steady_clock`, submit + wait), verification against a host reference,
JSON to `-o` or stdout. Exit codes: 0 OK, 1 VERIFY_FAIL, 2 usage error, 3 ERROR, 4 UNSUPPORTED.
Fields: `docs/json_schema.md`. Building single flows by hand and the tests: `docs/building.md`.

## Workflow (local PC ↔ server)

The repository on the local PC is the only source. The server gets a plain copy of the sources (no `.git`)
through the sshfs mount of the server home; results come back the same way.

**Local configuration (once):** `cp scripts/local/targets.example.sh scripts/local/targets.local.sh` and set
the mount point, the benchmark directory on the server and the local analysis directory
(`targets.local.sh` is git-ignored).

**Server layout** (`~/SYCL_RISC_V_Project/GEMM_Benchmark` on Monte Cimone):

```
src/       synced sources (+ SOURCE_VERSION, SOURCE_PATCH)
build/<machine>/<flow>/   (json, cxxopts, googletest fetched by CMake per build, compute node)
results/<date>_<tag>_<name>/<host>/<flow>/<stage>_m<M>_n<N>_k<K>_t<threads>_r<rep>.json
logs/
```

### 1. Sync the sources (local)

```bash
scripts/local/sync_to_server.sh monte_cimone            # dry run: what would change
scripts/local/sync_to_server.sh monte_cimone --apply    # transfer
scripts/local/sync_to_server.sh monte_cimone --apply --final   # refuses uncommitted changes
```

Only a whitelist is transferred (`CMakeLists.txt README.md cmake docs include kernels scripts src tests`),
never dotfiles (`.git`, `.idea`) or `targets.local.sh`; the transfer aborts if a whitelisted file contains
something that looks like a credential. The version goes along as `src/SOURCE_VERSION` (commit, describe,
dirty), the uncommitted changes as `src/SOURCE_PATCH`; the build info (and every result JSON) reads it.

### 2. Build and test all flows (server, compute node)

```bash
cd ~/SYCL_RISC_V_Project/GEMM_Benchmark
sbatch -p peak src/scripts/server/build.sbatch                  # all flows of the machine
sbatch -p peak src/scripts/server/build.sbatch acpp_omp_accel   # single flows
sbatch -p peak --export=ALL,GB_CLEAN=1 src/scripts/server/build.sbatch
```

Per flow: module load, CMake, build, `ctest`; logs in `logs/build_<machine>_<flow>.log`, summary at the end
of the job output (`logs/gemm-build_<job>.out`).

### 3. Run a campaign (server, compute node)

```bash
sbatch -p peak src/scripts/server/campaign.sbatch src/scripts/server/campaigns/smoke.sh
sbatch -p peak --export=ALL,STAGE=naive src/scripts/server/campaign.sbatch src/scripts/server/campaigns/stage_dev.sh
```

A campaign definition (`scripts/server/campaigns/*.sh`) lists tag, name, stages, sizes, thread counts,
flows, repetitions and further `gemm_bench` options; see the header of `campaign.sh`. Single runs:
`bash src/scripts/server/run_one.sh <flow> <threads> <out.json> -s naive --size 1000`.

### 4. Pull the results (local)

```bash
scripts/local/pull_results.sh monte_cimone            # dry run
scripts/local/pull_results.sh monte_cimone --apply    # all campaigns, or name single campaigns
```

Sorted by the campaign tag into the analysis directory: `data/dev/S1_naive|S2_interchange|S3_regblock|S4_packed/`,
`data/dev/<tag>/` (e.g. smoke) and `data/final/`. Nothing is ever deleted.

## Flows and machines

| Flow | Module | Build | Run time |
|---|---|---|---|
| `acpp_omp_accel` | `acpp/88ae0ef_omp` | acpp, `omp.accelerated` | OpenMP, `OMP_PROC_BIND=close`, `OMP_PLACES=cores` |
| `acpp_omp_libonly` | `acpp/88ae0ef_omp` | acpp, `omp.library-only` | same |
| `acpp_generic` | `acpp/88ae0ef_omp` | acpp, `generic` (JIT) | same |
| `acpp_pocl` | `acpp/88ae0ef_omp_pocl` | acpp (OpenCL backend), `generic` | PoCL pthread, `POCL_AFFINITY=1` |
| `dpcpp_native` | `dpcpp/7.1.0_nativecpu` | DPC++ `native_cpu` (Vecz) | Native CPU pool (no pinning option) |
| `dpcpp_native_novecz` | `dpcpp/7.1.0_nativecpu` | DPC++ `native_cpu`, Vecz off | same |
| `dpcpp_pocl` | `dpcpp/7.1.0_nativecpu_pocl` | DPC++ `spir64` (JIT by PoCL) | PoCL pthread, `POCL_AFFINITY=1` |
| `omp_twin` | `llvm/21.1.8` | clang++, `SYCL_IMPL=none` | OpenMP, as above |

Every run is confined to the first t cores with `taskset`; JIT and kernel caches go to a fresh
temporary directory per run (the cold iteration always includes the kernel compilation).
Definitions: `scripts/server/flows.sh`. Machines (`scripts/server/machines/`): `peak` (SG2044) and `blade`
(SG2042), detected from the host name; a new machine = a new profile (partition, module tree, thread counts,
ISA flags, flow list).
