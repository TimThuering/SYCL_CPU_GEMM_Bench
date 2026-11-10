# Building

One build directory per flow. The executable is always `gemm_bench`; which implementation
and backend it uses is recorded in its build info (and later in every result JSON).

Requirements: CMake >= 3.20, a C++17 compiler, internet access at configure time
(nlohmann/json v3.12.0 and cxxopts v3.3.1 are fetched via `FetchContent`, shallow git clones).

## CMake options

| Option | Values | Meaning |
|---|---|---|
| `CMAKE_CXX_COMPILER` | `acpp`, DPC++ `clang++`, `clang++` | compiler of the flow |
| `SYCL_IMPL` | `acpp` \| `dpcpp` \| `none` | `none` = plain C++/OpenMP twin |
| `SYCL_TARGETS` | e.g. `omp.accelerated`, `generic`, `native_cpu`, `spir64` | `--acpp-targets` / `-fsycl-targets` |
| `SYCL_EXTRA_FLAGS` | string | extra SYCL flags (compile + link), e.g. Native CPU device flags |
| `ARCH_FLAGS` | string | host ISA flags (compile + link) |
| `GEMM_MR/NR/KC/MC` | integers | kernel parameters (`kernels/params.hpp`) |
| `BENCH_BUILD_TESTS` | `OFF` (default) \| `ON` | build the GoogleTest tests (googletest v1.17.0 fetched) |
| `CMAKE_BUILD_TYPE` | default `Release` | |

## Flows (templates)

`$ARCH` stands for the host ISA flag of the machine (e.g. `-march=...` or `-mcpu=native`).
Use build directories outside the source tree (or add them to your `.gitignore`).

```bash
SRC=~/SYCL_RISC_V_Project/Benchmark_Suite/SYCL_CPU_GEMM_Bench
B=~/gemm_builds        # build root

# AdaptiveCpp, omp.accelerated / omp.library-only / generic (generic JIT also serves ACpp -> PoCL)
cmake -S $SRC -B $B/acpp_omp_accel -DCMAKE_CXX_COMPILER=acpp -DSYCL_IMPL=acpp \
      -DSYCL_TARGETS=omp.accelerated -DARCH_FLAGS="$ARCH"
cmake -S $SRC -B $B/acpp_omp_libonly -DCMAKE_CXX_COMPILER=acpp -DSYCL_IMPL=acpp \
      -DSYCL_TARGETS=omp.library-only -DARCH_FLAGS="$ARCH"
cmake -S $SRC -B $B/acpp_generic -DCMAKE_CXX_COMPILER=acpp -DSYCL_IMPL=acpp \
      -DSYCL_TARGETS=generic

# DPC++ Native CPU (Vecz) / DPC++ -> PoCL
cmake -S $SRC -B $B/dpcpp_native -DCMAKE_CXX_COMPILER=$DPCPP/bin/clang++ -DSYCL_IMPL=dpcpp \
      -DSYCL_TARGETS=native_cpu -DARCH_FLAGS="$ARCH" \
      -DSYCL_EXTRA_FLAGS="-Xsycl-target-frontend=native_cpu $ARCH"
cmake -S $SRC -B $B/dpcpp_spir64 -DCMAKE_CXX_COMPILER=$DPCPP/bin/clang++ -DSYCL_IMPL=dpcpp \
      -DSYCL_TARGETS=spir64 -DARCH_FLAGS="$ARCH"

# Plain C++/OpenMP twin
cmake -S $SRC -B $B/omp_twin -DCMAKE_CXX_COMPILER=clang++ -DSYCL_IMPL=none -DARCH_FLAGS="$ARCH"

cmake --build $B/<flow> -j
$B/<flow>/gemm_bench
```

Backend selection at run time (environment, as before): `ACPP_VISIBILITY_MASK`, `ONEAPI_DEVICE_SELECTOR`,
`OCL_ICD_VENDORS`/`OCL_ICD_FILENAMES`, `POCL_DEVICES`, ...

## Tests

Configure a flow with `-DBENCH_BUILD_TESTS=ON`, build, then run `ctest` in the build directory
(with the same run-time environment as the benchmark, e.g. `ACPP_VISIBILITY_MASK`, `ONEAPI_DEVICE_SELECTOR`).

| Test | Content |
|---|---|
| `bench_unit_tests` | harness core: CLI, value function, layout, init check, verification, timing, JSON |
| `bench_stage_tests` | every stage of this flow × 16 problem shapes (odd, rectangular, 1×1×1, with/without row padding), each run twice (catches accumulating onto C), verified; stages that are not implemented are reported as *skipped* |
| `cli_ref`, `cli_usage_exit_code` | end-to-end runs of `gemm_bench` (exit codes 0 and 2) |

Single tests: `ctest -R regblock`, or `./tests/bench_stage_tests --gtest_filter='*regblock*'` for the
GoogleTest output with the failure message.

## Build info

- Configure time → `<build>/generated/build_config.hpp`: impl, targets, flags, compiler (`--version`), params.
- Every build → `<build>/generated/git_info.hpp`: commit, `describe`, dirty flag.
  Git is only read (`--no-optional-locks`), nothing is written to the repository.
