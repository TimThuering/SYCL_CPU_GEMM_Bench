# Result JSON (schema version 1)

Every run writes exactly one JSON document (to `--output FILE` or stdout), also when it fails after
the command line was parsed. Command line errors (exit code 2) produce no JSON.
Parts that were not reached are `null`. Non-finite numbers (inf, NaN) are written as `null`.

## Status and exit code

| `status` | Exit code | Meaning |
|---|---|---|
| `OK` | 0 | ran; verified, or verification disabled (`--no-verify`) |
| `VERIFY_FAIL` | 1 | ran; result does not match the host reference |
| `UNSUPPORTED` | 4 | the selected stage is not available in this build |
| `ERROR` | 3 | any other failure (allocation, SYCL exception, ...); message in `error` |

(Exit code 2: invalid command line, no JSON. Exit code 3 is also returned if the JSON cannot be written.)

## Fields

| Field | Content |
|---|---|
| `schema_version` | `1` |
| `status`, `error` | see above; `error` is `null` unless `UNSUPPORTED` / `ERROR` |
| `label` | `--label` text (empty if not given) |
| `timestamp` | local start time, ISO 8601 with UTC offset |
| `command_line` | full command line |
| `config` | `stage`, `m`, `n`, `k`, `lda`, `ldb`, `ldc`, `ld_pad`, `seed`, `warmup`, `reps`, `verify` |
| `build` | `sycl_impl`, `sycl_targets`, `sycl_extra_flags`, `arch_flags`, `effective_flags`, `cxx_flags`, `build_type`, `compiler` {`path`, `id`, `version`, `version_output`}, `cmake_version`, `git` {`commit`, `describe`, `dirty`}, `build_host` {`hostname`, `cpu`, `logical_cores`, `os`, `arch`}, `params` {`MR`, `NR`, `KC`, `MC`} |
| `host` | machine running the binary: `hostname`, `cpu_model`, `os`, `online_cpus`, `affinity` {`count`, `list`} |
| `env` | relevant environment variables (`OMP_*`, `GOMP_*`, `KMP_*`, `ACPP_*`, `HIPSYCL_*`, `POCL_*`, `SYCL_*`, `ONEAPI_*`, `UR_*`, `OCL_ICD*`, `TBB_*`, `LD_LIBRARY_PATH`, `LD_PRELOAD`) |
| `device` | `name`, `vendor`, `driver_version`, `device_version`, `platform_name`, `platform_version`, `max_compute_units` |
| `arena` | `alloc_kind`, `alignment`, `total_bytes`, `regions` {`A`, `B`, `C`, `workspace`: {`offset`, `bytes`}} |
| `timing` | `clock`, `flops` (2·M·N·K), `cold_s`, `warmup_s` [], `times_s` [] (timed iterations), `mean_s`, `min_s`, `max_s`, `gflops_mean`, `gflops_best` (from `min_s`) |
| `os_threads` | OS threads of the process after the timed iterations |
| `verify` | `performed`, `passed` (`null` if not performed), `entries_checked`, `max_abs_error`, `tolerance`, `failure` (first failure or `null`) |

All times are in seconds.

## Example (shortened)

```json
{
  "schema_version": 1,
  "status": "OK",
  "error": null,
  "label": "peak_s1",
  "timestamp": "2026-11-10T09:38:28+0100",
  "command_line": "./gemm_bench -s naive --size 1000 --label peak_s1 -o r.json",
  "config": { "stage": "naive", "m": 1000, "n": 1000, "k": 1000, "lda": 1000, "ldb": 1000, "ldc": 1000,
              "ld_pad": 0, "seed": 123, "warmup": 2, "reps": 10, "verify": true },
  "build": { "sycl_impl": "acpp", "sycl_targets": "omp.accelerated", "...": "..." },
  "host": { "hostname": "...", "cpu_model": "...", "os": "...", "online_cpus": 64,
            "affinity": { "count": 64, "list": "0-63" } },
  "env": { "ACPP_VISIBILITY_MASK": "omp", "OMP_NUM_THREADS": "64", "OMP_PLACES": "cores", "OMP_PROC_BIND": "close" },
  "device": { "name": "...", "platform_name": "...", "max_compute_units": 64, "...": "..." },
  "arena": { "alloc_kind": "sycl::aligned_alloc_device", "alignment": 4096, "total_bytes": 24010752,
             "regions": { "A": { "offset": 0, "bytes": 8000000 }, "...": "..." } },
  "timing": { "clock": "std::chrono::steady_clock", "flops": 2e9, "cold_s": 0.41,
              "warmup_s": [0.021, 0.020], "times_s": [0.020, 0.020, "..."],
              "mean_s": 0.0201, "min_s": 0.0199, "max_s": 0.0205, "gflops_mean": 99.5, "gflops_best": 100.5 },
  "os_threads": 65,
  "verify": { "performed": true, "passed": true, "entries_checked": 1000000, "max_abs_error": 1.1e-13,
              "tolerance": 1e-08, "failure": null }
}
```
