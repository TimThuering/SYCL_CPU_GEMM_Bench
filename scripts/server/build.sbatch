#!/bin/bash -l
#SBATCH --job-name=gemm-build
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# Builds and tests flows on a compute node (wrapper around build.sh).
# Submit from the benchmark directory (= GB_ROOT, so logs/ and src/ resolve):
#   cd ~/SYCL_RISC_V_Project/GEMM_Benchmark
#   sbatch -p peak src/scripts/server/build.sbatch [flow ...]
#   sbatch -p peak --export=ALL,GB_CLEAN=1 src/scripts/server/build.sbatch
[ -f "$SLURM_SUBMIT_DIR/src/scripts/server/build.sh" ] ||
  { echo "submit from the benchmark directory (no src/scripts/server/build.sh in $SLURM_SUBMIT_DIR)"; exit 1; }
exec bash "$SLURM_SUBMIT_DIR/src/scripts/server/build.sh" "$@"
