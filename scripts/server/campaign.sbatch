#!/bin/bash -l
#SBATCH --job-name=gemm-campaign
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
# Runs a campaign on an exclusive compute node (wrapper around campaign.sh).
# Submit from the benchmark directory (= GB_ROOT, so logs/ and src/ resolve):
#   cd ~/SYCL_RISC_V_Project/GEMM_Benchmark
#   sbatch -p peak src/scripts/server/campaign.sbatch src/scripts/server/campaigns/smoke.sh
[ -f "$SLURM_SUBMIT_DIR/src/scripts/server/campaign.sh" ] ||
  { echo "submit from the benchmark directory (no src/scripts/server/campaign.sh in $SLURM_SUBMIT_DIR)"; exit 1; }
cd "$SLURM_SUBMIT_DIR" || exit 1
exec bash src/scripts/server/campaign.sh "$@"
