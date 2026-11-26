# Overview of all stages and all flows on the whole node (64 threads on peak).
#   sbatch -p peak src/scripts/server/campaign.sbatch src/scripts/server/campaigns/overview.sh
CAMPAIGN_TAG=overview
CAMPAIGN_NAME=all
STAGES="naive interchange regblock packed"
SIZES=("1000 1000 1000" "1024 1024 1024")
THREADS="node"
REPS=1
GEMM_ARGS=(--warmup 2 --reps 10)
