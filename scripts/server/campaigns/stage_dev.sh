# Development campaign for one stage: all flows, N = 1000 and 1024, 1 thread and the whole node.
#   sbatch -p peak --export=ALL,STAGE=naive src/scripts/server/campaign.sbatch src/scripts/server/campaigns/stage_dev.sh
STAGE=${STAGE:-naive}
case $STAGE in
  naive) CAMPAIGN_TAG=S1 ;;
  interchange) CAMPAIGN_TAG=S2 ;;
  regblock) CAMPAIGN_TAG=S3 ;;
  packed) CAMPAIGN_TAG=S4 ;;
  *) echo "stage_dev.sh: unknown STAGE '$STAGE'" >&2; exit 1 ;;
esac
CAMPAIGN_NAME=${NAME:-dev}
STAGES=$STAGE
SIZES=("1000 1000 1000" "1024 1024 1024")
THREADS="1 node"
REPS=1
GEMM_ARGS=(--warmup 2 --reps 10)
