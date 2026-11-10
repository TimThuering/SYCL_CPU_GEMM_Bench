# Smoke test: every flow and stage once on small sizes, 1 thread and the whole node.
# Stages that are not implemented yet end with exit code 4 (UNSUPPORTED).
CAMPAIGN_TAG=smoke
CAMPAIGN_NAME=all
STAGES="ref naive interchange regblock packed"
SIZES=("64 64 64" "33 17 65")
THREADS="1 node"
REPS=1
GEMM_ARGS=(--warmup 1 --reps 3)
