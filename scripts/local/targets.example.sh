# Targets of sync_to_server.sh / pull_results.sh.
# Copy to targets.local.sh (git-ignored) and adapt the paths.
#
#   TARGET_MOUNT[<target>]  local mount point of the server home (sshfs)
#   TARGET_DIR[<target>]    benchmark directory on the server, relative to the mount point
#   ANALYSIS_DIR            local analysis directory (results are sorted into $ANALYSIS_DIR/data)

declare -A TARGET_MOUNT=(
  [monte_cimone]=/path/to/mount/of/monte_cimone_home
)
declare -A TARGET_DIR=(
  [monte_cimone]=SYCL_RISC_V_Project/GEMM_Benchmark
)
ANALYSIS_DIR=/path/to/GEMM_Benchmark_Analysis
