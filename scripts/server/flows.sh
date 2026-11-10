# Flow definitions (sourced by env.sh): which modules, how to configure, how to run.
#
#   flow             module(s)                     build                             run time
#   acpp_omp_accel   acpp/88ae0ef_omp              acpp, omp.accelerated             OpenMP threads, pinned
#   acpp_omp_libonly acpp/88ae0ef_omp              acpp, omp.library-only            OpenMP threads, pinned
#   acpp_generic     acpp/88ae0ef_omp              acpp, generic (JIT)               OpenMP threads, pinned
#   acpp_pocl        acpp/88ae0ef_omp_pocl         acpp (OpenCL backend), generic    PoCL pthread, POCL_AFFINITY
#   dpcpp_native     dpcpp/7.1.0_nativecpu         DPC++ native_cpu (Vecz)           Native CPU pool, not pinned
#   dpcpp_native_novecz  dpcpp/7.1.0_nativecpu     DPC++ native_cpu, Vecz off        Native CPU pool, not pinned
#   dpcpp_pocl       dpcpp/7.1.0_nativecpu_pocl    DPC++ spir64 (JIT by PoCL)        PoCL pthread, POCL_AFFINITY
#   omp_twin         llvm/21.1.8                   clang++, SYCL_IMPL=none           OpenMP threads, pinned
#
# Every run is confined with taskset to the first t cores (same core set for all flows);
# pinning inside that set only where the runtime supports it natively.

GB_ALL_FLOWS="acpp_omp_accel acpp_omp_libonly acpp_generic acpp_pocl dpcpp_native dpcpp_native_novecz dpcpp_pocl omp_twin"

flow_check() {  # <flow>
  case " $GB_ALL_FLOWS " in *" $1 "*) ;; *) die "unknown flow '$1' (known: $GB_ALL_FLOWS)" ;; esac
}

flow_modules() {  # <flow> -> module names
  case $1 in
    acpp_omp_accel|acpp_omp_libonly|acpp_generic) echo acpp/88ae0ef_omp ;;
    acpp_pocl) echo acpp/88ae0ef_omp_pocl ;;
    dpcpp_native|dpcpp_native_novecz) echo dpcpp/7.1.0_nativecpu ;;
    dpcpp_pocl) echo dpcpp/7.1.0_nativecpu_pocl ;;
    omp_twin) echo llvm/21.1.8 ;;
  esac
}

# Clean module environment with only the flow's modules
flow_load() {  # <flow>
  module purge
  module use "$GB_MODULE_TREE"
  local m
  for m in $(flow_modules "$1"); do module load "$m" || die "module load $m failed"; done
}

# CMake options of the flow -> array FLOW_CMAKE_ARGS (after flow_load)
flow_cmake_args() {  # <flow>
  local arch; arch=$(gb_arch "$1")
  case $1 in
    acpp_omp_accel)   FLOW_CMAKE_ARGS=(-DCMAKE_CXX_COMPILER=acpp -DSYCL_IMPL=acpp -DSYCL_TARGETS=omp.accelerated "-DARCH_FLAGS=$arch") ;;
    acpp_omp_libonly) FLOW_CMAKE_ARGS=(-DCMAKE_CXX_COMPILER=acpp -DSYCL_IMPL=acpp -DSYCL_TARGETS=omp.library-only "-DARCH_FLAGS=$arch") ;;
    # generic: the kernel ISA is chosen by the JIT at run time; host code gets the native ISA
    acpp_generic|acpp_pocl)
                      FLOW_CMAKE_ARGS=(-DCMAKE_CXX_COMPILER=acpp -DSYCL_IMPL=acpp -DSYCL_TARGETS=generic "-DARCH_FLAGS=$arch") ;;
    # Native CPU: -march only reaches host code, the device part needs -Xsycl-target-frontend=native_cpu
    dpcpp_native)     FLOW_CMAKE_ARGS=("-DCMAKE_CXX_COMPILER=$DPCPP_ROOT/bin/clang++" -DSYCL_IMPL=dpcpp -DSYCL_TARGETS=native_cpu
                                       "-DARCH_FLAGS=$arch" "-DSYCL_EXTRA_FLAGS=-Xsycl-target-frontend=native_cpu $arch") ;;
    dpcpp_native_novecz)
                      FLOW_CMAKE_ARGS=("-DCMAKE_CXX_COMPILER=$DPCPP_ROOT/bin/clang++" -DSYCL_IMPL=dpcpp -DSYCL_TARGETS=native_cpu
                                       "-DARCH_FLAGS=$arch"
                                       "-DSYCL_EXTRA_FLAGS=-Xsycl-target-frontend=native_cpu $arch -mllvm -sycl-native-cpu-no-vecz") ;;
    # spir64: kernels are compiled by PoCL at run time (ISA fixed by the PoCL build); host code native ISA
    dpcpp_pocl)       FLOW_CMAKE_ARGS=("-DCMAKE_CXX_COMPILER=$DPCPP_ROOT/bin/clang++" -DSYCL_IMPL=dpcpp -DSYCL_TARGETS=spir64 "-DARCH_FLAGS=$arch") ;;
    omp_twin)         FLOW_CMAKE_ARGS=(-DCMAKE_CXX_COMPILER=clang++ -DSYCL_IMPL=none "-DARCH_FLAGS=$arch") ;;
  esac
}

# Run-time environment for t threads (after flow_load; exported in the current shell).
# cache_dir: empty directory for JIT / kernel caches, so every run compiles its kernels in the
# cold iteration (no reuse of caches from earlier runs).
flow_run_env() {  # <flow> <threads> <cache_dir>
  local f=$1 t=$2 cache=$3
  case $f in
    acpp_omp_accel|acpp_omp_libonly|acpp_generic|omp_twin)
      export OMP_NUM_THREADS=$t OMP_PROC_BIND=close OMP_PLACES=cores
      if [ "$f" != omp_twin ]; then export ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=$cache/acpp; fi ;;
    acpp_pocl)
      export ACPP_VISIBILITY_MASK=ocl ACPP_APPDB_DIR=$cache/acpp
      export POCL_DEVICES=pthread POCL_CPU_MAX_CU_COUNT=$t POCL_AFFINITY=1 POCL_CACHE_DIR=$cache/pocl ;;
    dpcpp_native|dpcpp_native_novecz)
      export ONEAPI_DEVICE_SELECTOR=native_cpu:cpu SYCL_NATIVE_CPU_HOST_THREADS=$t ;;
    dpcpp_pocl)
      export ONEAPI_DEVICE_SELECTOR=opencl:cpu
      export POCL_DEVICES=pthread POCL_CPU_MAX_CU_COUNT=$t POCL_AFFINITY=1 POCL_CACHE_DIR=$cache/pocl ;;
  esac
}
