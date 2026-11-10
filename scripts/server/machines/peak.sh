# Machine profile: Monte Cimone peak nodes (SOPHGO SG2044, 64 cores, RVV 1.0, 1 NUMA node)

GB_PARTITION=peak
GB_MODULE_TREE=$HOME/.modulefiles_sg2044
GB_THREADS_NODE=64
GB_THREADS_SOCKET=64

GB_FLOWS="acpp_omp_accel acpp_omp_libonly acpp_generic acpp_pocl dpcpp_native dpcpp_native_novecz dpcpp_pocl omp_twin"

# CPU list for t threads (taskset): cores 0 .. t-1
gb_cpu_list() { echo "0-$(($1 - 1))"; }

# ISA flags (evaluated after the flow's modules are loaded; needs clang in PATH)
gb_arch() {  # <flow>
  case $1 in
    # DPC++ 7.1 Native CPU: Vecz crashes with V in the device features -> scalar ISA for the Vecz flow;
    # dpcpp_native_novecz gets the vector ISA (LLVM loop vectorizer instead of Vecz)
    dpcpp_native) gb_native_march novec ;;
    *) gb_native_march vec ;;
  esac
}
