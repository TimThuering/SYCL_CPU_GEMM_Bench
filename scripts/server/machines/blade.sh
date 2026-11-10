# Machine profile: Monte Cimone blade nodes (2x SOPHGO SG2042, 128 cores, 8 NUMA nodes,
# RVV 0.7.1 not usable by the compilers -> scalar ISA with the T-Head extensions)

GB_PARTITION=blade
GB_MODULE_TREE=$HOME/.modulefiles_sg2042
GB_THREADS_NODE=128
GB_THREADS_SOCKET=64

GB_FLOWS="acpp_omp_accel acpp_omp_libonly acpp_generic acpp_pocl dpcpp_native dpcpp_native_novecz dpcpp_pocl omp_twin"

gb_cpu_list() { echo "0-$(($1 - 1))"; }

GB_SG2042_MARCH=rv64gc_xtheadba_xtheadbb_xtheadbs_xtheadcmo_xtheadcondmov_xtheadfmemidx_xtheadmac_xtheadmemidx_xtheadmempair_xtheadsync
gb_arch() { echo "-march=$GB_SG2042_MARCH"; }
