// Stage "interchange" (OpenCL reference): S2, one C row per work-item, k outer / j inner

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// One work-item per row of C: global range {M}; inside, k outer / j inner (contiguous in B and C)
__kernel void gemm_interchange(__global const double *a, __global const double *b, __global double *c,
                               ulong m, ulong n, ulong k, ulong lda, ulong ldb, ulong ldc) {
    const size_t i = get_global_id(0);

    for (size_t j = 0; j < n; j++) {
        c[i * ldc + j] = 0.0;
    }

    for (size_t kk = 0; kk < k; kk++) {
        const double a_ik = a[i * lda + kk];
        for (size_t j = 0; j < n; j++) {
            c[i * ldc + j] +=  a_ik * b[kk * ldb + j];
        }
    }
}
