// Stage "naive" (OpenCL reference): S1, one C element per work-item

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// One work-item per element of C: global range {N, M}, dimension 0 (fastest) = j
__kernel void gemm_naive(__global const double *a, __global const double *b, __global double *c,
                         ulong m, ulong n, ulong k, ulong lda, ulong ldb, ulong ldc) {
    const size_t i = get_global_id(1);
    const size_t j = get_global_id(0);

    double sum = 0.0;
    for (size_t kk = 0; kk < k; ++kk) {
        sum += a[i * lda + kk] * b[kk * ldb + j];
    }
    c[i * ldc + j] = sum;
}
