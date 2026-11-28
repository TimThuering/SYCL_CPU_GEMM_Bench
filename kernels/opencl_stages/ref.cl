// Stage "ref" (OpenCL reference): triple loop on a single work-item (counterpart of the SYCL
// single_task). For testing the harness on small sizes only, not part of the study.

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void gemm_ref(__global const double* a, __global const double* b, __global double* c,
                       ulong m, ulong n, ulong k, ulong lda, ulong ldb, ulong ldc) {
  for (ulong i = 0; i < m; ++i)
    for (ulong j = 0; j < n; ++j) {
      double sum = 0.0;
      for (ulong kk = 0; kk < k; ++kk) sum += a[i * lda + kk] * b[kk * ldb + j];
      c[i * ldc + j] = sum;
    }
}
