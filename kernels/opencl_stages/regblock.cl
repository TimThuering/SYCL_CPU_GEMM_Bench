// Stage "regblock" (OpenCL reference): S3, MR x NR register block per work-item.
// MR, NR: compile-time parameters, passed as -DMR=... -DNR=... when the program is built.

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Global range {blocks_n, blocks_m}: dimension 0 (fastest) = block column j, dimension 1 = block row i
__kernel void gemm_regblock(__global const double *a, __global const double *b, __global double *c,
                            ulong m, ulong n, ulong k, ulong lda, ulong ldb, ulong ldc) {
    const size_t i = get_global_id(1);
    const size_t j = get_global_id(0);

    // small arrays to be kept in registers
    double acc_r[MR][NR] = {{0.0}};
    double a_r[MR] = {0.0};
    double b_r[NR] = {0.0};

    const size_t i0 = i * MR;
    const size_t j0 = j * NR;
    const bool full = (i0 + MR <= m) && (j0 + NR <= n);

    if (full) {
        // no need for border checks
        for (size_t kk = 0; kk < k; ++kk) {
            // load a_r values
            for (size_t ii = 0; ii < MR; ++ii) {
                    a_r[ii] = a[(i0 + ii) * lda + kk];
            }

            // load b_r values
            for (size_t jj = 0; jj < NR; ++jj) {
                    b_r[jj] = b[kk * ldb + j0 + jj];
            }

            // a_r^T * b_r (outer product)
            for (size_t ii = 0; ii < MR; ++ii) {
                for (size_t jj = 0; jj < NR; ++jj) {
                    acc_r[ii][jj] += a_r[ii] * b_r[jj];
                }
            }
        }

        // write back local result to c
        for (size_t ii = 0; ii < MR; ++ii) {
            for (size_t jj = 0; jj < NR; ++jj) {
                    c[(i0 + ii) * ldc + j0 + jj] = acc_r[ii][jj];
            }
        }
    } else {
        // check borders as block does not fully overlay with matrix
        for (size_t kk = 0; kk < k; ++kk) {
            // load a_r values
            for (size_t ii = 0; ii < MR; ++ii) {
                if (i0 + ii < m) {
                    a_r[ii] = a[(i0 + ii) * lda + kk];
                }
            }

            // load b_r values
            for (size_t jj = 0; jj < NR; ++jj) {
                if (j0 + jj < n) {
                    b_r[jj] = b[kk * ldb + j0 + jj];
                }
            }

            // a_r^T * b_r (outer product)
            for (size_t ii = 0; ii < MR; ++ii) {
                for (size_t jj = 0; jj < NR; ++jj) {
                    acc_r[ii][jj] += a_r[ii] * b_r[jj];
                }
            }
        }

        // write back local result to c
        for (size_t ii = 0; ii < MR; ++ii) {
            for (size_t jj = 0; jj < NR; ++jj) {
                if (i0 + ii < m && j0 + jj < n) {
                    c[(i0 + ii) * ldc + j0 + jj] = acc_r[ii][jj];
                }
            }
        }
    }
}
