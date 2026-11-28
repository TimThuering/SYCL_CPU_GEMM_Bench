// Stage "packed" (OpenCL reference): S4, packed panels + KC/MC cache blocking around the S3 micro-kernel.
// Block sizes: compile-time parameters, passed as -DMR=... -DNR=... -DKC=... -DMC=... when the program is built:
//   MR x NR  micro-tile computed in registers
//   KC       panel length along K (columns of A / rows of B packed at a time)
//   MC       rows of C per work-item in the compute kernel (multiple of MR)
// Workspace: Ap = ws[0 ...], Bp = ws[bp_offset ...] (bp_offset = rows_p · KC elements, set by the host)

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// ceil(a / b) for positive integers, e.g. number of micro-panels needed to cover a rows
size_t ceil_div(size_t a, size_t b) { return (a + b - 1) / b; }

// KERNEL 1: pack rows pc … pc+kc-1 of B (all n columns) into NR-wide micro-panels
//   bp[(jr·KC + p)·NR + s] = B(pc + p, jr·NR + s),  0 for columns >= n (zero padding)
// one work-item per panel row p (global range {kc}): copies row pc + p of B sequentially
// (contiguous reads), NR values at a time into micro-panel jr
__kernel void pack_b(__global const double *b, ulong ldb, ulong n, ulong pc,
                     __global double *ws, ulong bp_offset) {
    __global double *bp = ws + bp_offset;
    const size_t cols_p = ceil_div(n, NR) * NR;

    const size_t p = get_global_id(0);             // k step within the panel   (0 … kc-1)
    for (size_t jj = 0; jj < cols_p; ++jj) {       // column incl. padding      (0 … cols_p-1)
        const size_t jr = jj / NR; // which micro-panel
        const size_t s = jj % NR;  // position inside it
        const double value = (jj < n) ? b[(pc + p) * ldb + jj] : 0.0; // global row = pc + p
        bp[(jr * KC + p) * NR + s] = value;
    }
}

// KERNEL 2: pack columns pc … pc+kc-1 of A (all m rows) into MR-tall micro-panels
//   ap[(ir·KC + p)·MR + r] = A(ir·MR + r, pc + p),  0 for rows >= m (zero padding)
// one work-item per row ii incl. padding (global range {rows_p}): copies the kc panel values of
// row ii of A sequentially (contiguous reads) into micro-panel ir, at position r of every k step
__kernel void pack_a(__global const double *a, ulong lda, ulong m, ulong pc, ulong kc,
                     __global double *ws) {
    __global double *ap = ws;

    const size_t ii = get_global_id(0); // row incl. padding         (0 … rows_p-1)
    const size_t ir = ii / MR; // which micro-panel
    const size_t r = ii % MR;  // position inside it
    for (size_t p = 0; p < kc; ++p) {  // k step within the panel   (0 … kc-1)
        const double value = (ii < m) ? a[ii * lda + (pc + p)] : 0.0; // global column = pc + p
        ap[(ir * KC + p) * MR + r] = value;
    }
}

// KERNEL 3: C (+)= Ap · Bp for the current panel
//   work-item (ic, jr): MC rows x NR columns of C = up to MC/MR micro-tiles, all with Bp micro-panel jr
//   first panel (first != 0): C = result, later panels: C += result
// global range {blocks_m · blocks_n}, t = ic · blocks_n + jr: consecutive work-items share ic
// (same Ap micro-panels), so a contiguous chunk of t per thread reuses its Ap block from cache for many jr
__kernel void compute(__global const double *ws, ulong bp_offset, __global double *c, ulong ldc,
                      ulong m, ulong n, ulong kc, int first) {
    __global const double *ap = ws;
    __global const double *bp = ws + bp_offset;
    const size_t blocks_n = ceil_div(n, NR); // column strips of NR columns = Bp micro-panels
    const size_t tiles_m = ceil_div(m, MR);  // Ap micro-panels in total

    const size_t t = get_global_id(0);
    const size_t ic = t / blocks_n; // row block of MC rows
    const size_t jr = t % blocks_n; // column strip of NR columns
    const size_t ir_begin = ic * (MC / MR);
    const size_t ir_end = (ir_begin + MC / MR < tiles_m) ? ir_begin + MC / MR : tiles_m;

    __global const double *bpanel = bp + jr * KC * NR; // Bp micro-panel jr (reused for all micro-tiles)

    for (size_t ir = ir_begin; ir < ir_end; ++ir) {
        __global const double *apanel = ap + ir * KC * MR; // Ap micro-panel ir

        // micro-kernel: always a full MR x NR tile (padding in Ap/Bp is zero)
        double acc[MR][NR] = {{0.0}};
        for (size_t p = 0; p < kc; ++p) {
            double a_r[MR];
            double b_s[NR];
            for (size_t r = 0; r < MR; ++r) a_r[r] = apanel[p * MR + r];
            for (size_t s = 0; s < NR; ++s) b_s[s] = bpanel[p * NR + s];
            for (size_t r = 0; r < MR; ++r)
                for (size_t s = 0; s < NR; ++s)
                    acc[r][s] += a_r[r] * b_s[s];
        }

        // write-back, masked at the matrix edge (padding results are dropped)
        for (size_t r = 0; r < MR; ++r) {
            const size_t i = ir * MR + r;
            if (i >= m) break;
            for (size_t s = 0; s < NR; ++s) {
                const size_t j = jr * NR + s;
                if (j >= n) break;
                if (first) c[i * ldc + j] = acc[r][s];
                else       c[i * ldc + j] += acc[r][s];
            }
        }
    }
}
