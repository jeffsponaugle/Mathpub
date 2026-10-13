/*
 * a171775_2d_cuda.cu -- CUDA (DGX Spark / GB10, sm_121) port of the swapped 2D A171775 search.
 *
 * Host logic comes from ../a171775_2d.c (included as a library, see below):
 * problem setup, units (c, e_0), the per-block ranges of check bases B, the
 * per-(c, B) constants and lookup tables, the survivor post-processing and
 * the state files.  Read the theory of operations there first.  The kernel
 * is a port of the one in ../metal/a171775_2d.metal.  So this tool, a171775_2d
 * (CPU) and a171775_metal produce the same statistics and share checkpoint
 * files.
 *
 * ========================================================================
 * THEORY OF OPERATIONS (the GPU part)
 * ========================================================================
 *
 * 1. Work.  A batch is a run of consecutive units of one base c (whole
 *    units, sized so a launch takes about 0.25 s; g_target_steps adapts to
 *    the measured rate).  Its blocks, each with its range of check bases B,
 *    are listed (batch_blocks(), as do_block() does), and for every B the
 *    consecutive blocks whose range contains B are cut into items of at most
 *    256 blocks (make_items()).  One CUDA block of 256 threads processes one
 *    item: one base pair (c, B), one thread per block of the search.
 *
 * 2. Tables.  The pair's lookup table (val, ys, bst) and digit-k tables
 *    (ay/az) are built on the host by cctx_set() and copied to the GPU per
 *    base c (slot_set_c()); each CUDA block copies its pair's tables into
 *    shared memory (c <= 1024).  The pair's constants are converted to
 *    PairC, with 96-bit values as three 32-bit limbs.
 *
 * 3. A thread's start state.  The host provides the base-B digits of each
 *    block weight w_t (blkw); the thread adds e_t times them with carries to
 *    get the base-B digits D[] of Mblock, so it never divides a 128-bit
 *    number: q1 = the top k digits, rem = M1 mod P from the low n-k digits,
 *    M1 mod B^k, digit k (mk_hi) and W = (T - M1) u mod N follow directly.
 *    Then it walks x = 0..c-1 exactly as block_B(): one or two lookups per x
 *    (W div g by a "magic" multiply: mulhi(W, gm) >> gs, exact for W < 2^31
 *    and verified on the host), with rem kept in 96 bits.
 *
 * 4. A match (process_match()) gets the same steps as in block_B(): the
 *    region test, the [LO, HI] test (only for items with blocks reaching
 *    outside the range), the digit pair (k, n-1-k) -- digit n-1-k from a
 *    float quotient, corrected exactly near a digit boundary -- and the full
 *    n-digit palindrome test from the digits of rem and q.  A survivor goes
 *    to the host as (q1, rem, B, c, x, j); the host rebuilds M, checks both
 *    palindromes again (a mismatch stops the run) and calls survivor().
 *    Each CUDA block sums its lookups, hits and f1 (warp shuffles) into
 *    tgstat[].
 *
 * 5. Pipeline and checkpoints.  Two slots (streams) alternate: the host
 *    prepares and launches one batch while the GPU runs the other, and
 *    batches finish in issue order, so the statistics always cover exactly
 *    the units below the oldest batch in flight; that frontier is what -S
 *    saves, so every checkpoint is exact.
 *
 * 6. Limits.  n = 8..14 (n + 2 <= MAXL_G digits, 1..3 block digits);
 *    c <= 1024 (MAXC, shared memory) with at most 1100 check bases;
 *    P, Ptop and the weights must fit in 96 bits; N <= 2^31 when g > 1; at
 *    most 2^15 items and 2^20 survivors per launch.  Every violation stops
 *    the run with a message (exit status 1).
 *
 * The file is compiled as C++ by nvcc, so ../a171775_2d.c is included with
 * A171775_2D_LIB (no main) and A171775_2D_NO_DRIVER (no C11 thread driver).
 *
 * Usage
 *   a171775_2d_cuda selftest              n = 8, 9, 10 full searches vs the CPU tool's statistics
 *   a171775_2d_cuda search n LO HI [-S state] [-i sec] [-E end_unit]
 * Exit status: 0 = done, 3 = interrupted, 1 = usage or a fatal error (CUDA
 * errors name the failing call).
 */
#define A171775_2D_LIB
#define A171775_2D_NO_DRIVER
#include "../a171775_2d.c"
#include <cuda_runtime.h>

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

#define TG 256                        /* threads per CUDA block = blocks of the search per item */
#define NWARP (TG / 32)
#define MAXK_G 3                      /* largest k (digit pairs imposed by the lookup) */
#define MAXL_G 16                     /* digit buffer size: n + 2 <= MAXL_G */

/*
 * PairC -- the constants of one pair (c, B) in the GPU layout (identical to
 * the Metal kernel's): the bconst_t fields of ../a171775_2d.c as 32-bit
 * values (B, k, N, g, N', u, lim, R, Wx, Wc[]), 96-bit values as three
 * limbs (P, Rc, Ptop, w_x, w_y, w_z, and N' w_z, the step between the z of
 * one match), the table offsets toff/boff of this pair in the slot's arrays,
 * Lc = n and Lk = n - k, the digit-k split (ox_lo, ox_hi, Bk), the base-B
 * digits of the block weights (blkw, for the start state), 1/Ptop and 1/g
 * as floats, and the magic multiplier gm, shift gs for W div g.
 */
struct PairC {                        /* identical to the Metal layout */
    uint32_t B, k, N, g, Np, u, lim, R;
    uint32_t Wx, sh, nb, toff, boff, Lc, Lk, c;
    uint32_t Wc[4];
    uint32_t P[4], Rc[4], Ptop[4];
    uint32_t wx[4], wy[4], wz[4], NpWz[4];
    uint32_t ox_lo, ox_hi, Bk, bd;
    uint32_t blkw[3][MAXL_G];
    float invPtop, invg;
    uint32_t gm, gs;
};
/* GItem -- one CUDA block's work: the pair index (B - Bbase), the linear
   index j0 of its first block of the search, the number of blocks (<= 256),
   and pad bit 0 = some block reaches outside [lo, hi] (range test on).
   Surv -- a survivor: q1 and rem = r0..r2 (M = q1 P + rem), B, c, x and the
   block index j.  Params -- [lo, hi] as 32-bit limbs, the survivor buffer
   size and n.  PC -- the part of PairC that process_match() reads, copied to
   shared memory once per CUDA block. */
struct GItem { uint32_t pair, j0, count, pad; };
struct Surv { uint32_t q, r0, r1, r2, B, c, x, j; };
struct Params { uint32_t lo[4], hi[4]; uint32_t outcap, n, pad0, pad1; };
struct PC { uint32_t B, k, Lc, Lk, Bk, toff, c, pad; uint32_t P[4], Ptop[4], wy[4], wz[4]; float invPtop; };

/* ------------------------------------------------------------------ */
/* device code                                                          */
/* ------------------------------------------------------------------ */

/* 96-bit arithmetic on three 32-bit limbs (a0 least significant), in 32-bit
   operations only:
     ge96(a, b)       a >= b
     add96(a, b)      a += b      (no overflow check: callers stay below 2^96)
     sub96(a, b)      a -= b      (a >= b)
     mul96(a, m)      a *= m      (m < 2^21 here, product < 2^96 by construction)
     div96(a, d)      a /= d, returns a mod d (d < 2^16), in 16-bit chunks */
__device__ __forceinline__ bool ge96(unsigned a0, unsigned a1, unsigned a2, unsigned b0, unsigned b1, unsigned b2)
{
    if (a2 != b2) return a2 > b2;
    if (a1 != b1) return a1 > b1;
    return a0 >= b0;
}

__device__ __forceinline__ void add96(unsigned &a0, unsigned &a1, unsigned &a2, unsigned b0, unsigned b1, unsigned b2)
{
    unsigned s0 = a0 + b0, c0 = s0 < b0;
    unsigned s1 = a1 + b1, c1 = s1 < b1;
    unsigned t1 = s1 + c0;
    c1 += t1 < c0;
    a0 = s0; a1 = t1; a2 = a2 + b2 + c1;
}

__device__ __forceinline__ void sub96(unsigned &a0, unsigned &a1, unsigned &a2, unsigned b0, unsigned b1, unsigned b2)
{
    unsigned br0 = a0 < b0, d0 = a0 - b0;
    unsigned br1 = a1 < b1, d1 = a1 - b1;
    br1 += d1 < br0;
    a0 = d0; a1 = d1 - br0; a2 = a2 - b2 - br1;
}

__device__ __forceinline__ void mul96(unsigned &a0, unsigned &a1, unsigned &a2, unsigned m)
{
    unsigned l0 = a0 * m, h0 = __umulhi(a0, m);
    unsigned l1 = a1 * m, h1 = __umulhi(a1, m);
    unsigned n1 = l1 + h0, c1 = n1 < h0;
    a0 = l0; a1 = n1; a2 = a2 * m + h1 + c1;
}

__device__ __forceinline__ unsigned div96(unsigned &a0, unsigned &a1, unsigned &a2, unsigned d)
{
    unsigned r = 0, q, h;
    h = (r << 16) | (a2 >> 16); q = h / d; r = h - q * d; unsigned q2 = q << 16;
    h = (r << 16) | (a2 & 0xffff); q = h / d; r = h - q * d; q2 |= q;
    h = (r << 16) | (a1 >> 16); q = h / d; r = h - q * d; unsigned q1 = q << 16;
    h = (r << 16) | (a1 & 0xffff); q = h / d; r = h - q * d; q1 |= q;
    h = (r << 16) | (a0 >> 16); q = h / d; r = h - q * d; unsigned q0 = q << 16;
    h = (r << 16) | (a0 & 0xffff); q = h / d; r = h - q * d; q0 |= q;
    a0 = q0; a1 = q1; a2 = q2;
    return r;
}

/*
 * process_match -- (device) one lookup match (y, z) of the thread's current
 * x: the hit code of block_B() in ../a171775_2d.c.
 *
 * In:     r0..r2 = rem (M1 mod P), q1, mklo/mk_hi (M1 mod B^(k+1)), reg (the
 *         region the lookup was for: 0 = q1, 1 = q1+1), y, z, x, tid; K (the
 *         pair, in shared memory); prm; the digit-k tables; the survivor
 *         buffer and its counter; j0 (first block index of the item);
 *         need_range (test [lo, hi]); the thread's hit and f1 counters.
 * Out:    counts the hit if (y, z) lies in region reg, and f1 if it passes
 *         the digit-pair filter; a full n-digit palindrome is appended to
 *         outs[] (dropped past prm.outcap; the host then stops with
 *         "survivor buffer overflow").
 * Errors: none.
 */
__device__ __forceinline__ void process_match(unsigned r0, unsigned r1, unsigned r2, unsigned q1, unsigned mklo,
                                              unsigned mk_hi, unsigned reg, unsigned y, unsigned z, unsigned x,
                                              unsigned tid, const PC &K, const Params &prm, const uint32_t *say_lo,
                                              const uint16_t *say_hi, const uint32_t *saz_lo, const uint16_t *saz_hi,
                                              Surv *outs, unsigned *nout, unsigned j0, bool need_range,
                                              unsigned long long &nhits, unsigned long long &nf1)
{
    const unsigned B = K.B, k = K.k, Lc = K.Lc, Lk = K.Lk, Bk = K.Bk;
    const unsigned P0 = K.P[0], P1 = K.P[1], P2 = K.P[2];
    unsigned d0 = K.wy[0], d1 = K.wy[1], d2 = K.wy[2];
    mul96(d0, d1, d2, y);
    unsigned f0 = K.wz[0], f1 = K.wz[1], f2 = K.wz[2];
    mul96(f0, f1, f2, z);
    add96(d0, d1, d2, f0, f1, f2);
    unsigned s0 = r0, s1 = r1, s2 = r2;
    add96(s0, s1, s2, d0, d1, d2);
    bool ge = ge96(s0, s1, s2, P0, P1, P2);
    if ((reg != 0) != ge) return;             /* (y, z) is in the other q-region */
    nhits++;
    if (need_range) {                         /* M = q1 P + rf as u128 against [lo, hi] */
        unsigned long long t = (unsigned long long)P0 * q1 + s0;
        unsigned m0 = (unsigned)t;
        t = (unsigned long long)P1 * q1 + s1 + (t >> 32);
        unsigned m1 = (unsigned)t;
        t = (unsigned long long)P2 * q1 + s2 + (t >> 32);
        unsigned m2 = (unsigned)t, m3 = (unsigned)(t >> 32);
        bool below = m3 != prm.lo[3] ? m3 < prm.lo[3] : m2 != prm.lo[2] ? m2 < prm.lo[2]
                   : m1 != prm.lo[1] ? m1 < prm.lo[1] : m0 < prm.lo[0];
        bool above = m3 != prm.hi[3] ? m3 > prm.hi[3] : m2 != prm.hi[2] ? m2 > prm.hi[2]
                   : m1 != prm.hi[1] ? m1 > prm.hi[1] : m0 > prm.hi[0];
        if (below || above) return;
    }
    /* digit k of M from (M1 + y w_y + z w_z) mod B^(k+1), 32-bit halves */
    unsigned lo = mklo + say_lo[y] + saz_lo[z];
    unsigned hi = mk_hi + (unsigned)say_hi[y] + (unsigned)saz_hi[z];
    if (lo >= Bk) { lo -= Bk; hi++; }
    if (lo >= Bk) { lo -= Bk; hi++; }
    while (hi >= B) hi -= B;
    const unsigned dlo = hi;
    /* digit Lc-1-k of M = top digit of r (r = rf, or rf - P in the q1+1 region) */
    unsigned a0 = s0, a1 = s1, a2 = s2;
    if (reg) sub96(a0, a1, a2, P0, P1, P2);
    /* float32 quotient: relative error about 3e-7, so below 3.1e-4 for digits
       < 1024 (c <= 1024), inside the 0.00049 margin; otherwise corrected exactly */
    float qf = ((float)a2 * 18446744073709551616.0f + (float)a1 * 4294967296.0f + (float)a0) * K.invPtop;
    int dh = (int)qf;
    float fr = qf - (float)dh;
    if (fr < 0.00049f || fr > 0.99951f) {    /* near a digit boundary: exact correction */
        if (dh < 0) dh = 0;
        for (;;) {
            unsigned p0 = K.Ptop[0], p1 = K.Ptop[1], p2 = K.Ptop[2];
            mul96(p0, p1, p2, (unsigned)dh);
            if (!ge96(a0, a1, a2, p0, p1, p2)) { dh--; continue; }
            add96(p0, p1, p2, K.Ptop[0], K.Ptop[1], K.Ptop[2]);
            if (ge96(a0, a1, a2, p0, p1, p2)) { dh++; continue; }
            break;
        }
    }
    if (dlo != (unsigned)dh) return;
    nf1++;
    /* full check: digits of r (Lk of them) and of q = q1 + reg (k of them) */
    unsigned dig[MAXL_G + 2];
    for (unsigned t = 0; t < Lk; t++) dig[t] = div96(a0, a1, a2, B);
    unsigned qv = q1 + reg;
    for (unsigned t = 0; t < k; t++) { unsigned qd = qv / B; dig[Lk + t] = qv - qd * B; qv = qd; }
    bool pal = dig[Lc - 1] != 0 && qv == 0 && (a0 | a1 | a2) == 0;
    for (unsigned t = 0; pal && t < Lc / 2; t++) pal = dig[t] == dig[Lc - 1 - t];
    if (!pal) return;
    unsigned idx = atomicAdd(nout, 1u);
    if (idx < prm.outcap) {
        Surv sv;
        sv.q = q1; sv.r0 = s0; sv.r1 = s1; sv.r2 = s2;
        sv.B = B; sv.c = K.c; sv.x = x; sv.j = j0 + tid;
        outs[idx] = sv;
    }
}

/*
 * scan2 -- (device) one CUDA block = one item: TG threads, thread tid works
 * on block j0 + tid of the search (if tid < count) for the item's pair
 * (c, B), i.e. one call of block_B() per thread.
 *
 * In:     items, pairs (the slot's PairC array), the slot's table arrays
 *         (tvals/tys/tbst = the key table, tay_*, taz_* = digit-k tables, at
 *         the pair's offsets toff and boff), prm; outs/nout for survivors;
 *         tgstat for the per-item counts.
 * Out:    survivors in outs[] (via process_match()); tgstat[3 item] =
 *         lookups, hits, f1 of the item.
 * Errors: none; a thread whose Mblock already has more than n base-B digits
 *         stays idle.
 */
__global__ void __launch_bounds__(TG) scan2(const GItem *__restrict__ items, const PairC *__restrict__ pairs,
                                            const uint32_t *__restrict__ tvals, const uint16_t *__restrict__ tys,
                                            const uint16_t *__restrict__ tbst, const uint32_t *__restrict__ tay_lo,
                                            const uint16_t *__restrict__ tay_hi, const uint32_t *__restrict__ taz_lo,
                                            const uint16_t *__restrict__ taz_hi, Params prm, Surv *outs,
                                            unsigned *nout, unsigned long long *tgstat)
{
    __shared__ uint32_t sval[1024];
    __shared__ uint16_t sys[1024];
    __shared__ uint16_t sbst[1032];
    __shared__ uint32_t say_lo[1024], saz_lo[1024];
    __shared__ uint16_t say_hi[1024], saz_hi[1024];
    __shared__ unsigned long long red[3 * NWARP];
    __shared__ PC pc;

    const unsigned tg = blockIdx.x, tid = threadIdx.x, tgsz = blockDim.x;
    const GItem it = items[tg];
    const PairC &K = pairs[it.pair];
    const unsigned c = K.c, B = K.B;
    /* stage the pair's tables in shared memory (all threads), then sync */
    if (tid == 0) {
        pc.B = B; pc.k = K.k; pc.Lc = K.Lc; pc.Lk = K.Lk; pc.Bk = K.Bk; pc.toff = K.toff; pc.c = c;
        for (int i = 0; i < 4; i++) { pc.P[i] = K.P[i]; pc.Ptop[i] = K.Ptop[i]; pc.wy[i] = K.wy[i]; pc.wz[i] = K.wz[i]; }
        pc.invPtop = K.invPtop;
    }
    for (unsigned i = tid; i < c; i += tgsz) {
        sval[i] = tvals[K.toff + i]; sys[i] = tys[K.toff + i];
        say_lo[i] = tay_lo[K.toff + i]; say_hi[i] = tay_hi[K.toff + i];
        saz_lo[i] = taz_lo[K.toff + i]; saz_hi[i] = taz_hi[K.toff + i];
    }
    for (unsigned i = tid; i <= K.nb; i += tgsz) sbst[i] = tbst[K.boff + i];
    __syncthreads();

    const unsigned k = K.k, N = K.N, g = K.g, Np = K.Np, sh = K.sh, Lc = K.Lc, Lk = K.Lk;
    const unsigned Wx = K.Wx, bd = K.bd, Bk = K.Bk, ox_lo = K.ox_lo, ox_hi = K.ox_hi, gm = K.gm, gs = K.gs;
    const bool need_range = (it.pad & 1u) != 0;
    const unsigned P0 = K.P[0], P1 = K.P[1], P2 = K.P[2];
    const unsigned Rc0 = K.Rc[0], Rc1 = K.Rc[1], Rc2 = K.Rc[2];
    const unsigned wx0 = K.wx[0], wx1 = K.wx[1], wx2 = K.wx[2];

    unsigned long long nlook = 0, nhits = 0, nf1 = 0;
    bool active = tid < it.count;
    /* block_B()'s state: digits E[] and value q1 of floor(M1 / P), W, rem =
       (r0, r1, r2), mk = M1 mod B^(k+1), wnext and last */
    unsigned E[MAXK_G] = {0, 0, 0};
    unsigned q1 = 0, W = 0, r0 = 0, r1 = 0, r2 = 0, mk_lo = 0, mk_hi = 0, wnext = 0;
    bool last = false;
    /* RECOMPUTE(): as in block_B(), wnext and last from the digits E[] */
#define RECOMPUTE()                                                                          \
    {                                                                                        \
        int jj = (int)k - 1;                                                                 \
        while (jj >= 0 && E[jj] == B - 1) jj--;                                              \
        last = jj < 0;                                                                       \
        if (jj == (int)k - 1) wnext = K.Wc[k - 1];                                           \
        else if (jj >= 0) {                                                                  \
            wnext = K.Wc[jj + 1] + K.Wc[jj];                                                 \
            if (wnext >= N) wnext -= N;                                                      \
        } else wnext = 0;                                                                    \
    }
    if (active) {
        /* block digits (e_0 >= 1) and base-B digits of Mblock: Mblock = sum
           e_t w_t, digit by digit from the base-B digits of the weights
           (blkw), with carries; then q1, T, M1 mod B^k, rem and W from D[] */
        unsigned j = it.j0 + tid, e[3] = {0, 0, 0};
        {
            unsigned t = j;
            for (int i = (int)bd - 1; i >= 1; i--) { e[i] = t % c; t /= c; }
            e[0] = 1 + t;
        }
        unsigned D[MAXL_G + 2];
        unsigned carry = 0;
        for (unsigned i = 0; i < Lc + 2; i++) {
            unsigned s = carry;
            for (unsigned t = 0; t < bd; t++) s += e[t] * K.blkw[t][i];
            carry = s / B;
            D[i] = s - carry * B;
        }
        if (carry == 0 && D[Lc] == 0 && D[Lc + 1] == 0) {            /* Mblock < B^Lc */
            for (unsigned i = 0; i < k; i++) E[i] = D[Lc - 1 - i];  /* E[0] = most significant digit of q */
            unsigned T = 0, MN = 0;
            for (unsigned i = 0; i < k; i++) q1 = q1 * B + E[i];
            for (int i = (int)k - 1; i >= 0; i--) { T = T * B + E[i]; MN = MN * B + D[i]; }
            unsigned tmn = T >= MN ? T - MN : T + N - MN;
            W = (unsigned)(((unsigned long long)tmn * K.u) % N);
            for (int i = (int)Lk - 1; i >= 0; i--) {
                mul96(r0, r1, r2, B);
                add96(r0, r1, r2, D[i], 0, 0);
            }
            mk_lo = MN;                  /* M1 mod B^(k+1) = mk_hi B^k + mk_lo */
            mk_hi = D[k];
            RECOMPUTE();
        } else active = false;
    }

    /* MATCH(z): hand the match (y, z) of the current lookup to process_match() */
#define MATCH(ZZ)                                                                            \
    process_match(r0, r1, r2, q1, mk_lo, mk_hi, (unsigned)reg, y, (ZZ), x, tid, pc, prm, say_lo, say_hi, \
                  saz_lo, saz_hi, outs, nout, it.j0, need_range, nhits, nf1)

    /* the x loop of block_B(): lookup(s), then advance x */
    for (unsigned x = 0; x < c && active; x++) {
        for (int reg = 0; reg < 2; reg++) {
            unsigned Wt;
            if (reg == 0) Wt = W;
            else {
                if (last || !ge96(r0, r1, r2, Rc0, Rc1, Rc2)) break;
                Wt = W + wnext;
                if (Wt >= N) Wt -= N;
            }
            nlook++;
            /* keys are (v mod g) N' + v div g: the matches form a window of width
               min(c, N') inside residue class r = Wt mod g, ending at m = Wt div g */
            unsigned rr = 0, mW = Wt;
            if (g > 1) { mW = __umulhi(Wt, gm) >> gs; rr = Wt - mW * g; }
            const unsigned base = rr * Np;
            unsigned m_lo, m_hi = mW;
            bool wrap;
            if (c >= Np) { m_lo = 0; m_hi = Np - 1; wrap = false; }
            else {
                m_lo = mW + 1 >= c ? mW + 1 - c : mW + 1 + Np - c;
                wrap = m_lo > m_hi;
            }
            for (int part = 0; part <= (int)wrap; part++) {
                unsigned a = base + (part == 0 ? m_lo : 0), b = base + ((wrap && part == 0) ? Np - 1 : m_hi);
                unsigned i0 = sbst[a >> sh], i1 = sbst[(b >> sh) + 1];
                for (unsigned i = i0; i < i1; i++) {
                    unsigned v = sval[i];
                    if (v < a) continue;
                    if (v > b) break;
                    unsigned m = v - base;
                    unsigned y = sys[i];
                    unsigned z0 = mW >= m ? mW - m : mW + Np - m;
                    if (Np >= c) {           /* only z0 < c: process_match() tests the region */
                        MATCH(z0);
                    } else {                 /* several z: walk them here with the region test */
                        unsigned d0 = K.wy[0], d1 = K.wy[1], d2 = K.wy[2];
                        mul96(d0, d1, d2, y);
                        unsigned f0 = K.wz[0], f1 = K.wz[1], f2 = K.wz[2];
                        mul96(f0, f1, f2, z0);
                        add96(d0, d1, d2, f0, f1, f2);
                        unsigned s0 = r0, s1 = r1, s2 = r2;
                        add96(s0, s1, s2, d0, d1, d2);
                        for (unsigned z = z0; z < c; z += Np) {
                            bool ge = ge96(s0, s1, s2, P0, P1, P2);
                            if (reg == 0) {
                                if (ge) break;
                                MATCH(z);
                            } else if (ge) MATCH(z);
                            add96(s0, s1, s2, K.NpWz[0], K.NpWz[1], K.NpWz[2]);
                        }
                    }
                }
            }
        }
        /* advance x */
        add96(r0, r1, r2, wx0, wx1, wx2);
        if (ge96(r0, r1, r2, P0, P1, P2)) {
            sub96(r0, r1, r2, P0, P1, P2);
            if (last) active = false;                /* q1 reaches B^k: M >= B^n from here on */
            else {
                W += wnext;
                if (W >= N) W -= N;
                int jj = (int)k - 1;
                while (jj >= 0) {
                    E[jj]++;
                    if (E[jj] < B) break;
                    E[jj] = 0;
                    jj--;
                }
                q1++;
                RECOMPUTE();
            }
        }
        W = W >= Wx ? W - Wx : W + N - Wx;
        mk_lo += ox_lo;
        mk_hi += ox_hi;
        if (mk_lo >= Bk) { mk_lo -= Bk; mk_hi++; }
        if (mk_hi >= B) mk_hi -= B;
    }
#undef RECOMPUTE
#undef MATCH
    /* per-item counts: sum within each warp by shuffles (every thread gets
       here, active or not), then thread 0 adds up the warps */
    for (int off = 16; off > 0; off >>= 1) {
        nlook += __shfl_down_sync(0xffffffffu, nlook, off);
        nhits += __shfl_down_sync(0xffffffffu, nhits, off);
        nf1 += __shfl_down_sync(0xffffffffu, nf1, off);
    }
    if ((tid & 31) == 0) { red[3 * (tid >> 5)] = nlook; red[3 * (tid >> 5) + 1] = nhits; red[3 * (tid >> 5) + 2] = nf1; }
    __syncthreads();
    if (tid == 0) {
        unsigned long long a = 0, b = 0, cc = 0;
        for (unsigned i = 0; i < (tgsz + 31) / 32; i++) { a += red[3 * i]; b += red[3 * i + 1]; cc += red[3 * i + 2]; }
        tgstat[3 * tg] = a;
        tgstat[3 * tg + 1] = b;
        tgstat[3 * tg + 2] = cc;
    }
}

/* ------------------------------------------------------------------ */
/* host side                                                            */
/* ------------------------------------------------------------------ */

/* put96 -- store v as three 32-bit limbs in d[0..2] (d[3] = 0); exits with
   status 1 ("<what> does not fit in 96 bits") if v >= 2^96. */
static void put96(uint32_t *d, u128 v, const char *what)
{
    if (v >> 96) { fprintf(stderr, "%s does not fit in 96 bits\n", what); exit(1); }
    d[0] = (uint32_t)v; d[1] = (uint32_t)(v >> 32); d[2] = (uint32_t)(v >> 64); d[3] = 0;
}

#define MAXPAIRS 1100                  /* check bases per c (B < c <= 1024) */
#define MAXC 1024                      /* largest c (shared-memory tables) */
#define MAXITEMS (1 << 15)             /* items (CUDA blocks) per launch */
#define OUTCAP (1u << 20)              /* survivors per launch */

/* Slot -- one of the two pipeline slots: a stream, timing events, device and
   pinned host buffers sized for the largest c, the host tables (cctx_t) of
   the slot's current base c, and the batch in flight: its items, units
   [u_begin, u_end), steps and whether it is busy. */
struct Slot {
    cudaStream_t stream;
    cudaEvent_t ev0, ev1;
    /* device */
    GItem *d_items; PairC *d_pairs; uint32_t *d_tvals, *d_tay_lo, *d_taz_lo; uint16_t *d_tys, *d_tbst, *d_tay_hi, *d_taz_hi;
    Surv *d_outs; unsigned *d_nout; unsigned long long *d_tgstat;
    /* pinned host */
    GItem *h_items; PairC *h_pairs; uint32_t *h_tvals, *h_tay_lo, *h_taz_lo; uint16_t *h_tys, *h_tbst, *h_tay_hi, *h_taz_hi;
    Surv *h_outs; unsigned *h_nout; unsigned long long *h_tgstat;
    cctx_t cc;
    u64 c;
    size_t nitems;
    u64 u_begin, u_end, steps;
    int busy;
};

/* slot_alloc -- create the slot's stream and events and allocate all its
   buffers for the largest sizes (MAXPAIRS pairs of c = MAXC); about 30 MB per
   slot on each side.  CUDA errors exit (CK). */
static void slot_alloc(Slot *s)
{
    memset(s, 0, sizeof *s);
    CK(cudaStreamCreate(&s->stream));
    CK(cudaEventCreate(&s->ev0));
    CK(cudaEventCreate(&s->ev1));
    size_t ne = (size_t)MAXPAIRS * MAXC, nbst = (size_t)MAXPAIRS * (MAXC + 3);
    CK(cudaMalloc(&s->d_items, MAXITEMS * sizeof(GItem)));
    CK(cudaMalloc(&s->d_pairs, MAXPAIRS * sizeof(PairC)));
    CK(cudaMalloc(&s->d_tvals, ne * 4)); CK(cudaMalloc(&s->d_tay_lo, ne * 4)); CK(cudaMalloc(&s->d_taz_lo, ne * 4));
    CK(cudaMalloc(&s->d_tys, ne * 2)); CK(cudaMalloc(&s->d_tay_hi, ne * 2)); CK(cudaMalloc(&s->d_taz_hi, ne * 2));
    CK(cudaMalloc(&s->d_tbst, nbst * 2));
    CK(cudaMalloc(&s->d_outs, OUTCAP * sizeof(Surv)));
    CK(cudaMalloc(&s->d_nout, sizeof(unsigned)));
    CK(cudaMalloc(&s->d_tgstat, 3 * MAXITEMS * sizeof(unsigned long long)));
    CK(cudaMallocHost(&s->h_items, MAXITEMS * sizeof(GItem)));
    CK(cudaMallocHost(&s->h_pairs, MAXPAIRS * sizeof(PairC)));
    CK(cudaMallocHost(&s->h_tvals, ne * 4)); CK(cudaMallocHost(&s->h_tay_lo, ne * 4)); CK(cudaMallocHost(&s->h_taz_lo, ne * 4));
    CK(cudaMallocHost(&s->h_tys, ne * 2)); CK(cudaMallocHost(&s->h_tay_hi, ne * 2)); CK(cudaMallocHost(&s->h_taz_hi, ne * 2));
    CK(cudaMallocHost(&s->h_tbst, nbst * 2));
    CK(cudaMallocHost(&s->h_outs, OUTCAP * sizeof(Surv)));
    CK(cudaMallocHost(&s->h_nout, sizeof(unsigned)));
    CK(cudaMallocHost(&s->h_tgstat, 3 * MAXITEMS * sizeof(unsigned long long)));
}

/*
 * slot_set_c -- make the slot ready for base c (nothing to do if it already
 * is): cctx_set() builds the host tables, every pair is converted to PairC
 * (96-bit limbs, the digits of the block weights, the magic division for g,
 * spot-checked on 4096 values) and the tables are packed and uploaded on the
 * slot's stream.  Call only when the slot is idle.
 * Errors: exits with status 1 if n or c exceed the kernel's limits, a value
 * does not fit in 96 bits, N > 2^31 with g > 1, or the magic division fails
 * its check.
 */
static void slot_set_c(Slot *s, const prob_t *pr, u64 c)
{
    if (s->c == c) return;
    cctx_set(&s->cc, pr, c);
    s->c = c;
    const cctx_t *cc = &s->cc;
    const int n = pr->n;
    if (n + 2 > MAXL_G) { fprintf(stderr, "n too large for the GPU kernel\n"); exit(1); }
    if (cc->Bcount > MAXPAIRS || c > MAXC) { fprintf(stderr, "c = %llu too large for the GPU buffers\n", (unsigned long long)c); exit(1); }
    size_t toff = 0, boff = 0;
    for (u64 i = 0; i < cc->Bcount; i++) {
        const bconst_t *K = &cc->tab[i];
        PairC *p = &s->h_pairs[i];
        memset(p, 0, sizeof *p);
        p->B = (uint32_t)K->B; p->k = (uint32_t)K->k; p->N = (uint32_t)K->N; p->g = (uint32_t)K->g;
        p->Np = (uint32_t)K->Np; p->u = (uint32_t)K->u; p->lim = (uint32_t)K->lim; p->R = (uint32_t)K->R;
        p->Wx = (uint32_t)K->Wx; p->sh = (uint32_t)K->sh; p->nb = K->nb;
        p->toff = (uint32_t)toff; p->boff = (uint32_t)boff;
        p->Lc = (uint32_t)n; p->Lk = (uint32_t)(n - (int)K->k); p->c = (uint32_t)c;
        for (int j = 0; j < (int)K->k; j++) p->Wc[j] = (uint32_t)K->Wc[j];
        put96(p->P, K->P, "P");
        put96(p->Rc, K->Rc, "Rc");
        put96(p->Ptop, K->Ptop, "Ptop");
        put96(p->wx, cc->wx, "w_x");
        put96(p->wy, cc->wy, "w_y");
        put96(p->wz, cc->wz, "w_z");
        if (K->Np < c) put96(p->NpWz, (u128)K->Np * cc->wz, "N' w_z");
        p->ox_lo = (uint32_t)(K->oxK % K->Bk);
        p->ox_hi = (uint32_t)(K->oxK / K->Bk);
        p->Bk = (uint32_t)K->Bk;
        p->bd = (uint32_t)pr->bd;
        for (int t = 0; t < pr->bd; t++) {
            u128 v = cc->w[t];
            for (int d = 0; d < MAXL_G; d++) { p->blkw[t][d] = (uint32_t)(v % K->B); v /= K->B; }
            if (v) { fprintf(stderr, "block weight has more than %d base-B digits\n", MAXL_G); exit(1); }
        }
        p->invPtop = (float)(1.0 / u128d(K->Ptop));
        p->invg = (float)(1.0 / (double)K->g);
        if (K->g > 1) {                   /* W div g = mulhi(W, M) >> (p-1), exact for W < 2^31 */
            if (K->N > (1ull << 31)) { fprintf(stderr, "N too large for the magic division\n"); exit(1); }
            int pl = 0;
            while ((1ull << pl) < K->g) pl++;
            u64 M = (u64)((((u128)1 << (31 + pl)) + K->g - 1) / K->g);
            if (M >> 32) { fprintf(stderr, "magic overflow\n"); exit(1); }
            p->gm = (uint32_t)M;
            p->gs = (uint32_t)(pl - 1);
            for (u64 t = 0; t < 4096; t++) {
                u64 w = (t * 2654435761ull) % K->N;
                if (((w * M) >> 32 >> (pl - 1)) != w / K->g) { fprintf(stderr, "magic division check failed\n"); exit(1); }
            }
        }
        for (u64 t = 0; t < c; t++) {
            s->h_tvals[toff + t] = K->val[t];
            s->h_tys[toff + t] = K->ys[t];
            s->h_tay_lo[toff + t] = K->ay_lo[t]; s->h_tay_hi[toff + t] = K->ay_hi[t];
            s->h_taz_lo[toff + t] = K->az_lo[t]; s->h_taz_hi[toff + t] = K->az_hi[t];
        }
        for (u32 t = 0; t <= K->nb; t++) s->h_tbst[boff + t] = (uint16_t)K->bst[t];
        toff += c;
        boff += K->nb + 1;
    }
    size_t np = cc->Bcount;
    CK(cudaMemcpyAsync(s->d_pairs, s->h_pairs, np * sizeof(PairC), cudaMemcpyHostToDevice, s->stream));
    CK(cudaMemcpyAsync(s->d_tvals, s->h_tvals, toff * 4, cudaMemcpyHostToDevice, s->stream));
    CK(cudaMemcpyAsync(s->d_tys, s->h_tys, toff * 2, cudaMemcpyHostToDevice, s->stream));
    CK(cudaMemcpyAsync(s->d_tay_lo, s->h_tay_lo, toff * 4, cudaMemcpyHostToDevice, s->stream));
    CK(cudaMemcpyAsync(s->d_tay_hi, s->h_tay_hi, toff * 2, cudaMemcpyHostToDevice, s->stream));
    CK(cudaMemcpyAsync(s->d_taz_lo, s->h_taz_lo, toff * 4, cudaMemcpyHostToDevice, s->stream));
    CK(cudaMemcpyAsync(s->d_taz_hi, s->h_taz_hi, toff * 2, cudaMemcpyHostToDevice, s->stream));
    CK(cudaMemcpyAsync(s->d_tbst, s->h_tbst, boff * 2, cudaMemcpyHostToDevice, s->stream));
}

/* BlkR -- one block of the search: its linear index j among the blocks of
   base c ((e_0 - 1) c^(bd-1) + e_1 c^(bd-2) + ...), its range [Blo, Bhi] of
   check bases, and whether it lies wholly inside [lo, hi]. */
struct BlkR { u64 j; u64 Blo, Bhi; int inside; };

/*
 * batch_blocks -- the blocks of the units [u0, u1) of base c that do_unit()
 * and do_block() would process, with their B ranges.
 * In:     pr; cc (the context of c); u0, u1; out (room for cap entries).
 * Out:    the number of blocks written to out[]; *steps = their x-steps
 *         (c per block and B, the CPU tool's "steps").
 * Errors: exits with status 1 if more than cap blocks ("block buffer
 *         overflow").
 */
static size_t batch_blocks(const prob_t *pr, const cctx_t *cc, u64 u0, u64 u1, BlkR *out, size_t cap, u64 *steps)
{
    size_t nb_ = 0;
    u64 c = cc->c;
    *steps = 0;
    for (u64 u = u0; u < u1; u++) {
        u64 pref[MAXN];
        unit_decode(pr, u, pref);
        u128 Mu = 0;
        for (int i = 0; i < pr->ud; i++) Mu += (u128)pref[i] * cc->w[i];
        if (Mu + cc->span_unit < pr->lo || Mu > pr->hi) continue;
        int nbd = pr->bd - pr->ud;
        u64 nblk = nbd ? c : 1;
        u64 jbase = 0;
        for (int i = 0; i < pr->ud; i++) jbase = jbase * c + (i == 0 ? pref[0] - 1 : pref[i]);
        for (int i = pr->ud; i < pr->bd; i++) jbase *= c;
        for (u64 d = 0; d < nblk; d++) {
            u128 Mb = Mu + (nbd ? (u128)d * cc->w[pr->ud] : 0);
            if (nbd && Mb > pr->hi) break;
            if (!cc->Bcount) continue;
            u128 Mmin = Mb, Mmax = Mb + cc->span_block;
            if (Mmax < pr->lo || Mmin > pr->hi) continue;
            int inside = Mmin >= pr->lo && Mmax <= pr->hi;
            if (Mmin < pr->lo) Mmin = pr->lo;
            if (Mmax > pr->hi) Mmax = pr->hi;
            u64 Blo = iroot(Mmin, pr->Lc) + 1, Bhi = iroot(Mmax, pr->Lc - 1);
            if (Blo < cc->Bbase) Blo = cc->Bbase;
            if (Bhi > cc->Bbase + cc->Bcount - 1) Bhi = cc->Bbase + cc->Bcount - 1;
            if (Bhi < Blo) continue;
            if (nb_ == cap) { fprintf(stderr, "block buffer overflow\n"); exit(1); }
            out[nb_].j = jbase + d;
            out[nb_].Blo = Blo;
            out[nb_].Bhi = Bhi;
            out[nb_].inside = inside;
            nb_++;
            *steps += (Bhi - Blo + 1) * c;
        }
    }
    return nb_;
}

/*
 * make_items -- cut the batch's blocks into items: for each check base B,
 * the runs of consecutive blocks (consecutive j) whose range contains B, in
 * pieces of at most TG blocks.
 * In:     cc; bl[0..nbl-1] (in order of j); it (room for cap items).
 * Out:    the number of items, or (size_t)-1 if more than cap are needed
 *         (the caller stops with "item buffer overflow").
 */
static size_t make_items(const cctx_t *cc, const BlkR *bl, size_t nbl, GItem *it, size_t cap)
{
    size_t ni = 0;
    if (!nbl) return 0;
    u64 Bmin = bl[0].Blo, Bmax = bl[0].Bhi;
    for (size_t i = 1; i < nbl; i++) {
        if (bl[i].Blo < Bmin) Bmin = bl[i].Blo;
        if (bl[i].Bhi > Bmax) Bmax = bl[i].Bhi;
    }
    for (u64 B = Bmin; B <= Bmax; B++) {
        size_t i = 0;
        while (i < nbl) {
            if (bl[i].Blo > B || bl[i].Bhi < B) { i++; continue; }
            size_t a = i;
            while (i + 1 < nbl && bl[i + 1].j == bl[i].j + 1 && bl[i + 1].Blo <= B && bl[i + 1].Bhi >= B &&
                   i + 1 - a < TG)
                i++;
            if (ni == cap) return (size_t)-1;
            it[ni].pair = (uint32_t)(B - cc->Bbase);
            it[ni].j0 = (uint32_t)bl[a].j;
            it[ni].count = (uint32_t)(i - a + 1);
            it[ni].pad = 0;
            for (size_t t = a; t <= i; t++)
                if (!bl[t].inside) it[ni].pad = 1;   /* some block reaches outside [lo, hi]: range check on */
            ni++;
            i++;
        }
    }
    return ni;
}

static u64 g_target_steps = 1ull << 26;    /* thread-steps per launch, adapted to ~0.25 s */
static double g_gpu_rate, g_gpu_time, g_max_kernel;   /* smoothed steps/s; kernel seconds: total, max */
static u64 g_nlaunch;

/*
 * slot_finish -- wait for the slot's batch and account for it: adapt
 * g_target_steps to the measured kernel rate, copy the survivors back,
 * rebuild each M = q1 P + rem, re-check both palindromes on the host and
 * call survivor(), add the items' lookups, hits and f1 and the batch's
 * steps, and move done_units to the end of the batch.
 * In:     s (busy); st (statistics); done_units; n.
 * Errors: exits with status 1 on CUDA errors, a survivor buffer overflow, or
 *         a GPU survivor that fails the host check (a kernel bug).
 */
static void slot_finish(Slot *s, wstat_t *st, u64 *done_units, int n)
{
    CK(cudaStreamSynchronize(s->stream));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, s->ev0, s->ev1));
    double gt = ms * 1e-3;
    g_gpu_time += gt;
    if (gt > g_max_kernel) g_max_kernel = gt;
    g_nlaunch++;
    if (gt > 0.002 && s->steps > 0) {
        double r = s->steps / gt;
        g_gpu_rate = g_gpu_rate > 0 ? 0.7 * g_gpu_rate + 0.3 * r : r;
        double tgt = (r < g_gpu_rate ? r : g_gpu_rate) * 0.25;
        if (tgt < 1e6) tgt = 1e6;
        if (tgt > 2e11) tgt = 2e11;
        g_target_steps = (u64)tgt;
    }
    unsigned no = *s->h_nout;
    if (no > OUTCAP) { fprintf(stderr, "survivor buffer overflow\n"); exit(1); }
    if (no) {
        CK(cudaMemcpyAsync(s->h_outs, s->d_outs, (size_t)no * sizeof(Surv), cudaMemcpyDeviceToHost, s->stream));
        CK(cudaStreamSynchronize(s->stream));
    }
    for (unsigned i = 0; i < no; i++) {
        const Surv *sv = &s->h_outs[i];
        const bconst_t *K = &s->cc.tab[sv->B - s->cc.Bbase];
        u128 rf = (u128)sv->r0 | ((u128)sv->r1 << 32) | ((u128)sv->r2 << 64);
        u128 M = (u128)sv->q * K->P + rf;
        if (!is_pal_base(M, sv->B, n) || !is_pal_base(M, sv->c, n - 1)) {
            char buf[48];
            fprintf(stderr, "GPU survivor fails the host check: M=%s B=%u c=%u\n", u128s(M, buf), sv->B, sv->c);
            exit(1);
        }
        survivor(st, M, sv->B, sv->c);
    }
    for (size_t i = 0; i < s->nitems; i++) {
        st->lookups += s->h_tgstat[3 * i];
        st->hits += s->h_tgstat[3 * i + 1];
        st->f1 += s->h_tgstat[3 * i + 2];
    }
    st->steps += s->steps;
    if (s->u_end > *done_units) *done_units = s->u_end;
    s->busy = 0;
}

/*
 * gpu_search -- the "search" command (and the selftest's searches); the
 * counterpart of run_search() in ../a171775_2d.c.
 *
 * In:     n (8..14); lo, hi; state (checkpoint file or NULL); interval
 *         (seconds between progress lines); quiet; result (may be NULL).
 * Out:    SOLUTION lines, progress, summary and checkpoints as run_search();
 *         returns 0 when complete (also without units), 3 if interrupted,
 *         1 if n is not supported.
 * How:    loop: finish the slot's previous batch; build a new batch of whole
 *         units of one base c (at most g_target_steps steps and TG
 *         MAXITEMS/2 threads), upload its items, launch scan2, queue the
 *         copies back; switch slots.  When nothing is left to issue (or on
 *         g_stop), finish both slots, oldest first.  Checkpoints use the
 *         first unit of the oldest batch in flight, so they are exact.
 * Errors: exits with status 1 on CUDA errors and the limits of slot_set_c(),
 *         batch_blocks() and make_items(); load_state() refuses a state file
 *         with a frontier beyond -E.
 */
static int gpu_search(int n, u128 lo, u128 hi, const char *state, double interval, int quiet, wstat_t *result)
{
    char a[48], b[48];
    prob_t pr;
    prob_init(&pr, n, lo, hi);
    if (g_end_unit < pr.nunits) pr.nunits = g_end_unit;
    if (pr.bd < 1 || pr.bd > 3) { fprintf(stderr, "n = %d not supported by the GPU kernel\n", n); return 1; }
    wstat_t st, base;
    memset(&st, 0, sizeof st);
    memset(&base, 0, sizeof base);
    st.pr = &pr;
    u64 frontier = 0;
    if (state && load_state(state, &pr, &frontier, &base) && !quiet)
        fprintf(stderr, "resuming at unit %llu of %llu\n", (unsigned long long)frontier, (unsigned long long)pr.nunits);
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    if (!quiet)
        fprintf(stderr, "cuda search n=%d [%s, %s] on %s (%d SMs): bases c in [%llu, %llu], %llu units\n", n,
                u128s(lo, a), u128s(hi, b), prop.name, prop.multiProcessorCount, (unsigned long long)pr.cmin,
                (unsigned long long)pr.cmax, (unsigned long long)pr.nunits);
    Params prm;
    memset(&prm, 0, sizeof prm);
    for (int i = 0; i < 4; i++) { prm.lo[i] = (uint32_t)(pr.lo >> (32 * i)); prm.hi[i] = (uint32_t)(pr.hi >> (32 * i)); }
    prm.outcap = OUTCAP;
    prm.n = (uint32_t)n;

    static Slot sl[2];                     /* allocated once, reused by the selftest's searches */
    static int allocated = 0;
    if (!allocated) { slot_alloc(&sl[0]); slot_alloc(&sl[1]); allocated = 1; }
    for (int i = 0; i < 2; i++) { sl[i].c = 0; sl[i].busy = 0; }   /* c = 0: tables must be rebuilt */
    size_t blkcap = 1 << 22;
    BlkR *blk = (BlkR *)malloc(blkcap * sizeof(BlkR));
    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);
    double t0 = now_sec(), tlast = t0, tsave = t0;
    u64 unext = frontier, done_units = frontier;
    int cur = 0;
    for (;;) {
        Slot *s = &sl[cur];
        if (s->busy) slot_finish(s, &st, &done_units, n);   /* the older batch in flight */
        int issued = 0;
        if (unext < pr.nunits && !g_stop) {
            /* a batch: whole units of one base c, until the step or thread budget is used */
            u64 pref[MAXN];
            u64 c = unit_decode(&pr, unext, pref);
            slot_set_c(s, &pr, c);
            u64 ub = unext, ue = unext, steps = 0, threads = 0;
            size_t nbl = 0;
            while (ue < pr.nunits) {
                if (unit_decode(&pr, ue, pref) != c) break;
                u64 st_, th = 0;
                size_t add = batch_blocks(&pr, &s->cc, ue, ue + 1, blk + nbl, blkcap - nbl, &st_);
                for (size_t i = nbl; i < nbl + add; i++) th += blk[i].Bhi - blk[i].Blo + 1;
                if (ue > ub && (steps + st_ > g_target_steps || threads + th > (u64)TG * (MAXITEMS / 2))) break;
                nbl += add;
                threads += th;
                steps += st_;
                ue++;
            }
            size_t ni = make_items(&s->cc, blk, nbl, s->h_items, MAXITEMS);
            if (ni == (size_t)-1) { fprintf(stderr, "item buffer overflow\n"); exit(1); }
            s->nitems = ni;
            s->u_begin = ub;
            s->u_end = ue;
            s->steps = steps;
            /* on the slot's stream: reset the survivor count, upload, launch,
               copy the counts back (the events time the kernel) */
            *s->h_nout = 0;
            CK(cudaMemcpyAsync(s->d_nout, s->h_nout, sizeof(unsigned), cudaMemcpyHostToDevice, s->stream));
            CK(cudaEventRecord(s->ev0, s->stream));
            if (ni) {
                CK(cudaMemcpyAsync(s->d_items, s->h_items, ni * sizeof(GItem), cudaMemcpyHostToDevice, s->stream));
                scan2<<<(unsigned)ni, TG, 0, s->stream>>>(s->d_items, s->d_pairs, s->d_tvals, s->d_tys, s->d_tbst,
                                                          s->d_tay_lo, s->d_tay_hi, s->d_taz_lo, s->d_taz_hi, prm,
                                                          s->d_outs, s->d_nout, s->d_tgstat);
                CK(cudaGetLastError());
                CK(cudaMemcpyAsync(s->h_tgstat, s->d_tgstat, 3 * ni * sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                                   s->stream));
            }
            CK(cudaEventRecord(s->ev1, s->stream));
            CK(cudaMemcpyAsync(s->h_nout, s->d_nout, sizeof(unsigned), cudaMemcpyDeviceToHost, s->stream));
            s->busy = 1;
            unext = ue;
            issued = 1;
        }
        cur ^= 1;
        int drained = 0;
        if (!issued) {                     /* nothing left to issue: drain, oldest first */
            if (sl[cur].busy) slot_finish(&sl[cur], &st, &done_units, n);
            if (sl[cur ^ 1].busy) slot_finish(&sl[cur ^ 1], &st, &done_units, n);
            drained = 1;
        }
        double t = now_sec();
        if (!quiet && (t - tlast >= interval || drained)) {
            u64 pr_[MAXN];
            u64 cc_ = unit_decode(&pr, done_units < pr.nunits ? done_units : pr.nunits - 1, pr_);
            fprintf(stderr, "[%7.0fs] frontier %llu/%llu (c=%llu)  %.3e steps/s %.3e lookups/s  st2 %llu canon %llu sol %llu\n",
                    t - t0, (unsigned long long)done_units, (unsigned long long)pr.nunits, (unsigned long long)cc_,
                    st.steps / (t - t0), st.lookups / (t - t0), (unsigned long long)(base.st2 + st.st2),
                    (unsigned long long)(base.canon + st.canon), (unsigned long long)(base.sol + st.sol));
            tlast = t;
        }
        if (state && (t - tsave >= 60 || drained)) {
            wstat_t all = st;                  /* the finished batches: exactly the units below fr */
            add_stats(&all, &base);
            u64 fr = done_units;               /* = the first unit of the oldest batch in flight */
            for (int i = 0; i < 2; i++) if (sl[i].busy && sl[i].u_begin < fr) fr = sl[i].u_begin;
            save_state(state, &pr, fr, &all);
            tsave = t;
        }
        if (drained) break;
    }
    wstat_t all = st;
    add_stats(&all, &base);
    int complete = done_units >= pr.nunits;
    if (!quiet) {
        print_summary(stdout, &pr, &all, complete, now_sec() - t0);
        fprintf(stderr, "gpu: %llu launches, kernel time %.1f s (max %.3f s per launch)\n", (unsigned long long)g_nlaunch,
                g_gpu_time, g_max_kernel);
    }
    if (result) *result = all;
    free(blk);
    free(pr.uoff);
    return complete ? 0 : 3;
}

/*
 * cuda_selftest -- complete searches over [1, a(n)] for n = 8, 9, 10 (the
 * last takes a few minutes on the GB10): each must find exactly one solution,
 * a(n), with all seven statistics of the CPU tool.
 * Out: one line per n; returns 0 if all agree, 1 otherwise.  Errors: as
 * gpu_search(); tmpfile() is not checked; -E must not be given.
 */
static int cuda_selftest(void)
{
    /* statistics of a171775_2d (CPU) for the same searches */
    struct { int n; const char *hi; u64 steps, lookups, hits, f1, st2, canon, csum; } ref[] = {
        {8, "2^42", 2795327ULL, 3386590ULL, 61277350ULL, 1488018ULL, 4725, 4724, 0x13747afea2768e7aULL},
        {9, "2^56", 42728678ULL, 50351034ULL, 1504469668ULL, 19481395ULL, 19140, 19140, 0x380d9946bb59f02eULL},
        {10, "2^72", 158029794515ULL, 178478637779ULL, 46118188570ULL, 343754929ULL, 135382, 135382, 0xe9eed6127c51a50fULL},
    };
    int bad = 0;
    for (int i = 0; i < 3; i++) {
        FILE *mem = tmpfile(), *save = sol_out;
        sol_out = mem;
        wstat_t t;
        double t0 = now_sec();
        gpu_search(ref[i].n, 1, parse_or_die(ref[i].hi), NULL, 1e9, 1, &t);
        sol_out = save;
        rewind(mem);
        char line[1024];
        int nsol = 0;
        u128 best = 0;
        while (fgets(line, sizeof line, mem)) {
            char *p = strstr(line, "M=");
            if (!p) continue;
            char num[64];
            sscanf(p + 2, "%63[0-9]", num);
            u128 M = parse_or_die(num);
            if (!best || M < best) best = M;
            nsol++;
        }
        fclose(mem);
        int ok = nsol == 1 && best == parse_or_die(ref[i].hi) && t.steps == ref[i].steps && t.lookups == ref[i].lookups &&
                 t.hits == ref[i].hits && t.f1 == ref[i].f1 && t.st2 == ref[i].st2 && t.canon == ref[i].canon &&
                 t.csum == ref[i].csum;
        char buf[48];
        double el = now_sec() - t0;
        printf("  n=%d [1, %s]: %.2fs (%.3e lookups/s), steps %llu, lookups %llu, hits %llu, f1 %llu, st2 %llu, "
               "canonical %llu, checksum %016llx, %d solution(s), smallest %s  %s\n",
               ref[i].n, ref[i].hi, el, t.lookups / el, (unsigned long long)t.steps, (unsigned long long)t.lookups,
               (unsigned long long)t.hits, (unsigned long long)t.f1, (unsigned long long)t.st2,
               (unsigned long long)t.canon, (unsigned long long)t.csum, nsol, best ? u128s(best, buf) : "none",
               ok ? "ok" : "MISMATCH");
        fflush(stdout);
        bad += !ok;
    }
    printf(bad ? "SELFTEST FAILED\n" : "selftest passed\n");
    return bad ? 1 : 0;
}

/* main -- parse -S, -i, -E and the command (search needs n = 8..14); the
   blocking-sync flag lets the host sleep while it waits for the GPU.
   Unknown commands print the usage and return 1. */
int main(int argc, char **argv)
{
    sol_out = stdout;
    const char *state = NULL;
    double interval = 10;
    char *pos[8];
    int npos = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-S") && i + 1 < argc) state = argv[++i];
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) interval = atof(argv[++i]);
        else if (!strcmp(argv[i], "-E") && i + 1 < argc) g_end_unit = strtoull(argv[++i], NULL, 10);   /* see ../a171775_2d.c */
        else if (npos < 8) pos[npos++] = argv[i];
    }
    CK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    if (npos >= 1 && !strcmp(pos[0], "selftest")) return cuda_selftest();
    if (npos == 4 && !strcmp(pos[0], "search")) {
        int n = atoi(pos[1]);
        if (n < 8 || n > 14) { fprintf(stderr, "n must be 8..14\n"); return 1; }
        return gpu_search(n, parse_or_die(pos[2]), parse_or_die(pos[3]), state, interval, 0, NULL);
    }
    fprintf(stderr, "usage: a171775_2d_cuda selftest | search n LO HI [-S state] [-i sec] [-E end_unit]\n");
    return 1;
}
