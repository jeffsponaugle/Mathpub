/*
 * a171775_2d.metal -- Metal kernel for the swapped 2D A171775 search (see ../a171775_2d.c,
 * whose block_B() this kernel runs, and a171775_metal.m, its host).  cuda/a171775_2d_cuda.cu
 * contains a CUDA port of the same kernel.
 *
 * One threadgroup = one base pair (c, B) and up to 256 consecutive blocks of the search; one
 * thread = one block.  The pair's lookup table (keys (v mod g) N' + v div g, sorted,
 * bucketed) and digit-k tables are copied into threadgroup memory (c <= 1024).  All wide
 * values are little-endian 32-bit limbs: the remainder rem = M1 mod P and the weights are
 * 96-bit, M itself (for the range test) 128-bit.  A thread builds its start state from the
 * base-B digits of Mblock (the sum of precomputed digit vectors of the block weights, so no
 * wide division is needed) and walks x = 0..c-1 exactly like block_B() in the C tool.
 *
 * Each lookup match is checked at once by the thread that found it (process_rec(): region,
 * range, next digit pair, full palindrome test), with the same steps as block_B(), so the
 * statistics are identical to the CPU tool's.  Only full n-digit palindromes are written out.
 * (An earlier version queued the matches per SIMD group to reduce divergence; the queue
 * overflowed too often and was removed.  Rec, QS and the fourth tgstat word are left from
 * it.)
 *
 * Compile-time switches: ABLATE (set by the host from A171775_ABLATE) removes parts of the
 * work for profiling and gives wrong statistics: 12 = process_rec() returns at once, >= 21 =
 * matches are not processed at all, >= 23 = no lookups either.
 */
#include <metal_stdlib>
using namespace metal;

#define MAXK 3                       /* largest k (digit pairs imposed by the lookup) */
#define MAXL 16                      /* digit buffers: n + 2 <= MAXL */
#define TG 256                       /* threads per threadgroup (the host uses the same) */
#define NSG (TG / 32)                /* SIMD groups per threadgroup */
#ifndef QS
#define QS 128                       /* unused: the queue size of the removed match queue */
#endif

struct PairC {                       /* must match the host definition */
    uint B, k, N, g, Np, u, lim, R;
    uint Wx, sh, nb, toff, boff, Lc, Lk, c;
    uint Wc[4];
    uint P[4], Rc[4], Ptop[4];       /* u96, limb 3 unused */
    uint wx[4], wy[4], wz[4], NpWz[4];
    uint ox_lo, ox_hi, Bk, bd;       /* w_x mod B^(k+1) as (mod B^k, div B^k); Bk = B^k */
    uint blkw[3][MAXL];              /* base-B digits of the block weights w_0..w_{bd-1} */
    float invPtop, invg;
    uint gm, gs;                     /* W div g = mulhi(W, gm) >> gs for W < 2^31 */
};

/* Item -- one threadgroup's work: pair index, first block index j0, number of blocks,
   pad bit 0 = range test needed.  Surv -- a survivor: q1, rem (r0..r2), B, c, x and the
   block index j; the host rebuilds M = q1 P + rem.  Params -- [lo, hi] as 32-bit limbs,
   the survivor buffer size and n. */
struct Item { uint pair, j0, count, pad; };
struct Surv { uint q, r0, r1, r2, B, c, x, j; };
struct Params { uint lo[4], hi[4]; uint outcap, n, pad0, pad1; };
/* a lookup match: state of the pushing thread at its x, and (y, z) */
struct Rec { uint r0, r1, r2, q1, mklo, info, yz, xt; };   /* info = mk_hi | reg << 16, yz = y | z << 16, xt = x | tid << 16 */

/* 96-bit arithmetic on three 32-bit limbs (a0 least significant), 32-bit operations only:
   ge96 (a >= b), add96 (a += b, no overflow by construction), sub96 (a -= b, a >= b),
   mul96 and div96 below. */
static inline bool ge96(uint a0, uint a1, uint a2, uint b0, uint b1, uint b2)
{
    if (a2 != b2) return a2 > b2;
    if (a1 != b1) return a1 > b1;
    return a0 >= b0;
}

static inline void add96(thread uint &a0, thread uint &a1, thread uint &a2, uint b0, uint b1, uint b2)
{
    uint s0 = a0 + b0, c0 = s0 < b0;
    uint s1 = a1 + b1, c1 = s1 < b1;
    uint t1 = s1 + c0;
    c1 += t1 < c0;
    a0 = s0; a1 = t1; a2 = a2 + b2 + c1;
}

static inline void sub96(thread uint &a0, thread uint &a1, thread uint &a2, uint b0, uint b1, uint b2)
{
    uint br0 = a0 < b0, d0 = a0 - b0;
    uint br1 = a1 < b1, d1 = a1 - b1;
    br1 += d1 < br0;
    a0 = d0; a1 = d1 - br0; a2 = a2 - b2 - br1;
}

/* (a0,a1,a2) * m, m < 2^21, result u96 (no overflow by construction); 32-bit mul/mulhi only */
static inline void mul96(thread uint &a0, thread uint &a1, thread uint &a2, uint m)
{
    uint l0 = a0 * m, h0 = mulhi(a0, m);
    uint l1 = a1 * m, h1 = mulhi(a1, m);
    uint n1 = l1 + h0, c1 = n1 < h0;
    a0 = l0; a1 = n1; a2 = a2 * m + h1 + c1;
}

/* divide u96 by d < 2^16 in place, return remainder (16-bit chunks, 32-bit divisions only) */
static inline uint div96(thread uint &a0, thread uint &a1, thread uint &a2, uint d)
{
    uint r = 0, q, h;
    h = (r << 16) | (a2 >> 16); q = h / d; r = h - q * d; uint q2 = q << 16;
    h = (r << 16) | (a2 & 0xffff); q = h / d; r = h - q * d; q2 |= q;
    h = (r << 16) | (a1 >> 16); q = h / d; r = h - q * d; uint q1 = q << 16;
    h = (r << 16) | (a1 & 0xffff); q = h / d; r = h - q * d; q1 |= q;
    h = (r << 16) | (a0 >> 16); q = h / d; r = h - q * d; uint q0 = q << 16;
    h = (r << 16) | (a0 & 0xffff); q = h / d; r = h - q * d; q0 |= q;
    a0 = q0; a1 = q1; a2 = q2;
    return r;
}

struct PC {                          /* per-threadgroup copy of what process_rec needs */
    uint B, k, Lc, Lk, Bk, toff, c, pad;
    uint P[4], Ptop[4], wy[4], wz[4];
    float invPtop;
};

/*
 * process_rec -- check one lookup match: the same steps as the match code of block_B() in
 * ../a171775_2d.c (candidate() in a171775.c).
 *
 * In:     rc (the thread's rem, q1, M1 mod B^(k+1) and region at this x, plus y, z, x and
 *         the thread index); K (the pair, in threadgroup memory); prm; the digit-k tables;
 *         the survivor buffer and counter; j0; need_range; the thread's hit and f1 counters.
 * Out:    counts the hit if (y, z) lies in the region rc was made for, and f1 if it passes
 *         the digit pair (k, n-1-k); a full n-digit palindrome is appended to outs[]
 *         (dropped past prm.outcap; the host then stops with "survivor buffer overflow").
 * Errors: none.  Digit n-1-k comes from a float32 quotient (relative error about 3e-7,
 *         below 3.1e-4 for digits < 1024, inside the 0.00049 margin) and is corrected
 *         exactly near a digit boundary.
 */
static inline void process_rec(Rec rc, threadgroup const PC &K, constant Params &prm, threadgroup const uint *tay_lo,
                               threadgroup const ushort *tay_hi, threadgroup const uint *taz_lo, threadgroup const ushort *taz_hi,
                               device Surv *outs, device atomic_uint *nout, uint j0, bool need_range,
                               thread ulong &nhits, thread ulong &nf1)
{
    const uint B = K.B, k = K.k, Lc = K.Lc, Lk = K.Lk, Bk = K.Bk;
    const uint P0 = K.P[0], P1 = K.P[1], P2 = K.P[2];
    const uint y = rc.yz & 0xffff, z = rc.yz >> 16, reg = (rc.info >> 16) & 1, mk_hi = rc.info & 0xffff;
    const uint q1 = rc.q1;
#if defined(ABLATE) && ABLATE == 12
    nhits += y & 1;
    return;
#endif
    /* rf = rem + y w_y + z w_z, and the q-region it falls in */
    uint d0 = K.wy[0], d1 = K.wy[1], d2 = K.wy[2];
    mul96(d0, d1, d2, y);
    uint f0 = K.wz[0], f1 = K.wz[1], f2 = K.wz[2];
    mul96(f0, f1, f2, z);
    add96(d0, d1, d2, f0, f1, f2);
    uint s0 = rc.r0, s1 = rc.r1, s2 = rc.r2;
    add96(s0, s1, s2, d0, d1, d2);
    bool ge = ge96(s0, s1, s2, P0, P1, P2);
    if ((reg != 0) != ge) return;
    nhits++;
    if (need_range) {                /* M = q1 P + rf as u128 against [lo, hi] */
        ulong t = (ulong)P0 * q1 + s0;
        uint m0 = (uint)t;
        t = (ulong)P1 * q1 + s1 + (t >> 32);
        uint m1 = (uint)t;
        t = (ulong)P2 * q1 + s2 + (t >> 32);
        uint m2 = (uint)t, m3 = (uint)(t >> 32);
        bool below = m3 != prm.lo[3] ? m3 < prm.lo[3] : m2 != prm.lo[2] ? m2 < prm.lo[2]
                   : m1 != prm.lo[1] ? m1 < prm.lo[1] : m0 < prm.lo[0];
        bool above = m3 != prm.hi[3] ? m3 > prm.hi[3] : m2 != prm.hi[2] ? m2 > prm.hi[2]
                   : m1 != prm.hi[1] ? m1 > prm.hi[1] : m0 > prm.hi[0];
        if (below || above) return;
    }
    /* digit k of M from (M1 + y w_y + z w_z) mod B^(k+1), 32-bit halves */
    uint lo = rc.mklo + tay_lo[y] + taz_lo[z];
    uint hi = mk_hi + (uint)tay_hi[y] + (uint)taz_hi[z];
    if (lo >= Bk) { lo -= Bk; hi++; }
    if (lo >= Bk) { lo -= Bk; hi++; }
    while (hi >= B) hi -= B;
    const uint dlo = hi;
    /* digit Lc-1-k of M = top digit of r (r = rf, or rf - P in the q1+1 region) */
    uint a0 = s0, a1 = s1, a2 = s2;
    if (reg) sub96(a0, a1, a2, P0, P1, P2);
    float qf = ((float)a2 * 18446744073709551616.0f + (float)a1 * 4294967296.0f + (float)a0) * K.invPtop;
    int dh = (int)qf;
    float fr = qf - (float)dh;
    if (fr < 0.00049f || fr > 0.99951f) {     /* near a digit boundary: exact correction */
        if (dh < 0) dh = 0;
        for (;;) {
            uint p0 = K.Ptop[0], p1 = K.Ptop[1], p2 = K.Ptop[2];
            mul96(p0, p1, p2, (uint)dh);
            if (!ge96(a0, a1, a2, p0, p1, p2)) { dh--; continue; }
            add96(p0, p1, p2, K.Ptop[0], K.Ptop[1], K.Ptop[2]);
            if (ge96(a0, a1, a2, p0, p1, p2)) { dh++; continue; }
            break;
        }
    }
    if (dlo != (uint)dh) return;
    nf1++;
    /* full check: digits of r (Lk of them) and of q = q1 + reg (k of them) */
    uint dig[MAXL + 2];
    for (uint t = 0; t < Lk; t++) dig[t] = div96(a0, a1, a2, B);
    uint qv = q1 + reg;
    for (uint t = 0; t < k; t++) { uint qd = qv / B; dig[Lk + t] = qv - qd * B; qv = qd; }
    bool pal = dig[Lc - 1] != 0 && qv == 0 && (a0 | a1 | a2) == 0;
    for (uint t = 0; pal && t < Lc / 2; t++) pal = dig[t] == dig[Lc - 1 - t];
    if (!pal) return;
    uint idx = atomic_fetch_add_explicit(nout, 1u, memory_order_relaxed);
    if (idx < prm.outcap) {
        Surv sv;
        sv.q = q1; sv.r0 = s0; sv.r1 = s1; sv.r2 = s2;
        sv.B = B; sv.c = K.c; sv.x = rc.xt & 0xffff; sv.j = j0 + (rc.xt >> 16);
        outs[idx] = sv;
    }
}

/*
 * scan2 -- one threadgroup = one item: thread tid runs block_B() for block j0 + tid (if
 * tid < count) of the item's pair (c, B).
 *
 * In:     the item list, the slot's pair constants and packed tables (key table tvals,
 *         tys, tbst; digit-k tables tay_lo, tay_hi, taz_lo, taz_hi; at the pair's offsets
 *         toff and boff), prm.
 * Out:    survivors in outs[] (counter nout); tgstat[4 tg] = the item's lookups, hits, f1
 *         and 0.
 * Errors: none; a thread whose Mblock already has more than n base-B digits stays idle.
 */
kernel void scan2(device const Item *items [[buffer(0)]],
                  device const PairC *pairs [[buffer(1)]],
                  device const uint *tvals [[buffer(2)]],
                  device const ushort *tys [[buffer(3)]],
                  device const ushort *tbst [[buffer(4)]],
                  constant Params &prm [[buffer(5)]],
                  device Surv *outs [[buffer(6)]],
                  device atomic_uint *nout [[buffer(7)]],
                  device ulong *tgstat [[buffer(8)]],
                  device const uint *tay_lo [[buffer(9)]],
                  device const ushort *tay_hi [[buffer(10)]],
                  device const uint *taz_lo [[buffer(11)]],
                  device const ushort *taz_hi [[buffer(12)]],
                  uint tg [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]],
                  uint tgsz [[threads_per_threadgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]])
{
    /* the pair's tables, copied from device memory once per threadgroup (c <= 1024) */
    threadgroup uint sval[1024];
    threadgroup ushort sys[1024];
    threadgroup ushort sbst[1032];
    threadgroup uint say_lo[1024], saz_lo[1024];
    threadgroup ushort say_hi[1024], saz_hi[1024];
    threadgroup ulong red[4 * NSG];
    threadgroup PC pc;

    const Item it = items[tg];
    device const PairC &K = pairs[it.pair];
    const uint c = K.c, B = K.B;
    if (tid == 0) {
        pc.B = B; pc.k = K.k; pc.Lc = K.Lc; pc.Lk = K.Lk; pc.Bk = K.Bk; pc.toff = K.toff; pc.c = c;
        for (int i = 0; i < 4; i++) { pc.P[i] = K.P[i]; pc.Ptop[i] = K.Ptop[i]; pc.wy[i] = K.wy[i]; pc.wz[i] = K.wz[i]; }
        pc.invPtop = K.invPtop;
    }
    for (uint i = tid; i < c; i += tgsz) {
        sval[i] = tvals[K.toff + i]; sys[i] = tys[K.toff + i];
        say_lo[i] = tay_lo[K.toff + i]; say_hi[i] = tay_hi[K.toff + i];
        saz_lo[i] = taz_lo[K.toff + i]; saz_hi[i] = taz_hi[K.toff + i];
    }
    for (uint i = tid; i <= K.nb; i += tgsz) sbst[i] = tbst[K.boff + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint k = K.k, N = K.N, g = K.g, Np = K.Np, sh = K.sh, Lc = K.Lc, Lk = K.Lk;
    const uint Wx = K.Wx, bd = K.bd, Bk = K.Bk, ox_lo = K.ox_lo, ox_hi = K.ox_hi, gm = K.gm, gs = K.gs;
    const bool need_range = (it.pad & 1u) != 0;
    const uint P0 = K.P[0], P1 = K.P[1], P2 = K.P[2];
    const uint Rc0 = K.Rc[0], Rc1 = K.Rc[1], Rc2 = K.Rc[2];
    const uint wx0 = K.wx[0], wx1 = K.wx[1], wx2 = K.wx[2];

    ulong nlook = 0, nhits = 0, nf1 = 0, novf = 0;   /* novf: unused (removed queue) */
    bool active = tid < it.count;
    uint E[MAXK] = {0, 0, 0};
    uint q1 = 0, W = 0, r0 = 0, r1 = 0, r2 = 0, mk_lo = 0, mk_hi = 0, wnext = 0;
    bool last = false;
    /* RECOMPUTE(): as in block_B(), wnext and last from the digits E[] of q1 */
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
        /* block digits (e_0 >= 1) and base-B digits of Mblock */
        uint j = it.j0 + tid, e[3] = {0, 0, 0};
        {
            uint t = j;
            for (int i = (int)bd - 1; i >= 1; i--) { e[i] = t % c; t /= c; }
            e[0] = 1 + t;
        }
        uint D[MAXL + 2];
        uint carry = 0;
        for (uint i = 0; i < Lc + 2; i++) {
            uint s = carry;
            for (uint t = 0; t < bd; t++) s += e[t] * K.blkw[t][i];
            carry = s / B;
            D[i] = s - carry * B;
        }
        if (carry == 0 && D[Lc] == 0 && D[Lc + 1] == 0) {            /* Mblock < B^Lc */
            for (uint i = 0; i < k; i++) E[i] = D[Lc - 1 - i];      /* E[0] = most significant digit of q */
            uint T = 0, MN = 0;
            for (uint i = 0; i < k; i++) q1 = q1 * B + E[i];
            for (int i = (int)k - 1; i >= 0; i--) { T = T * B + E[i]; MN = MN * B + D[i]; }
            uint tmn = T >= MN ? T - MN : T + N - MN;
            W = (uint)(((ulong)tmn * K.u) % N);
            for (int i = (int)Lk - 1; i >= 0; i--) {
                mul96(r0, r1, r2, B);
                add96(r0, r1, r2, D[i], 0, 0);
            }
            mk_lo = MN;                  /* M1 mod B^(k+1) = mk_hi B^k + mk_lo */
            mk_hi = D[k];
            RECOMPUTE();
        } else active = false;
    }

    /* PUSH(z): check the match (y, z) of the current lookup at once (process_rec) */
#if defined(ABLATE) && ABLATE >= 21
#define PUSH(ZZ) { nhits += (ZZ) & 1; }
#else
#define PUSH(ZZ)                                                                             \
    {                                                                                        \
        Rec rc_;                                                                             \
        rc_.r0 = r0; rc_.r1 = r1; rc_.r2 = r2; rc_.q1 = q1; rc_.mklo = mk_lo;                \
        rc_.info = mk_hi | ((uint)reg << 16);                                                \
        rc_.yz = y | ((ZZ) << 16);                                                           \
        rc_.xt = x | (tid << 16);                                                            \
        process_rec(rc_, pc, prm, say_lo, say_hi, saz_lo, saz_hi, outs, nout, it.j0, need_range, nhits, nf1); \
    }
#endif

    /* the x loop of block_B(); inactive threads keep looping (idle) so that every thread
       reaches the reduction below */
    for (uint x = 0; x < c; x++) {
        if (active) {
#if defined(ABLATE) && ABLATE >= 23
            nlook += W & 1;
            if (false)
#endif
            for (int reg = 0; reg < 2; reg++) {
                uint Wt;
                if (reg == 0) Wt = W;
                else {
                    if (last || !ge96(r0, r1, r2, Rc0, Rc1, Rc2)) break;
                    Wt = W + wnext;
                    if (Wt >= N) Wt -= N;
                }
                nlook++;
                /* keys are (v mod g) N' + v div g: the matches form a window of width
                   min(c, N') inside residue class r = Wt mod g, ending at m = Wt div g */
                uint rr = 0, mW = Wt;
                if (g > 1) { mW = mulhi(Wt, gm) >> gs; rr = Wt - mW * g; }
                const uint base = rr * Np;
                uint m_lo, m_hi = mW;
                bool wrap;
                if (c >= Np) { m_lo = 0; m_hi = Np - 1; wrap = false; }
                else {
                    m_lo = mW + 1 >= c ? mW + 1 - c : mW + 1 + Np - c;
                    wrap = m_lo > m_hi;
                }
                for (int part = 0; part <= (int)wrap; part++) {
                    uint a = base + (part == 0 ? m_lo : 0), b = base + ((wrap && part == 0) ? Np - 1 : m_hi);
                    uint i0 = sbst[a >> sh], i1 = sbst[(b >> sh) + 1];
                    for (uint i = i0; i < i1; i++) {
                        uint v = sval[i];
                        if (v < a) continue;
                        if (v > b) break;
                        uint m = v - base;
                        uint y = sys[i];
                        uint z0 = mW >= m ? mW - m : mW + Np - m;
                        if (Np >= c) {
                            PUSH(z0);
                        } else {             /* several z: walk them here with the region test */
                            uint d0 = K.wy[0], d1 = K.wy[1], d2 = K.wy[2];
                            mul96(d0, d1, d2, y);
                            uint f0 = K.wz[0], f1 = K.wz[1], f2 = K.wz[2];
                            mul96(f0, f1, f2, z0);
                            add96(d0, d1, d2, f0, f1, f2);
                            uint s0 = r0, s1 = r1, s2 = r2;
                            add96(s0, s1, s2, d0, d1, d2);
                            for (uint z = z0; z < c; z += Np) {
                                bool ge = ge96(s0, s1, s2, P0, P1, P2);
                                if (reg == 0) {
                                    if (ge) break;
                                    PUSH(z);
                                } else if (ge) PUSH(z);
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
                if (last) active = false;            /* q1 reaches B^k: M >= B^n from here on */
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
    }
#undef RECOMPUTE
#undef PUSH
    /* per-item counts: SIMD-group sums (32-bit: at most 2c lookups per thread), then thread 0
       adds the groups */
    uint snl = simd_sum((uint)nlook), snh = simd_sum((uint)nhits), snf = simd_sum((uint)nf1);
    if (lane == 0) { red[4 * sg] = snl; red[4 * sg + 1] = snh; red[4 * sg + 2] = snf; red[4 * sg + 3] = 0; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        ulong a = 0, b = 0, cc = 0, o = 0;
        for (uint i = 0; i < (tgsz + 31) / 32; i++) { a += red[4 * i]; b += red[4 * i + 1]; cc += red[4 * i + 2]; }
        tgstat[4 * tg] = a;
        tgstat[4 * tg + 1] = b;
        tgstat[4 * tg + 2] = cc;
        tgstat[4 * tg + 3] = o;
    }
}
