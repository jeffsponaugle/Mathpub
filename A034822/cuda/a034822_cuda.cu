/*
 * a034822_cuda.cu -- GPU exhaustive search for palindromic squares of even length L
 * (OEIS A034822 / A263618).
 *
 * ===========================================================================
 * PURPOSE
 * ===========================================================================
 *
 * Lists every palindromic square S = n^2 with exactly L decimal digits, for even
 * L = 40..70 (the lengths compiled into VARIANTS[]).  If none exists for an L,
 * then L is a term of A034822; the first open case is L = 70 (all lengths <= 69
 * were searched by De Geest / Xiao).  This is the GPU version of ../a034822.c and
 * uses the same mathematics; read the THEORY OF OPERATION in that file first.
 * This header summarizes it and describes what is specific to the GPU.
 *
 * ===========================================================================
 * THEORY OF OPERATION (summary; details in ../a034822.c)
 * ===========================================================================
 *
 * Notation: D = L/2 digits of n, split n = t*10^h + B with B < 10^h and h = L/4
 * rounded down (h = 17 for L = 70); e = L - 3h; P = the top h digits of S.
 *
 *   - B fixes the low h digits of S (B^2 mod 10^h).  For a palindrome they are
 *     the reversed top digits P, so t must lie in the window
 *         isqrt(P*10^e) <= t <= isqrt((P+1)*10^e)     (width ~5..16 for L = 70).
 *   - B is built by a digit DFS from the units digit up, carrying
 *     Q = floor(B^2/10^i) so each new digit yields one more digit of S.
 *   - Eight roots share one window: B, B + H5, -B, H5 - B (H5 = 5*10^(h-1)) and
 *     their images under sigma2: B -> B*E' mod 10^h with E' = -1 (mod 2^h),
 *     E' = 1 (mod 5^h).  Canonical B: unit digit 1..5 (unit 5: tens 0..4), top
 *     digit 0..4, and the digit v2(B)+1 (which sigma2 shifts by 5) 0..4.
 *   - Even-length palindromes are divisible by 11, so 11 | n: each root has one
 *     residue class of t mod 11, i.e. at most 2 candidates in the window.
 *
 * GPU decomposition
 * -----------------
 *   - A "base" = the low IB digits of B (unit digit + IB-1 more), numbered
 *     0 .. 5*10^(IB-1)-1 (IB = 10 for L = 70, i.e. 5*10^9 bases).  One thread
 *     handles one base: it replays the IB digits (dstep_rt), then walks the
 *     remaining K = h - IB levels (dfs_iter) down to the leaves (leafs).
 *     Bases that violate a digit restriction are skipped (invalid).
 *   - The host launches bases in batches (-b, default 2^20 per launch), collects
 *     survivors, verifies them exactly, and checkpoints after every launch.
 *
 * Window computation (per leaf, integers only -- FP64 is slow on GB10)
 * --------------------------------------------------------------------
 *   Inside one thread P = Pb + u with 0 <= u < 10^K.  The thread computes
 *   R = sqrt(Pb*10^e) once (exact 128-bit isqrt + double fraction) and stores
 *   the Taylor polynomial sqrt((Pb+u)*10^e) - Rb ~ frac + u*s1 - u^2*s2 in fixed
 *   point with 36 fractional bits (Base: FRAC, S1, S2 scaled by 2^(36+SQ)).  At
 *   each level-(h-1) parent it evaluates V0 = v(u0) and the slope D0 =
 *   v(u0+1) - v(u0); its 5 children (top digit 0..4) have P = P0 + s for their
 *   S digit s, so their windows are [V0 + s*D0, V0 + (s+1)*D0], widened by
 *   2^-10 on both sides.  The host checks that V < 2^64 can never overflow
 *   (this exact overflow was a real bug before 2026-09-30, see README).
 *
 * Candidate filter (per root, 64-bit)
 * -----------------------------------
 *   D = t^2 + floor(2*t*f/2^64) - P*10^e (mod 2^64), f ~ root*2^64/10^h, is the
 *   part of floor(S/10^(2h)) below the prefix P; a true hit has 0 <= D < 10^e,
 *   so D mod 2^64 is exact where it matters.  The main filter compares 3 digit
 *   pairs: the low digits low3 = (Bq + 2*t*root) mod 1000 (= S digits h..h+2)
 *   must equal, reversed, the top digits floor(D / 10^(e-3)).  This is done as
 *   "D in [rev3(low3)*10^(e-3) - SLACK, ... + 10^(e-3) + SLACK)" with a
 *   1000-entry shared-memory table of rev3(l)*10^(e-3).  SLACK covers the error
 *   of the f-based estimate.
 *   Hits go into a small per-thread buffer and are processed once per leaf by
 *   rare(): a 6-digit-pair 64-bit check (NB digits), then an exact 128-bit
 *   check (exact_check) of digits h..2h-1 against their mirrors.  Survivors are
 *   written to Out and verified on the host by full decimal conversion.
 *
 * Arithmetic notes
 * ----------------
 *   Several expressions wrap modulo 2^64 on purpose (D, the f of negated roots,
 *   0 - fx, ...).  For the four roots x, x+H5, 10^h-x, H5-x of a group:
 *     Bq(x+H5) = Q + x + Q25,  Bq(10^h-x) = 10^h - 2x + Q,  Bq(H5-x) = Q25 - x + Q
 *     f(x+H5) = f + 2^63,      f(10^h-x) = -f,              f(H5-x) = 2^63 - f
 *     G = floor(2*tlo*f/2^64) for f+2^63 is G + tlo; for -f it is 2*tlo - G - 1,
 *     for 2^63 - f it is tlo - G - 1 (each within 1 of exact, covered by SLACK).
 *
 * Validation builds (all slower; abort/exit 3 on the first problem)
 * -----------------------------------------------------------------
 *   -DWINDOW_CHECK  compares every fixed-point window with exact 128-bit isqrt.
 *   -DFILTER_CHECK  counts candidates that exact arithmetic accepts but the
 *                   3-digit filter rejects (must be 0).
 *   The independent cross-checks (../xcheck_subtree.py, ../base_of.py) and the
 *   selftest are described in ../README.md.
 *
 * Tuning macros: -DMINB=n (min blocks per SM in __launch_bounds__; production
 *   uses 2), -DLEAF_UNROLL=n (unroll of the 5-leaf loop; 1 is fastest).
 *
 * ===========================================================================
 * USAGE
 * ===========================================================================
 *
 *   a034822_cuda search L [-b bases_per_launch] [-S state] [-r a:b] [-B blockdim]
 *       L       even, 40..70
 *       -b      bases per kernel launch (default 2^20 ~ 6 min at L = 70);
 *               1 .. 2^30
 *       -S      state file: written after every launch; a rerun with the same
 *               L and -r range resumes from it
 *       -r a:b  base range [a, b) (b = 0 or > count means "to the end"), for
 *               splitting a run between GPUs/machines
 *       -B      threads per block, a multiple of 32 in 32..256 (default 256)
 *   a034822_cuda selftest [maxL]   even L = 40..maxL vs A263618 (maxL <= 68)
 *
 * Output: "FOUND L n n^2" lines on stdout as they are found, then
 *   "DONE ..." (range finished) or "PARTIAL ..." (stopped by SIGINT/SIGTERM,
 *   which is honoured between launches).  Progress on stderr every 10 s.
 * Exit: 0 after a search, 1 on bad parameters/CUDA errors/output overflow,
 *   2 on usage errors, 3 if a validation build detects a problem; selftest
 *   returns 1 on any mismatch.
 *
 * Build (DGX Spark / GB10, sm_121):
 *   nvcc -O3 -std=c++17 -arch=sm_121 -DMINB=2 -o a034822_cuda a034822_cuda.cu
 */
#include <cuda_runtime.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef uint64_t u64;
typedef int64_t i64;
typedef uint32_t u32;
typedef unsigned __int128 u128;
typedef __int128 i128;

/* CK -- abort with a message on any CUDA runtime error. */
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

/* A263618: number of palindromic squares with n digits, n = 1..69 (68, 69 from
 * De Geest's table).  KNOWN[70] = -1 marks the unknown L = 70.  Selftest only. */
static const int KNOWN[71] = {0,
    4,0,3,0,7,1,5,0,11,0,5,1,19,0,13,1,25,0,18,0,48,1,31,0,70,1,44,2,105,
    0,70,1,153,1,98,3,209,0,132,0,291,1,181,1,384,0,234,2,496,1,301,1,
    636,0,383,0,798,1,474,1,981,0,578,0,1199,2,701,1,1443,-1};

/* p10 -- 10^k as a compile-time constant (k <= 19). */
__host__ __device__ constexpr u64 p10(int k) { return k == 0 ? 1ull : 10ull * p10(k - 1); }

/* SLACK: tolerance of the 64-bit filter.  floor(2*t*f/2^64) differs from the
 * exact floor((2*t*root + Bq)/10^h) by < ~20 at L = 70 (f is a truncated
 * binary fraction, plus the +-1 group identities); SLACK = 64 covers that and
 * is negligible next to the digit spacing 10^(e-3) = 10^16. */
#define SLACK 64ull
#ifndef LEAF_UNROLL
#define LEAF_UNROLL 1
#endif
constexpr int kLeafUnroll = LEAF_UNROLL; /* #pragma unroll needs a constant expression */
#ifndef MINB
#define MINB 1
#endif
#define MAXOUT 4096 /* survivors per launch; at L = 70 essentially only true palindromes survive */

/* Per-launch output, zeroed by the host before each launch.
 *   nodes/filt/passB/pass3  leaves, 3-digit filter hits, 6-digit passes, exact passes
 *   nout, t[], b[]          survivors (n = t*10^h + b), at most MAXOUT stored
 *   dbg[0]                  FILTER_CHECK failures, dbg[1] WINDOW_CHECK failures */
struct Out {
    unsigned long long nodes, filt, passB, pass3;
    unsigned int nout;
    u64 dbg[8];
    u64 t[MAXOUT], b[MAXOUT];
};

/* ---------------- device helpers ---------------- */

/* d_u128_to_double -- u128 -> double (rounded), as in the CPU version. */
__device__ __forceinline__ double d_u128_to_double(u128 x)
{
    return (double)(u64)(x >> 64) * 18446744073709551616.0 + (double)(u64)x;
}

/*
 * d_udiv128_64 -- exact floor(x/d).
 *   In:  x (128-bit), d > 0.  Precondition: the quotient fits in 64 bits.
 *   Out: the quotient.  How: double estimate, exact 128-bit remainder, second
 *        estimate, +-1 fix-up.  Used once per thread (base isqrt) and in
 *        WINDOW_CHECK builds.  Garbage if the preconditions are violated.
 */
__device__ __forceinline__ u64 d_udiv128_64(u128 x, u64 d)
{
    double qd = d_u128_to_double(x) / (double)d;
    u64 q = qd >= 18446744073709549568.0 ? ~0ull : (u64)qd;
    i128 r = (i128)(x - (u128)q * d);
    double rd = r < 0 ? -d_u128_to_double((u128)(-r)) : d_u128_to_double((u128)r);
    i64 adj = (i64)(rd / (double)d);
    q += (u64)adj;
    r -= (i128)adj * (i128)d;
    while (r < 0) { q--; r += d; }
    while (r >= (i128)d) { q++; r -= d; }
    return q;
}

/*
 * d_isqrt128 -- exact floor(sqrt(x)) for x < 2^127 (double estimate, one Newton
 * step, +-1 fix-up).  x = 0 returns 0.  Larger x can overflow the (r+1)^2 test.
 */
__device__ __forceinline__ u64 d_isqrt128(u128 x)
{
    if (x == 0) return 0;
    u64 r = (u64)sqrt(d_u128_to_double(x));
    if (r == 0) r = 1;
    u128 s = ((u128)r + d_udiv128_64(x, r)) >> 1;
    r = (u64)s;
    while ((u128)r * r > x) r--;
    while ((u128)(r + 1) * (r + 1) <= x) r++;
    return r;
}

/* Constant tables, uploaded by the host (main(), run()). */
__constant__ u64 c_f64[20]; /* floor(2^64/10^k); c_f64[0] unused */
__constant__ u64 c_p10[20]; /* 10^k (for runtime-indexed powers) */

/* Compile-time constants for one (L, h). */
template <int L, int H>
struct Cfg {
    static constexpr int e = L - 3 * H;
    static constexpr u64 PE = p10(e);          /* 10^e */
    static constexpr u64 PEX = p10(e - 2);     /* 10^(e-2): spacing of the 2-digit prefix in exact_check */
    static constexpr int NB = (e >= 12) ? 6 : (e >= 9) ? 4 : 0; /* digit pairs checked in rare() */
    static constexpr u64 PN = p10(e - NB);     /* spacing for the NB-digit check */
    static constexpr u64 PH = p10(H);          /* 10^h */
    static constexpr u64 H5 = 5 * p10(H - 1);  /* 10^h / 2 */
    static constexpr u64 Q25 = 25 * p10(H - 2); /* H5^2 / 10^h */
    static constexpr u64 H5M11 = H5 % 11;
    static constexpr u64 PHM11 = PH % 11;
};

/* Window polynomial of one thread (see "Window computation" above).
 *   Pb, Rb   base P and floor(sqrt(Pb*10^e))
 *   S1, FRAC slope and fraction in 2^-36 units
 *   S2       curvature, scaled by 2^(36+SQ) (SQ chosen by the host)
 *   rb11, rb1000   Rb mod 11 and mod 1000 (for residues of t = Rb + offset) */
struct Base {
    u64 Pb, Rb, S1, FRAC, S2; /* fixed point 2^-36; S2 = s2 * 2^(36+SQ) */
    u32 rb11, rb1000;
};

/* Per-thread filter-hit buffer.  At most 2 groups x 4 roots x 2 candidates = 16
 * hits can be pushed per leaf, and the buffer is flushed after every leaf, so
 * HITBUF = 16 can never overflow. */
#define HITBUF 16
struct Hit { u64 t, Bp, Bq, D; u32 y; };
/* Per-thread statistics and the hit buffer pointer (kept separate from the
 * array so the counters stay in registers). */
struct Thr {
    unsigned long long nodes, filt, passB, pass3;
    int nh;
    Hit *hb; /* per-thread local array, touched only on filter hits */
};

/*
 * exact_check -- exact test of a candidate that passed the 64-bit filters.
 *   In:  t, Bp, Bq = floor(Bp^2/10^h), P (top h digits), y = the 2 digits that
 *        must follow P (reverse of S digits h, h+1).
 *   Out: 1 if the top h+2 digits of S are exactly P*100 + y and S digits
 *        h+2 .. min(2h-1, L-1-2h) mirror their partners in Y = floor(S/10^(2h)),
 *        else 0.  Remaining middle digits are checked by the host.
 *   Uses 128-bit divisions (slow, but reached by ~1e-7 of the candidates).
 */
template <int L, int H>
__device__ __forceinline__ int exact_check(u64 t, u64 Bp, u64 Bq, u64 P, u64 y)
{
    typedef Cfg<L, H> C;
    u128 Cc = (u128)2 * t * Bp + Bq;          /* S = t^2 10^(2h) + Cc 10^h + (Bp^2 mod 10^h) */
    u64 X = (u64)(Cc / C::PH);
    u64 c0 = (u64)(Cc - (u128)X * C::PH);     /* S digits h .. 2h-1 */
    u128 Y = (u128)t * t + X;                 /* floor(S / 10^(2h)) */
    if ((u64)(Y / C::PEX) != P * 100 + y) return 0;
    u64 Ym = (u64)(Y % 10000000000000000000ull); /* low 19 digits of Y */
    int pmax = 2 * H - 1;
    if (pmax > L - 1 - 2 * H) pmax = L - 1 - 2 * H;
    u64 pw = 100;
    for (int p = H + 2; p <= pmax; p++, pw *= 10) {
        int q = L - 1 - p - 2 * H; /* S digit p mirrors S digit L-1-p = Y digit q */
        if (q > 18) continue;
        u64 pq = c_p10[q];
        if ((c0 / pw) % 10 != (Ym / pq) % 10) return 0;
    }
    return 1;
}

/* rev_digits -- reverse the n low decimal digits of x (x < 10^n). */
__device__ __forceinline__ u32 rev_digits(u32 x, int n)
{
    u32 r = 0;
    for (int i = 0; i < n; i++) { r = 10 * r + x % 10; x /= 10; }
    return r;
}

/*
 * rare -- process one filter hit (rare path, one code copy per leaf loop).
 *   In:  t, Bp, Bq     candidate and root
 *        P             leaf's top digits
 *        D             64-bit filter value (approximate within SLACK)
 *        y             2 digits that must follow P
 *        st, out       statistics; survivors are appended to out
 *   Steps: (1) NB-digit check: low = (Bq + 2 t Bp) mod 10^NB must equal the
 *        reversed top digits floor(D / 10^(e-NB)), unless D is within SLACK of
 *        a digit boundary (then undecided -> continue); (2) exact_check;
 *        (3) store (t, Bp) for host verification.
 *   Errors: if more than MAXOUT survivors occur in one launch, the extras are
 *        counted in nout but not stored; the host then stops with an error
 *        (rerun that launch with a smaller -b).
 */
template <int L, int H>
__device__ __forceinline__ void rare(u64 t, u64 Bp, u64 Bq, u64 P, u64 D, u32 y, Thr &st, Out *out)
{
    typedef Cfg<L, H> C;
    st.filt++;
    if (C::NB) {
        constexpr u64 M = p10(C::NB);
        u64 low = (Bq % M + 2 * (Bp % M) * (t % M)) % M;
        u64 yN = D / C::PN, rem = D - yN * C::PN;
        if (!(rem < SLACK || rem > C::PN - SLACK) && yN != rev_digits((u32)low, C::NB)) return;
    }
    st.passB++;
    if (!exact_check<L, H>(t, Bp, Bq, P, y)) return;
    st.pass3++;
    unsigned int k = atomicAdd(&out->nout, 1u);
    if (k < MAXOUT) { out->t[k] = t; out->b[k] = Bp; }
}

__constant__ u64 c_ce[20];  /* 10^i * E' mod 10^h, E' = -1 (mod 2^h), 1 (mod 5^h) */

/* main filter: 3 digit pairs after P.  low3 = (Bq + 2 t Bp) mod 1000 = S digits h..h+2;
 * the top digits after P are D / 10^(e-3) and must equal rev3(low3); s_y3p[l] = rev3(l) * 10^(e-3). */
/* rev3 -- reverse a 3-digit number (l < 1000); red -- x mod m for x < 2m. */
__device__ __forceinline__ u32 rev3(u32 l) { u32 a = l / 100, r = l - 100 * a, b = r / 10, c = r - 10 * b; return 100 * c + 10 * b + a; }
__device__ __forceinline__ u32 red(u32 x, u32 m) { return x >= m ? x - m : x; } /* x < 2m */

/*
 * root -- apply the 3-digit filter to the candidates of one root.
 *   In:  Bp, Bq, f   the root, floor(Bp^2/10^h), f ~ Bp*2^64/10^h
 *        G           floor(2*tlo*f/2^64) (+-1), shared per group
 *        m           2*Bp mod 1000;  q1000 = Bq mod 1000
 *        k           t0 = tlo + k is the root's first candidate (0 <= k <= 10)
 *        E0          tlo^2 - P*10^e (mod 2^64);  A2 = 2*tlo
 *        tlo, thi    window;  tm1000 = tlo mod 1000
 *        y3p         shared table rev3(l)*10^(e-3)
 *        NE          template: number of candidates per root (1 if the
 *                    thread's windows are <= 11 wide, else 2)
 *   Out: pushes (t, Bp, Bq, D, y) into st.hb for each candidate t <= thi that
 *        passes the filter.  Branch-free except the (rare) push.
 *   How: D(t0) = E0 + (A2 + k)*k + G + floor(2k*f/2^64), with the last term
 *        approximated as (2k*(f>>32))>>32 (error <= 1).
 */
template <int L, int H, int NE>
__device__ __forceinline__ void root(u64 Bp, u64 Bq, u64 f, u64 G, u32 m, u32 q1000, u32 k, u64 E0,
                                     u64 A2, u64 tlo, u32 tm1000, u64 thi, const u64 *y3p, Thr &st, Out *out)
{
    typedef Cfg<L, H> C;
    constexpr u64 PX3 = p10(C::e - 3);
    const u64 fh = f >> 32;
    u64 t0 = tlo + k;
    /* D = t^2 + floor(2 t f/2^64) - P*10^e (mod 2^64), E0 = tlo^2 - P*10^e, A2 = 2 tlo */
    u64 D0 = E0 + (A2 + k) * k + G + ((2ull * k * fh) >> 32);
    u32 tt = tm1000 + k; /* t0 mod 1000, possibly + 1000 (harmless: only m*tt mod 1000 is used) */
    u32 l0 = (q1000 + m * tt) % 1000;
    /* unsigned trick: D0 in [y3p - SLACK, y3p + PX3 + SLACK) */
    bool h0 = (t0 <= thi) & (D0 + SLACK - y3p[l0] < PX3 + 2 * SLACK);
#ifdef FILTER_CHECK
    {
        /* exact D mod 2^64 (exact in the accept region): t^2 + floor((2 t Bp + Bq)/10^h) - P 10^e,
         * with P 10^e = tlo^2 - E0 (mod 2^64) */
        u128 Cc = (u128)2 * t0 * Bp + Bq;
        u64 X = (u64)(Cc / C::PH);
        u64 Dex = t0 * t0 + X - (tlo * tlo - E0);
        u64 w = y3p[l0];
        if (t0 <= thi && Dex >= w && Dex < w + PX3 && !h0) atomicAdd((unsigned long long *)&out->dbg[0], 1ull);
    }
#endif
    if (h0) { int i = st.nh++; st.hb[i].t = t0; st.hb[i].Bp = Bp; st.hb[i].Bq = Bq; st.hb[i].D = D0; st.hb[i].y = rev3(l0) / 10; }
    if (NE == 2) {
        /* second candidate t1 = t0 + 11 (only exists when the window is > 11 wide) */
        u64 t1 = t0 + 11;
        u32 k1 = k + 11;
        u64 D1 = E0 + (A2 + k1) * k1 + G + ((2ull * k1 * fh) >> 32);
        u32 l1 = (q1000 + m * (tt + 11)) % 1000;
        bool h1 = (t1 <= thi) & (D1 + SLACK - y3p[l1] < PX3 + 2 * SLACK);
        if (h1) { int i = st.nh++; st.hb[i].t = t1; st.hb[i].Bp = Bp; st.hb[i].Bq = Bq; st.hb[i].D = D1; st.hb[i].y = rev3(l1) / 10; }
    }
}

/*
 * group -- the four roots x, x + H5, 10^h - x, H5 - x (x < H5) sharing one window.
 *   In:  x, Qx = floor(x^2/10^h), fx ~ x*2^64/10^h, bx11 = x mod 11,
 *        bx1000 = x mod 1000, qx1000 = Qx mod 1000; E0, tlo, thi, tm11 = tlo mod 11,
 *        tm1000 = tlo mod 1000, y3p, st, out.
 *   Out: calls root() four times.  One 64x64 multiply-high (G) per group; the
 *        partners' G, Bq, f, residues follow from the identities in the header.
 *   The mod-11 offsets: root r needs t = -(-1)^h r (mod 11) (because 10 = -1),
 *        and k = (that - tlo) mod 11.
 */
template <int L, int H, int NE>
__device__ __forceinline__ void group(u64 x, u64 Qx, u64 fx, u32 bx11, u32 bx1000, u32 qx1000, u64 E0,
                                      u64 tlo, u32 tm11, u32 tm1000, u64 thi, const u64 *y3p, Thr &st, Out *out)
{
    typedef Cfg<L, H> C;
    u64 A2 = 2 * tlo;
    u64 G = __umul64hi(A2, fx);
    /* residues mod 11: root r needs t = -(-1)^h r (mod 11); k = (that - tlo) mod 11 */
    u32 rA = bx11, rB = red(bx11 + (u32)C::H5M11, 11), rC = red((u32)C::PHM11 + 11 - bx11, 11),
        rD = red((u32)C::H5M11 + 11 - bx11, 11);
    if (!(H & 1)) { rA = red(11 - rA, 11); rB = red(11 - rB, 11); rC = red(11 - rC, 11); rD = red(11 - rD, 11); }
    u32 kA = red(rA + 11 - tm11, 11), kB = red(rB + 11 - tm11, 11), kC = red(rC + 11 - tm11, 11), kD = red(rD + 11 - tm11, 11);
    u32 m = red(2 * bx1000, 1000), nm = red(1000 - m, 1000); /* 2 b mod 1000 for x, x+H5 and for the negatives */
    root<L, H, NE>(x, Qx, fx, G, m, qx1000, kA, E0, A2, tlo, tm1000, thi, y3p, st, out);
    root<L, H, NE>(x + C::H5, Qx + x + C::Q25, fx + (1ull << 63), G + tlo, m, red(qx1000 + bx1000, 1000), kB,
                   E0, A2, tlo, tm1000, thi, y3p, st, out);
    root<L, H, NE>(C::PH - x, C::PH - 2 * x + Qx, 0 - fx, 2 * tlo - G - 1, nm, red(qx1000 + 1000 - m, 1000), kC,
                   E0, A2, tlo, tm1000, thi, y3p, st, out);
    root<L, H, NE>(C::H5 - x, C::Q25 - x + Qx, (1ull << 63) - fx, tlo - G - 1, nm, red(qx1000 + 1000 - bx1000, 1000), kD,
                   E0, A2, tlo, tm1000, thi, y3p, st, out);
}

/* digit state for B (low I digits) and B2 = B E' mod 10^h:
 *   B, Q = floor(B^2/10^I), P (S digits 0..I-1 reversed into the top), f, b11
 *   B2f = B*E' mod 10^h (full), and B2, Q2, f2, b211 for the low I digits of B2. */
struct DS {
    u64 B, Q, P, f, B2f, B2, Q2, f2;
    u32 b11, b211;
};

/*
 * s2_level -- is digit i the one sigma2 shifts by 5 (so it must be 0..4)?
 *   True iff ctz(B mod 10^i) == i-1, 1 <= i <= h-2 and the unit digit is not 5.
 *   (Same rule as the CPU version; B = 0 never occurs because the unit digit
 *   is 1..5.)
 */
__device__ __forceinline__ bool s2_level(int i, u64 B, bool unit5, int H)
{
    return !unit5 && i >= 1 && i <= H - 2 && B != 0 && (__ffsll((long long)B) - 1) == i - 1;
}

/*
 * leafs -- process the 5 leaves below one level-(h-1) parent (top digit 0..4).
 *   In:  a       state with h-1 digits fixed
 *        s2      sigma2 restriction applied (test the 4 sigma2 roots too)
 *        bs, SQ  window polynomial and its S2 scale
 *        b10k, b210k   B mod 10^4 and B2 mod 10^4 (constant per thread: the
 *                low digits are fixed by the base)
 *        y3p, st, out
 *   Per leaf: S digit s and Q from v mod 10^4 (32-bit), the window from the
 *        fixed-point polynomial, E0 = tlo^2 - P*10^e, residues of tlo, then
 *        group() for B and (if s2) for the sigma2 image, then the hit buffer is
 *        flushed through rare().
 *   Sigma2 image of the top digit: E' = 1 (mod 10), so the top digit of
 *        B*E' mod 10^h is (top digit of a.B2f + d) mod 10.  If the image is
 *        >= H5 it is reduced by H5 (the group covers both).
 */
template <int L, int H, int NE>
__device__ __forceinline__ void leafs(const DS &a, bool s2, const Base &bs, int SQ, u32 b10k, u32 b210k,
                                      const u64 *y3p, Thr &st, Out *out)
{
    typedef Cfg<L, H> C;
    constexpr int I = H - 1;
    constexpr u32 TOP10K = (u32)(p10(I) % 10000);
    /* V(u0 + s) = V0 + s*D0 with u0 = P - Pb */
    u64 u0 = a.P - bs.Pb;
    u128 q2 = (u128)(u0 * u0) * bs.S2;
    u64 V0 = bs.FRAC + u0 * bs.S1 - (u64)(q2 >> SQ);
    u128 q3 = (u128)(2 * u0 + 1) * bs.S2;
    u64 D0 = bs.S1 - (u64)(q3 >> SQ);
    u32 q10k = (u32)(a.Q % 10000), q210k = (u32)(a.Q2 % 10000);
    u32 b2top = (u32)(a.B2f / p10(I));
    u64 W0base = a.P * C::PE;
    const u32 b1000 = b10k % 1000, b21000 = b210k % 1000;
    const i64 EPSF = 1ll << 26; /* 2^-10 */
#pragma unroll (kLeafUnroll)
    for (u32 d = 0; d <= 4; d++) {
        u64 v = a.Q + 2 * d * a.B + (u64)(d * d) * p10(I);
        u32 vm = (q10k + 2 * d * b10k + d * d * TOP10K) % 10000; /* v mod 10^4 */
        u32 s = vm % 10;
        u64 B = a.B + d * p10(I), Q = v / 10, P = a.P + s;
        u64 f = a.f + d * c_f64[1];
        u32 b11 = (a.b11 + d * (u32)(p10(I) % 11)) % 11;
        st.nodes++;
        u64 Vs = V0 + s * D0;
        /* Vs, Vs + D0 < 2^64 (host asserts); unsigned until after the shift */
        i64 wa = (Vs >= (u64)EPSF) ? (i64)((Vs - (u64)EPSF) >> 36) : -1;
        i64 wb = (i64)((Vs + D0 + (u64)EPSF) >> 36);
        u64 tlo = bs.Rb + (u64)wa, thi = bs.Rb + (u64)wb;
#ifdef WINDOW_CHECK
        {
            u64 elo = d_isqrt128((u128)P * C::PE), ehi = d_isqrt128((u128)(P + 1) * C::PE);
            if (tlo > elo || thi < ehi || thi > ehi + 2 || tlo + 2 < elo) atomicAdd((unsigned long long *)&out->dbg[1], 1ull);
        }
#endif
        u64 E0 = tlo * tlo - (W0base + s * C::PE);
        u32 tm11 = (u32)((bs.rb11 + 11 + wa) % 11), tm1000 = (u32)((bs.rb1000 + 1000 + wa) % 1000);
        group<L, H, NE>(B, Q, f, b11, b1000, vm / 10, E0, tlo, tm11, tm1000, thi, y3p, st, out);
        if (s2) {
            u32 d2 = (b2top + d) % 10; /* E' = 1 (mod 10): top digit of B2 shifts with d */
            u64 B2 = a.B2 + d2 * p10(I);
            u64 v2 = a.Q2 + 2 * d2 * a.B2 + (u64)(d2 * d2) * p10(I);
            u32 vm2 = (q210k + 2 * d2 * b210k + d2 * d2 * TOP10K) % 10000;
            u64 Q2 = v2 / 10, f2 = a.f2 + d2 * c_f64[1];
            u32 b211 = (a.b211 + d2 * (u32)(p10(I) % 11)) % 11;
            u32 q21000 = vm2 / 10;
            if (B2 >= C::H5) {
                Q2 = Q2 - B2 + C::Q25; B2 -= C::H5; f2 -= 1ull << 63;
                b211 = (b211 + 11 - (u32)C::H5M11) % 11; q21000 = (q21000 + 1000 - b21000) % 1000;
            }
            group<L, H, NE>(B2, Q2, f2, b211, b21000, q21000, E0, tlo, tm11, tm1000, thi, y3p, st, out);
        }
        /* flush the filter hits of this leaf (one copy of the rare path) */
        if (st.nh) {
#pragma unroll 1
            for (int i = 0; i < st.nh; i++) rare<L, H>(st.hb[i].t, st.hb[i].Bp, st.hb[i].Bq, P, st.hb[i].D, st.hb[i].y, st, out);
            st.nh = 0;
        }
    }
}

/*
 * dstep -- append digit d at the compile-time level I (see the CPU dstep()).
 *   S digit I = v mod 10 with v = Q + 2 d B + d^2 10^I; the sigma2 image's
 *   digit I is read from B2f = B*E' mod 10^h and appended to B2/Q2/f2/b211.
 *   All divisors are compile-time constants (multiply-high, no division calls).
 */
template <int H, int I>
__device__ __forceinline__ void dstep(const DS &a, DS &n, u32 d)
{
    u64 v = a.Q + 2ull * d * a.B + (u64)(d * d) * p10(I);
    u64 s = v % 10;
    n.P = a.P + s * p10(H - 1 - I);
    n.B = a.B + d * p10(I);
    n.Q = v / 10;
    n.f = a.f + d * c_f64[H - I];
    n.b11 = (a.b11 + d * (u32)(p10(I) % 11)) % 11;
    u64 b2f = (a.B2f + d * c_ce[I]) % p10(H);
    n.B2f = b2f;
    u32 d2 = (u32)((b2f / p10(I)) % 10);
    u64 v2 = a.Q2 + 2ull * d2 * a.B2 + (u64)(d2 * d2) * p10(I);
    n.B2 = a.B2 + d2 * p10(I);
    n.Q2 = v2 / 10;
    n.f2 = a.f2 + d2 * c_f64[H - I];
    n.b211 = (a.b211 + d2 * (u32)(p10(I) % 11)) % 11;
}

/* dstep_rt -- same as dstep with a runtime level I (powers from c_p10; uses real
 * 64-bit divisions, so only used for the per-thread base replay). */
__device__ __forceinline__ void dstep_rt(const DS &a, DS &n, u32 d, int I, int H)
{
    u64 pI = c_p10[I], ph = c_p10[H];
    u64 v = a.Q + 2ull * d * a.B + (u64)(d * d) * pI;
    n.P = a.P + (v % 10) * c_p10[H - 1 - I];
    n.B = a.B + d * pI;
    n.Q = v / 10;
    n.f = a.f + d * c_f64[H - I];
    n.b11 = (u32)((a.b11 + d * (pI % 11)) % 11);
    u64 b2f = (a.B2f + d * c_ce[I]) % ph;
    n.B2f = b2f;
    u32 d2 = (u32)((b2f / pI) % 10);
    u64 v2 = a.Q2 + 2ull * d2 * a.B2 + (u64)(d2 * d2) * pI;
    n.B2 = a.B2 + d2 * pI;
    n.Q2 = v2 / 10;
    n.f2 = a.f2 + d2 * c_f64[H - I];
    n.b211 = (u32)((a.b211 + d2 * (pI % 11)) % 11);
}

/* dstep with a runtime level I in [I0, H-2], dispatched to compile-time constants
 * (a chain of comparisons resolved at compile time; the dstep_rt fallback is
 * never reached at run time) */
template <int H, int I0>
__device__ __forceinline__ void dstep_ct(const DS &a, DS &n, u32 d, int I)
{
    if constexpr (I0 >= H - 1) {
        dstep_rt(a, n, d, I, H);
    } else {
        if (I == I0) dstep<H, I0>(a, n, d);
        else dstep_ct<H, I0 + 1>(a, n, d, I);
    }
}

/*
 * dfs_iter -- iterative DFS over levels IB..H-2 of one thread's subtree.
 *   In:  a0 (state after the IB base digits), s2_0 (sigma2 restriction already
 *        applied in the base), bs, SQ, b1000/b21000 (B, B2 mod 10^4), y3p, st, out.
 *   Out: calls leafs() for every level-(h-1) parent.
 *   The explicit stack (stk/dig/dmx/s2s, K entries) lives in local memory and
 *   is touched once per parent; keeping a recursive template version in
 *   registers needed > 255 registers.  Digit limit per level is 9, or 4 where
 *   s2_level() applies (the unit-5 exception only matters at level 1, which is
 *   always inside the base).
 */
template <int L, int H, int IB, int NE>
__device__ __forceinline__ void dfs_iter(const DS &a0, bool s2_0, const Base &bs, int SQ, u32 b1000, u32 b21000,
                                         const u64 *y3p, Thr &st, Out *out)
{
    constexpr int K = H - 1 - IB; /* internal levels IB .. H-2 */
    if constexpr (K <= 0) {
        leafs<L, H, NE>(a0, s2_0, bs, SQ, b1000, b21000, y3p, st, out);
    } else {
        DS stk[K];
        unsigned char dig[K], dmx[K]; /* next digit to try, max digit, per level */
        bool s2s[K];                  /* sigma2 flag for the children of each level */
        stk[0] = a0;
        {
            bool r = s2_level(IB, a0.B, false, H);
            dmx[0] = r ? 4 : 9;
            s2s[0] = s2_0 | r;
            dig[0] = 0;
        }
        int lv = 0;
#pragma unroll 1
        while (lv >= 0) {
            if (dig[lv] > dmx[lv]) { lv--; continue; } /* level exhausted: back up */
            u32 d = dig[lv]++;
            DS n;
            dstep_ct<H, IB>(stk[lv], n, d, IB + lv);
            if (lv == K - 1) {
                leafs<L, H, NE>(n, s2s[lv], bs, SQ, b1000, b21000, y3p, st, out);
            } else {
                bool r = s2_level(IB + lv + 1, n.B, false, H);
                bool s2n = s2s[lv] | r;
                lv++;
                stk[lv] = n;
                dmx[lv] = r ? 4 : 9;
                s2s[lv] = s2n;
                dig[lv] = 0;
            }
        }
    }
}

/*
 * search_kernel -- one thread per base k = base0 + global thread index.
 *   In:  base0, nbases (bases of this launch), SQ (S2 scale), out.
 *   Out: per-launch counters (warp-reduced, then atomically added) and the
 *        survivors in out.  Threads with idx >= nbases or an invalid base only
 *        take part in the shared-memory setup and the warp reduction (blockDim
 *        must be a multiple of 32 for the full-mask shuffles).
 *   Steps per valid base: replay the IB digits (validity: unit 5 needs tens
 *        <= 4, sigma2-restricted digits <= 4); exact sqrt base and fixed-point
 *        Taylor coefficients; choose NE (1 if s1 < 9.9, i.e. every window in
 *        this thread is at most 11 wide, else 2); run the DFS.
 */
template <int L, int H, int IB>
__global__ void __launch_bounds__(256, MINB) search_kernel(u64 base0, u64 nbases, int SQ, Out *out)
{
    typedef Cfg<L, H> C;
    /* shared table: s_y3p[l] = rev3(l) * 10^(e-3), the filter target for low digits l */
    __shared__ u64 s_y3p[1000];
    for (int i = threadIdx.x; i < 1000; i += blockDim.x) s_y3p[i] = (u64)rev3(i) * p10(C::e - 3);
    __syncthreads();
    u64 idx = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    Hit hbuf[HITBUF];
    Thr st;
    st.nodes = st.filt = st.passB = st.pass3 = 0;
    st.nh = 0;
    st.hb = hbuf;
    if (idx < nbases) {
        u64 k = base0 + idx;
        u64 per = p10(IB - 1);
        u64 u = 1 + k / per, rest = k % per;   /* unit digit 1..5, digits 1..IB-1 in rest */
        bool unit5 = (u == 5);
        bool valid = !(unit5 && (rest % 10) > 4);
        DS a = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, n;
        bool s2 = false;
        u64 rr = rest;
        for (int i = 0; i < IB && valid; i++) {
            u32 d = (i == 0) ? (u32)u : (u32)(rr % 10);
            if (i > 0) rr /= 10;
            if (s2_level(i, a.B, unit5, H)) { if (d > 4) valid = false; s2 = true; }
            dstep_rt(a, n, d, i, H);
            a = n;
        }
        if (valid) {
            /* window base: exact isqrt, then the Taylor coefficients in fixed point */
            Base bs;
            u128 X = (u128)a.P * C::PE;
            u64 r = d_isqrt128(X);
            double R = (double)r;
            double frac = d_u128_to_double(X - (u128)r * r) / (2.0 * R);
            double Rreal = R + frac, Pd = (double)a.P;
            double s1 = Rreal / (2.0 * Pd), s2d = Rreal / (8.0 * Pd * Pd);
            bs.Pb = a.P;
            bs.Rb = r;
            bs.FRAC = (u64)(frac * 68719476736.0); /* 2^36 */
            bs.S1 = (u64)(s1 * 68719476736.0);
            bs.S2 = (u64)(s2d * ldexp(1.0, 36 + SQ));
            bs.rb11 = (u32)(r % 11);
            bs.rb1000 = (u32)(r % 1000);
            u32 b1000 = (u32)(a.B % 10000), b21000 = (u32)(a.B2 % 10000); /* mod 10^4 */
            /* window width <= 11 everywhere in this thread: one candidate per root */
            bool one = s1 < 9.9;
            if (one) dfs_iter<L, H, IB, 1>(a, s2, bs, SQ, b1000, b21000, s_y3p, st, out);
            else dfs_iter<L, H, IB, 2>(a, s2, bs, SQ, b1000, b21000, s_y3p, st, out);
        }
    }
    /* warp reduction of the counters, then one atomic per warp */
    for (int o = 16; o > 0; o >>= 1) {
        st.nodes += __shfl_down_sync(0xffffffff, st.nodes, o);
        st.filt += __shfl_down_sync(0xffffffff, st.filt, o);
        st.passB += __shfl_down_sync(0xffffffff, st.passB, o);
        st.pass3 += __shfl_down_sync(0xffffffff, st.pass3, o);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&out->nodes, st.nodes);
        atomicAdd(&out->filt, st.filt);
        atomicAdd(&out->passB, st.passB);
        atomicAdd(&out->pass3, st.pass3);
    }
}

/* ---------------- host ---------------- */

/* square_digits -- decimal digits of n^2 into out (>= 80 chars); returns the
 * digit count.  Base-10^9 schoolbook multiplication; see the CPU version. */
static int square_digits(u128 n, char *out)
{
    u64 a[6] = {0}, s[12] = {0};
    int na = 0;
    while (n) { a[na++] = (u64)(n % 1000000000u); n /= 1000000000u; }
    for (int i = 0; i < na; i++) {
        u64 carry = 0;
        for (int j = 0; j < na; j++) {
            u64 v = s[i + j] + a[i] * a[j] + carry;
            s[i + j] = v % 1000000000u;
            carry = v / 1000000000u;
        }
        int k = i + na;
        while (carry) { u64 v = s[k] + carry; s[k] = v % 1000000000u; carry = v / 1000000000u; k++; }
    }
    int ns = 2 * na;
    while (ns > 0 && s[ns - 1] == 0) ns--;
    char buf[128];
    int len = 0;
    for (int i = 0; i < ns; i++) {
        u64 v = s[i];
        for (int k = 0; k < 9; k++) { buf[len++] = (char)('0' + v % 10); v /= 10; }
    }
    while (len > 1 && buf[len - 1] == '0') len--;
    for (int i = 0; i < len; i++) out[i] = buf[len - 1 - i];
    out[len] = 0;
    return len;
}

/* u128_str -- decimal string of n into out (>= 40 chars). */
static void u128_str(u128 n, char *out)
{
    char buf[64];
    int len = 0;
    do { buf[len++] = (char)('0' + (int)(n % 10)); n /= 10; } while (n);
    for (int i = 0; i < len; i++) out[i] = buf[len - 1 - i];
    out[len] = 0;
}

/* now_sec -- monotonic time in seconds. */
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* SIGINT/SIGTERM set stop_flag; run() stops after the current launch. */
static volatile sig_atomic_t stop_flag = 0;
static void on_sig(int s) { (void)s; stop_flag = 1; }

/* Each compiled (L, h, IB) has a launcher with this signature. */
typedef void (*launch_fn)(u64 base0, u64 nb, int SQ, Out *out, int bd);

/* launch -- start search_kernel<L,H,IB> for bases [base0, base0+nb) with bd
 * threads per block (asynchronous; errors are picked up by cudaGetLastError). */
template <int L, int H, int IB>
static void launch(u64 base0, u64 nb, int SQ, Out *out, int bd)
{
    u64 blocks = (nb + bd - 1) / bd;
    search_kernel<L, H, IB><<<(unsigned)blocks, bd>>>(base0, nb, SQ, out);
}

struct Variant { int L, H, IB; launch_fn fn; };

/* ib_for -- base depth IB: the smallest depth with the cubic Taylor term
 * Rb*U^3/(16 Pb^3) < 1e-8 (U = 10^(h-IB)) and K = h - IB <= 7; same rule as
 * the CPU setup(), evaluated at compile time. */
static constexpr int ib_for(int L, int H)
{
    int e = L - 3 * H;
    for (int ib = 2; ib <= H - 1; ib++) {
        double lg = (H + e) / 2.0 + 3.0 * (1 - ib) - 1.2041199826559248; /* log10(16) */
        if (lg < -8 && H - ib <= 7) return ib;
    }
    return H - 1;
}

/* Compiled lengths.  h = floor(L/4) keeps every window <= 22 wide. */
#define VAR(L, H) { L, H, ib_for(L, H), launch<L, H, ib_for(L, H)> }
static const Variant VARIANTS[] = {
    VAR(40, 10), VAR(42, 10), VAR(44, 11), VAR(46, 11), VAR(48, 12), VAR(50, 12), VAR(52, 13), VAR(54, 13),
    VAR(56, 14), VAR(58, 14), VAR(60, 15), VAR(62, 15), VAR(64, 16), VAR(66, 16), VAR(68, 17), VAR(70, 17),
};

static u64 P10H[20]; /* 10^k on the host */

/*
 * write_state -- atomically replace the state file (write "<state>.tmp", rename).
 *   Called after every completed launch, so pos/found/counters always describe
 *   whole launches: a resumed run neither skips nor double-counts anything.
 *   Errors: prints a warning and keeps the previous checkpoint if the file
 *   cannot be written, flushed or renamed; the run continues.
 */
static void write_state(const char *state, int L, u64 ra, u64 rb, u64 pos, u64 found,
                        unsigned long long nodes, unsigned long long filt, unsigned long long passB,
                        unsigned long long pass3)
{
    char tmp[1024];
    snprintf(tmp, sizeof tmp, "%s.tmp", state);
    FILE *f = fopen(tmp, "w");
    if (!f) { fprintf(stderr, "\nwarning: cannot write %s: %s\n", tmp, strerror(errno)); return; }
    int ok = fprintf(f, "A034822cuda L=%d range=%" PRIu64 ":%" PRIu64 " pos=%" PRIu64 " found=%" PRIu64
                        " nodes=%llu filt=%llu passB=%llu pass3=%llu\n",
                     L, ra, rb, pos, found, nodes, filt, passB, pass3) > 0;
    int err = errno;
    if (fclose(f) != 0 && ok) { ok = 0; err = errno; }
    if (ok && rename(tmp, state) != 0) { ok = 0; err = errno; }
    if (!ok) {
        fprintf(stderr, "\nwarning: state file %s not updated: %s\n", state, strerror(err));
        remove(tmp);
    }
}

/*
 * run -- search length L over the base range [ra, rb).
 *   In:  L          even length (must be in VARIANTS)
 *        bpl        bases per launch (1 .. 2^30)
 *        state      state file or NULL
 *        ra, rb     base range; rb = 0 or rb > count means "to the end"
 *        bd         threads per block (multiple of 32, 32..256)
 *        quiet      selftest: no FOUND lines or progress
 *   Out: returns the number of palindromic squares found in the range
 *        (including those recorded in a resumed state file); prints FOUND lines,
 *        progress, and a final DONE/PARTIAL line.
 *   Setup: uploads c_ce for this h, picks SQ so S2 = s2*2^(36+SQ) < 2^62, and
 *        checks that the 128-bit curvature product and the 2^-36 fixed-point
 *        window cannot overflow for u < 10^K.
 *   Each launch: zero Out, run the kernel, copy Out back, verify every survivor
 *        by full decimal conversion (only genuine L-digit palindromes count),
 *        add the counters, advance pos, write the state file.
 *   Errors (exit 1): L not compiled in; empty or out-of-range base range;
 *        state file present but for a different L or range; any CUDA error;
 *        more than MAXOUT survivors in one launch (rerun with a smaller -b; the
 *        state file still points at the start of that launch); overflow checks.
 *        Validation builds exit 3 on a detected problem.
 */
static u64 run(int L, u64 bpl, const char *state, u64 ra, u64 rb, int bd, int quiet)
{
    const Variant *vv = NULL;
    for (size_t i = 0; i < sizeof VARIANTS / sizeof VARIANTS[0]; i++)
        if (VARIANTS[i].L == L) vv = &VARIANTS[i];
    if (!vv) { fprintf(stderr, "L=%d not compiled in (even 40..70)\n", L); exit(1); }
    int H = vv->H, IB = vv->IB, e = L - 3 * H;
    u64 nbases = 5 * P10H[IB - 1];
    {
        /* E' = -1 (mod 2^h), 1 (mod 5^h); c_ce[k] = 10^k E' mod 10^h */
        u64 m2 = 1ull << H, m5 = P10H[H] >> H, M = P10H[H], e1 = 1;
        while (e1 % m2 != m2 - 1) e1 += m5;
        u64 ES = e1 % M, ce[20] = {0};
        if ((u64)(((u128)ES * ES) % M) != 1) { fprintf(stderr, "bad E'\n"); exit(1); }
        for (int k = 0; k < H; k++) ce[k] = (u64)(((u128)P10H[k] * ES) % M);
        CK(cudaMemcpyToSymbol(c_ce, ce, sizeof ce));
    }
    if (rb == 0 || rb > nbases) rb = nbases;
    if (ra >= rb) {
        fprintf(stderr, "empty base range [%" PRIu64 ",%" PRIu64 ") (L=%d has %" PRIu64 " bases)\n", ra, rb, L, nbases);
        exit(1);
    }
    /* SQ: S2 = s2 * 2^(36+SQ) < 2^63 for the largest s2 = s1max/(4*10^(h-1)) */
    double s1max = sqrt(pow(10.0, e - H + 1)) / 2.0;
    double s2max = s1max / (4.0 * pow(10.0, H - 1));
    int SQ = (int)floor(log2(ldexp(1.0, 62) / (s2max * ldexp(1.0, 36))));
    if (SQ > 90) SQ = 90;
    /* u0^2 * S2 must fit in 128 bits: u0 < 10^K */
    int K = H - IB;
    if (2 * K * log2(10.0) + 63 > 127) { fprintf(stderr, "K too large\n"); exit(1); }
    if (SQ >= 64 + 63) { fprintf(stderr, "bad SQ\n"); exit(1); }
    /* fixed-point window V = frac + u s1 (2^-36 units) must stay below 2^64 for u < 10^K */
    if ((s1max + 1.0) * ldexp(1.0, 36) * (pow(10.0, K) + 2) + ldexp(1.0, 37) >= ldexp(1.0, 64)) {
        fprintf(stderr, "fixed-point window overflow for L=%d (K=%d)\n", L, K); exit(1);
    }

    u64 pos = ra, found = 0;
    unsigned long long nodes = 0, filt = 0, passB = 0, pass3 = 0;
    if (state) {
        FILE *f = fopen(state, "r");
        if (f) {
            int sL; u64 sra, srb, spos, sfound;
            if (fscanf(f, "A034822cuda L=%d range=%" SCNu64 ":%" SCNu64 " pos=%" SCNu64 " found=%" SCNu64
                          " nodes=%llu filt=%llu passB=%llu pass3=%llu",
                       &sL, &sra, &srb, &spos, &sfound, &nodes, &filt, &passB, &pass3) == 9 &&
                sL == L && sra == ra && srb == rb && spos >= ra && spos <= rb) {
                pos = spos;
                found = sfound;
                if (!quiet) fprintf(stderr, "resumed at base %" PRIu64 "\n", pos);
            } else { fprintf(stderr, "state file %s does not match L=%d range %" PRIu64 ":%" PRIu64 "\n", state, L, ra, rb); exit(1); }
            fclose(f);
        }
    }
    if (!quiet)
        fprintf(stderr, "L=%d h=%d e=%d IB=%d K=%d SQ=%d bases [%" PRIu64 ",%" PRIu64 ") of %" PRIu64
                        ", %" PRIu64 " per launch\n", L, H, e, IB, K, SQ, ra, rb, nbases, bpl);
    Out *d_out, *h_out;
    CK(cudaMalloc(&d_out, sizeof(Out)));
    CK(cudaMallocHost(&h_out, sizeof(Out)));
    double t0 = now_sec(), tlast = t0;
    u64 pos0 = pos;
    while (pos < rb && !stop_flag) {
        u64 nb = rb - pos < bpl ? rb - pos : bpl;
        CK(cudaMemset(d_out, 0, sizeof(Out)));
        vv->fn(pos, nb, SQ, d_out, bd);
        CK(cudaGetLastError());
        CK(cudaMemcpy(h_out, d_out, sizeof(Out), cudaMemcpyDeviceToHost)); /* waits for the kernel */
        if (h_out->nout > MAXOUT) {
            fprintf(stderr, "output overflow (%u survivors > %d) in bases [%" PRIu64 ",%" PRIu64 "); rerun with smaller -b\n",
                    h_out->nout, MAXOUT, pos, pos + nb);
            exit(1);
        }
        /* host verification: only genuine L-digit palindromic squares count */
        for (unsigned i = 0; i < h_out->nout; i++) {
            u128 n = (u128)h_out->t[i] * P10H[H] + h_out->b[i];
            char ss[128], ns[64];
            int len = square_digits(n, ss);
            int pal = 1;
            for (int a = 0, b = len - 1; a < b; a++, b--) if (ss[a] != ss[b]) pal = 0;
            if (len == L && pal) {
                found++;
                u128_str(n, ns);
                if (!quiet) { printf("FOUND %d %s %s\n", L, ns, ss); fflush(stdout); }
            }
        }
#ifdef WINDOW_CHECK
        if (h_out->dbg[1]) { fprintf(stderr, "WINDOW_CHECK: %llu bad windows!\n", (unsigned long long)h_out->dbg[1]); exit(3); }
#endif
#ifdef FILTER_CHECK
        if (h_out->dbg[0]) { fprintf(stderr, "FILTER_CHECK: %llu exact hits rejected by the filter!\n", (unsigned long long)h_out->dbg[0]); exit(3); }
#endif
        nodes += h_out->nodes; filt += h_out->filt; passB += h_out->passB; pass3 += h_out->pass3;
        pos += nb;
        double t = now_sec();
        if (state) write_state(state, L, ra, rb, pos, found, nodes, filt, passB, pass3);
        if (!quiet && (t - tlast > 10 || pos >= rb)) {
            double frac = (double)(pos - ra) / (double)(rb - ra);
            double rate = (double)(pos - pos0) / (t - t0);
            fprintf(stderr, "\r%8.4f%%  base %" PRIu64 "  leaves %.3e  %.3e leaves/s  found %" PRIu64
                            "  eta %.0fs    ", 100 * frac, pos, (double)nodes,
                    (double)nodes / (t - t0 + 1e-9),
                    found, rate > 0 ? (rb - pos) / rate : 0.0);
            tlast = t;
        }
    }
    double el = now_sec() - t0;
    if (!quiet) fprintf(stderr, "\n");
    printf("%s L=%d count=%" PRIu64 " bases=[%" PRIu64 ",%" PRIu64 ") pos=%" PRIu64
           " leaves=%llu filt=%llu passB=%llu pass3=%llu time=%.2fs rate=%.3e leaves/s\n",
           pos >= rb ? "DONE" : "PARTIAL", L, found, ra, rb, pos, nodes, filt, passB, pass3, el,
           el > 0 ? (double)nodes / el : 0.0);
    fflush(stdout);
    cudaFree(d_out);
    cudaFreeHost(h_out);
    return found;
}

/*
 * main -- set up the constant tables and dispatch "search" / "selftest".
 *   Validates -b (1..2^30), -B (multiple of 32 in 32..256) and the selftest
 *   range (maxL <= 68: KNOWN[70] is unknown).  See USAGE for exit codes.
 */
int main(int argc, char **argv)
{
    P10H[0] = 1;
    for (int i = 1; i < 20; i++) P10H[i] = P10H[i - 1] * 10;
    u64 f64[20];
    for (int k = 0; k < 20; k++) f64[k] = k == 0 ? ~0ull : (u64)(((u128)1 << 64) / P10H[k]);
    CK(cudaMemcpyToSymbol(c_f64, f64, sizeof f64));
    CK(cudaMemcpyToSymbol(c_p10, P10H, sizeof P10H));
    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);
    int bd = 256;
    if (argc >= 2 && !strcmp(argv[1], "selftest")) {
        int maxL = argc >= 3 ? atoi(argv[2]) : 52, bad = 0;
        if (maxL > 68) {
            fprintf(stderr, "selftest: counts are known only up to L = 69 (odd L not supported); using maxL = 68\n");
            maxL = 68;
        }
        if (maxL < 40) { fprintf(stderr, "selftest: maxL must be >= 40\n"); return 2; }
        for (int L = 40; L <= maxL; L += 2) {
            u64 c = run(L, 1 << 20, NULL, 0, 0, bd, 1);
            fprintf(stderr, "L=%d count=%" PRIu64 " expected %d %s\n", L, c, KNOWN[L], (int)c == KNOWN[L] ? "ok" : "MISMATCH");
            if ((int)c != KNOWN[L]) bad++;
        }
        fprintf(stderr, bad ? "SELFTEST FAILED\n" : "SELFTEST PASSED\n");
        return bad != 0;
    }
    if (argc >= 3 && !strcmp(argv[1], "search")) {
        int L = atoi(argv[2]);
        u64 bpl = 1 << 20, ra = 0, rb = 0;
        const char *state = NULL;
        for (int i = 3; i < argc; i++) {
            if (!strcmp(argv[i], "-b") && i + 1 < argc) bpl = strtoull(argv[++i], NULL, 10);
            else if (!strcmp(argv[i], "-S") && i + 1 < argc) state = argv[++i];
            else if (!strcmp(argv[i], "-B") && i + 1 < argc) bd = atoi(argv[++i]);
            else if (!strcmp(argv[i], "-r") && i + 1 < argc) {
                if (sscanf(argv[++i], "%" SCNu64 ":%" SCNu64, &ra, &rb) != 2) goto usage;
            } else goto usage;
        }
        if (bpl < 1 || bpl > (1ull << 30)) { fprintf(stderr, "-b must be in 1..2^30\n"); return 1; }
        if (bd < 32 || bd > 256 || bd % 32) { fprintf(stderr, "-B must be a multiple of 32 in 32..256\n"); return 1; }
        run(L, bpl, state, ra, rb, bd, 0);
        return 0;
    }
usage:
    fprintf(stderr, "usage: %s search L [-b bases_per_launch] [-S state] [-r a:b] [-B blockdim]\n"
                    "       %s selftest [maxL]\n", argv[0], argv[0]);
    return 2;
}
