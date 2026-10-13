/*
 * a171775_cuda.cu -- CUDA (NVIDIA GPU) version of the 1D A171775 search
 * (OEIS A171775: smallest M that is a k-digit palindrome in some base for
 * every k = 2..n).  Written for the DGX Spark (GB10, sm_121); see Makefile.
 *
 * Same method, work units, statistics and state files as ../a171775.c:
 * read its theory of operations first.  This file keeps its own copies of
 * the host helpers (it does not include a171775.c), so the two must be kept
 * in step: the unit numbering, the per-(B, c) constants, the canonical-base
 * rule, surv_hash() and the state-file format are what make the results
 * comparable (the selftest checks it) and the checkpoints interchangeable.
 *
 * ========================================================================
 * THEORY OF OPERATIONS (the GPU part)
 * ========================================================================
 *
 * 1. Work.  Units (B, d_0[, d_1]) are numbered base by base as in the CPU
 *    tool.  The host turns a batch of consecutive units into an item list,
 *    one item per (block, c) -- one call of block_c() in the CPU tool
 *    (gen_unit()) -- and one GPU thread processes one item.  The constants
 *    of every (B, c) of the whole search are built once (build_tables()) and
 *    live in device memory; an item holds Mblock and the index of its (B, c).
 *
 * 2. The thread's state, all in registers (scan_kernel()):
 *      rem  = M0 mod P   (u64)                 P = c^(h+1)
 *      w    = (T(q) - M0) u mod N              N = c^k; type WT = u32 if every
 *                                              N of the search is < 2^31, else u64
 *      E[]  = the k base-c digits of q = floor(M0 / P)
 *    The kernel is a template over K = k (2..5) and WT, so the digit loops
 *    unroll and E[] stays in registers.
 *
 * 3. The step.  (x, y) runs over [0, B)^2 as in block_c().  The coarse test
 *    is  w < gB  or  (w - a) mod 2^bits < gB,  with a = N - wnext: the
 *    second term is w1 = w + wnext - N, the w of region 1, when w >= a; when
 *    w < a, w1 = w + wnext >= w, so w1 < gB implies the first term.  So no
 *    hit is missed, and rare_path() redoes both regions exactly in 64 bits
 *    (divisibility by g, the region bounds) as block_c() does.  The update
 *    is rem += w_y (dx at a y wrap) with a carry into q when rem >= P - w_y
 *    (precomputed), and w += D0 or D1 (mod N): D0 = -w_y u, D1 = wnext - w_y u
 *    (a step that also carries q).  cnt counts the increments of q left
 *    until E[k-1] reaches c-1 (phase 0) or until the carry (phase 1), the
 *    elast/trig logic of block_c(); RECOMPUTE() then refreshes wnext.
 *
 * 4. Survivors.  rare_path() counts every z (the "hits") and emit() checks
 *    the range and the full (n-1)-digit palindrome in base c on the GPU; the
 *    few that pass (st2) are written to an output buffer and go back to the
 *    host, which applies the canonical-base rule and tests the lengths
 *    n-2..3 with the CPU tool's code (post_process(), over host threads).
 *
 * 5. Pipeline and checkpoints.  Two slots (CUDA streams) alternate: the host
 *    builds the next batch while the GPU runs the other one, and batches are
 *    finished in the order they were issued.  So the statistics always cover
 *    exactly the units below the oldest unfinished batch -- the frontier
 *    saved with -S -- and every checkpoint is exact, whenever the run ends.
 *
 * 6. Limits.  n = 7..14 (k = n-2-h must be 2..5); c^(h+1) < 2^62 and
 *    c^k < 2^42 as in the CPU tool; at most 2^20 survivors per batch (lower
 *    -b if a batch overflows); the items of one unit must fit in a batch
 *    (raise -b).  A range without n-digit numbers gives a note and an empty
 *    summary.
 *
 * Usage
 *   a171775_cuda selftest            a(7), a(8), a(9) + CPU-tool statistics
 *   a171775_cuda search n LO HI [-S state] [-i sec] [-b items] [-t host threads]
 * -b items per batch (default 2^20), -t host threads for post_process()
 * (default: all cores).  Output and exit status as in a171775.c, except that
 * an interrupted search exits with status 3; CUDA errors exit with status 1
 * and name the failing call (CK()).
 */
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#include <unistd.h>
#include <cuda_runtime.h>

typedef unsigned __int128 u128;
typedef uint64_t u64;
typedef uint32_t u32;

#define MAXN 24                     /* largest length n the arrays allow */
#define MAXK 6                      /* size of CConst.Wc[]; the kernels handle k <= 5 */
#define U128_MAX (~(u128)0)         /* also the saturation value of pow_sat() */

/* CK(call) -- run a CUDA runtime call; on failure print the error with the
   source line and exit with status 1. */
#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

/* ------------------------------------------------------------------ */
/* host integer helpers (same as the CPU tool)                          */
/* ------------------------------------------------------------------ */
/* These are copies of the functions in ../a171775.c; see the comments there
   for inputs, outputs and limits.  Briefly:
     pow_sat(b, e)        b^e, saturating at U128_MAX
     iroot(x, k)          floor(x^(1/k)), exact
     u128s(x, buf)        decimal string (buf >= 40 bytes)
     parse_term/_or_die   number expressions like 2^72-1 (exit 1 on bad input)
     gcd64, modinv64      gcd; inverse mod m (0 if none)
     mulmod64, mod128_64  a b mod m; 128-bit M mod 64-bit b
     is_pal_base(M, b, L) M is an L-digit palindrome in base b
     find_pal_base(M, L)  smallest such base (M-1 for L = 2), 0 if none
     surv_hash(M)         survivor hash; the checksum is its sum mod 2^64
     now_sec()            monotonic seconds */

static u128 pow_sat(u64 b, int e)
{
    u128 r = 1;
    for (int i = 0; i < e; i++) {
        if (b && r > U128_MAX / b) return U128_MAX;
        r *= b;
    }
    return r;
}

static u64 iroot(u128 x, int k)
{
    if (x == 0) return 0;
    double a = pow((double)x, 1.0 / k);
    u64 r = a >= 1.8e19 ? (u64)1.8e19 : (u64)a;
    if (r == 0) r = 1;
    while (r > 1 && pow_sat(r, k) > x) r--;
    while (pow_sat(r + 1, k) <= x) r++;
    return r;
}

static char *u128s(u128 x, char *buf)
{
    char t[48];
    int i = 0;
    do { t[i++] = (char)('0' + (int)(x % 10)); x /= 10; } while (x);
    for (int j = 0; j < i; j++) buf[j] = t[i - 1 - j];
    buf[i] = 0;
    return buf;
}

static int parse_term(const char **ps, u128 *out)
{
    const char *s = *ps;
    u128 v = 0;
    int nd = 0;
    while (*s >= '0' && *s <= '9') { v = v * 10 + (u128)(*s - '0'); s++; nd++; }
    if (!nd) return 0;
    if (*s == '^' || *s == 'e' || *s == 'E') {
        char op = *s++;
        char *e;
        unsigned long ex = strtoul(s, &e, 10);
        if (e == s) return 0;
        s = e;
        if (op == '^') {
            if (v > UINT64_MAX) return 0;
            v = pow_sat((u64)v, (int)ex);
        } else v *= pow_sat(10, (int)ex);
    }
    *ps = s;
    *out = v;
    return 1;
}

static u128 parse_or_die(const char *s0)
{
    const char *s = s0;
    u128 acc, t;
    if (!parse_term(&s, &acc)) goto bad;
    while (*s == '+' || *s == '-') {
        char op = *s++;
        if (!parse_term(&s, &t)) goto bad;
        acc = op == '+' ? acc + t : acc - t;
    }
    if (*s) goto bad;
    return acc;
bad:
    fprintf(stderr, "bad number: %s\n", s0);
    exit(1);
}

static u64 gcd64(u64 a, u64 b)
{
    while (b) { u64 t = a % b; a = b; b = t; }
    return a;
}

static u64 modinv64(u64 a, u64 m)
{
    if (m == 1) return 0;
    __int128 t = 0, nt = 1, r = m, nr = a % m;
    while (nr) {
        __int128 q = r / nr, tmp;
        tmp = t - q * nt; t = nt; nt = tmp;
        tmp = r - q * nr; r = nr; nr = tmp;
    }
    if (r != 1) return 0;
    if (t < 0) t += m;
    return (u64)t;
}

static inline u64 mulmod64(u64 a, u64 b, u64 m) { return (u64)((u128)a * b % m); }

static inline u64 mod128_64(u128 M, u64 b)
{
    u64 hi = (u64)(M >> 64), lo = (u64)M;
    if (!hi) return lo % b;
    if (b <= 0xffffffffULL) {
        u64 r = hi % b;
        r = ((r << 32) | (lo >> 32)) % b;
        r = ((r << 32) | (lo & 0xffffffffULL)) % b;
        return r;
    }
    return (u64)(M % b);
}

static int is_pal_base(u128 M, u64 b, int L)
{
    u64 d[130];
    int len = 0;
    if (b < 2) return 0;
    u128 x = M;
    while (x) {
        if (len >= L) return 0;
        d[len++] = (u64)(x % b);
        x /= b;
    }
    if (len != L) return 0;
    for (int i = 0; i < L / 2; i++)
        if (d[i] != d[L - 1 - i]) return 0;
    return 1;
}

static u64 find_pal_base(u128 M, int L)
{
    if (L < 2 || M == 0) return 0;
    if (L == 2) {
        if (M < 3) return 0;
        return (M - 1 > UINT64_MAX) ? UINT64_MAX : (u64)(M - 1);
    }
    u64 blo = iroot(M, L) + 1, bhi = iroot(M, L - 1);
    if (blo < 2) blo = 2;
    double Md = (double)M;
    for (u64 b = blo; b <= bhi; b++) {
        u64 r = mod128_64(M, b);
        if (r == 0) continue;
        double la = Md / pow((double)b, (double)(L - 1));
        if (la < (double)r - 1.5 || la > (double)r + 1.5) continue;
        u128 P = pow_sat(b, L - 1);
        if (M / P != r) continue;
        if (is_pal_base(M, b, L)) return b;
    }
    return 0;
}

static inline u64 surv_hash(u128 M)
{
    u64 a = (u64)M * 0x9E3779B97F4A7C15ULL, b = (u64)(M >> 64) * 0xC2B2AE3D27D4EB4FULL;
    a ^= a >> 29;
    return a * 0xBF58476D1CE4E5B9ULL + b;
}

static double now_sec()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

/* ------------------------------------------------------------------ */
/* problem, per-B and per-(B,c) constants                              */
/* ------------------------------------------------------------------ */

/* Prob -- the search: as prob_t in ../a171775.c (n, h = n/2 digit pairs,
   D = ceil(n/2) free digits, k = n-2-h, bd block digits, ud unit digits,
   [lo, hi], the bases B with n-digit numbers in it, the unit offsets). */
struct Prob {
    int n, h, D, k, bd, ud;
    u128 lo, hi;
    u64 Bmin, Bmax, nunits;
    std::vector<u64> uoff;
};

/* CConst -- the constants of one pair (B, c), read by the kernel: the
   cconst_t fields of ../a171775.c (N = c^k, P = c^(h+1), g, N', the inverse
   u, limit = g B, Wy, Wx, Rcross, Wc[]) plus what a thread needs without a
   per-B table: the weights s, w_y, dx and w_x (128-bit, as two halves). */
struct CConst {                 /* per (B, c); read by the kernel */
    u64 B, c, N, P, g, Np, inv, limit, Wy, Wx, Rcross, s, wy, dx, wx_lo, wx_hi;
    u64 Wc[MAXK];
};

/* BInfo -- per base B (host only): the digit weights, the spans of a block
   and of a unit, the range [cmin, cmax] of bases c, and the index of
   (B, cmin) in the CConst table (the entries for one B are consecutive). */
struct BInfo {
    u128 w[MAXN];
    u128 span_block, span_unit;
    u64 cmin, cmax;
    u64 off;                    /* index of (B, cmin) in the table */
};

/* Item -- one GPU thread's work: Mblock (two halves) and the index cc of its
   (B, c) in the CConst table.  Out -- one survivor for the host: M (two
   halves), B and c. */
struct Item { u64 Mlo, Mhi; u32 cc, pad; };
struct Out { u64 Mlo, Mhi; u32 B, c; };

/* prob_init -- as in ../a171775.c: fill pr for n and [lo, hi] (lo >= 1) and
   number the units, (B-1) B^(ud-1) per base B.  No errors; nunits = 0 if no
   base has n-digit numbers in the range. */
static void prob_init(Prob &pr, int n, u128 lo, u128 hi)
{
    pr.n = n;
    pr.h = n / 2;
    pr.D = (n + 1) / 2;
    pr.k = n - 2 - pr.h;
    pr.bd = pr.D - 3;
    pr.ud = pr.bd - 1 >= 1 ? pr.bd - 1 : 1;
    if (lo < 1) lo = 1;
    pr.lo = lo;
    pr.hi = hi;
    u64 Bmax = iroot(hi, n - 1), Bmin = 2;
    while (Bmin <= Bmax && pow_sat(Bmin, n) - 1 < lo) Bmin++;
    pr.Bmin = Bmin;
    pr.Bmax = Bmax;
    u64 nb = Bmax >= Bmin ? Bmax - Bmin + 1 : 0;
    pr.uoff.assign(nb + 1, 0);
    u64 tot = 0;
    for (u64 i = 0; i < nb; i++) {
        u64 B = Bmin + i;
        pr.uoff[i] = tot;
        tot += (u64)((u128)(B - 1) * pow_sat(B, pr.ud - 1));
    }
    pr.uoff[nb] = tot;
    pr.nunits = tot;
}

/*
 * build_tables -- the per-B data and the CConst of every (B, c) of the
 * search, computed as in bctx_set() of ../a171775.c.
 *
 * In:     pr (with at least one base B).
 * Out:    bi[B - Bmin] for every base; tab = all (B, c) constants, grouped by
 *         B; maxN = the largest c^k (decides 32- or 64-bit w).
 * Errors: exits with status 1 if c^(h+1) >= 2^62 or c^k >= 2^42, or if a
 *         weight bound fails (as bctx_set()).
 */
static void build_tables(const Prob &pr, std::vector<BInfo> &bi, std::vector<CConst> &tab, u64 &maxN)
{
    int n = pr.n, h = pr.h, D = pr.D, k = pr.k;
    u64 nb = pr.Bmax - pr.Bmin + 1;
    bi.resize(nb);
    tab.clear();
    maxN = 0;
    for (u64 i = 0; i < nb; i++) {
        u64 B = pr.Bmin + i;
        BInfo &b = bi[i];
        for (int j = 0; j < h; j++) b.w[j] = pow_sat(B, n - 1 - j) + pow_sat(B, j);
        if (n & 1) b.w[h] = pow_sat(B, h);
        u64 s = (u64)b.w[D - 1], wy = (u64)b.w[D - 2];
        u128 wx = b.w[D - 3];
        u64 dx = (u64)(b.w[D - 3] - (u128)(B - 1) * b.w[D - 2]);
        b.span_block = (u128)(B - 1) * (b.w[D - 3] + b.w[D - 2] + b.w[D - 1]);
        u128 su = 0;
        for (int j = pr.ud; j < D; j++) su += b.w[j];
        b.span_unit = (u128)(B - 1) * su;
        u128 Mlo = pow_sat(B, n - 1), Mhi = pow_sat(B, n) - 1;
        if (Mlo < pr.lo) Mlo = pr.lo;
        if (Mhi > pr.hi) Mhi = pr.hi;
        u64 cmin = iroot(Mlo, n - 1) + 1, cmax = iroot(Mhi, n - 2);
        if (cmax < cmin) cmax = cmin;
        b.cmin = cmin;
        b.cmax = cmax;
        b.off = tab.size();
        for (u64 c = cmin; c <= cmax; c++) {
            CConst K;
            memset(&K, 0, sizeof K);
            u128 P128 = pow_sat(c, h + 1), N128 = pow_sat(c, k);
            if (P128 >= ((u128)1 << 62) || N128 >= ((u128)1 << 42)) {
                fprintf(stderr, "c = %llu too large for 64-bit state (n = %d)\n", (unsigned long long)c, n);
                exit(1);
            }
            u64 P = (u64)P128, N = (u64)N128;
            if ((u128)(B - 1) * s >= P || wy >= P || dx >= P) {
                fprintf(stderr, "internal: weight bound violated\n");
                exit(1);
            }
            K.B = B;
            K.c = c;
            K.N = N;
            K.P = P;
            u64 sN = s % N, g = gcd64(sN, N), Np = N / g;   /* z s = t (mod N): see a171775.c, section 4 */
            u64 sp = Np > 1 ? (sN / g) % Np : 0, spp = sp;
            while (gcd64(spp % N, N) != 1) spp += Np;       /* a unit s'' = s/g (mod N') */
            K.g = g;
            K.Np = Np;
            K.inv = modinv64(spp % N, N);
            K.limit = g * B;
            K.Wy = mulmod64(wy % N, K.inv, N);
            K.Wx = mulmod64(dx % N, K.inv, N);
            u64 cj = 1;
            for (int j = 0; j < k; j++) { K.Wc[j] = mulmod64(cj % N, K.inv, N); cj *= c; }
            K.Rcross = P - (B - 1) * s;
            K.s = s;
            K.wy = wy;
            K.dx = dx;
            K.wx_lo = (u64)wx;
            K.wx_hi = (u64)(wx >> 64);
            if (N > maxN) maxN = N;
            tab.push_back(K);
        }
    }
}

/* unit_decode -- as in ../a171775.c: unit u -> base B (returned) and the
   unit's digits pref[0..ud-1].  u is not range-checked; without any base it
   returns Bmin and 1, 0, ... */
static u64 unit_decode(const Prob &pr, u64 u, u64 *pref)
{
    if (pr.Bmax < pr.Bmin) {              /* no bases at all: avoid the underflow below */
        pref[0] = 1;
        for (int i = 1; i < pr.ud; i++) pref[i] = 0;
        return pr.Bmin;
    }
    u64 lo = 0, hi = pr.Bmax - pr.Bmin;
    while (lo < hi) {
        u64 mid = (lo + hi + 1) / 2;
        if (pr.uoff[mid] <= u) lo = mid; else hi = mid - 1;
    }
    u64 B = pr.Bmin + lo, local = u - pr.uoff[lo];
    for (int i = pr.ud - 1; i >= 1; i--) { pref[i] = local % B; local /= B; }
    pref[0] = 1 + local;
    return B;
}

/*
 * gen_unit -- the items of unit u: do_unit() and do_block() of ../a171775.c,
 * but instead of running block_c() for each (block, c) it appends an Item.
 *
 * In:     pr; bi; the unit u; items (appended to).
 * Out:    returns the unit's tuples (B^2 per item), the CPU tool's count.
 * Errors: none.
 */
static u64 gen_unit(const Prob &pr, const std::vector<BInfo> &bi, u64 u, std::vector<Item> &items)
{
    u64 pref[MAXN];
    u64 B = unit_decode(pr, u, pref);
    const BInfo &b = bi[B - pr.Bmin];
    u128 Mu = 0;
    for (int i = 0; i < pr.ud; i++) Mu += (u128)pref[i] * b.w[i];
    if (Mu + b.span_unit < pr.lo || Mu > pr.hi) return 0;
    int nb = pr.bd - pr.ud;
    u64 tuples = 0;
    u64 nblk = nb ? B : 1;
    for (u64 d = 0; d < nblk; d++) {
        u128 Mb = Mu + (nb ? (u128)d * b.w[pr.ud] : 0);
        u128 Mmin = Mb, Mmax = Mb + b.span_block;
        if (Mmax < pr.lo || Mmin > pr.hi) continue;
        if (Mmin < pr.lo) Mmin = pr.lo;
        if (Mmax > pr.hi) Mmax = pr.hi;
        u64 clo = iroot(Mmin, pr.n - 1) + 1, chi = iroot(Mmax, pr.n - 2);
        if (clo < b.cmin) clo = b.cmin;
        if (chi > b.cmax) chi = b.cmax;
        for (u64 c = clo; c <= chi; c++) {
            Item it;
            it.Mlo = (u64)Mb;
            it.Mhi = (u64)(Mb >> 64);
            it.cc = (u32)(b.off + (c - b.cmin));
            it.pad = 0;
            items.push_back(it);
        }
        if (chi >= clo) tuples += (chi - clo + 1) * B * B;
    }
    return tuples;
}

/* ------------------------------------------------------------------ */
/* device code                                                         */
/* ------------------------------------------------------------------ */

/* is_pal_dev -- (device) is M an L-digit palindrome in base b?  As
   is_pal_base(); digits are kept as u32 (b < 2^32 here) and L <= MAXN + 1. */
__device__ bool is_pal_dev(u128 M, u64 b, int L)
{
    u32 d[MAXN + 2];
    int len = 0;
    while (M) {
        if (len >= L) return false;
        d[len++] = (u32)(M % b);
        M /= b;
    }
    if (len != L) return false;
    for (int i = 0; i < L / 2; i++)
        if (d[i] != d[L - 1 - i]) return false;
    return true;
}

/* emit -- (device) the full check of one hit, as in candidate(): if M is in
   [lo, hi] and an (n-1)-digit palindrome in base c, append (M, B, c) to out[]
   (an atomic slot counter; entries beyond outcap are counted but dropped, and
   the host stops with "output buffer overflow"). */
__device__ __forceinline__ void emit(u128 M, u128 lo, u128 hi, u64 B, u64 c, int n, Out *out, u32 *nout, u32 outcap)
{
    if (M < lo || M > hi) return;
    if (!is_pal_dev(M, c, n - 1)) return;
    u32 idx = atomicAdd(nout, 1u);
    if (idx < outcap) {
        out[idx].Mlo = (u64)M;
        out[idx].Mhi = (u64)(M >> 64);
        out[idx].B = (u32)B;
        out[idx].c = (u32)c;
    }
}

/* DevParams -- per search, the same for every launch, in constant memory:
   [lo, hi] as 64-bit halves, the output capacity and n.  Set once before the
   first launch (per-launch data such as output pointers are kernel arguments,
   so two streams can run at once). */
struct DevParams {                  /* per search (same for every launch), in constant memory */
    u64 lo_lo, lo_hi, hi_lo, hi_hi;
    u32 outcap;
    int n;
};
__constant__ DevParams dp;

/*
 * rare_path -- (device) the exact check of one (x, y) after the coarse test
 * fired: the hit code of block_c() in ../a171775.c, in 64/128-bit
 * arithmetic.  Kept out of line (__noinline__) so the hot loop stays small.
 *
 * In:     items, i (this thread's item), tab; x, y; the state w, wnext,
 *         last, rem of this step.
 * Out:    returns the number of z found (the "hits"); each goes through
 *         emit().  Region 0: z = w/g + i N' while rem + z s < P; region 1
 *         (not if last): z = w1/g + i N' from the first z with
 *         rem + z s >= P.
 * Errors: none.
 */
__device__ __noinline__ u32 rare_path(const Item *__restrict__ items, u32 i, const CConst *__restrict__ tab, u32 x,
                                      u32 y, u64 w, u64 wnext, bool last, u64 rem, Out *out, u32 *nout)
{
    const Item it = items[i];
    const CConst *Kc = tab + it.cc;
    const u64 B = Kc->B, c = Kc->c, P = Kc->P, s = Kc->s, g = Kc->g, Np = Kc->Np, N = Kc->N;
    const u64 limit = Kc->limit < N ? Kc->limit : N;
    const u128 Mblock = ((u128)it.Mhi << 64) | it.Mlo;
    const u128 wx = ((u128)Kc->wx_hi << 64) | Kc->wx_lo;
    const u128 M0 = Mblock + (u128)x * wx + (u128)y * Kc->wy;
    const u128 lo = ((u128)dp.lo_hi << 64) | dp.lo_lo, hi = ((u128)dp.hi_hi << 64) | dp.hi_lo;
    u32 hits = 0;
    if (w < limit && (g == 1 || w % g == 0)) {          /* solutions with q unchanged */
        for (u64 z = w / g; z < B; z += Np) {
            if (rem + z * s >= P) break;
            hits++;
            emit(M0 + (u128)z * s, lo, hi, B, c, dp.n, out, nout, dp.outcap);
        }
    }
    u64 w1 = w + wnext;
    if (w1 >= N) w1 -= N;
    if (!last && rem >= Kc->Rcross && w1 < limit && (g == 1 || w1 % g == 0)) {   /* solutions with q+1 */
        u64 z = w1 / g;
        u64 zs = (P - rem + s - 1) / s;            /* first z in the q+1 range */
        if (z < zs) z += ((zs - z + Np - 1) / Np) * Np;
        for (; z < B; z += Np) {
            hits++;
            emit(M0 + (u128)z * s, lo, hi, B, c, dp.n, out, nout, dp.outcap);
        }
    }
    return hits;
}

/*
 * scan_kernel<K, WT> -- (device) one thread = one item (block, c): all B^2
 * steps (x, y) of block_c() in ../a171775.c, with the state in registers
 * (theory, sections 2-3).  K = k, WT = the type of w (u32 when every N of
 * the search is below 2^31).
 *
 * In:     items[0..nitems-1]; tab (the CConst table); per launch, nhits (an
 *         accumulator) and the output buffer out[] with its counter nout.
 * Out:    adds the thread's hits to *nhits; survivors through emit().
 *         A thread stops early once q passes c^k - 1 (M >= c^(n-1)).
 * Errors: none.
 */
template <int K, typename WT>
__global__ void __launch_bounds__(128) scan_kernel(const Item *__restrict__ items, u32 nitems,
                                                   const CConst *__restrict__ tab, unsigned long long *nhits,
                                                   Out *__restrict__ out, u32 *nout)
{
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nitems) return;
    const Item it = items[i];
    const CConst *Kc = tab + it.cc;
    const u32 B = (u32)Kc->B, c = (u32)Kc->c;
    const u64 P = Kc->P, wy = Kc->wy, Pwy = P - wy, N64 = Kc->N;   /* rem >= P - w_y <=> rem + w_y >= P */
    const WT N = (WT)N64, Wy = (WT)Kc->Wy, Wx = (WT)Kc->Wx;
    const WT limit = (WT)(Kc->limit < N64 ? Kc->limit : N64);   /* w < N always, so clamping is exact */
    const WT D0y = N - Wy, D0x = N - Wx;   /* w steps by -W (mod N) without a carry of q */
    WT Wc[K];
#pragma unroll
    for (int j = 0; j < K; j++) Wc[j] = (WT)Kc->Wc[j];

    u64 rem;
    u32 E[K];
    WT w;
    {   /* initial state for (x, y) = (0, 0), as in block_c(); the only 128-bit divisions */
        const u128 Mblock = ((u128)it.Mhi << 64) | it.Mlo;
        rem = (u64)(Mblock % P);
        const u128 q128 = Mblock / P;
        if (q128 >= N64) return;                   /* M >= c^(n-1) already */
        u64 q = (u64)q128, T = 0;
#pragma unroll
        for (int j = K - 1; j >= 0; j--) { E[j] = (u32)(q % c); q /= c; }
#pragma unroll
        for (int j = K - 1; j >= 0; j--) T = T * c + E[j];
        const u64 MN = (u64)(Mblock % N64);
        w = (WT)(((u128)(T >= MN ? T - MN : T + N64 - MN) * Kc->inv) % N64);
    }

    WT wnext, a, D1y;       /* a = N - wnext; D1y = the w step of a y step that carries q */
    bool last, phase;       /* last: q = c^k - 1; phase: false until E[k-1] = c-1 */
    u32 cnt;                /* increments of q left until the next phase change */
    /* q -> q+1 with the carry stopping at digit m (the largest j with E[j] != c-1) changes T by
       c^(m+1) + c^m (mod c^k) for m < k-1 and by c^(k-1) for m = k-1; all in registers */
    /* RECOMPUTE(): find m (jj), set last, wnext, a and D1y.  The loops have
       constant bounds so E[] and Wc[] stay in registers (no dynamic indexing). */
#define RECOMPUTE()                                                                          \
    do {                                                                                     \
        int jj = -1;                                                                         \
        _Pragma("unroll") for (int j = 0; j < K; j++) if (E[j] != c - 1) jj = j;             \
        last = jj < 0;                                                                       \
        WT wn = 0;                                                                           \
        _Pragma("unroll") for (int j = 0; j < K; j++) {                                      \
            if (j == jj) {                                                                   \
                if (j == K - 1) wn = Wc[K - 1];                                              \
                else {                                                                       \
                    WT t2 = Wc[j + 1 < K ? j + 1 : K - 1] + Wc[j];                           \
                    wn = t2 >= N ? t2 - N : t2;                                              \
                }                                                                            \
            }                                                                                \
        }                                                                                    \
        wnext = wn;                                                                          \
        a = N - wn;                                                                          \
        D1y = wn >= Wy ? wn - Wy : wn + (N - Wy);                                            \
    } while (0)
    RECOMPUTE();
    if (E[K - 1] < c - 1) { phase = false; cnt = c - 1 - E[K - 1]; }
    else { phase = true; cnt = 1; }
    u32 hits = 0;

    /* CHECK(x, y): the coarse test of theory section 3 (branch-free), with
       the exact check out of line.  STEP(D0, D1, DR, PDR): M0 += DR (w_y or
       dx), with a carry into q when rem >= PDR = P - DR; w += D0, or D1 when
       q carries; then the phase bookkeeping of E[k-1] (block_c()'s slow path). */
#define CHECK(X, Y)                                                                          \
    if (__builtin_expect((w < limit) | ((WT)(w - a) < limit), 0))                            \
        hits += rare_path(items, i, tab, (X), (Y), (u64)w, (u64)wnext, last, rem, out, nout);
#define STEP(D0, D1, DR, PDR)                                                                \
    do {                                                                                     \
        const bool inc = rem >= (PDR);                                                       \
        rem = inc ? rem - (PDR) : rem + (DR);                                                \
        const WT wa = w + (inc ? (D1) : (D0));                                               \
        w = wa >= N ? wa - N : wa;                                                           \
        cnt -= inc;                                                                          \
        if (__builtin_expect(cnt == 0, 0)) {                                                 \
            if (!phase) {              /* E[k-1] reached c-1: the next increment carries */  \
                E[K - 1] = c - 1;                                                            \
                RECOMPUTE();                                                                 \
                phase = true;                                                                \
                cnt = 1;                                                                     \
            } else {                   /* carry; q = c^k means M >= c^(n-1) from here on */  \
                if (last) goto done;                                                         \
                E[K - 1] = 0;                                                                \
                bool carry = true;                                                           \
                _Pragma("unroll") for (int j = K - 2; j >= 0; j--) {                         \
                    if (carry) {                                                             \
                        E[j]++;                                                              \
                        if (E[j] == c) E[j] = 0;                                             \
                        else carry = false;                                                  \
                    }                                                                        \
                }                                                                            \
                if (carry) goto done;                                                        \
                RECOMPUTE();                                                                 \
                phase = false;                                                               \
                cnt = c - 1;                                                                 \
            }                                                                                \
        }                                                                                    \
    } while (0)

    for (u32 x = 0; x < B; x++) {
        for (u32 y = 0; y + 1 < B; y++) {          /* y steps: M0 += w_y */
            CHECK(x, y);
            STEP(D0y, D1y, wy, Pwy);
        }
        CHECK(x, B - 1);
        {                                          /* y wraps, x steps: M0 += dx */
            const u64 dx = Kc->dx;
            const WT D1x = wnext >= Wx ? wnext - Wx : wnext + (N - Wx);
            STEP(D0x, D1x, dx, P - dx);
        }
    }
done:
#undef RECOMPUTE
#undef CHECK
#undef STEP
    if (hits) atomicAdd(nhits, (unsigned long long)hits);
}

/* kern_t, pick_kernel -- the kernel instance for k (2..5) and the w width
   (narrow = 32-bit w, possible when every N of the search is < 2^31).
   Exits with status 1 for any other k. */
typedef void (*kern_t)(const Item *, u32, const CConst *, unsigned long long *, Out *, u32 *);

static kern_t pick_kernel(int k, bool narrow)
{
    switch (k) {
    case 2: return narrow ? scan_kernel<2, u32> : scan_kernel<2, u64>;
    case 3: return narrow ? scan_kernel<3, u32> : scan_kernel<3, u64>;
    case 4: return narrow ? scan_kernel<4, u32> : scan_kernel<4, u64>;
    case 5: return narrow ? scan_kernel<5, u32> : scan_kernel<5, u64>;
    }
    fprintf(stderr, "unsupported k = %d\n", k);
    exit(1);
}

/* ------------------------------------------------------------------ */
/* host side: survivors                                                */
/* ------------------------------------------------------------------ */

/* Stats -- the statistics of ../a171775.c, section 8 (tuples, hits, st2,
   canon, csum, sol, fail[]); add_stats() adds two of them. */
struct Stats {
    u64 tuples = 0, hits = 0, st2 = 0, canon = 0, csum = 0, sol = 0;
    u64 fail[MAXN + 1] = {0};
};

static void add_stats(Stats &t, const Stats &s)
{
    t.tuples += s.tuples; t.hits += s.hits; t.st2 += s.st2; t.canon += s.canon; t.csum += s.csum; t.sol += s.sol;
    for (int L = 0; L <= MAXN; L++) t.fail[L] += s.fail[L];
}

static std::mutex out_mtx;              /* one SOLUTION line at a time */
static FILE *sol_out = stdout;          /* a temporary file during the selftest */

/*
 * post_process -- the host part of the survivor check (candidate() in
 * ../a171775.c after the palindrome test): the canonical-base rule, the
 * checksum and the lengths n-2..3, with SOLUTION lines for the survivors
 * that pass them all.  The survivors are shared out over nthreads threads.
 *
 * In:     pr; o[0..no-1], the batch's survivors; st; nthreads.
 * Out:    st.st2 += no; canon, csum, sol and fail[] updated.
 * Errors: none.
 */
static void post_process(const Prob &pr, const Out *o, size_t no, Stats &st, int nthreads)
{
    std::atomic<size_t> next(0);
    std::vector<Stats> ts(nthreads);
    auto work = [&](int id) {
        Stats &s = ts[id];
        char buf[48];
        for (;;) {
            size_t i = next.fetch_add(1);
            if (i >= no) break;
            u128 M = ((u128)o[i].Mhi << 64) | o[i].Mlo;
            u64 B = o[i].B, c = o[i].c;
            int n = pr.n;
            if (find_pal_base(M, n - 1) != c) continue;
            if (find_pal_base(M, n) != B) continue;
            s.canon++;
            s.csum += surv_hash(M);
            int L;
            for (L = n - 2; L >= 3; L--)
                if (!find_pal_base(M, L)) break;
            if (L >= 3) { s.fail[L]++; continue; }
            s.sol++;
            std::lock_guard<std::mutex> lk(out_mtx);
            char buf2[48];
            fprintf(sol_out, "SOLUTION n=%d M=%s bases(L=2..%d)=%s", n, u128s(M, buf), n, u128s(M - 1, buf2));
            for (int LL = 3; LL <= n; LL++) fprintf(sol_out, ",%llu", (unsigned long long)find_pal_base(M, LL));
            fprintf(sol_out, "  [B=%llu c=%llu]\n", (unsigned long long)B, (unsigned long long)c);
            fflush(sol_out);
        }
    };
    std::vector<std::thread> th;
    for (int i = 0; i < nthreads; i++) th.emplace_back(work, i);
    for (auto &t : th) t.join();
    for (auto &s : ts) {
        st.canon += s.canon;
        st.csum += s.csum;
        st.sol += s.sol;
        for (int L = 0; L <= MAXN; L++) st.fail[L] += s.fail[L];
    }
    st.st2 += no;
}

/* ------------------------------------------------------------------ */
/* search driver                                                       */
/* ------------------------------------------------------------------ */

/* g_stop -- set by SIGINT/SIGTERM: run_search() stops issuing batches,
   finishes the ones in flight, saves the state and prints the summary. */
static volatile sig_atomic_t g_stop;
static void on_sig(int) { g_stop = 1; }

/*
 * load_state -- read a checkpoint (same format as ../a171775.c, so either
 * tool can resume the other's runs).
 *
 * In:     path; pr.
 * Out:    0 if the file cannot be opened (a fresh start); otherwise 1, the
 *         frontier and the saved totals (csum and failL= are optional).
 * Errors: exits with status 1 if the file cannot be parsed, is for another
 *         n, LO or HI, or has a frontier beyond the last unit.
 */
static int load_state(const char *path, const Prob &pr, u64 &frontier, Stats &st)
{
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char lo[64], hi[64];
    int n;
    unsigned long long fr, tu, hi_, s2, ca, so, cs = 0;
    int ok = fscanf(f, "n=%d lo=%63s hi=%63s frontier=%llu tuples=%llu hits=%llu st2=%llu canon=%llu sol=%llu",
                    &n, lo, hi, &fr, &tu, &hi_, &s2, &ca, &so) == 9;
    if (ok && fscanf(f, " csum=%llu", &cs) != 1) cs = 0;
    unsigned long long fl[MAXN + 1] = {0};
    if (ok) {                          /* optional " failL=count" fields */
        for (int i = 0; i <= MAXN; i++) {
            int LL;
            unsigned long long v;
            if (fscanf(f, " fail%d=%llu", &LL, &v) != 2) break;
            if (LL >= 0 && LL <= MAXN) fl[LL] = v;
        }
    }
    fclose(f);
    if (!ok) { fprintf(stderr, "cannot parse %s\n", path); exit(1); }
    if (n != pr.n || parse_or_die(lo) != pr.lo || parse_or_die(hi) != pr.hi) {
        fprintf(stderr, "state file %s is for a different search\n", path);
        exit(1);
    }
    if (fr > pr.nunits) {
        fprintf(stderr, "state file %s: frontier %llu is beyond the last unit (%llu units)\n", path, fr,
                (unsigned long long)pr.nunits);
        exit(1);
    }
    frontier = fr;
    st.tuples = tu; st.hits = hi_; st.st2 = s2; st.canon = ca; st.sol = so; st.csum = cs;
    for (int L = 0; L <= MAXN; L++) st.fail[L] = fl[L];
    return 1;
}

/*
 * save_state -- write a checkpoint line "n=.. lo=.. hi=.. frontier=..
 * tuples=.. hits=.. st2=.. canon=.. sol=.. csum=.. failL=.." to path.tmp and
 * rename it over path.
 * In:     path; pr; the frontier; t (exactly the units below the frontier,
 *         including a resumed state's totals).
 * Errors: if path.tmp cannot be written completely, prints the reason,
 *         removes it and keeps the previous checkpoint.
 */
static void save_state(const char *path, const Prob &pr, u64 frontier, const Stats &t)
{
    char tmp[4096], a[48], b[48];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "w");
    if (!f) { perror(tmp); return; }
    fprintf(f, "n=%d lo=%s hi=%s frontier=%llu tuples=%llu hits=%llu st2=%llu canon=%llu sol=%llu csum=%llu",
            pr.n, u128s(pr.lo, a), u128s(pr.hi, b), (unsigned long long)frontier, (unsigned long long)t.tuples,
            (unsigned long long)t.hits, (unsigned long long)t.st2, (unsigned long long)t.canon,
            (unsigned long long)t.sol, (unsigned long long)t.csum);
    for (int L = pr.n - 2; L >= 3; L--) fprintf(f, " fail%d=%llu", L, (unsigned long long)t.fail[L]);
    fprintf(f, "\n");
    int bad = ferror(f);
    if (fclose(f) != 0) bad = 1;
    if (bad) {                         /* keep the old checkpoint rather than a truncated one */
        fprintf(stderr, "cannot write %s; checkpoint not updated\n", tmp);
        remove(tmp);
        return;
    }
    rename(tmp, path);
}

/* print_summary -- the two final lines, in the format of ../a171775.c:
   "# n=.. [LO, HI] complete|INTERRUPTED: ..." and "# failed at length: ...". */
static void print_summary(const Prob &pr, const Stats &t, bool complete, double el)
{
    char a[48], b[48];
    printf("# n=%d [%s, %s] %s: %.1fs, tuples %llu, hits %llu, (n-1)-palindromes %llu, canonical %llu, "
           "checksum %016llx, solutions %llu\n",
           pr.n, u128s(pr.lo, a), u128s(pr.hi, b), complete ? "complete" : "INTERRUPTED", el,
           (unsigned long long)t.tuples, (unsigned long long)t.hits, (unsigned long long)t.st2,
           (unsigned long long)t.canon, (unsigned long long)t.csum, (unsigned long long)t.sol);
    printf("# failed at length:");
    for (int L = pr.n - 2; L >= 3; L--) printf(" L=%d:%llu", L, (unsigned long long)t.fail[L]);
    printf("\n");
    fflush(stdout);
}

/* Slot -- one of the two pipeline slots: a CUDA stream with pinned host and
   device buffers for one batch (items, survivors, their count, the hit
   count), the batch's units [u_begin, u_end) and tuples, and whether it is
   in flight (busy). */
struct Slot {
    cudaStream_t stream;
    Item *h_items, *d_items;
    Out *h_out, *d_out;
    u32 *d_nout, *h_nout;
    unsigned long long *d_hits, *h_hits;
    size_t nitems;
    u64 u_begin, u_end, tuples;
    bool busy;
};

/*
 * run_search -- the "search" command (and the selftest's searches).
 *
 * In:     n; lo, hi; state (checkpoint file or NULL); interval (seconds
 *         between progress lines); batch (items per batch, -b); hthreads
 *         (host threads for post_process); result (may be NULL); quiet (no
 *         progress or summary).
 * Out:    SOLUTION lines on sol_out, progress on stderr and the summary on
 *         stdout unless quiet; the checkpoint if requested; *result = the
 *         totals including a resumed state's.  Returns 0 when complete (also
 *         for a range without units), 3 if interrupted, 1 if n < 7.
 * How:    build all (B, c) constants and copy them to the GPU; then loop:
 *         build the next batch (whole units, up to `batch` items), finish the
 *         slot's previous batch (wait, copy the survivors back,
 *         post_process), launch the new batch in that slot, and switch slots;
 *         when everything is issued or g_stop is set, finish both slots,
 *         oldest first.  Batches finish in issue order, so done_units is a
 *         frontier with exactly known statistics.
 * Errors: exits with status 1 on CUDA errors (CK), table limits
 *         (build_tables), a unit with more items than a batch holds ("raise
 *         -b"), or more than 2^20 survivors in a batch ("lower -b").
 */
static int run_search(int n, u128 lo, u128 hi, const char *state, double interval, size_t batch, int hthreads,
                      Stats *result, int quiet)
{
    char a[48], b[48];
    Prob pr;
    prob_init(pr, n, lo, hi);
    if (pr.D < 4) { fprintf(stderr, "n = %d: use the CPU tool (n >= 7 here)\n", n); return 1; }
    if (pr.nunits == 0) {              /* e.g. HI < 2^(n-1), or LO > HI */
        Stats none;
        if (!quiet) {
            fprintf(stderr, "gpu search n=%d [%s, %s]: no base has %d-digit numbers in this range, nothing to search\n",
                    n, u128s(lo, a), u128s(hi, b), n);
            print_summary(pr, none, true, 0.0);
        }
        if (result) *result = none;
        return 0;
    }
    std::vector<BInfo> bi;
    std::vector<CConst> tab;
    u64 maxN;
    build_tables(pr, bi, tab, maxN);
    bool narrow = maxN < ((u64)1 << 31);   /* 32-bit w is enough for every (B, c) */
    kern_t kern = pick_kernel(pr.k, narrow);
    CConst *d_tab;
    CK(cudaMalloc(&d_tab, tab.size() * sizeof(CConst)));
    CK(cudaMemcpy(d_tab, tab.data(), tab.size() * sizeof(CConst), cudaMemcpyHostToDevice));

    Stats st, base;
    u64 frontier = 0;
    if (state && load_state(state, pr, frontier, base) && !quiet)
        fprintf(stderr, "resuming at unit %llu of %llu\n", (unsigned long long)frontier, (unsigned long long)pr.nunits);
    if (!quiet)
        fprintf(stderr, "gpu search n=%d [%s, %s]: B in [%llu, %llu], %llu units, k=%d, %zu (B,c) constants, %s w\n", n,
                u128s(lo, a), u128s(hi, b), (unsigned long long)pr.Bmin, (unsigned long long)pr.Bmax,
                (unsigned long long)pr.nunits, pr.k, tab.size(), narrow ? "32-bit" : "64-bit");

    const u32 outcap = 1u << 20;           /* survivors per batch */
    {
        DevParams h;
        h.lo_lo = (u64)pr.lo; h.lo_hi = (u64)(pr.lo >> 64);
        h.hi_lo = (u64)pr.hi; h.hi_hi = (u64)(pr.hi >> 64);
        h.outcap = outcap;
        h.n = n;
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpyToSymbol(dp, &h, sizeof h));
    }
    Slot sl[2];
    for (auto &s : sl) {
        CK(cudaStreamCreate(&s.stream));
        CK(cudaMallocHost(&s.h_items, batch * sizeof(Item)));
        CK(cudaMalloc(&s.d_items, batch * sizeof(Item)));
        CK(cudaMallocHost(&s.h_out, outcap * sizeof(Out)));
        CK(cudaMalloc(&s.d_out, outcap * sizeof(Out)));
        CK(cudaMalloc(&s.d_nout, sizeof(u32)));
        CK(cudaMallocHost(&s.h_nout, sizeof(u32)));
        CK(cudaMalloc(&s.d_hits, sizeof(unsigned long long)));
        CK(cudaMallocHost(&s.h_hits, sizeof(unsigned long long)));
        s.busy = false;
    }
    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);
    double t0 = now_sec(), tlast = t0, tsave = t0;
    u64 unext = frontier, done_units = frontier;
    std::vector<Item> items;
    items.reserve(batch + 4096);
    int cur = 0;
    /* finish(s): wait for slot s, copy its survivors back and post-process
       them, add its counts; the frontier moves to the end of its batch */
    auto finish = [&](Slot &s) {
        CK(cudaStreamSynchronize(s.stream));
        u32 no = *s.h_nout;
        if (no > outcap) {
            fprintf(stderr, "output buffer overflow (%u > %u); lower -b\n", no, outcap);
            exit(1);
        }
        if (no) {
            CK(cudaMemcpyAsync(s.h_out, s.d_out, (size_t)no * sizeof(Out), cudaMemcpyDeviceToHost, s.stream));
            CK(cudaStreamSynchronize(s.stream));
        }
        post_process(pr, s.h_out, no, st, hthreads);
        st.hits += *s.h_hits;
        st.tuples += s.tuples;
        if (s.u_end > done_units) done_units = s.u_end;
        s.busy = false;
    };
    for (;;) {
        /* build the next batch on the host while the other slot runs */
        items.clear();
        u64 ub = unext, tuples = 0;
        while (unext < pr.nunits && items.size() < batch) {
            size_t before = items.size();
            u64 t = gen_unit(pr, bi, unext, items);
            if (items.size() > batch) {
                if (before == 0) {
                    fprintf(stderr, "unit %llu has %zu items > batch %zu; raise -b\n", (unsigned long long)unext,
                            items.size(), batch);
                    exit(1);
                }
                items.resize(before);      /* this unit goes into the next batch */
                break;
            }
            tuples += t;
            unext++;
        }
        Slot &s = sl[cur];
        if (s.busy) finish(s);                  /* the older of the two batches in flight */
        if (ub < unext) {                       /* launch the new batch in this slot */
            memcpy(s.h_items, items.data(), items.size() * sizeof(Item));
            s.nitems = items.size();
            s.u_begin = ub;
            s.u_end = unext;
            s.tuples = tuples;
            CK(cudaMemcpyAsync(s.d_items, s.h_items, s.nitems * sizeof(Item), cudaMemcpyHostToDevice, s.stream));
            CK(cudaMemsetAsync(s.d_nout, 0, sizeof(u32), s.stream));
            CK(cudaMemsetAsync(s.d_hits, 0, sizeof(unsigned long long), s.stream));
            if (s.nitems) {
                u32 threads = 128, blocks = (u32)((s.nitems + threads - 1) / threads);
                kern<<<blocks, threads, 0, s.stream>>>(s.d_items, (u32)s.nitems, d_tab, s.d_hits, s.d_out,
                                                       s.d_nout);
                CK(cudaGetLastError());
            }
            CK(cudaMemcpyAsync(s.h_nout, s.d_nout, sizeof(u32), cudaMemcpyDeviceToHost, s.stream));
            CK(cudaMemcpyAsync(s.h_hits, s.d_hits, sizeof(unsigned long long), cudaMemcpyDeviceToHost, s.stream));
            s.busy = true;
        }
        cur ^= 1;
        bool all_issued = unext >= pr.nunits;
        if (all_issued || g_stop) {             /* drain, oldest first */
            if (sl[cur].busy) finish(sl[cur]);
            if (sl[cur ^ 1].busy) finish(sl[cur ^ 1]);
        }
        double t = now_sec();
        bool fin = all_issued && !sl[0].busy && !sl[1].busy;
        if (!quiet && (t - tlast >= interval || fin || g_stop)) {
            u64 pref[MAXN];
            u64 Bc = unit_decode(pr, done_units < pr.nunits ? done_units : pr.nunits - 1, pref);
            fprintf(stderr, "[%7.0fs] frontier %llu/%llu (B=%llu)  %.3e tuples/s  st2 %llu canon %llu sol %llu\n",
                    t - t0, (unsigned long long)done_units, (unsigned long long)pr.nunits, (unsigned long long)Bc,
                    st.tuples / (t - t0 > 0 ? t - t0 : 1), (unsigned long long)(base.st2 + st.st2),
                    (unsigned long long)(base.canon + st.canon), (unsigned long long)(base.sol + st.sol));
            tlast = t;
        }
        if (state && (t - tsave >= 60 || fin || g_stop)) {
            Stats all = st;                /* the finished batches: exactly the units below fr */
            add_stats(all, base);
            u64 fr = done_units;           /* = the first unit of the oldest batch in flight */
            for (auto &x : sl) if (x.busy && x.u_begin < fr) fr = x.u_begin;
            save_state(state, pr, fr, all);
            tsave = t;
        }
        if (fin || g_stop) break;
    }
    double el = now_sec() - t0;
    Stats all = st;
    add_stats(all, base);
    bool complete = done_units >= pr.nunits;
    if (!quiet) print_summary(pr, all, complete, el);
    for (auto &s : sl) {
        cudaStreamDestroy(s.stream);
        cudaFreeHost(s.h_items); cudaFree(s.d_items); cudaFreeHost(s.h_out); cudaFree(s.d_out);
        cudaFree(s.d_nout); cudaFreeHost(s.h_nout); cudaFree(s.d_hits); cudaFreeHost(s.h_hits);
    }
    cudaFree(d_tab);
    if (result) *result = all;
    return complete ? 0 : 3;
}

/* ------------------------------------------------------------------ */

/*
 * cmd_selftest -- complete searches over [1, a(n)] for n = 7, 8, 9: each
 * must find exactly one solution, a(n), with the CPU tool's tuples, hits,
 * st2, canonical count and checksum.
 * In: batch, hthreads (as for search).  Out: one line per n; returns 0 if all
 * agree, 1 otherwise.  Errors: as run_search(); tmpfile() is not checked.
 */
static int cmd_selftest(size_t batch, int hthreads)
{
    /* expected statistics from the CPU tool (a171775 search n 1 a(n)) */
    struct { int n; const char *hi; u64 tuples, hits, st2, canon, csum; } ref[] = {
        {7, "2^30", 1663504ULL, 25964ULL, 639, 639, 0x048e08fa43539a8eULL},
        {8, "2^42", 45519324ULL, 319612ULL, 4725, 4724, 0x13747afea2768e7aULL},
        {9, "2^56", 90554063896ULL, 2449285ULL, 19140, 19140, 0x380d9946bb59f02eULL},
    };
    int bad = 0;
    for (auto &r : ref) {
        Stats s;
        FILE *mem = tmpfile();
        FILE *save = sol_out;
        sol_out = mem;
        double t0 = now_sec();
        run_search(r.n, 1, parse_or_die(r.hi), NULL, 1e9, batch, hthreads, &s, 1);
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
        char buf[48];
        int ok = nsol == 1 && best == parse_or_die(r.hi) && s.tuples == r.tuples && s.hits == r.hits &&
                 s.st2 == r.st2 && s.canon == r.canon && (r.csum == 0 || s.csum == r.csum);
        printf("  n=%d [1, %s]: %.2fs, %.3e tuples (%.3e/s), hits %llu, st2 %llu, canonical %llu, checksum %016llx, "
               "%d solution(s), smallest %s  %s\n",
               r.n, r.hi, now_sec() - t0, (double)s.tuples, s.tuples / (now_sec() - t0), (unsigned long long)s.hits,
               (unsigned long long)s.st2, (unsigned long long)s.canon, (unsigned long long)s.csum, nsol,
               best ? u128s(best, buf) : "none", ok ? "ok" : "MISMATCH");
        bad += !ok;
    }
    printf(bad ? "SELFTEST FAILED\n" : "selftest passed\n");
    return bad ? 1 : 0;
}

/* main -- parse -S, -i, -b, -t and the command; set the blocking-sync device
   flag (the host sleeps while waiting for the GPU).  Unknown commands print
   the usage and return 1; bad numbers exit with status 1. */
int main(int argc, char **argv)
{
    const char *state = NULL;
    double interval = 10;
    size_t batch = 1 << 20;
    int hthreads = (int)std::thread::hardware_concurrency();
    char *pos[8];
    int npos = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-S") && i + 1 < argc) state = argv[++i];
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) interval = atof(argv[++i]);
        else if (!strcmp(argv[i], "-b") && i + 1 < argc) batch = (size_t)atof(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) hthreads = atoi(argv[++i]);
        else if (npos < 8) pos[npos++] = argv[i];
    }
    if (hthreads < 1) hthreads = 1;
    if (batch < 1) batch = 1;              /* -b 0 would never issue a batch */
    CK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "device: %s, %d SMs\n", prop.name, prop.multiProcessorCount);
    if (npos >= 1 && !strcmp(pos[0], "selftest")) return cmd_selftest(batch, hthreads);
    if (npos == 4 && !strcmp(pos[0], "search")) {
        int n = atoi(pos[1]);
        return run_search(n, parse_or_die(pos[2]), parse_or_die(pos[3]), state, interval, batch, hthreads, NULL, 0);
    }
    fprintf(stderr, "usage: a171775_cuda selftest | search n LO HI [-S state] [-i sec] [-b items] [-t host threads]\n");
    return 1;
}
