/*
 * a171775_2d.c -- faster exhaustive search for OEIS A171775 (the "swapped
 * 2D" method).
 *
 *   a(n) = smallest M such that for every k = 2..n there is a base b_k in
 *   which M is a k-digit palindrome.  Known: a(7..9) = 2^30, 2^42, 2^56;
 *   a(10) = 2^72 (a171775.c, confirmed here) and a(11) = 2^90 (this
 *   program, Sep 2026); conjecture a(n) = 2^((n-1)(n-2)).
 *
 * Read the theory of operations in a171775.c first.  This program keeps its
 * reduction (a survivor is an n-digit palindrome in some base B and an
 * (n-1)-digit palindrome in some base c > B; the lengths n-2..3 are tested
 * afterwards), its canonical-base rule and its checksum, and swaps the roles
 * of the two lengths.
 *
 * Layout of this file
 *   integer helpers        as in a171775.c, plus u128d, surv_hash, now_sec
 *   problem and constants  prob_t, bconst_t, cctx_t, wstat_t; prob_init,
 *                          bconst_set (per pair (c, B)), cctx_set (per c)
 *   survivors              report_solution, record_pair, survivor
 *   one block, one B       block_B (the hot loop), do_block, unit_decode,
 *                          do_unit
 *   checkpoints, summary   cctx_free, add_stats, load_state, save_state,
 *                          print_summary
 *   threaded search        finish_unit, search_totals, worker, unit_weight,
 *                          run_search
 *   selftest               naive references, compare_block, cmd_selftest,
 *                          cmd_paranoid (-DPARANOID builds)
 *   command line           usage, main
 *
 * The file is also a library.  cuda/a171775_2d_cuda.cu (C++, compiled with
 * A171775_2D_LIB and A171775_2D_NO_DRIVER) and metal/a171775_metal.m (with
 * A171775_2D_LIB) include it and replace only the inner loop of block_B:
 * they copy fields of bconst_t and cctx_t to the GPU, call survivor(), and
 * read and write the same state files.  Changes to those structures, to the
 * statistics or to the state-file format must be made in the ports too.
 * A171775_2D_NO_DRIVER leaves out the threaded driver, the selftest and main
 * (everything from search_t on, and stdatomic.h, which C++ lacks);
 * A171775_2D_LIB leaves out main.  This is also why the file compiles as
 * C++ (casts on malloc results).
 *
 * ========================================================================
 * THEORY OF OPERATIONS
 * ========================================================================
 *
 * 1. Why swap.  a171775.c enumerates the n-digit palindromes and solves for
 *    one digit, which saves a factor B.  For odd n the (n-1)-digit
 *    palindromes are the smaller set (below 2^90 there are 7.0e15 ten-digit
 *    ones and 6.1e16 eleven-digit ones), and solving for two digits at once
 *    saves a factor c^2.  So this program enumerates the Le = (n-1)-digit
 *    palindromes M in base c (the enumeration base) and, for every check
 *    base B < c in which M can have Lc = n digits (B^(n-1) <= M < B^n),
 *    imposes part of the n-digit condition on two digits through a lookup
 *    table.
 *
 * 2. The enumerated palindromes.  Free digits e_0..e_{De-1} (De =
 *    ceil(Le/2), e_0 >= 1) with weights w_i = c^(Le-1-i) + c^i, and
 *    w_mid = c^(Le/2) for the middle digit of odd Le.  The three innermost
 *    free digits are x = e_{De-3}, y = e_{De-2}, z = e_{De-1}; the others
 *    form the block:
 *
 *        M = M1 + y w_y + z w_z,   M1 = Mblock + x w_x.
 *
 *    The work is nested like this:
 *      unit    c and the first ud = max(1, De-4) free digits: (c, e_0) for
 *              n <= 11, (c, e_0, e_1) for n = 12 and 13; units are
 *              numbered base by base and handed to threads
 *      block   do_unit loops over the last block digit if the unit does not
 *              fix it
 *      B       every check base for which some M of the block can have n
 *              digits (do_block)
 *      x       c steps per (block, B): block_B, the hot loop
 *      (y, z)  all solutions of a congruence, found by one table lookup
 *              (two near a carry of q)
 *
 * 3. The congruence.  For each pair (c, B), k is the largest k <= 3 with
 *    max(span, w_x) < P = B^(n-k), where span = (c-1)(w_y + w_z) is the
 *    range of y w_y + z w_z.  Then over the (y, z) box, q = floor(M / P)
 *    (the k leading base-B digits of M) takes only the two values
 *    q1 = floor(M1 / P) and q1+1, and x -> x+1 raises q1 by at most one.
 *    The k trailing digits must mirror the leading ones: M = T(q) (mod N),
 *    N = B^k, T = the digits of q reversed.  Let g = gcd(w_z, N),
 *    N' = N/g, u = the inverse mod N of a unit s'' = w_z/g (mod N'),
 *    R = w_y u mod N and W = (T - M1) u mod N.  As w_z u = g (mod N),
 *
 *      y w_y + z w_z = T - M1 (mod N)  <=>  g | V(y)  and  z = V(y)/g + i N',
 *                                           where V(y) = (W - y R) mod N.
 *
 * 4. The lookup.  So the y that give some z < c are those whose
 *    v = y R mod N has v = W (mod g) and (W div g - v div g) mod N' < c.
 *    With key(v) = (v mod g) N' + v div g these keys form one window of
 *    width min(c, N') inside the block of keys of W's residue class (the
 *    window can wrap around the end of that block).  The table holds
 *    key(y R mod N) for y = 0..c-1, sorted, with y, plus about c buckets on
 *    the high bits of the key, so a window costs O(1) plus its entries.
 *    Every entry found gives the solutions z = (W div g - v div g) mod N'
 *    + i N' < c; each z is then kept only if M falls in the region (q1 or
 *    q1+1) the lookup was made for.
 *
 * 5. Incremental update.  As x runs, M1 grows by w_x: W drops by w_x u
 *    (Wx), rem = M1 mod P grows by w_x, and a pass over P raises q1 by one,
 *    which moves T by B^(k-1), or by B^(m+1) + B^m (mod B^k) when the carry
 *    stops at digit m (as in a171775.c).  Each x costs one lookup, or two
 *    when the box reaches into q1+1 (rem >= Rc = P - span).
 *
 * 6. Filtering a match.  A match (a "hit") has digits 0..k-1 of M in base
 *    B mirroring digits n-1..n-k.  The next pair is compared before the full
 *    test: digit k from M mod B^(k+1), kept in two parts (mod B^k and div
 *    B^k) as M1's part plus table entries for y w_y and z w_z; digit n-1-k
 *    is the top digit of r = M mod P, from a double-precision quotient
 *    r / B^(n-1-k), replaced by an exact division when the quotient is
 *    within 1e-9 of an integer.  The rounding error is below B 2^-50 < 6e-11
 *    for B < 65536, so the result is always exact.  About one hit in B
 *    passes ("f1") and gets the full n-digit test.
 *
 * 7. Survivors (survivor()) get the same treatment as in a171775.c: the
 *    canonical pair (smallest B and c), the checksum, the lengths n-2..3,
 *    fail[] and SOLUTION lines.  So st2, canonical, checksum and fail[]
 *    agree with a171775.c over the same range (the selftest checks this for
 *    n = 8, 9 and, with "full", 10), while steps, lookups, hits and f1
 *    belong to this method:
 *      steps     (block, B, x) steps, c per block and B
 *      lookups   table lookups (1 or 2 per step)
 *      hits      (y, z) solutions of the congruence in their region
 *      f1        hits that pass the next-digit filter
 *      st2       (M, B, c) with M in [LO, HI] an n-digit palindrome in base
 *                B (and, by construction, an (n-1)-digit one in base c)
 *
 * 8. Threads, checkpoints, signals: as in a171775.c (theory, section 9):
 *    each unit's statistics are committed in frontier order, so every
 *    checkpoint (written every 60 s and at the end, with fail[]) holds
 *    exactly the units below its frontier, and a resumed run gives identical
 *    results however the earlier run ended.  The GPU hosts finish their
 *    batches in order, which gives them the same property.  -E u stops
 *    before unit u and -u n c HI prints the first unit of base c (unit
 *    numbers as for LO = 1); with a state file whose frontier is set to a
 *    start unit, any slice of units can be run, e.g. to compare the CPU and
 *    GPU tools on the same slice.  A state file whose frontier lies beyond
 *    the -E unit is refused.
 *
 * 9. Limits: n = 8..20; c <= 65535 (y is stored in 16 bits); B^k < 2^32
 *    (32-bit keys); per thread, the tables of one c take about 22 c bytes per
 *    check base (a few MB at c = 1024).  A range in which no base c >= 3 has
 *    (n-1)-digit numbers is reported as "nothing to search" with an empty
 *    summary.
 *
 * Usage
 *   a171775_2d selftest [full]        naive comparisons; n = 8, 9 [, 10] full searches vs a171775.c
 *   a171775_2d search n LO HI [-t T] [-S state] [-i sec] [-v] [-E end_unit]
 *   a171775_2d -u n c HI              print the index of the first unit of base c
 *   a171775_2d paranoid [reps]        (build with -DPARANOID) state checked at every step
 * Numbers accept 2^90, 1e27, 2^90-1, ...  Options as in a171775.c.
 * Exit status: 0 = done, 3 = search interrupted, 1 = usage, number or
 * state-file error, a fatal limit, or a failed selftest.  A PARANOID build
 * calls abort() on a mismatch.
 */
#ifndef _GNU_SOURCE                /* the including GPU host may have defined it */
#define _GNU_SOURCE
#endif
#include <math.h>
#include <pthread.h>
#include <signal.h>
#ifndef A171775_2D_NO_DRIVER       /* C11 atomics: not available to the C++ (CUDA) host */
#include <stdatomic.h>
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef unsigned __int128 u128;    /* M and anything that scales like it */
typedef uint64_t u64;
typedef int64_t i64;
typedef uint32_t u32;              /* table keys and bucket starts */
typedef uint16_t u16;              /* table y values, high parts of digit sums */

#define MAXN 24                    /* largest length n the arrays allow (search: n <= 20) */
#define MAXK 4                     /* size of the digit arrays E[] and Wc[] */
#define KMAX 3                     /* largest k (base-B digit pairs imposed by the lookup) */
#define U128_MAX (~(u128)0)        /* also the saturation value of pow_sat() */

/* ------------------------------------------------------------------ */
/* integer helpers (as in a171775.c)                                   */
/* ------------------------------------------------------------------ */

/* pow_sat -- b^e, or U128_MAX if that does not fit in 128 bits (e <= 0 gives
   1).  No errors; a saturated value compares above every M handled here. */
static u128 pow_sat(u64 b, int e)
{
    u128 r = 1;
    for (int i = 0; i < e; i++) {
        if (b && r > U128_MAX / b) return U128_MAX;
        r *= b;
    }
    return r;
}

/*
 * iroot -- floor(x^(1/k)) for k >= 2, exact: a double estimate corrected by
 * integer steps until r^k <= x < (r+1)^k.  0 for x = 0.  No errors for the
 * inputs used here (a root above 1.8e19, i.e. k = 2 with x > 3.2e38, would
 * make the correction loop extremely long).
 */
static u64 iroot(u128 x, int k)
{
    if (x == 0) return 0;
    double a = pow((double)x, 1.0 / k);
    u64 r = a >= 1.8e19 ? (u64)1.8e19 : (u64)a;   /* (u64) of a larger double is undefined */
    if (r == 0) r = 1;
    while (r > 1 && pow_sat(r, k) > x) r--;
    while (pow_sat(r + 1, k) <= x) r++;
    return r;
}

/* u128s -- decimal digits of x in buf (at least 40 bytes; callers use 48);
   returns buf.  No errors (buf size not checked). */
static char *u128s(u128 x, char *buf)
{
    char t[48];
    int i = 0;
    do { t[i++] = (char)('0' + (int)(x % 10)); x /= 10; } while (x);
    for (int j = 0; j < i; j++) buf[j] = t[i - 1 - j];
    buf[i] = 0;
    return buf;
}

/*
 * parse_term -- parse one term DIGITS, DIGITS^EXP or DIGITSeEXP at *ps.
 * Out:    1, the value in *out and *ps advanced; 0 on a syntax error (no
 *         digits, no exponent digits, base above 2^64-1 before '^').
 * Errors: overflow is not detected (decimal and 'e' values wrap mod 2^128,
 *         b^e saturates).
 */
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

/*
 * parse_or_die -- parse a number expression: terms (parse_term) joined by
 * '+' or '-', left to right, e.g. "2^90-1".
 * In:     s0, a NUL-terminated string.
 * Out:    the value (arithmetic wraps mod 2^128).
 * Errors: on a syntax error or trailing text prints "bad number: <s0>" and
 *         exits with status 1.
 */
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

/* gcd64 -- greatest common divisor; gcd(a, 0) = a.  No errors. */
static u64 gcd64(u64 a, u64 b)
{
    while (b) { u64 t = a % b; a = b; b = t; }
    return a;
}

/*
 * modinv64 -- the inverse of a mod m (m >= 1), in [0, m), by the extended
 * Euclidean algorithm in signed 128-bit arithmetic.  Returns 0 if
 * gcd(a, m) != 1 (and for m = 1).  m = 0 divides by zero; bconst_set() only
 * passes units, so the failure value never occurs.
 */
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

/* mulmod64 -- a b mod m through a 128-bit product (no overflow; m != 0). */
static inline u64 mulmod64(u64 a, u64 b, u64 m) { return (u64)((u128)a * b % m); }

/* mod128_64 -- M mod b (b != 0), avoiding the slow 128-bit division when b <
   2^32 by reducing 32 bits at a time (see a171775.c).  b = 0 divides by zero. */
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

/* u128d -- a 128-bit value as a double (relative error a few 2^-53; used
   only where an exact check follows or the error is accounted for). */
static inline double u128d(u128 x) { return (double)(u64)(x >> 64) * 18446744073709551616.0 + (double)(u64)x; }

/*
 * is_pal_base -- 1 if M has exactly L base-b digits (L <= 130) that read the
 * same in both directions, else 0 (also for b < 2 or M = 0).  Stops as soon
 * as M has more than L digits.  No errors.
 */
static int is_pal_base(u128 M, u64 b, int L)
{
    u64 d[130];
    int len = 0;
    if (b < 2) return 0;
    if ((M >> 64) == 0) {
        u64 x = (u64)M;
        while (x) { if (len >= L) return 0; d[len++] = x % b; x /= b; }
    } else {
        u128 x = M;
        while (x) { if (len >= L) return 0; d[len++] = (u64)(x % b); x /= b; }
    }
    if (len != L) return 0;
    for (int i = 0; i < L / 2; i++)
        if (d[i] != d[L - 1 - i]) return 0;
    return 1;
}

/*
 * find_pal_base -- the smallest base in which M is an L-digit palindrome
 * (L >= 3), or 0 if there is none; for L = 2 just M-1 (M = "11" in base M-1,
 * capped at 2^64-1), 0 if M < 3.
 * How:    scans b with b^(L-1) <= M < b^L; per base the last digit must be
 *         the (nonzero) leading digit, pre-checked with a double quotient
 *         (safe: see find_pal_base in a171775.c), then all digits.
 * Errors: none.  The cost is the number of bases scanned: large for L = 3
 *         and big M (up to 2^45 bases at M ~ 2^90); in a search only
 *         survivors that pass all longer lengths get there.
 */
static u64 find_pal_base(u128 M, int L)
{
    if (L < 2 || M == 0) return 0;
    if (L == 2) {                       /* only existence matters */
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

/* surv_hash -- the survivor hash; the checksum is its sum over the canonical
   survivors mod 2^64.  Must stay identical to a171775.c and the GPU ports,
   or checksums stop being comparable.  No errors. */
static inline u64 surv_hash(u128 M)
{
    u64 a = (u64)M * 0x9E3779B97F4A7C15ULL, b = (u64)(M >> 64) * 0xC2B2AE3D27D4EB4FULL;
    a ^= a >> 29;
    return a * 0xBF58476D1CE4E5B9ULL + b;
}

/* now_sec -- monotonic time in seconds (elapsed times, rates, ETA). */
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------ */
/* problem and per-base constants                                      */
/* ------------------------------------------------------------------ */

/*
 * prob_t -- one search: the order n, the range [lo, hi] and its work units.
 *
 *   n, Lc        the order; Lc = n is the length checked in base B
 *   Le           n-1, the length of the enumerated palindromes (base c)
 *   he, De       Le/2 mirrored digit pairs, De = ceil(Le/2) free digits
 *   bd           De-3 block digits (the free digits before x, y, z)
 *   ud           max(1, bd-1) block digits fixed by a unit; the rest (0 or
 *                1 digit) is looped over in do_unit
 *   lo, hi       the range of M (lo >= 1)
 *   cmin, cmax   the enumeration bases with Le-digit numbers in [lo, hi]
 *                (c >= 3, since some B with 2 <= B < c is needed)
 *   uoff         uoff[c-cmin] = the first unit of base c; uoff[cmax-cmin+1]
 *                = nunits; base c has (c-1) c^(ud-1) units
 *   nunits       the total number of units (run_search may lower it, -E)
 */
typedef struct {
    int n, Le, Lc, he, De, bd, ud; /* check length Lc = n, enumerated length Le = n-1 */
    u128 lo, hi;
    u64 cmin, cmax;                /* enumeration bases */
    u64 *uoff;                     /* uoff[c-cmin] = first unit of base c */
    u64 nunits;
} prob_t;

/*
 * bconst_t -- the constants and the lookup table of one pair (c, B), built
 * by bconst_set() (theory, sections 3-6).  The GPU ports copy most fields.
 *
 *   B, k       the check base and the number of digit pairs imposed
 *   N, Bk      both B^k (N as the modulus, Bk for splitting digit sums)
 *   g, Np      g = gcd(w_z, N), N' = N/g
 *   u          the inverse mod N of a unit s'' = w_z/g (mod N')
 *   lim        min(g c, N), the width of the window of matching y R mod N
 *              (copied into the GPU ports' pair structs, but nothing reads it
 *              at present; block_B works with the keys instead)
 *   R          w_y u mod N
 *   Wx         w_x u mod N (W drops by it at each x step)
 *   Wc[j]      B^j u mod N, for the change of T when q1 is incremented
 *   P, Rc      P = B^(n-k); the (y, z) box reaches q1+1 iff rem >= Rc =
 *              P - span
 *   BK1        B^(k+1); oxK, oyK, ozK = w_x, w_y, w_z mod B^(k+1)
 *   Ptop, invPtop  B^(n-1-k) and 1/B^(n-1-k): digit n-1-k of M is
 *              floor((M mod P) / Ptop)
 *   val, ys    the table: val[i] = key(y R mod N) in increasing order, ys[i]
 *              = its y (c entries)
 *   bst, sh, nb  buckets: bst[b] = first i with val[i] >> sh >= b, for
 *              b = 0..nb (nb about c, sh chosen so that N >> sh <= c)
 *   ay_lo, ay_hi, az_lo, az_hi  t w_y and t w_z mod B^(k+1) for t < c, split
 *              into (mod B^k) and (div B^k) parts, for digit k of M
 *   ox_lo, ox_hi  w_x mod B^(k+1), split the same way
 * The arrays live in the owning cctx_t's mem block.
 */
typedef struct {                   /* per (c, B) */
    u64 B, k, N, g, Np, u, lim, R, Wx, Wc[MAXK];
    u128 P, Rc;                    /* P = B^(n-k); crossing iff rem >= Rc = P - span */
    u64 BK1, oxK, oyK, ozK;        /* B^(k+1) and w_x, w_y, w_z mod B^(k+1) */
    u64 Bk;                        /* B^k */
    u128 Ptop;                     /* B^(n-1-k): digit n-1-k = floor(r / Ptop) */
    double invPtop;
    u32 *val;                      /* table: key(y R mod N) sorted, with y */
    u16 *ys;
    u32 *bst;                      /* bucket starts, nb + 1 entries */
    int sh;
    u32 nb;
    u32 *ay_lo, *az_lo;            /* t w_y, t w_z mod B^(k+1) for t < c, split as (mod B^k, div B^k) */
    u16 *ay_hi, *az_hi;
    u64 ox_lo, ox_hi;              /* w_x mod B^(k+1), split the same way */
} bconst_t;

/*
 * cctx_t -- everything that depends on the enumeration base c, plus the
 * bconst_t of every check base B that can occur with c.  One per thread;
 * cctx_set() rebuilds it when the thread's next unit has another c (units
 * come in order of c, so that is rare).  cctx_free() releases it.
 *
 *   c            the enumeration base
 *   w[i]         the weights w_0..w_{De-1}
 *   wx, wy, wz   the weights of x, y, z
 *   span         (c-1)(w_y + w_z): the range of y w_y + z w_z
 *   span_block   (c-1)(w_x + w_y + w_z): a block's M lie in
 *                [Mblock, Mblock + span_block]
 *   span_unit    the same for all digits a unit does not fix
 *   Bbase, Bcount  the check bases of the table: tab[B - Bbase], Bcount
 *                entries (Bcount = 0 if no B is possible)
 *   tab, tabcap  the bconst_t table and its allocated length
 *   mem, memcap  one allocation holding all lookup tables of tab[]
 */
typedef struct {                   /* per enumeration base c */
    u64 c;
    u128 w[MAXN];
    u128 wx, wy, wz, span, span_block, span_unit;
    u64 Bbase, Bcount;
    bconst_t *tab;
    size_t tabcap;
    char *mem;                     /* table storage */
    size_t memcap;
} cctx_t;

/* pair_t -- a pair (M, B) recorded by the selftest hooks (wstat_t.record). */
typedef struct { u128 M; u64 B; } pair_t;

/*
 * wstat_t -- statistics (theory, section 7) plus test hooks.  A worker
 * collects one unit's statistics in a wstat_t of its own and hands them to
 * finish_unit(); add_stats() adds two of them.
 * The GPU ports fill the same fields.
 *
 *   steps, lookups, hits, f1, st2   see section 7
 *   canon, csum, sol, fail[]        as in a171775.c
 *   record   selftest hook: 0 = normal; 1 = store every (M, B) with M an
 *            n-digit palindrome in base B instead of processing it; 2 = store
 *            every hit with B^(n-1) <= M < B^n (every k-digit match)
 *   pairs, np, pcap   the stored pairs, their count and the capacity
 *   verbose  -v: print DEEP lines
 *   pchk, pz (PARANOID) brute-force checked x steps and the (y, z) matches
 *            among them
 */
typedef struct {
    const prob_t *pr;
    u64 steps, lookups, hits, f1, st2, canon, csum, sol;
    u64 fail[MAXN + 1];
    int record;                    /* test hook: 1 = full n-digit palindromes, 2 = all k-digit matches */
    pair_t *pairs;
    size_t np, pcap;
    int verbose;
#ifdef PARANOID
    u64 pchk, pz;                  /* brute-force steps checked, (y,z) hits checked */
#endif
} wstat_t;

static pthread_mutex_t out_lock = PTHREAD_MUTEX_INITIALIZER;  /* one output line at a time */
static FILE *sol_out;              /* SOLUTION/DEEP lines: stdout, or a temporary file in the selftest */

/*
 * prob_init -- set up the problem (n, [lo, hi]) and number its work units.
 *
 * In:     pr (overwritten); n (8..20); lo, hi (lo < 1 is raised to 1).
 * Out:    *pr (see prob_t); pr->uoff is allocated, the caller frees it.
 * How:    the enumeration bases are the c >= 3 with c^(Le-1) <= hi and
 *         c^Le - 1 >= lo; base c has (c-1) c^(ud-1) units (leading digit
 *         1..c-1, other unit digits 0..c-1), numbered in order of c.
 * Errors: none reported; calloc failure is not checked.  No base c gives
 *         nunits = 0 (see the header, section 9).  n < 8 makes the digit
 *         split meaningless; main() refuses it.
 */
static void prob_init(prob_t *pr, int n, u128 lo, u128 hi)
{
    memset(pr, 0, sizeof *pr);
    pr->n = n;
    pr->Lc = n;
    pr->Le = n - 1;
    pr->he = pr->Le / 2;
    pr->De = (pr->Le + 1) / 2;
    pr->bd = pr->De - 3;
    pr->ud = pr->bd - 1 >= 1 ? pr->bd - 1 : 1;
    if (lo < 1) lo = 1;
    pr->lo = lo;
    pr->hi = hi;
    /* enumeration bases c with c^(Le-1) <= hi and c^Le - 1 >= lo */
    u64 cmax = iroot(hi, pr->Le - 1), cmin = 3;
    while (cmin <= cmax && pow_sat(cmin, pr->Le) - 1 < lo) cmin++;
    pr->cmin = cmin;
    pr->cmax = cmax;
    u64 nc = cmax >= cmin ? cmax - cmin + 1 : 0;
    pr->uoff = (u64 *)calloc(nc + 1, sizeof(u64));
    u64 tot = 0;
    for (u64 i = 0; i < nc; i++) {        /* prefix sum of the units per base */
        u64 c = cmin + i;
        pr->uoff[i] = tot;
        tot += (u64)((u128)(c - 1) * pow_sat(c, pr->ud - 1));
    }
    pr->uoff[nc] = tot;
    pr->nunits = tot;
}

/* u32cmp_pairs -- qsort() comparator for the packed table entries
   (key << 16 | y) of bconst_set(): orders by key, then by y.  (The entries
   are u64; the name refers to the 32-bit keys.) */
static int u32cmp_pairs(const void *a, const void *b)
{
    const u64 *x = (const u64 *)a, *y = (const u64 *)b;   /* packed (val << 16 | y) */
    return *x < *y ? -1 : *x > *y;
}

/*
 * bconst_set -- the constants and the lookup table of the pair (c, B).
 *
 * In:     K (overwritten); pr; cc (the context of c: weights and spans);
 *         the check base B < c; *memp, a pointer into cc->mem with room for
 *         this pair's arrays (reserved by cctx_set).
 * Out:    *K filled in (see bconst_t); *memp advanced past the arrays used
 *         (each array is padded to 8 bytes).
 * How:    k = the largest k <= KMAX with max(span, w_x) < B^(n-k); then N,
 *         g, N', u, R, Wx, Wc[] as in a171775.c's bctx_set (u from the
 *         smallest unit s'' = w_z/g + j N'); the digit-k tables; and the
 *         key table: key(y R mod N) for y < c, sorted, plus the buckets.
 * Errors: exits with status 1 if no k >= 1 works ("internal: no usable k";
 *         would mean c is far too large for B) or if B^k >= 2^32 ("B^k too
 *         large for 32-bit tables").  malloc failure is not checked.
 */
static void bconst_set(bconst_t *K, const prob_t *pr, const cctx_t *cc, u64 B, char **memp)
{
    const u64 c = cc->c;
    const int Lc = pr->Lc;
    u128 need = cc->span > cc->wx ? cc->span : cc->wx;   /* both must be < P (theory, section 3) */
    int k = 0;
    while (k + 1 <= KMAX && need < pow_sat(B, Lc - (k + 1))) k++;
    if (k < 1) {
        fprintf(stderr, "internal: no usable k for c=%llu B=%llu\n", (unsigned long long)c, (unsigned long long)B);
        exit(1);
    }
    memset(K, 0, sizeof *K);
    K->B = B;
    K->k = (u64)k;
    u128 N128 = pow_sat(B, k);
    if (N128 >= ((u128)1 << 32)) {
        fprintf(stderr, "B^k too large for 32-bit tables (B=%llu k=%d)\n", (unsigned long long)B, k);
        exit(1);
    }
    u64 N = (u64)N128;
    K->N = N;
    K->Bk = N;
    K->P = pow_sat(B, Lc - k);
    K->Rc = K->P - cc->span;
    /* y w_y + z w_z = t (mod N): with u w_z = g (mod N) this becomes z g = t u - y R */
    u64 sN = (u64)(cc->wz % N), g = gcd64(sN, N), Np = N / g;
    u64 sp = Np > 1 ? (sN / g) % Np : 0, spp = sp;   /* w_z/g mod N' */
    while (gcd64(spp % N, N) != 1) spp += Np;        /* lift to a unit mod N */
    K->g = g;
    K->Np = Np;
    K->u = modinv64(spp % N, N);
    K->lim = g * c < N ? g * c : N;
    K->R = mulmod64((u64)(cc->wy % N), K->u, N);
    K->Wx = mulmod64((u64)(cc->wx % N), K->u, N);
    u64 bj = 1;
    for (int j = 0; j < k; j++) { K->Wc[j] = mulmod64(bj % N, K->u, N); bj *= B; }
    K->BK1 = (u64)pow_sat(B, k + 1);
    K->oxK = (u64)(cc->wx % K->BK1);
    K->oyK = (u64)(cc->wy % K->BK1);
    K->ozK = (u64)(cc->wz % K->BK1);
    K->Ptop = pow_sat(B, Lc - 1 - k);
    K->invPtop = 1.0 / u128d(K->Ptop);
    /* table: carve the arrays out of the caller's block (sizes as in cctx_set's "need") */
    K->val = (u32 *)*memp; *memp += c * sizeof(u32);
    K->ys = (u16 *)*memp; *memp += ((c * sizeof(u16) + 7) & ~(size_t)7);
    int sh = 0;
    while ((N >> sh) > c) sh++;            /* about c buckets for c keys */
    K->sh = sh;
    K->nb = (u32)(((N - 1) >> sh) + 1);
    K->bst = (u32 *)*memp; *memp += ((K->nb + 1) * sizeof(u32) + 7) & ~(size_t)7;
    K->ay_lo = (u32 *)*memp; *memp += c * sizeof(u32);
    K->az_lo = (u32 *)*memp; *memp += c * sizeof(u32);
    K->ay_hi = (u16 *)*memp; *memp += ((c * sizeof(u16) + 7) & ~(size_t)7);
    K->az_hi = (u16 *)*memp; *memp += ((c * sizeof(u16) + 7) & ~(size_t)7);
    /* digit-k tables: t w mod B^(k+1) = hi B^k + lo, hi < B fits in 16 bits */
    for (u64 t = 0; t < c; t++) {
        u64 vy = (u64)((u128)t * K->oyK % K->BK1), vz = (u64)((u128)t * K->ozK % K->BK1);
        K->ay_lo[t] = (u32)(vy % K->Bk); K->ay_hi[t] = (u16)(vy / K->Bk);
        K->az_lo[t] = (u32)(vz % K->Bk); K->az_hi[t] = (u16)(vz / K->Bk);
    }
    K->ox_lo = K->oxK % K->Bk;
    K->ox_hi = K->oxK / K->Bk;
    /* key(v) = (v mod g) N' + v div g: the y whose v = y R mod N gives a solution z < c for target W
       are exactly those with v = W (mod g) and (W div g - v div g) mod N' < c, i.e. a window of width
       min(c, N') inside one residue class, whatever g is */
    u64 *tmp = (u64 *)malloc(c * sizeof(u64));
    u64 v = 0;                             /* v = y R mod N, by repeated addition */
    for (u64 y = 0; y < c; y++) {
        u64 key = (v % g) * Np + v / g;    /* < N < 2^32 */
        tmp[y] = (key << 16) | y;          /* y < 2^16 (c <= 65535) */
        v += K->R;
        if (v >= N) v -= N;
    }
    qsort(tmp, c, sizeof(u64), u32cmp_pairs);
    for (u64 i = 0; i < c; i++) { K->val[i] = (u32)(tmp[i] >> 16); K->ys[i] = (u16)(tmp[i] & 0xffff); }
    free(tmp);
    u32 b = 0;                             /* bst[b] = first i with val[i] >> sh >= b */
    for (u64 i = 0; i < c; i++) {
        u32 bi = K->val[i] >> sh;
        while (b <= bi) K->bst[b++] = (u32)i;
    }
    while (b <= K->nb) K->bst[b++] = (u32)c;
}

/*
 * cctx_set -- build the context of the enumeration base c: weights, spans,
 * the range of check bases B and every pair's constants and tables.
 *
 * In:     cc (reused: tab and mem grow as needed, the rest is overwritten);
 *         pr; the base c.
 * Out:    *cc (see cctx_t).  The check bases are those with some Le-digit
 *         base-c number M in [lo, hi] (M in [max(c^(Le-1), lo),
 *         min(c^Le - 1, hi)]) that has n digits in base B, and B < c.
 *         Bcount = 0 if there is none (do_block then skips the base).
 * Errors: exits with status 1 if c > 65535 ("c too large"), or through
 *         bconst_set().  malloc failures are not checked.  The tables of
 *         all pairs take about Bcount (22 c + 64) bytes.
 */
static void cctx_set(cctx_t *cc, const prob_t *pr, u64 c)
{
    int Le = pr->Le, he = pr->he, De = pr->De, Lc = pr->Lc;
    cc->c = c;
    for (int i = 0; i < he; i++) cc->w[i] = pow_sat(c, Le - 1 - i) + pow_sat(c, i);
    if (Le & 1) cc->w[he] = pow_sat(c, he);
    cc->wx = cc->w[De - 3];
    cc->wy = cc->w[De - 2];
    cc->wz = cc->w[De - 1];
    cc->span = (u128)(c - 1) * (cc->wy + cc->wz);
    cc->span_block = (u128)(c - 1) * (cc->wx + cc->wy + cc->wz);
    u128 su = 0;
    for (int i = pr->ud; i < De; i++) su += cc->w[i];
    cc->span_unit = (u128)(c - 1) * su;
    if (c > 65535) { fprintf(stderr, "c too large\n"); exit(1); }   /* y must fit in 16 bits */
    /* B range for this c: B^(n-1) <= M < B^n for some M in [Mlo, Mhi], and B < c */
    u128 Mlo = pow_sat(c, Le - 1), Mhi = pow_sat(c, Le) - 1;
    if (Mlo < pr->lo) Mlo = pr->lo;
    if (Mhi > pr->hi) Mhi = pr->hi;
    u64 Bmin = iroot(Mlo, Lc) + 1, Bmax = iroot(Mhi, Lc - 1);
    if (Bmin < 2) Bmin = 2;
    if (Bmax >= c) Bmax = c - 1;
    if (Bmax < Bmin) { cc->Bbase = Bmin; cc->Bcount = 0; return; }
    cc->Bbase = Bmin;
    cc->Bcount = Bmax - Bmin + 1;
    if (cc->Bcount > cc->tabcap) {
        free(cc->tab);
        cc->tabcap = cc->Bcount;
        cc->tab = (bconst_t *)malloc(cc->tabcap * sizeof(bconst_t));
    }
    /* per pair: val 4c, ys 2c (+8 padding), bst 4(c+2) (+16), ay/az 12c (+32) bytes */
    size_t need = cc->Bcount * (c * 4 + c * 2 + 8 + (c + 2) * 4 + 16 + c * 12 + 32) + 64;
    if (need > cc->memcap) {
        free(cc->mem);
        cc->memcap = need;
        cc->mem = (char *)malloc(need);
    }
    char *mp = cc->mem;
    for (u64 i = 0; i < cc->Bcount; i++) bconst_set(&cc->tab[i], pr, cc, Bmin + i, &mp);
}

/* ------------------------------------------------------------------ */
/* survivors                                                           */
/* ------------------------------------------------------------------ */

/*
 * report_solution -- print a SOLUTION line: M, a base for each length 2..n
 * (M-1 for L = 2, the smallest base otherwise) and the pair (B, c).
 * In: st (for n), M, B, c.  Out: one line on sol_out, under out_lock,
 * flushed.  No errors; repeating find_pal_base() for L = 3 can take seconds.
 */
static void report_solution(wstat_t *st, u128 M, u64 B, u64 c)
{
    const prob_t *pr = st->pr;
    char buf[48];
    u64 bases[MAXN + 1];
    for (int L = 3; L <= pr->n; L++) bases[L] = find_pal_base(M, L);
    pthread_mutex_lock(&out_lock);
    fprintf(sol_out, "SOLUTION n=%d M=%s", pr->n, u128s(M, buf));
    fprintf(sol_out, " bases(L=2..%d)=%s", pr->n, u128s(M - 1, buf));
    for (int L = 3; L <= pr->n; L++) fprintf(sol_out, ",%llu", (unsigned long long)bases[L]);
    fprintf(sol_out, "  [B=%llu c=%llu]\n", (unsigned long long)B, (unsigned long long)c);
    fflush(sol_out);
    pthread_mutex_unlock(&out_lock);
}

/* record_pair -- append (M, B) to st->pairs (selftest record modes), growing
   the array by doubling.  realloc failure is not checked. */
static void record_pair(wstat_t *st, u128 M, u64 B)
{
    if (st->np == st->pcap) {
        st->pcap = st->pcap ? 2 * st->pcap : 1024;
        st->pairs = (pair_t *)realloc(st->pairs, st->pcap * sizeof(pair_t));
    }
    st->pairs[st->np].M = M;
    st->pairs[st->np].B = B;
    st->np++;
}

static void survivor(wstat_t *st, u128 M, u64 B, u64 c);

/*
 * survivor -- process a survivor: M in [lo, hi] is an n-digit palindrome in
 * base B and an (n-1)-digit one in base c (theory, section 7).  Also called
 * by the GPU hosts for the survivors their kernels find.
 *
 * In:     st; M; B; c.
 * Out:    st2++; with record = 1 the pair is stored and nothing else
 *         happens.  Otherwise only the canonical pair (c and B the smallest
 *         bases of their lengths) continues: canon++, csum += surv_hash(M),
 *         lengths n-2 down to 3 tested; the first failing length is counted
 *         in fail[L] (a DEEP line with -v if L <= n-3); if none fails, sol++
 *         and a SOLUTION line.
 * Errors: none.
 */
static void survivor(wstat_t *st, u128 M, u64 B, u64 c)
{
    const prob_t *pr = st->pr;
    const int n = pr->n;
    st->st2++;
    if (st->record == 1) { record_pair(st, M, B); return; }
    if (find_pal_base(M, n - 1) != c) return;          /* canonical: smallest bases of both lengths */
    if (find_pal_base(M, n) != B) return;
    st->canon++;
    st->csum += surv_hash(M);
    for (int L = n - 2; L >= 3; L--) {
        if (!find_pal_base(M, L)) {
            st->fail[L]++;
            if (st->verbose && L <= n - 3) {
                char buf[48];
                pthread_mutex_lock(&out_lock);
                fprintf(sol_out, "DEEP n=%d M=%s fails at L=%d [B=%llu c=%llu]\n", n, u128s(M, buf), L,
                        (unsigned long long)B, (unsigned long long)c);
                fflush(sol_out);
                pthread_mutex_unlock(&out_lock);
            }
            return;
        }
    }
    st->sol++;
    report_solution(st, M, B, c);
}

/* ------------------------------------------------------------------ */
/* one block, one check base B                                         */
/* ------------------------------------------------------------------ */

#ifdef PARANOID
/* u64cmp -- (PARANOID) qsort() comparator for u64 values. */
static int u64cmp(const void *a, const void *b)
{
    u64 x = *(const u64 *)a, y = *(const u64 *)b;
    return x < y ? -1 : x > y;
}

/* T_of -- (PARANOID) T(q): the k base-B digits of q (q < B^k) in reverse
   order, read as a number.  No errors. */
static u64 T_of(u64 q, u64 B, int k)
{
    u64 E[MAXK], T = 0;
    for (int j = k - 1; j >= 0; j--) { E[j] = q % B; q /= B; }
    for (int j = k - 1; j >= 0; j--) T = T * B + E[j];
    return T;
}
#endif

/*
 * block_B -- the hot loop: all (x, y, z) of one block for one check base B
 * (theory, sections 3-6).
 *
 * In:     st      the calling thread's statistics
 *         cc      the context of the enumeration base c (weights)
 *         K       the constants and tables of (c, B)
 *         Mblock  the value of the block digits (x = y = z = 0)
 *         inside  1 if the whole block lies in [lo, hi] (skips the range
 *                 test per hit)
 * Out:    for every x, looks up the (y, z) whose M = Mblock + x w_x + y w_y
 *         + z w_z satisfies M = T(floor(M / P)) (mod B^k); each such hit is
 *         counted, range-checked, filtered by digit pair (k, n-1-k), tested
 *         as an n-digit palindrome in B and handed to survivor().  In record
 *         mode 2 every hit with n digits in B is stored instead.  Adds the
 *         number of lookups to st->lookups.  Stops early once M >= B^n for
 *         all remaining x.
 * Errors: none (a PARANOID build aborts on any mismatch).
 *
 * State for the current x, M1 = Mblock + x w_x:
 *   rem          M1 mod P (128 bits: P can exceed 2^64)
 *   q1           floor(M1 / P) < B^k, as its base-B digits E[0..k-1]
 *   W            (T(q1) - M1) u mod N
 *   wnext, last  (T(q1+1) - T(q1)) u mod N, and last = 1 if q1 = B^k - 1
 *   mk_lo, mk_hi M1 mod B^(k+1) = mk_hi B^k + mk_lo, for digit k of M
 *   M1           itself, to form M on a hit
 */
static void block_B(wstat_t *st, const cctx_t *cc, const bconst_t *K, u128 Mblock, int inside)
{
    const prob_t *pr = st->pr;
    const int n = pr->n;
    const u64 c = cc->c, B = K->B, N = K->N, g = K->g, Np = K->Np, Wx = K->Wx;
    const int k = (int)K->k;
    const u128 P = K->P, Rc = K->Rc, wx = cc->wx, wy = cc->wy, wz = cc->wz;
    const u64 BK1 = K->BK1, Bk = K->Bk, ox_lo = K->ox_lo, ox_hi = K->ox_hi;
    const u128 Ptop = K->Ptop;
    const double invPtop = K->invPtop;
    const u32 *ay_lo = K->ay_lo, *az_lo = K->az_lo;
    const u16 *ay_hi = K->ay_hi, *az_hi = K->az_hi;
    const u32 *val = K->val, *bst = K->bst;
    const u16 *ys = K->ys;
    const int sh = K->sh;

    /* initial state for x = 0, M1 = Mblock; after this the x loop divides only
       in lookups (by g) and on hits */
    u128 q128 = Mblock / P;
    if (q128 >= N) return;                 /* M >= B^n for the whole block */
    u128 rem = Mblock % P;
    u64 q = (u64)q128, E[MAXK];
    for (int j = k - 1; j >= 0; j--) { E[j] = q % B; q /= B; }   /* E[0] = leading digit */
    u64 T = 0;
    for (int j = k - 1; j >= 0; j--) T = T * B + E[j];          /* T(q1): digits reversed */
    u64 MN = (u64)(Mblock % N);
    u64 W = mulmod64(T >= MN ? T - MN : T + N - MN, K->u, N);
    u64 mk_lo, mk_hi;                      /* M1 mod B^(k+1) = mk_hi B^k + mk_lo */
    {
        u64 m = (u64)(Mblock % BK1);
        mk_lo = m % Bk;
        mk_hi = m / Bk;
    }
    u128 M1 = Mblock;
    u64 wnext;
    int last;
    /* q -> q+1 with the carry stopping at digit m changes T by B^(m+1) + B^m (mod B^k), or by B^(k-1) */
    /* RECOMPUTE_NEXT(): from the digits E[], find that m (jj) and set wnext =
       (change of T) u mod N from Wc[]; all digits B-1 means q1 = B^k - 1:
       last = 1 (no q1+1), wnext = 0.  Called after every carry of q1. */
#define RECOMPUTE_NEXT()                                                        \
    do {                                                                        \
        int jj = k - 1;                                                         \
        while (jj >= 0 && E[jj] == B - 1) jj--;                                 \
        last = jj < 0;                                                          \
        if (jj == k - 1) wnext = K->Wc[k - 1];                                  \
        else if (jj >= 0) {                                                     \
            wnext = K->Wc[jj + 1] + K->Wc[jj];                                  \
            if (wnext >= N) wnext -= N;                                         \
        } else wnext = 0;                                                       \
    } while (0)
    RECOMPUTE_NEXT();

    u64 nlook = 0;                         /* lookups, added to st->lookups at the end */
#ifdef PARANOID
    /* the (y, z) found for one x (packed y << 16 | z) and the brute-force set */
    u64 *pfound = malloc((c * c + 1) * sizeof(u64)), *want = malloc((c * c + 1) * sizeof(u64));
    const int pcap = (int)(c * c + 1);
#endif
    for (u64 x = 0; x < c; x++) {
#ifdef PARANOID
        {   /* recompute the whole state for M1 from scratch; abort on any difference */
            u128 M1p = Mblock + (u128)x * wx;
            u128 qp = M1p / P;
            int bad = 0;
            if (qp >= N) bad |= 1;
            else {
                if (M1p % P != rem) bad |= 2;
                u64 qq = (u64)qp, Et[MAXK];
                for (int j = k - 1; j >= 0; j--) { Et[j] = qq % B; qq /= B; }
                for (int j = 0; j < k; j++) if (Et[j] != E[j]) bad |= 4;
                u64 Tt = T_of((u64)qp, B, k), MNt = (u64)(M1p % N);
                if (mulmod64(Tt >= MNt ? Tt - MNt : Tt + N - MNt, K->u, N) != W) bad |= 8;
                if ((u64)qp + 1 < N) {
                    u64 T1 = T_of((u64)qp + 1, B, k);
                    if (mulmod64(T1 >= Tt ? T1 - Tt : T1 + N - Tt, K->u, N) != wnext) bad |= 16;
                    if (last) bad |= 32;
                } else if (!last) bad |= 32;
                if ((u64)(M1p % BK1) != mk_lo + mk_hi * Bk || M1p != M1) bad |= 64;
            }
            if (bad) {
                char buf[48];
                fprintf(stderr, "PARANOID state mismatch %#x: n=%d c=%llu B=%llu x=%llu M1=%s\n", bad, st->pr->n,
                        (unsigned long long)c, (unsigned long long)B, (unsigned long long)x, u128s(M1p, buf));
                abort();
            }
        }
        int npf = 0;
#endif
        /* the two lookups: target T(q1) (region r < P) and, if the box reaches it, T(q1+1) */
        for (int reg = 0; reg < 2; reg++) {
            u64 Wt;
            if (reg == 0) Wt = W;
            else {
                if (last || rem < Rc) break;
                Wt = W + wnext;
                if (Wt >= N) Wt -= N;
            }
            nlook++;
            /* matching keys: residue class r = Wt mod g, i.e. keys r N' + m, with
               (mW - m) mod N' < c for mW = Wt div g: the window m in
               [mW - c + 1, mW] (mod N'), or the whole class if c >= N' */
            u64 r = 0, mW = Wt;
            if (g > 1) { r = Wt % g; mW = Wt / g; }
            const u64 base = r * Np;
            u64 m_lo, m_hi = mW;
            int wrap;
            if (c >= Np) { m_lo = 0; m_hi = Np - 1; wrap = 0; }
            else {
                m_lo = mW + 1 >= c ? mW + 1 - c : mW + 1 + Np - c;
                wrap = m_lo > m_hi;                /* window wraps: [m_lo, N'-1] and [0, mW] */
            }
            for (int part = 0; part <= wrap; part++) {
                u64 a = base + (part == 0 ? m_lo : 0), b = base + ((wrap && part == 0) ? Np - 1 : m_hi);
                u32 i = bst[a >> sh], e = bst[(b >> sh) + 1];   /* buckets covering keys [a, b] */
                for (; i < e; i++) {
                    u64 v = val[i];
                    if (v < a) continue;
                    if (v > b) break;
                    u64 m = v - base;              /* (y R mod N) div g */
                    u64 y = ys[i];
                    /* z = (mW - m) mod N' + i N' < c, kept if M lies in this lookup's region */
                    for (u64 z = mW >= m ? mW - m : mW + Np - m; z < c; z += Np) {
                        u128 d = (u128)y * wy + (u128)z * wz;
                        u128 r2 = rem + d;         /* M - q1 P */
                        if (reg == 0) {
                            if (r2 >= P) break;    /* M already in q1+1 (and further z too) */
                        } else {
                            if (r2 < P) continue;  /* not yet in q1+1 */
                            r2 -= P;
                        }
                        /* now r2 = M mod P */
#ifdef PARANOID
                        if (npf < pcap) pfound[npf++] = (y << 16) | z;
#endif
                        /* same steps as candidate() in a171775.c: count, range, next digit pair, full test */
                        st->hits++;
                        u128 M = M1 + d;
                        if (!inside && (M < pr->lo || M > pr->hi)) continue;
                        if (st->record == 2) {
                            if (pow_sat(B, n - 1) <= M && M < pow_sat(B, n)) record_pair(st, M, B);
                            continue;
                        }
                        /* M mod B^(k+1) = (M1 + y w_y + z w_z) mod B^(k+1), in (div, mod B^k)
                           parts: three lo parts < 3 B^k, so at most two carries */
                        u64 dlo = mk_lo + ay_lo[y] + az_lo[z], dhi2 = mk_hi + ay_hi[y] + az_hi[z];
                        if (dlo >= Bk) { dlo -= Bk; dhi2++; }
                        if (dlo >= Bk) { dlo -= Bk; dhi2++; }
                        while (dhi2 >= B) dhi2 -= B;           /* digit k of M */
                        double dq = u128d(r2) * invPtop;       /* digit n-1-k of M: top digit of r2 */
                        u64 dtop = (u64)dq;
                        double fr = dq - (double)dtop;
                        /* error < B 2^-50 << 1e-9: only near an integer can the floor be off */
                        if (fr < 1e-9 || fr > 1 - 1e-9) dtop = (u64)(r2 / Ptop);
                        if (dhi2 != dtop) continue;
                        st->f1++;
                        if (!is_pal_base(M, B, n)) continue;
                        survivor(st, M, B, c);
                    }
                }
            }
        }
#ifdef PARANOID
        /* compare the (y, z) found above with all solutions in the box, for
           every x when c <= 48 and every 29th x otherwise */
        if (c <= 48 || x % 29 == 0) {       /* brute force over the (y, z) box */
            u128 M1 = Mblock + (u128)x * wx;
            int nw = 0;
            for (u64 y = 0; y < c; y++)
                for (u64 z = 0; z < c; z++) {
                    u128 M = M1 + (u128)y * wy + (u128)z * wz;
                    u128 qz = M / P;
                    if (qz >= N) continue;
                    if ((u64)(M % N) == T_of((u64)qz, B, k) && nw < pcap) want[nw++] = (y << 16) | z;
                }
            /* both lists are small; compare as sorted sets */
            qsort(pfound, npf, sizeof(u64), u64cmp);
            int same = nw == npf;
            for (int i = 0; same && i < nw; i++) same = want[i] == pfound[i];
            if (!same) {
                char buf[48];
                fprintf(stderr, "PARANOID (y,z) mismatch: n=%d c=%llu B=%llu x=%llu M1=%s fast %d brute %d\n",
                        st->pr->n, (unsigned long long)c, (unsigned long long)B, (unsigned long long)x,
                        u128s(M1, buf), npf, nw);
                abort();
            }
            st->pchk++;
            st->pz += nw;
        }
#endif
        /* advance x: M1 += w_x (w_x < P, so q1 grows by at most one) */
        rem += wx;
        if (rem >= P) {                        /* q1 -> q1+1 */
            rem -= P;
            if (last) break;                   /* q1 reaches B^k: M >= B^n from here on */
            W += wnext;                        /* T(q1) -> T(q1+1) */
            if (W >= N) W -= N;
            int j = k - 1;                     /* increment the digits of q1 */
            while (j >= 0) {
                E[j]++;
                if (E[j] < B) break;
                E[j] = 0;
                j--;
            }
            RECOMPUTE_NEXT();
        }
        W = W >= Wx ? W - Wx : W + N - Wx;     /* W -= w_x u (mod N) */
        mk_lo += ox_lo;                        /* M1 mod B^(k+1) += w_x mod B^(k+1) */
        mk_hi += ox_hi;
        if (mk_lo >= Bk) { mk_lo -= Bk; mk_hi++; }
        if (mk_hi >= B) mk_hi -= B;
        M1 += wx;
    }
#undef RECOMPUTE_NEXT
#ifdef PARANOID
    free(pfound);
    free(want);
#endif
    st->lookups += nlook;
}

/*
 * do_block -- process one block for all its check bases B.
 *
 * In:     st; cc (the context of c); Mblock.
 * Out:    clips the block's range [Mblock, Mblock + span_block] to [lo, hi]
 *         and runs block_B() for every B from floor(Mmin^(1/n))+1 to
 *         floor(Mmax^(1/(n-1))) (the bases in which some M of the block has
 *         n digits) that is in cc's table; adds (number of B) c to
 *         st->steps.  Tells block_B whether the whole block is inside
 *         [lo, hi].
 * Errors: none; skips blocks outside [lo, hi] and bases c without check
 *         bases.
 */
static void do_block(wstat_t *st, const cctx_t *cc, u128 Mblock)
{
    const prob_t *pr = st->pr;
    if (!cc->Bcount) return;
    u128 Mmin = Mblock, Mmax = Mblock + cc->span_block;
    if (Mmax < pr->lo || Mmin > pr->hi) return;
    int inside = Mmin >= pr->lo && Mmax <= pr->hi;
    if (Mmin < pr->lo) Mmin = pr->lo;
    if (Mmax > pr->hi) Mmax = pr->hi;
    u64 Blo = iroot(Mmin, pr->Lc) + 1, Bhi = iroot(Mmax, pr->Lc - 1);
    if (Blo < cc->Bbase) Blo = cc->Bbase;
    if (Bhi > cc->Bbase + cc->Bcount - 1) Bhi = cc->Bbase + cc->Bcount - 1;
    for (u64 B = Blo; B <= Bhi; B++) block_B(st, cc, &cc->tab[B - cc->Bbase], Mblock, inside);
    if (Bhi >= Blo) st->steps += (Bhi - Blo + 1) * cc->c;
}

/*
 * unit_decode -- unit index -> enumeration base c and the unit's digits.
 *
 * In:     pr; u < pr->nunits; pref with room for ud digits.
 * Out:    returns c (binary search for uoff[i] <= u < uoff[i+1]); the local
 *         index is read as the mixed-radix number (e_0 - 1, e_1, ...) in
 *         base c: pref[0] = e_0 in 1..c-1, the others in 0..c-1.
 * Errors: u is not range-checked (u >= nunits gives the last base and
 *         meaningless digits).  A problem without units (cmax < cmin) gives
 *         cmin and the digits 1, 0, ...: the GPU hosts call this for their
 *         progress line even then.
 */
static u64 unit_decode(const prob_t *pr, u64 u, u64 *pref)
{
    if (pr->cmax < pr->cmin) {              /* no bases at all: avoid the underflow below */
        pref[0] = 1;
        for (int i = 1; i < pr->ud; i++) pref[i] = 0;
        return pr->cmin;
    }
    u64 lo = 0, hi = pr->cmax - pr->cmin;   /* find i with uoff[i] <= u < uoff[i+1] */
    while (lo < hi) {
        u64 mid = (lo + hi + 1) / 2;
        if (pr->uoff[mid] <= u) lo = mid; else hi = mid - 1;
    }
    u64 c = pr->cmin + lo, local = u - pr->uoff[lo];
    for (int i = pr->ud - 1; i >= 1; i--) { pref[i] = local % c; local /= c; }
    pref[0] = 1 + local;
    return c;
}

/*
 * do_unit -- process one work unit: all blocks that start with its digits.
 *
 * In:     st; cc (the thread's context, rebuilt if the unit has another c);
 *         the unit index u.
 * Out:    do_block() for the unit's block, or, if the unit fixes one block
 *         digit less than a block has (n >= 10), for each value of that
 *         digit in increasing order, stopping once the block starts above hi.
 *         Units whose whole range misses [lo, hi] are skipped.
 * Errors: as cctx_set() (exits for c > 65535 or an impossible table).
 */
static void do_unit(wstat_t *st, cctx_t *cc, u64 u)
{
    const prob_t *pr = st->pr;
    u64 pref[MAXN];
    u64 c = unit_decode(pr, u, pref);
    if (cc->c != c) cctx_set(cc, pr, c);
    u128 Mu = 0;                           /* value of the unit's digits */
    for (int i = 0; i < pr->ud; i++) Mu += (u128)pref[i] * cc->w[i];
    if (Mu + cc->span_unit < pr->lo || Mu > pr->hi) return;
    int nb = pr->bd - pr->ud;              /* block digits left to loop over: 0 or 1 */
    if (nb == 0) do_block(st, cc, Mu);
    else {
        u128 wb = cc->w[pr->ud];
        for (u64 d = 0; d < c; d++) {
            u128 Mb = Mu + (u128)d * wb;
            if (Mb > pr->hi) break;
            do_block(st, cc, Mb);
        }
    }
}

/* ------------------------------------------------------------------ */
/* threaded search                                                     */
/* ------------------------------------------------------------------ */

/* g_stop -- set by SIGINT/SIGTERM (on_sig): workers stop taking new units,
   run_search() saves the state and prints the summary.  Also used by the
   GPU hosts. */
static volatile sig_atomic_t g_stop;
static u64 g_end_unit = UINT64_MAX;       /* -E: process units below this index only */
static void on_sig(int s) { (void)s; g_stop = 1; }

/* cctx_free -- release the tables of a cctx_t (the struct itself is the
   caller's).  No errors. */
static void cctx_free(cctx_t *cc) { free(cc->tab); free(cc->mem); }

/* add_stats -- *t += *s for every statistic (not the test fields).  Used to
   sum the threads and to add a resumed state's totals.  No errors. */
static void add_stats(wstat_t *t, const wstat_t *s)
{
    t->steps += s->steps; t->lookups += s->lookups; t->hits += s->hits; t->f1 += s->f1;
    t->st2 += s->st2; t->canon += s->canon; t->csum += s->csum; t->sol += s->sol;
    for (int L = 0; L <= MAXN; L++) t->fail[L] += s->fail[L];
}

/*
 * load_state -- read a checkpoint written by save_state() (also by the GPU
 * tools, which share the format and the unit numbering).
 *
 * In:     path; pr (the current problem).
 * Out:    0 if the file cannot be opened (normally: it does not exist yet);
 *         otherwise 1, the frontier in *frontier and all saved totals,
 *         including fail[] (optional "failL=" fields), in *acc.
 * Errors: exits with status 1 if the file cannot be parsed, belongs to
 *         another n, LO or HI, or has a frontier beyond pr->nunits: a
 *         damaged file, or an end unit (-E) below the saved frontier, since
 *         the callers lower nunits for -E before loading.
 */
static int load_state(const char *path, const prob_t *pr, u64 *frontier, wstat_t *acc)
{
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char lo[64], hi[64];
    int n;
    unsigned long long fr, a[9];
    int ok = fscanf(f, "n=%d lo=%63s hi=%63s frontier=%llu steps=%llu lookups=%llu hits=%llu f1=%llu st2=%llu "
                       "canon=%llu csum=%llu sol=%llu",
                    &n, lo, hi, &fr, &a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6], &a[7]) == 12;
    unsigned long long fl[MAXN + 1] = {0};
    if (ok) {
        for (int L = 3; L <= MAXN; L++) {
            int LL;
            unsigned long long v;
            if (fscanf(f, " fail%d=%llu", &LL, &v) != 2) break;
            if (LL >= 0 && LL <= MAXN) fl[LL] = v;
        }
    }
    fclose(f);
    if (!ok) { fprintf(stderr, "cannot parse state file %s\n", path); exit(1); }
    if (n != pr->n || parse_or_die(lo) != pr->lo || parse_or_die(hi) != pr->hi) {
        fprintf(stderr, "state file %s is for a different search\n", path);
        exit(1);
    }
    if (fr > pr->nunits) {
        fprintf(stderr, "state file %s: frontier %llu is beyond the last unit of this run (%llu units; -E below the "
                        "saved frontier?)\n", path, fr, (unsigned long long)pr->nunits);
        exit(1);
    }
    *frontier = fr;
    acc->steps = a[0]; acc->lookups = a[1]; acc->hits = a[2]; acc->f1 = a[3]; acc->st2 = a[4];
    acc->canon = a[5]; acc->csum = a[6]; acc->sol = a[7];
    for (int L = 0; L <= MAXN; L++) acc->fail[L] = fl[L];
    return 1;
}

/*
 * save_state -- write a checkpoint, one line: "n=.. lo=.. hi=.. frontier=..
 * steps=.. lookups=.. hits=.. f1=.. st2=.. canon=.. csum=.. sol=.." and
 * "failL=.." for L = n-2..3 (decimal numbers).
 *
 * In:     path; pr; the frontier; t (the statistics of exactly the units
 *         below the frontier, including a resumed state's).
 * Out:    path.tmp is written and renamed over path (atomic replacement).
 * Errors: if path.tmp cannot be written completely (e.g. a full disk),
 *         prints the reason, removes path.tmp and keeps the previous
 *         checkpoint; the search goes on.
 */
static void save_state(const char *path, const prob_t *pr, u64 frontier, const wstat_t *t)
{
    char tmp[4096], a[48], b[48];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "w");
    if (!f) { perror(tmp); return; }
    fprintf(f, "n=%d lo=%s hi=%s frontier=%llu steps=%llu lookups=%llu hits=%llu f1=%llu st2=%llu canon=%llu "
               "csum=%llu sol=%llu",
            pr->n, u128s(pr->lo, a), u128s(pr->hi, b), (unsigned long long)frontier, (unsigned long long)t->steps,
            (unsigned long long)t->lookups, (unsigned long long)t->hits, (unsigned long long)t->f1,
            (unsigned long long)t->st2, (unsigned long long)t->canon, (unsigned long long)t->csum,
            (unsigned long long)t->sol);
    for (int L = pr->n - 2; L >= 3; L--) fprintf(f, " fail%d=%llu", L, (unsigned long long)t->fail[L]);
    fprintf(f, "\n");
    int bad = ferror(f);
    if (fclose(f) != 0) bad = 1;
    if (bad) {                             /* keep the old checkpoint rather than a truncated one */
        fprintf(stderr, "cannot write %s; checkpoint not updated\n", tmp);
        remove(tmp);
        return;
    }
    rename(tmp, path);                     /* atomic replacement */
}

/*
 * print_summary -- the two final lines: "# n=.. [LO, HI] complete|INTERRUPTED:
 * time, steps, lookups, hits, f1, (n-1)-palindromes (st2), canonical,
 * checksum, solutions" and "# failed at length: L=..:..".  The labels match
 * a171775.c where the statistics are comparable.
 * In: o (the stream), pr, t (totals), complete, el (seconds).  Flushes o.
 * No errors.
 */
static void print_summary(FILE *o, const prob_t *pr, const wstat_t *t, int complete, double el)
{
    char a[48], b[48];
    fprintf(o, "# n=%d [%s, %s] %s: %.1fs, steps %llu, lookups %llu, hits %llu, f1 %llu, (n-1)-palindromes %llu, "
               "canonical %llu, checksum %016llx, solutions %llu\n",
            pr->n, u128s(pr->lo, a), u128s(pr->hi, b), complete ? "complete" : "INTERRUPTED", el,
            (unsigned long long)t->steps, (unsigned long long)t->lookups, (unsigned long long)t->hits,
            (unsigned long long)t->f1, (unsigned long long)t->st2, (unsigned long long)t->canon,
            (unsigned long long)t->csum, (unsigned long long)t->sol);
    fprintf(o, "# failed at length:");
    for (int L = pr->n - 2; L >= 3; L--) fprintf(o, " L=%d:%llu", L, (unsigned long long)t->fail[L]);
    fprintf(o, "\n");
    fflush(o);
}

#ifndef A171775_2D_NO_DRIVER
/* From here on: the CPU driver (threads, selftest, main); left out when the
   file is included by the C++ CUDA host. */

/* pend_t -- a finished unit at or above the frontier, with its statistics. */
typedef struct { u64 u; wstat_t st; } pend_t;

/*
 * search_t -- the shared state of one threaded search.
 *   pr         the problem
 *   next       the next unit to hand out (atomic counter)
 *   done       done[u] = 1 once unit u is finished (nunits + 1 bytes)
 *   frontier   the lowest unfinished unit: every unit below it is done
 *   committed  the statistics of exactly the units in [start, frontier) of
 *              this session, so a checkpoint (frontier, committed + resumed
 *              totals) is always exact
 *   pend, npend, pendcap  units finished above the frontier (out of order);
 *              their statistics move into committed when the frontier
 *              passes them.  Usually about one entry per thread.
 *   lock       protects done[], frontier, committed and pend[]
 *   verbose    -v, copied into each unit's statistics
 */
typedef struct {
    prob_t *pr;
    _Atomic u64 next;
    unsigned char *done;
    u64 frontier;
    wstat_t committed;
    pend_t *pend;
    size_t npend, pendcap;
    pthread_mutex_t lock;
    int verbose;
} search_t;

/*
 * finish_unit -- record a finished unit (call with S->lock held).
 * In:     S; the unit u; ust, its statistics.
 * Out:    done[u] = 1; the statistics wait in pend[] until the frontier
 *         reaches u; the frontier advances over all finished units, moving
 *         their statistics into committed.
 * Errors: exits with status 1 if pend[] cannot grow (out of memory).
 */
static void finish_unit(search_t *S, u64 u, const wstat_t *ust)
{
    if (S->npend == S->pendcap) {
        S->pendcap = S->pendcap ? 2 * S->pendcap : 64;
        S->pend = (pend_t *)realloc(S->pend, S->pendcap * sizeof(pend_t));
        if (!S->pend) { fprintf(stderr, "out of memory\n"); exit(1); }
    }
    S->pend[S->npend].u = u;
    S->pend[S->npend].st = *ust;
    S->npend++;
    S->done[u] = 1;
    while (S->frontier < S->pr->nunits && S->done[S->frontier]) {
        for (size_t i = 0; i < S->npend; i++)  /* every finished unit >= frontier is in pend[] */
            if (S->pend[i].u == S->frontier) {
                add_stats(&S->committed, &S->pend[i].st);
                S->pend[i] = S->pend[--S->npend];
                break;
            }
        S->frontier++;
    }
}

/*
 * search_totals -- a consistent view of a running search.
 * In:     S; below and finished (either may be NULL).
 * Out:    returns the frontier; *below = the statistics of the units below
 *         it (exact: what a checkpoint needs); *finished = those plus the
 *         units finished above it (for progress lines).  Neither includes a
 *         resumed state's totals.
 * Errors: none.
 */
static u64 search_totals(search_t *S, wstat_t *below, wstat_t *finished)
{
    pthread_mutex_lock(&S->lock);
    u64 fr = S->frontier;
    if (below) *below = S->committed;
    if (finished) {
        *finished = S->committed;
        for (size_t i = 0; i < S->npend; i++) add_stats(finished, &S->pend[i].st);
    }
    pthread_mutex_unlock(&S->lock);
    return fr;
}

/*
 * worker -- thread body: take units from S->next until they run out or
 * g_stop is set; run do_unit() on each with fresh statistics, then hand them
 * to finish_unit().
 * In: arg, the search_t.  Out: the units' statistics, through finish_unit();
 * returns NULL.
 * Errors: none of its own (cctx_set may exit).  g_stop is checked only
 * between units, so every unit taken is finished.
 */
static void *worker(void *arg)
{
    search_t *S = (search_t *)arg;
    cctx_t cc;
    memset(&cc, 0, sizeof cc);
    wstat_t ust;                           /* the current unit's statistics */
    while (!g_stop) {
        u64 u = atomic_fetch_add(&S->next, 1);
        if (u >= S->pr->nunits) break;
        memset(&ust, 0, sizeof ust);
        ust.pr = S->pr;
        ust.verbose = S->verbose;
        do_unit(&ust, &cc, u);
        pthread_mutex_lock(&S->lock);
        finish_unit(S, u, &ust);
        pthread_mutex_unlock(&S->lock);
    }
    cctx_free(&cc);
    return NULL;
}

/*
 * unit_weight -- rough work of a unit, for the progress percentage and the
 * ETA only.  In: pr, c, the leading digit d0.  Out: (number of check
 * bases B for M in [d0 c^(Le-1), (d0+1) c^(Le-1)] clipped to [lo, hi]) x
 * c^(De-3) (x steps of the blocks with e_0 = d0) x (the fraction of that M
 * range inside [lo, hi]); 0 outside.  run_search divides it by c when a unit
 * also fixes e_1.  No errors; an estimate in double precision.
 */
static double unit_weight(const prob_t *pr, u64 c, u64 d0)
{
    int Le = pr->Le, Lc = pr->Lc;
    u128 Mmin = (u128)d0 * pow_sat(c, Le - 1), Mmax = (u128)(d0 + 1) * pow_sat(c, Le - 1);
    if (Mmax < pr->lo || Mmin > pr->hi) return 0;
    if (Mmin < pr->lo) Mmin = pr->lo;
    if (Mmax > pr->hi) Mmax = pr->hi;
    double bl = pow((double)Mmin, 1.0 / Lc), bh = pow((double)Mmax, 1.0 / (Lc - 1));
    double nB = bh - bl + 1;
    if (nB < 0) nB = 0;
    double frac = (double)(Mmax - Mmin) / (double)pow_sat(c, Le - 1);
    return nB * pow((double)c, pr->De - 3) * frac;
}

/*
 * run_search -- a complete threaded search for n over [lo, hi] (the
 * "search" command, also used by the selftest).
 *
 * In:     n (8..20); lo, hi; nthreads; state (checkpoint file or NULL);
 *         interval (seconds between progress lines); verbose (-v); quiet
 *         (no progress, no summary: selftest); result (may be NULL).
 * Out:    SOLUTION/DEEP lines on sol_out, progress on stderr and the
 *         summary on stdout unless quiet; the checkpoint if requested;
 *         *result = the totals including a resumed state's.  Returns 0 if
 *         every unit (below -E, if given) is done, 3 if it was interrupted.
 *         Without units (an empty range, or -E 0) it prints a note and an
 *         empty "complete" summary.
 * How:    prob_init (nunits lowered to g_end_unit if that is smaller);
 *         resume from the state file; progress weights cw[u] = estimated
 *         steps of the units before u; start the threads; poll every 0.1 s:
 *         a progress line every `interval` seconds and a checkpoint every
 *         60 s, both from search_totals() (the checkpoint holds exactly the
 *         units below the frontier), until all units are done or g_stop is
 *         set; join; final checkpoint and summary.
 * Errors: exits through prob_init/cctx_set/load_state on fatal conditions
 *         (load_state also refuses a frontier beyond -E's end unit);
 *         allocation failures are not checked.
 */
static int run_search(int n, u128 lo, u128 hi, int nthreads, const char *state, double interval, int verbose,
                      int quiet, wstat_t *result)
{
    char a[48], b[48];
    prob_t pr;
    prob_init(&pr, n, lo, hi);
    if (g_end_unit < pr.nunits) pr.nunits = g_end_unit;   /* -E: stop before unit g_end_unit */
    if (pr.nunits == 0) {
        wstat_t none;
        memset(&none, 0, sizeof none);
        if (!quiet) {
            if (pr.cmax < pr.cmin)
                fprintf(stderr, "search2 n=%d [%s, %s]: no base c >= 3 has %d-digit numbers in this range, nothing to "
                                "search\n", n, u128s(lo, a), u128s(hi, b), pr.Le);
            else
                fprintf(stderr, "search2 n=%d [%s, %s]: -E 0, nothing to search\n", n, u128s(lo, a), u128s(hi, b));
            print_summary(stdout, &pr, &none, 1, 0.0);
        }
        if (result) *result = none;
        free(pr.uoff);
        return 0;
    }
    search_t S;
    memset(&S, 0, sizeof S);
    S.pr = &pr;
    S.verbose = verbose;
    S.done = (unsigned char *)calloc(pr.nunits + 1, 1);
    pthread_mutex_init(&S.lock, NULL);
    wstat_t base;                          /* the totals of a resumed state */
    memset(&base, 0, sizeof base);
    u64 start = 0;
    if (state && load_state(state, &pr, &start, &base) && !quiet)
        fprintf(stderr, "resuming at unit %llu of %llu\n", (unsigned long long)start, (unsigned long long)pr.nunits);
    for (u64 u = 0; u < start; u++) S.done[u] = 1;   /* units below the saved frontier are done */
    S.frontier = start;
    atomic_store(&S.next, start);

    /* progress weights: cw[u] = estimated steps of units 0..u-1 */
    double *cw = (double *)malloc((pr.nunits + 1) * sizeof(double));
    cw[0] = 0;
    for (u64 u = 0; u < pr.nunits; u++) {
        u64 pref[MAXN];
        u64 c = unit_decode(&pr, u, pref);
        double wgt = unit_weight(&pr, c, pref[0]);
        if (pr.ud > 1) wgt /= (double)c;
        cw[u + 1] = cw[u] + wgt;
    }
    double wtot = cw[pr.nunits] > 0 ? cw[pr.nunits] : 1;
    if (!quiet)
        fprintf(stderr, "search2 n=%d [%s, %s]: enumerate %d-digit palindromes in bases c in [%llu, %llu], %llu units, "
                        "%d threads\n",
                n, u128s(lo, a), u128s(hi, b), pr.Le, (unsigned long long)pr.cmin, (unsigned long long)pr.cmax,
                (unsigned long long)pr.nunits, nthreads);
    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);
    pthread_t *th = (pthread_t *)malloc(nthreads * sizeof(pthread_t));
    double t0 = now_sec(), tlast = t0, tsave = t0, w0 = cw[start];
    for (int i = 0; i < nthreads; i++) pthread_create(&th[i], NULL, worker, &S);
    for (;;) {                             /* the main thread only reports and checkpoints */
        usleep(100000);
        wstat_t below, fin;                /* units below the frontier; all finished units */
        u64 fr = search_totals(&S, &below, &fin);
        u64 nx = atomic_load(&S.next);
        int finished = fr >= pr.nunits;
        double t = now_sec();
        if (!quiet && (t - tlast >= interval || finished || g_stop)) {
            double el = t - t0, wd = cw[fr < pr.nunits ? fr : pr.nunits];
            double eta = (wd - w0) > 0 ? (wtot - wd) * el / (wd - w0) : 0;
            u64 pref[MAXN];
            u64 cc_ = unit_decode(&pr, fr < pr.nunits ? fr : pr.nunits - 1, pref);
            fprintf(stderr,
                    "[%7.0fs] frontier %llu/%llu (c=%llu) %.3f%%  %.3e steps/s %.3e lookups/s  st2 %llu canon %llu sol %llu  "
                    "ETA %.0fs\n",
                    el, (unsigned long long)fr, (unsigned long long)pr.nunits, (unsigned long long)cc_, 100 * wd / wtot,
                    fin.steps / (el > 0 ? el : 1), fin.lookups / (el > 0 ? el : 1),
                    (unsigned long long)(base.st2 + fin.st2), (unsigned long long)(base.canon + fin.canon),
                    (unsigned long long)(base.sol + fin.sol), eta);
            tlast = t;
        }
        if (state && (t - tsave >= 60 || finished || g_stop)) {
            add_stats(&below, &base);      /* exactly the units below fr */
            save_state(state, &pr, fr, &below);
            tsave = t;
        }
        if (finished || (g_stop && nx >= pr.nunits) || g_stop) break;
    }
    for (int i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
    wstat_t all;                           /* every unit taken has finished, so none is pending */
    u64 fr = search_totals(&S, &all, NULL);
    add_stats(&all, &base);
    if (state) save_state(state, &pr, fr, &all);
    int complete = fr >= pr.nunits;
    if (!quiet) print_summary(stdout, &pr, &all, complete, now_sec() - t0);
    if (result) *result = all;
    free(th); free(cw); free(S.done); free(S.pend); free(pr.uoff);
    return complete ? 0 : 3;
}

/* ------------------------------------------------------------------ */
/* selftest                                                            */
/* ------------------------------------------------------------------ */

/* rnd -- xorshift64 pseudo-random numbers; the fixed seed makes every
   selftest run check the same blocks. */
static u64 rng_state = 0x2545F4914F6CDD1DULL;
static u64 rnd(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

/* pair_cmp -- qsort() comparator: order pairs by M, then by B. */
static int pair_cmp(const void *a, const void *b)
{
    const pair_t *x = a, *y = b;
    if (x->M != y->M) return x->M < y->M ? -1 : 1;
    if (x->B != y->B) return x->B < y->B ? -1 : 1;
    return 0;
}

/* ends_mirror -- 1 if M has exactly L base-B digits and digit i = digit
   L-1-i for i < k (the naive version of a hit), else 0.  No errors. */
static int ends_mirror(u128 M, u64 B, int L, int k)
{
    u64 d[130];
    int len = 0;
    u128 x = M;
    while (x) { d[len++] = (u64)(x % B); x /= B; }
    if (len != L) return 0;
    for (int i = 0; i < k; i++)
        if (d[i] != d[L - 1 - i]) return 0;
    return 1;
}

/* all (M, B) for M in the block: mode 1 = n-digit palindrome in B, mode 2 = the k(c,B) ends mirror */
/*
 * naive_block -- brute-force reference for one block: every M = Mblock +
 * x w_x + y w_y + z w_z in [lo, hi] (all x, y, z < c) and every B with
 * B^(n-1) <= M < B^n, kept if (mode 1) M is an n-digit palindrome in B, or
 * (mode 2) its k(c, B) outer base-B digit pairs match.
 * Out:    *out = a malloc'ed array of the pairs (the caller frees it),
 *         *np = their number.
 * Errors: exits with status 1 if such a B is outside cc's table (the table
 *         range would be wrong).  Allocation failures are not checked.
 *         Cost c^3 times the number of B: small c only.
 */
static void naive_block(const prob_t *pr, const cctx_t *cc, u128 Mblock, int mode, pair_t **out, size_t *np)
{
    size_t cap = 1024, cnt = 0;
    pair_t *p = malloc(cap * sizeof(pair_t));
    u64 c = cc->c;
    for (u64 x = 0; x < c; x++)
        for (u64 y = 0; y < c; y++)
            for (u64 z = 0; z < c; z++) {
                u128 M = Mblock + (u128)x * cc->wx + (u128)y * cc->wy + (u128)z * cc->wz;
                if (M < pr->lo || M > pr->hi) continue;
                u64 Blo = iroot(M, pr->Lc) + 1, Bhi = iroot(M, pr->Lc - 1);
                if (Blo < 2) Blo = 2;
                for (u64 B = Blo; B <= Bhi; B++) {
                    if (B < cc->Bbase || B >= cc->Bbase + cc->Bcount) {
                        fprintf(stderr, "naive: B=%llu outside the table range\n", (unsigned long long)B);
                        exit(1);
                    }
                    int ok = mode == 1 ? is_pal_base(M, B, pr->Lc)
                                       : ends_mirror(M, B, pr->Lc, (int)cc->tab[B - cc->Bbase].k);
                    if (!ok) continue;
                    if (cnt == cap) { cap *= 2; p = realloc(p, cap * sizeof(pair_t)); }
                    p[cnt].M = M;
                    p[cnt].B = B;
                    cnt++;
                }
            }
    *out = p;
    *np = cnt;
}

/*
 * compare_block -- selftest: run do_block() in a record mode on the block
 * (e_0, e_1, 0, ...) of base c and compare its (M, B) pairs with
 * naive_block().
 *
 * In:     n; c; e0 (1..c-1); e1 (used if the block has a second digit);
 *         mode (1 = n-digit palindromes, 2 = all k-digit matches); npairs
 *         (the number of naive pairs is added to it).
 * Out:    1 if both lists agree, 0 otherwise (and a MISMATCH line).
 * Errors: none; the range is effectively unlimited (hi = 2^126 - 1), so c
 *         must stay small for the naive side.
 */
static int compare_block(int n, u64 c, u64 e0, u64 e1, int mode, size_t *npairs)
{
    prob_t pr;
    prob_init(&pr, n, 1, U128_MAX >> 2);
    cctx_t cc;
    memset(&cc, 0, sizeof cc);
    cctx_set(&cc, &pr, c);
    u128 Mb = (u128)e0 * cc.w[0] + (pr.bd >= 2 ? (u128)e1 * cc.w[1] : 0);   /* other block digits 0 */
    wstat_t st;
    memset(&st, 0, sizeof st);
    st.pr = &pr;
    st.record = mode;
    do_block(&st, &cc, Mb);
    pair_t *np;
    size_t nn;
    naive_block(&pr, &cc, Mb, mode, &np, &nn);
    qsort(st.pairs, st.np, sizeof(pair_t), pair_cmp);
    qsort(np, nn, sizeof(pair_t), pair_cmp);
    int ok = st.np == nn;
    for (size_t i = 0; ok && i < nn; i++) ok = st.pairs[i].M == np[i].M && st.pairs[i].B == np[i].B;
    if (!ok) {
        char buf[48];
        printf("  MISMATCH n=%d c=%llu e0=%llu e1=%llu mode %d (Mblock=%s): fast %zu, naive %zu\n", n,
               (unsigned long long)c, (unsigned long long)e0, (unsigned long long)e1, mode, u128s(Mb, buf), st.np, nn);
    }
    *npairs += nn;
    free(st.pairs);
    free(np);
    cctx_free(&cc);
    free(pr.uoff);
    return ok;
}

/*
 * cmd_selftest -- the "selftest" command.
 *
 * In:     full (also the complete n = 10 search over [1, 2^72], about ten
 *         minutes on 10 cores); nthreads.
 * Out:    prints two parts and returns 0 if all passed, 1 otherwise:
 *         1. the lookup agrees with naive_block() on random blocks of small
 *            c for n = 8..12, in both record modes, including the first and
 *            last block (e_0, e_1) = (1, 0) and (c-1, c-1);
 *         2. complete searches over [1, a(n)] for n = 8, 9 (and 10) give
 *            exactly one solution, a(n), and the same st2, canonical count,
 *            checksum and fail[] as a171775.c (reference values below).
 * Errors: none beyond a failed check (returned as 1).  tmpfile() failure is
 *         not checked; -E must not be given (it would cut the searches).
 */
static int cmd_selftest(int full, int nthreads)
{
    int bad = 0;
    printf("1. fast lookup vs naive enumeration on random blocks\n");
    struct { int n; u64 cmin, cmax; int reps; } cfg[] = {
        {8, 4, 40, 60}, {9, 4, 36, 60}, {10, 4, 26, 50}, {11, 4, 22, 40}, {12, 4, 12, 20},
    };
    for (size_t ci = 0; ci < sizeof cfg / sizeof cfg[0]; ci++) {
        int n = cfg[ci].n, okc = 0, tot = 0;
        size_t p1 = 0, p2 = 0;
        for (int r = 0; r < cfg[ci].reps; r++) {
            u64 c = cfg[ci].cmin + rnd() % (cfg[ci].cmax - cfg[ci].cmin + 1);
            u64 e0 = 1 + rnd() % (c - 1), e1 = rnd() % c;
            if (r == 0) { e0 = 1; e1 = 0; }
            if (r == 1) { e0 = c - 1; e1 = c - 1; }
            int ok = compare_block(n, c, e0, e1, 1, &p1) & compare_block(n, c, e0, e1, 2, &p2);
            okc += ok;
            tot++;
        }
        printf("  n=%d c in [%llu,%llu]: %d/%d blocks agree (%zu n-digit palindrome pairs, %zu k-digit matches)\n", n,
               (unsigned long long)cfg[ci].cmin, (unsigned long long)cfg[ci].cmax, okc, tot, p1, p2);
        fflush(stdout);
        bad += okc != tot;
    }
    printf("2. full searches vs a171775.c (1D) statistics\n");
    /* reference values from complete a171775.c runs over [1, a(n)]: st2,
       canonical, checksum, fail[n-2], fail[n-3], fail[n-4] */
    struct { int n; const char *hi; u64 st2, canon, csum; u64 fails[3]; } ref[] = {
        {8, "2^42", 4725, 4724, 0x13747afea2768e7aULL, {4723, 0, 0}},
        {9, "2^56", 19140, 19140, 0x380d9946bb59f02eULL, {19138, 1, 0}},
        {10, "2^72", 135382, 135382, 0xe9eed6127c51a50fULL, {135381, 0, 0}},
    };
    int nref = full ? 3 : 2;
    for (int i = 0; i < nref; i++) {
        FILE *mem = tmpfile();             /* capture the SOLUTION lines */
        FILE *save = sol_out;
        sol_out = mem;
        wstat_t t;
        double t0 = now_sec();
        run_search(ref[i].n, 1, parse_or_die(ref[i].hi), nthreads, NULL, 1e9, 0, 1, &t);
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
        int n = ref[i].n;
        int ok = nsol == 1 && best == parse_or_die(ref[i].hi) && t.st2 == ref[i].st2 && t.canon == ref[i].canon &&
                 t.csum == ref[i].csum && t.fail[n - 2] == ref[i].fails[0] && t.fail[n - 3] == ref[i].fails[1] &&
                 t.fail[n - 4] == ref[i].fails[2];
        char buf[48];
        printf("  n=%d [1, %s]: %.2fs, %.3e steps, %.3e lookups, hits %llu, f1 %llu, st2 %llu, canonical %llu, "
               "checksum %016llx, fails %llu/%llu/%llu, %d solution(s), smallest %s  %s\n",
               n, ref[i].hi, now_sec() - t0, (double)t.steps, (double)t.lookups, (unsigned long long)t.hits,
               (unsigned long long)t.f1, (unsigned long long)t.st2, (unsigned long long)t.canon,
               (unsigned long long)t.csum, (unsigned long long)t.fail[n - 2], (unsigned long long)t.fail[n - 3],
               (unsigned long long)t.fail[n - 4], nsol, best ? u128s(best, buf) : "none", ok ? "ok" : "WRONG");
        fflush(stdout);
        bad += !ok;
    }
    printf(bad ? "SELFTEST FAILED (%d)\n" : "selftest passed\n", bad);
    return bad ? 1 : 0;
}

#ifdef PARANOID
/*
 * cmd_paranoid -- (PARANOID builds) the "paranoid" command: do_block() on
 * random blocks for several (n, c) ranges, up to the production sizes
 * (c up to 1024 for n = 11), with block_B's state recomputed from scratch at
 * every x and the (y, z) box brute-forced on part of the steps.
 * In:     reps, the number of blocks per configuration (each configuration
 *         also stops after 60 s).
 * Out:    one summary line per configuration; returns 0.
 * Errors: abort() with a diagnostic on any mismatch.  Solutions go to
 *         /dev/null.
 */
static int cmd_paranoid(int reps)
{
    struct { int n; u64 cmin, cmax; u128 hi; } cfg[] = {
        {8, 4, 64, (u128)1 << 60}, {9, 4, 64, (u128)1 << 62}, {10, 4, 64, (u128)1 << 72},
        {11, 4, 64, (u128)1 << 90}, {12, 4, 40, (u128)1 << 100},
        {10, 200, 512, (u128)1 << 72}, {11, 400, 1024, (u128)1 << 90},
    };
    FILE *devnull = fopen("/dev/null", "w");
    FILE *save = sol_out;
    sol_out = devnull;
    for (size_t ci = 0; ci < sizeof cfg / sizeof cfg[0]; ci++) {
        int n = cfg[ci].n;
        u64 blocks = 0, chk = 0, zh = 0, steps = 0;
        double t0 = now_sec();
        for (int r = 0; r < reps && now_sec() - t0 < 60; r++) {
            u64 c = cfg[ci].cmin + rnd() % (cfg[ci].cmax - cfg[ci].cmin + 1);
            prob_t pr;
            prob_init(&pr, n, 1, cfg[ci].hi);
            cctx_t cc;
            memset(&cc, 0, sizeof cc);
            cctx_set(&cc, &pr, c);
            u128 dm = cfg[ci].hi / cc.w[0];     /* leading digits with Mblock <= hi */
            u64 dmax = dm < c - 1 ? (u64)dm : c - 1;
            if (dmax >= 1 && cc.Bcount) {
                u64 e0 = 1 + rnd() % dmax, e1 = rnd() % c;
                u128 Mb = (u128)e0 * cc.w[0] + (pr.bd >= 2 ? (u128)e1 * cc.w[1] : 0);
                wstat_t st;
                memset(&st, 0, sizeof st);
                st.pr = &pr;
                do_block(&st, &cc, Mb);
                blocks++;
                chk += st.pchk;
                zh += st.pz;
                steps += st.steps;
            }
            cctx_free(&cc);
            free(pr.uoff);
        }
        printf("  paranoid n=%d c in [%llu,%llu]: %llu blocks, %llu x-steps state-checked, %llu brute-force (y,z) boxes, "
               "%llu matches: ok\n",
               n, (unsigned long long)cfg[ci].cmin, (unsigned long long)cfg[ci].cmax, (unsigned long long)blocks,
               (unsigned long long)steps, (unsigned long long)chk, (unsigned long long)zh);
        fflush(stdout);
    }
    sol_out = save;
    return 0;
}
#endif

/* ------------------------------------------------------------------ */

/* usage -- print the command summary to stderr and exit with status 1
   (-E and -u are described in the header). */
static void usage(void)
{
    fprintf(stderr,
            "usage: a171775_2d selftest [full] [-t T]\n"
            "       a171775_2d search n LO HI [-t T] [-S state] [-i sec] [-v]\n"
#ifdef PARANOID
            "       a171775_2d paranoid [reps]\n"
#endif
    );
    exit(1);
}

#ifndef A171775_2D_LIB
/*
 * main -- parse the options (-t, -S, -i, -v, -E, anywhere on the line; -u
 * acts at once) and up to eight positional arguments, and run the command.
 *
 * Errors: an unknown command or a wrong number of arguments prints the
 *         usage and exits with status 1; search with n outside 8..20 prints
 *         a message and returns 1, as does -u with missing arguments, n
 *         outside 8..20 or c outside [cmin, cmax]; bad numbers exit with
 *         status 1.  A thread count below 1 becomes 1.
 */
int main(int argc, char **argv)
{
    sol_out = stdout;
    int nthreads = (int)sysconf(_SC_NPROCESSORS_ONLN);   /* default: one thread per online core */
    const char *state = NULL;
    double interval = 10;
    int verbose = 0;
    char *pos[8];                          /* positional arguments: the command and its operands */
    int npos = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-t") && i + 1 < argc) nthreads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-S") && i + 1 < argc) state = argv[++i];
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) interval = atof(argv[++i]);
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else if (!strcmp(argv[i], "-E") && i + 1 < argc) g_end_unit = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-u")) {                   /* print the first unit of base c */
            if (i + 3 >= argc) { fprintf(stderr, "-u needs three arguments: n c HI\n"); return 1; }
            int n = atoi(argv[++i]);
            u64 c = strtoull(argv[++i], NULL, 10);
            if (n < 8 || n > 20) { fprintf(stderr, "-u: n must be 8..20\n"); return 1; }
            prob_t pr;
            prob_init(&pr, n, 1, parse_or_die(argv[++i]));    /* unit numbering for LO = 1 */
            if (c < pr.cmin || c > pr.cmax) {
                fprintf(stderr, "-u: c must be in [%llu, %llu] for this n and HI\n", (unsigned long long)pr.cmin,
                        (unsigned long long)pr.cmax);
                return 1;
            }
            printf("%llu\n", (unsigned long long)pr.uoff[c - pr.cmin]);
            return 0;
        }
        else if (npos < 8) pos[npos++] = argv[i];
        else usage();
    }
    if (nthreads < 1) nthreads = 1;
    if (npos < 1) usage();
    if (!strcmp(pos[0], "selftest")) return cmd_selftest(npos > 1 && !strcmp(pos[1], "full"), nthreads);
#ifdef PARANOID
    if (!strcmp(pos[0], "paranoid")) return cmd_paranoid(npos > 1 ? atoi(pos[1]) : 300);
#endif
    if (!strcmp(pos[0], "search") && npos == 4) {
        int n = atoi(pos[1]);
        /* n >= 8 for at least one block digit (De >= 4), n <= 20 for the array sizes */
        if (n < 8 || n > 20) { fprintf(stderr, "n must be 8..20 (use a171775 for smaller n)\n"); return 1; }
        return run_search(n, parse_or_die(pos[2]), parse_or_die(pos[3]), nthreads, state, interval, verbose, 0, NULL);
    }
    usage();
    return 1;
}
#endif /* A171775_2D_LIB */
#endif /* A171775_2D_NO_DRIVER */
