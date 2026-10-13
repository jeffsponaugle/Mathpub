/*
 * a171775.c -- exhaustive search for OEIS A171775 (the "1D" method).
 *
 *   "a(n) = smallest number M such that there exist bases b_2, b_3, ..., b_n
 *    with the property that M written in base b_k is a k-digit palindrome
 *    for all k = 2..n."
 *
 * Terms (n = 1..11):
 *   1, 3, 5, 52, 130, 1885, 1073741824 = 2^30, 4398046511104 = 2^42,
 *   72057594037927936 = 2^56 (a(9): Max Alekseyev, Jun 2026),
 *   2^72 (a(10): found with this program, Sep 2026),
 *   2^90 (a(11): found with a171775_2d.c, Sep 2026).
 * Bound: a(n) <= 2^((n-1)(n-2)) (James G. Merickel); conjecture (Alekseyev):
 * equality for n >= 7.
 *
 * Why 2^((n-1)(n-2)) works: in base b = 2^e - 1, 2^r (b+1)^(L-1) has the
 * digits 2^r C(L-1, j), a palindrome of L digits whenever 2^r C(L-1, j) < b.
 * With N = (n-1)(n-2) = e(L-1) + r every length L = 2..n finds such (e, r).
 *
 * Layout of this file
 *   integer helpers      pow_sat, iroot, u128s, number parsing, gcd, inverse, mod
 *   palindrome tests     is_pal_base, find_pal_base
 *   problem setup        prob_t, cconst_t, bctx_t, wstat_t; prob_init, bctx_set
 *   survivors            candidate (full checks of one hit), report_solution
 *   the fast filter      block_c (the hot loop), do_block, unit_decode, do_unit
 *   naive references     brute-force versions, used by the selftest
 *   threaded search      worker, checkpoints, cmd_search
 *   selftest             compare_block, check_value, cmd_selftest
 *   paranoid, main       cmd_paranoid (-DPARANOID builds), usage, main
 *
 * ========================================================================
 * THEORY OF OPERATIONS
 * ========================================================================
 *
 * Notation.  M is an L-digit palindrome in base b when b^(L-1) <= M < b^L
 * and its L base-b digits read the same in both directions.  All tests are
 * exact integer arithmetic: M is an unsigned __int128, and the per-step
 * state of the hot loop fits in 64 bits.  Floating point is used only as a
 * pre-filter whose verdict is always confirmed exactly, and for progress
 * estimates.
 *
 * 1. Two lengths do the work.
 *    L = 2 is free: every M >= 3 is "11" in base M-1.  So a solution for n
 *    is an n-digit palindrome in some base B and an (n-1)-digit palindrome
 *    in some base c (necessarily c > B, as B^(n-1) <= M < c^(n-1)) that
 *    also passes the lengths n-2, ..., 3.  The program finds every M in
 *    [LO, HI] with the first two properties, the "survivors" (135382 below
 *    2^72 for n = 10), and tests the other lengths on those directly.  The
 *    n-digit palindromes are the smaller of the two sets (3.7e12 below 2^72
 *    for n = 10), so they are enumerated, and the (n-1)-digit condition is
 *    imposed by solving a congruence for the innermost digit instead of
 *    trying each of its values.
 *
 * 2. The n-digit palindromes in base B.
 *    There are h = n/2 mirrored digit pairs and D = ceil(n/2) free digits
 *    d_0..d_{D-1} (d_0 >= 1 is the leading digit) with weights
 *    w_i = B^(n-1-i) + B^i (i < h) and, for odd n, w_h = B^h for the middle
 *    digit, so M = sum d_i w_i.  The three innermost free digits are
 *    x = d_{D-3}, y = d_{D-2}, z = d_{D-1}, with weights w_x, w_y and s; the
 *    others form the "block" d_0..d_{D-4}:
 *
 *        M = Mblock + x w_x + y w_y + z s,    Mblock = sum_{i<D-3} d_i w_i.
 *
 *    The work is nested like this:
 *
 *      unit     the base B and the first ud = max(1, D-4) free digits:
 *               (B, d_0) for n <= 10, (B, d_0, d_1) for n = 11 and 12;
 *               the piece of work a thread takes (do_unit), numbered base
 *               by base
 *      block    all D-3 block digits fixed; do_unit loops over the last one
 *               when the unit does not fix it
 *      c        every base in which some M of the block can have n-1
 *               digits, c^(n-2) <= M < c^(n-1) (do_block)
 *      (x, y)   B^2 steps per (block, c): block_c, the hot loop
 *      z        solved from a congruence: no value, one, or a few
 *
 * 3. The (n-1)-digit condition as a congruence in z.
 *    Let k = n-2-h, P = c^(h+1) = c^((n-1)-k) and N = c^k.  For fixed
 *    (x, y), z moves M by at most (B-1) s < B^(h+1) < P, so q = floor(M / P)
 *    takes at most two values, q and q+1.  (With one more digit pair,
 *    P = c^h, that would fail.)  q holds the k leading base-c digits of M,
 *    E_0 (the leading digit), ..., E_{k-1}; M being a palindrome, its k
 *    trailing digits are the same in reverse order:
 *
 *        M = T(q) (mod N),   T = E_0 + E_1 c + ... + E_{k-1} c^(k-1),
 *
 *    which, with M0 = Mblock + x w_x + y w_y, is the linear congruence
 *
 *        z s = T - M0 (mod N).
 *
 * 4. Solving it with one comparison.
 *    Let g = gcd(s, N), N' = N/g, and s'' = s/g + j N' with the smallest
 *    j >= 0 that makes s'' coprime to N.  Put u = s''^(-1) mod N and
 *    w = (T - M0) u mod N.  Since u is a unit and u = (s/g)^(-1) mod N', a
 *    solution exists iff g | w, and the solutions are z = w/g + i N'.  The
 *    smallest one must be < B, so  w < g B  is the whole test per (x, y);
 *    divisibility and the range of z are checked only on the rare hits.
 *
 * 5. Incremental update with additions only.
 *    (x, y) runs through [0, B)^2 with y fastest: y -> y+1 adds w_y to M0,
 *    the wrap (y = B-1 -> 0, x -> x+1) adds dx = w_x - (B-1) w_y; both are
 *    < P, so q grows by at most one per step.  The state is
 *      rem = M0 mod P             (+ w_y or dx; passing P means q -> q+1)
 *      w   = (T(q) - M0) u mod N  (- w_y u or - dx u: precomputed Wy, Wx)
 *      the digits E_j of q        (touched only when E_{k-1} carries).
 *    q -> q+1 changes T by c^(k-1), or, when the carry stops at digit m
 *    (E_{m+1..k-1} = c-1 wrap to 0 and E_m += 1), by
 *      -(c-1)(c^(m+1) + ... + c^(k-1)) + c^m = c^(m+1) + c^m (mod c^k).
 *    That change times u (mod N) is wnext, recomputed only when E_{k-1}
 *    reaches c-1 or wraps.  A step costs a few additions and comparisons.
 *
 * 6. The two regions.
 *    Region 0 (q):   the z with rem + z s < P; solutions from w.
 *    Region 1 (q+1): the z with rem + z s >= P; solutions from
 *                    w1 = w + wnext, i.e. from T(q+1).  Only possible when
 *                    rem >= Rcross = P - (B-1) s and q+1 < c^k.
 *
 * 7. Hits, survivors and the other lengths (candidate()).
 *    A hit (about B/c^k per step) says only that the k outer base-c digit
 *    pairs of M match.  It is checked in full: M in [LO, HI] and all n-1
 *    digits in base c.  A survivor can be an n-digit palindrome in several
 *    bases B and an (n-1)-digit one in several c, so it is found once per
 *    pair (B, c); it is counted only for the pair in which B and c are the
 *    smallest such bases (the canonical pair).  Canonical survivors are
 *    tested for L = n-2, ..., 3 by scanning every base with
 *    b^(L-1) <= M < b^L (find_pal_base).  The first length that fails is
 *    counted in fail[L]; an M that passes them all is printed as a SOLUTION.
 *    Nearly every survivor fails at L = n-2 already.
 *
 * 8. Statistics and checksum (the final "# n=..." line).
 *    tuples              (block, c, x, y) steps
 *    hits                z values produced by the congruence
 *    (n-1)-palindromes   (M, B, c) with M in range and a palindrome of both
 *                        lengths ("st2" in the code)
 *    canonical           survivors, each M counted once ("canon")
 *    checksum            sum of surv_hash(M) over the canonical survivors,
 *                        mod 2^64.  It does not depend on the order of the
 *                        work, the thread count or checkpoints, and
 *                        a171775_2d.c and the GPU ports compute the same
 *                        value, so runs with different methods and hardware
 *                        can be compared.
 *    solutions           canonical survivors that pass every length
 *    failed at length    fail[L] for L = n-2..3
 *
 * 9. Threads and checkpoints.
 *    Threads take units from an atomic counter.  Each unit's statistics are
 *    collected separately (wstat_t) and handed over when the unit is done
 *    (finish_unit).  The lowest unit not yet finished is the frontier, and
 *    the statistics are committed in frontier order: a unit finished out of
 *    order waits in a short pending list until the frontier reaches it.  So
 *    the totals of exactly the units below the frontier are always known.
 *    With -S, the frontier, those totals and fail[] are written to a state
 *    file every 60 s and at the end, and a run with the same n, LO, HI
 *    resumes at the frontier with identical results, however the earlier
 *    run ended (Ctrl-C, SIGTERM, kill -9, a crash or a power cut).  SIGINT
 *    and SIGTERM stop cleanly: every thread finishes the unit it has.
 *
 * 10. Limits.
 *    The fast search needs n = 7..20 (for n = 3..6 "search" runs the plain
 *    enumeration naive_search) and, for every base c that occurs,
 *    c^(h+1) < 2^62 and c^k < 2^42 (bctx_set exits otherwise).  That covers
 *    n = 10 up to 2^72 and n = 11 up to 2^90, but not n = 12 up to 2^110.
 *    A range without n-digit numbers (e.g. HI < 2^(n-1), or LO > HI) is
 *    reported as "nothing to search" with an empty summary.
 *
 * 11. Verification.
 *    "selftest" checks the known terms, compares the fast filter with a
 *    naive enumeration on random blocks (every k-digit match and every
 *    palindrome pair), and reproduces a(7), a(8) (and a(9) with "full") by
 *    complete searches.  A build with -DPARANOID recomputes the incremental
 *    state from scratch at every step and brute-forces z; it runs the
 *    selftest that way, and its "paranoid" command checks random blocks
 *    up to the production sizes.
 *
 * Usage
 *   a171775 selftest [full]        known terms, fast filter vs naive, a(7), a(8) [, a(9)]
 *   a171775 search n LO HI [-t T] [-S state] [-i sec] [-v]
 *   a171775 check M [n]            bases for each length 2..n (n defaults to 12)
 *   a171775 naive n LO HI          reference enumeration (small ranges)
 *   a171775 paranoid [reps]        state checks (only in a -DPARANOID build)
 * Options: -t threads (default: all cores), -S state file, -i seconds between
 * progress lines (default 10), -v print DEEP lines (survivors that pass
 * length n-2).  Numbers accept 2^72, 1e21, 2^72-1, ...
 *
 * Output: SOLUTION and DEEP lines and the final summary on stdout, progress
 * on stderr.  Exit status: 0 = done (also after an interrupted search; the
 * summary line says INTERRUPTED), 1 = usage, number or state-file error, a
 * fatal limit, or a failed selftest, 2 = "check" found a length without a
 * palindrome base.  A PARANOID build calls abort() on a mismatch.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef unsigned __int128 u128;    /* M and anything that scales like it */
typedef uint64_t u64;               /* bases, digits, per-step state */
typedef int64_t i64;
typedef uint32_t u32;

#define MAXN 24                     /* largest length n the arrays allow (search: n <= 20) */
#define MAXK 8                      /* largest k = n-2-h, the number of base-c digit pairs */
#define U128_MAX (~(u128)0)         /* also the saturation value of pow_sat() */

/* ------------------------------------------------------------------ */
/* integer helpers                                                     */
/* ------------------------------------------------------------------ */

/*
 * pow_sat -- b^e, saturating at U128_MAX.
 *
 * In:     b  the base (0 and 1 allowed); e  the exponent (e <= 0 gives 1).
 * Out:    b^e, or U128_MAX if b^e does not fit in 128 bits.
 * Errors: none.  A saturated value compares above every M the program
 *         handles, which is how the callers use it ("b^e > M").
 */
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
 * iroot -- floor(x^(1/k)), the integer k-th root.
 *
 * In:     x  any 128-bit value; k  the root order (k >= 2 here).
 * Out:    the largest r with r^k <= x (0 for x = 0).
 * How:    a double-precision estimate, then exact correction steps until
 *         r^k <= x < (r+1)^k, so the result is exact whatever pow() returns.
 * Errors: none for the inputs used here.  The estimate is capped at 1.8e19,
 *         so a root above that (only possible for k = 1, or k = 2 with
 *         x > 3.2e38) would need an extremely long correction loop.
 */
static u64 iroot(u128 x, int k)
{
    if (x == 0) return 0;
    double a = pow((double)x, 1.0 / k);
    u64 r = a >= 1.8e19 ? (u64)1.8e19 : (u64)a;   /* (u64) of a larger double is undefined */
    if (r == 0) r = 1;
    while (r > 1 && pow_sat(r, k) > x) r--;      /* estimate too high */
    while (pow_sat(r + 1, k) <= x) r++;          /* estimate too low */
    return r;
}

/*
 * u128s -- decimal representation of a 128-bit value.
 *
 * In:     x; buf with room for at least 40 bytes (callers use 48).
 * Out:    buf, holding the digits of x and a terminating NUL, so the call
 *         can be used directly as a printf argument.
 * Errors: none; the size of buf is not checked.
 */
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
 * parse_term -- parse one term of a number expression: DIGITS, DIGITS^EXP or
 * DIGITSeEXP (also 'E'), e.g. "52", "2^72", "1e21".
 *
 * In:     *ps  the text; parsing starts there.
 * Out:    1, the value in *out and *ps advanced past the term; or 0 on a
 *         syntax error (no digits, no exponent digits, or a base above
 *         2^64-1 before '^'), with *ps unchanged.
 * Errors: overflow is not detected: decimal values of 2^128 and more wrap,
 *         b^e saturates at U128_MAX, the "e" product wraps, and huge
 *         exponents are truncated to int.
 */
static int parse_term(const char **ps, u128 *out)
{
    const char *s = *ps;
    u128 v = 0;
    int nd = 0;
    while (*s >= '0' && *s <= '9') { v = v * 10 + (u128)(*s - '0'); s++; nd++; }
    if (!nd) return 0;
    if (*s == '^') {
        s++;
        char *e;
        unsigned long ex = strtoul(s, &e, 10);
        if (e == s) return 0;
        s = e;
        if (v > UINT64_MAX) return 0;
        v = pow_sat((u64)v, (int)ex);
    } else if (*s == 'e' || *s == 'E') {
        s++;
        char *e;
        unsigned long ex = strtoul(s, &e, 10);
        if (e == s) return 0;
        s = e;
        v *= pow_sat(10, (int)ex);
    }
    *ps = s;
    *out = v;
    return 1;
}

/*
 * parse_u128 -- parse a number expression: terms (see parse_term) joined by
 * '+' or '-' and evaluated left to right, e.g. "2^72-1", "2^90+2^80".
 *
 * In:     s  a NUL-terminated string (no spaces allowed).
 * Out:    1 and the value in *out, or 0 on a syntax error or trailing text.
 * Errors: the arithmetic wraps mod 2^128 ("1-2" gives 2^128-1).
 */
static int parse_u128(const char *s, u128 *out)
{
    u128 acc, t;
    if (!parse_term(&s, &acc)) return 0;
    while (*s == '+' || *s == '-') {
        char op = *s++;
        if (!parse_term(&s, &t)) return 0;
        acc = op == '+' ? acc + t : acc - t;
    }
    if (*s) return 0;
    *out = acc;
    return 1;
}

/*
 * parse_or_die -- parse_u128() for command-line and state-file numbers.
 *
 * In:     s.
 * Out:    the value.
 * Errors: on a syntax error prints "bad number: <s>" and exits with status 1.
 */
static u128 parse_or_die(const char *s)
{
    u128 v;
    if (!parse_u128(s, &v)) { fprintf(stderr, "bad number: %s\n", s); exit(1); }
    return v;
}

/* gcd64 -- greatest common divisor (Euclid); gcd(a, 0) = a, gcd(0, 0) = 0.
   No error conditions. */
static u64 gcd64(u64 a, u64 b)
{
    while (b) { u64 t = a % b; a = b; b = t; }
    return a;
}

/*
 * modinv64 -- inverse of a modulo m, by the extended Euclidean algorithm in
 * signed 128-bit arithmetic (so the intermediate values cannot overflow).
 *
 * In:     a; the modulus m >= 1.
 * Out:    the x in [0, m) with a x = 1 (mod m); 0 if gcd(a, m) != 1, and 0
 *         for m = 1 (which is the inverse mod 1).
 * Errors: m = 0 divides by zero.  0 doubles as the failure value; bctx_set()
 *         only passes units, so it never gets the failure case.
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

/*
 * mod128_64 -- M mod b for a 128-bit M and a 64-bit b.
 *
 * In:     M; b != 0.
 * Out:    M mod b.
 * How:    M < 2^64: one 64-bit division.  b < 2^32: reduce the high word,
 *         then the two 32-bit halves of the low word, each time as
 *         (r << 32 | half) mod b, which fits in 64 bits because r < b < 2^32;
 *         this avoids the slow 128-bit division.  Otherwise a 128-bit '%'.
 * Errors: b = 0 divides by zero.
 */
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

/* ------------------------------------------------------------------ */
/* palindrome tests                                                    */
/* ------------------------------------------------------------------ */

/*
 * is_pal_base -- is M an L-digit palindrome in base b?
 *
 * In:     M; the base b; the length L (at most 130).
 * Out:    1 if M has exactly L base-b digits (b^(L-1) <= M < b^L) and they
 *         read the same in both directions, else 0.  The conversion stops as
 *         soon as M turns out to have more than L digits.
 * Errors: none; b < 2 and M = 0 give 0.
 */
static int is_pal_base(u128 M, u64 b, int L)
{
    u64 d[130];                         /* d[0] = least significant digit */
    int len = 0;
    if (b < 2) return 0;
    if ((M >> 64) == 0) {               /* 64-bit divisions are much faster */
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
 * find_pal_base -- the smallest base in which M is an L-digit palindrome.
 *
 * In:     M; the length L.
 * Out:    for L >= 3: the smallest b >= 2 in which M is an L-digit
 *         palindrome, or 0 if there is none.  For L = 2 only existence
 *         matters: the result is M-1 (M = "11" in base M-1; capped at
 *         2^64-1), which need not be the smallest base, or 0 for M < 3.
 *         0 for L < 2 or M = 0.
 * How:    scans the bases with b^(L-1) <= M < b^L, i.e. b from
 *         floor(M^(1/L))+1 to floor(M^(1/(L-1))).  Per base, the last digit
 *         r = M mod b must equal the leading digit, which is >= 1; a
 *         double-precision quotient M / b^(L-1) skips the bases whose leading
 *         digit is clearly not r; then the exact leading digit and finally
 *         all digits are compared.  The double filter never skips a real
 *         match: the leading digit is below M^(1/L) <= 2^43, so the rounding
 *         error (relative 2^-51 at most) is below 2^-8, far inside the 1.5
 *         margin.
 * Errors: none.  The cost is the number of bases scanned, about
 *         M^(1/(L-1)) - M^(1/L): small for long lengths, but up to 2^45
 *         bases for L = 3 and M ~ 2^90.  In a search only survivors that
 *         pass every longer length get to L = 3 (so far only the solutions,
 *         where the scan stops at the first base found: about 10^9 bases,
 *         seconds, for 2^90).
 */
static u64 find_pal_base(u128 M, int L)
{
    if (L < 2 || M == 0) return 0;
    if (L == 2) {                       /* "11" in base M-1; only existence matters */
        if (M < 3) return 0;
        return (M - 1 > UINT64_MAX) ? UINT64_MAX : (u64)(M - 1);
    }
    u64 blo = iroot(M, L) + 1, bhi = iroot(M, L - 1);   /* exactly L digits */
    if (blo < 2) blo = 2;
    double Md = (double)M;
    for (u64 b = blo; b <= bhi; b++) {
        u64 r = mod128_64(M, b);        /* last digit = leading digit */
        if (r == 0) continue;           /* the leading digit is never 0 */
        double Pd = pow((double)b, (double)(L - 1));
        double la = Md / Pd;            /* approximately the leading digit */
        if (la < (double)r - 1.5 || la > (double)r + 1.5) continue;
        u128 P = pow_sat(b, L - 1);
        if (M / P != r) continue;       /* exact leading digit */
        if (is_pal_base(M, b, L)) return b;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* problem setup                                                       */
/* ------------------------------------------------------------------ */

/*
 * prob_t -- one search: the order n, the range [lo, hi] and its work units.
 *
 *   n        the order: palindromes of lengths 2..n are needed
 *   h        n/2, the number of mirrored digit pairs of an n-digit palindrome
 *   D        ceil(n/2), its number of free digits d_0..d_{D-1}
 *   k        n-2-h, the number of base-c digit pairs the congruence imposes
 *   bd       D-3, the number of block digits (the free digits before x, y, z)
 *   ud       max(1, bd-1), the number of block digits a unit fixes; the
 *            other bd-ud block digits (0 or 1) are looped over in do_unit
 *   lo, hi   the range of M (lo >= 1)
 *   Bmin, Bmax  the bases B that have n-digit numbers in [lo, hi]
 *   uoff     uoff[B-Bmin] = index of the first unit of base B, and
 *            uoff[Bmax-Bmin+1] = nunits; base B has (B-1) B^(ud-1) units
 *   nunits   the total number of units
 *
 * prob_init() fills it in; the caller frees uoff.
 */
typedef struct {
    int n, h, D, k, bd, ud;     /* length, pairs, free digits, base-c digits used, block digits, unit digits */
    u128 lo, hi;
    u64 Bmin, Bmax;
    u64 *uoff;                  /* uoff[B-Bmin] = first unit of base B */
    u64 nunits;
} prob_t;

/*
 * cconst_t -- the constants of one base pair (B, c), built by bctx_set()
 * (theory, sections 3-6).  All values fit in 64 bits (P < 2^62, N < 2^42).
 *
 *   c          the base of the (n-1)-digit condition
 *   N          c^k, the modulus of the congruence
 *   P          c^(h+1): q = floor(M / P) holds the k leading base-c digits
 *   g, Np      g = gcd(s, N) and N' = N/g
 *   inv        u = s''^(-1) mod N, s'' a unit congruent to s/g mod N'
 *   limit      g B: w < limit is the per-step test (smallest z = w/g < B)
 *   Wy, NWy    w_y u mod N and N - Wy (subtracting Wy mod N = adding NWy)
 *   Wx, NWx    the same for dx = w_x - (B-1) w_y, the step at a y wrap
 *   Rcross     P - (B-1) s: region 1 (q+1) is reachable iff rem >= Rcross
 *   Wc[j]      c^j u mod N, for the change of T when q is incremented
 */
typedef struct {
    u64 c, N, P, g, Np, inv, limit, Wy, NWy, Wx, NWx, Rcross;
    u64 Wc[MAXK];
} cconst_t;

/*
 * bctx_t -- everything that depends only on the base B, plus the cconst_t of
 * every base c that can occur with B.  One per thread; bctx_set() rebuilds
 * it when the thread's next unit has another B (units come in order of B,
 * so that is rare).
 *
 *   B            the base of the n-digit palindromes
 *   w[i]         the weights w_0..w_{D-1} of the free digits
 *   s, wy, wx    the weights of z, y and x (s and wy are < P, so 64 bits)
 *   dx           w_x - (B-1) w_y: the change of M0 when y wraps and x steps
 *   span_block   (B-1)(w_x + w_y + s): a block's M lie in [Mblock, Mblock + span_block]
 *   span_unit    the same for all digits a unit does not fix
 *   cbase, ccount  the bases c of the table: tab[c - cbase], ccount entries
 *   tab, tabcap  the cconst_t table and its allocated length
 */
typedef struct {
    u64 B;
    u128 w[MAXN];
    u64 s, wy, dx;              /* innermost weight, y weight, x-wrap delta */
    u128 wx;
    u128 span_block, span_unit;
    u64 cbase, ccount;          /* table of c constants */
    cconst_t *tab;
    size_t tabcap;
} bctx_t;

/* pair_t -- a pair (M, c) recorded by the selftest hooks (wstat_t.record). */
typedef struct { u128 M; u64 c; } pair_t;

/*
 * wstat_t -- statistics (theory, section 8) plus test hooks.  A worker
 * collects one unit's statistics in a wstat_t of its own and hands them to
 * finish_unit(); add_stats() adds two of them.
 *
 *   pr       the problem
 *   tuples, hits, st2, canon, csum, fail[], sol   see section 8
 *   record   selftest hook: 0 = normal search; 1 = store every (M, c) with M
 *            an (n-1)-digit palindrome in base c and stop there; 2 = store
 *            every hit with c^(n-2) <= M < c^(n-1) (every k-digit match)
 *   pairs, np, pcap   the stored pairs, their count and the capacity
 *   verbose  -v: print a DEEP line for survivors that pass length n-2
 *   pchk, pz (PARANOID) brute-force checked (x, y) steps and the z hits
 *            among them
 */
typedef struct {
    const prob_t *pr;
    u64 tuples, hits, st2, canon;
    u64 csum;                   /* order-independent checksum of the canonical survivors */
    u64 fail[MAXN + 1];         /* fail[L]: canonical survivors that failed at length L */
    u64 sol;
    int record;                 /* test hook (0, 1, 2; see above): record (M, c) pairs instead of testing them */
    pair_t *pairs;
    size_t np, pcap;
    int verbose;
#ifdef PARANOID
    u64 pchk, pz;               /* brute-force steps checked, z hits checked */
#endif
} wstat_t;

/* add_stats -- *t += *s for every statistic (not the test fields).  Used to
   collect finished units and to add a resumed state's totals.  No errors. */
static void add_stats(wstat_t *t, const wstat_t *s)
{
    t->tuples += s->tuples; t->hits += s->hits; t->st2 += s->st2; t->canon += s->canon;
    t->csum += s->csum; t->sol += s->sol;
    for (int L = 0; L <= MAXN; L++) t->fail[L] += s->fail[L];
}

static pthread_mutex_t out_lock = PTHREAD_MUTEX_INITIALIZER;  /* one output line at a time */
static FILE *sol_out;           /* solutions go here (stdout; a temporary file in the selftest) */

/*
 * prob_init -- set up the problem (n, [lo, hi]) and number its work units.
 *
 * In:     pr (overwritten); n (7..20 for the fast search); lo, hi (lo < 1 is
 *         raised to 1).
 * Out:    *pr filled in (see prob_t); pr->uoff is allocated, the caller
 *         frees it.
 * How:    the bases B are those with an n-digit number in [lo, hi]:
 *         B^(n-1) <= hi and B^n - 1 >= lo.  Base B has (B-1) B^(ud-1) units
 *         (leading digit 1..B-1, other unit digits 0..B-1), numbered
 *         consecutively in order of B, so uoff[] is a prefix sum.
 * Errors: none reported; calloc failure is not checked.  A range without
 *         n-digit numbers gives nunits = 0 (see the header, section 10).
 *         For n < 7 the digit split is meaningless (bd = D-3 < 1); the
 *         callers use naive_search() there.
 */
static void prob_init(prob_t *pr, int n, u128 lo, u128 hi)
{
    memset(pr, 0, sizeof *pr);
    pr->n = n;
    pr->h = n / 2;
    pr->D = (n + 1) / 2;
    pr->k = n - 2 - pr->h;
    pr->bd = pr->D - 3;
    pr->ud = pr->bd - 1 >= 1 ? pr->bd - 1 : 1;
    if (lo < 1) lo = 1;
    pr->lo = lo;
    pr->hi = hi;
    /* B with B^(n-1) <= hi and B^n - 1 >= lo */
    u64 Bmax = iroot(hi, n - 1);
    u64 Bmin = 2;
    while (Bmin <= Bmax && pow_sat(Bmin, n) - 1 < lo) Bmin++;
    if (Bmin < 2) Bmin = 2;
    pr->Bmin = Bmin;
    pr->Bmax = Bmax;
    u64 nb = Bmax >= Bmin ? Bmax - Bmin + 1 : 0;
    pr->uoff = calloc(nb + 1, sizeof(u64));
    u64 tot = 0;
    for (u64 i = 0; i < nb; i++) {          /* prefix sum of the units per base */
        u64 B = Bmin + i;
        pr->uoff[i] = tot;
        u128 cnt = (u128)(B - 1) * pow_sat(B, pr->ud - 1);
        tot += (u64)cnt;
    }
    pr->uoff[nb] = tot;
    pr->nunits = tot;
}

/*
 * bctx_set -- build the context of base B and the constants of every (B, c).
 *
 * In:     bc (reused: its table grows as needed, the rest is overwritten);
 *         pr; the base B.
 * Out:    bc->w[], s, wy, wx, dx, the spans, and bc->tab[c - cbase] for
 *         every c in [cbase, cbase + ccount) (see cconst_t).
 * How:    c runs over the bases in which an n-digit base-B number in
 *         [lo, hi] can have n-1 digits: M in [max(B^(n-1), lo),
 *         min(B^n - 1, hi)] and c^(n-2) <= M < c^(n-1).  Per c: N = c^k,
 *         P = c^(h+1), g = gcd(s mod N, N), N' = N/g, and the unit
 *         s'' = s/g + j N' with the smallest j >= 0 that is coprime to N (it
 *         exists by the Chinese remainder theorem: only primes dividing g
 *         but not N' need avoiding, one residue class of j each), u = its
 *         inverse, and the step constants derived from u.
 * Errors: exits with status 1 if c^(h+1) >= 2^62 or c^k >= 2^42 ("too large
 *         for 64-bit state": the hot loop keeps rem, w and g B in 64 bits),
 *         or if (B-1) s, w_y or dx is not below P ("weight bound violated";
 *         impossible since c > B, but the "q changes by at most one"
 *         argument depends on it).  malloc failure is not checked.
 */
static void bctx_set(bctx_t *bc, const prob_t *pr, u64 B)
{
    int n = pr->n, h = pr->h, D = pr->D;
    bc->B = B;
    for (int i = 0; i < h; i++) bc->w[i] = pow_sat(B, n - 1 - i) + pow_sat(B, i);
    if (n & 1) bc->w[h] = pow_sat(B, h);   /* middle digit */
    bc->s = (u64)bc->w[D - 1];             /* z = d_{D-1} */
    bc->wy = (u64)bc->w[D - 2];            /* y = d_{D-2} */
    bc->wx = bc->w[D - 3];                 /* x = d_{D-3} */
    bc->dx = (u64)(bc->w[D - 3] - (u128)(B - 1) * bc->w[D - 2]);   /* > 0: w_x > (B-1) w_y */
    bc->span_block = (u128)(B - 1) * (bc->w[D - 3] + bc->w[D - 2] + bc->w[D - 1]);
    u128 su = 0;
    for (int i = pr->ud; i < D; i++) su += bc->w[i];
    bc->span_unit = (u128)(B - 1) * su;

    /* c range for this B: some M in [Mlo, Mhi] must have n-1 digits in base c */
    u128 lo = pr->lo, hi = pr->hi;
    u128 Mlo = pow_sat(B, n - 1), Mhi = pow_sat(B, n) - 1;
    if (Mlo < lo) Mlo = lo;
    if (Mhi > hi) Mhi = hi;
    u64 cmin = iroot(Mlo, n - 1) + 1, cmax = iroot(Mhi, n - 2);
    if (cmax < cmin) cmax = cmin;          /* keep one entry; do_block clips c per block */
    bc->cbase = cmin;
    bc->ccount = cmax - cmin + 1;
    if (bc->ccount > bc->tabcap) {
        free(bc->tab);
        bc->tabcap = bc->ccount;
        bc->tab = malloc(bc->tabcap * sizeof(cconst_t));
    }
    int k = pr->k;
    for (u64 i = 0; i < bc->ccount; i++) {
        u64 c = cmin + i;
        cconst_t *K = &bc->tab[i];
        u128 P128 = pow_sat(c, h + 1), N128 = pow_sat(c, k);
        if (P128 >= ((u128)1 << 62) || N128 >= ((u128)1 << 42)) {
            fprintf(stderr, "c = %llu too large for 64-bit state (n = %d)\n", (unsigned long long)c, n);
            exit(1);
        }
        u64 P = (u64)P128, N = (u64)N128;
        if ((u128)(B - 1) * bc->s >= P || bc->wy >= P || bc->dx >= P) {
            fprintf(stderr, "internal: weight bound violated (B=%llu c=%llu)\n",
                    (unsigned long long)B, (unsigned long long)c);
            exit(1);
        }
        K->c = c;
        K->N = N;
        K->P = P;
        /* z s = t (mod N) has solutions iff g | t: z = (t/g)(s/g)^(-1) (mod N') */
        u64 sN = bc->s % N;
        u64 g = gcd64(sN, N);          /* gcd(0, N) = N */
        u64 Np = N / g;
        u64 sp = Np > 1 ? (sN / g) % Np : 0;     /* s/g mod N' (a unit mod N') */
        u64 spp = sp;
        while (gcd64(spp % N, N) != 1) spp += Np; /* lift it to a unit mod N */
        K->g = g;
        K->Np = Np;
        K->inv = modinv64(spp % N, N); /* u: also (s/g)^(-1) mod N' */
        K->limit = g * B;              /* w < g B  <=>  w/g < B */
        K->Wy = mulmod64(bc->wy % N, K->inv, N);
        K->NWy = N - K->Wy;
        K->Wx = mulmod64(bc->dx % N, K->inv, N);
        K->NWx = N - K->Wx;
        u64 cj = 1;
        for (int j = 0; j < k; j++) { K->Wc[j] = mulmod64(cj % N, K->inv, N); cj *= c; }
        K->Rcross = P - (B - 1) * bc->s;   /* rem >= Rcross: some z reaches q+1 */
    }
}

/* ------------------------------------------------------------------ */
/* survivors                                                           */
/* ------------------------------------------------------------------ */

/*
 * report_solution -- print one SOLUTION line: M, one base per length 2..n
 * (M-1 for L = 2, the smallest base for L >= 3) and the pair (B, c) that
 * found it.
 *
 * In:     st (for n); M; B; c.
 * Out:    one line on sol_out, written under out_lock and flushed at once.
 * Errors: none.  It repeats find_pal_base() for every length, which takes
 *         seconds for L = 3 and M ~ 2^90 (see find_pal_base).
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

/*
 * surv_hash -- 64-bit hash of a canonical survivor M.  The run checksum is
 * the sum of these hashes mod 2^64, so it does not depend on the order in
 * which survivors are found.  a171775_2d.c and the GPU ports use the same
 * function; changing it changes every checksum in the logs and README.
 * No error conditions.
 */
static inline u64 surv_hash(u128 M)
{
    u64 a = (u64)M * 0x9E3779B97F4A7C15ULL, b = (u64)(M >> 64) * 0xC2B2AE3D27D4EB4FULL;
    a ^= a >> 29;
    return a * 0xBF58476D1CE4E5B9ULL + b;
}

/*
 * candidate -- check one hit of the fast filter in full (theory, section 7).
 *
 * In:     st (statistics and test mode); M = M0 + z s, an n-digit palindrome
 *         in base B whose k outer base-c digit pairs match; B; c.
 * Out:    st->hits is always counted.  Then, for M in [lo, hi]:
 *           record = 2: store (M, c) if M has n-1 digits in base c;
 *           otherwise, if M is an (n-1)-digit palindrome in base c, st2++
 *           and: record = 1 stores (M, c); record = 0 continues only for
 *           the canonical pair (c = find_pal_base(M, n-1) and
 *           B = find_pal_base(M, n)), counts it (canon, csum) and tests the
 *           lengths n-2 down to 3.  The first failing length is counted in
 *           fail[L] (with -v and L <= n-3 also printed as a DEEP line); if
 *           none fails, sol++ and report_solution().
 * Errors: none; realloc failure in the record modes is not checked.
 */
static void candidate(wstat_t *st, u128 M, u64 B, u64 c)
{
    const prob_t *pr = st->pr;
    st->hits++;
    if (M < pr->lo || M > pr->hi) return;
    int n = pr->n;
    if (st->record == 2) {              /* test hook: every k-digit match with n-1 digits */
        if (pow_sat(c, n - 2) <= M && M < pow_sat(c, n - 1)) {
            if (st->np == st->pcap) {
                st->pcap = st->pcap ? 2 * st->pcap : 1024;
                st->pairs = realloc(st->pairs, st->pcap * sizeof(pair_t));
            }
            st->pairs[st->np].M = M;
            st->pairs[st->np].c = c;
            st->np++;
        }
        return;
    }
    if (!is_pal_base(M, c, n - 1)) return;
    st->st2++;
    if (st->record) {
        if (st->np == st->pcap) {
            st->pcap = st->pcap ? 2 * st->pcap : 1024;
            st->pairs = realloc(st->pairs, st->pcap * sizeof(pair_t));
        }
        st->pairs[st->np].M = M;
        st->pairs[st->np].c = c;
        st->np++;
        return;
    }
    /* canonical: c and B are the smallest bases of their lengths */
    if (find_pal_base(M, n - 1) != c) return;
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
/* the fast filter: one block, one base c                              */
/* ------------------------------------------------------------------ */

#ifdef PARANOID
/* debug build: recompute the incremental state from scratch at every step */

/* T_of -- (PARANOID) T(q): the k base-c digits of q (q < c^k) in reverse
   order, read as a number.  No error conditions. */
static u64 T_of(u64 q, u64 c, int k)
{
    u64 E[MAXK], T = 0;
    for (int j = k - 1; j >= 0; j--) { E[j] = q % c; q /= c; }
    for (int j = k - 1; j >= 0; j--) T = T * c + E[j];
    return T;
}

/*
 * paranoid_state -- (PARANOID) recompute block_c's incremental state for M0
 * from scratch and compare.
 *
 * In:     pr, bc, K; M0 = Mblock + x w_x + y w_y; block_c's state for it:
 *         w, rem, the digits E[0..k-2], elast (= E[k-1]), wnext, last.
 * Out:    returns if everything agrees.
 * Errors: on a mismatch prints the bit mask (1 rem, 2 q >= c^k, 4 elast,
 *         8 E[], 16 w, 32 wnext, 64 last) and calls abort().
 */
static void paranoid_state(const prob_t *pr, const bctx_t *bc, const cconst_t *K, u128 M0, u64 w, u64 rem,
                           const u64 *E, u64 elast, u64 wnext, int last)
{
    const int k = pr->k;
    const u64 c = K->c, N = K->N, P = K->P;
    char buf[48];
    u128 q = M0 / P;
    int bad = 0;
    if ((u64)(M0 % P) != rem) bad |= 1;
    if (q >= N) bad |= 2;
    else {
        u64 qq = (u64)q, Et[MAXK];
        for (int j = k - 1; j >= 0; j--) { Et[j] = qq % c; qq /= c; }
        if (Et[k - 1] != elast) bad |= 4;
        for (int j = 0; j < k - 1; j++)
            if (Et[j] != E[j]) bad |= 8;
        u64 T = T_of((u64)q, c, k), MN = (u64)(M0 % N);
        u64 wt = mulmod64(T >= MN ? T - MN : T + N - MN, K->inv, N);
        if (wt != w) bad |= 16;
        if ((u64)q + 1 < N) {
            u64 T1 = T_of((u64)q + 1, c, k);
            u64 wn = mulmod64(T1 >= T ? T1 - T : T1 + N - T, K->inv, N);
            if (wn != wnext) bad |= 32;
            if (last) bad |= 64;
        } else if (!last) bad |= 64;
    }
    if (bad) {
        fprintf(stderr, "PARANOID state mismatch %#x: n=%d B=%llu c=%llu M0=%s\n", bad, pr->n,
                (unsigned long long)bc->B, (unsigned long long)c, u128s(M0, buf));
        abort();
    }
}

/* u64cmp -- (PARANOID) qsort() comparator for u64 values. */
static int u64cmp(const void *a, const void *b)
{
    u64 x = *(const u64 *)a, y = *(const u64 *)b;
    return x < y ? -1 : x > y;
}
#endif

/*
 * block_c -- the hot loop: every (x, y, z) of one block for one base c
 * (theory, sections 3-6).
 *
 * In:     st      the calling thread's statistics
 *         bc      the context of base B (weights)
 *         K       the constants of (B, c)
 *         Mblock  the value of the block digits (x = y = z = 0)
 * Out:    calls candidate() for every z in [0, B) for which M = Mblock +
 *         x w_x + y w_y + z s satisfies M = T(floor(M / P)) (mod c^k), i.e.
 *         whose k outer base-c digit pairs match.  Returns early once
 *         M >= c^(n-1) for all remaining (x, y).
 * Errors: none (a PARANOID build aborts on any state mismatch).
 *
 * State for the current (x, y), M0 = Mblock + x w_x + y w_y:
 *   rem     M0 mod P
 *   q       floor(M0 / P) < c^k, kept as its base-c digits: E[0..k-2] and
 *           elast for E[k-1] (the stored E[k-1] is refreshed only when
 *           elast reaches c-1 or c)
 *   w       (T(q) - M0) u mod N
 *   wnext   (T(q+1) - T(q)) u mod N; last = 1 if q = c^k - 1 (no q+1)
 *   trig    the elast value at which the slow path runs: c-1 while
 *           E[k-1] < c-1 (the slow path then sets up wnext for the coming
 *           carry), c once E[k-1] = c-1 (it then performs the carry)
 */
static void block_c(wstat_t *st, const bctx_t *bc, const cconst_t *K, u128 Mblock)
{
    const prob_t *pr = st->pr;
    const int k = pr->k;
    const u64 B = bc->B, c = K->c, N = K->N, P = K->P, limit = K->limit;
    const u64 Wy = K->Wy, NWy = K->NWy, Wx = K->Wx, NWx = K->NWx;
    const u64 wy = bc->wy, dx = bc->dx, s = bc->s, Rcross = K->Rcross;
    const u64 g = K->g, Np = K->Np;

    /* initial state for (x, y) = (0, 0), i.e. M0 = Mblock; after this the loop
       divides only on hits */
    u64 rem = (u64)(Mblock % P);
    u128 q128 = Mblock / P;
    if (q128 >= N) return;             /* M >= c^(n-1) already */
    u64 q = (u64)q128;
    u64 E[MAXK];
    for (int j = k - 1; j >= 0; j--) { E[j] = q % c; q /= c; }   /* E[0] = leading digit */
    u64 T = 0;
    for (int j = k - 1; j >= 0; j--) T = T * c + E[j];          /* T(q): digits reversed */
    u64 MN = (u64)(Mblock % N);
    u64 t = T >= MN ? T - MN : T + N - MN;
    u64 w = mulmod64(t, K->inv, N);    /* (T - M0) u mod N */

    u64 wnext;
    int last;
    u64 elast, trig;
    /* q -> q+1 with the carry stopping at digit m (E[m+1..k-1] = c-1 wrap to 0,
       E[m] += 1) changes T by -(c-1)(c^(m+1) + ... + c^(k-1)) + c^m, which is
       c^(m+1) + c^m (mod c^k) for m < k-1 and c^(k-1) for m = k-1. */
    /* RECOMPUTE_NEXT(): from the stored digits E[] (E[k-1] must be current),
       find that m (jj) and set wnext = (change of T) u mod N from Wc[]; if all
       digits are c-1 there is no q+1 below c^k: last = 1, wnext = 0. */
#define RECOMPUTE_NEXT()                                                        \
    do {                                                                        \
        int jj = k - 1;                                                         \
        while (jj >= 0 && E[jj] == c - 1) jj--;                                 \
        last = jj < 0;                                                          \
        if (jj == k - 1) wnext = K->Wc[k - 1];                                  \
        else if (jj >= 0) {                                                     \
            wnext = K->Wc[jj + 1] + K->Wc[jj];                                  \
            if (wnext >= N) wnext -= N;                                         \
        } else wnext = 0;                                                       \
    } while (0)
    RECOMPUTE_NEXT();
    elast = E[k - 1];                  /* the fast-path copy of the last digit */
    trig = (elast == c - 1) ? c : c - 1;

    /* CAND(z): hand M = M0 + z s to candidate() (a PARANOID build also notes z
       for the brute-force comparison) */
#ifdef PARANOID
#define CAND(zv)                                                                \
    do {                                                                        \
        if (nfound < 256) found[nfound++] = (zv);                               \
        candidate(st, M0 + (u128)(zv) * s, B, c);                               \
    } while (0)
#else
#define CAND(zv) candidate(st, M0 + (u128)(zv) * s, B, c)
#endif
    for (u64 x = 0; x < B; x++) {
        for (u64 y = 0; y < B; y++) {
#ifdef PARANOID
            u64 found[256];
            int nfound = 0;
            {
                u128 M0p = Mblock + (u128)x * bc->wx + (u128)y * wy;
                paranoid_state(pr, bc, K, M0p, w, rem, E, elast, wnext, last);
            }
#endif
            u64 w1 = w + wnext;            /* the w of region 1, i.e. of T(q+1) */
            w1 = (w1 >= N) ? w1 - N : w1;
            /* the whole per-step test: a smallest solution w/g < B in region 0,
               or in region 1 if some z reaches it (branch-free) */
            int hit = (w < limit) | ((rem >= Rcross) & (w1 < limit));
            if (__builtin_expect(hit, 0)) {
                u128 M0 = Mblock + (u128)x * bc->wx + (u128)y * wy;
                /* region 0: z = w/g + i N' while M0 + z s stays below (q+1) P */
                if (w < limit && (g == 1 || w % g == 0)) {
                    for (u64 z = w / g; z < B; z += Np) {
                        if (rem + z * s >= P) break;
                        CAND(z);
                    }
                }
                /* region 1 (not if q = c^k - 1): z = w1/g + i N', from the first
                   z with rem + z s >= P on (no upper limit: (B-1) s < P) */
                if (!last && rem >= Rcross && w1 < limit && (g == 1 || w1 % g == 0)) {
                    u64 z = w1 / g;
                    u64 zs = (P - rem + s - 1) / s;    /* first z in the q+1 range */
                    if (z < zs) z += ((zs - z + Np - 1) / Np) * Np;
                    for (; z < B; z += Np) CAND(z);
                }
            }
#ifdef PARANOID
            /* compare the z found above with all z that satisfy the congruence,
               on every step for B <= 64 and every 97th step otherwise */
            if (B <= 64 || (x * B + y) % 97 == 0) {   /* brute force over z */
                u128 M0 = Mblock + (u128)x * bc->wx + (u128)y * wy;
                u64 want[256];
                int nw = 0;
                for (u64 z = 0; z < B; z++) {
                    u128 M = M0 + (u128)z * s;
                    u128 qz = M / P;
                    if (qz >= N) continue;
                    if ((u64)(M % N) == T_of((u64)qz, c, k) && nw < 256) want[nw++] = z;
                }
                qsort(found, nfound, sizeof(u64), u64cmp);
                int same = nw == nfound;
                for (int i = 0; same && i < nw; i++) same = want[i] == found[i];
                if (!same) {
                    char buf[48];
                    fprintf(stderr, "PARANOID z mismatch: n=%d B=%llu c=%llu M0=%s fast %d brute %d\n", pr->n,
                            (unsigned long long)B, (unsigned long long)c, u128s(M0, buf), nfound, nw);
                    abort();
                }
                st->pchk++;                        /* count checked steps */
                st->pz += nw;                      /* count checked z hits */
            }
#endif
            /* advance to the next (x, y): M0 grows by dR = w_y, or by dx when
               y wraps to 0 and x steps */
            u64 dW, nW, dR;
            if (y + 1 < B) { dW = Wy; nW = NWy; dR = wy; }
            else { dW = Wx; nW = NWx; dR = dx; }
            w = (w >= dW) ? w - dW : w + nW;   /* w -= dR u (mod N) */
            rem += dR;
            u64 inc = rem >= P;                /* q -> q+1 (at most once: dR < P) */
            rem -= inc ? P : 0;
            u64 wa = w + (inc ? wnext : 0);    /* T(q) -> T(q+1) */
            w = (wa >= N) ? wa - N : wa;
            elast += inc;
            /* slow path, about once per c increments of q: E[k-1] reaches c-1
               (prepare the carry) or c (do it) */
            if (__builtin_expect(elast >= trig, 0)) {
                if (elast == c - 1) {          /* next increment carries */
                    E[k - 1] = c - 1;
                    RECOMPUTE_NEXT();
                    trig = c;
                } else {                       /* elast == c: carry happened */
                    if (last) goto done;       /* q reached c^k: M >= c^(n-1) from here on */
                    E[k - 1] = 0;
                    int j = k - 2;             /* propagate the carry into E[k-2], E[k-3], ... */
                    while (j >= 0) {
                        E[j]++;
                        if (E[j] < c) break;
                        E[j] = 0;
                        j--;
                    }
                    if (j < 0) goto done;      /* defensive: "last" already covers this */
                    elast = 0;
                    RECOMPUTE_NEXT();
                    trig = c - 1;
                }
            }
        }
    }
done:
#undef RECOMPUTE_NEXT
#undef CAND
    return;
}

/*
 * do_block -- process one block for all its bases c.
 *
 * In:     st; bc (the context of B); Mblock.
 * Out:    clips the block's range [Mblock, Mblock + span_block] to [lo, hi],
 *         runs block_c() for every c from floor(Mmin^(1/(n-1)))+1 to
 *         floor(Mmax^(1/(n-2))) (the bases in which some M of the block has
 *         n-1 digits) that is in the table of bctx_set(), and adds
 *         (number of c) B^2 to st->tuples.
 * Errors: none; a block entirely outside [lo, hi] is skipped.
 */
static void do_block(wstat_t *st, const bctx_t *bc, u128 Mblock)
{
    const prob_t *pr = st->pr;
    u128 Mmin = Mblock, Mmax = Mblock + bc->span_block;
    if (Mmax < pr->lo || Mmin > pr->hi) return;
    if (Mmin < pr->lo) Mmin = pr->lo;
    if (Mmax > pr->hi) Mmax = pr->hi;
    u64 clo = iroot(Mmin, pr->n - 1) + 1, chi = iroot(Mmax, pr->n - 2);
    if (clo < bc->cbase) clo = bc->cbase;
    if (chi > bc->cbase + bc->ccount - 1) chi = bc->cbase + bc->ccount - 1;
    for (u64 c = clo; c <= chi; c++) block_c(st, bc, &bc->tab[c - bc->cbase], Mblock);
    if (chi >= clo) st->tuples += (chi - clo + 1) * bc->B * bc->B;
}

/*
 * unit_decode -- unit index -> base B and the digits the unit fixes.
 *
 * In:     pr; u < pr->nunits; pref with room for ud digits.
 * Out:    returns B, found by binary search for uoff[i] <= u < uoff[i+1];
 *         the local index u - uoff[i] is read as the mixed-radix number
 *         (d_0 - 1, d_1, ..., d_{ud-1}) in base B: pref[0] = d_0 in 1..B-1,
 *         pref[1..ud-1] in 0..B-1.
 * Errors: u is not range-checked (u >= nunits gives the last base and
 *         meaningless digits).  A problem without units (Bmax < Bmin) gives
 *         Bmin and the digits 1, 0, ...; cmd_search never gets that far.
 */
static u64 unit_decode(const prob_t *pr, u64 u, u64 *pref)
{
    if (pr->Bmax < pr->Bmin) {              /* no bases at all: avoid the underflow below */
        pref[0] = 1;
        for (int i = 1; i < pr->ud; i++) pref[i] = 0;
        return pr->Bmin;
    }
    u64 lo = 0, hi = pr->Bmax - pr->Bmin;   /* find i with uoff[i] <= u < uoff[i+1] */
    while (lo < hi) {
        u64 mid = (lo + hi + 1) / 2;
        if (pr->uoff[mid] <= u) lo = mid; else hi = mid - 1;
    }
    u64 B = pr->Bmin + lo;
    u64 local = u - pr->uoff[lo];
    for (int i = pr->ud - 1; i >= 1; i--) { pref[i] = local % B; local /= B; }
    pref[0] = 1 + local;
    return B;
}

/*
 * do_unit -- process one work unit: all blocks that start with the unit's
 * digits.
 *
 * In:     st; bc (the thread's context, rebuilt if the unit has another B);
 *         the unit index u.
 * Out:    do_block() for the unit's block, or, if the unit fixes one block
 *         digit less than a block has (n >= 9), for each of the B values of
 *         that digit.  Units whose whole range misses [lo, hi] are skipped.
 * Errors: as bctx_set() (exit on a too large c).
 */
static void do_unit(wstat_t *st, bctx_t *bc, u64 u)
{
    const prob_t *pr = st->pr;
    u64 pref[MAXN];
    u64 B = unit_decode(pr, u, pref);
    if (bc->B != B) bctx_set(bc, pr, B);
    u128 Mu = 0;                       /* value of the unit's digits */
    for (int i = 0; i < pr->ud; i++) Mu += (u128)pref[i] * bc->w[i];
    if (Mu + bc->span_unit < pr->lo || Mu > pr->hi) return;
    int nb = pr->bd - pr->ud;          /* extra block digits: 0 or 1 */
    if (nb == 0) {
        do_block(st, bc, Mu);
    } else {
        u128 wb = bc->w[pr->ud];
        for (u64 d = 0; d < B; d++) do_block(st, bc, Mu + (u128)d * wb);
    }
}

/* ------------------------------------------------------------------ */
/* naive references                                                    */
/* ------------------------------------------------------------------ */

/*
 * ends_mirror -- does M have exactly L base-c digits whose k outer pairs
 * match (digit i = digit L-1-i for i < k)?  The naive version of a hit.
 *
 * In:     M; c >= 2; L; k <= L/2.
 * Out:    1 or 0.
 * Errors: none (M < 2^128 has at most 128 digits, so d[] cannot overflow).
 */
static int ends_mirror(u128 M, u64 c, int L, int k)
{
    u64 d[130];
    int len = 0;
    u128 x = M;
    while (x) { d[len++] = (u64)(x % c); x /= c; }
    if (len != L) return 0;
    for (int i = 0; i < k; i++)
        if (d[i] != d[L - 1 - i]) return 0;
    return 1;
}

/*
 * naive_block_pairs -- brute-force reference for one block: every (M, c)
 * with M = Mblock + x w_x + y w_y + z s in [lo, hi] (all x, y, z < B) and
 * c^(n-2) <= M < c^(n-1), such that
 *   mode 1: M is an (n-1)-digit palindrome in base c;
 *   mode 2: the k outer digit pairs of M in base c match (ends_mirror).
 *
 * In:     pr, bc (the context of B), Mblock, mode.
 * Out:    *out = a malloc'ed array of the pairs (the caller frees it),
 *         *np = their number.
 * Errors: none; allocation failures are not checked.  Cost B^3 times the
 *         number of bases c, so only for small B or narrow ranges.
 */
static void naive_block_pairs(const prob_t *pr, const bctx_t *bc, u128 Mblock, pair_t **out, size_t *np, int mode)
{
    size_t cap = 1024, cnt = 0;
    pair_t *p = malloc(cap * sizeof(pair_t));
    u64 B = bc->B;
    u128 xspan = (u128)(B - 1) * (bc->wy + bc->s);
    for (u64 x = 0; x < B; x++) {
        u128 Mx = Mblock + (u128)x * bc->wx;
        if (Mx + xspan < pr->lo || Mx > pr->hi) continue;
        for (u64 y = 0; y < B; y++)
            for (u64 z = 0; z < B; z++) {
                u128 M = Mblock + (u128)x * bc->wx + (u128)y * bc->wy + (u128)z * bc->s;
                if (M < pr->lo || M > pr->hi) continue;
                u64 clo = iroot(M, pr->n - 1) + 1, chi = iroot(M, pr->n - 2);
                for (u64 c = clo; c <= chi; c++) {
                    u64 r = mod128_64(M, c);
                    if (!r) continue;                  /* quick pre-test, valid in both modes: */
                    u128 Pc = pow_sat(c, pr->n - 2);
                    if (M / Pc != r) continue;         /* last digit = leading digit */
                    if (mode == 2 ? !ends_mirror(M, c, pr->n - 1, pr->k) : !is_pal_base(M, c, pr->n - 1)) continue;
                    if (cnt == cap) { cap *= 2; p = realloc(p, cap * sizeof(pair_t)); }
                    p[cnt].M = M;
                    p[cnt].c = c;
                    cnt++;
                }
            }
    }
    *out = p;
    *np = cnt;
}

/* pair_cmp -- qsort() comparator: order pairs by M, then by c. */
static int pair_cmp(const void *a, const void *b)
{
    const pair_t *x = a, *y = b;
    if (x->M != y->M) return x->M < y->M ? -1 : 1;
    if (x->c != y->c) return x->c < y->c ? -1 : 1;
    return 0;
}

/*
 * naive_search -- plain enumeration: every n-digit palindrome in every base
 * B (digits counted up like an odometer), with all lengths n-1..3 checked by
 * find_pal_base().  Used for n <= 6 by "search", by "naive", and by the
 * selftest for a(1..6).
 *
 * In:     n; [lo, hi]; print (1 = print every new smallest solution).
 * Out:    the smallest M in [lo, hi] with palindrome bases for all lengths
 *         2..n, or 0 if there is none.  n <= 2 is answered directly (every
 *         M >= 1 is a 1-digit palindrome, every M >= 3 is "11" in base M-1;
 *         hi is not looked at).
 * Errors: none; the work grows with the number of n-digit palindromes up
 *         to hi (about hi^((D+1)/(n-1))), so it is only usable for small hi
 *         (the selftest uses hi = 100000).
 */
static u128 naive_search(int n, u128 lo, u128 hi, int print)
{
    u128 best = 0;
    char buf[48];
    if (n <= 1) return lo <= 1 ? 1 : lo;
    if (n == 2) return lo <= 3 ? 3 : lo;
    u64 Bmax = iroot(hi, n - 1);
    int D = (n + 1) / 2, h = n / 2;
    for (u64 B = 2; B <= Bmax; B++) {
        u128 w[MAXN];
        for (int i = 0; i < h; i++) w[i] = pow_sat(B, n - 1 - i) + pow_sat(B, i);
        if (n & 1) w[h] = pow_sat(B, h);
        u64 d[MAXN] = {0};
        d[0] = 1;
        for (;;) {
            u128 M = 0;
            for (int i = 0; i < D; i++) M += (u128)d[i] * w[i];
            if (M >= lo && M <= hi && (!best || M < best)) {
                int ok = 1;
                for (int L = n - 1; L >= 3 && ok; L--) ok = find_pal_base(M, L) != 0;
                if (ok) {
                    best = M;
                    if (print) printf("  naive n=%d: %s (B=%llu)\n", n, u128s(M, buf), (unsigned long long)B);
                }
            }
            int i = D - 1;             /* next digit tuple; d_0 runs 1..B-1 */
            while (i >= 0) {
                d[i]++;
                if (d[i] < B) break;
                d[i] = (i == 0) ? 1 : 0;
                i--;
            }
            if (i < 0) break;          /* all palindromes of base B done */
        }
    }
    return best;
}

/* ------------------------------------------------------------------ */
/* threaded search                                                     */
/* ------------------------------------------------------------------ */

/* g_stop -- set by SIGINT/SIGTERM (on_sig); workers stop taking new units
   and cmd_search() saves the state and prints the summary. */
static volatile sig_atomic_t g_stop;
static void on_sig(int s) { (void)s; g_stop = 1; }

/* pend_t -- a finished unit at or above the frontier, with its statistics. */
typedef struct { u64 u; wstat_t st; } pend_t;

/*
 * search_t -- the shared state of one threaded search.
 *
 *   pr         the problem
 *   next       the next unit to hand out (atomic counter)
 *   done       done[u] = 1 once unit u is finished (nunits + 1 bytes)
 *   frontier   the lowest unfinished unit: every unit below it is done
 *   committed  the statistics of exactly the units in [start, frontier) of
 *              this session, so a checkpoint (frontier, committed + resumed
 *              totals) is always exact
 *   pend, npend, pendcap  finished units above the frontier (finished out of
 *              order); their statistics move into committed when the
 *              frontier passes them.  Usually about one entry per thread.
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
 *
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
        S->pend = realloc(S->pend, S->pendcap * sizeof(pend_t));
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
 *
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
 * worker -- thread body: take units from S->next until they run out or a
 * stop is requested; run do_unit() on each with fresh statistics, then hand
 * them to finish_unit().
 *
 * In:     arg, the search_t.
 * Out:    the units' statistics, through finish_unit(); returns NULL.
 * Errors: none of its own (bctx_set may exit).  g_stop is checked only
 *         between units, so every unit taken is finished.
 */
static void *worker(void *arg)
{
    search_t *S = arg;
    bctx_t bc;
    memset(&bc, 0, sizeof bc);
    wstat_t ust;                       /* the current unit's statistics */
    while (!g_stop) {
        u64 u = atomic_fetch_add(&S->next, 1);
        if (u >= S->pr->nunits) break;
        memset(&ust, 0, sizeof ust);
        ust.pr = S->pr;
        ust.verbose = S->verbose;
        do_unit(&ust, &bc, u);
        pthread_mutex_lock(&S->lock);
        finish_unit(S, u, &ust);
        pthread_mutex_unlock(&S->lock);
    }
    free(bc.tab);
    return NULL;
}

/* now_sec -- monotonic time in seconds, for elapsed times, rates and the ETA. */
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/*
 * unit_weight -- estimated tuples of a unit (for the progress percentage
 * and the ETA only).
 *
 * In:     pr; B; the leading digit d0.
 * Out:    (number of bases c for M in [d0 B^(n-1), (d0+1) B^(n-1)]
 *         clipped to [lo, hi]) x B^(D-2) (blocks with d0 fixed, times B^2
 *         steps each) x (the fraction of that M range inside [lo, hi]);
 *         0 if it is outside.  The caller divides by B when a unit also
 *         fixes d_1.
 * Errors: none; double precision, approximate by design.
 */
static double unit_weight(const prob_t *pr, u64 B, u64 d0)
{
    int n = pr->n;
    u128 Mmin = (u128)d0 * pow_sat(B, n - 1), Mmax = (u128)(d0 + 1) * pow_sat(B, n - 1);
    if (Mmax < pr->lo || Mmin > pr->hi) return 0;
    if (Mmin < pr->lo) Mmin = pr->lo;
    if (Mmax > pr->hi) Mmax = pr->hi;
    double cl = pow((double)Mmin, 1.0 / (n - 1)), ch = pow((double)Mmax, 1.0 / (n - 2));
    double nc = ch - cl + 1;
    if (nc < 0) nc = 0;
    double frac = (double)(Mmax - Mmin) / (double)pow_sat(B, n - 1);
    return nc * pow((double)B, pr->D - 2) * frac;   /* blocks (d0 fixed) x B^2 */
}

/*
 * load_state -- read a checkpoint written by save_state().
 *
 * In:     path; pr (the current problem).
 * Out:    0 if the file cannot be opened (normally: it does not exist yet,
 *         a fresh start); otherwise 1, the frontier in *frontier and the
 *         saved totals (tuples, hits, st2, canon, sol, csum, fail[]) in
 *         *acc.  Older files without csum or the failL= fields load with
 *         zeros there.  The format is shared with cuda/a171775_cuda.cu.
 * Errors: exits with status 1 if the file cannot be parsed ("cannot parse
 *         state file"), belongs to another n, LO or HI ("is for a different
 *         search"), or has a frontier beyond the last unit (a damaged file).
 */
static int load_state(const char *path, const prob_t *pr, u64 *frontier, wstat_t *acc)
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
    if (!ok) { fprintf(stderr, "cannot parse state file %s\n", path); exit(1); }
    if (n != pr->n || parse_or_die(lo) != pr->lo || parse_or_die(hi) != pr->hi) {
        fprintf(stderr, "state file %s is for a different search\n", path);
        exit(1);
    }
    if (fr > pr->nunits) {
        fprintf(stderr, "state file %s: frontier %llu is beyond the last unit (%llu units)\n", path, fr,
                (unsigned long long)pr->nunits);
        exit(1);
    }
    *frontier = fr;
    acc->tuples = tu; acc->hits = hi_; acc->st2 = s2; acc->canon = ca; acc->sol = so; acc->csum = cs;
    for (int L = 0; L <= MAXN; L++) acc->fail[L] = fl[L];
    return 1;
}

/*
 * save_state -- write a checkpoint: one line "n=.. lo=.. hi=.. frontier=..
 * tuples=.. hits=.. st2=.. canon=.. sol=.. csum=.. failL=.." (decimal
 * numbers, failL= for L = n-2..3).
 *
 * In:     path; pr; the frontier; t (the statistics of exactly the units
 *         below the frontier, including a resumed state's).
 * Out:    writes path.tmp and renames it over path, so an interruption
 *         during the write leaves the previous checkpoint intact.
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
    fprintf(f, "n=%d lo=%s hi=%s frontier=%llu tuples=%llu hits=%llu st2=%llu canon=%llu sol=%llu csum=%llu",
            pr->n, u128s(pr->lo, a), u128s(pr->hi, b), (unsigned long long)frontier, (unsigned long long)t->tuples,
            (unsigned long long)t->hits, (unsigned long long)t->st2, (unsigned long long)t->canon,
            (unsigned long long)t->sol, (unsigned long long)t->csum);
    for (int L = pr->n - 2; L >= 3; L--) fprintf(f, " fail%d=%llu", L, (unsigned long long)t->fail[L]);
    fprintf(f, "\n");
    int bad = ferror(f);
    if (fclose(f) != 0) bad = 1;
    if (bad) {                         /* keep the old checkpoint rather than a truncated one */
        fprintf(stderr, "cannot write %s; checkpoint not updated\n", tmp);
        remove(tmp);
        return;
    }
    rename(tmp, path);                 /* atomic replacement */
}

/*
 * print_summary -- the two final lines of a search: "# n=.. [LO, HI]
 * complete|INTERRUPTED: time, tuples, hits, (n-1)-palindromes (st2),
 * canonical, checksum, solutions" and "# failed at length: L=..:..".
 * In: pr, t (totals including a resumed state's), complete, el (seconds).
 * Out: on stdout.  No errors.
 */
static void print_summary(const prob_t *pr, const wstat_t *t, int complete, double el)
{
    char a[48], b[48];
    printf("# n=%d [%s, %s] %s: %.1fs, tuples %llu, hits %llu, (n-1)-palindromes %llu, canonical %llu, "
           "checksum %016llx, solutions %llu\n",
           pr->n, u128s(pr->lo, a), u128s(pr->hi, b), complete ? "complete" : "INTERRUPTED", el,
           (unsigned long long)t->tuples, (unsigned long long)t->hits, (unsigned long long)t->st2,
           (unsigned long long)t->canon, (unsigned long long)t->csum, (unsigned long long)t->sol);
    printf("# failed at length:");
    for (int L = pr->n - 2; L >= 3; L--) printf(" L=%d:%llu", L, (unsigned long long)t->fail[L]);
    printf("\n");
}

/*
 * cmd_search -- the "search" command: find every solution for n in [lo, hi].
 *
 * In:     n; lo, hi; nthreads; state (checkpoint file, or NULL); interval
 *         (seconds between progress lines); verbose (-v).
 * Out:    SOLUTION (and DEEP) lines and two summary lines on stdout,
 *         progress on stderr, and the checkpoint file if requested.  For
 *         n <= 6 runs naive_search() instead and prints the smallest
 *         solution.  A range without n-digit numbers gets a note on stderr
 *         and an empty "complete" summary.  Returns 0, also for an
 *         interrupted search (its summary line says INTERRUPTED).
 * How:    prob_init; resume from the state file if it exists; progress
 *         weights cw[u] = estimated work of the units before u; start the
 *         threads; then poll every 0.2 s: a progress line every `interval`
 *         seconds and a checkpoint every 60 s, both taken from
 *         search_totals() (the checkpoint holds exactly the units below the
 *         frontier), until all units are done or SIGINT/SIGTERM set g_stop.
 *         Then join the threads, save the final checkpoint and print the
 *         summary; all counts include the resumed totals.
 * Errors: exits through prob_init/bctx_set/load_state on fatal conditions.
 *         Allocation failures are not checked; done[] takes nunits bytes and
 *         cw[] 8 nunits bytes (about 0.4 GB for n = 11 over [1, 2^90]).
 */
static int cmd_search(int n, u128 lo, u128 hi, int nthreads, const char *state, double interval, int verbose)
{
    char a[48], b[48];
    if (n <= 6) {
        u128 m = naive_search(n, lo, hi, 1);
        printf("naive: smallest solution %s\n", m ? u128s(m, a) : "none");
        return 0;
    }
    prob_t pr;
    prob_init(&pr, n, lo, hi);
    if (pr.nunits == 0) {              /* e.g. HI < 2^(n-1), or LO > HI */
        wstat_t none;
        memset(&none, 0, sizeof none);
        fprintf(stderr, "search n=%d [%s, %s]: no base has %d-digit numbers in this range, nothing to search\n", n,
                u128s(lo, a), u128s(hi, b), n);
        print_summary(&pr, &none, 1, 0.0);
        free(pr.uoff);
        return 0;
    }
    search_t S;
    memset(&S, 0, sizeof S);
    S.pr = &pr;
    S.verbose = verbose;
    S.done = calloc(pr.nunits + 1, 1);
    pthread_mutex_init(&S.lock, NULL);
    wstat_t base;                      /* the totals of a resumed state */
    memset(&base, 0, sizeof base);
    u64 start = 0;
    if (state && load_state(state, &pr, &start, &base))
        fprintf(stderr, "resuming at unit %llu of %llu\n", (unsigned long long)start, (unsigned long long)pr.nunits);
    for (u64 u = 0; u < start; u++) S.done[u] = 1;   /* units below the saved frontier are done */
    S.frontier = start;
    atomic_store(&S.next, start);

    /* progress weights: cw[u] = estimated tuples of units 0..u-1 */
    double *cw = malloc((pr.nunits + 1) * sizeof(double));
    cw[0] = 0;
    for (u64 u = 0; u < pr.nunits; u++) {
        u64 pref[MAXN];
        u64 B = unit_decode(&pr, u, pref);
        double wgt = unit_weight(&pr, B, pref[0]);
        if (pr.ud > 1) wgt /= (double)B;
        cw[u + 1] = cw[u] + wgt;
    }
    double wtot = cw[pr.nunits] > 0 ? cw[pr.nunits] : 1;

    fprintf(stderr, "search n=%d [%s, %s]: B in [%llu, %llu], %llu units, k=%d, %d threads\n", n, u128s(lo, a),
            u128s(hi, b), (unsigned long long)pr.Bmin, (unsigned long long)pr.Bmax,
            (unsigned long long)pr.nunits, pr.k, nthreads);
    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);
    pthread_t *th = malloc(nthreads * sizeof(pthread_t));
    double t0 = now_sec(), tlast = t0, tsave = t0;
    for (int i = 0; i < nthreads; i++) pthread_create(&th[i], NULL, worker, &S);
    double w0 = cw[start];
    for (;;) {                         /* the main thread only reports and checkpoints */
        usleep(200000);
        wstat_t below, fin;            /* units below the frontier; all finished units */
        u64 fr = search_totals(&S, &below, &fin);
        u64 nx = atomic_load(&S.next);
        int finished = fr >= pr.nunits;
        double t = now_sec();
        if (t - tlast >= interval || finished || g_stop) {
            double el = t - t0;
            double wd = cw[fr < pr.nunits ? fr : pr.nunits];   /* work below the frontier */
            double prog = wd / wtot;
            double rate = fin.tuples / (el > 0 ? el : 1);
            double eta = (wd - w0) > 0 ? (wtot - wd) * el / (wd - w0) : 0;
            u64 pref[MAXN];
            u64 Bc = unit_decode(&pr, fr < pr.nunits ? fr : pr.nunits - 1, pref);
            fprintf(stderr,
                    "[%7.0fs] frontier %llu/%llu (B=%llu) %.3f%%  %.3e tuples/s  st2 %llu canon %llu sol %llu  ETA %.0fs\n",
                    el, (unsigned long long)fr, (unsigned long long)pr.nunits, (unsigned long long)Bc, 100 * prog,
                    rate, (unsigned long long)(base.st2 + fin.st2), (unsigned long long)(base.canon + fin.canon),
                    (unsigned long long)(base.sol + fin.sol), eta);
            tlast = t;
        }
        if (state && (t - tsave >= 60 || finished || g_stop)) {
            add_stats(&below, &base);  /* exactly the units below fr */
            save_state(state, &pr, fr, &below);
            tsave = t;
        }
        if (finished || (g_stop && nx >= pr.nunits)) break;
        if (g_stop) break;
    }
    for (int i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
    wstat_t all;                       /* every unit taken has finished, so none is pending */
    u64 fr = search_totals(&S, &all, NULL);
    add_stats(&all, &base);
    if (state) save_state(state, &pr, fr, &all);
    print_summary(&pr, &all, fr >= pr.nunits, now_sec() - t0);
    free(th); free(cw); free(S.done); free(S.pend); free(pr.uoff);
    return 0;
}

/* ------------------------------------------------------------------ */
/* selftest                                                            */
/* ------------------------------------------------------------------ */

/* rnd -- xorshift64 pseudo-random numbers; the fixed seed makes every
   selftest run check the same blocks. */
static u64 rng_state = 88172645463325252ULL;
static u64 rnd(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

/*
 * compare_block -- selftest: run do_block() on one block in a record mode
 * and compare its (M, c) pairs with naive_block_pairs().
 *
 * In:     n; B; blockidx (which block of base B: d_0 = 1 + blockidx /
 *         B^(bd-1), the other block digits from the remainder); [lo, hi];
 *         quiet (print only mismatches); win (if > 0, restrict M to
 *         [Mblock + x0 w_x, Mblock + (x0 + win) w_x] for a random x0, so
 *         that large B stay cheap for the naive side); mode (1 = (n-1)-digit
 *         palindromes, 2 = all k-digit matches, see wstat_t.record); npairs
 *         (if not NULL, the number of naive pairs is added to it).
 * Out:    1 if both sides give the same pairs (also when blockidx is past
 *         the last block), 0 otherwise; a line per block unless quiet.
 * Errors: none; allocation failures are not checked.
 */
static int compare_block(int n, u64 B, u64 blockidx, u128 lo, u128 hi, int quiet, int win, int mode, size_t *npairs)
{
    prob_t pr;
    prob_init(&pr, n, lo, hi);
    bctx_t bc;
    memset(&bc, 0, sizeof bc);
    bctx_set(&bc, &pr, B);
    /* block digits d_0..d_{bd-1} from blockidx: d_0 = 1 + idx / B^(bd-1) */
    u64 d[MAXN];
    u64 t = blockidx;
    for (int i = pr.bd - 1; i >= 1; i--) { d[i] = t % B; t /= B; }
    d[0] = 1 + t;
    if (d[0] >= B) { free(bc.tab); free(pr.uoff); return 1; }
    u128 Mb = 0;
    for (int i = 0; i < pr.bd; i++) Mb += (u128)d[i] * bc.w[i];
    if (win) {                         /* narrow [lo, hi] to the window; both sides see it */
        u64 x0 = rnd() % B;
        u128 wlo = Mb + (u128)x0 * bc.wx, whi = wlo + (u128)win * bc.wx;
        if (wlo < lo) wlo = lo;
        if (whi > hi) whi = hi;
        if (wlo > whi) { wlo = Mb; whi = Mb + (u128)win * bc.wx; }
        free(pr.uoff);
        prob_init(&pr, n, wlo, whi);
        bctx_set(&bc, &pr, B);
    }
    wstat_t st;
    memset(&st, 0, sizeof st);
    st.pr = &pr;
    st.record = mode;
    do_block(&st, &bc, Mb);
    pair_t *np;
    size_t nn;
    naive_block_pairs(&pr, &bc, Mb, &np, &nn, mode);
    if (npairs) *npairs += nn;
    qsort(st.pairs, st.np, sizeof(pair_t), pair_cmp);
    qsort(np, nn, sizeof(pair_t), pair_cmp);
    int ok = st.np == nn;
    for (size_t i = 0; ok && i < nn; i++) ok = st.pairs[i].M == np[i].M && st.pairs[i].c == np[i].c;
    char buf[48];
    if (!ok || !quiet)
        printf("  n=%d B=%llu block %llu (Mblock=%s) mode %d: fast %zu pairs, naive %zu pairs, hits %llu  %s\n", n,
               (unsigned long long)B, (unsigned long long)blockidx, u128s(Mb, buf), mode, st.np, nn,
               (unsigned long long)st.hits, ok ? "ok" : "MISMATCH");
    free(st.pairs);
    free(np);
    free(bc.tab);
    free(pr.uoff);
    return ok;
}

/*
 * check_value -- the "check" command (and selftest part 2): does M have a
 * palindrome base for every length 2..n?
 *
 * In:     M; n; print (1 = print the smallest base of each length and, for
 *         L <= 12, its digits; L = 2 shows base M-1).
 * Out:    1 if every length 2..n has a base, else 0.
 * Errors: none; can take seconds for large M (find_pal_base at L = 3).
 */
static int check_value(u128 M, int n, int print)
{
    char buf[48];
    int ok = 1;
    if (print) printf("M = %s\n", u128s(M, buf));
    for (int L = 2; L <= n; L++) {
        u64 b = find_pal_base(M, L);
        if (print) {
            if (b) {
                printf("  L=%2d base %llu digits:", L, (unsigned long long)b);
                u128 x = M;
                u64 dg[130];
                int len = 0;
                while (x) { dg[len++] = (u64)(x % b); x /= b; }
                if (L <= 12)
                    for (int i = len - 1; i >= 0; i--) printf(" %llu", (unsigned long long)dg[i]);
                printf("\n");
            } else printf("  L=%2d none\n", L);
        }
        if (!b) ok = 0;
    }
    return ok;
}

/*
 * cmd_selftest -- the "selftest" command.
 *
 * In:     full (also run the complete a(9) search, tens of seconds);
 *         nthreads (for the complete searches).
 * Out:    prints four parts and returns 0 if all passed, 1 otherwise:
 *         1. naive_search() gives a(1..6);
 *         2. 2^((n-1)(n-2)) has palindrome bases for all lengths 2..n, and
 *            none for length n+1, for n = 7..12;
 *         3. the fast filter agrees with naive_block_pairs() on random blocks
 *            for n = 7..12 in both record modes, including the first and
 *            last block of a base and windows at large B (up to 511);
 *         4. complete searches over [1, a(n)] find exactly one solution,
 *            a(n), for n = 7, 8 (and 9 with full).
 *         A PARANOID build runs the same checks with every step verified
 *         (much slower).
 * Errors: none beyond a failed check (returned as 1).  tmpfile() failure is
 *         not checked.
 */
static int cmd_selftest(int full, int nthreads)
{
    int bad = 0;
    char buf[48];
    const char *known[] = {"1", "3", "5", "52", "130", "1885", "2^30", "2^42", "2^56"};
    printf("1. naive a(1..6)\n");
    for (int n = 1; n <= 6; n++) {
        u128 m = naive_search(n, 1, 100000, 0);
        u128 want = parse_or_die(known[n - 1]);
        printf("  a(%d) = %s %s\n", n, u128s(m, buf), m == want ? "ok" : "WRONG");
        bad += m != want;
    }
    printf("2. 2^((n-1)(n-2)) satisfies lengths 2..n (n = 7..12), and fails n+1\n");
    for (int n = 7; n <= 12; n++) {
        u128 M = (u128)1 << ((n - 1) * (n - 2));
        int ok = check_value(M, n, 0) && !find_pal_base(M, n + 1);
        printf("  n=%d 2^%d: %s\n", n, (n - 1) * (n - 2), ok ? "ok" : "FAIL");
        bad += !ok;
    }
    printf("3. fast filter vs naive on random blocks\n");
    struct { int n; u64 Bmin, Bmax; int reps, win; } cfg[] = {
        {7, 3, 40, 60, 0}, {8, 3, 30, 60, 0}, {9, 3, 24, 60, 0}, {10, 3, 20, 60, 0}, {11, 3, 14, 40, 0},
        {12, 3, 10, 20, 0}, {9, 60, 90, 4, 0}, {10, 40, 60, 3, 0}, {11, 30, 40, 2, 0},
        {10, 100, 255, 8, 2}, {11, 200, 511, 4, 1}, {9, 100, 127, 8, 2}, {8, 50, 63, 8, 0},
    };
    for (size_t ci = 0; ci < sizeof cfg / sizeof cfg[0]; ci++) {
        int n = cfg[ci].n, okc = 0, tot = 0;
        size_t pairs1 = 0, pairs2 = 0;
        for (int r = 0; r < cfg[ci].reps; r++) {
            u64 B = cfg[ci].Bmin + rnd() % (cfg[ci].Bmax - cfg[ci].Bmin + 1);
            u128 hi = U128_MAX >> 1;
            if (n == 10) hi = (u128)1 << 72;     /* production limits: c^k and c^(h+1) fit */
            if (n == 11) hi = (u128)1 << 90;
            if (n == 12) hi = (u128)1 << 80;
            u128 dm = hi / pow_sat(B, n - 1);    /* leading digits with Mblock <= hi */
            u64 dmax = dm < B - 1 ? (u64)dm : B - 1;
            if (dmax < 1) { r--; continue; }     /* B too large for hi: draw another */
            int bdg = (n + 1) / 2 - 3;           /* block digits (prob_t.bd) */
            u64 nblocks = dmax * (u64)pow_sat(B, bdg - 1);
            u64 bi = rnd() % nblocks;
            if (r == 0) bi = 0;                  /* always include the first ... */
            if (r == 1) bi = nblocks - 1;        /* ... and the last block */
            u64 save = rng_state;
            int ok = compare_block(n, B, bi, 1, hi, 1, cfg[ci].win, 1, &pairs1);
            rng_state = save;                    /* same window for mode 2 */
            ok &= compare_block(n, B, bi, 1, hi, 1, cfg[ci].win, 2, &pairs2);
            okc += ok;
            tot++;
        }
        printf("  n=%d B in [%llu,%llu]%s: %d/%d blocks agree (%zu palindrome pairs, %zu k-digit matches)\n", n,
               (unsigned long long)cfg[ci].Bmin, (unsigned long long)cfg[ci].Bmax, cfg[ci].win ? " windowed" : "",
               okc, tot, pairs1, pairs2);
        bad += okc != tot;
    }
    printf("4. full searches reproduce a(7), a(8)%s\n", full ? ", a(9)" : "");
    int nmax = full ? 9 : 8;
    for (int n = 7; n <= nmax; n++) {
        /* search [1, a(n)] with the workers directly (no checkpoint, no
           progress lines); any smaller solution would show up as a second one */
        u128 want = parse_or_die(known[n - 1]);
        prob_t pr;
        prob_init(&pr, n, 1, want);
        search_t S;
        memset(&S, 0, sizeof S);
        S.pr = &pr;
        S.done = calloc(pr.nunits + 1, 1);
        pthread_mutex_init(&S.lock, NULL);
        /* capture solutions */
        FILE *mem = tmpfile();
        FILE *save = sol_out;
        sol_out = mem;
        double t0 = now_sec();
        pthread_t *th = malloc(nthreads * sizeof(pthread_t));
        for (int i = 0; i < nthreads; i++) pthread_create(&th[i], NULL, worker, &S);
        for (int i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
        sol_out = save;
        wstat_t tot;
        search_totals(&S, &tot, NULL);
        rewind(mem);                   /* read back the SOLUTION lines */
        char line[1024];
        u128 best = 0;
        int nsol = 0;
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
        int ok = best == want && nsol == 1;
        printf("  n=%d [1, %s]: %.2fs, %.3e tuples, %llu (n-1)-pals, %llu canonical, %d solution(s), smallest %s  %s\n", n,
               known[n - 1], now_sec() - t0, (double)tot.tuples, (unsigned long long)tot.st2,
               (unsigned long long)tot.canon, nsol, best ? u128s(best, buf) : "none", ok ? "ok" : "WRONG");
        bad += !ok;
        free(th); free(S.done); free(S.pend); free(pr.uoff);
    }
    printf(bad ? "SELFTEST FAILED (%d)\n" : "selftest passed\n", bad);
    return bad ? 1 : 0;
}

/* ------------------------------------------------------------------ */

#ifdef PARANOID
/*
 * cmd_paranoid -- (PARANOID builds) the "paranoid" command: do_block() on
 * random blocks of several (n, B) ranges, with the incremental state
 * recomputed from scratch at every step (paranoid_state) and z brute-forced
 * on part of the steps (see block_c).
 *
 * In:     reps, the number of blocks per configuration (each configuration
 *         also stops after 60 s).
 * Out:    one summary line per configuration; returns 0.
 * Errors: abort() with a diagnostic on any mismatch.  Solutions go to
 *         /dev/null.
 */
static int cmd_paranoid(int reps)
{
    struct { int n; u64 Bmin, Bmax; u128 hi; } cfg[] = {
        {7, 3, 60, (u128)1 << 62}, {8, 3, 60, (u128)1 << 62}, {9, 3, 60, (u128)1 << 62},
        {10, 3, 60, (u128)1 << 72}, {11, 3, 60, (u128)1 << 90}, {12, 3, 40, (u128)1 << 80},
        {10, 100, 255, (u128)1 << 72}, {11, 200, 511, (u128)1 << 90}, {9, 64, 127, (u128)1 << 62},
    };
    FILE *devnull = fopen("/dev/null", "w");
    FILE *save = sol_out;
    sol_out = devnull;
    for (size_t ci = 0; ci < sizeof cfg / sizeof cfg[0]; ci++) {
        int n = cfg[ci].n;
        u64 steps = 0, zh = 0, blocks = 0;
        double t0 = now_sec();
        int r;
        for (r = 0; r < reps && now_sec() - t0 < 60; r++) {
            u64 B = cfg[ci].Bmin + rnd() % (cfg[ci].Bmax - cfg[ci].Bmin + 1);
            u128 hi = cfg[ci].hi;
            u128 dm = hi / pow_sat(B, n - 1);
            u64 dmax = dm < B - 1 ? (u64)dm : B - 1;
            if (dmax < 1) continue;
            prob_t pr;
            prob_init(&pr, n, 1, hi);
            bctx_t bc;
            memset(&bc, 0, sizeof bc);
            bctx_set(&bc, &pr, B);
            u64 d[MAXN];
            d[0] = 1 + rnd() % dmax;
            for (int i = 1; i < pr.bd; i++) d[i] = rnd() % B;
            u128 Mb = 0;
            for (int i = 0; i < pr.bd; i++) Mb += (u128)d[i] * bc.w[i];
            wstat_t st;
            memset(&st, 0, sizeof st);
            st.pr = &pr;
            do_block(&st, &bc, Mb);
            steps += st.pchk;                  /* brute-force checked steps */
            zh += st.pz;                       /* z hits among them */
            blocks++;
            free(bc.tab);
            free(pr.uoff);
        }
        printf("  paranoid n=%d B in [%llu,%llu]: %llu blocks, every step state-checked, %llu brute-force steps, %llu z hits: ok\n",
               n, (unsigned long long)cfg[ci].Bmin, (unsigned long long)cfg[ci].Bmax, (unsigned long long)blocks,
               (unsigned long long)steps, (unsigned long long)zh);
        fflush(stdout);
    }
    sol_out = save;
    return 0;
}
#endif

/* usage -- print the command summary to stderr and exit with status 1. */
static void usage(void)
{
    fprintf(stderr,
            "usage: a171775 selftest [full] [-t T]\n"
            "       a171775 search n LO HI [-t T] [-S state] [-i sec] [-v]\n"
            "       a171775 check M [n]\n"
            "       a171775 naive n LO HI\n");
    exit(1);
}

/*
 * main -- parse the options (-t, -S, -i, -v, anywhere on the line) and up to
 * eight positional arguments, and run the command (see the header).
 *
 * Errors: an unknown command or a wrong number of arguments prints the
 *         usage and exits with status 1; search needs 3 <= n <= 20; bad
 *         numbers exit with status 1 (parse_or_die).  A thread count below 1
 *         (including a non-numeric -t) becomes 1.
 */
int main(int argc, char **argv)
{
    sol_out = stdout;
    int nthreads = (int)sysconf(_SC_NPROCESSORS_ONLN);   /* default: one thread per online core */
    const char *state = NULL;
    double interval = 10;
    int verbose = 0;
    char *pos[8];                      /* positional arguments: the command and its operands */
    int npos = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-t") && i + 1 < argc) nthreads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-S") && i + 1 < argc) state = argv[++i];
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) interval = atof(argv[++i]);
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else if (npos < 8) pos[npos++] = argv[i];
        else usage();
    }
    if (nthreads < 1) nthreads = 1;
    if (npos < 1) usage();
    if (!strcmp(pos[0], "selftest")) return cmd_selftest(npos > 1 && !strcmp(pos[1], "full"), nthreads);
#ifdef PARANOID
    if (!strcmp(pos[0], "paranoid")) return cmd_paranoid(npos > 1 ? atoi(pos[1]) : 200);
#endif
    if (!strcmp(pos[0], "search") && npos == 4) {
        int n = atoi(pos[1]);
        if (n < 3 || n > 20) usage();  /* MAXN/MAXK leave room up to 20 */
        return cmd_search(n, parse_or_die(pos[2]), parse_or_die(pos[3]), nthreads, state, interval, verbose);
    }
    if (!strcmp(pos[0], "check") && npos >= 2) {   /* exit status 2 if some length fails */
        u128 M = parse_or_die(pos[1]);
        int n = npos > 2 ? atoi(pos[2]) : 12;
        return check_value(M, n, 1) ? 0 : 2;
    }
    if (!strcmp(pos[0], "naive") && npos == 4) {
        char buf[48];
        u128 m = naive_search(atoi(pos[1]), parse_or_die(pos[2]), parse_or_die(pos[3]), 1);
        printf("smallest: %s\n", m ? u128s(m, buf) : "none");
        return 0;
    }
    usage();
    return 1;
}
