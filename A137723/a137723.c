/*
 * a137723.c -- compute and extend OEIS A137723
 * =============================================
 *
 *   A137723: "First occurrence of a set of n consecutive numbers having at
 *   least one prime gap in their factorization: a(n) = smallest number of
 *   this set."
 *
 * The original entry listed a(1..31) (built in below as KNOWN[]) and noted
 * a(32) > 10^11.  This program found a(32) = 2^54 - 32 and every term up to
 * a(193); they were added to the OEIS in Sep 2026 (b-file n = 1..193).  The
 * first open term is a(194) > 2^64 - 194.  README.md describes the results
 * and the method in prose; submission.md tracks the OEIS status.
 *
 *
 * THEORY OF OPERATION
 * ===================
 *
 * 1. Definitions
 * --------------
 * p_1 = 2, p_2 = 3, p_3 = 5, ... are the primes in increasing order.  A number
 * m >= 1 is GAPLESS (OEIS A073491; "gap-free" in the README) when its distinct
 * prime factors are consecutive primes p_i, p_(i+1), ..., p_j, with arbitrary
 * exponents >= 1.  The gapless numbers are 1, the primes, the prime powers and
 * products such as 12 = 2^2*3, 35 = 5*7 and 2940 = 2^2*3*5*7^2.  Every other
 * number is GAPFUL (A073492): its factorization skips a prime, e.g. 10 = 2*5
 * skips 3 and 84 = 2^2*3*7 skips 5.  A073490(m) counts the skipped places;
 * A073490(110) = 2 because 110 = 2*5*11 skips 3 and 7.
 *
 * The problem is really about the gaps between consecutive gapless numbers:
 *
 *     If L < R are CONSECUTIVE gapless numbers, then L+1, ..., R-1 is a
 *     maximal run of n = R-L-1 gapful numbers, and a(n) is L+1 for the
 *     smallest such L with R - L = n + 1.
 *
 * Example: 83 and 89 are consecutive gapless numbers and 84..88 are gapful,
 * so a(5) = 84.  Notation used throughout the code:
 *
 *     d = n + 1     the distance R - L between the two bounding numbers
 *     pair (L, R)   two consecutive gapless numbers, i.e. one maximal run
 *     E, O          for even n: the even one and the odd one of L and R
 *     k             the number of distinct primes of E
 *
 * Two facts carry the correctness arguments below: every number strictly
 * inside a run is gapful, and therefore two different runs never overlap.
 *
 *
 * 2. Method 1, `scan N`: exhaustive enumeration (every n)
 * -------------------------------------------------------
 * Instead of factoring every integer up to N, the scan generates the gapless
 * numbers in increasing order from two sorted streams:
 *
 *   - the primes, from a primesieve iterator (one per thread);
 *   - the composite gapless numbers (p^e with e >= 2, and products of two or
 *     more consecutive primes).  Each has a prime factor <= sqrt(N), so there
 *     are only O(sqrt N) of them, about 9*10^4 below 10^11.
 *     gen_composite_gapless() enumerates them recursively and sorts them once.
 *
 * [1, N] is cut into chunks that the worker threads claim through an atomic
 * counter.  A worker merges the two streams from the start of its chunk and,
 * for every pair of consecutive gapless numbers (g, g2) with g inside the
 * chunk, records g+1 as a start for the length n = g2-g-1 if it has none
 * yet.  The pair that crosses the end of a chunk belongs to the chunk that
 * contains g (the worker reads past the end until it has seen g2), so no pair
 * is lost or counted twice.  The per-thread minima are merged at the end.
 *
 * Guarantee: every pair with g <= N is examined, so each reported a(n) is
 * exact, and an n that is not reported has a(n) > N + 1.  The cost is
 * essentially one prime sieve of [1, N]: 10^11 takes about 3 s on 10 cores.
 * Odd n can only be found this way (both bounding numbers are odd; see 3).
 *
 *
 * 3. Method 2, `search n`: structural search for even n (128-bit)
 * ---------------------------------------------------------------
 * PARITY LEMMA.  For even n the distance d = n+1 is odd, so exactly one of L
 * and R is even.  An even gapless number contains the prime 2 and its primes
 * are consecutive from 2 on, so it has the form
 *
 *         E = 2^e1 * 3^e2 * 5^e3 * ... * p_k^ek        (every e_i >= 1).
 *
 * These are very sparse: 2,537,666 of them lie below 2^64 and about
 * 1.74*10^10 below 2^127.  run_search() enumerates every such E up to the
 * limit (depth first over the exponents) and tests both possible partners
 * O = E + d and O = E - d: O must be gapless and the d-1 numbers between E
 * and O must all be gapful (interior_gapful()).  Each hit is a genuine pair.
 *
 * COVERAGE LEMMA.  (a) A pair with L <= limit - d has its even endpoint at or
 * below the limit, so it is found.  (b) If any pair is found, the smallest one
 * found, (Lf, Rf), gives a(n) exactly.  Lf is at most the limit, so an unfound
 * pair (L', R') with L' < Lf must have its even endpoint above the limit: L'
 * is odd and R' > limit >= Lf.  The gapless Lf would then lie strictly inside
 * the run (L', R'), which is impossible.  (c) If no pair is found, then
 * a(n) > limit - n.
 *
 * CONSTRAINT PROPAGATION.  Let p be an odd prime with p <= p_k, so that p | E.
 * Then p | d implies p | O, and p not dividing d implies p does not divide O.
 * The primes of O must be consecutive, so the odd primes <= p_k that divide d
 * must form one block of consecutive primes.  Once E would contain a second,
 * separate prime factor of d, no partner can be gapless; that caps the number
 * of primes of E at kmax (compute_kmax()).  In the same way O can only be a
 * large prime when no odd prime <= p_k divides d (compute_kprime(), reported
 * for information).  Example: d = 33 = 3*11 gives kmax = 4, and a prime
 * partner forces k = 1, i.e. E = 2^a.  This is why a(32) = 2^54 - 32 is so
 * large, and why every a(n) with n == 2 (mod 6) found so far is next to a
 * power of 2.
 *
 * The top three levels of the enumeration (2^a, 2^a*3^b, 2^a*3^b*5^c) form a
 * task list consumed through an atomic counter; the workers recurse into
 * 7, 11, ... themselves.
 *
 *
 * 4. Method 3, `search n -l LIMIT` with LIMIT >= 2^127: GMP mode
 * --------------------------------------------------------------
 * Above 2^127 there are far too many even gapless numbers to visit (well over
 * 10^40 below 2^1000), so the same theorem is applied more selectively.  Let
 * p_j be the smallest odd prime dividing d, and {p_j, ..., p_m} the maximal
 * block of consecutive primes from p_j on that all divide d (z_analyze()).
 * For an even endpoint with k primes, the odd partner O = E +- d is:
 *
 *     k <  j        coprime to p_1..p_k, may be prime      -> enumerate E
 *     j <= k <= m   divisible by p_j..p_k, may continue    -> enumerate E
 *     k >  m        exactly a product over {p_j..p_m}      -> enumerate O
 *
 * In the last case p_j..p_m divide O because they divide both E and d, while
 * p_(m+1) divides E but not d, so it cannot divide O, and the primes of O are
 * consecutive.  Hence E is enumerated only for k <= m (the "E side": 2^a,
 * 2^a*3^b, ...), and for k > m the program enumerates the products O over the
 * block instead (the "O side") and tests E = O +- d.  The two sides are
 * disjoint and together cover every pair whose even endpoint is at most the
 * limit, so the coverage lemma of section 3 still applies.
 *
 * For d = 63 = 3^2*7 the whole search is 2^a +- 63, 2^a*3^b +- 63 and the
 * powers of 3: about 3*10^5 candidates below 2^1000, under a second.  Before
 * starting, the program prints an estimate of the work (simplex volumes,
 * z_est_E() and z_est_S()) and refuses to run if it exceeds --max-nodes.
 *
 *
 * 5. The gapless test: is_gapless() and z_is_gapfree()
 * ----------------------------------------------------
 *   (i)   Trial division by the 6542 primes below SMALL_LIMIT = 2^16.  Once the
 *         smallest prime factor p is known, m is gapless iff dividing out p,
 *         nextprime(p), nextprime(nextprime(p)), ... reaches 1 (chain_from()).
 *         Nearly every candidate is decided here, most after a few divisions.
 *   (ii)  No factor below 2^16: m is gapless if it is prime.
 *   (iii) Composite with no factor below 2^16: suppose m is gapless with t
 *         prime factors counted with multiplicity, p_1 <= ... <= p_s.  Then
 *         p_1^t <= m <= p_s^t.  With c = floor(m^(1/t)), either m = c^t with c
 *         prime, or p_1 <= c < p_s; then the primes in [p_1, c] are among
 *         p_1..p_(s-1), so p_1 is one of the t-1 largest primes <= c.  Every
 *         factor exceeds 2^16, so t < log2(m)/16 (at most 7 below 2^128).
 *         root_method() tries those few candidates with chain_from(); no
 *         general factoring is needed.
 *
 *
 * 6. Primality and rigor
 * ----------------------
 * is_prime64() is deterministic: Miller-Rabin with bases 2, 7, 61 below
 * 4759123141 and Sinclair's seven bases below 2^64.  Above 2^64,
 * is_prime128() is a strong probable-prime test to the first 24 prime bases,
 * and z_is_prime() is GMP's mpz_probab_prime_p() with 30 rounds (Baillie-PSW
 * plus Miller-Rabin).  A "composite" verdict is always correct; a "prime"
 * verdict above 2^64 is probabilistic.  The only step of a reported term above
 * 2^64 that is not a proof is therefore the primality of its prime endpoint;
 * certify those with a proving algorithm such as PARI/GP isprime() (see
 * primes_to_certify.txt).  Everything else -- scan results, the completeness
 * of the enumerations, the gaplessness of E, the gapfulness of the interior --
 * is exact.
 *
 *
 * 7. Factorization (for display and independent checks only)
 * ----------------------------------------------------------
 * `check`, `run` and verify_pair() use a separate full factorizer: trial
 * division, perfect-power detection and Pollard-Brent rho.  rho64() always
 * succeeds; rho128() has an iteration budget and may leave a cofactor
 * unfactored (shown as [C...]).  verify_pair() re-checks the smallest pair of
 * a 128-bit search without calling is_gapless() and reports OK, FAILED or
 * INCOMPLETE.  GMP mode prints a verify_run.py command for that purpose.
 *
 *
 * 8. Commands (see main())
 * ------------------------
 *   scan N [-t T] [-m MAXN] [-o FILE]   exhaustive scan of 1..N
 *                                       (100 <= N <= 2^63)
 *   search n.. [-l LIMIT] [-t T]        structural search, even n only.
 *                                       LIMIT below 2^127: 128-bit mode
 *                                       (default 2^64); LIMIT >= 2^127: GMP
 *                                       mode.  --gmp forces GMP mode and
 *                                       --max-nodes X caps its work.
 *                                       Without GMP the 128-bit mode accepts
 *                                       LIMIT up to 2^127.
 *   next [-N SCAN] [-l LIMIT] [-t T] [-m MAXN] [-o FILE]
 *                                       scan to SCAN (default 10^10), then a
 *                                       128-bit search for every even n the
 *                                       scan left unknown
 *   check m                             factor m, count its prime gaps
 *   run m                               print the maximal run of gapful
 *                                       numbers around m
 *   selftest [-t T]                     consistency tests, about half a second
 *
 * Numbers may be written in decimal, as B^E or MeE (1e15), and as sums and
 * differences such as 2^54-32.  Results go to stdout and progress to stderr
 * (a live \r line on a terminal, one line every 15 s otherwise).  Exit status:
 * 0 on success, 1 on errors (die()), 2 on usage errors; selftest exits with 0
 * or 1.
 *
 *
 * 9. Threads, memory and errors
 * -----------------------------
 * Each search keeps its parameters in one global descriptor (SC, SQ or ZQ)
 * that is written before the workers start and only read while they run.
 * Work is handed out through an atomic counter, and each worker owns its
 * result buffer (and, in GMP mode, its own mpz scratch space zws_t), so no
 * locks are needed.  The small-prime tables are built once in main() and are
 * read-only afterwards.  The search functions are therefore not reentrant:
 * one search at a time.
 *
 * Fatal conditions (bad arguments, out of memory, thread creation or
 * primesieve failure) call die(), which prints "a137723: <message>" and exits
 * with status 1.  Limits: scan N <= 2^63 (the prime table up to 2*sqrt(N)
 * takes 8 bytes per prime, about 2.3 GB at N = 2^63); n <= 100000; the 128-bit
 * search limit is at most 2^127; run lengths below MAXRUN = 65536 (prime gaps
 * below 2^64 are shorter than 1600).
 *
 *
 * 10. Self-test (cmd_selftest())
 * ------------------------------
 *   [1] primality code against a sieve and known primes and pseudoprimes
 *   [2] is_gapless() against brute-force factorization for every m <= 2*10^6
 *   [3] the root method on products of consecutive primes near 2^32, 2^41
 *       and 2^62, and the gap counts of fmt_factor()
 *   [4] the threaded scan against a brute-force scan and the 31 OEIS terms
 *   [5] the 128-bit search against the even OEIS terms, and compute_kmax()
 *   [6] search against scan for every even run length below 10^7
 *   [7] (GMP builds) the big-number gap test, and the GMP search against the
 *       OEIS terms, against a(32) and against the scan
 *
 *
 * 11. Source map (in file order)
 * ------------------------------
 *   generic helpers ........ now, die, xmalloc, u128_str, parse_u128
 *   small primes ........... sp[], small_isp[], r64tab[], mod_small
 *   modular arithmetic ..... mulmod/powmod, sprp, is_prime64/128, next/prev
 *   integer roots .......... isqrt128, iroot128
 *   gapless test ........... chain_from, root_method, is_gapless
 *   full factorization ..... rho64, rho128, factor128, fmt_factor
 *   scan ................... gen_composite_gapless, scan_worker, run_scan
 *   search (128-bit) ....... compute_kmax, test_E, dfs, run_search,
 *                            verify_pair, cmd_search
 *   next, check, run ....... cmd_next, cmd_check, print_run, cmd_run
 *   GMP mode ............... z_is_gapfree, z_test_E, z_test_O,
 *                            run_search_gmp, parse_mpz, cmd_*_gmp
 *   selftest, main
 *
 * Build:  make    (cc -O2 -std=gnu11 -pthread, links -lprimesieve; adds
 *                  -DHAVE_GMP -lgmp when gmp.h is installed)
 */

/*
 * Standard C11 and POSIX headers (threads, atomics, sysconf/isatty/usleep,
 * clock_gettime) and primesieve (https://github.com/kimwalisch/primesieve),
 * which supplies the small-prime table and the fast prime iterators of the
 * scan.  gmp.h is included further down, only when HAVE_GMP is defined.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>
#include <time.h>
#include <inttypes.h>
#include <primesieve.h>

/*
 * Short names for the integer types used everywhere.  unsigned __int128 is a
 * GCC/Clang extension; it carries the 128-bit search and the exact products
 * in the 64-bit modular arithmetic.
 */
typedef uint64_t u64;
typedef uint32_t u32;
typedef unsigned __int128 u128;

/*
 * Compile-time limits (see also header section 9):
 *   SMALL_LIMIT  trial-division bound.  The 6542 primes below 2^16 are
 *                tabulated, and a number with none of them as a factor has
 *                all its prime factors >= 65537.
 *   MAXRUN       size of the per-length result tables; longer runs would be
 *                clamped.
 *   NKNOWN       number of terms built into KNOWN[].
 *   MAXPE        capacity of a factorization record.
 * stderr_tty is set by main(): progress lines are repainted with \r only on a
 * terminal.
 */
#define SMALL_LIMIT 65536u   /* trial-division bound (table of primes below it) */
#define MAXRUN      65536    /* longest run we can record; prime gaps < 2^64 are < 1600 */
#define NKNOWN      31
#define MAXPE       48       /* max distinct primes in a 128-bit number is 26 */

static bool stderr_tty;   /* live \r progress only on a terminal */

/*
 * KNOWN[1..31]: the terms of the OEIS entry as it stood before this program
 * extended it (KNOWN[0] is unused).  The self-test checks the scan and both
 * searches against them, and the tables mark agreement with "ok (OEIS)".  The
 * later terms (several do not fit in 64 bits) are in b137723.txt.
 */
static const u64 KNOWN[NKNOWN + 1] = { 0,
    10, 33, 20, 55, 84, 114, 390, 513, 182, 200, 468, 2941, 774, 65522, 1832,
    1261, 1130, 1332, 1638, 524289, 1952, 4298, 4524, 69960, 5120, 16385, 2972,
    4832, 5352, 10801, 5592 };

/* ------------------------------------------------------------------ */
/* Generic helpers                                                     */
/* ------------------------------------------------------------------ */
/*
 * Timing, fatal errors, checked allocation, 128-bit decimal output and the
 * parser for numeric command-line arguments.
 */

/*
 * now - monotonic time in seconds.
 *
 * Function: reads CLOCK_MONOTONIC; used only for elapsed-time reports.
 * Inputs:   none.
 * Outputs:  seconds since an arbitrary fixed origin.
 * Errors:   none (the clock_gettime() result is not checked; it cannot fail
 *           for CLOCK_MONOTONIC on supported systems).
 */
static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

/*
 * die - report a fatal error and exit.
 *
 * Function: prints "a137723: <message>" and a newline to stderr, then exits
 *           with status 1.  Every unrecoverable condition ends here.
 * Inputs:   a printf-style format string and its arguments.
 * Outputs:  never returns.  exit() flushes stdout, so results printed before
 *           the error are kept.
 * Errors:   none.  Called from a worker thread it ends the whole process,
 *           which is the intended behaviour.
 */
static void die(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fputs("a137723: ", stderr);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(1);
}

/*
 * xmalloc - malloc() that never returns NULL.
 *
 * Function: allocates n bytes, or 1 byte when n is 0, so the result is always
 *           a valid pointer that can be passed to free().
 * Inputs:   n, the size in bytes.
 * Outputs:  a pointer to uninitialized memory; the caller frees it.
 * Errors:   allocation failure -> die("out of memory").  A size computed as
 *           count * size must not overflow (the caller's responsibility).
 */
static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

/*
 * xcalloc - calloc() that never returns NULL.
 *
 * Function: allocates n zero-filled elements of sz bytes (at least one).
 * Inputs:   n, the number of elements; sz, the element size.
 * Outputs:  a pointer to zeroed memory; the caller frees it.
 * Errors:   allocation failure -> die("out of memory").  calloc() itself
 *           rejects an n * sz that overflows.
 */
static void *xcalloc(size_t n, size_t sz)
{
    void *p = calloc(n ? n : 1, sz);
    if (!p) die("out of memory");
    return p;
}

/*
 * u128_str - decimal text of a 128-bit unsigned value.
 *
 * Function: converts v to decimal digits; printf() has no conversion for
 *           unsigned __int128.
 * Inputs:   v; buf, which must hold at least 40 bytes (2^128 - 1 has 39
 *           digits, plus the terminating NUL).
 * Outputs:  buf, NUL-terminated.  The same pointer is returned, so the call
 *           can be used directly as a printf() argument (with a separate
 *           buffer for every value printed by one call).
 * Errors:   none; a buffer shorter than 40 bytes is a caller bug.
 */
static char *u128_str(u128 v, char *buf)
{
    char tmp[48];
    int i = 0, j = 0;
    if (v == 0) tmp[i++] = '0';
    while (v) { tmp[i++] = (char)('0' + (int)(v % 10)); v /= 10; }
    while (i) buf[j++] = tmp[--i];
    buf[j] = 0;
    return buf;
}

/*
 * bitlen128 - number of significant bits of v.
 *
 * Function: floor(log2(v)) + 1, from count-leading-zeros on the two 64-bit
 *           halves.
 * Inputs:   v.
 * Outputs:  0 for v = 0, otherwise 1..128.
 * Errors:   none; v = 0 is handled separately because __builtin_clzll(0) is
 *           undefined.
 */
static int bitlen128(u128 v)
{
    u64 hi = (u64)(v >> 64), lo = (u64)v;
    if (hi) return 128 - __builtin_clzll(hi);
    return lo ? 64 - __builtin_clzll(lo) : 0;
}

/*
 * parse_dec - parse exactly len decimal digits into a 128-bit value.
 *
 * Function: r = 10*r + digit for each character, checking for overflow
 *           before every step.
 * Inputs:   s (need not be NUL-terminated); len; out.
 * Outputs:  true with *out set, or false with *out unchanged.
 * Errors:   false for len = 0, for any character other than '0'..'9' (signs
 *           and spaces included) and for values of 2^128 or more.
 */
static bool parse_dec(const char *s, size_t len, u128 *out)
{
    u128 r = 0;
    if (len == 0) return false;
    for (size_t i = 0; i < len; i++) {
        if (s[i] < '0' || s[i] > '9') return false;
        unsigned dgt = (unsigned)(s[i] - '0');
        if (r > (((u128)0 - 1) - dgt) / 10) return false;
        r = r * 10 + dgt;
    }
    *out = r;
    return true;
}

/*
 * parse_u128 - parse a numeric command-line argument into 128 bits.
 *
 * Function: accepts
 *             decimal              "123456"
 *             power B^E            "2^64", "10^12"      (E <= 400)
 *             scientific MeE       "1e15", "3E9"        (decimal M, E <= 40)
 *             sums, differences    "2^54-32", "10^12+39", "2^64-2^32+1"
 *           An expression is split at its LAST '+' or '-' and both sides are
 *           parsed recursively, so the operators associate to the left.
 * Inputs:   s, a NUL-terminated string; out.
 * Outputs:  true with *out set, or false (*out unspecified).
 * Errors:   false for empty or malformed text, a leading sign, results of
 *           2^128 or more, negative differences, and a left operand longer
 *           than 127 characters.
 */
static bool parse_u128(const char *s, u128 *out)
{
    const char *pm = NULL;                       /* the LAST '+' or '-', not at s[0] */
    for (const char *c = s; *c; c++)
        if ((*c == '+' || *c == '-') && c != s) pm = c;
    if (pm) {
        char head[128];
        size_t hl = (size_t)(pm - s);
        u128 a, b;
        if (hl >= sizeof head) return false;
        memcpy(head, s, hl);
        head[hl] = 0;
        if (!parse_u128(head, &a) || !parse_u128(pm + 1, &b)) return false;
        if (*pm == '+') { if (a > ((u128)0 - 1) - b) return false; *out = a + b; }
        else            { if (b > a) return false; *out = a - b; }
        return true;
    }
    const char *caret = strchr(s, '^');
    const char *ee = strpbrk(s, "eE");
    if (caret) {
        u128 b, e, r = 1;
        if (!parse_dec(s, (size_t)(caret - s), &b)) return false;
        if (!parse_dec(caret + 1, strlen(caret + 1), &e) || e > 400) return false;
        for (unsigned i = 0; i < (unsigned)e; i++) {
            if (b != 0 && r > ((u128)0 - 1) / b) return false;
            r *= b;
        }
        *out = r;
        return true;
    }
    if (ee) {
        u128 m, e;
        if (!parse_dec(s, (size_t)(ee - s), &m)) return false;
        if (!parse_dec(ee + 1, strlen(ee + 1), &e) || e > 40) return false;
        for (unsigned i = 0; i < (unsigned)e; i++) {
            if (m > ((u128)0 - 1) / 10) return false;
            m *= 10;
        }
        *out = m;
        return true;
    }
    return parse_dec(s, strlen(s), out);
}

/*
 * arg_u128 - parse a numeric command-line argument or exit.
 *
 * Function: parse_u128() with a fatal error instead of a return code.
 * Inputs:   s, the argument text; what, its name for the error message.
 * Outputs:  the value.
 * Errors:   malformed or out-of-range input -> die("bad <what>: '<s>'").
 */
static u128 arg_u128(const char *s, const char *what)
{
    u128 v;
    if (!parse_u128(s, &v)) die("bad %s: '%s'", what, s);
    return v;
}

/*
 * arg_u64 - parse a numeric argument that must fit in 64 bits, or exit.
 *
 * Function: arg_u128() plus a range check.
 * Inputs:   s; what, its name for the error message.
 * Outputs:  the value.
 * Errors:   malformed input or a value of 2^64 or more -> die().
 */
static u64 arg_u64(const char *s, const char *what)
{
    u128 v = arg_u128(s, what);
    if (v > UINT64_MAX) die("%s too large (max 2^64-1): '%s'", what, s);
    return (u64)v;
}

/*
 * cmp_u64 - qsort() comparator for u64 values in increasing order.
 *
 * Function: compares with two comparisons (a subtraction could overflow the
 *           int result).
 * Inputs:   pointers to two u64.
 * Outputs:  -1, 0 or 1.
 * Errors:   none.
 */
static int cmp_u64(const void *a, const void *b)
{
    u64 x = *(const u64 *)a, y = *(const u64 *)b;
    return x < y ? -1 : x > y;
}

/* ------------------------------------------------------------------ */
/* Small primes (below SMALL_LIMIT)                                    */
/* ------------------------------------------------------------------ */
/*
 * The 6542 primes below 2^16 drive trial division everywhere, and their
 * indices give the p_k notation of the header (p_k = sp[k-1]).
 */

/*
 * Tables built once by init_small_primes() and read-only afterwards, so every
 * thread may use them without locking:
 *   sp[0..nsp-1]  the nsp = 6542 primes below 2^16 (sp[0] = 2, sp[k-1] = p_k,
 *                 sp[nsp-1] = 65521)
 *   small_isp[x]  1 iff x is prime, for 0 <= x < 2^16
 *   r64tab[i]     2^64 mod sp[i], used by mod_small() to reduce 128-bit values
 */
static u32     *sp;          /* primes < SMALL_LIMIT                       */
static size_t   nsp;
static uint8_t *small_isp;   /* small_isp[x] = 1 iff x is prime, x < 65536 */
static u32     *r64tab;      /* r64tab[i] = 2^64 mod sp[i]                 */

/*
 * init_small_primes - build the small-prime tables sp[], nsp, small_isp[] and
 *                     r64tab[].
 *
 * Function: asks primesieve for every prime below 2^16 (as 32-bit values),
 *           marks them in small_isp[] and precomputes 2^64 mod p for each.
 * Inputs:   none.  Must run once, before anything else uses the tables;
 *           main() calls it first.
 * Outputs:  the global tables.  They live for the whole run and are never
 *           freed.
 * Errors:   primesieve failure -> die(); out of memory -> die().
 */
static void init_small_primes(void)
{
    size_t n = 0;
    u32 *p = (u32 *)primesieve_generate_primes(2, SMALL_LIMIT - 1, &n, UINT32_PRIMES);
    if (!p || n == 0) die("primesieve_generate_primes failed");
    sp = p;
    nsp = n;
    small_isp = xcalloc(SMALL_LIMIT, 1);
    r64tab = xmalloc(n * sizeof(u32));
    for (size_t i = 0; i < n; i++) {
        small_isp[p[i]] = 1;
        r64tab[i] = (u32)((((u128)1) << 64) % p[i]);
    }
}

/*
 * mod_small - m mod sp[i], using only 64-bit divisions.
 *
 * Function: below 2^64 a single 64-bit remainder.  Otherwise, writing
 *           m = hi*2^64 + lo:  m mod p = ((hi mod p)*(2^64 mod p) + lo mod p)
 *           mod p, with 2^64 mod p from r64tab[].  This avoids the slow
 *           software 128-bit division.
 * Inputs:   m; i, an index into sp[] (0 <= i < nsp).
 * Outputs:  m mod sp[i].
 * Errors:   none; every intermediate value stays below 2^33.  An out-of-range
 *           i is a caller bug.
 */
static inline u32 mod_small(u128 m, size_t i)
{
    u32 p = sp[i];
    if (m <= UINT64_MAX) return (u32)((u64)m % p);
    u64 hi = (u64)(m >> 64), lo = (u64)m;
    return (u32)(((hi % p) * (u64)r64tab[i] + lo % p) % p);
}

/* ------------------------------------------------------------------ */
/* Modular arithmetic and primality                                    */
/* ------------------------------------------------------------------ */
/*
 * Two layers: 64-bit arithmetic (exact products through unsigned __int128)
 * for moduli below 2^64, and a slower shift-and-add 128-bit layer that is
 * only needed for the rare 128-bit candidates without small factors.  All
 * functions are pure and thread-safe.
 */

/*
 * mulmod64 - (a * b) mod m for 64-bit operands.
 *
 * Function: a plain 64-bit product when both factors are below 2^32,
 *           otherwise the exact 128-bit product, followed by one remainder.
 * Inputs:   a, b (need not be reduced); m >= 1.
 * Outputs:  a*b mod m.
 * Errors:   m = 0 would divide by zero; no caller passes it.
 */
static inline u64 mulmod64(u64 a, u64 b, u64 m)
{
    if ((a | b) < ((u64)1 << 32)) return (a * b) % m;
    return (u64)(((u128)a * b) % m);
}

/*
 * powmod64 - a^e mod m by square-and-multiply.
 *
 * Function: scans the bits of e from the lowest, squaring a at every step.
 * Inputs:   a (any); e; m >= 1.
 * Outputs:  a^e mod m.  (For e = 0 the result is 1, even when m = 1.)
 * Errors:   m = 0 would divide by zero; no caller passes it.
 */
static u64 powmod64(u64 a, u64 e, u64 m)
{
    u64 r = 1;
    a %= m;
    while (e) {
        if (e & 1) r = mulmod64(r, a, m);
        a = mulmod64(a, a, m);
        e >>= 1;
    }
    return r;
}

/*
 * sprp64 - strong probable-prime (Miller-Rabin) test of n to base a.
 *
 * Function: writes n - 1 = d*2^s with d odd and accepts n if a^d = 1, or
 *           a^(d*2^r) = n - 1 for some 0 <= r < s (all mod n).
 * Inputs:   n odd, n > 2; a, any base.  The base is reduced mod n, and a base
 *           that is a multiple of n counts as a pass, as Sinclair's base set
 *           requires.
 * Outputs:  false: n is certainly composite.  true: n is a strong probable
 *           prime to base a.
 * Errors:   n = 1 would loop forever (d = 0 never becomes odd); is_prime64()
 *           only calls it for n > 1681.
 */
static bool sprp64(u64 n, u64 a)
{
    a %= n;
    if (a == 0) return true;
    u64 d = n - 1;
    int s = 0;
    while (!(d & 1)) { d >>= 1; s++; }
    u64 x = powmod64(a, d, n);
    if (x == 1 || x == n - 1) return true;
    for (int i = 1; i < s; i++) {
        x = mulmod64(x, x, n);
        if (x == n - 1) return true;
        if (x == 1) return false;
    }
    return false;
}

/*
 * is_prime64 - deterministic primality test for 64-bit numbers.
 *
 * Function: trial division by the primes up to 37 (which settles every
 *           n < 41^2), then Miller-Rabin: bases 2, 7 and 61 for n < 4759123141
 *           (Jaeschke), otherwise Sinclair's seven bases, which no composite
 *           below 2^64 passes.
 * Inputs:   any n.
 * Outputs:  true iff n is prime (0 and 1 are not prime).
 * Errors:   none; the answer is exact for every 64-bit input.
 */
static bool is_prime64(u64 n)
{
    static const u32 tiny[] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 };
    static const u64 b3[] = { 2, 7, 61 };
    static const u64 b7[] = { 2, 325, 9375, 28178, 450775, 9780504, 1795265022 };
    if (n < 2) return false;
    for (int i = 0; i < 12; i++)
        if (n % tiny[i] == 0) return n == tiny[i];
    /* no factor up to 37, so every n below 41^2 = 1681 is prime */
    if (n < 41u * 41u) return true;
    /* bases 2, 7 and 61 are a proof below 4759123141 (Jaeschke) */
    if (n < 4759123141ULL) {
        for (int i = 0; i < 3; i++) if (!sprp64(n, b3[i])) return false;
        return true;
    }
    /* Sinclair's bases: no composite below 2^64 passes all seven */
    for (int i = 0; i < 7; i++) if (!sprp64(n, b7[i])) return false;
    return true;
}

/*
 * addmod128 - (a + b) mod m for reduced 128-bit operands.
 *
 * Function: adds, then subtracts m once if the sum reached m.  When m > 2^127
 *           the sum can carry out of 128 bits; that shows as s < a, and the
 *           wrapped subtraction then still gives the right residue.
 * Inputs:   a < m, b < m.
 * Outputs:  (a + b) mod m.
 * Errors:   unreduced inputs give a wrong result (caller contract).
 */
static inline u128 addmod128(u128 a, u128 b, u128 m)   /* a, b < m */
{
    u128 s = a + b;
    if (s < a || s >= m) s -= m;
    return s;
}

/*
 * mulmod128 - (a * b) mod m for 128-bit operands.
 *
 * Function: delegates to mulmod64() when m < 2^64.  Otherwise double-and-add
 *           over the bits of the smaller factor, i.e. up to 128 modular
 *           additions.  Slow, but only used for the few 128-bit candidates
 *           that survive trial division.
 * Inputs:   a, b (any); m >= 1.
 * Outputs:  a*b mod m.
 * Errors:   m = 0 would divide by zero; no caller passes it.
 */
static u128 mulmod128(u128 a, u128 b, u128 m)
{
    if (m <= UINT64_MAX) return mulmod64((u64)(a % m), (u64)(b % m), (u64)m);
    if (a >= m) a %= m;
    if (b >= m) b %= m;
    if (a < b) { u128 t = a; a = b; b = t; }
    u128 r = 0;
    while (b) {
        if (b & 1) r = addmod128(r, a, m);
        a = addmod128(a, a, m);
        b >>= 1;
    }
    return r;
}

/*
 * powmod128 - a^e mod m for 128-bit operands.
 *
 * Function: square-and-multiply with mulmod128().
 * Inputs:   a, e (any); m >= 1.
 * Outputs:  a^e mod m.
 * Errors:   m = 0 would divide by zero; no caller passes it.
 */
static u128 powmod128(u128 a, u128 e, u128 m)
{
    u128 r = 1;
    a %= m;
    while (e) {
        if (e & 1) r = mulmod128(r, a, m);
        a = mulmod128(a, a, m);
        e >>= 1;
    }
    return r;
}

/*
 * sprp128 - strong probable-prime test of n to base a (128-bit sprp64()).
 *
 * Function: n - 1 = d*2^s with d odd; accept if a^d = 1 or a^(d*2^r) = n - 1
 *           (mod n) for some 0 <= r < s.
 * Inputs:   n odd, n > 2; a, any base (reduced mod n; a multiple of n passes).
 * Outputs:  false: certainly composite.  true: strong probable prime to a.
 * Errors:   n = 1 would loop forever; is_prime128() only calls it for
 *           n > 2^64.
 */
static bool sprp128(u128 n, u128 a)
{
    a %= n;
    if (a == 0) return true;
    u128 d = n - 1;
    int s = 0;
    while (!(d & 1)) { d >>= 1; s++; }
    u128 x = powmod128(a, d, n);
    if (x == 1 || x == n - 1) return true;
    for (int i = 1; i < s; i++) {
        x = mulmod128(x, x, n);
        if (x == n - 1) return true;
        if (x == 1) return false;
    }
    return false;
}

/*
 * is_prime128 - primality test for 128-bit numbers.
 *
 * Function: below 2^64 is_prime64() (exact).  Above, trial division by the
 *           primes below 2000, then strong probable-prime tests to the first
 *           24 prime bases (2 .. 89).
 * Inputs:   any n.
 * Outputs:  true iff n is prime, exactly below 2^64; above 2^64 true means
 *           "strong probable prime to 24 bases".
 * Errors:   above 2^64 a composite could in principle pass.  None can below
 *           3.3*10^24, where the first 13 prime bases are already a proof, and
 *           none is known above; reported primes are certified externally
 *           anyway (header section 6).
 */
static bool is_prime128(u128 n)
{
    if (n <= UINT64_MAX) return is_prime64((u64)n);
    for (size_t i = 0; i < nsp && sp[i] < 2000; i++)
        if (mod_small(n, i) == 0) return false;
    for (int i = 0; i < 24; i++)
        if (!sprp128(n, sp[i])) return false;
    return true;
}

/*
 * next_prime128 - smallest prime > x.
 *
 * Function: a lookup in small_isp[] for x < 65521, otherwise odd candidates
 *           tested with is_prime128().
 * Inputs:   x below 2^128 - 159 (the largest prime below 2^128).
 * Outputs:  the next prime (a probable prime above 2^64).
 * Errors:   from x >= 2^128 - 159 on the candidates would wrap around 2^128
 *           and the result would be meaningless; no caller comes near that.
 */
static u128 next_prime128(u128 x)
{
    if (x < 65521) {                 /* 65521 is the largest prime < 2^16 */
        u32 c = (u32)x + 1;
        while (!small_isp[c]) c++;
        return c;
    }
    u128 c = x + 1;
    if (!(c & 1)) c++;
    while (!is_prime128(c)) c += 2;
    return c;
}

/*
 * prev_prime128 - largest prime < x.
 *
 * Function: odd candidates below x tested with is_prime128().
 * Inputs:   any x.
 * Outputs:  the previous prime, or 0 when x <= 2 (there is none).
 * Errors:   none.
 */
static u128 prev_prime128(u128 x)
{
    if (x <= 2) return 0;
    if (x == 3) return 2;
    u128 c = x - 1;
    if (!(c & 1)) c--;
    while (!is_prime128(c)) c -= 2;
    return c;
}

/* ------------------------------------------------------------------ */
/* Integer roots                                                       */
/* ------------------------------------------------------------------ */
/*
 * Exact integer square and t-th roots for the root method (header section 5)
 * and for perfect-power detection in the factorizer.
 */

/*
 * isqrt128 - floor(sqrt(n)) for a 128-bit n.
 *
 * Function: Newton's iteration x <- (x + n/x)/2 from a power of two that is at
 *           least sqrt(n).  The values decrease until they reach
 *           floor(sqrt(n)), and the first step that does not decrease ends it.
 * Inputs:   n.
 * Outputs:  floor(sqrt(n)); n itself for n < 2.
 * Errors:   none; x never exceeds 2^64, so x + n/x cannot overflow.
 */
static u128 isqrt128(u128 n)
{
    if (n < 2) return n;
    u128 x = (u128)1 << ((bitlen128(n) + 1) / 2);   /* x >= sqrt(n) */
    for (;;) {
        u128 y = (x + n / x) >> 1;
        if (y >= x) return x;
        x = y;
    }
}

/*
 * pow_le - is x^t <= n?  Overflow-safe.
 *
 * Function: builds r = x^i one factor at a time and stops as soon as the next
 *           product would exceed n (tested as r > n/x), so no intermediate
 *           value can overflow.
 * Inputs:   x; t >= 0; n.
 * Outputs:  true iff x^t <= n.
 * Errors:   none.
 */
static bool pow_le(u128 x, int t, u128 n)   /* x^t <= n, overflow-safe */
{
    u128 r = 1;
    for (int i = 0; i < t; i++) {
        if (x != 0 && r > n / x) return false;
        r *= x;
    }
    return r <= n;
}

/*
 * pow_eq - is x^t == n?  Overflow-safe, like pow_le().
 *
 * Function: the loop of pow_le() with an equality test at the end.
 * Inputs:   x; t >= 0; n.
 * Outputs:  true iff x^t == n.
 * Errors:   none.
 */
static bool pow_eq(u128 x, int t, u128 n)
{
    u128 r = 1;
    for (int i = 0; i < t; i++) {
        if (x != 0 && r > n / x) return false;
        r *= x;
    }
    return r == n;
}

/*
 * iroot128 - floor(n^(1/t)) for a 128-bit n.
 *
 * Function: t = 2 uses isqrt128().  Otherwise a binary search between 1 and
 *           2^ceil(bits/t) that keeps lo^t <= n < hi^t.
 * Inputs:   n; t.
 * Outputs:  the integer t-th root; n itself when t <= 1 or n < 2.
 * Errors:   none.
 */
static u128 iroot128(u128 n, int t)         /* floor(n^(1/t)) */
{
    if (t <= 1 || n < 2) return n;
    if (t == 2) return isqrt128(n);
    int bits = bitlen128(n);
    u128 lo = 1, hi = (u128)1 << ((bits + t - 1) / t);   /* hi^t > n */
    while (hi - lo > 1) {
        u128 mid = lo + (hi - lo) / 2;
        if (pow_le(mid, t, n)) lo = mid; else hi = mid;
    }
    return lo;
}

/* ------------------------------------------------------------------ */
/* Gapless test (A073490(m) == 0 ?)                                    */
/* ------------------------------------------------------------------ */
/*
 * The core predicate of the program, used by every method (header section 5).
 * The GMP version is z_is_gapfree(), in the GMP section.
 */

/*
 * pe_t: one prime power p^e of a factorization.
 * gl_t: the result of the gapless test: the verdict and, for a gapless number,
 *       its factorization f[0..nf-1] in increasing order of the primes.
 *       MAXPE = 48 entries are plenty: a number below 2^128 has at most 26
 *       distinct primes (2*3*5*...*101 < 2^128 < 2*3*5*...*103).
 */
typedef struct { u128 p; int e; } pe_t;
typedef struct { pe_t f[MAXPE]; int nf; bool gapless; } gl_t;

/*
 * gl_push - append p^e to the factorization in g.
 *
 * Function: stores (p, e) at g->f[g->nf] and increments nf.
 * Inputs:   g; a prime p; an exponent e >= 1.
 * Outputs:  g updated.
 * Errors:   past MAXPE entries the factor is dropped but still counted in nf;
 *           impossible below 2^128 (see gl_t).
 */
static void gl_push(gl_t *g, u128 p, int e)
{
    if (g->nf < MAXPE) { g->f[g->nf].p = p; g->f[g->nf].e = e; }
    g->nf++;
}

/*
 * chain_from - is r exactly p^e1 * nextprime(p)^e2 * nextprime(...)^e3 * ...?
 *
 * Function: divides out p completely, then the next prime, and so on.  It
 *           succeeds when the quotient reaches 1 and fails as soon as the
 *           current prime does not divide.  Called with p = the smallest prime
 *           factor of r, it decides whether r is gapless.
 * Inputs:   r >= 1; a prime p; g for the result.
 * Outputs:  true with *g = the factorization (g->gapless = true), or false
 *           with *g unchanged.
 * Errors:   none.  Cost: one division per prime power plus one next_prime128()
 *           per distinct prime (a table lookup below 2^16, a primality test
 *           above).
 */
static bool chain_from(u128 r, u128 p, gl_t *g)
{
    gl_t t;
    u128 x = r;
    t.nf = 0;
    for (;;) {
        if (x % p) return false;
        int e = 0;
        do { x /= p; e++; } while (x % p == 0);
        gl_push(&t, p, e);
        if (x == 1) break;
        p = next_prime128(p);
    }
    t.gapless = true;
    *g = t;
    return true;
}

/*
 * root_method - gapless test for a composite r without prime factors below
 *               2^16 (step (iii) of header section 5).
 *
 * Function: for t = 2, 3, ... while c = floor(r^(1/t)) > 2^16 (every factor
 *           exceeds 2^16, so r has fewer than log2(r)/16 of them): if r = c^t
 *           with c prime, r is gapless.  Otherwise, were r gapless with t
 *           factors, its smallest prime would be one of the t-1 largest primes
 *           <= c; each of those is tried with chain_from().
 * Inputs:   r, composite, with no prime factor below SMALL_LIMIT (so
 *           r > 2^32); g.
 * Outputs:  true with *g filled if r is gapless; false (g->gapless = false)
 *           otherwise.
 * Errors:   none.  t never exceeds 7 below 2^128 (the loop bound 8 is a
 *           margin).  Primality above 2^64 is probabilistic (is_prime128()).
 */
static bool root_method(u128 r, gl_t *g)
{
    for (int t = 2; t <= 8; t++) {
        u128 c = iroot128(r, t);
        if (c <= SMALL_LIMIT) break;
        bool exact = pow_eq(c, t, r);
        if (exact && is_prime128(c)) {
            g->nf = 0;
            gl_push(g, c, t);
            g->gapless = true;
            return true;
        }
        /* q = the largest prime <= c (below c when c^t = r with c composite) */
        u128 q = exact ? prev_prime128(c) : (is_prime128(c) ? c : prev_prime128(c));
        for (int j = 0; j < t - 1 && q > SMALL_LIMIT; j++) {
            if (chain_from(r, q, g)) return true;
            if (j + 1 < t - 1) q = prev_prime128(q);       /* only when another candidate follows */
        }
    }
    g->gapless = false;
    return false;
}

/*
 * is_gapless - is A073490(m) = 0, i.e. are the distinct primes of m
 *              consecutive?  The hot path of every search.
 *
 * Function: (i) trial division by the primes below 2^16 until the smallest
 *           prime factor p appears, then chain_from(m, p).  Below 2^64 the
 *           loop also stops once p^2 > m, which proves m prime.  (ii) No
 *           factor below 2^16: m is gapless if it is prime.  (iii) Otherwise
 *           root_method().  See header section 5.
 * Inputs:   any m; g, optional (NULL when only the verdict is needed).
 * Outputs:  true iff m is gapless (1 is gapless, 0 is not).  If g is given it
 *           receives the verdict and, for a gapless m, the factorization (one
 *           entry m^1 for a prime m).
 * Errors:   none.  Exact below 2^64; above, the primality steps use
 *           is_prime128(), whose "prime" verdict is probabilistic.
 *           Thread-safe: it only reads the global tables.
 */
static bool is_gapless(u128 m, gl_t *g)
{
    gl_t tmp;
    if (!g) g = &tmp;
    g->nf = 0;
    g->gapless = false;
    if (m == 0) return false;
    if (m == 1) { g->gapless = true; return true; }
    if (m <= UINT64_MAX) {
        u64 m64 = (u64)m;
        for (size_t i = 0; i < nsp; i++) {
            u64 p = sp[i];
            if (p * p > m64) {                 /* m is prime */
                gl_push(g, m, 1);
                g->gapless = true;
                return true;
            }
            if (m64 % p == 0) return chain_from(m, p, g);
        }
    } else {
        for (size_t i = 0; i < nsp; i++)
            if (mod_small(m, i) == 0) return chain_from(m, sp[i], g);
    }
    /* no prime factor <= SMALL_LIMIT, and m > SMALL_LIMIT^2 */
    if (is_prime128(m)) {
        gl_push(g, m, 1);
        g->gapless = true;
        return true;
    }
    return root_method(m, g);
}

/* ------------------------------------------------------------------ */
/* Full factorization (for display; Pollard-Brent rho)                 */
/* ------------------------------------------------------------------ */
/*
 * Only for output (`check`, `run`, the run tables) and for verify_pair().  The
 * searches never factor anything completely: is_gapless() decides
 * gaplessness without it.
 */

/*
 * fac_t: a factorization for display, possibly partial.  f[0..nf-1] are the
 * primes found (sorted by factor128_ex()), and unf^unf_e is a composite
 * cofactor that could not be split (unf = 0 when the factorization is
 * complete).
 */
typedef struct { pe_t f[MAXPE]; int nf; u128 unf; int unf_e; } fac_t;

/*
 * fac_add - add p^e to a factorization, merging with an existing entry for p.
 *
 * Function: a linear search for p; increments its exponent or appends it.
 * Inputs:   F; a prime p; e >= 1.
 * Outputs:  F updated.
 * Errors:   more than MAXPE distinct primes cannot occur below 2^128; extra
 *           primes would be dropped silently.
 */
static void fac_add(fac_t *F, u128 p, int e)
{
    for (int i = 0; i < F->nf; i++)
        if (F->f[i].p == p) { F->f[i].e += e; return; }
    if (F->nf < MAXPE) { F->f[F->nf].p = p; F->f[F->nf].e = e; F->nf++; }
}

/*
 * gcd64 - greatest common divisor of two 64-bit numbers (Euclid).
 *
 * Function: repeated remainders.
 * Inputs:   a, b.
 * Outputs:  gcd(a, b); gcd(a, 0) = a.
 * Errors:   none.
 */
static u64 gcd64(u64 a, u64 b)
{
    while (b) { u64 t = a % b; a = b; b = t; }
    return a;
}

/*
 * gcd128 - greatest common divisor of two 128-bit numbers (Euclid).
 *
 * Function: repeated remainders.
 * Inputs:   a, b.
 * Outputs:  gcd(a, b); gcd(a, 0) = a.
 * Errors:   none.
 */
static u128 gcd128(u128 a, u128 b)
{
    while (b) { u128 t = a % b; a = b; b = t; }
    return a;
}

/*
 * addc64 - (x + c) mod n for x, c < n, safe against a carry out of 64 bits.
 *
 * Function: adds, and subtracts n if the sum carried or reached n.
 * Inputs:   x < n; c < n.
 * Outputs:  (x + c) mod n.
 * Errors:   unreduced inputs give a wrong result (caller contract).
 */
static inline u64 addc64(u64 x, u64 c, u64 n)   /* (x + c) mod n, x < n, c < n */
{
    u64 s = x + c;
    if (s < x || s >= n) s -= n;
    return s;
}

/*
 * rho64 - a nontrivial factor of a 64-bit composite (Pollard-Brent rho).
 *
 * Function: iterates y <- y^2 + c mod n with Brent's cycle detection,
 *           multiplying the differences |x - y| into q and taking one gcd per
 *           batch of up to 128 steps.  If a batch overshoots (gcd = n) it
 *           backtracks one step at a time from ys; if the sequence still only
 *           yields n, the next constant c is tried.
 * Inputs:   n, composite and not a perfect power (factor_rec() removes those
 *           first).
 * Outputs:  a divisor g with 1 < g < n (2 for even n).
 * Errors:   none for valid input; the expected work is about n^(1/4) steps.
 *           For a prime n or n = 1 it would loop forever, which the callers
 *           rule out.
 */
static u64 rho64(u64 n)
{
    if (n % 2 == 0) return 2;
    for (u64 c = 1;; c++) {
        u64 y = 2, r = 1, q = 1, g = 1, x = 0, ys = 0;
        do {
            x = y;
            for (u64 i = 0; i < r; i++) y = addc64(mulmod64(y, y, n), c, n);
            u64 k = 0;
            do {
                ys = y;
                /* up to 128 steps share one gcd */
                u64 lim = (r - k < 128) ? r - k : 128;
                for (u64 i = 0; i < lim; i++) {
                    y = addc64(mulmod64(y, y, n), c, n);
                    q = mulmod64(q, x > y ? x - y : y - x, n);
                }
                g = gcd64(q, n);
                k += lim;
            } while (k < r && g == 1);
            r <<= 1;
        } while (g == 1);
        if (g == n) {
            do {
                ys = addc64(mulmod64(ys, ys, n), c, n);
                g = gcd64(x > ys ? x - ys : ys - x, n);
            } while (g == 1);
        }
        if (g != n) return g;
    }
}

/*
 * rho128 - Pollard-Brent rho for a 128-bit n, with an iteration budget.
 *
 * Function: the algorithm of rho64() with mulmod128() arithmetic.  Only the
 *           batched steps count against the budget; the steps that advance y
 *           before each batch do not, so the work can be about twice the
 *           budget.
 * Inputs:   n, composite and not a perfect power; budget (factor_rec() uses
 *           2^22).
 * Outputs:  a nontrivial divisor, or 0 if the budget runs out (typical when
 *           the smallest prime factor of n exceeds about 2^44).
 * Errors:   none; 0 is the failure value.
 */
static u128 rho128(u128 n, u64 budget)
{
    if (n % 2 == 0) return 2;
    u64 used = 0;
    for (u128 c = 1; used < budget; c++) {
        u128 y = 2, r = 1, q = 1, g = 1, x = 0, ys = 0;
        do {
            x = y;
            for (u128 i = 0; i < r; i++) y = addmod128(mulmod128(y, y, n), c, n);
            u128 k = 0;
            do {
                ys = y;
                u128 lim = (r - k < 128) ? r - k : 128;
                for (u128 i = 0; i < lim; i++) {
                    y = addmod128(mulmod128(y, y, n), c, n);
                    q = mulmod128(q, x > y ? x - y : y - x, n);
                }
                g = gcd128(q, n);
                k += lim;
                used += (u64)lim;
            } while (k < r && g == 1);
            r <<= 1;
        } while (g == 1 && used < budget);
        if (g == 1) return 0;
        if (g == n) {
            do {
                ys = addmod128(mulmod128(ys, ys, n), c, n);
                g = gcd128(x > ys ? x - ys : ys - x, n);
            } while (g == 1);
        }
        if (g != n) return g;
    }
    return 0;
}

/*
 * factor_rec - split n into primes, recording each with multiplicity mult.
 *
 * Function: n prime: record it.  n = c^t, a perfect power: recurse on c with
 *           multiplicity mult*t (rho does not split prime powers reliably).
 *           Otherwise split n with rho64() (below 2^64) or, if heavy, with
 *           rho128(), and recurse on both parts.
 * Inputs:   n >= 1; F, the factorization being built; mult >= 1; heavy:
 *           whether to try the slow 128-bit rho.
 * Outputs:  primes added to F.  A part that cannot be split is stored as
 *           F->unf^F->unf_e.
 * Errors:   none fatal.  Only one unsplit cofactor can be stored, and a second
 *           would overwrite the first.  That cannot happen for 128-bit input
 *           after trial division: rho128() only fails on parts whose factors
 *           all exceed about 2^44, and two such parts would exceed 2^128.
 */
static void factor_rec(u128 n, fac_t *F, int mult, bool heavy)
{
    if (n == 1) return;
    if (is_prime128(n)) { fac_add(F, n, mult); return; }
    for (int t = 2; t <= 127; t++) {           /* perfect powers */
        u128 c = iroot128(n, t);
        if (c < 2) break;
        if (pow_eq(c, t, n)) { factor_rec(c, F, mult * t, heavy); return; }
    }
    u128 d = 0;
    if (n <= UINT64_MAX) d = rho64((u64)n);
    else if (heavy) d = rho128(n, (u64)1 << 22);
    if (d == 0 || d == n) { F->unf = n; F->unf_e = mult; return; }
    factor_rec(d, F, mult, heavy);
    factor_rec(n / d, F, mult, heavy);
}

/*
 * cmp_pe - qsort() comparator for pe_t by prime, increasing.
 *
 * Function: compares the p fields (without a subtraction).
 * Inputs:   pointers to two pe_t.
 * Outputs:  -1, 0 or 1.
 * Errors:   none.
 */
static int cmp_pe(const void *a, const void *b)
{
    const pe_t *x = a, *y = b;
    return x->p < y->p ? -1 : x->p > y->p;
}

/*
 * factor128_ex - factor m as far as possible, for display or verification.
 *
 * Function: trial division by the primes below 2^16 (for a remainder below
 *           2^64 it stops once p^2 exceeds it, which proves the rest prime),
 *           then factor_rec() on the cofactor, then sorts the primes.
 * Inputs:   m; F, the result; heavy: also try rho128() on a composite cofactor
 *           above 2^64 (slower, and it can still fail).
 * Outputs:  *F with the primes found.  If the factorization is incomplete,
 *           the unsplit cofactor is in F->unf; every prime below 2^16 was then
 *           tried, so all its prime factors exceed 2^16.  m < 2 gives an empty
 *           F.
 * Errors:   none fatal.  A cofactor above 2^64 stays unfactored when heavy is
 *           false or when rho128() exhausts its budget.
 */
static void factor128_ex(u128 m, fac_t *F, bool heavy)
{
    memset(F, 0, sizeof *F);
    if (m < 2) return;
    for (size_t i = 0; i < nsp; i++) {
        if (mod_small(m, i) == 0) {
            int e = 0;
            do { m /= sp[i]; e++; } while (mod_small(m, i) == 0);
            fac_add(F, sp[i], e);
        }
        if (m <= UINT64_MAX && (u64)sp[i] * sp[i] > (u64)m) break;
    }
    factor_rec(m, F, 1, heavy);
    qsort(F->f, (size_t)F->nf, sizeof(pe_t), cmp_pe);
}

/*
 * factor128 - full-effort factorization: factor128_ex(m, F, true).
 *
 * Function: wrapper.
 * Inputs:   m; F.
 * Outputs:  *F as described at factor128_ex().
 * Errors:   as factor128_ex(); a cofactor above 2^64 may stay unfactored.
 */
static void factor128(u128 m, fac_t *F)
{
    factor128_ex(m, F, true);
}

/*
 * fmt_factor_ex - human-readable factorization with prime-gap markers.
 *
 * Function: factor128() (full effort), then prints the primes in increasing
 *           order: consecutive primes are joined by '*' and a prime gap is
 *           shown as " | ", e.g. 84 -> "2^2*3 | 7" (with markers false every
 *           separator is '*').  An unsplit cofactor is appended as "[C...]".
 * Inputs:   m; buf and its size cap; markers.
 * Outputs:  buf, always NUL-terminated (truncated to cap).  Returns the
 *           number of gaps, i.e. A073490(m), or -1 when a cofactor stayed
 *           unfactored.  For m = 0 and m = 1 it prints "0" or "1" and
 *           returns 0.
 * Errors:   truncation is silent.  The 0 returned for m = 0 is not a claim
 *           that 0 is gapless; cmd_check() rejects m = 0.
 */
static int fmt_factor_ex(u128 m, char *buf, size_t cap, bool markers)
{
    fac_t F;
    size_t pos = 0;
    int gaps = 0;
    if (m < 2) { snprintf(buf, cap, "%s", m ? "1" : "0"); return 0; }
    factor128(m, &F);
    for (int i = 0; i < F.nf && pos < cap; i++) {
        char pb[48];
        if (i) {
            bool gap = next_prime128(F.f[i - 1].p) != F.f[i].p;
            if (gap) gaps++;
            pos += (size_t)snprintf(buf + pos, cap - pos, (gap && markers) ? " | " : "*");
            if (pos >= cap) break;
        }
        pos += (size_t)snprintf(buf + pos, cap - pos, "%s", u128_str(F.f[i].p, pb));
        if (pos >= cap) break;
        if (F.f[i].e > 1) pos += (size_t)snprintf(buf + pos, cap - pos, "^%d", F.f[i].e);
    }
    if (F.unf && pos < cap) {
        char pb[48];
        pos += (size_t)snprintf(buf + pos, cap - pos, "%s[C%s]", F.nf ? "*" : "", u128_str(F.unf, pb));
        if (F.unf_e > 1 && pos < cap) snprintf(buf + pos, cap - pos, "^%d", F.unf_e);
        gaps = -1;
    }
    return gaps;
}

/*
 * fmt_factor - fmt_factor_ex() with gap markers.
 *
 * Function: wrapper; see fmt_factor_ex().
 * Inputs:   m; buf; cap.
 * Outputs:  as fmt_factor_ex(): the text and the gap count (or -1).
 * Errors:   as fmt_factor_ex().
 */
static int fmt_factor(u128 m, char *buf, size_t cap)
{
    return fmt_factor_ex(m, buf, cap, true);
}

/*
 * describe_gapless - short structural description of a gapless number.
 *
 * Function: "1", "prime", or the factorization from is_gapless(), e.g.
 *           "2^24*3^28" or "31*37".
 * Inputs:   m; buf and its size cap.
 * Outputs:  buf (truncated to cap).
 * Errors:   if m is not gapless the text is "NOT GAPLESS?!", an alarm that
 *           should never appear because callers only pass bounds of runs.
 *           m = 0 also produces it.
 */
static void describe_gapless(u128 m, char *buf, size_t cap)
{
    gl_t g;
    if (m == 1) { snprintf(buf, cap, "1"); return; }
    if (!is_gapless(m, &g)) { snprintf(buf, cap, "NOT GAPLESS?!"); return; }
    if (g.nf == 1 && g.f[0].e == 1) { snprintf(buf, cap, "prime"); return; }
    size_t pos = 0;
    for (int i = 0; i < g.nf && i < MAXPE && pos < cap; i++) {
        char pb[48];
        pos += (size_t)snprintf(buf + pos, cap - pos, "%s%s", i ? "*" : "", u128_str(g.f[i].p, pb));
        if (pos < cap && g.f[i].e > 1) pos += (size_t)snprintf(buf + pos, cap - pos, "^%d", g.f[i].e);
    }
}

/* ------------------------------------------------------------------ */
/* scan: exhaustive enumeration of gapless numbers up to N              */
/* ------------------------------------------------------------------ */
/*
 * Method 1 of the header (section 2): generate the gapless numbers up to N in
 * increasing order and record the first run of every length.
 */

/*
 * vec64_t: a growable array of u64; v[0..n-1] are in use, cap are allocated.
 */
typedef struct { u64 *v; size_t n, cap; } vec64_t;

/*
 * vec64_push - append x to a vec64_t.
 *
 * Function: doubles the capacity when full (4096 entries at first).
 * Inputs:   v (zero-initialized before the first push); x.
 * Outputs:  v->v[v->n++] = x.
 * Errors:   out of memory -> die().
 */
static void vec64_push(vec64_t *v, u64 x)
{
    if (v->n == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 4096;
        v->v = realloc(v->v, v->cap * sizeof(u64));
        if (!v->v) die("out of memory");
    }
    v->v[v->n++] = x;
}

/*
 * comp_extend - push every v * pr[idx]^e1 * pr[idx+1]^e2 * ... <= N.
 *
 * Function: v is gapless and its largest prime is pr[idx-1].  Each product
 *           v * pr[idx]^e (e >= 1) up to N is gapless with one more prime: it
 *           is pushed, and the recursion then tries to append pr[idx+1].
 * Inputs:   out; v; idx; the prime table pr[0..np-1]; np; N.
 * Outputs:  the new numbers appended to out (not sorted).
 * Errors:   none.  It returns when idx >= np, which loses nothing: if
 *           v * pr[idx] <= N then pr[idx-1] < sqrt(N), so by Bertrand's
 *           postulate pr[idx] < 2*sqrt(N), which is inside the table.  The
 *           u128 products cannot overflow (v <= N < 2^64, pr[idx] < 2^33), and
 *           the recursion is at most 15 deep (2*3*...*47 < 2^64).
 */
static void comp_extend(vec64_t *out, u128 v, size_t idx, const u64 *pr, size_t np, u64 N)
{
    if (idx >= np) return;    /* pr[idx] > 2 sqrt(N): no product can fit */
    u128 w = v * pr[idx];
    while (w <= N) {
        vec64_push(out, (u64)w);
        comp_extend(out, w, idx + 1, pr, np, N);
        w *= pr[idx];
    }
}

/*
 * gen_composite_gapless - every composite gapless number <= N, plus 1,
 *                         sorted.
 *
 * Function: for each prime p with p^2 <= N: the powers p^e (e >= 2) and,
 *           through comp_extend(), every p^e * nextprime(p)^f * ... <= N.
 *           Each composite gapless number arises exactly once, from its
 *           smallest prime and its exponents.  The number 1 is included
 *           because it is gapless too, so the scan can start with it.
 * Inputs:   N < 2^64; out, an empty vec64_t.
 * Outputs:  out->v[0..n-1] in increasing order, starting with 1.
 * Errors:   primesieve failure or out of memory -> die().  Memory is
 *           dominated by the prime table up to P = 2*sqrt(N) + 10^5, 8 bytes
 *           per prime (about 2.3 GB at N = 2^63); the output itself is small
 *           (about 9*10^4 entries at N = 10^11).
 */
static void gen_composite_gapless(u64 N, vec64_t *out)
{
    u64 P = 2 * (u64)isqrt128(N) + 100000;
    size_t np = 0;
    u64 *pr = (u64 *)primesieve_generate_primes(2, P, &np, UINT64_PRIMES);
    if (!pr || np == 0) die("primesieve_generate_primes failed");
    vec64_push(out, 1);
    for (size_t i = 0; i < np; i++) {
        u128 p = pr[i];
        if (p * p > N) break;
        u128 v = p;
        while (v <= N) {
            if (v != p) vec64_push(out, (u64)v);          /* p^e, e >= 2 */
            comp_extend(out, v, i + 1, pr, np, N);        /* p^e * next primes */
            v *= p;
        }
    }
    primesieve_free(pr);
    qsort(out->v, out->n, sizeof(u64), cmp_u64);
}

/*
 * lower_bound64 - the first index i with a[i] >= x in a sorted array.
 *
 * Function: binary search.
 * Inputs:   a[0..n-1], sorted increasing; n; x.
 * Outputs:  the index, or n if every element is below x.
 * Errors:   none.
 */
static size_t lower_bound64(const u64 *a, size_t n, u64 x)
{
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (a[mid] < x) lo = mid + 1; else hi = mid;
    }
    return lo;
}

/*
 * scanres_t: the results of one scan worker: first[n] = the smallest start of
 *            a run of length n it has seen (0 = none), and maxrun = the
 *            longest run it has seen.  About 512 KB, one per thread.
 * SC:        the scan descriptor shared by all workers, set by run_scan()
 *            before they start: the bound N, the chunk size and count, the
 *            atomic counters next (next chunk to claim) and done (chunks
 *            finished, for the progress line), and the sorted list of
 *            composite gapless numbers.
 */
typedef struct { u64 first[MAXRUN]; int maxrun; } scanres_t;

static struct {
    u64 N, chunk, nchunks;
    atomic_uint_fast64_t next, done;
    const u64 *comp;
    size_t ncomp;
} SC;

/*
 * scan_worker - thread body of the scan.
 *
 * Function: claims chunks [lo, hi) of [1, N+1) until none are left.  For each
 *           chunk it positions a primesieve iterator at lo, binary-searches
 *           the composite list for lo, and merges the two streams.  For every
 *           consecutive pair (g, g2) with lo <= g < hi it records g+1 as a
 *           start for the length n = g2 - g - 1 if none was seen yet.  The last
 *           g of the chunk is paired with the first gapless number past hi, so
 *           a run that crosses a chunk boundary is counted once, by the chunk
 *           that holds its lower bound.
 * Inputs:   arg, this thread's zeroed scanres_t; the global SC.
 * Outputs:  res->first[] and res->maxrun; SC.done is incremented per chunk.
 * Errors:   primesieve errors are not checked: primesieve_next_prime() would
 *           return PRIMESIEVE_ERROR (UINT64_MAX), which cannot happen because
 *           main() keeps N <= 2^63.  Lengths of MAXRUN or more would be
 *           clamped to MAXRUN-1; prime gaps below 2^64 are shorter than 1600.
 *           A composite never equals a prime, so the merge sees no ties.
 */
static void *scan_worker(void *arg)
{
    scanres_t *res = arg;
    primesieve_iterator it;
    primesieve_init(&it);
    for (;;) {
        u64 ci = atomic_fetch_add(&SC.next, 1);
        if (ci >= SC.nchunks) break;
        u64 lo = ci * SC.chunk, hi = (ci + 1) * SC.chunk;
        if (lo < 1) lo = 1;
        if (hi > SC.N + 1) hi = SC.N + 1;         /* process pairs (g, g') with lo <= g < hi */
        if (lo >= hi) { atomic_fetch_add(&SC.done, 1); continue; }

        /* both streams start at the first gapless number >= lo; g is the current one */
        primesieve_jump_to(&it, lo, hi + 1000000);
        u64 p = primesieve_next_prime(&it);
        size_t idx = lower_bound64(SC.comp, SC.ncomp, lo);
        u64 c = idx < SC.ncomp ? SC.comp[idx++] : UINT64_MAX;
        u64 g;
        if (c < p) { g = c; c = idx < SC.ncomp ? SC.comp[idx++] : UINT64_MAX; }
        else       { g = p; p = primesieve_next_prime(&it); }
        /* every pair (g, g2) with g in this chunk; the last g2 may lie past hi */
        while (g < hi) {
            u64 g2;
            if (c < p) { g2 = c; c = idx < SC.ncomp ? SC.comp[idx++] : UINT64_MAX; }
            else       { g2 = p; p = primesieve_next_prime(&it); }
            u64 n = g2 - g - 1;
            if (n) {
                if (n >= MAXRUN) n = MAXRUN - 1;
                if (!res->first[n]) res->first[n] = g + 1;
                if ((int)n > res->maxrun) res->maxrun = (int)n;
            }
            g = g2;
        }
        atomic_fetch_add(&SC.done, 1);
    }
    primesieve_free_iterator(&it);
    return NULL;
}

/*
 * run_scan - exhaustive scan of 1..N with `threads` workers.
 *
 * Function: builds the composite gapless list up to N + 2*10^6 (a margin for
 *           the pair that crosses N), sizes the chunks (about 64 per thread
 *           for load balance, at least 2^22 numbers each), runs the workers,
 *           prints progress if asked, and merges the per-thread minima.
 * Inputs:   N <= 2^63; threads >= 1; first, an array of MAXRUN entries;
 *           maxrun; verbose (progress on stderr).
 * Outputs:  first[n] = a(n) for every n whose first run starts at or below
 *           N + 1, and 0 for the others (first[0] is unused); *maxrun = the
 *           longest run seen.
 * Errors:   out of memory, primesieve or pthread_create() failure -> die().
 *           Not reentrant (uses the global SC).
 */
static void run_scan(u64 N, int threads, u64 *first, int *maxrun, bool verbose)
{
    double t0 = now();
    vec64_t comp = { 0, 0, 0 };
    /* the margin covers the pair that crosses N (gaps below 2^64 are < 1600) */
    gen_composite_gapless(N + 2000000, &comp);
    if (verbose)
        fprintf(stderr, "[scan] %zu composite gapless numbers <= %" PRIu64 " generated in %.2fs\n",
                comp.n - 1, N, now() - t0);

    SC.N = N;
    SC.comp = comp.v;
    SC.ncomp = comp.n;
    /* about 64 chunks per thread for load balance, each at least 2^22 numbers */
    u64 target = (u64)threads * 64;
    u64 chunk = (N + target) / target;
    if (chunk < ((u64)1 << 22)) chunk = (u64)1 << 22;
    SC.chunk = chunk;
    SC.nchunks = (N + chunk) / chunk;
    atomic_store(&SC.next, 0);
    atomic_store(&SC.done, 0);

    scanres_t *res = xcalloc((size_t)threads, sizeof *res);
    pthread_t *tid = xmalloc((size_t)threads * sizeof *tid);
    double t1 = now();
    for (int i = 0; i < threads; i++)
        if (pthread_create(&tid[i], NULL, scan_worker, &res[i])) die("pthread_create failed");
    if (verbose) {
        double last = 0;
        for (;;) {
            u64 done = atomic_load(&SC.done);
            double el = now() - t1;
            if (done >= SC.nchunks) break;
            if (done && (stderr_tty || el - last >= 15.0)) {
                double eta = el * (double)(SC.nchunks - done) / (double)done;
                fprintf(stderr, "%s[scan] %5.1f%%  %" PRIu64 "/%" PRIu64 " chunks  %.0fs elapsed  ETA %.0fs   %s",
                        stderr_tty ? "\r" : "", 100.0 * (double)done / (double)SC.nchunks, done, SC.nchunks,
                        el, eta, stderr_tty ? "" : "\n");
                last = el;
            }
            usleep(200000);
        }
        fprintf(stderr, "%s[scan] done: %" PRIu64 " chunks of %" PRIu64 " in %.2fs%40s\n",
                stderr_tty ? "\r" : "", SC.nchunks, chunk, now() - t1, "");
    }
    for (int i = 0; i < threads; i++) pthread_join(tid[i], NULL);

    memset(first, 0, sizeof(u64) * MAXRUN);
    *maxrun = 0;
    for (int i = 0; i < threads; i++) {
        if (res[i].maxrun > *maxrun) *maxrun = res[i].maxrun;
        for (int n = 1; n <= res[i].maxrun; n++)
            if (res[i].first[n] && (!first[n] || res[i].first[n] < first[n]))
                first[n] = res[i].first[n];
    }
    free(res);
    free(tid);
    free(comp.v);
}

/*
 * print_table - the result table: one line per n with a(n), both bounding
 *               gapless numbers with their structure, and a note.
 *
 * Function: for n = 1..top prints "n  a(n)  L (structure)  R (structure)
 *           note".  The note compares n <= 31 with the built-in OEIS terms
 *           ("ok (OEIS)" or "MISMATCH!"), marks values above the scan bound N
 *           as found by the structural search, and calls the rest "new"
 *           (relative to the 31 built-in terms).
 * Inputs:   fp; a[1..top] (0 = unknown; 128-bit so search results fit); top;
 *           N, the scan bound; note_unknown, the text for unknown entries.
 * Outputs:  text on fp.
 * Errors:   none.  "NOT GAPLESS?!" in a structure column would indicate a bug
 *           elsewhere.
 */
static void print_table(FILE *fp, const u128 *a, int top, u64 N, const char *note_unknown)
{
    char b1[48], b2[48], b3[48], d1[200], d2[200];
    fprintf(fp, "%4s  %-40s  %-30s  %-30s  %s\n", "n", "a(n)", "L = a(n)-1", "R = a(n)+n", "note");
    for (int n = 1; n <= top; n++) {
        if (!a[n]) {
            fprintf(fp, "%4d  %-40s  %-30s  %-30s  %s\n", n, note_unknown, "", "",
                    (n <= NKNOWN) ? "!! OEIS has a value here" : "");
            continue;
        }
        u128 L = a[n] - 1, R = a[n] + (u128)n;
        describe_gapless(L, d1, sizeof d1);
        describe_gapless(R, d2, sizeof d2);
        char lb[256], rb[256];
        snprintf(lb, sizeof lb, "%s (%s)", u128_str(L, b2), d1);
        snprintf(rb, sizeof rb, "%s (%s)", u128_str(R, b3), d2);
        const char *note;
        char nb[80];
        if (n <= NKNOWN) {
            if (a[n] == KNOWN[n]) note = "ok (OEIS)";
            else { snprintf(nb, sizeof nb, "MISMATCH! OEIS has %" PRIu64, KNOWN[n]); note = nb; }
        } else if (a[n] > N) {
            note = "NEW (structural search)";
        } else {
            note = "new";
        }
        fprintf(fp, "%4d  %-40s  %-30s  %-30s  %s\n", n, u128_str(a[n], b1), lb, rb, note);
    }
}

/*
 * print_summary - the consecutive range, the DATA line and the unknown n.
 *
 * Function: finds the largest k with a(1..k) all known, prints those terms as
 *           an OEIS DATA line, and lists up to 40 unknown n <= top.
 * Inputs:   fp; a[1..top]; top.
 * Outputs:  text on fp.
 * Errors:   none.
 */
static void print_summary(FILE *fp, const u128 *a, int top)
{
    int k = 0;
    while (k + 1 <= top && a[k + 1]) k++;
    fprintf(fp, "\nConsecutive terms known: a(1..%d)\n", k);
    if (k) {
        fprintf(fp, "DATA: ");
        for (int n = 1; n <= k; n++) {
            char b[48];
            fprintf(fp, "%s%s", n > 1 ? ", " : "", u128_str(a[n], b));
        }
        fprintf(fp, "\n");
        if (k > NKNOWN) fprintf(fp, "(%d term%s beyond the %d built-in OEIS terms)\n", k - NKNOWN, k - NKNOWN > 1 ? "s" : "", NKNOWN);
    }
    int miss = 0;
    for (int n = k + 1; n <= top; n++) if (!a[n]) miss++;
    if (miss) {
        fprintf(fp, "Unknown n up to %d:", top);
        int shown = 0;
        for (int n = k + 1; n <= top && shown < 40; n++)
            if (!a[n]) { fprintf(fp, " %d", n); shown++; }
        if (shown < miss) fprintf(fp, " ... (%d total)", miss);
        fprintf(fp, "\n");
    }
}

/*
 * cmd_scan - the `scan N` command.
 *
 * Function: run_scan() with progress, then the table for n = 1..top (top =
 *           the longest run found, capped by -m, but at least 31) and the
 *           summary, on stdout and, with -o, in FILE.
 * Inputs:   N (100 <= N <= 2^63, enforced by main()); threads; maxn (0 = no
 *           cap); outfile (NULL = none).
 * Outputs:  the table on stdout and in FILE; progress on stderr.
 * Errors:   FILE cannot be created -> die(), after the table went to stdout.
 *           Uses static arrays of about 1.5 MB, so it is not reentrant.
 */
static void cmd_scan(u64 N, int threads, int maxn, const char *outfile)
{
    static u64 first[MAXRUN];
    static u128 a[MAXRUN];
    int maxrun = 0;
    double t0 = now();
    fprintf(stderr, "[scan] A137723: exhaustive scan of 1..%" PRIu64 " with %d threads\n", N, threads);
    run_scan(N, threads, first, &maxrun, true);
    fprintf(stderr, "[scan] total %.2fs, longest run found: %d\n", now() - t0, maxrun);
    for (int n = 0; n < MAXRUN; n++) a[n] = first[n];
    int top = maxrun;
    if (maxn > 0 && maxn < top) top = maxn;
    if (top < NKNOWN) top = NKNOWN;
    char nb[64];
    snprintf(nb, sizeof nb, "> %" PRIu64, N);
    printf("A137723 scan: every a(n) <= %" PRIu64 " (longest run found: %d)\n", N, maxrun);
    print_table(stdout, a, top, N, nb);
    print_summary(stdout, a, top);
    if (outfile) {
        FILE *fp = fopen(outfile, "w");
        if (!fp) die("cannot write %s", outfile);
        fprintf(fp, "A137723 scan: every a(n) <= %" PRIu64 " (longest run found: %d)\n", N, maxrun);
        print_table(fp, a, top, N, nb);
        print_summary(fp, a, top);
        fclose(fp);
        fprintf(stderr, "[scan] table written to %s\n", outfile);
    }
}

/* ------------------------------------------------------------------ */
/* search: structural search for even n                                */
/* ------------------------------------------------------------------ */
/*
 * Method 2 of the header (section 3): for even n, enumerate the even gapless
 * endpoints E up to a 128-bit limit and test both partners E +- d.
 */

/*
 * pair_t:  one run found by a search: its bounding gapless numbers L and
 *          R = L + d.
 * sres_t:  the results of one search worker: the pairs v[0..n-1] (growable)
 *          and the number of even endpoints tested.
 * stask_t: one unit of search work: an even endpoint E with k primes
 *          (2^a, 2^a*3^b or 2^a*3^b*5^c) and whether the worker should go on to
 *          append 7, 11, ... (only for k = 3; the extensions of the k = 1 and
 *          k = 2 tasks by 3 and 5 are tasks of their own).
 * SQ:      the search descriptor shared by the workers: limit, d = n + 1, n,
 *          kmax, the task list, and the atomic counters next (next task to
 *          claim) and done (tasks finished).
 */
typedef struct { u128 L, R; } pair_t;
typedef struct { pair_t *v; size_t n, cap; u64 tested; } sres_t;

typedef struct { u128 E; int k; bool recurse; } stask_t;

static struct {
    u128 limit;
    u64 d;
    int n, kmax;
    stask_t *tasks;
    size_t ntasks;
    atomic_size_t next, done;
} SQ;

/*
 * compute_kmax - the most primes an even endpoint can have for this d.
 *
 * Function: walks the odd primes 3, 5, 7, ... (up to p_60 = 281) and tracks
 *           the first block of consecutive primes dividing d.  At the first
 *           prime factor of d that is not adjacent to that block, E can no
 *           longer contain it (header section 3), so the answer is its index
 *           minus one.
 * Inputs:   d = n + 1, odd.
 * Outputs:  kmax (always >= 3), or INT_MAX if there is no such factor (the
 *           odd prime factors of d up to 281 form one block, or there are
 *           none).
 * Errors:   none.  Factors of d above 281 are not examined, so kmax can come
 *           out larger than necessary.  That only weakens the pruning, never
 *           the correctness; below 2^128 an even gapless number has at most
 *           26 primes anyway.
 *           Examples: d = 33 = 3*11 -> 4;  d = 21 = 3*7 -> 3;  d = 35 -> INT_MAX.
 */
static int compute_kmax(u64 d)
{
    int first = -1, last = -1;
    for (int k = 2; k <= 60; k++) {
        u32 p = sp[k - 1];
        if (d % p == 0) {
            if (first < 0) first = last = k;
            else if (last == k - 1) last = k;
            else return k - 1;
        }
    }
    return INT_MAX;
}

/*
 * compute_kprime - the most primes E can have while its partner can still be
 *                  a (large) prime.
 *
 * Function: the smallest odd prime p_k dividing d divides every E with at
 *           least k primes, and therefore its partner O, so O can only be
 *           prime when E has fewer than k primes.
 * Inputs:   d, odd.
 * Outputs:  k - 1 for the index k of the smallest odd prime factor of d (if
 *           that factor is at most p_60 = 281), otherwise INT_MAX.
 * Errors:   none.  Only used in the progress report; the search tests every
 *           partner regardless.  (O = p_k itself is a prime divisible by p_k;
 *           such tiny cases are covered by the search anyway.)
 */
static int compute_kprime(u64 d)
{
    for (int k = 2; k <= 60; k++)
        if (d % sp[k - 1] == 0) return k - 1;
    return INT_MAX;
}

/*
 * add_pair - append the pair (L, R) to a result list.
 *
 * Function: doubles the capacity when full (64 entries at first).
 * Inputs:   r (zero-initialized before the first use); L; R.
 * Outputs:  r->v[r->n++] = (L, R).
 * Errors:   out of memory -> die().
 */
static void add_pair(sres_t *r, u128 L, u128 R)
{
    if (r->n == r->cap) {
        r->cap = r->cap ? r->cap * 2 : 64;
        r->v = realloc(r->v, r->cap * sizeof(pair_t));
        if (!r->v) die("out of memory");
    }
    r->v[r->n].L = L;
    r->v[r->n].R = R;
    r->n++;
}

/*
 * interior_gapful - is every number strictly between L and R gapful?
 *
 * Function: is_gapless() on L+1, ..., R-1, stopping at the first gapless one.
 * Inputs:   L < R.
 * Outputs:  true iff none of them is gapless.
 * Errors:   none.  Up to d-1 tests, but only reached when the partner O is
 *           gapless, which is rare.
 */
static bool interior_gapful(u128 L, u128 R)
{
    for (u128 m = L + 1; m < R; m++)
        if (is_gapless(m, NULL)) return false;
    return true;
}

/*
 * test_E - try both partners of one even endpoint E (128-bit search).
 *
 * Function: O = E + d gives the candidate pair (E, O), and O = E - d (when
 *           E > d) gives (O, E).  A pair is recorded when O is gapless and the
 *           numbers in between are all gapful.  E is gapless by construction
 *           and is not tested here.
 * Inputs:   E, even and gapless, at most SQ.limit <= 2^127 (so E + d cannot
 *           overflow); res, the worker's result list.
 * Outputs:  pairs appended to res; res->tested incremented.
 * Errors:   none.  O = 1 (for E = d + 1) counts as gapless, consistent with
 *           the scan, which treats 1 as gapless too.
 */
static void test_E(u128 E, sres_t *res)
{
    u128 O;
    res->tested++;
    O = E + SQ.d;
    if (is_gapless(O, NULL) && interior_gapful(E, O)) add_pair(res, E, O);
    if (E > SQ.d) {
        O = E - SQ.d;
        if (is_gapless(O, NULL) && interior_gapful(O, E)) add_pair(res, O, E);
    }
}

/*
 * dfs - extend an even endpoint by the next prime, depth first.
 *
 * Function: E has the k-1 primes p_1..p_(k-1).  For e = 1, 2, ... while
 *           E * p_k^e <= limit, the product is tested and, if k+1 <= kmax,
 *           extended in turn by p_(k+1).
 * Inputs:   E; k, the index of the prime to append (k >= 4, since the tasks
 *           go up to 5); res.
 * Outputs:  pairs appended to res through test_E().
 * Errors:   none.  The test p > limit/w prevents overflow.  The depth is at
 *           most 26 below 2^128.
 */
static void dfs(u128 E, int k, sres_t *res)
{
    u128 p = sp[k - 1], w = E;
    for (;;) {
        if (p > SQ.limit / w) break;
        w *= p;
        test_E(w, res);
        if (k + 1 <= SQ.kmax) dfs(w, k + 1, res);
    }
}

/*
 * search_worker - thread body of the 128-bit search.
 *
 * Function: claims tasks through SQ.next.  For each task it tests E and, if
 *           the task is marked for recursion (k = 3 and kmax >= 4), explores
 *           every extension by 7, 11, ... with dfs().
 * Inputs:   arg, this thread's zeroed sres_t; the global SQ.
 * Outputs:  pairs and counts in the sres_t; SQ.done incremented per task.
 * Errors:   out of memory inside add_pair() -> die().
 */
static void *search_worker(void *arg)
{
    sres_t *res = arg;
    for (;;) {
        size_t ti = atomic_fetch_add(&SQ.next, 1);
        if (ti >= SQ.ntasks) break;
        stask_t *t = &SQ.tasks[ti];
        test_E(t->E, res);
        if (t->recurse) dfs(t->E, t->k + 1, res);
        atomic_fetch_add(&SQ.done, 1);
    }
    return NULL;
}

/*
 * cmp_pair - qsort() comparator for pair_t by L, increasing.
 *
 * Function: compares the L fields (without a subtraction).
 * Inputs:   pointers to two pair_t.
 * Outputs:  -1, 0 or 1.
 * Errors:   none.
 */
static int cmp_pair(const void *a, const void *b)
{
    const pair_t *x = a, *y = b;
    return x->L < y->L ? -1 : x->L > y->L;
}

/*
 * add_task - append a task to a growable stask_t array.
 *
 * Function: doubles the capacity when full (1024 entries at first).
 * Inputs:   t, n, cap (the array, its length and capacity); E; k; rec
 *           (whether the worker should recurse).
 * Outputs:  the task appended; *t, *n and *cap updated.
 * Errors:   out of memory -> die().
 */
static void add_task(stask_t **t, size_t *n, size_t *cap, u128 E, int k, bool rec)
{
    if (*n == *cap) {
        *cap = *cap ? *cap * 2 : 1024;
        *t = realloc(*t, *cap * sizeof(stask_t));
        if (!*t) die("out of memory");
    }
    (*t)[*n].E = E;
    (*t)[*n].k = k;
    (*t)[*n].recurse = rec;
    (*n)++;
}

/*
 * gapful_by_partial - does a partial factorization already prove a prime gap?
 *
 * Function: F holds the primes found and an unsplit cofactor whose primes all
 *           exceed 2^16 (see factor128_ex()).  Every prime below 2^16 was
 *           tried by trial division, so a prime q < 2^16 missing from F does
 *           not divide m.  A gap is therefore certain when, for some known
 *           prime p, the next prime q is below 2^16 and missing while m has a
 *           prime larger than p: the next known prime, or any prime of the
 *           cofactor when p is the largest known prime.
 * Inputs:   F from factor128_ex(), with F->unf != 0.
 * Outputs:  true: m certainly has a prime gap.  false: undecided.
 * Errors:   none.  Primes above 2^16 that are absent from F may be hiding in
 *           the cofactor, so they never count as missing.
 */
static bool gapful_by_partial(const fac_t *F)
{
    if (F->nf == 0) return false;
    for (int i = 1; i < F->nf; i++) {
        u128 q = next_prime128(F->f[i - 1].p);
        if (q != F->f[i].p && q < SMALL_LIMIT) return true;   /* q was tried: missing */
    }
    return next_prime128(F->f[F->nf - 1].p) < SMALL_LIMIT;
}

/*
 * verify_pair - independent re-check of one run, using full factorizations.
 *
 * Function: factors L and R (fmt_factor()) and every interior number, and
 *           checks the gap counts taken from the factorizations: 0 for L and
 *           R, at least 1 for each interior number.  For an interior number a
 *           cheap factorization (no 128-bit rho) comes first and often already
 *           proves a gap through gapful_by_partial(); otherwise the
 *           full-effort one decides.  is_gapless() is not used, so an error in
 *           the search's gap test would show up here.
 * Inputs:   L < R; why and its size cap, for a message.
 * Outputs:  1: verified.  0: contradiction, described in why.  2: could not be
 *           certified because a number above 2^64 could not be factored
 *           completely, described in why.
 * Errors:   none fatal.  Primality inside the factorizer is the same
 *           is_prime128() the search uses, so this is not an independent
 *           primality check (use verify_run.py or PARI/GP for that).  Interior
 *           numbers may be factored up to three times, which is only slow for
 *           runs above 2^64.
 */
static int verify_pair(u128 L, u128 R, char *why, size_t cap)
{
    char fb[512], b[48];
    fac_t F;
    int g, result = 1;
    g = fmt_factor(L, fb, sizeof fb);
    if (g > 0) { snprintf(why, cap, "L not gapless: %s", fb); return 0; }
    if (g < 0) { snprintf(why, cap, "L not fully factored: %s", fb); result = 2; }
    g = fmt_factor(R, fb, sizeof fb);
    if (g > 0) { snprintf(why, cap, "R not gapless: %s", fb); return 0; }
    if (g < 0) { snprintf(why, cap, "R not fully factored: %s", fb); result = 2; }
    for (u128 m = L + 1; m < R; m++) {
        factor128_ex(m, &F, false);                 /* cheap: no 128-bit rho */
        if (F.unf && gapful_by_partial(&F)) continue;   /* gap already certain */
        g = fmt_factor(m, fb, sizeof fb);           /* full effort */
        if (g == 0) { snprintf(why, cap, "%s is gapless: %s", u128_str(m, b), fb); return 0; }
        if (g < 0) {
            factor128(m, &F);
            if (!gapful_by_partial(&F)) {
                snprintf(why, cap, "%s not fully factored: %s", u128_str(m, b), fb);
                result = 2;
            }
        }
    }
    return result;
}

/*
 * run_search - the 128-bit structural search for one even n (header
 *              section 3).
 *
 * Function: sets up SQ (d = n + 1, kmax), builds the task list -- 2^a (k = 1),
 *           2^a*3^b (k = 2) and 2^a*3^b*5^c (k = 3, recursing further when
 *           kmax >= 4), each product checked against the limit before it is
 *           formed -- runs the workers, collects and sorts the pairs by L, and
 *           re-checks every pair with is_gapless() and interior_gapful() (the
 *           enumeration never tests E itself, so this guards it).
 * Inputs:   n, even, >= 2; limit <= 2^127; threads >= 1; verbose (report and
 *           progress on stderr); out, optional.
 * Outputs:  a(n) = L + 1 for the smallest pair found, or 0 if no pair has its
 *           even endpoint at or below the limit.  By the coverage lemma a
 *           nonzero result is exactly a(n), and 0 means a(n) > limit - n.  If
 *           out is given it receives all pairs sorted by L (the caller frees
 *           out->v); otherwise they are freed here.
 * Errors:   out of memory or pthread_create() failure -> die().  A pair that
 *           fails the re-check is reported as "INTERNAL ERROR" on stderr and
 *           dropped.  Not reentrant (global SQ).  A limit below 2 gives no
 *           tasks and returns 0.
 */
static u128 run_search(int n, u128 limit, int threads, bool verbose, sres_t *out)
{
    char lb[48];
    u64 d = (u64)n + 1;
    SQ.limit = limit;
    SQ.d = d;
    SQ.n = n;
    SQ.kmax = compute_kmax(d);
    int kprime = compute_kprime(d);

    /* tasks: 2^a (k=1), 2^a 3^b (k=2), 2^a 3^b 5^c (k=3, recursing into 7, 11, ...) */
    stask_t *tasks = NULL;
    size_t ntasks = 0, cap = 0;
    /* every product is guarded against 128-bit overflow before it is formed */
    for (u128 A = 2; A <= limit; A *= 2) {
        add_task(&tasks, &ntasks, &cap, A, 1, false);
        if (SQ.kmax >= 2 && A <= limit / 3) {
            for (u128 B = A * 3; B <= limit; B *= 3) {
                add_task(&tasks, &ntasks, &cap, B, 2, false);
                if (SQ.kmax >= 3 && B <= limit / 5) {
                    for (u128 C = B * 5; C <= limit; C *= 5) {
                        add_task(&tasks, &ntasks, &cap, C, 3, SQ.kmax >= 4);
                        if (C > limit / 5) break;
                    }
                }
                if (B > limit / 3) break;
            }
        }
        if (A > limit / 2) break;
    }
    SQ.tasks = tasks;
    SQ.ntasks = ntasks;
    atomic_store(&SQ.next, 0);
    atomic_store(&SQ.done, 0);

    if (verbose) {
        char db[200];
        fmt_factor_ex(d, db, sizeof db, false);
        fprintf(stderr, "[search] n = %d, d = n+1 = %" PRIu64 " = %s, limit = %s, %d threads\n",
                n, d, db, u128_str(limit, lb), threads);
        fprintf(stderr, "[search] even endpoint E = 2^e1*3^e2*...*p_k^ek: ");
        if (SQ.kmax == INT_MAX) fprintf(stderr, "any k; ");
        else fprintf(stderr, "k <= %d; ", SQ.kmax);
        if (kprime == INT_MAX) fprintf(stderr, "a prime odd partner is possible for every k\n");
        else if (kprime == 1) fprintf(stderr, "a prime odd partner needs k = 1 (E = 2^a)\n");
        else fprintf(stderr, "a prime odd partner needs k <= %d\n", kprime);
        fprintf(stderr, "[search] %zu top-level tasks\n", ntasks);
    }

    sres_t *res = xcalloc((size_t)threads, sizeof *res);
    pthread_t *tid = xmalloc((size_t)threads * sizeof *tid);
    double t0 = now();
    for (int i = 0; i < threads; i++)
        if (pthread_create(&tid[i], NULL, search_worker, &res[i])) die("pthread_create failed");
    if (verbose) {
        double last = 0;
        for (;;) {
            size_t done = atomic_load(&SQ.done);
            if (done >= ntasks) break;
            double el = now() - t0;
            if (el > 2.0 && (stderr_tty || el - last >= 15.0)) {
                fprintf(stderr, "%s[search] %zu/%zu tasks  %.0fs   %s", stderr_tty ? "\r" : "",
                        done, ntasks, el, stderr_tty ? "" : "\n");
                last = el;
            }
            usleep(200000);
        }
    }
    for (int i = 0; i < threads; i++) pthread_join(tid[i], NULL);

    sres_t all = { NULL, 0, 0, 0 };
    for (int i = 0; i < threads; i++) {
        all.tested += res[i].tested;
        for (size_t j = 0; j < res[i].n; j++) add_pair(&all, res[i].v[j].L, res[i].v[j].R);
        free(res[i].v);
    }
    free(res);
    free(tid);
    free(tasks);
    qsort(all.v, all.n, sizeof(pair_t), cmp_pair);
    if (verbose)
        fprintf(stderr, "%s[search] enumerated %" PRIu64 " even gapless E in %.2fs, %zu run%s of exactly %d found\n",
                stderr_tty ? "\r" : "", all.tested, now() - t0, all.n, all.n == 1 ? "" : "s", n);

    /* consistency check of every pair (the enumeration never tested E itself) */
    size_t kept = 0;
    for (size_t i = 0; i < all.n; i++) {
        char b1[48], b2[48];
        if (is_gapless(all.v[i].L, NULL) && is_gapless(all.v[i].R, NULL) && interior_gapful(all.v[i].L, all.v[i].R)) {
            all.v[kept++] = all.v[i];
        } else {
            fprintf(stderr, "[search] INTERNAL ERROR: dropped invalid pair L=%s R=%s\n",
                    u128_str(all.v[i].L, b1), u128_str(all.v[i].R, b2));
        }
    }
    all.n = kept;

    u128 result = 0;
    if (all.n) result = all.v[0].L + 1;
    if (out) *out = all; else free(all.v);
    return result;
}

/*
 * cmd_search - the `search n` command in 128-bit mode.
 *
 * Function: validates the arguments, runs run_search(), prints up to 40 runs
 *           found with the structure of their bounds, re-checks the smallest
 *           with verify_pair(), and prints a(n), or the bound a(n) > limit - n
 *           when nothing was found.
 * Inputs:   n; limit, with n + 1 <= limit <= 2^127; threads.
 * Outputs:  the report on stdout (flushed); progress on stderr.
 * Errors:   n < 1, odd n, limit > 2^127 or limit < n + 1 -> die().  When the
 *           smallest pair found starts above limit - n the report calls it a
 *           candidate; by the coverage lemma it is still exactly a(n), so that
 *           wording is merely conservative.
 */
static void cmd_search(int n, u128 limit, int threads)
{
    char b1[48], b2[48], b3[48], d1[200], d2[200];
    if (n < 1) die("n must be >= 1");
    if (n % 2) die("search handles even n only (odd n: the two bounding gapless numbers "
                   "have the same parity, so use 'scan')");
    if (limit > ((u128)1 << 127)) die("limit must be <= 2^127");
    if (limit < (u128)n + 1) die("limit must be at least n+1 = %d", n + 1);
    sres_t all;
    u128 a = run_search(n, limit, threads, true, &all);
    u64 d = (u64)n + 1;
    printf("\n=== A137723 search: n = %d (d = %" PRIu64 "), even endpoint <= %s ===\n", n, d, u128_str(limit, b1));
    if (all.n == 0) {
        printf("No run of exactly %d gapful numbers starts at or below %s.\n", n, u128_str(limit - d + 1, b1));
        printf("a(%d) > %s\n", n, u128_str(limit - d + 1, b1));
    } else {
        size_t show = all.n < 40 ? all.n : 40;
        printf("Runs of exactly %d gapful numbers (bounded by gapless L and R = L+%" PRIu64 "):\n", n, d);
        for (size_t i = 0; i < show; i++) {
            describe_gapless(all.v[i].L, d1, sizeof d1);
            describe_gapless(all.v[i].R, d2, sizeof d2);
            printf("  start %-40s L = %s (%s)  R = %s (%s)\n", u128_str(all.v[i].L + 1, b1),
                   u128_str(all.v[i].L, b2), d1, u128_str(all.v[i].R, b3), d2);
        }
        if (show < all.n) printf("  ... %zu more\n", all.n - show);
        char why[600];
        int vr = verify_pair(all.v[0].L, all.v[0].R, why, sizeof why);
        printf("\nIndependent verification of the smallest run (full factorizations): %s%s\n",
               vr == 1 ? "OK" : vr == 0 ? "FAILED: " : "INCOMPLETE: ", vr == 1 ? "" : why);
        printf("a(%d) = %s", n, u128_str(a, b1));
        /* the second message is only conservative: by the coverage lemma the
         * smallest pair found is a(n) either way */
        if (a - 1 <= limit - d) printf("   (exhaustive: every run of exactly %d starting <= %s was enumerated)\n",
                                       n, u128_str(limit - d + 1, b2));
        else printf("   (candidate only; smaller L in (%s, %s] were not all covered)\n",
                    u128_str(limit - d, b2), u128_str(limit, b3));
        if (n <= NKNOWN) printf("OEIS value: %" PRIu64 " -> %s\n", KNOWN[n], a == KNOWN[n] ? "match" : "MISMATCH");
    }
    fflush(stdout);
    free(all.v);
}

/* ------------------------------------------------------------------ */
/* next: scan, then structurally search the missing even n             */
/* ------------------------------------------------------------------ */
/*
 * The combined pipeline: an exhaustive scan for every n, then the 128-bit
 * structural search for each even n that the scan could not reach.
 */

/*
 * cmd_next - the `next` command: scan, then fill in the even n by search.
 *
 * Function: run_scan(N), then run_search(n, limit) for every even n <= top
 *           (the longest run found, capped by -m, at least 32) that the scan
 *           did not find; odd n can only come from the scan.  Prints the
 *           merged table and summary on stdout and, with -o, in FILE.
 * Inputs:   N, the scan bound; limit <= 2^127 (main() rejects GMP-sized
 *           limits); threads; maxn (0 = no cap); outfile (NULL = none).
 * Outputs:  the table on stdout and in FILE; per-n results and progress on
 *           stderr.
 * Errors:   limit > 2^127 or FILE not writable -> die().  For n >= limit the
 *           bound printed for an unresolved n is the trivial a(n) > 0.
 */
static void cmd_next(u64 N, u128 limit, int threads, int maxn, const char *outfile)
{
    static u64 first[MAXRUN];
    static u128 a[MAXRUN];
    int maxrun = 0;
    char lb[48];
    if (limit > ((u128)1 << 127)) die("limit must be <= 2^127");
    fprintf(stderr, "[next] step 1: exhaustive scan to %" PRIu64 "\n", N);
    run_scan(N, threads, first, &maxrun, true);
    for (int n = 0; n < MAXRUN; n++) a[n] = first[n];
    int top = maxrun;
    if (maxn > 0 && maxn < top) top = maxn;
    if (top < NKNOWN + 1) top = NKNOWN + 1;

    fprintf(stderr, "[next] step 2: structural search (even endpoint <= %s) for each missing even n <= %d\n",
            u128_str(limit, lb), top);
    for (int n = 2; n <= top; n += 2) {
        if (a[n]) continue;
        u128 r = run_search(n, limit, threads, true, NULL);
        if (r) {
            a[n] = r;
            fprintf(stderr, "[next]   a(%d) = %s\n", n, u128_str(r, lb));
            fflush(stderr);
        } else {
            fprintf(stderr, "[next]   a(%d) > %s\n", n, u128_str(limit > (u128)n ? limit - (u128)n : 0, lb));
        }
    }
    char nb[100];
    snprintf(nb, sizeof nb, "> %" PRIu64 " (odd n) / > limit-n (even n)", N);
    FILE *outs[2] = { stdout, NULL };
    if (outfile && !(outs[1] = fopen(outfile, "w"))) die("cannot write %s", outfile);
    for (int i = 0; i < 2 && outs[i]; i++) {
        fprintf(outs[i], "A137723: exhaustive scan to %" PRIu64 " plus structural search (even endpoint <= %s) for even n\n",
                N, u128_str(limit, lb));
        print_table(outs[i], a, top, N, nb);
        print_summary(outs[i], a, top);
    }
    if (outs[1]) { fclose(outs[1]); fprintf(stderr, "[next] table written to %s\n", outfile); }
}

/* ------------------------------------------------------------------ */
/* check / run                                                         */
/* ------------------------------------------------------------------ */
/*
 * Inspection commands for single numbers below 2^127 (GMP builds) or below
 * 2^128 (without GMP).  The GMP versions are in the GMP section.
 */

/*
 * cmd_check - the `check m` command for m below 2^127 (any 128-bit m without
 *             GMP).
 *
 * Function: prints the factorization with gap markers and A073490(m),
 *           cross-checks the verdict against is_gapless(), and says whether m
 *           is prime.
 * Inputs:   m >= 1.
 * Outputs:  text on stdout.
 * Errors:   m = 0 -> die() (0 has no factorization).  If a cofactor cannot be
 *           factored, A073490(m) is reported as unknown.
 */
static void cmd_check(u128 m)
{
    char b[48], fb[600];
    if (m == 0) die("m must be >= 1 (0 has no factorization)");
    int g = fmt_factor(m, fb, sizeof fb);
    printf("%s = %s\n", u128_str(m, b), fb);
    if (g < 0) printf("A073490(m): unknown (unfactored part)\n");
    else printf("A073490(%s) = %d  ->  %s\n", u128_str(m, b), g, g ? "has a prime gap (gapful)" : "no prime gap (gapless, in A073491)");
    bool gl = is_gapless(m, NULL);
    if (g >= 0 && gl != (g == 0)) printf("!! is_gapless() disagrees with the factorization\n");
    if (is_prime128(m)) printf("%s is prime\n", u128_str(m, b));
}

/*
 * print_run - print one run (L, R) with the factorization of every number.
 *
 * Function: a header with both bounds and their structure; for runs of up to
 *           200 numbers one fmt_factor() line per interior number; then, for
 *           n <= 31, whether this is the first run of its length according to
 *           the built-in OEIS terms.
 * Inputs:   L < R, consecutive gapless numbers.
 * Outputs:  text on stdout.
 * Errors:   none.  Fully factoring 128-bit interior numbers can take a moment
 *           and may leave [C...] cofactors.  For n > 31 the last line only
 *           says "candidate"; compare it with the b-file (b137723.txt).
 */
static void print_run(u128 L, u128 R)
{
    char b1[48], b2[48], d1[200], d2[200], fb[600];
    u128 n = R - L - 1;
    describe_gapless(L, d1, sizeof d1);
    describe_gapless(R, d2, sizeof d2);
    printf("Run of %s gapful numbers: %s .. %s\n", u128_str(n, b1), u128_str(L + 1, b2), u128_str(R - 1, fb));
    printf("  bounded below by %s (%s) and above by %s (%s)\n", u128_str(L, b1), d1, u128_str(R, b2), d2);
    if (n <= 200) {
        for (u128 m = L + 1; m < R; m++) {
            fmt_factor(m, fb, sizeof fb);
            printf("  %s = %s\n", u128_str(m, b1), fb);
        }
    } else {
        printf("  (%s numbers, factorizations omitted)\n", u128_str(n, b1));
    }
    if (n <= NKNOWN && n >= 1) {
        int ni = (int)n;
        if (L + 1 == KNOWN[ni]) printf("  -> this is a(%d) = %" PRIu64 " (OEIS)\n", ni, KNOWN[ni]);
        else printf("  -> a(%d) = %" PRIu64 " (OEIS), so this is a later run of that length\n", ni, KNOWN[ni]);
    } else if (n >= 1) {
        printf("  -> candidate for a(%s) = %s (beyond the %d built-in terms; compare with b137723.txt)\n",
               u128_str(n, b1), u128_str(L + 1, b2), NKNOWN);
    }
}

/*
 * cmd_run - the `run m` command for m below 2^127 (any 128-bit m without
 *           GMP).
 *
 * Function: if m is gapless, prints the runs just below and just above it;
 *           otherwise finds the gapless numbers L < m < R around m and prints
 *           that run.
 * Inputs:   m >= 2.
 * Outputs:  text on stdout.
 * Errors:   m < 2 -> die().  Without GMP, an m in the last prime gap below
 *           2^128 (m > 2^128 - 159) would make R wrap around 2^128; with GMP
 *           such m go to cmd_run_gmp() instead.
 */
static void cmd_run(u128 m)
{
    char b[48], d[200];
    if (m < 2) die("m must be >= 2");
    if (is_gapless(m, NULL)) {
        describe_gapless(m, d, sizeof d);
        printf("%s is gapless (%s); showing the runs on both sides.\n\n", u128_str(m, b), d);
        u128 L = m - 1;
        while (L >= 1 && !is_gapless(L, NULL)) L--;
        if (L < m - 1) print_run(L, m);
        else printf("(no run below: %s is also gapless)\n", u128_str(L, b));
        u128 R = m + 1;
        while (!is_gapless(R, NULL)) R++;
        printf("\n");
        if (R > m + 1) print_run(m, R);
        else printf("(no run above: %s is also gapless)\n", u128_str(R, b));
        return;
    }
    u128 L = m - 1, R = m + 1;
    while (!is_gapless(L, NULL)) L--;
    while (!is_gapless(R, NULL)) R++;
    print_run(L, R);
}

/* ------------------------------------------------------------------ */
/* GMP mode: structural search for even n with LIMIT >= 2^127          */
/* ------------------------------------------------------------------ */
/*
 * Same theorem as the 128-bit search, applied more selectively so that the
 * enumeration stays small at hundreds or thousands of bits (header section 4).
 * Let d = n+1 (odd), p_j the smallest odd prime dividing d, and {p_j..p_m} the
 * maximal block of consecutive primes from p_j on that all divide d.  For an
 * even endpoint E = 2^e1 3^e2 ... p_k^ek the odd partner O = E +- d satisfies
 *
 *   k <  j      : O is coprime to p_1..p_k             -> may be prime: enumerate E
 *   j <= k <= m : O = p_j^f..p_k^f.. times a consecutive continuation beyond p_k
 *                                                       -> open-ended: enumerate E
 *   m <  k      : O = p_j^f * ... * p_m^f EXACTLY       -> enumerate these O instead
 *
 * (k > kmax is impossible, kmax = the index of the next odd prime factor of d
 * minus one).  So E is enumerated only for k <= m (the "E side"), and for k > m
 * the partner O ranges over the products of the fixed primes p_j..p_m (the "O
 * side").  For d = 63 = 3^2*7: j = m = 2 and kmax = 3; the E side is 2^a and
 * 2^a*3^b, the O side the powers of 3, about 3*10^5 candidates below 2^1000.
 *
 * Everything here works on mpz_t; numbers below 2^127 are handed to the
 * 128-bit routines.  Prime verdicts come from z_is_prime() (probabilistic
 * above 2^64).  Each worker owns a zws_t, so the hot paths never allocate.
 */
#ifdef HAVE_GMP
#include <gmp.h>
#include <math.h>

/*
 * ZDEPTH: the number of mpz slots per worker for the E-side and O-side
 * recursions (one per prime appended).  The E side needs at most m of them
 * (m = the end of the block of d, below 10 for every d whose search is
 * feasible), so 128 is plenty; run_search_gmp() checks the bound.
 */
#define ZDEPTH 128

/*
 * zws_t: one worker's GMP scratch space, so the hot loops never allocate.
 *   x        the running quotient in z_chain_from() and z_partial_line()
 *   p, q, c  prime candidates and roots in z_chain_from() and z_root_method()
 *   tmp      the loop variable of z_interior_gapful(); also used for printing
 *   E, O     the current even endpoint (z_test_O()) and odd partner
 *            (z_test_E()); z_partial_line() keeps its cofactor in E
 *   stk[]    one value per recursion level of z_dfs() and z_dfs_O()
 * z_is_gapfree() clobbers x, p, q and c, never tmp, E, O or stk[]; the
 * functions below rely on that.
 */
typedef struct {
    mpz_t x, p, q, c, tmp, E, O;
    mpz_t stk[ZDEPTH];
} zws_t;

/*
 * zws_init - initialize every mpz_t of a zws_t.
 *
 * Function: mpz_init() on all fields (no large allocations yet).
 * Inputs:   w, uninitialized.
 * Outputs:  w ready for use; release it with zws_clear().
 * Errors:   GMP aborts the process if memory runs out (its default policy).
 */
static void zws_init(zws_t *w)
{
    mpz_inits(w->x, w->p, w->q, w->c, w->tmp, w->E, w->O, NULL);
    for (int i = 0; i < ZDEPTH; i++) mpz_init(w->stk[i]);
}

/*
 * zws_clear - release every mpz_t of a zws_t.
 *
 * Function: mpz_clear() on all fields.
 * Inputs:   w, initialized by zws_init().
 * Outputs:  w must not be used again until it is re-initialized.
 * Errors:   none.
 */
static void zws_clear(zws_t *w)
{
    mpz_clears(w->x, w->p, w->q, w->c, w->tmp, w->E, w->O, NULL);
    for (int i = 0; i < ZDEPTH; i++) mpz_clear(w->stk[i]);
}

/*
 * z_is_prime - probable-prime test for a number of any size.
 *
 * Function: mpz_probab_prime_p(m, 30): trial division, a Baillie-PSW test and
 *           6 more Miller-Rabin rounds (GMP 6.2 and later).
 * Inputs:   m >= 0.
 * Outputs:  true if m is prime or a probable prime; false if m is certainly
 *           composite or m < 2.
 * Errors:   no Baillie-PSW pseudoprime is known, but a true result above 2^64
 *           is not a proof; reported endpoints should be certified.
 */
static bool z_is_prime(const mpz_t m)
{
    return mpz_probab_prime_p(m, 30) > 0;
}

/*
 * z_prevprime - largest (probable) prime below x.
 *
 * Function: steps down through the odd numbers below x with z_is_prime().
 * Inputs:   out; x >= 4 (callers only pass values above 2^16).  out may be the
 *           same variable as x.
 * Outputs:  out = the previous prime.
 * Errors:   for x <= 3 the search would continue below 2 into negative
 *           numbers; no caller does that.
 */
static void z_prevprime(mpz_t out, const mpz_t x)
{
    mpz_sub_ui(out, x, 1);
    if (mpz_even_p(out)) mpz_sub_ui(out, out, 1);
    while (!z_is_prime(out)) mpz_sub_ui(out, out, 2);
}

/*
 * z_to_u128 - convert an mpz_t below 2^128 to unsigned __int128.
 *
 * Function: exports the two least significant 64-bit words.
 * Inputs:   m with 0 <= m < 2^128 (callers check mpz_sizeinbase(m, 2) first).
 * Outputs:  the value.
 * Errors:   m >= 2^128 would overflow the two-word buffer (memory
 *           corruption), and a negative m would lose its sign; callers must
 *           check the size first.
 */
static u128 z_to_u128(const mpz_t m)
{
    u64 w[2] = { 0, 0 };
    size_t cnt = 0;
    mpz_export(w, &cnt, -1, 8, 0, 0, m);
    return ((u128)w[1] << 64) | w[0];
}

/*
 * zdesc_add - append "p^e" to a description string, with a '*' separator
 *             after the first factor.
 *
 * Function: writes p in decimal and the exponent if e > 1; if the factor would
 *           not fit, appends "..." instead.
 * Inputs:   d, a NUL-terminated string in a buffer of cap bytes; p; e >= 1.
 * Outputs:  d extended in place.
 * Errors:   none; it never writes past cap (mpz_sizeinbase() may overestimate
 *           the length by one, which only makes the check stricter).
 */
static void zdesc_add(char *d, size_t cap, const mpz_t p, unsigned e)
{
    size_t len = strlen(d);
    if (len + 4 >= cap) return;
    if (len) { d[len++] = '*'; d[len] = 0; }
    size_t need = mpz_sizeinbase(p, 10) + 2;
    if (len + need + 16 >= cap) { snprintf(d + len, cap - len, "..."); return; }
    mpz_get_str(d + len, 10, p);
    len = strlen(d);
    if (e > 1) snprintf(d + len, cap - len, "^%u", e);
}

/*
 * z_chain_from - GMP version of chain_from(): is r exactly
 *                p^e1 * nextprime(p)^e2 * ...?
 *
 * Function: divides out p completely, then the next prime, and so on, until
 *           the quotient is 1 (success) or the current prime does not divide
 *           (failure).
 * Inputs:   r >= 1; p, a prime, which is CLOBBERED (it ends at the last prime
 *           used); w (x is clobbered); desc and cap, optional (desc may be
 *           NULL), for the factorization text; nprimes, optional, for the
 *           number of distinct primes.
 * Outputs:  true with desc and *nprimes filled, or false.  After a failure
 *           desc may hold a partial text and must be ignored.
 * Errors:   none.  mpz_nextprime() is probabilistic in the same sense as
 *           z_is_prime().
 */
static bool z_chain_from(const mpz_t r, mpz_t p, zws_t *w, char *desc, size_t cap, int *nprimes)
{
    int cnt = 0;
    mpz_set(w->x, r);
    if (desc) desc[0] = 0;
    for (;;) {
        if (!mpz_divisible_p(w->x, p)) return false;
        unsigned e = 0;
        do { mpz_divexact(w->x, w->x, p); e++; } while (mpz_divisible_p(w->x, p));
        cnt++;
        if (desc) zdesc_add(desc, cap, p, e);
        if (mpz_cmp_ui(w->x, 1) == 0) break;
        mpz_nextprime(p, p);
    }
    if (nprimes) *nprimes = cnt;
    return true;
}

/*
 * z_root_method - GMP version of root_method() for a composite r without
 *                 prime factors below 2^16.
 *
 * Function: for t = 2, 3, ... while c = floor(r^(1/t)) > 2^16: r = c^t with c
 *           prime is gapless; otherwise the t-1 largest primes <= c are tried
 *           as the smallest prime of a consecutive chain (see root_method()
 *           for the proof).
 * Inputs:   r; w (clobbers x, p, q and c); desc and cap, optional.
 * Outputs:  true iff r is gapless (desc filled on success).
 * Errors:   none.  The cost is dominated by the prime searches below r^(1/2)
 *           and r^(1/3): for a 1000-bit r, a few hundred probable-prime tests
 *           on 300- to 500-bit numbers.
 */
static bool z_root_method(const mpz_t r, zws_t *w, char *desc, size_t cap)
{
    for (unsigned long t = 2; t < 100000; t++) {
        int exact = mpz_root(w->c, r, t);
        if (mpz_cmp_ui(w->c, SMALL_LIMIT) <= 0) break;
        if (exact && z_is_prime(w->c)) {
            if (desc) { desc[0] = 0; zdesc_add(desc, cap, w->c, (unsigned)t); }
            return true;
        }
        if (exact || !z_is_prime(w->c)) z_prevprime(w->q, w->c);
        else mpz_set(w->q, w->c);
        for (unsigned long jj = 0; jj < t - 1 && mpz_cmp_ui(w->q, SMALL_LIMIT) > 0; jj++) {
            mpz_set(w->p, w->q);
            if (z_chain_from(r, w->p, w, desc, cap, NULL)) return true;
            if (jj + 1 < t - 1) z_prevprime(w->q, w->q);   /* only when another candidate follows */
        }
    }
    return false;
}

/*
 * z_is_gapfree - gapless test for a number of any size.
 *
 * Function: below 2^127 it calls is_gapless().  Otherwise: trial division by
 *           the primes below 2^16 and z_chain_from() from the smallest one
 *           found; with no small factor, a probable prime is gapless and
 *           anything else goes to z_root_method().
 * Inputs:   m >= 0; w (clobbers x, p, q and c, never tmp, E, O or stk[]);
 *           desc and cap, optional, for a description: the structure from
 *           describe_gapless() or z_chain_from(), "probable prime", or
 *           "gapful" (the last only below 2^127).
 * Outputs:  true iff m is gapless.
 * Errors:   none.  Prime verdicts above 2^64 are probabilistic.  Above 2^127
 *           desc is only meaningful when the result is true.
 */
static bool z_is_gapfree(const mpz_t m, zws_t *w, char *desc, size_t cap)
{
    if (mpz_sizeinbase(m, 2) <= 127) {
        u128 v = z_to_u128(m);
        bool r = is_gapless(v, NULL);
        if (desc) { if (r) describe_gapless(v, desc, cap); else snprintf(desc, cap, "gapful"); }
        return r;
    }
    for (size_t i = 0; i < nsp; i++)
        if (mpz_divisible_ui_p(m, sp[i])) {
            mpz_set_ui(w->p, sp[i]);
            return z_chain_from(m, w->p, w, desc, cap, NULL);
        }
    if (z_is_prime(m)) { if (desc) snprintf(desc, cap, "probable prime"); return true; }
    return z_root_method(m, w, desc, cap);
}

/*
 * z_interior_gapful - is every number strictly between L and R gapful?  (GMP)
 *
 * Function: z_is_gapfree() on L+1, ..., R-1, stopping at the first gapless
 *           one.
 * Inputs:   L < R, neither of them w->tmp (the loop variable); w.
 * Outputs:  true iff none of them is gapless.
 * Errors:   none.  Up to d-1 tests, only reached when the partner is gapless.
 */
static bool z_interior_gapful(const mpz_t L, const mpz_t R, zws_t *w)
{
    for (mpz_add_ui(w->tmp, L, 1); mpz_cmp(w->tmp, R) < 0; mpz_add_ui(w->tmp, w->tmp, 1))
        if (z_is_gapfree(w->tmp, w, NULL, 0)) return false;
    return true;
}

/*
 * ---- GMP search state ----
 *
 * ztask_t: one unit of GMP work.  side 0 (E side): E = 2^a * 3^b (b = 0 for
 *          k = 1); the worker tests E and, when b > 0, extends it by 5, 7, ...
 *          up to kE primes.  side 1 (O side): O starts as p_j^f1 and the
 *          worker multiplies in p_(j+1), ..., p_m.
 * zpair_t: one run found, as mpz values owned by the list.
 * zres_t:  the results of one worker: the pairs v[0..n-1] and the number of
 *          candidates tested (nodes).
 * ZQ:      the GMP search descriptor shared by the workers: limit and
 *          limit + d, d, n, the block (j, m), kmax, kE = min(m, kmax) (the most
 *          primes enumerated on the E side), the task list and its atomic
 *          counters.
 */

typedef struct { int side; unsigned a, b, f1; } ztask_t;
typedef struct { mpz_t L, R; } zpair_t;
typedef struct { zpair_t *v; size_t n, cap; u64 nodes; } zres_t;

static struct {
    mpz_t limit, limitd;          /* limit, limit + d */
    unsigned long d;
    int n, j, m, kmax, kE;
    ztask_t *tasks;
    size_t ntasks;
    atomic_size_t next, done;
} ZQ;

/*
 * z_analyze - the prime structure of d that drives the GMP search.
 *
 * Function: finds j, the index of the smallest odd prime factor of d
 *           (p_1 = 2); m, the end of the maximal block p_j, p_(j+1), ..., p_m
 *           of primes that all divide d; and kmax, the index of the next prime
 *           factor of d after a gap, minus one (INT_MAX if there is none below
 *           2^16).
 * Inputs:   d, odd; pointers for j, m and kmax.
 * Outputs:  *j, *m and *kmax, with kmax > m.
 * Errors:   d without an odd prime factor below 2^16 (d = 1, or a prime d
 *           above 65521, i.e. n >= 65536) -> die().
 *           Example: d = 63 = 3^2*7 gives j = m = 2 (the block {3}), kmax = 3.
 */
static void z_analyze(u64 d, int *j, int *m, int *kmax)
{
    *j = 0; *m = 0; *kmax = INT_MAX;
    for (int k = 2; k <= (int)nsp; k++) {
        if (d % sp[k - 1] == 0) {
            if (*j == 0) { *j = k; *m = k; }
            else if (*m == k - 1) *m = k;
            else { *kmax = k - 1; break; }
        }
    }
    if (*j == 0) die("d = %" PRIu64 " has no odd prime factor below 2^16 (n too large for this mode)", d);
}

/*
 * z_est_E - estimated number of E = p_1^e1 * ... * p_k^ek <= X with every
 *           e_i >= 1.
 *
 * Function: such exponent vectors satisfy sum (e_i - 1)*log p_i <=
 *           ln X - sum log p_i, a k-dimensional simplex of volume
 *           (ln X - sum log p_i)^k / (k! * prod log p_i).
 * Inputs:   k >= 1; lnX = ln X.
 * Outputs:  the estimate; 0 if p_1 * ... * p_k already exceeds X.
 * Errors:   none.  The volume undercounts the lattice points: it is close for
 *           the few primes of the E side, but summed over every k it is about
 *           5 times too small below 2^64.  It only feeds the --max-nodes
 *           guard.
 */
static double z_est_E(int k, double lnX)
{
    double s = lnX, denom = 1;
    for (int i = 0; i < k; i++) { s -= log((double)sp[i]); denom *= log((double)sp[i]) * (i + 1); }
    return s <= 0 ? 0 : pow(s, k) / denom;
}

/*
 * z_est_S - estimated number of products over the block p_j..p_m (every
 *           exponent >= 1) up to X.
 *
 * Function: the simplex volume of z_est_E() with the primes p_j..p_m.
 * Inputs:   j <= m, the block; lnX = ln X.
 * Outputs:  the estimate.
 * Errors:   none.
 */
static double z_est_S(int j, int m, double lnX)
{
    double s = lnX, denom = 1;
    int cnt = 0;
    for (int i = j; i <= m; i++) { s -= log((double)sp[i - 1]); cnt++; denom *= log((double)sp[i - 1]) * cnt; }
    return s <= 0 ? 0 : pow(s, cnt) / denom;
}

/*
 * zres_add - append copies of (L, R) to a GMP result list.
 *
 * Function: doubles the capacity when full (16 entries at first) and stores
 *           new mpz copies.
 * Inputs:   r (zero-initialized before the first use); L; R.
 * Outputs:  the list owns the copies; the caller eventually mpz_clear()s
 *           every entry and frees r->v.
 * Errors:   out of memory -> die() (or a GMP abort inside mpz_init_set()).
 */
static void zres_add(zres_t *r, const mpz_t L, const mpz_t R)
{
    if (r->n == r->cap) {
        r->cap = r->cap ? r->cap * 2 : 16;
        r->v = realloc(r->v, r->cap * sizeof(zpair_t));
        if (!r->v) die("out of memory");
    }
    mpz_init_set(r->v[r->n].L, L);
    mpz_init_set(r->v[r->n].R, R);
    r->n++;
}

/*
 * z_test_E - GMP version of test_E(): try O = E + d and O = E - d.
 *
 * Function: records (E, O) or (O, E) when O is gapless and every number in
 *           between is gapful.
 * Inputs:   E, even and gapless, at most the limit, and neither w->O nor
 *           w->tmp; w (uses O and tmp; z_is_gapfree() clobbers x, p, q, c);
 *           res.
 * Outputs:  pairs appended to res; res->nodes incremented.
 * Errors:   none.
 */
static void z_test_E(const mpz_t E, zws_t *w, zres_t *res)
{
    res->nodes++;
    mpz_add_ui(w->O, E, ZQ.d);
    if (z_is_gapfree(w->O, w, NULL, 0) && z_interior_gapful(E, w->O, w)) zres_add(res, E, w->O);
    if (mpz_cmp_ui(E, ZQ.d) > 0) {
        mpz_sub_ui(w->O, E, ZQ.d);
        if (z_is_gapfree(w->O, w, NULL, 0) && z_interior_gapful(w->O, E, w)) zres_add(res, w->O, E);
    }
}

/*
 * z_dfs - E-side recursion: extend E (k primes) by powers of
 *         p_(k+1) = sp[k].
 *
 * Function: for e = 1, 2, ... while E * p_(k+1)^e <= limit: test the product
 *           and recurse.  It stops at kE primes, because larger k belong to
 *           the O side (header section 4).
 * Inputs:   E (not w->stk[k]); k >= 2; w (uses stk[k]); res.
 * Outputs:  pairs through z_test_E().
 * Errors:   none; the depth stays below kE < ZDEPTH - 2 (checked by
 *           run_search_gmp()).
 */
static void z_dfs(const mpz_t E, int k, zws_t *w, zres_t *res)
{
    if (k >= ZQ.kE || k >= ZDEPTH) return;
    mpz_t *wk = &w->stk[k];
    mpz_set(*wk, E);
    for (;;) {
        mpz_mul_ui(*wk, *wk, sp[k]);
        if (mpz_cmp(*wk, ZQ.limit) > 0) break;
        z_test_E(*wk, w, res);
        z_dfs(*wk, k + 1, w, res);
    }
}

/*
 * z_test_O - O-side test: O is a product over the block {p_j..p_m}; try the
 *            even endpoints E = O + d and E = O - d.
 *
 * Function: E must be at most the limit, even, and gapless with k > m primes
 *           (z_chain_from() from 2 counts them).  Smaller k are covered by the
 *           E side, so no pair is reported twice.  Then the interior must be
 *           gapful.  O itself is gapless by construction.
 * Inputs:   O (neither w->E nor w->tmp); w (uses E, p, x and tmp); res.
 * Outputs:  pairs appended to res; res->nodes incremented.
 * Errors:   none.
 */
static void z_test_O(const mpz_t O, zws_t *w, zres_t *res)
{
    res->nodes++;
    for (int sgn = 0; sgn < 2; sgn++) {
        if (sgn == 0) mpz_add_ui(w->E, O, ZQ.d);
        else { if (mpz_cmp_ui(O, ZQ.d) <= 0) break; mpz_sub_ui(w->E, O, ZQ.d); }
        if (mpz_cmp(w->E, ZQ.limit) > 0 || mpz_cmp_ui(w->E, 2) < 0) continue;
        int k = 0;
        mpz_set_ui(w->p, 2);
        if (!z_chain_from(w->E, w->p, w, NULL, 0, &k)) continue;
        /* k <= m was covered on the E side */
        if (k <= ZQ.m) continue;
        if (sgn == 0) { if (z_interior_gapful(O, w->E, w)) zres_add(res, O, w->E); }
        else          { if (z_interior_gapful(w->E, O, w)) zres_add(res, w->E, O); }
    }
}

/*
 * z_dfs_O - O-side recursion: multiply in p_i, p_(i+1), ..., p_m, each at
 *           least once, and test every complete product.
 *
 * Function: for e = 1, 2, ... while O * p_i^e <= limit + d: recurse with
 *           i + 1; past p_m the product is complete and goes to z_test_O().
 * Inputs:   O, a product over p_j..p_(i-1); i, with j+1 <= i <= m+1; w (uses
 *           stk[i-j+1]); res.
 * Outputs:  z_test_O() on every product over the whole block up to limit + d
 *           (the bound that keeps E = O - d within the limit).
 * Errors:   none; the depth is the block length (at most 5 for n <= 100000).
 */
static void z_dfs_O(const mpz_t O, int i, zws_t *w, zres_t *res)   /* multiply by p_i, i = j+1..m */
{
    if (i > ZQ.m) { z_test_O(O, w, res); return; }
    int depth = i - ZQ.j + 1;
    if (depth >= ZDEPTH) return;
    mpz_t *wk = &w->stk[depth];
    mpz_set(*wk, O);
    for (;;) {
        mpz_mul_ui(*wk, *wk, sp[i - 1]);
        if (mpz_cmp(*wk, ZQ.limitd) > 0) break;
        z_dfs_O(*wk, i + 1, w, res);
    }
}

/*
 * z_worker - thread body of the GMP search.
 *
 * Function: claims tasks through ZQ.next.  An E-side task (a, b) sets
 *           E = 2^a * 3^b, tests it and, when b > 0, extends it with z_dfs().
 *           An O-side task f1 sets O = p_j^f1 and completes it over the block
 *           with z_dfs_O().
 * Inputs:   arg, this thread's zeroed zres_t; the global ZQ.
 * Outputs:  pairs and node counts in the zres_t; ZQ.done per task.
 * Errors:   out of memory -> die() or a GMP abort.
 */
static void *z_worker(void *arg)
{
    zres_t *res = arg;
    zws_t w;
    zws_init(&w);
    for (;;) {
        size_t ti = atomic_fetch_add(&ZQ.next, 1);
        if (ti >= ZQ.ntasks) break;
        ztask_t *t = &ZQ.tasks[ti];
        /* E side: E = 2^a * 3^b, then 5, 7, ... through z_dfs(); O side: O = p_j^f1,
         * completed over the block by z_dfs_O() */
        if (t->side == 0) {
            mpz_ui_pow_ui(w.stk[0], 2, t->a);
            if (t->b) { mpz_ui_pow_ui(w.tmp, 3, t->b); mpz_mul(w.stk[0], w.stk[0], w.tmp); }
            z_test_E(w.stk[0], &w, res);
            if (t->b) z_dfs(w.stk[0], 2, &w, res);
        } else {
            mpz_ui_pow_ui(w.stk[0], sp[ZQ.j - 1], t->f1);
            z_dfs_O(w.stk[0], ZQ.j + 1, &w, res);
        }
        atomic_fetch_add(&ZQ.done, 1);
    }
    zws_clear(&w);
    return NULL;
}

/*
 * cmp_zpair - qsort() comparator for zpair_t by L, increasing.
 *
 * Function: mpz_cmp() of the L fields.
 * Inputs:   pointers to two zpair_t.
 * Outputs:  negative, zero or positive.
 * Errors:   none.
 */
static int cmp_zpair(const void *a, const void *b)
{
    return mpz_cmp(((const zpair_t *)a)->L, ((const zpair_t *)b)->L);
}

/*
 * ztask_add - append a task to a growable ztask_t array.
 *
 * Function: doubles the capacity when full (1024 entries at first).
 * Inputs:   t, n, cap (the array, its length and capacity); side; a, b (E
 *           side); f1 (O side).
 * Outputs:  the task appended; *t, *n and *cap updated.
 * Errors:   out of memory -> die().
 */
static void ztask_add(ztask_t **t, size_t *n, size_t *cap, int side, unsigned a, unsigned b, unsigned f1)
{
    if (*n == *cap) {
        *cap = *cap ? *cap * 2 : 1024;
        *t = realloc(*t, *cap * sizeof(ztask_t));
        if (!*t) die("out of memory");
    }
    (*t)[*n].side = side; (*t)[*n].a = a; (*t)[*n].b = b; (*t)[*n].f1 = f1;
    (*n)++;
}

/*
 * run_search_gmp - the GMP structural search for one even n (header
 *                  section 4).
 *
 * Function: z_analyze(d); the E side covers up to kE = min(m, kmax) primes and
 *           the O side the products over the block.  It estimates the work and
 *           refuses above max_nodes, builds the tasks (2^a and 2^a*3^b for the
 *           E side, p_j^f for the O side), runs the workers, and merges and
 *           sorts the pairs by L.
 * Inputs:   n, even, >= 2 (cmd_search_gmp() checks); limit, of any size;
 *           threads >= 1; max_nodes; verbose; out.
 * Outputs:  the number of pairs found, and *out = the pairs sorted by L (the
 *           caller mpz_clear()s every L and R and frees out->v).  By the
 *           coverage lemma the first pair gives a(n) exactly; no pair means
 *           a(n) > limit - n.
 * Errors:   d without a small odd prime factor, an E side of ZDEPTH - 2 or
 *           more primes, an estimate above max_nodes, out of memory or
 *           pthread_create() failure -> die().  Not reentrant (global ZQ).
 *           Unlike run_search() the pairs are not re-checked here;
 *           cmd_search_gmp() prints a verify_run.py command instead.
 */
static size_t run_search_gmp(int n, const mpz_t limit, int threads, double max_nodes, bool verbose, zres_t *out)
{
    unsigned long d = (unsigned long)n + 1;
    int j, m, kmax;
    z_analyze(d, &j, &m, &kmax);
    mpz_init_set(ZQ.limit, limit);
    mpz_init(ZQ.limitd);
    mpz_add_ui(ZQ.limitd, limit, d);
    ZQ.d = d; ZQ.n = n; ZQ.j = j; ZQ.m = m; ZQ.kmax = kmax;
    ZQ.kE = m < kmax ? m : kmax;
    if (ZQ.kE >= ZDEPTH - 2) die("too many primes in the even endpoint (kE = %d)", ZQ.kE);

    /* estimate the work first (ln X rounded up to whole bits) */
    double lnX = mpz_sizeinbase(limit, 2) * log(2.0);
    double estE = 0;
    for (int k = 1; k <= ZQ.kE; k++) estE += z_est_E(k, lnX);
    double estO = z_est_S(j, m, lnX);
    if (verbose) {
        char *ls = mpz_get_str(NULL, 10, limit);
        size_t bits = mpz_sizeinbase(limit, 2);
        fprintf(stderr, "[gmp] n = %d, d = %lu, limit = %s%s (%zu bits), %d threads\n", n, d,
                strlen(ls) > 40 ? "" : ls, strlen(ls) > 40 ? "(long)" : "", bits, threads);
        free(ls);
        fprintf(stderr, "[gmp] smallest odd prime of d: p_%d = %u; block p_%d..p_%d; kmax = ", j, sp[j - 1], j, m);
        if (kmax == INT_MAX) fprintf(stderr, "unbounded\n"); else fprintf(stderr, "%d\n", kmax);
        fprintf(stderr, "[gmp] E side: k <= %d, ~%.3g candidates; O side: products of {", ZQ.kE, estE);
        for (int i = j; i <= m; i++) fprintf(stderr, "%s%u", i > j ? "," : "", sp[i - 1]);
        fprintf(stderr, "} for k > %d, ~%.3g candidates\n", m, estO);
    }
    if (estE + estO > max_nodes)
        die("estimated %.3g candidates exceeds --max-nodes %.3g; lower the limit (kE = %d primes on the E side)",
            estE + estO, max_nodes, ZQ.kE);

    /* tasks */
    ztask_t *tasks = NULL;
    size_t ntasks = 0, cap = 0;
    mpz_t t1, t2;
    mpz_inits(t1, t2, NULL);
    /* E-side tasks 2^a and 2^a*3^b, then O-side tasks p_j^f */
    for (unsigned a = 1;; a++) {
        mpz_ui_pow_ui(t1, 2, a);
        if (mpz_cmp(t1, limit) > 0) break;
        ztask_add(&tasks, &ntasks, &cap, 0, a, 0, 0);
        if (ZQ.kE >= 2) {
            for (unsigned b = 1;; b++) {
                mpz_ui_pow_ui(t2, 3, b);
                mpz_mul(t2, t2, t1);
                if (mpz_cmp(t2, limit) > 0) break;
                ztask_add(&tasks, &ntasks, &cap, 0, a, b, 0);
            }
        }
    }
    for (unsigned f = 1;; f++) {
        mpz_ui_pow_ui(t1, sp[j - 1], f);
        if (mpz_cmp(t1, ZQ.limitd) > 0) break;
        ztask_add(&tasks, &ntasks, &cap, 1, 0, 0, f);
    }
    mpz_clears(t1, t2, NULL);
    ZQ.tasks = tasks; ZQ.ntasks = ntasks;
    atomic_store(&ZQ.next, 0);
    atomic_store(&ZQ.done, 0);
    if (verbose) fprintf(stderr, "[gmp] %zu tasks\n", ntasks);

    zres_t *res = xcalloc((size_t)threads, sizeof *res);
    pthread_t *tid = xmalloc((size_t)threads * sizeof *tid);
    double t0 = now();
    for (int i = 0; i < threads; i++)
        if (pthread_create(&tid[i], NULL, z_worker, &res[i])) die("pthread_create failed");
    if (verbose) {
        double last = 0;
        for (;;) {
            size_t done = atomic_load(&ZQ.done);
            if (done >= ntasks) break;
            double el = now() - t0;
            if (el > 2.0 && (stderr_tty || el - last >= 15.0)) {
                fprintf(stderr, "%s[gmp] %zu/%zu tasks  %.0fs   %s", stderr_tty ? "\r" : "", done, ntasks, el,
                        stderr_tty ? "" : "\n");
                last = el;
            }
            usleep(200000);
        }
    }
    for (int i = 0; i < threads; i++) pthread_join(tid[i], NULL);

    zres_t all = { NULL, 0, 0, 0 };
    for (int i = 0; i < threads; i++) {
        all.nodes += res[i].nodes;
        for (size_t k = 0; k < res[i].n; k++) {
            zres_add(&all, res[i].v[k].L, res[i].v[k].R);
            mpz_clears(res[i].v[k].L, res[i].v[k].R, NULL);
        }
        free(res[i].v);
    }
    free(res); free(tid); free(tasks);
    qsort(all.v, all.n, sizeof(zpair_t), cmp_zpair);
    if (verbose)
        fprintf(stderr, "%s[gmp] %" PRIu64 " candidates tested in %.2fs, %zu run%s of exactly %d found\n",
                stderr_tty ? "\r" : "", all.nodes, now() - t0, all.n, all.n == 1 ? "" : "s", n);
    mpz_clears(ZQ.limit, ZQ.limitd, NULL);
    *out = all;
    return all.n;
}

/*
 * cmd_search_gmp - the `search n` command in GMP mode (LIMIT >= 2^127, or
 *                  --gmp).
 *
 * Function: validates the arguments, runs run_search_gmp(), prints up to 20
 *           runs with the structure of their bounds, then a(n) with its digit
 *           count and the verify_run.py command for an independent check, or
 *           the bound a(n) > limit - n.
 * Inputs:   n; limit, at least n + 1; threads; max_nodes.
 * Outputs:  the report on stdout (flushed); progress on stderr.
 * Errors:   n < 1, odd n or limit < n + 1 -> die(), plus the errors of
 *           run_search_gmp().
 */
static void cmd_search_gmp(int n, const mpz_t limit, int threads, double max_nodes)
{
    if (n < 1) die("n must be >= 1");
    if (n % 2) die("search handles even n only (odd n: the two bounding gapless numbers "
                   "have the same parity, so use 'scan')");
    if (mpz_cmp_ui(limit, (unsigned long)n + 1) < 0) die("limit must be at least n+1 = %d", n + 1);
    zres_t all;
    run_search_gmp(n, limit, threads, max_nodes, true, &all);
    unsigned long d = (unsigned long)n + 1;
    zws_t w;
    zws_init(&w);
    mpz_t lim_d;
    mpz_init(lim_d);
    mpz_sub_ui(lim_d, limit, d - 1);          /* limit - d + 1 */
    char *s1 = mpz_get_str(NULL, 10, lim_d);
    printf("\n=== A137723 search (GMP): n = %d (d = %lu), even endpoint <= 2^%zu-ish, %zu bits ===\n",
           n, d, mpz_sizeinbase(limit, 2) - 1, mpz_sizeinbase(limit, 2));
    if (all.n == 0) {
        printf("No run of exactly %d gapful numbers starts at or below %s.\n", n, s1);
        printf("a(%d) > %s\n", n, s1);
    } else {
        char dL[512], dR[512];
        size_t show = all.n < 20 ? all.n : 20;
        printf("Runs of exactly %d gapful numbers (bounded by gapless L and R = L+%lu):\n", n, d);
        for (size_t i = 0; i < show; i++) {
            z_is_gapfree(all.v[i].L, &w, dL, sizeof dL);
            z_is_gapfree(all.v[i].R, &w, dR, sizeof dR);
            char *sL = mpz_get_str(NULL, 10, all.v[i].L), *sR = mpz_get_str(NULL, 10, all.v[i].R);
            mpz_add_ui(w.tmp, all.v[i].L, 1);
            char *sA = mpz_get_str(NULL, 10, w.tmp);
            printf("  start %s\n    L = %s (%s)\n    R = %s (%s)\n", sA, sL, dL, sR, dR);
            free(sL); free(sR); free(sA);
        }
        if (show < all.n) printf("  ... %zu more\n", all.n - show);
        mpz_add_ui(w.tmp, all.v[0].L, 1);
        char *sA = mpz_get_str(NULL, 10, w.tmp);
        printf("\na(%d) = %s\n", n, sA);
        printf("  = %zu digits; exhaustive: every run of exactly %d starting <= %s was enumerated.\n",
               strlen(sA), n, s1);
        char *sL = mpz_get_str(NULL, 10, all.v[0].L), *sR = mpz_get_str(NULL, 10, all.v[0].R);
        printf("  Primality above 2^64 is probabilistic here; certify and cross-check with:\n"
               "    python3 verify_run.py %s %s\n", sL, sR);
        free(sA); free(sL); free(sR);
        for (size_t i = 0; i < all.n; i++) mpz_clears(all.v[i].L, all.v[i].R, NULL);
    }
    free(all.v);
    free(s1);
    mpz_clear(lim_d);
    zws_clear(&w);
    fflush(stdout);
}

/*
 * parse_mpz - parse a numeric argument of any size (GMP builds).
 *
 * Function: the syntax of parse_u128(): decimal, B^E (B <= 10^6, E <= 200000),
 *           MeE (E <= 100000), and sums and differences split at the LAST '+'
 *           or '-' (left-associative).
 * Inputs:   s, NUL-terminated; out, an initialized mpz_t.
 * Outputs:  true with out set; false otherwise (out may have been changed).
 * Errors:   false for empty or malformed text, a leading sign, negative
 *           differences, and left operands of 256 characters or more.
 */
static bool parse_mpz(const char *s, mpz_t out)
{
    const char *pm = NULL;                       /* the LAST '+' or '-', not at s[0] */
    for (const char *c = s; *c; c++)
        if ((*c == '+' || *c == '-') && c != s) pm = c;
    if (pm) {
        char head[256];
        size_t hl = (size_t)(pm - s);
        if (hl >= sizeof head) return false;
        memcpy(head, s, hl); head[hl] = 0;
        mpz_t a, b;
        mpz_inits(a, b, NULL);
        bool ok = parse_mpz(head, a) && parse_mpz(pm + 1, b);
        if (ok) {
            if (*pm == '+') mpz_add(out, a, b);
            else { if (mpz_cmp(a, b) < 0) ok = false; else mpz_sub(out, a, b); }
        }
        mpz_clears(a, b, NULL);
        return ok;
    }
    const char *caret = strchr(s, '^');
    const char *ee = strpbrk(s, "eE");
    if (caret) {
        u128 b, e;
        if (!parse_dec(s, (size_t)(caret - s), &b) || b > 1000000) return false;
        if (!parse_dec(caret + 1, strlen(caret + 1), &e) || e > 200000) return false;
        mpz_ui_pow_ui(out, (unsigned long)b, (unsigned long)e);
        return true;
    }
    if (ee) {
        u128 e;
        char head[256];
        size_t hl = (size_t)(ee - s);
        if (hl >= sizeof head) return false;
        memcpy(head, s, hl); head[hl] = 0;
        if (mpz_set_str(out, head, 10) != 0) return false;
        if (!parse_dec(ee + 1, strlen(ee + 1), &e) || e > 100000) return false;
        mpz_t p10;
        mpz_init(p10);
        mpz_ui_pow_ui(p10, 10, (unsigned long)e);
        mpz_mul(out, out, p10);
        mpz_clear(p10);
        return true;
    }
    for (const char *c = s; *c; c++) if (*c < '0' || *c > '9') return false;
    return *s && mpz_set_str(out, s, 10) == 0;
}

/*
 * z_partial_line - one display line for a big number, without a full
 *                  factorization.
 *
 * Function: divides out the primes below 2^16 (printed with gap markers like
 *           fmt_factor()) and shows the rest as a probable prime, as its
 *           gapless structure, or as "[C...: composite with a prime gap]".
 *           Whether the last small prime and the cofactor are consecutive is
 *           decided exactly: the next prime after it must divide the cofactor,
 *           which only needs a test when that prime is 65537.
 * Inputs:   m; w (x and E are clobbered, plus whatever z_is_gapfree() uses);
 *           buf and its size cap; exact.
 * Outputs:  buf; returns the number of prime gaps found.  *exact is false when
 *           that number is only a lower bound, which happens when the cofactor
 *           is composite with a gap (it may contain more than one).
 * Errors:   none; the text is truncated to cap.
 */
static int z_partial_line(const mpz_t m, zws_t *w, char *buf, size_t cap, bool *exact)
{
    size_t pos = 0;
    unsigned long prev = 0;
    int gaps = 0;
    bool any = false;
    *exact = true;
    buf[0] = 0;
    mpz_set(w->x, m);
    for (size_t i = 0; i < nsp && pos < cap; i++) {
        if (!mpz_divisible_ui_p(w->x, sp[i])) continue;
        unsigned e = 0;
        do { mpz_divexact_ui(w->x, w->x, sp[i]); e++; } while (mpz_divisible_ui_p(w->x, sp[i]));
        if (any) {
            bool gap = next_prime128(prev) != sp[i];
            if (gap) gaps++;
            pos += (size_t)snprintf(buf + pos, cap - pos, gap ? " | " : "*");
        }
        if (pos < cap) pos += (size_t)snprintf(buf + pos, cap - pos, "%u", sp[i]);
        if (e > 1 && pos < cap) pos += (size_t)snprintf(buf + pos, cap - pos, "^%u", e);
        prev = sp[i];
        any = true;
        if (mpz_cmp_ui(w->x, 1) == 0) break;
    }
    if (mpz_cmp_ui(w->x, 1) > 0 && pos < cap) {
        /* every prime of the cofactor exceeds SMALL_LIMIT */
        if (any) {
            /* gap-free here iff the prime after prev divides the cofactor; a
             * prime below 2^16 was already tried, so only 65537 needs a test */
            u128 q = next_prime128(prev);
            bool joined = q >= SMALL_LIMIT && mpz_divisible_ui_p(w->x, (unsigned long)q);
            if (!joined) gaps++;
            pos += (size_t)snprintf(buf + pos, cap - pos, joined ? "*" : " | ");
        }
        mpz_set(w->E, w->x);                       /* z_is_gapfree clobbers w->x */
        char d2[600];
        char *cs = mpz_get_str(NULL, 10, w->E);
        if (z_is_prime(w->E)) {
            if (pos < cap) snprintf(buf + pos, cap - pos, "%s (probable prime)", cs);
        } else if (z_is_gapfree(w->E, w, d2, sizeof d2)) {
            if (pos < cap) snprintf(buf + pos, cap - pos, "%s", d2);
        } else {
            gaps++;                                /* at least one gap inside the cofactor */
            *exact = false;
            if (pos < cap) snprintf(buf + pos, cap - pos, "[C%s: composite with a prime gap]", cs);
        }
        free(cs);
    }
    return gaps;
}

/*
 * cmd_check_gmp - the `check m` command for m >= 2^127 (GMP builds).
 *
 * Function: prints z_partial_line(), the A073490 count (exact or a lower
 *           bound), the verdict of z_is_gapfree() with the structure, and a
 *           warning if the two disagree.
 * Inputs:   m.
 * Outputs:  text on stdout.
 * Errors:   none.  Prime verdicts are probabilistic.
 */
static void cmd_check_gmp(const mpz_t m)
{
    zws_t w;
    zws_init(&w);
    char line[4096], desc[600];
    bool exact;
    int gaps = z_partial_line(m, &w, line, sizeof line, &exact);
    char *ms = mpz_get_str(NULL, 10, m);
    printf("%s = %s\n", ms, line);
    bool gf = z_is_gapfree(m, &w, desc, sizeof desc);
    printf("A073490 %s %d  ->  %s\n", exact ? "=" : ">=", gaps,
           gf ? "no prime gap (gapless, in A073491)" : "has a prime gap (gapful)");
    if (gf) printf("structure: %s\n", desc);
    if (gf == (gaps > 0)) printf("!! z_is_gapfree() disagrees with the display factorization\n");
    free(ms);
    zws_clear(&w);
}

/*
 * z_print_run - GMP version of print_run().
 *
 * Function: prints the bounds of the run and their structure and, for runs of
 *           up to 300 numbers, one z_partial_line() per interior number.
 * Inputs:   L < R, consecutive gapless numbers; w.
 * Outputs:  text on stdout.
 * Errors:   none.
 */
static void z_print_run(const mpz_t L, const mpz_t R, zws_t *w)
{
    char dL[600], dR[600], line[4096];
    bool exact;
    mpz_t n, m;
    mpz_inits(n, m, NULL);
    mpz_sub(n, R, L);
    mpz_sub_ui(n, n, 1);
    z_is_gapfree(L, w, dL, sizeof dL);
    z_is_gapfree(R, w, dR, sizeof dR);
    char *sn = mpz_get_str(NULL, 10, n), *sL = mpz_get_str(NULL, 10, L), *sR = mpz_get_str(NULL, 10, R);
    mpz_add_ui(m, L, 1);
    char *sA = mpz_get_str(NULL, 10, m);
    printf("Run of %s gapful numbers starting at %s\n", sn, sA);
    printf("  bounded below by %s (%s)\n  and above by %s (%s)\n", sL, dL, sR, dR);
    if (mpz_cmp_ui(n, 300) <= 0) {
        for (mpz_add_ui(m, L, 1); mpz_cmp(m, R) < 0; mpz_add_ui(m, m, 1)) {
            char *s = mpz_get_str(NULL, 10, m);
            z_partial_line(m, w, line, sizeof line, &exact);
            printf("  %s = %s\n", s, line);
            free(s);
        }
    }
    if (mpz_cmp_ui(n, 1) >= 0 && mpz_fits_ulong_p(n))
        printf("  -> candidate for a(%s) = %s\n", sn, sA);
    free(sn); free(sL); free(sR); free(sA);
    mpz_clears(n, m, NULL);
}

/*
 * cmd_run_gmp - the `run m` command for m >= 2^127 (GMP builds).
 *
 * Function: like cmd_run(): steps down and up from m to the nearest gapless
 *           numbers with z_is_gapfree() and prints the run, or both runs if m
 *           itself is gapless.
 * Inputs:   m.
 * Outputs:  text on stdout.
 * Errors:   none.  Each step can need a root-method test on a big composite,
 *           so at thousands of bits this takes seconds.
 */
static void cmd_run_gmp(const mpz_t m0)
{
    zws_t w;
    zws_init(&w);
    mpz_t L, R;
    mpz_inits(L, R, NULL);
    char desc[600];
    if (z_is_gapfree(m0, &w, desc, sizeof desc)) {
        char *s = mpz_get_str(NULL, 10, m0);
        printf("%s is gapless (%s); showing the runs on both sides.\n\n", s, desc);
        free(s);
        mpz_sub_ui(L, m0, 1);
        while (!z_is_gapfree(L, &w, NULL, 0)) mpz_sub_ui(L, L, 1);
        z_print_run(L, m0, &w);
        printf("\n");
        mpz_add_ui(R, m0, 1);
        while (!z_is_gapfree(R, &w, NULL, 0)) mpz_add_ui(R, R, 1);
        z_print_run(m0, R, &w);
    } else {
        mpz_sub_ui(L, m0, 1);
        while (!z_is_gapfree(L, &w, NULL, 0)) mpz_sub_ui(L, L, 1);
        mpz_add_ui(R, m0, 1);
        while (!z_is_gapfree(R, &w, NULL, 0)) mpz_add_ui(R, R, 1);
        z_print_run(L, R, &w);
    }
    mpz_clears(L, R, NULL);
    zws_clear(&w);
}
#endif /* HAVE_GMP */

/* ------------------------------------------------------------------ */
/* selftest                                                            */
/* ------------------------------------------------------------------ */
/*
 * Internal consistency tests (header section 10).
 */

/*
 * tfail: the number of failed expectations in the running self-test.
 */
static int tfail;

/*
 * expect - record one self-test expectation.
 *
 * Function: on failure increments tfail and prints "  FAIL: <message>".
 * Inputs:   cond; a printf-style message, only formatted on failure.
 * Outputs:  nothing on success.
 * Errors:   none.
 */
static void expect(bool cond, const char *fmt, ...)
{
    va_list ap;
    if (cond) return;
    tfail++;
    va_start(ap, fmt);
    fputs("  FAIL: ", stdout);
    vprintf(fmt, ap);
    va_end(ap);
    fputc('\n', stdout);
}

/*
 * brute_gapful - reference prime-gap indicator for 2..M, by plain
 *                factorization.
 *
 * Function: a smallest-prime-factor sieve, then for each m its distinct primes
 *           in order, each compared with the prime after the previous one.
 * Inputs:   M (the self-test uses 2*10^6).
 * Outputs:  g[0..M], freed by the caller: g[m] = 1 iff m >= 2 has a prime gap
 *           (g[0] = g[1] = 0).
 * Errors:   out of memory -> die().  Memory 5*(M+1) bytes.
 */
static uint8_t *brute_gapful(u64 M)
{
    u32 *spf = xcalloc(M + 1, sizeof(u32));
    for (u64 i = 2; i <= M; i++)
        if (!spf[i])
            for (u64 j = i; j <= M; j += i)
                if (!spf[j]) spf[j] = (u32)i;
    uint8_t *g = xcalloc(M + 1, 1);
    for (u64 m = 2; m <= M; m++) {
        u64 x = m;
        u32 prev = 0;
        bool gap = false;
        while (x > 1) {
            u32 p = spf[x];
            if (prev) {
                u64 q = prev + 1;
                while (spf[q] != q) q++;          /* next prime after prev */
                if (q != p) gap = true;
            }
            prev = p;
            while (x % p == 0) x /= p;
        }
        g[m] = gap;
    }
    free(spf);
    return g;
}

/*
 * cmd_selftest - the `selftest` command: groups [1]..[7] of header
 *                section 10.
 *
 * Function: runs every group and prints each failed expectation.
 * Inputs:   threads, for the scans and searches it runs.
 * Outputs:  progress and failures on stdout.  Never returns: exits with 0 if
 *           everything passed, 1 otherwise.
 * Errors:   resource failures -> die().
 */
static void cmd_selftest(int threads)
{
    double t0 = now();
    printf("A137723 selftest\n");

    /* 1. primality */
    printf("[1] primality tests\n");
    for (u64 x = 0; x < SMALL_LIMIT; x++)
        expect(is_prime64(x) == (x < SMALL_LIMIT && small_isp[x]), "is_prime64(%" PRIu64 ")", x);
    expect(is_prime64(((u64)1 << 61) - 1), "2^61-1 prime");
    expect(is_prime64(18446744073709551557ULL), "2^64-59 prime");
    expect(!is_prime64(3215031751ULL), "3215031751 composite");
    expect(!is_prime64(3825123056546413051ULL), "3825123056546413051 composite (spsp to bases 2..23)");
    expect(!is_prime64(UINT64_MAX), "2^64-1 composite");
    expect(is_prime128(((u128)1 << 89) - 1), "2^89-1 prime");
    expect(is_prime128(((u128)1 << 107) - 1), "2^107-1 prime");
    expect(is_prime128(((u128)1 << 127) - 1), "2^127-1 prime");
    expect(!is_prime128(((u128)1 << 67) - 1), "2^67-1 composite");
    expect(!is_prime128(((u128)1 << 64) + 1), "2^64+1 composite");
    {
        u128 psp;
        parse_u128("318665857834031151167461", &psp);      /* spsp to the first 12 prime bases */
        expect(!is_prime128(psp), "318665857834031151167461 composite");
        parse_u128("3317044064679887385961981", &psp);     /* spsp to the first 13 prime bases */
        expect(!is_prime128(psp), "3317044064679887385961981 composite");
    }
    expect(next_prime128(2) == 3 && next_prime128(65521) == 65537 && prev_prime128(65537) == 65521,
           "next/prev prime around 2^16");
    expect(next_prime128(((u128)1 << 64)) == ((u128)1 << 64) + 13, "nextprime(2^64) = 2^64+13");
    expect(prev_prime128(((u128)1 << 64)) == ((u128)1 << 64) - 59, "prevprime(2^64) = 2^64-59");

    /* 2. gapless test vs brute force */
    u64 M = 2000000;
    printf("[2] is_gapless vs brute force for 1..%" PRIu64 "\n", M);
    uint8_t *bg = brute_gapful(M);
    for (u64 m = 1; m <= M; m++)
        expect(is_gapless(m, NULL) == !bg[m], "is_gapless(%" PRIu64 ")", m);
    {
        gl_t g;
        expect(is_gapless(2940, &g) && g.nf == 4 && g.f[3].p == 7 && g.f[3].e == 2, "2940 = 2^2*3*5*7^2 structure");
        expect(!is_gapless(84, NULL) && !is_gapless(10, NULL) && is_gapless(35, NULL) && is_gapless(1, NULL), "small cases");
    }

    /* 3. root method: consecutive-prime products with large primes */
    printf("[3] gapless test on large consecutive-prime products\n");
    {
        u128 p = prev_prime128((u128)1 << 32), q = next_prime128(p), r = next_prime128(q), s = next_prime128(r);
        expect(is_gapless(p * q, NULL), "p*q (p~2^32)");
        expect(!is_gapless(p * r, NULL), "p*r skips q");
        expect(is_gapless(p * p, NULL), "p^2");
        expect(is_gapless(p * p * q, NULL), "p^2*q");
        expect(is_gapless(p * q * q, NULL), "p*q^2");
        expect(is_gapless(p * q * r, NULL), "p*q*r");
        expect(!is_gapless(p * q * s, NULL), "p*q*s skips r");
        expect(is_gapless(p * p * p, NULL), "p^3");
        expect(!is_gapless(p * p * r, NULL), "p^2*r skips q");
        u128 P = prev_prime128((u128)1 << 62), Q = next_prime128(P), R = next_prime128(Q);
        expect(is_gapless(P * Q, NULL), "P*Q (P~2^62)");
        expect(!is_gapless(P * R, NULL), "P*R skips Q");
        expect(is_gapless(P * P, NULL), "P^2 (~2^124)");
        u128 a = prev_prime128((u128)1 << 41), b = next_prime128(a), c = next_prime128(b);
        expect(is_gapless(a * b * c, NULL), "a*b*c (~2^123)");
        expect(is_gapless(a * a * b, NULL), "a^2*b (~2^123)");
        expect(!is_gapless(a * b * next_prime128(c), NULL), "a*b*d skips c");
        char fb[300];
        expect(fmt_factor(p * q * s, fb, sizeof fb) == 1, "fmt_factor gap count p*q*s: %s", fb);
        expect(fmt_factor(2940, fb, sizeof fb) == 0 && strcmp(fb, "2^2*3*5*7^2") == 0, "fmt_factor(2940) = %s", fb);
        expect(fmt_factor(110, fb, sizeof fb) == 2 && strcmp(fb, "2 | 5 | 11") == 0, "fmt_factor(110) = %s", fb);
    }

    /* 4. scan vs brute force and OEIS */
    u64 S = M - 5000;
    printf("[4] scan to %" PRIu64 " vs brute force and OEIS\n", S);
    {
        static u64 first[MAXRUN];
        static u64 bfirst[MAXRUN];
        int maxrun = 0, bmax = 0;
        u64 g = 1;
        memset(bfirst, 0, sizeof bfirst);
        for (u64 m = 2; m <= M; m++) {
            if (bg[m]) continue;
            if (g <= S) {
                u64 n = m - g - 1;
                if (n) { if (!bfirst[n]) bfirst[n] = g + 1; if ((int)n > bmax) bmax = (int)n; }
            }
            g = m;
        }
        run_scan(S, threads, first, &maxrun, false);
        expect(maxrun == bmax, "scan maxrun %d vs brute %d", maxrun, bmax);
        for (int n = 1; n < MAXRUN; n++)
            expect(first[n] == bfirst[n], "scan first[%d] = %" PRIu64 " vs brute %" PRIu64, n, first[n], bfirst[n]);
        for (int n = 1; n <= NKNOWN; n++)
            expect(first[n] == KNOWN[n], "a(%d): scan %" PRIu64 " vs OEIS %" PRIu64, n, first[n], KNOWN[n]);
        expect(first[32] == 0, "a(32) not below %" PRIu64, S);
    }
    free(bg);

    /* 5. structural search reproduces every even OEIS term */
    printf("[5] structural search for even n <= 30 with limit 2^20\n");
    for (int n = 2; n <= NKNOWN; n += 2) {
        u128 a = run_search(n, (u128)1 << 20, threads, false, NULL);
        char b[48];
        expect(a == KNOWN[n], "search a(%d) = %s vs OEIS %" PRIu64, n, u128_str(a, b), KNOWN[n]);
    }
    {
        expect(compute_kmax(33) == 4 && compute_kprime(33) == 1, "kmax(33)=4, kprime(33)=1");
        expect(compute_kmax(35) == INT_MAX && compute_kprime(35) == 2, "kmax(35)=inf, kprime(35)=2");
        expect(compute_kmax(21) == 3 && compute_kmax(15) == INT_MAX && compute_kmax(55) == 4, "kmax 21/15/55");
        expect(compute_kmax(37) == INT_MAX && compute_kprime(37) == 11, "kmax(37)=inf, kprime(37)=11");
    }

    /* 6. the structural search must agree with the exhaustive scan for every even n */
    {
        static u64 first[MAXRUN];
        int maxrun = 0, checked = 0;
        u64 T = 10000000;
        printf("[6] structural search vs exhaustive scan for every even n up to %" PRIu64 "\n", T);
        run_scan(T, threads, first, &maxrun, false);
        for (int n = 2; n <= maxrun; n += 2) {
            u64 d = (u64)n + 1;
            u128 a = run_search(n, T, threads, false, NULL);
            char b[48];
            if (first[n] && first[n] - 1 <= T - d)
                expect(a == first[n], "n=%d: search %s vs scan %" PRIu64, n, u128_str(a, b), first[n]);
            else
                expect(a == 0 || a == first[n], "n=%d: search %s but scan %" PRIu64, n, u128_str(a, b), first[n]);
            checked++;
        }
        printf("    %d even values of n compared (longest run below %" PRIu64 ": %d)\n", checked, T, maxrun);
    }

#ifdef HAVE_GMP
    /* 7. GMP mode: big-number gap-free test, and the E-side/O-side enumeration
     *    must agree with the 128-bit search and with the exhaustive scan       */
    printf("[7] GMP mode\n");
    {
        zws_t w;
        zws_init(&w);
        mpz_t P, Q, R2, X, Y;
        mpz_inits(P, Q, R2, X, Y, NULL);
        mpz_ui_pow_ui(X, 2, 521); mpz_sub_ui(X, X, 1);
        expect(z_is_gapfree(X, &w, NULL, 0), "2^521-1 (Mersenne prime) gap-free");
        mpz_ui_pow_ui(X, 2, 500); mpz_sub_ui(X, X, 1);
        expect(!z_is_gapfree(X, &w, NULL, 0), "2^500-1 = 3*5*... gapful");
        mpz_ui_pow_ui(X, 3, 100);
        expect(z_is_gapfree(X, &w, NULL, 0), "3^100 gap-free");
        mpz_ui_pow_ui(X, 2, 100); mpz_ui_pow_ui(Y, 3, 50); mpz_mul(X, X, Y); mpz_ui_pow_ui(Y, 5, 20); mpz_mul(X, X, Y);
        expect(z_is_gapfree(X, &w, NULL, 0), "2^100*3^50*5^20 gap-free");
        mpz_mul_ui(Y, X, 11);
        expect(!z_is_gapfree(Y, &w, NULL, 0), "2^100*3^50*5^20*11 gapful");
        mpz_ui_pow_ui(P, 2, 70); mpz_nextprime(P, P); mpz_nextprime(Q, P); mpz_nextprime(R2, Q);
        mpz_mul(X, P, Q);
        expect(z_is_gapfree(X, &w, NULL, 0), "P*Q with P ~ 2^70 gap-free (GMP root method, t=2)");
        mpz_mul(X, P, R2);
        expect(!z_is_gapfree(X, &w, NULL, 0), "P*R skips Q");
        mpz_mul(X, P, Q); mpz_mul(X, X, R2);
        expect(z_is_gapfree(X, &w, NULL, 0), "P*Q*R gap-free (t=3)");
        mpz_mul(X, P, P); mpz_mul(X, X, Q);
        expect(z_is_gapfree(X, &w, NULL, 0), "P^2*Q gap-free");
        mpz_mul(X, P, P); mpz_mul(X, X, R2);
        expect(!z_is_gapfree(X, &w, NULL, 0), "P^2*R skips Q");
        mpz_mul(X, P, P); mpz_mul(X, X, P);
        expect(z_is_gapfree(X, &w, NULL, 0), "P^3 gap-free");
        char desc[256];
        mpz_ui_pow_ui(X, 2, 200); mpz_mul_ui(X, X, 9);
        expect(z_is_gapfree(X, &w, desc, sizeof desc) && strcmp(desc, "2^200*3^2") == 0, "describe 2^200*3^2: %s", desc);
        mpz_clears(P, Q, R2, X, Y, NULL);
        zws_clear(&w);

        mpz_t lim;
        mpz_init(lim);
        for (int n = 2; n <= NKNOWN; n += 2) {
            zres_t all;
            mpz_ui_pow_ui(lim, 2, 20);
            size_t cnt = run_search_gmp(n, lim, threads, 1e12, false, &all);
            bool ok = cnt > 0 && mpz_cmp_ui(all.v[0].L, (unsigned long)KNOWN[n] - 1) == 0;
            expect(ok, "GMP search a(%d) vs OEIS %" PRIu64, n, KNOWN[n]);
            for (size_t i = 0; i < all.n; i++) mpz_clears(all.v[i].L, all.v[i].R, NULL);
            free(all.v);
        }
        {
            zres_t all;
            mpz_ui_pow_ui(lim, 2, 64);
            size_t cnt = run_search_gmp(32, lim, threads, 1e12, false, &all);
            bool ok = cnt == 2 && mpz_cmp_ui(all.v[0].L, 18014398509481951ULL) == 0;
            expect(ok, "GMP search a(32) to 2^64");
            for (size_t i = 0; i < all.n; i++) mpz_clears(all.v[i].L, all.v[i].R, NULL);
            free(all.v);
        }
        {
            static u64 first[MAXRUN];
            int maxrun = 0, checked = 0;
            u64 T = 10000000;
            run_scan(T, threads, first, &maxrun, false);
            mpz_set_ui(lim, T);
            for (int n = 2; n <= maxrun; n += 2) {
                u64 d = (u64)n + 1;
                zres_t all;
                run_search_gmp(n, lim, threads, 1e12, false, &all);
                u64 a = all.n ? mpz_get_ui(all.v[0].L) + 1 : 0;
                if (first[n] && first[n] - 1 <= T - d)
                    expect(a == first[n], "GMP n=%d: search %" PRIu64 " vs scan %" PRIu64, n, a, first[n]);
                else
                    expect(a == 0 || a == first[n], "GMP n=%d: search %" PRIu64 " but scan %" PRIu64, n, a, first[n]);
                for (size_t i = 0; i < all.n; i++) mpz_clears(all.v[i].L, all.v[i].R, NULL);
                free(all.v);
                checked++;
            }
            printf("    %d even values of n compared with the exhaustive scan\n", checked);
        }
        mpz_clear(lim);
    }
#endif

    printf("%s (%d failure%s, %.1fs)\n", tfail ? "SELFTEST FAILED" : "SELFTEST PASSED", tfail, tfail == 1 ? "" : "s", now() - t0);
    exit(tfail ? 1 : 0);
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */
/*
 * Option parsing and command dispatch.
 */

/*
 * usage - print the command summary to stderr and exit with status 2.
 *
 * Function: called for a missing or unknown command, a wrong number of
 *           arguments, or an unknown option.
 * Inputs:   none.
 * Outputs:  never returns.
 * Errors:   none.
 */
static void usage(void)
{
    fprintf(stderr,
        "usage:\n"
        "  a137723 scan N [-t T] [-m MAXN] [-o FILE]   exhaustive: every a(n) <= N\n"
        "  a137723 search n [n ...] [-l LIMIT] [-t T]  structural search for even n (default LIMIT 2^64);\n"
        "                                              LIMIT >= 2^127 (e.g. 2^1000) uses GMP; --gmp forces it,\n"
        "                                              --max-nodes X caps the estimated enumeration (default 5e9)\n"
        "  a137723 next [-N SCAN] [-l LIMIT] [-t T] [-m MAXN] [-o FILE]\n"
        "                                              scan to SCAN (default 10^10), then search missing even n\n"
        "  a137723 check m                             factor m and count its prime gaps (A073490)\n"
        "  a137723 run m                               show the maximal run of gapful numbers around m\n"
        "                                              (check/run accept numbers above 2^128 when built with GMP)\n"
        "  a137723 selftest [-t T]\n"
        "numbers: decimal, 2^k, 10^k, 1e15, or sums/differences like 2^54-32;  -t defaults to the number of CPUs\n");
    exit(2);
}

/*
 * main - command-line entry point.
 *
 * Function: builds the small-prime tables, parses the options (they may
 *           appear anywhere after the command; everything else is a
 *           positional argument), chooses 128-bit or GMP mode from the size of
 *           -l, and dispatches to the command.
 * Inputs:   argv[1], the command; options -t T (threads; default: the online
 *           CPUs, at least 1), -m MAXN, -o FILE, -l LIMIT, -N SCAN, --gmp,
 *           --max-nodes X.
 * Outputs:  exit status 0 on success (selftest: 0 or 1), 1 after die(), 2
 *           after usage().
 * Errors:   unknown options -> usage(); malformed numbers -> die(); a scan N
 *           below 100 is raised to 100; more than 64 positional arguments are
 *           ignored; -t with a non-number gives one thread (atoi()).
 */
int main(int argc, char **argv)
{
    if (argc < 2) usage();
    init_small_primes();
    stderr_tty = isatty(2);
    int threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (threads < 1) threads = 1;
    const char *cmd = argv[1];
    const char *pos[64];
    int npos = 0, maxn = 0;
    const char *outfile = NULL;
    const char *limit_str = NULL;
    u128 limit = (u128)1 << 64;
    u64 scanN = 10000000000ULL;
    bool force_gmp = false;
    double max_nodes = 5e9;

    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "-t") && i + 1 < argc) { threads = atoi(argv[++i]); if (threads < 1) threads = 1; }
        else if (!strcmp(argv[i], "-m") && i + 1 < argc) maxn = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) outfile = argv[++i];
        else if (!strcmp(argv[i], "-l") && i + 1 < argc) limit_str = argv[++i];
        else if (!strcmp(argv[i], "--gmp")) force_gmp = true;
        else if (!strcmp(argv[i], "--max-nodes") && i + 1 < argc) max_nodes = atof(argv[++i]);
        else if (!strcmp(argv[i], "-N") && i + 1 < argc) scanN = arg_u64(argv[++i], "scan limit");
        else if (argv[i][0] == '-' && argv[i][1]) usage();
        else if (npos < 64) pos[npos++] = argv[i];
    }

    /* 128-bit mode for LIMIT < 2^127, GMP mode for LIMIT >= 2^127 or --gmp */
    bool gmp_mode = force_gmp;
#ifdef HAVE_GMP
    mpz_t zlimit;
    mpz_init(zlimit);
    mpz_ui_pow_ui(zlimit, 2, 64);
    if (limit_str) {
        if (!parse_mpz(limit_str, zlimit)) die("bad limit: '%s'", limit_str);
        if (mpz_sizeinbase(zlimit, 2) <= 127) limit = z_to_u128(zlimit);
        else gmp_mode = true;
    }
#else
    if (limit_str) limit = arg_u128(limit_str, "limit");
    if (limit > ((u128)1 << 127)) die("limit above 2^127 needs the GMP build (make with gmp.h available)");
    if (force_gmp) die("this build has no GMP support");
    (void)max_nodes;                             /* --max-nodes only matters in GMP mode */
#endif

    if (!strcmp(cmd, "scan")) {
        if (npos != 1) usage();
        u64 N = arg_u64(pos[0], "N");
        if (N < 100) N = 100;
        if (N > ((u64)1 << 63)) die("N must be <= 2^63");
        cmd_scan(N, threads, maxn, outfile);
    } else if (!strcmp(cmd, "search")) {
        if (npos < 1) usage();
        for (int i = 0; i < npos; i++) {
            u128 n = arg_u128(pos[i], "n");
            if (n > 100000) die("n too large");
#ifdef HAVE_GMP
            if (gmp_mode) { cmd_search_gmp((int)n, zlimit, threads, max_nodes); continue; }
#endif
            cmd_search((int)n, limit, threads);
        }
    } else if (!strcmp(cmd, "next")) {
        if (npos) usage();
        if (gmp_mode) die("'next' uses the 128-bit search; use 'search n -l LIMIT' above 2^127");
        cmd_next(scanN, limit, threads, maxn, outfile);
    } else if (!strcmp(cmd, "check") || !strcmp(cmd, "run")) {
        if (npos != 1) usage();
        bool is_check = !strcmp(cmd, "check");
#ifdef HAVE_GMP
        mpz_t zm;
        mpz_init(zm);
        if (!parse_mpz(pos[0], zm)) die("bad m: '%s'", pos[0]);
        /* numbers of 2^127 and above go to the GMP versions */
        if (mpz_sizeinbase(zm, 2) > 127) {
            if (is_check) cmd_check_gmp(zm); else cmd_run_gmp(zm);
            mpz_clear(zm);
            return 0;
        }
        mpz_clear(zm);
#endif
        if (is_check) cmd_check(arg_u128(pos[0], "m"));
        else cmd_run(arg_u128(pos[0], "m"));
    } else if (!strcmp(cmd, "selftest")) {
        cmd_selftest(threads);
    } else {
        usage();
    }
    return 0;
}
