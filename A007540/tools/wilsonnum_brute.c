/*
 * wilsonnum_brute.c -- decide the Wilson-number condition directly from the definition.
 *
 * ================================================================================================
 * THEORY OF OPERATION
 * ================================================================================================
 *
 *    n is a Wilson number (OEIS A157250) if
 *
 *        P(n) + e(n) == 0   (mod n^2),
 *
 *    where P(n) is the product of the a in [1, n] with gcd(a, n) = 1, and e(n) = +1 if n has a
 *    primitive root (n = 1, 2, 4, p^k or 2p^k with p an odd prime) and e(n) = -1 otherwise.
 *
 *    This program evaluates the definition literally: it runs through every a in [1, n], keeps
 *    the a coprime to n, and multiplies them modulo n^2.  It shares no code and no mathematics
 *    with wilsonnum.c (no Wilson-quotient table, no decomposition into prime-power components),
 *    which makes it the independent check for every number that wilsonnum.c reports.  A third,
 *    pure-Python check of the same definition is tools/wilson_verify.py.
 *
 *    Arithmetic: for n < 2^32, n^2 < 2^64, so every residue fits in 64 bits and each step needs
 *    one 64x64->128-bit product and one remainder.  The cost is about n gcd computations plus
 *    phi(n) multiplications, i.e. linear in n: about 5 s for n = 3.4*10^8 on 8 threads of an
 *    M1 Pro.  (wilsonnum.c needs a few minutes for every n <= 2^32 together; this program is
 *    for checking individual numbers, or for scanning small ranges with -all.)
 *
 *    Modes:
 *      single numbers   the range [1, n] is split into nt contiguous pieces; each thread
 *                       multiplies the units of its piece mod n^2, and the partial products are
 *                       multiplied together at the end (numbers below 100000 use one thread);
 *      -all N           every n in [2, N] is tested, each one in a single thread; threads take
 *                       blocks of 64 consecutive n from a shared counter.
 *
 * USAGE
 *
 *    wilsonnum_brute [-t threads] n [n ...]   print "n  P(n) mod n^2 = r  e(n)=+-1  verdict" per n
 *    wilsonnum_brute [-t threads] -all N      print "WILSON n" for every Wilson number 2 <= n <= N
 *                                             (in no particular order), then "DONE scan to N"
 *
 *    -t, if present, must be the first argument (default 8 threads).
 *
 * ERROR CONDITIONS
 *
 *    - n >= 2^32 is rejected ("n too large", exit status 1).
 *    - Not checked: n = 0 (division by zero); a thread count outside 1..256 (0 threads silently
 *      gives a wrong verdict, more than 256 overflows the thread arrays); -all with N >= 2^32
 *      (n^2 overflows) or without a value.
 *    - Verdict overflow for n > 3.04*10^9 (n^2 > 2^63): the verdict adds e(n) = -1 as n^2 - 1,
 *      and r + n^2 - 1 can then exceed 2^64 and wrap.  A true Wilson number is never affected
 *      (it has r = 1 when e(n) = -1, and r = n^2 - 1 when e(n) = +1, and neither sum wraps), but
 *      a non-Wilson n with P(n) == 1 + 2^64 (mod n^2), probability about 1/n^2, would be
 *      labelled "WILSON NUMBER".  The printed residue r is always exact (128-bit products), so
 *      for such n read the criterion from r directly: r = 1 with e(n) = -1, or r = n^2 - 1 with
 *      e(n) = +1.  All 17 new terms of 2026-10-04 printed r = 1 with e(n) = -1.
 *    - Numbers are read with strtoull, which stops at the first non-digit: "1e8" is read as 1,
 *      and a single argument "5 13" is read as 5 alone.  Pass each n as a separate plain decimal
 *      argument (in zsh an unquoted $VAR is not split into words; use ${=VAR}).
 *    - Exit status 0 otherwise.
 *
 * BUILD
 *
 *    cc -O2 -o wilsonnum_brute wilsonnum_brute.c -lpthread
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
typedef unsigned long long u64; typedef unsigned __int128 u128;

/* gcd -- greatest common divisor by Euclid's algorithm.  In: any a, b.  Out: gcd(a, b) (gcd(a, 0) = a). */
static u64 gcd(u64 a, u64 b) { while (b) { u64 t = a % b; a = b; b = t; } return a; }

/*
 * e_of -- the Gauss sign e(n): +1 iff n has a primitive root (n = 1, 2, 4, p^k, 2p^k with p odd).
 *   In:     n >= 1.
 *   Out:    +1 or -1.
 *   How:    strip the factors of 2; 4 | n (n != 4) means -1; otherwise n is p^k or 2p^k iff the
 *           odd part is a prime power, which is tested by finding its smallest prime factor by
 *           trial division (at most sqrt(n)/2 steps) and dividing it out.
 *   Errors: none for n >= 1 (n = 0 returns +1 and is meaningless).
 */
static int e_of(u64 n) {            /* +1 iff n in {1,2,4}, p^k, 2p^k (p odd) */
    if (n <= 4 && n != 3) return 1;                 /* n = 1, 2, 4 (and 0) */
    u64 m = n; int two = 0; while (m % 2 == 0) { m /= 2; two++; }
    if (two >= 2) return (m == 1 && two <= 2) ? 1 : -1;
    /* m odd: must be a prime power */
    if (m == 1) return 1; /* n = 2 */
    u64 q = m; u64 d = 3; while (d * d <= q && q % d) d += 2; u64 p = (d * d <= q) ? d : q;   /* p = smallest prime factor of m */
    while (q % p == 0) q /= p;
    return q == 1 ? 1 : -1;
}

/*
 * job_t / worker -- one thread's share of P(n): the product, modulo n^2, of the a in [lo, hi)
 * with gcd(a, n) = 1.
 *   In:     *a = job with n, lo, hi filled in (1 <= lo <= hi <= n + 1, n < 2^32).
 *   Out:    j->prod = that product mod n^2 (1 for an empty range); returns NULL.
 */
typedef struct { u64 n, lo, hi, prod; } job_t;
static void *worker(void *a) { job_t *j = a; u64 n2 = j->n * j->n, r = 1; for (u64 x = j->lo; x < j->hi; x++) if (gcd(x, j->n) == 1) r = (u64)(((u128)r * x) % n2); j->prod = r; return 0; }

/*
 * P_mod_n2 -- P(n) mod n^2 using nt threads.
 *   In:     1 <= n < 2^32; 1 <= nt <= 256 (forced to 1 for n < 100000, where threads cost more
 *           than they save).
 *   Out:    P(n) mod n^2.
 *   How:    split [1, n] into nt nearly equal contiguous ranges, one worker thread each, then
 *           multiply the partial products mod n^2 (multiplication is commutative, so the split
 *           does not change the result).
 *   Errors: none detected; see ERROR CONDITIONS for n = 0 and bad thread counts.
 */
static u64 P_mod_n2(u64 n, int nt) {
    if (n < 100000) nt = 1;
    pthread_t th[256]; job_t jb[256]; u64 n2 = n * n;
    for (int t = 0; t < nt; t++) { jb[t].n = n; jb[t].lo = 1 + n * t / nt; jb[t].hi = 1 + n * (t + 1) / nt; pthread_create(&th[t], 0, worker, &jb[t]); }
    u64 r = 1; for (int t = 0; t < nt; t++) { pthread_join(th[t], 0); r = (u64)(((u128)r * jb[t].prod) % n2); }
    return r;
}

/* range scan: threads take n's dynamically; each n done single-threaded */
static u64 scanN; static volatile u64 next_n = 2; static pthread_mutex_t lk = PTHREAD_MUTEX_INITIALIZER;   /* lk guards next_n and stdout */

/*
 * scan_worker -- thread body for -all: repeatedly claim the next block of 64 consecutive n (under
 * the mutex), test each n <= scanN from the definition in this thread, and print "WILSON n" for
 * every hit (under the mutex, so lines are not interleaved).  In: arg unused.  Out: NULL.
 * The condition is P(n) + e(n) == 0 mod n^2, with e(n) = -1 written as + (n^2 - 1); for N above
 * 3.04*10^9 that sum can wrap (see ERROR CONDITIONS).
 */
static void *scan_worker(void *a) { (void)a; for (;;) { pthread_mutex_lock(&lk); u64 n = next_n; next_n += 64; pthread_mutex_unlock(&lk); if (n > scanN) break;
    for (u64 m = n; m < n + 64 && m <= scanN; m++) { u64 n2 = m * m, r = 1; for (u64 x = 1; x <= m; x++) if (gcd(x, m) == 1) r = (u64)(((u128)r * x) % n2);
        int e = e_of(m); if ((r + (e > 0 ? 1 : n2 - 1)) % n2 == 0) { pthread_mutex_lock(&lk); printf("WILSON %llu\n", m); fflush(stdout); pthread_mutex_unlock(&lk); } } }
    return 0; }

/*
 * main -- parse the arguments and run one of the two modes (see USAGE).
 *   Out:    exit status 0, or 1 if some n >= 2^32 (numbers before it have already been printed).
 */
int main(int argc, char **argv) {
    int nt = 8, a = 1; if (argc > 2 && !strcmp(argv[1], "-t")) { nt = atoi(argv[2]); a = 3; }
    if (a < argc && !strcmp(argv[a], "-all")) { scanN = strtoull(argv[a + 1], 0, 10); pthread_t th[256]; for (int t = 0; t < nt; t++) pthread_create(&th[t], 0, scan_worker, 0); for (int t = 0; t < nt; t++) pthread_join(th[t], 0); printf("DONE scan to %llu\n", scanN); return 0; }
    for (; a < argc; a++) { u64 n = strtoull(argv[a], 0, 10); if (n >= (1ULL << 32)) { fprintf(stderr, "n too large\n"); return 1; }
        /* Wilson iff r + e(n) == 0 mod n^2; e(n) = -1 is added as n^2 - 1 to stay unsigned
           (can wrap for n > 3.04*10^9; see ERROR CONDITIONS: trust r itself for such n) */
        u64 r = P_mod_n2(n, nt); int e = e_of(n); u64 n2 = n * n; int ok = ((r + (e > 0 ? 1 : n2 - 1)) % n2) == 0;
        printf("%llu  P(n) mod n^2 = %llu  e(n)=%+d  %s\n", n, r, e, ok ? "WILSON NUMBER" : "not a Wilson number"); fflush(stdout); }
    return 0;
}
