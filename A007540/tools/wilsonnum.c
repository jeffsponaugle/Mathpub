/*
 * wilsonnum.c  --  exhaustive search for Wilson numbers (OEIS A157250) up to a bound N < 2^32.
 *
 * ================================================================================================
 * THEORY OF OPERATION
 * ================================================================================================
 *
 * 1. The problem
 *
 *    For n >= 1 let P(n) be the product of the integers a in [1, n] with gcd(a, n) = 1 (the
 *    "n-phi-torial", OEIS A001783), and let e(n) = +1 if n has a primitive root (n = 1, 2, 4,
 *    p^k or 2p^k with p an odd prime) and e(n) = -1 otherwise.  Gauss's generalisation of
 *    Wilson's theorem says P(n) == -e(n) (mod n).  n is a WILSON NUMBER when this holds one
 *    level deeper:
 *
 *        P(n) + e(n) == 0   (mod n^2),
 *
 *    i.e. n divides the generalised Wilson quotient (P(n) + e(n))/n (OEIS A157249).  For a prime
 *    p this is the Wilson-prime condition (A007540: only 5, 13, 563 are known, none other below
 *    2*10^13).  Checking a single n from the definition costs about n multiplications, so a
 *    direct scan of every n <= 2^32 would take about 10^19 operations.  This program instead
 *    reduces the condition to a few numbers per prime, and tests all n <= 2^32 in minutes.
 *
 * 2. Splitting n into prime-power components
 *
 *    By the Chinese remainder theorem, P(n) == -e(n) (mod n^2) holds iff it holds modulo
 *    p^(2k) for every prime power p^k exactly dividing n (written p^k || n).  Fix such a p^k
 *    and write n = p^k * m with p !| m.  Define
 *
 *        G_k  = product of the j in [1, p^k) with p !| j       (the Gauss factorial of p^k),
 *        eps  = G_k mod p^k = -1 for odd p and for 2^1, 2^2, and +1 for 2^k with k >= 3 (Gauss),
 *        A(m) = product over the primes l | m of l^(-phi(m)/(l-1))      (a unit mod p^(2k)).
 *
 *    Then
 *
 *        (F1)  P(n)  == A(m)^phi(p^k) * G_k^phi(m)   (mod p^(2k))   for every p^k except 2^1,
 *        (F2)  P(2m) == A(m) * (-1)^(phi(m)/2)       (mod 4)        for odd m >= 3.
 *
 *    Proof sketch of (F1).  Moebius inversion over the divisors d of m removes the condition
 *    gcd(a, m) = 1:  P(n) = prod_{d|m} ( d^(M*phi(p^k)) * prod_{b <= p^k*M, p !| b} b )^mu(d)
 *    with M = m/d.  Split b = c*p^k + j with 0 <= c < M and 1 <= j < p^k, p !| j.  Modulo p^(2k)
 *
 *        prod_j (c*p^k + j) == G_k * (1 + c*p^k * sum_j 1/j) == G_k,
 *
 *    because sum_j 1/j == 0 (mod p^k): the inverses run over all units mod p^k, whose sum is
 *    phi(p^k)*p^k/2, and that is 0 mod p^k for every p^k except p^k = 2 (hence (F2)).  Finally
 *    sum_{d|m} mu(d)*(m/d) = phi(m), and the exponent of a prime l in prod_{d|m} d^(mu(d)*m/d)
 *    is -phi(m)/(l-1).  (F2) comes from the same Moebius inversion with p^k = 2: the product of
 *    the M odd numbers below 2M is (-1)^((M-1)/2) (mod 4) for odd M, and summing the exponents
 *    over d | m gives (phi(m) - sum_{d|m} mu(d))/2 = phi(m)/2 for m > 1.  Both formulas were
 *    checked against the direct product for every prime-power component of every n <= 3000
 *    (6845 components, no mismatch).
 *
 * 3. Linearisation: each component test is one congruence mod p^k
 *
 *    Write G_k = eps + p^k * w (mod p^(2k)).  w is the generalised Wilson quotient of p^k; for
 *    k = 1 it is the ordinary Wilson quotient w_p = ((p-1)! + 1)/p mod p.  For a unit x let
 *
 *        q(x) = (x^phi(p^k) - 1)/p^k mod p^k           (generalised Fermat quotient).
 *
 *    Since x^phi(p^k) == 1 + p^k*q(x) (mod p^(2k)), q is additive: q(xy) == q(x) + q(y) (mod
 *    p^k).  Substituting into (F1) and dropping terms divisible by p^(2k):
 *
 *        P(n) == eps^phi(m) * (1 + p^k * (eps*phi(m)*w - t))   (mod p^(2k)),
 *        t    == sum over the primes l | m of (phi(m)/(l-1)) * q(l)   (mod p^k).
 *
 *    eps^phi(m) == -e(n) always (Gauss again; component_ok checks it), so the p^k-component of
 *    the Wilson condition is exactly
 *
 *        (C)   eps * phi(m) * w  ==  t   (mod p^k).
 *
 *    (C) needs only w(p^k), phi(m), the prime factors l of m and the Fermat quotients q(l).
 *    For k = 1 it is Theorem 5.1 of T. Agoh, K. Dilcher, L. Skula, "Wilson quotients for
 *    composite moduli", Math. Comp. 67 (1998) 843-861 ("ADS" below).  For p^k = 2^1, (F2)
 *    always evaluates to P(2m) == 1 (mod 4) (ADS Lemma 3.3), so that component passes iff
 *    e(n) = -1.  Heuristically a random n passes (C) with probability about 1/p^k.
 *
 * 4. Where the per-prime data comes from
 *
 *    - w_p for every prime 7 <= p <= N/2: binary table files written by the Costa-Gerbicz-
 *      Harvey Wilson-prime program pw13 (third_party/Wilson-prime/pw13.c, function printallres,
 *      patched for this project).  File format: a sequence of 16-byte records
 *      { uint64 p; uint64 w; } in native (little-endian) byte order, with w = ((p-1)! + 1)/p in
 *      [1, p] (w == p marks a Wilson prime).  Several files may be given, one per pw13 range;
 *      they are concatenated and sorted here.  p = 3 and 5 are computed directly.
 *    - w(p^k) for k >= 2 and p^k <= N (so p <= sqrt N): computed at start-up, in parallel.  k = 2
 *      uses an O(p) power-sum method (compute_wp2_fast), which is checked against the direct
 *      O(p^2) product at every start; k >= 3 uses the direct product (compute_wpk_direct).
 *    - q(l): computed on demand by one modular exponentiation and cached per thread (qcache_t).
 *
 * 5. Search organisation (every n is visited exactly once)
 *
 *    Let P be the largest prime factor of n and K its exponent, so n = P^K * m with every prime
 *    factor of m below P.  The search enumerates the pairs (P^K, m):
 *
 *      pass 1  worker(), multithreaded:            K = 1,  P an odd prime <= N/2,  m = 1 .. N/P;
 *      pass 2  prime_power_pass(), single thread:  K >= 2, P^K <= N,               m = 1 .. N/P^K.
 *
 *    In both passes m is factored with a smallest-prime-factor table and skipped unless P !| m
 *    and every prime of m is below P.  For each remaining n the P^K-component (C) is tested
 *    first; it is cheap and passes with probability about 1/P.  The rare survivors get every
 *    component tested (full_test) and are reported.  Never visited: n = 1 (a Wilson number by
 *    convention, printed in the LIST line), n = 2 (not a Wilson number: P(2) + e(2) = 2), and
 *    the primes in (N/2, N] (a prime Wilson number is a Wilson prime; there is none between 563
 *    and 2*10^13).  Consequently the "n tested" count in the SUMMARY line must equal
 *
 *        N - 2 - (pi(N) - pi(N/2)),
 *
 *    e.g. 191606 for N = 2*10^5, 97239677 for N = 10^8, 487323449 for N = 5*10^8 and
 *    4196784637 for N = 2^32 - 1.  All four values were observed, which confirms that the
 *    enumeration is complete and visits no n twice.
 *
 * 6. Arithmetic
 *
 *    n <= N < 2^32 implies p^(2k) <= n^2 < 2^64 for every component, so all modular arithmetic
 *    is single-word: Montgomery multiplication for odd moduli, masked arithmetic for powers of 2
 *    (unsigned multiplication wraps modulo 2^64, which is exact modulo 2^(2k)).
 *    compute_wp2_fast works modulo p^4 < 2^64 (p <= 65535); moduli that large need the
 *    overflow-safe mont_redc.
 *
 * 7. Safeguards built into the program
 *
 *    - The tables must contain every prime in [7, N/2] and only primes (else exit status 2).
 *    - compute_wp2_fast is compared with direct products for all 5 <= p <= 1500 and for
 *      p = 4099, 16411, 60013, 65521 at every start (exit status 2 on any disagreement).
 *    - "-check K" compares the formula with the definition (direct_wilson) on the first 11
 *      Wilson numbers > 1 and on K random n <= min(N, 3*10^7).
 *    - "-n n" prints the component-by-component trace for one n, plus the direct verdict.
 *    - Hits should be re-verified independently with the definition: tools/wilsonnum_brute.c
 *      (C) and tools/wilson_verify.py (Python).
 *
 * 8. Resources and performance (Apple M1 Pro, 10 threads, N = 2^32 - 1, 2026-10-04)
 *
 *    Memory about 9 GB at peak (observed resident size 8.0 GiB): smallest-prime-factor table
 *    4*(N/3) bytes = 5.7 GB, prime list 0.4 GB, Wilson-quotient table 16 bytes per prime
 *    (1.7 GB, about 3.8 GB while loading), byte sieve of odd numbers to N/2 1.1 GB (freed).
 *    Time 934 s in total: table load and sieves 24 s, w(p^2) self-check 39 s, w(p^k) table 209 s, "-check 500"
 *    424 s (single-threaded), search 238 s.
 *    Pass 1 scales poorly with the number of threads: primes are handed out in chunks of 256,
 *    and the first chunk (P < 1620) carries about 60% of all m iterations, so one thread does
 *    most of the work; pass 2 (about 0.45*N factorisations) is single-threaded.  Measured at
 *    N = 10^8 (2026-10-12): the search phase took 10 s with 1 thread and 5 s with 8 threads.
 *    Smaller chunks for small P, and a parallel pass 2, would cut the search time several-fold.
 *
 * 9. Limits and error conditions
 *
 *    - N must satisfy 1 <= N < 2^32 (checked; exit status 1).
 *    - Exit status: 0 success; 1 usage error, bad N or unreadable table file; 2 data problem
 *      (table incomplete, a table entry that is not prime or has w > p, w(p^2) self-check
 *      failure, a required Wilson quotient missing).
 *    - Not checked, the caller must ensure: 1 <= threads <= 256 (-j; with 0 threads pass 1 is
 *      silently skipped), at most 64 "-t" files, every option is followed by its value.
 *      Numbers are read with strtoull/atoi, which stop at the first non-digit: "-N 1e8" is read
 *      as N = 1 and silently searches nothing.  Give plain decimal integers.
 *      Memory allocation failures are not detected (the program would crash).  Duplicate table
 *      records for the same p are tolerated but not cross-checked; pw13 APPENDS to its output
 *      file, so do not rerun pw13 in a directory that already holds a wilsonres.bin.
 *    - "(known)" in the output only means n <= the -b bound (default 5*10^8).  It is not a
 *      comparison with the published list: the three terms below 5*10^8 that were missing from
 *      the literature (24365727, 86286991, 139459265) were printed as "(known)".
 *    - "-check" prints mismatches to stderr but does not change the exit status.
 *
 * ================================================================================================
 * USAGE
 *
 *    wilsonnum -N <N> -t <table.bin> [-t <table2.bin> ...] [-j <threads>] [-b <N0>]
 *              [-check <K>] [-n <n>] [-slowwpk]
 *
 *      -N N        search bound, 1 <= N < 2^32
 *      -t file     Wilson-quotient table(s) from the patched pw13; together they must cover every
 *                  prime 7 <= p <= N/2 (entries above N/2 are ignored)
 *      -j threads  number of threads (default 8; must be 1..256)
 *      -b N0       Wilson numbers <= N0 are labelled "(known)", larger ones "*** NEW ***"
 *                  (default 500000000, the bound of the 1998 ADS search)
 *      -check K    before the search, compare formula and definition on 5, 13, ..., 26921605 and
 *                  on K random n <= min(N, 3*10^7)
 *      -n n        trace the formula test of the single number n (n >= 2, n <= N) and exit
 *      -slowwpk    compute every w(p^k) by direct products (for testing compute_wp2_fast)
 *
 * OUTPUT (stdout; progress and diagnostics go to stderr)
 *
 *    WILSON <n>  (known)       or   WILSON <n>  *** NEW ***     one line per Wilson number > 1,
 *                                                                in the order the threads find them
 *    SUMMARY N=<N>: <a> n tested at their largest prime power, <b> full tests, <c> Wilson numbers > 1 found
 *    LIST 1 <n1> <n2> ...                                         all Wilson numbers <= N, sorted
 *
 * BUILD
 *
 *    cc -O2 -o wilsonnum wilsonnum.c -lpthread -lm
 *
 * HISTORY (details in ../A157250.md)
 *
 *    2026-10-04  First version.  Fixed a q-cache stamp collision in full_test that caused false
 *                negatives (341569806 and 139459265 were missed in the first 5*10^8 run); fixed a
 *                mont_redc overflow for moduli above about 1.14*10^19; added the O(p) method for
 *                w(p^2).  The run to 2^32 - 1 found 29 Wilson numbers > 1, 17 of them new.
 *    2026-10-12  Comments only (theory of operation and per-function documentation); removed two
 *                no-op statements (an empty loop with an unused variable in compute_wp2_fast, and
 *                a self-assignment in main).  The compiled object code is byte-identical.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <math.h>
#include <stdatomic.h>
#include <time.h>

typedef uint64_t u64; typedef uint32_t u32; typedef unsigned __int128 u128;

/* ================================================================================================
 * 64-bit Montgomery arithmetic for a fixed odd modulus n < 2^64
 * ================================================================================================
 * A residue a mod n is stored in "Montgomery form" a*R mod n with R = 2^64.  A product then costs
 * one 64x64->128-bit multiplication plus one reduction (mont_redc) instead of a 128-bit division.
 * Every modulus used with these routines is odd: p^(2k) for an odd prime p, or p^4 inside
 * compute_wp2_fast.  Powers of 2 are handled separately with bit masks (fermat_q2,
 * compute_wpk_direct).
 */
typedef struct { u64 n, ninv, r2; } mont_t;       /* ninv = -n^{-1} mod 2^64, r2 = 2^128 mod n */

/*
 * mont_init -- prepare a Montgomery context.
 *   In:     M = context to fill; n = modulus, odd, 3 <= n < 2^64.
 *   Out:    M->n = n, M->ninv = -n^(-1) mod 2^64, M->r2 = 2^128 mod n.
 *   How:    Newton's iteration inv <- inv*(2 - n*inv) doubles the number of correct low bits of
 *           n^(-1) at each step; starting from inv = 1 (correct mod 2), six steps give 64 bits.
 *   Errors: none reported.  An even n has no inverse mod 2^64 and gives a meaningless context.
 *           This happens only for n = 4 in wilson_quotient_prime(2), whose result is never used.
 */
static void mont_init(mont_t *M, u64 n) {
    u64 inv = 1; for (int i = 0; i < 6; i++) inv *= 2 - n * inv;   /* n^{-1} mod 2^64 */
    M->n = n; M->ninv = (u64)0 - inv;
    u128 r = ((u128)1 << 64) % n; M->r2 = (u64)((r * r) % n);
}

/*
 * mont_redc -- Montgomery reduction.
 *   In:     M = context for the odd modulus n; t < n * 2^64 (for example a product of two
 *           residues below n).
 *   Out:    t * 2^(-64) mod n, fully reduced into [0, n).
 *   How:    m = (t mod 2^64) * ninv makes t + m*n divisible by 2^64, and (t + m*n)/2^64 < 2n,
 *           so one conditional subtraction finishes.  The sum t + m*n (up to 129 bits) is never
 *           formed: its low 64 bits are zero, and adding the two low words carries exactly when
 *           the low word of t is nonzero.  Forming the sum directly overflowed for moduli above
 *           about 1.14*10^19 (p^4 with p > 58000); fixed 2026-10-04.
 *   Errors: none detected.  The result is wrong if t >= n * 2^64 or n is even.
 */
static inline u64 mont_redc(const mont_t *M, u128 t) {   /* valid for any odd n < 2^64 and t < n * 2^64 */
    u64 m = (u64)t * M->ninv;
    u128 mn = (u128)m * M->n;
    /* (t + mn) >> 64 without forming the possibly-overflowing sum: low words cancel to 0 mod 2^64,
       with a carry iff the low word of t is nonzero */
    u128 u = (t >> 64) + (mn >> 64) + ((u64)t != 0);
    if (u >= M->n) u -= M->n;
    return (u64)u;
}

/* mont_mul -- product of two Montgomery-form residues a, b < n; the result is in Montgomery form. */
static inline u64 mont_mul(const mont_t *M, u64 a, u64 b) { return mont_redc(M, (u128)a * b); }

/* mont_to -- convert any 64-bit integer a to Montgomery form, (a mod n) * 2^64 mod n. */
static inline u64 mont_to(const mont_t *M, u64 a) { return mont_mul(M, a % M->n, M->r2); }

/* mont_from -- convert a Montgomery-form residue back to an ordinary residue in [0, n). */
static inline u64 mont_from(const mont_t *M, u64 a) { return mont_redc(M, a); }

/*
 * mont_pow -- modular power by binary square-and-multiply.
 *   In:     M = context; a_m = base in Montgomery form; e = exponent (e = 0 is allowed).
 *   Out:    a^e mod n, in Montgomery form.
 *   Errors: none.
 */
static u64 mont_pow(const mont_t *M, u64 a_m, u64 e) {   /* a_m in Montgomery form; returns Montgomery form */
    u64 r = mont_to(M, 1);
    while (e) { if (e & 1) r = mont_mul(M, r, a_m); a_m = mont_mul(M, a_m, a_m); e >>= 1; }
    return r;
}

/*
 * fermat_q -- generalised Fermat quotient for an odd prime power p^k.
 *   In:     M = Montgomery context for the modulus p^(2k); pk = p^k; phipk = phi(p^k) =
 *           p^(k-1)*(p-1); x = an integer coprime to p (in this program always a prime l != p).
 *   Out:    q(x) = (x^phi(p^k) - 1)/p^k mod p^k, a value in [0, p^k).
 *   Why:    by Euler x^phi(p^k) == 1 (mod p^k), so x^phi(p^k) == 1 + p^k*q(x) (mod p^(2k)).  q is
 *           additive, which turns A(m)^phi(p^k) into the sum t of theory section 3.
 *   Errors: none detected.  If p | x the division is not exact and the result is meaningless.
 */
static inline u64 fermat_q(const mont_t *M, u64 pk, u64 phipk, u64 x) {
    u64 y = mont_from(M, mont_pow(M, mont_to(M, x), phipk));      /* x^phi(p^k) mod p^2k, == 1 mod p^k */
    u64 d = (y >= 1) ? y - 1 : y + M->n - 1;                         /* y - 1 mod p^2k (y = 0 cannot occur for valid x) */
    return (d / pk) % pk;                                            /* exact: d is a multiple of pk */
}

/*
 * fermat_q2 -- the same quotient for p = 2:  q(x) = (x^(2^(k-1)) - 1)/2^k mod 2^k.
 *   In:     k >= 2 with 2k <= 62 (true whenever 2^k <= N < 2^32); x odd.
 *   Out:    q(x) in [0, 2^k).
 *   How:    arithmetic modulo 2^(2k) by masking.  Unsigned multiplication wraps modulo 2^64,
 *           which is exact modulo 2^(2k), so the k-1 squarings never need 128 bits.
 *   Errors: none detected; meaningless for even x.  k = 1 is never used here, because the 2^1
 *           component has its own rule in component_ok.
 */
static inline u64 fermat_q2(int k, u64 x) {
    u64 mask = (2 * k >= 64) ? ~(u64)0 : (((u64)1 << (2 * k)) - 1);
    u64 y; x &= mask;
    y = x; for (int i = 0; i < k - 1; i++) y = (y * y) & mask;      /* x^(2^(k-1)) by repeated squaring */
    u64 d = (y - 1) & mask;
    return (d >> k) & (((u64)1 << k) - 1);
}

/* ================================================================================================
 * Global data.  main() fills all of it before any worker thread starts; afterwards it is read-only,
 * except for the result list and counters in the "results" section, which are synchronised.
 * ================================================================================================ */
static u64 N;                 /* search bound */
static u64 debug_n = 0, ncheck = 0;          /* -n: single number to trace;  -check: number of random cross-checks */
static u64 N0 = 500000000ULL; /* terms <= N0 are "known" */
static int nthreads = 8;                     /* -j: number of threads for the w(p^k) table and for pass 1 */
static u32 *spf;              /* smallest prime factor, index up to spf_max */
static u64 spf_max;                          /* max(N/3, sqrt N) + 2; every m factored by factor_spf is <= N/3 */
static u32 *primes; static u64 nprimes;      /* all primes <= N/2 */
static u64 *tab_p, *tab_w; static u64 tab_n; /* Wilson quotient table, sorted by p */
static u32 *pidx_small; static u64 pidx_max; /* prime -> index among primes <= sqrtN (for q caches) */
static u64 sqrtN;                            /* floor(sqrt(N)) + 1 */

/* generalised Wilson quotients for prime powers p^k, k >= 2, p^k <= N: sorted array of (p^k, w) */
typedef struct { u64 pk, w; u32 p; u32 k; } wpk_t;   /* w = (G_k - eps)/p^k mod p^k; sorted by pk */
static wpk_t *wpk; static u64 nwpk;

/*
 * wilson_quotient_prime -- Wilson quotient w_p = ((p-1)! + 1)/p mod p of a prime p.
 *   In:     p prime.
 *   Out:    w_p in [0, p).  w_p = 0 means p is a Wilson prime (5, 13, 563).
 *   How:    binary search in the sorted table (tab_p, tab_w), which main() has already reduced
 *           mod p.  Primes missing from the table but below 2*10^6 (in practice only 3 and 5)
 *           are computed directly from (p-1)! mod p^2, which costs p multiplications.
 *   Errors: a prime >= 2*10^6 missing from the table is fatal (message, exit status 2).  It
 *           cannot happen during a search, because main() verifies that the table holds every
 *           prime in [7, N/2], but it can happen for "-n n" with a prime factor above N/2.
 *           For p = 2 the direct branch would need the even modulus 4 and returns a meaningless
 *           value; that value is never used (the 2^1 component ignores w), but "-n" prints it.
 *   Thread safety: reads global tables only; safe to call from worker threads.
 */
static u64 wilson_quotient_prime(u64 p) {        /* w_p mod p from the table; small primes computed directly */
    u64 lo = 0, hi = tab_n;
    while (lo < hi) { u64 mid = (lo + hi) / 2; if (tab_p[mid] < p) lo = mid + 1; else hi = mid; }
    if (lo < tab_n && tab_p[lo] == p) return tab_w[lo];
    if (p < 2000000) {                            /* direct: (p-1)! mod p^2 */
        mont_t M; mont_init(&M, p * p); u64 r = mont_to(&M, 1);
        for (u64 i = 2; i < p; i++) r = mont_mul(&M, r, mont_to(&M, i));
        u64 f = mont_from(&M, r); return ((f + 1) / p) % p;
    }
    fprintf(stderr, "FATAL: prime %llu missing from Wilson quotient table\n", (unsigned long long)p); exit(2);
}

/*
 * wilson_quotient_pk -- generalised Wilson quotient w(p^k) for k >= 2.
 *   In:     p prime, k >= 2, with p^k <= N.
 *   Out:    w in [0, p^k), defined by G_k == eps + p^k * w (mod p^(2k)).
 *   How:    binary search on pk in the array wpk[], which main() fills before the search.
 *   Errors: fatal (message, exit status 2) if p^k was not precomputed, i.e. p^k > N or k < 2.
 *   Thread safety: read-only; safe from worker threads.
 */
static u64 wilson_quotient_pk(u64 p, u32 k) {    /* w(p^k) for k >= 2 from precomputed array */
    u64 pk = 1; for (u32 i = 0; i < k; i++) pk *= p;
    u64 lo = 0, hi = nwpk;
    while (lo < hi) { u64 mid = (lo + hi) / 2; if (wpk[mid].pk < pk) lo = mid + 1; else hi = mid; }
    if (lo < nwpk && wpk[lo].pk == pk) return wpk[lo].w;
    fprintf(stderr, "FATAL: w(%llu^%u) not precomputed\n", (unsigned long long)p, k); exit(2);
}

/*
 * addmod -- (a + b) mod n for a, b < n, without overflow even when n is close to 2^64.
 */
static inline u64 addmod(u64 a, u64 b, u64 n) { return (a >= n - b) ? a - (n - b) : a + b; }   /* a,b < n, no overflow */

/*
 * compute_wp2_fast -- w(p^2) in O(p) operations instead of the O(p^2) direct product.
 *   In:     p prime with 5 <= p <= 65535.
 *   Out:    w(p^2) in [0, p^2), defined by G_2 == -1 + p^2 * w (mod p^4).
 *   How:    G_2 = prod_{j < p^2, p !| j} j is computed modulo p^4.  Write j = c*p + i with
 *           0 <= c < p and 1 <= i <= p-1.  For one block c,
 *               prod_i (c*p + i) = (p-1)! * sum_r (c*p)^r * e_r,
 *           where e_r is the r-th elementary symmetric polynomial of {1/i : 1 <= i <= p-1}
 *           (p-integral).  Modulo p^4 only r <= 3 survive, so
 *               G_2 == ((p-1)!)^p * prod_{c=0}^{p-1} (1 + c*p*e1 + c^2*p^2*e2 + c^3*p^3*e3)  (mod p^4).
 *           e1, e2, e3 come from the power sums S_r = sum_i 1/i^r by Newton's identities:
 *               e1 = S1,  e2 = (S1^2 - S2)/2,  e3 = (S1^3 - 3*S1*S2 + 2*S3)/6.
 *           The inverses of 1 .. p-1 mod p^4 come from the linear-time recurrence
 *               inv[i] = -floor(M4/i) * inv[M4 mod i]   (from M4 = floor(M4/i)*i + (M4 mod i)).
 *   Errors: none detected.  p < 5 breaks the division by 6 (not invertible mod 2^4 or 3^4), and
 *           p > 65535 makes p^4 overflow 64 bits; compute_wpk() only calls this for
 *           5 <= p <= 65535, and main() compares it with compute_wpk_direct at every start.
 *           The malloc of p*8 bytes (at most 512 KB) is not checked.
 */
static u64 compute_wp2_fast(u64 p) {
    u64 M4 = p * p * p * p; mont_t M; mont_init(&M, M4);
    u64 *inv = malloc(p * sizeof(u64));             /* plain residues mod p^4 */
    inv[1] = 1;
    for (u64 i = 2; i < p; i++) { u64 q = M4 / i, r = M4 % i; inv[i] = (u64)((M4 - (u128)q * inv[r] % M4) % M4); }
    /* power sums and (p-1)! in Montgomery form */
    u64 S1 = 0, S2 = 0, S3 = 0, fact = mont_to(&M, 1), jm = mont_to(&M, 1), one = mont_to(&M, 1);
    for (u64 i = 1; i < p; i++) {
        u64 x = mont_to(&M, inv[i]); u64 x2 = mont_mul(&M, x, x);
        S1 = addmod(S1, x, M4); S2 = addmod(S2, x2, M4); u64 x3 = mont_mul(&M, x2, x); S3 = addmod(S3, x3, M4);
        fact = mont_mul(&M, fact, jm); jm = addmod(jm, one, M4);              /* jm = Montgomery form of i, incremented */
    }
    free(inv);
    /* e1, e2, e3 (Montgomery form) */
    u64 inv2 = mont_to(&M, (M4 + 1) / 2), inv6;                /* 1/2 mod p^4 (M4 odd) */
    { /* 1/6 mod p^4 by Euler's theorem: 6^(phi(p^4) - 1); valid because gcd(6, p) = 1 for p >= 5 */
      u64 g = 6 % M4; u128 a = 1; u64 phi = M4 / p * (p - 1); a = mont_from(&M, mont_pow(&M, mont_to(&M, g), phi - 1)); inv6 = mont_to(&M, (u64)a); }
    u64 e1 = S1;
    u64 e2 = mont_mul(&M, addmod(mont_mul(&M, S1, S1), M4 - S2, M4), inv2);
    u64 s1s2 = mont_mul(&M, S1, S2), m3 = addmod(addmod(M4 - s1s2, M4 - s1s2, M4), M4 - s1s2, M4);   /* -3 S1 S2 */
    u64 t3 = addmod(addmod(mont_mul(&M, mont_mul(&M, S1, S1), S1), m3, M4), addmod(S3, S3, M4), M4);   /* S1^3 - 3 S1 S2 + 2 S3 */
    u64 e3 = mont_mul(&M, t3, inv6);
    /* prod over c of (1 + c p e1 + c^2 p^2 e2 + c^3 p^3 e3) */
    u64 pm = mont_to(&M, p), pe1 = mont_mul(&M, pm, e1), p2e2 = mont_mul(&M, mont_mul(&M, pm, pm), e2), p3e3 = mont_mul(&M, mont_mul(&M, mont_mul(&M, pm, pm), pm), e3);
    u64 prod = one, cm = 0;                                     /* cm = Montgomery form of c */
    for (u64 c = 0; c < p; c++) {
        u64 c2 = mont_mul(&M, cm, cm), c3 = mont_mul(&M, c2, cm);
        u64 term = addmod(one, mont_mul(&M, cm, pe1), M4); term = addmod(term, mont_mul(&M, c2, p2e2), M4); term = addmod(term, mont_mul(&M, c3, p3e3), M4);
        prod = mont_mul(&M, prod, term);
        cm = addmod(cm, one, M4);
    }
    u64 G = mont_from(&M, mont_mul(&M, mont_pow(&M, fact, p), prod));      /* G_2 mod p^4, == -1 mod p^2 */
    return ((G + 1) / (p * p)) % (p * p);
}

/*
 * compute_wpk_direct -- generalised Wilson quotient w(p^k) from the definition.
 *   In:     p prime, k >= 1, with p^(2k) < 2^64 (true for p^k <= N < 2^32).
 *   Out:    w in [0, p^k) with G_k == eps + p^k * w (mod p^(2k)), where G_k = prod_{j < p^k,
 *           p !| j} j and eps = -1 (odd p, or p = 2 with k <= 2) or +1 (p = 2, k >= 3).
 *   How:    one multiplication per j, about p^k in total: masked arithmetic modulo 2^(2k) for
 *           p = 2, Montgomery arithmetic modulo p^(2k) otherwise.  Used for every k >= 3, for
 *           k = 2 when p < 5 or "-slowwpk" is given, and as the reference in main()'s self-check.
 *   Errors: none detected.  The odd-p branch relies on G_k == -1 (mod p^k) (Gauss); if that
 *           failed, (g + 1)/pk would not be exact and the result would be meaningless.
 *   Cost:   O(p^k).  The k = 3 entries dominate the w(p^k) table at N = 2^32 (about 3.5 minutes
 *           on 10 threads).
 */
static u64 compute_wpk_direct(u64 p, u32 k) {
    u64 pk = 1; for (u32 i = 0; i < k; i++) pk *= p;
    if (p == 2) {
        u64 mask = (2 * k >= 64) ? ~(u64)0 : (((u64)1 << (2 * k)) - 1);
        u64 g = 1; for (u64 j = 1; j < pk; j += 2) g = (g * j) & mask;
        u64 eps = (k <= 2) ? (mask) /* -1 */ : 1;
        return ((g - eps) & mask) >> k;
    }
    mont_t M; mont_init(&M, pk * pk); u64 r = mont_to(&M, 1);
    for (u64 j = 1; j < pk; j++) if (j % p) r = mont_mul(&M, r, mont_to(&M, j));
    u64 g = mont_from(&M, r);                   /* == -1 mod p^k */
    return ((g + 1) / pk) % pk;
}

/* ================================================================================================
 * Factorisation of the cofactor m, and the Gauss sign e(n)
 * ================================================================================================ */

/*
 * fac_t -- factorisation of a number: distinct primes l[0..cnt-1] with exponents e[], phi of the
 * number, and its largest prime factor maxp (1 for the number 1).  16 slots are more than enough:
 * a number below 2^32 has at most 9 distinct prime factors, and full_test adds one more.
 */
typedef struct { u32 l[16]; u32 e[16]; int cnt; u64 phi; u32 maxp; } fac_t;

/*
 * factor_spf -- factor m using the smallest-prime-factor table.
 *   In:     1 <= m <= spf_max; f = output structure.
 *   Out:    *f = factorisation of m, primes in increasing order, phi(m) and maxp; m = 1 gives
 *           cnt = 0, phi = 1, maxp = 1.
 *   How:    repeatedly divide by spf[m]; at most log2(m) table lookups.
 *   Errors: none detected.  m > spf_max reads past the end of the table (undefined behaviour);
 *           the callers only pass m <= N/3 (pass 1, since P >= 3) or m <= N/4 (pass 2).
 */
static inline void factor_spf(u64 m, fac_t *f) {
    f->cnt = 0; f->phi = 1; f->maxp = 1;
    while (m > 1) {
        u32 l = spf[m]; u32 e = 0; u64 pe = 1;
        while (m % l == 0) { m /= l; e++; pe *= l; }
        f->l[f->cnt] = l; f->e[f->cnt] = e; f->cnt++;
        f->phi *= pe / l * (l - 1); if (l > f->maxp) f->maxp = l;
    }
}

/*
 * e_of_n -- the Gauss sign e(n): +1 if n has a primitive root (n = 1, 2, 4, p^k, 2p^k with p odd),
 * -1 otherwise.
 *   In:     fn = factorisation of n with the primes in increasing order (so 2, if it divides n,
 *           is fn->l[0]).
 *   Out:    +1 or -1.
 *   Errors: none detected; an unsorted fn can give the wrong sign for n = 2p^k.
 */
static inline int e_of_n(const fac_t *fn) {       /* Gauss sign e(n) from the factorisation of n */
    if (fn->cnt == 0) return 1;                                      /* n = 1 */
    if (fn->cnt == 1) return (fn->l[0] == 2) ? (fn->e[0] <= 2 ? 1 : -1) : 1;   /* 2,4 -> +1; 8,16.. -> -1; p^k -> +1 */
    if (fn->cnt == 2 && fn->l[0] == 2 && fn->e[0] == 1) return 1;              /* 2 p^k */
    return -1;
}

/* ================================================================================================
 * Per-thread cache of Fermat quotients q(l) for the prime power currently being tested
 * ================================================================================================
 * Pass 1 visits about N/P cofactors m for each prime P, and the same small primes l divide many
 * of them, so q(l) (one modular exponentiation each) is cached.  Only primes l <= pidx_max
 * (about sqrt N) are cached, in an array indexed by the position of l among those primes
 * (pidx_small[l]); larger l are recomputed on each use.  Instead of clearing the array whenever
 * the prime power changes, every entry carries a "stamp": an entry is valid only while its stamp
 * equals the cache's current stamp, and qcache_set() starts a new stamp.  Each thread owns its own
 * cache, so no locking is needed.
 */
typedef struct {
    mont_t M; u64 p, pk, phipk; u32 k; u64 stamp;
    u64 *q; u64 *qstamp;          /* indexed by small-prime index; valid if qstamp[i] == stamp */
} qcache_t;

/*
 * qcache_set -- switch the cache to the prime power p^k and invalidate all cached q values.
 *   In:     c = cache (q and qstamp arrays already allocated with pidx_max + 2 entries);
 *           p prime, k >= 1.
 *   Out:    c->p, c->k, c->pk = p^k, c->phipk = phi(p^k), the Montgomery context for p^(2k)
 *           (odd p only; p = 2 uses masks), and a new stamp.
 *   Errors: none detected; p^(2k) must be < 2^64.
 */
static void qcache_set(qcache_t *c, u64 p, u32 k) {
    c->p = p; c->k = k; c->pk = 1; for (u32 i = 0; i < k; i++) c->pk *= p;
    c->phipk = c->pk / p * (p - 1);
    if (p != 2) mont_init(&c->M, c->pk * c->pk);
    c->stamp++;
}

/*
 * qcache_get -- Fermat quotient q(l) = (l^phi(p^k) - 1)/p^k mod p^k for the cache's prime power.
 *   In:     c = cache set to p^k; l = a prime different from p.
 *   Out:    q(l) in [0, p^k).  The value is cached when l <= pidx_max.
 *   Errors: none detected; l = p gives a meaningless value (see fermat_q).
 */
static inline u64 qcache_get(qcache_t *c, u32 l) {   /* q_{p^k}(l) mod p^k, l prime < p (or any l coprime to p) */
    u32 i = (l <= pidx_max) ? pidx_small[l] : 0xffffffffu;
    if (i != 0xffffffffu && c->qstamp[i] == c->stamp) return c->q[i];
    u64 v = (c->p == 2) ? fermat_q2(c->k, l) : fermat_q(&c->M, c->pk, c->phipk, l);
    if (i != 0xffffffffu) { c->q[i] = v; c->qstamp[i] = c->stamp; }
    return v;
}

/* ================================================================================================
 * The Wilson-number test
 * ================================================================================================ */

/*
 * component_ok -- test the p^k-component of the Wilson condition for n = p^k * m.
 *   In:     c   = q-cache set to p^k;
 *           fm  = factorisation of m (p !| m), including phi(m);
 *           m   = the cofactor itself (only compared with 1 here);
 *           w   = w(p^k) mod p^k (ignored for p^k = 2^1);
 *           e_n = e(n) for the full n.
 *   Out:    1 if P(n) == -e(n) (mod p^(2k)), 0 otherwise.
 *   How:    theory section 3.  For p^k = 2^1, formula (F2) gives P(n) mod 4 directly (it always
 *           comes out as 1, so this component passes iff e(n) = -1).  Otherwise compute
 *           t = sum_l (phi(m)/(l-1)) q(l) mod p^k and test congruence (C), eps*phi(m)*w == t.
 *           The sign test "eps^phi(m) == -e(n)" always succeeds for consistent input (Gauss); it
 *           is kept as a cheap internal consistency check.
 *   Errors: none detected.
 */
static int component_ok(qcache_t *c, const fac_t *fm, u64 m, u64 w, int e_n) {
    u64 p = c->p, pk = c->pk, phi = fm->phi;
    if (p == 2 && c->k == 1) {                      /* P(2m) == A(m) (-1)^(phi/2) mod 4, m odd >= 3; m = 1 -> P(2) = 1 */
        if (m == 1) return (1 + e_n) % 4 == 0;      /* n = 2: P=1, e=+1 -> 1+1=2 !=0 mod 4 -> false */
        int sgn = (phi / 2) & 1;                    /* exponent parity from (-1)^(phi/2) */
        for (int i = 0; i < fm->cnt; i++) if (fm->l[i] % 4 == 3) sgn ^= (phi / (fm->l[i] - 1)) & 1;   /* l^(-x) == (-1)^x mod 4 when l == 3 mod 4 */
        int P4 = sgn ? 3 : 1;                       /* P(n) mod 4 */
        return (P4 + e_n + 4) % 4 == 0;
    }
    /* eps = G_k mod p^k (Gauss): -1 for odd p and for 2^2 (2^1 was handled above), +1 for 2^k, k >= 3 */
    int eps = (p == 2) ? (c->k <= 2 ? -1 : 1) : -1;
    /* t = sum (phi/(l-1)) q(l) mod p^k */
    u64 t = 0;
    for (int i = 0; i < fm->cnt; i++) {
        u64 coef = (phi / (fm->l[i] - 1)) % pk;
        u64 q = qcache_get(c, fm->l[i]);
        t = (u64)(((u128)coef * q + t) % pk);
    }
    /* predicted P(n) = eps^phi * (1 + p^k (eps*phi*w - t)) mod p^2k ; must equal -e_n */
    int sign = (phi & 1) ? eps : 1;                 /* eps^phi */
    if (sign != -e_n) return 0;
    u64 term = (u64)(((u128)(phi % pk) * (w % pk)) % pk);
    if (eps == -1) term = (pk - term) % pk;         /* eps*phi*w */
    return term == t;                               /* eps*phi*w - t == 0 mod p^k */
}

/*
 * full_test -- test every prime-power component of n = P^K * m.
 *   In:     c  = the calling thread's q-cache (any state);
 *           P  = a prime with P !| m, K >= 1; in the search P is the largest prime factor of n;
 *           m  = cofactor, fm = its factorisation (at most 15 distinct primes);
 *           verbose = 1 to print one trace line per component to stderr.
 *   Out:    1 if n is a Wilson number, 0 otherwise.
 *   How:    build and sort the factorisation of n, compute e(n), then for each p^k || n build the
 *           factorisation of n/p^k, look up w(p^k) and call component_ok.  It stops at the first
 *           failing component.  The P^K-component, already tested by the caller, is tested
 *           again; that is redundant but cheap, because full tests are rare.
 *   Side effects: the q arrays of c are reused for the other prime powers, with fresh stamps,
 *           and c->stamp is advanced before returning, so every q value cached in c beforehand is
 *           invalid afterwards (c->p, c->pk and c->M are unchanged).  Without the final advance,
 *           entries written for the last component looked valid to the caller's next m: that was
 *           the stamp-collision bug, fixed 2026-10-04, which caused false negatives.
 *   Errors: through wilson_quotient_prime / wilson_quotient_pk, a missing Wilson quotient is
 *           fatal (exit status 2).  This cannot happen for n <= N once main()'s checks passed.
 */
static int full_test(qcache_t *c, u64 P, u32 K, u64 m, const fac_t *fm, int verbose) {
    fac_t fn = *fm;                                 /* factorisation of n = fm plus P^K */
    fn.l[fn.cnt] = (u32)P; fn.e[fn.cnt] = K; fn.cnt++;
    /* phi(n) not needed; e(n) needs sorted-ish structure: compute by rule */
    fac_t fs = fn;                                  /* sort by prime for e_of_n (2 first) */
    for (int i = 0; i < fs.cnt; i++) for (int j = i + 1; j < fs.cnt; j++) if (fs.l[j] < fs.l[i]) { u32 t1 = fs.l[i]; fs.l[i] = fs.l[j]; fs.l[j] = t1; u32 t2 = fs.e[i]; fs.e[i] = fs.e[j]; fs.e[j] = t2; }
    int e_n = e_of_n(&fs);
    u64 n = 1; for (int i = 0; i < fs.cnt; i++) for (u32 j = 0; j < fs.e[i]; j++) n *= fs.l[i];
    for (int i = 0; i < fs.cnt; i++) {
        u64 p = fs.l[i]; u32 k = fs.e[i];
        u64 pk = 1; for (u32 j = 0; j < k; j++) pk *= p;
        /* factorisation and phi of the cofactor mm = n / p^k, built from fs (no table lookup needed) */
        u64 mm = n / pk; fac_t fmm; fmm.cnt = 0; fmm.phi = 1; fmm.maxp = 1;
        for (int j = 0; j < fs.cnt; j++) if (j != i) { fmm.l[fmm.cnt] = fs.l[j]; fmm.e[fmm.cnt] = fs.e[j]; fmm.cnt++; u64 pe = 1; for (u32 q = 0; q < fs.e[j]; q++) pe *= fs.l[j]; fmm.phi *= pe / fs.l[j] * (fs.l[j] - 1); if (fs.l[j] > fmm.maxp) fmm.maxp = fs.l[j]; }
        u64 w = (k == 1) ? wilson_quotient_prime(p) : wilson_quotient_pk(p, k);
        /* temporary cache for p^k that shares c's arrays; qcache_set gives it a new stamp */
        qcache_t cc = *c; cc.q = c->q; cc.qstamp = c->qstamp; qcache_set(&cc, p, k);
        int ok = component_ok(&cc, &fmm, mm, w, e_n);
        c->stamp = cc.stamp;                        /* keep stamps monotone */
        if (verbose) fprintf(stderr, "   n=%llu component %llu^%u (m=%llu, w=%llu): %s\n", (unsigned long long)n, (unsigned long long)p, k, (unsigned long long)mm, (unsigned long long)w, ok ? "pass" : "FAIL");
        if (!ok) { c->stamp++; return 0; }          /* fresh stamp: cache entries written here belong to other prime powers */
    }
    c->stamp++;
    return 1;
}

/*
 * direct_wilson -- decide the Wilson condition from the definition (reference for -check and -n).
 *   In:     1 <= n <= 3.04*10^9 (so that n^2 < 2^63; the callers use n <= 5*10^7).
 *   Out:    1 if P(n) + e(n) == 0 (mod n^2), else 0.
 *   How:    multiply every a in [1, n] with gcd(a, n) = 1 modulo n^2, then compute e(n) by trial
 *           division: e = +1 for n = 1, 2, 4; -1 if 4 | n otherwise; else +1 iff the odd part of
 *           n is a prime power.  About n gcd computations, so only practical for n up to a few
 *           times 10^7.
 *   Errors: none detected.  n >= 2^32 overflows n^2; for n^2 > 2^63 the final r + n^2 - 1 (the
 *           way e(n) = -1 is added) can wrap past 2^64 and, with probability about 1/n^2, turn a
 *           non-Wilson n into a false "Wilson".  Neither can happen for the callers' n <= 5*10^7.
 */
static int direct_wilson(u64 n) {
    u64 n2 = n * n, r = 1;
    for (u64 a = 1; a <= n; a++) { u64 x = a, y = n; while (y) { u64 t = x % y; x = y; y = t; } if (x == 1) r = (u64)(((u128)r * a) % n2); }
    /* e(n) */
    u64 m = n; int two = 0; while (m % 2 == 0) { m /= 2; two++; }
    int e; if (n == 1 || n == 2 || n == 4) e = 1; else if (two >= 2) e = -1; else { u64 q = m, d = 3; while (d * d <= q && q % d) d += 2; u64 pp = (d * d <= q) ? d : q; while (q % pp == 0) q /= pp; e = (q == 1) ? 1 : -1; }
    return ((r + (e > 0 ? 1 : n2 - 1)) % n2) == 0;
}

/*
 * formula_test_n -- run the formula test on one arbitrary n (used by -n and -check).
 *   In:     2 <= n <= N; verbose = 1 for a component trace on stderr.
 *   Out:    1 if the formula says n is a Wilson number, else 0.
 *   How:    factor n by trial division (about sqrt(n) steps), take its largest prime P and the
 *           exponent K, and call full_test with a freshly allocated q-cache.
 *   Errors: n = 1 is not supported (empty factorisation; reads an uninitialised slot).  A prime
 *           factor p > N/2 with p >= 2*10^6, or a prime power p^k > N with k >= 2, has no Wilson
 *           quotient available and is fatal (exit status 2).  Allocation failures not checked.
 */
static int formula_test_n(u64 n, int verbose) {
    fac_t f; f.cnt = 0; f.phi = 1; f.maxp = 1; u64 m = n;
    for (u64 d = 2; d * d <= m; d++) if (m % d == 0) { u32 e = 0; while (m % d == 0) { m /= d; e++; } f.l[f.cnt] = (u32)d; f.e[f.cnt] = e; f.cnt++; }
    if (m > 1) { f.l[f.cnt] = (u32)m; f.e[f.cnt] = 1; f.cnt++; }
    /* largest prime power last; build cofactor factorisation */
    int li = 0; for (int i = 1; i < f.cnt; i++) if (f.l[i] > f.l[li]) li = i;
    u64 P = f.l[li]; u32 K = f.e[li]; u64 PK = 1; for (u32 j = 0; j < K; j++) PK *= P;
    fac_t fm; fm.cnt = 0; fm.phi = 1; fm.maxp = 1;
    for (int i = 0; i < f.cnt; i++) if (i != li) { fm.l[fm.cnt] = f.l[i]; fm.e[fm.cnt] = f.e[i]; fm.cnt++; u64 pe = 1; for (u32 q = 0; q < f.e[i]; q++) pe *= f.l[i]; fm.phi *= pe / f.l[i] * (f.l[i] - 1); if (f.l[i] > fm.maxp) fm.maxp = f.l[i]; }
    qcache_t c; memset(&c, 0, sizeof c); c.q = malloc((pidx_max + 2) * sizeof(u64)); c.qstamp = calloc(pidx_max + 2, sizeof(u64)); c.stamp = 1;
    qcache_set(&c, P, K);
    int ok = full_test(&c, P, K, n / PK, &fm, verbose);
    free(c.q); free(c.qstamp);
    return ok;
}

/* ================================================================================================
 * Results (shared between threads)
 * ================================================================================================ */
static pthread_mutex_t out_lock = PTHREAD_MUTEX_INITIALIZER;   /* guards found[], nfound, cap_found and stdout */
static u64 *found; static u64 nfound, cap_found;               /* every Wilson number > 1 found, unsorted until the end */
static atomic_ullong n_tested, n_fulltests;                    /* totals for the SUMMARY line (added once per thread) */

/*
 * report -- record and print one Wilson number.
 *   In:     n = the Wilson number; known = 1 if n <= N0 (the -b bound), which only selects the
 *           label "(known)" versus "*** NEW ***".
 *   Out:    n appended to found[]; one "WILSON" line written to stdout and flushed.
 *   Thread safety: serialised by out_lock.
 *   Errors: realloc failure not checked.
 */
static void report(u64 n, int known) {
    pthread_mutex_lock(&out_lock);
    if (nfound == cap_found) { cap_found = cap_found ? 2 * cap_found : 64; found = realloc(found, cap_found * sizeof(u64)); }
    found[nfound++] = n;
    printf("WILSON %llu%s\n", (unsigned long long)n, known ? "  (known)" : "  *** NEW ***");
    fflush(stdout);
    pthread_mutex_unlock(&out_lock);
}

/* ================================================================================================
 * Pass 1: n = P * m with P the largest prime factor of n and P || n  (multithreaded)
 * ================================================================================================ */
static atomic_ullong next_chunk;   /* index of the next unclaimed entry of primes[] */
#define CHUNK 256                  /* primes claimed per atomic operation */

/*
 * worker -- thread body of pass 1.
 *   In:     arg unused; reads the global tables.
 *   Out:    returns NULL; Wilson numbers go to report(); the counts are added to n_tested and
 *           n_fulltests when the thread finishes.
 *   How:    claim CHUNK consecutive primes at a time.  For each odd prime P (P = 2 cannot be the
 *           largest prime of any n > 2 with 2 || n): look up w_P, switch the q-cache to P, and
 *           for m = 1 .. N/P with P !| m and all primes of m below P, test the P-component; run
 *           full_test on the survivors.  e(n) for n = P*m is +1 exactly when m = 1 or m = 2.
 *   Load balance: the chunk holding the smallest primes has by far the most m values, so this
 *           pass scales poorly beyond a few threads (theory section 8).
 *   Errors: a missing Wilson quotient is fatal (cannot happen after main()'s table check);
 *           allocation failures not checked.
 */
static void *worker(void *arg) {
    (void)arg;
    qcache_t c; memset(&c, 0, sizeof c);
    c.q = malloc((pidx_max + 2) * sizeof(u64)); c.qstamp = calloc(pidx_max + 2, sizeof(u64)); c.stamp = 1;
    /* the q arrays need one slot per prime <= pidx_max; pidx_max + 2 slots is always enough */
    fac_t fm; u64 local_tested = 0, local_full = 0;
    for (;;) {
        u64 start = atomic_fetch_add(&next_chunk, CHUNK);
        if (start >= nprimes) break;
        u64 end = start + CHUNK < nprimes ? start + CHUNK : nprimes;
        for (u64 ip = start; ip < end; ip++) {
            u64 P = primes[ip]; if (P == 2) continue;
            u64 w = wilson_quotient_prime(P);
            qcache_set(&c, P, 1);
            u64 mmax = N / P;
            for (u64 m = 1; m <= mmax; m++) {
                if (m % P == 0) continue;
                factor_spf(m, &fm);
                if (fm.maxp > P) continue;                      /* P is not the largest prime of n */
                local_tested++;
                /* e(n) for n = P*m: +1 iff m == 1 or m == 2 */
                int e_n = (m <= 2) ? 1 : -1;
                if (!component_ok(&c, &fm, m, w, e_n)) continue;
                local_full++;
                if (full_test(&c, P, 1, m, &fm, 0)) report(P * m, P * m <= N0);
            }
        }
    }
    atomic_fetch_add(&n_tested, local_tested); atomic_fetch_add(&n_fulltests, local_full);
    free(c.q); free(c.qstamp);
    return NULL;
}

/* ================================================================================================
 * Pass 2: n = P^K * m with P the largest prime factor of n and K >= 2  (single thread)
 * ================================================================================================ */

/*
 * prime_power_pass -- the same scan as worker(), for the prime powers P^K (K >= 2, P^K <= N)
 * listed in wpk[].
 *   In:     global tables, including wpk[] with its Wilson quotients.
 *   Out:    Wilson numbers go to report(); the counts are added to n_tested and n_fulltests.
 *   How:    for each P^K and m = 1 .. N/P^K with P !| m and all primes of m below P: test the
 *           P^K-component, then full_test the survivors.  e(n) = +1 exactly for n = 4 (P = 2,
 *           K = 2, m = 1), n = p^K and n = 2p^K with p odd; -1 otherwise.
 *   Cost:   about 0.45*N factorisations in one thread; most m are rejected because they have a
 *           prime factor larger than P (e.g. P = 2 only keeps m = 1).
 *   Errors: as worker().
 */
static void prime_power_pass(void) {
    qcache_t c; memset(&c, 0, sizeof c);
    c.q = malloc((pidx_max + 2) * sizeof(u64)); c.qstamp = calloc(pidx_max + 2, sizeof(u64)); c.stamp = 1;
    fac_t fm; u64 tested = 0, full = 0;
    for (u64 i = 0; i < nwpk; i++) {
        u64 P = wpk[i].p; u32 K = wpk[i].k; u64 PK = wpk[i].pk; u64 w = wpk[i].w;
        qcache_set(&c, P, K);
        u64 mmax = N / PK;
        for (u64 m = 1; m <= mmax; m++) {
            if (m % P == 0) continue;
            factor_spf(m, &fm);
            if (fm.maxp > P) continue;
            tested++;
            int e_n;
            if (P == 2) e_n = (m == 1) ? (K <= 2 ? 1 : -1) : -1;    /* 2^k: e=+1 for 2,4 ; 2^k*m (m odd>=3): -1 */
            else e_n = (m <= 2) ? 1 : -1;                              /* p^k, 2p^k: +1 */
            if (!component_ok(&c, &fm, m, w, e_n)) continue;
            full++;
            if (full_test(&c, P, K, m, &fm, 0)) report(PK * m, PK * m <= N0);
        }
    }
    atomic_fetch_add(&n_tested, tested); atomic_fetch_add(&n_fulltests, full);
    free(c.q); free(c.qstamp);
}

/* ================================================================================================
 * Start-up helpers: the w(p^k) table and qsort comparators
 * ================================================================================================ */
static int slow_wpk = 0;      /* -slowwpk: always use compute_wpk_direct */

/*
 * compute_wpk -- choose the method for w(p^k): compute_wp2_fast for k = 2 and 5 <= p <= 65535
 * (unless -slowwpk), otherwise compute_wpk_direct.  In: p prime, k >= 2.  Out: w(p^k).
 */
static u64 compute_wpk(u64 p, u32 k) {
    if (!slow_wpk && k == 2 && p >= 5 && p <= 65535) return compute_wp2_fast(p);
    return compute_wpk_direct(p, k);
}

static atomic_ullong wpk_next;   /* next wpk[] entry to compute */

/*
 * wpk_worker -- thread body that fills wpk[i].w for the entries it claims, one index at a time
 * from the shared counter wpk_next (dynamic scheduling, because the cost per entry ranges from
 * microseconds to seconds).  In: arg unused.  Out: NULL.
 */
static void *wpk_worker(void *a) { (void)a; for (;;) { u64 i = atomic_fetch_add(&wpk_next, 1); if (i >= nwpk) break; wpk[i].w = compute_wpk(wpk[i].p, wpk[i].k); } return NULL; }

/* cmp_u64 -- qsort comparator on the first u64 of each element (used for found[] and for the
 * (p, w) record pairs, which it sorts by p). */
static int cmp_u64(const void *a, const void *b) { u64 x = *(const u64 *)a, y = *(const u64 *)b; return x < y ? -1 : x > y; }

/* cmp_wpk -- qsort comparator ordering wpk[] by the prime power pk. */
static int cmp_wpk(const void *a, const void *b) { u64 x = ((const wpk_t *)a)->pk, y = ((const wpk_t *)b)->pk; return x < y ? -1 : x > y; }

/* ================================================================================================
 * main -- parse options, build the tables, run the checks and the two search passes, print results
 * ================================================================================================
 *   In:     command line (see USAGE at the top of the file).
 *   Out:    results on stdout, progress on stderr.  Exit status 0 on success; 1 for a usage error,
 *           N out of range or an unreadable table file; 2 for a data problem (incomplete or invalid
 *           table, w(p^2) self-check failure, missing Wilson quotient).
 *   Steps:  1. options;  2. load and validate the Wilson-quotient tables;  3. sieve primes <= N/2
 *           and smallest prime factors <= N/3;  4. table completeness check;  5. index of small
 *           primes for the q-caches;  6. self-check of compute_wp2_fast;  7. w(p^k) for all
 *           prime powers p^k <= N, k >= 2;  8. optional -n trace (then exit) or -check;
 *           9. pass 1 (threads) and pass 2;  10. sorted summary.
 */
int main(int argc, char **argv) {
    /* --- 1. options (values are not validated beyond the range check on N; see theory section 9) --- */
    const char *tabfiles[64]; int ntab = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-N")) N = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "-t")) tabfiles[ntab++] = argv[++i];
        else if (!strcmp(argv[i], "-j")) nthreads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-b")) N0 = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "-n")) debug_n = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "-check")) ncheck = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "-slowwpk")) slow_wpk = 1;
        else { fprintf(stderr, "usage: %s -N N -t table.bin [-t ...] [-j threads] [-b knownbound]\n", argv[0]); return 1; }
    }
    if (!N || N >= ((u64)1 << 32)) { fprintf(stderr, "N must be in [1, 2^32)\n"); return 1; }
    time_t t0 = time(NULL);
    sqrtN = (u64)sqrtl((long double)N) + 1;

    /* --- 2. load Wilson quotient table ---
     * Read all files into one array of (p, w) pairs (grown by doubling), sort the pairs by p, and
     * split them into tab_p[] and tab_w[].  pw13 writes w = ((p-1)! + 1)/p, which lies in [1, p]
     * (w = p exactly for a Wilson prime); anything larger is a corrupt record.  w is reduced mod p
     * here so that lookups return w_p mod p.  Duplicate records are kept; lookups use the first. */
    { u64 cap = 1 << 20; tab_n = 0; u64 *pairs = malloc(cap * 2 * sizeof(u64));
      for (int i = 0; i < ntab; i++) {
          FILE *f = fopen(tabfiles[i], "rb"); if (!f) { perror(tabfiles[i]); return 1; }
          u64 buf[2];
          while (fread(buf, sizeof buf, 1, f) == 1) {
              if (tab_n == cap) { cap *= 2; pairs = realloc(pairs, cap * 2 * sizeof(u64)); }
              pairs[2 * tab_n] = buf[0]; pairs[2 * tab_n + 1] = buf[1]; tab_n++;
          }
          fclose(f);
      }
      /* sort pairs by p */
      qsort(pairs, tab_n, 2 * sizeof(u64), cmp_u64);
      tab_p = malloc(tab_n * sizeof(u64)); tab_w = malloc(tab_n * sizeof(u64));
      for (u64 i = 0; i < tab_n; i++) { tab_p[i] = pairs[2 * i]; tab_w[i] = pairs[2 * i + 1];
          if (tab_w[i] > tab_p[i]) { fprintf(stderr, "FATAL: bad table entry p=%llu w=%llu\n", (unsigned long long)tab_p[i], (unsigned long long)tab_w[i]); return 2; }
          tab_w[i] %= tab_p[i]; }       /* pw13 stores w = ((p-1)!+1)/p in [1, p]; w == p means Wilson prime */
      free(pairs);
      fprintf(stderr, "table: %llu primes, p in [%llu, %llu]\n", (unsigned long long)tab_n, (unsigned long long)(tab_n ? tab_p[0] : 0), (unsigned long long)(tab_n ? tab_p[tab_n - 1] : 0));
    }

    /* --- 3. sieve primes <= N/2, spf <= N/3 ---
     * spf[]: smallest prime factor of every integer up to spf_max (sieve of Eratosthenes that
     * records the first prime to strike each number; 4 bytes per entry, 5.7 GB at N = 2^32).
     * primes[]: every prime <= N/2, from a byte sieve over odd numbers that is freed afterwards. */
    spf_max = N / 3 + 2; if (spf_max < sqrtN + 2) spf_max = sqrtN + 2;
    spf = calloc(spf_max + 1, sizeof(u32));
    for (u64 i = 2; i <= spf_max; i++) if (!spf[i]) { spf[i] = (u32)i; if (i * i <= spf_max) for (u64 j = i * i; j <= spf_max; j += i) if (!spf[j]) spf[j] = (u32)i; }
    u64 pmax = N / 2;
    {
      uint8_t *comp = calloc(pmax / 2 + 1, 1);        /* odd numbers only: index i <-> 2i+1 */
      for (u64 i = 1; (2 * i + 1) * (2 * i + 1) <= pmax; i++) if (!comp[i]) { u64 p = 2 * i + 1; for (u64 j = (p * p - 1) / 2; j <= pmax / 2; j += p) comp[j] = 1; }
      u64 cnt = 1; for (u64 i = 1; 2 * i + 1 <= pmax; i++) if (!comp[i]) cnt++;
      primes = malloc(cnt * sizeof(u32)); nprimes = 0; primes[nprimes++] = 2;
      for (u64 i = 1; 2 * i + 1 <= pmax; i++) if (!comp[i]) primes[nprimes++] = (u32)(2 * i + 1);
      free(comp);
    }
    fprintf(stderr, "primes <= %llu: %llu;  spf table to %llu  (%.1f s)\n", (unsigned long long)pmax, (unsigned long long)nprimes, (unsigned long long)spf_max, difftime(time(NULL), t0));

    /* --- 4. table completeness check: every prime 7 <= p <= N/2 must be present ---
     * Merge-walk the sorted prime list against the sorted table.  Then check that every table
     * entry small enough to be covered by spf[] (p <= spf_max) is prime; entries above that are
     * only required to be present, not re-checked. */
    { u64 missing = 0, ti = 0;
      for (u64 i = 0; i < nprimes; i++) { u64 p = primes[i]; if (p < 7) continue; while (ti < tab_n && tab_p[ti] < p) ti++; if (ti == tab_n || tab_p[ti] != p) { if (missing < 5) fprintf(stderr, "  table missing p=%llu\n", (unsigned long long)p); missing++; } }
      if (missing) { fprintf(stderr, "FATAL: %llu primes missing from table\n", (unsigned long long)missing); return 2; }
      /* and every table entry p <= spf_max must be prime (w <= p was checked while loading) */
      for (u64 i = 0; i < tab_n; i++) if ((tab_p[i] <= spf_max && spf[tab_p[i]] != tab_p[i])) { fprintf(stderr, "FATAL: bad table entry p=%llu w=%llu\n", (unsigned long long)tab_p[i], (unsigned long long)tab_w[i]); return 2; }
      fprintf(stderr, "table complete for all primes in [7, %llu]\n", (unsigned long long)pmax);
    }

    /* --- 5. prime index table for l <= sqrtN ---
     * pidx_small[l] = position of l among the primes <= pidx_max, or 0xffffffff if l is not prime.
     * The q-caches are allocated with pidx_max + 2 slots, more than the number of such primes. */
    pidx_max = sqrtN + 1; pidx_small = malloc((pidx_max + 1) * sizeof(u32));
    { u32 idx = 0; for (u64 l = 0; l <= pidx_max; l++) { if (l >= 2 && spf[l] == l) pidx_small[l] = idx++; else pidx_small[l] = 0xffffffffu; } }

    /* --- 6. self-check of the O(p) w(p^2) method against direct products ---
     * Every prime 5 <= p <= 1500, plus the four larger primes in extra[] when they are <= N/2.
     * 60013 and 65521 lie above the old mont_redc overflow threshold (p ~ 58000).  The direct
     * products for the large primes take most of the ~40 s this step needs. */
    { u64 bad = 0, cnt = 0; static const u64 extra[] = {4099, 16411, 60013, 65521};
      for (u64 i = 0; i < nprimes && (primes[i] <= 1500 || cnt < 4); i++) {
          u64 p = primes[i]; if (p < 5) continue; if (p > 1500) { int hit = 0; for (int j = 0; j < 4; j++) if (extra[j] == p) hit = 1; if (!hit) continue; cnt++; }
          u64 a = compute_wp2_fast(p), b = compute_wpk_direct(p, 2);
          if (a != b) { bad++; fprintf(stderr, "w(p^2) SELF-CHECK MISMATCH p=%llu fast=%llu direct=%llu\n", (unsigned long long)p, (unsigned long long)a, (unsigned long long)b); }
      }
      if (bad) { fprintf(stderr, "FATAL: w(p^2) fast method disagrees with direct products\n"); return 2; }
      fprintf(stderr, "w(p^2) fast method agrees with direct products for all 5 <= p <= 1500 and p = 4099, 16411, 60013, 65521  (%.1f s)\n", difftime(time(NULL), t0)); }

    /* --- 7. generalised Wilson quotients for prime powers p^k <= N, k >= 2 (parallel over p) ---
     * List every prime power p^k <= N with k >= 2 (the "if (pk > N / p) break" stops before
     * p^(k+1) could overflow), compute their w in nthreads threads, then sort by p^k. */
    { nwpk = 0; u64 cap = 1 << 16; wpk = malloc(cap * sizeof(wpk_t));
      for (u64 i = 0; i < nprimes && (u64)primes[i] * primes[i] <= N; i++) {
          u64 p = primes[i], pk = p * p; u32 k = 2;
          while (pk <= N) { if (nwpk == cap) { cap *= 2; wpk = realloc(wpk, cap * sizeof(wpk_t)); } wpk[nwpk].p = (u32)p; wpk[nwpk].k = k; wpk[nwpk].pk = pk; wpk[nwpk].w = 0; nwpk++; if (pk > N / p) break; pk *= p; k++; }
      }
      /* parallel compute (dynamic) */
      atomic_store(&wpk_next, 0);
      pthread_t th[256]; for (int t = 0; t < nthreads; t++) pthread_create(&th[t], 0, wpk_worker, 0); for (int t = 0; t < nthreads; t++) pthread_join(th[t], 0);
      qsort(wpk, nwpk, sizeof(wpk_t), cmp_wpk);
      fprintf(stderr, "generalised Wilson quotients for %llu prime powers p^k <= N (k>=2)  (%.1f s)\n", (unsigned long long)nwpk, difftime(time(NULL), t0));
    }

    /* --- 8. optional single-number trace (-n) or formula-versus-definition cross-check (-check) --- */
    if (debug_n) { int ok = formula_test_n(debug_n, 1); printf("formula: %llu is %sa Wilson number; direct: %s\n", (unsigned long long)debug_n, ok ? "" : "NOT ", debug_n < 50000000 ? (direct_wilson(debug_n) ? "WILSON" : "not Wilson") : "(skipped, too large)"); return 0; }
    if (ncheck) {   /* randomized formula-vs-definition cross-check (also forces the known Wilson numbers) */
        /* single-threaded; each direct check costs O(n), hence the 3*10^7 cap.  The 11 entries of
           known[] are the Wilson numbers 5 .. 26921605; a mismatch is reported but does not stop
           the run or change the exit status. */
        u64 lim = N < 30000000 ? N : 30000000, bad = 0; srand(12345);
        static const u64 known[] = {5,13,563,5971,558771,1964215,8121909,12326713,23025711,24365727,26921605};
        for (u64 i = 0; i < ncheck + 11; i++) {
            u64 n = (i < 11) ? known[i] : 2 + (((u64)rand() << 31) ^ (u64)rand()) % (lim - 1);
            if (n > lim) continue;
            int a = formula_test_n(n, 0), b = direct_wilson(n);
            if (a != b) { bad++; fprintf(stderr, "CROSS-CHECK MISMATCH n=%llu formula=%d direct=%d\n", (unsigned long long)n, a, b); }
        }
        fprintf(stderr, "cross-check: %llu random n <= %llu plus 11 known terms vs direct definition: %llu mismatches  (%.0f s)\n", (unsigned long long)ncheck, (unsigned long long)lim, (unsigned long long)bad, difftime(time(NULL), t0));
    }
    /* --- 9. main search: pass 1 in nthreads threads, then pass 2 in this thread --- */
    { pthread_t th[256]; atomic_store(&next_chunk, 0);
      for (int t = 0; t < nthreads; t++) pthread_create(&th[t], 0, worker, 0);
      for (int t = 0; t < nthreads; t++) pthread_join(th[t], 0); }
    prime_power_pass();
    /* --- 10. results --- */
    /* n = 1 is a Wilson number by convention (P(1)=1, e(1)=+1: 1+1=2 == 0 mod 1) */
    qsort(found, nfound, sizeof(u64), cmp_u64);
    printf("SUMMARY N=%llu: %llu n tested at their largest prime power, %llu full tests, %llu Wilson numbers > 1 found  (%.0f s, %d threads)\n",
           (unsigned long long)N, (unsigned long long)atomic_load(&n_tested), (unsigned long long)atomic_load(&n_fulltests), (unsigned long long)nfound, difftime(time(NULL), t0), nthreads);
    printf("LIST 1");
    for (u64 i = 0; i < nfound; i++) printf(" %llu", (unsigned long long)found[i]);
    printf("\n");
    return 0;
}
