/*
 * wilson_brute.c - brute-force (p-1)! mod p^2 for p < 2^39, multithreaded.
 *
 * ================================================================================================
 * THEORY OF OPERATION
 * ================================================================================================
 *
 *    Purpose: an independent, definition-level check of Wilson quotients.  For a prime p,
 *    Wilson's theorem gives (p-1)! == -1 (mod p), so (p-1)! = -1 + w_p * p for an integer w_p
 *    (the Wilson quotient), and (p-1)! mod p^2 determines w_p mod p.  p is a Wilson prime
 *    (OEIS A007540: 5, 13, 563) iff w_p == 0 (mod p).  This program multiplies 1 * 2 * ... * (p-1)
 *    modulo p^2, one factor at a time, with no shortcuts.  It was used to spot-check the
 *    Costa-Gerbicz-Harvey program pw13 (at p ~ 10^6, 4*10^11 and 2.1*10^9) and therefore the
 *    Wilson-quotient tables that wilsonnum.c reads.
 *
 *    Representation.  p^2 can reach 2^78, beyond 64 bits, so a residue x mod p^2 is held as the
 *    pair (a, b) with x = a*p + b and 0 <= a, b < p.  Multiplying by a small factor i < p:
 *
 *        (a*p + b) * i = a*i*p + b*i,   b*i = c*p + d   (0 <= d < p),
 *        so  x*i == ((a*i + c) mod p) * p + d   (mod p^2).
 *
 *    Every intermediate value is below 2^79 and fits in unsigned __int128.  The divisions by p
 *    use a double-precision estimate of the quotient followed by an exact integer correction
 *    (divmod_p), which is much faster than a 128-bit division.
 *
 *    Parallelism.  [1, p-1] is split into nt contiguous ranges; each thread multiplies its range
 *    (mul_small per factor), and the nt partial products are combined with the general product
 *    mul_res:  (a*p + b)(a'*p + b') == (a*b' + a'*b)*p + b*b'  (mod p^2).
 *
 *    Cost.  p - 1 steps of two divmod_p each: about 30 s for p = 10^10 and about 20 minutes for
 *    p = 4*10^11 on 8 threads of an M1 Pro.
 *
 * USAGE
 *
 *    wilson_brute [-t threads] p [p ...]       (-t, if present, must come first; default 8)
 *
 * OUTPUT, one line per p
 *
 *    p  R  k=K     R = (p-1)! mod p^2 in decimal (up to 24 digits);
 *                  K = the integer with (p-1)! == -1 + K*p (mod p^2), reduced into (-p/2, p/2].
 *                  K == w_p (mod p); K = 0 iff p is a Wilson prime.  (pw13 tables store
 *                  w = ((p-1)! + 1)/p in [1, p], which is congruent to K mod p.)
 *
 * ERROR CONDITIONS
 *
 *    - p >= 2^39 is rejected ("p too large", exit status 1).
 *    - p is not tested for primality.  For composite p the output is still (p-1)! mod p^2 (which
 *      is 0 for most composites), and k has no meaning.
 *    - Not checked: p = 0 (p - 1 wraps around and the run never finishes) and p = 1 (meaningless
 *      output); a thread count outside 1..256 (0 threads silently reports (p-1)! = 1, more than
 *      256 overflows the thread arrays).
 *    - Numbers are read with strtoull, which stops at the first non-digit ("1e10" is read as 1,
 *      and a single argument "7 11" as 7 alone).  Pass each p as a separate plain decimal
 *      argument.
 *    - Exit status 0 otherwise.
 *
 * BUILD
 *
 *    cc -O2 -o wilson_brute wilson_brute.c -lpthread
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
typedef unsigned long long u64; typedef unsigned __int128 u128; typedef __int128 i128;
typedef struct { u64 a, b; } res_t;            /* a*p + b */
static u64 P; static double Pd;                /* the current modulus base p, and p as a double */

/*
 * divmod_p -- quotient and remainder of x by p.
 *   In:     x < 2^79; the globals P (2 <= P < 2^39) and Pd = (double)P.
 *   Out:    returns floor(x/P); *rem = x mod P.
 *   How:    the double-precision quotient is accurate to about 2^-52 relative error, so for the
 *           quotients that occur here (below 2^40) it is off by at most a unit or two; the two
 *           correction loops make the result exact.
 *   Errors: none detected; x >= 2^79 or P = 0 are outside the design range.
 */
static inline u64 divmod_p(u128 x, u64 *rem) {  /* x < 2^79: q = floor(x/P), rem = x mod P */
    u64 q = (u64)((double)x / Pd);
    i128 r = (i128)x - (i128)((u128)q * P);
    while (r >= (i128)P) { r -= P; q++; }
    while (r < 0)        { r += P; q--; }
    *rem = (u64)r; return q;
}

/*
 * mul_small -- multiply the residue x = a*p + b by a factor i < p, modulo p^2.
 *   In:     x with a, b < p; 1 <= i < p.
 *   Out:    x*i mod p^2 in the same (a, b) representation.
 *   How:    b*i = c*p + d, then the new a is (a*i + c) mod p and the new b is d (see the header).
 */
static inline res_t mul_small(res_t x, u64 i) {  /* (a*p+b)*i mod p^2, i < p */
    u64 d, c = divmod_p((u128)x.b * i, &d);      /* b*i = c*p + d */
    u64 f; divmod_p((u128)x.a * i + c, &f);      /* (a*i + c) mod p */
    return (res_t){ f, d };
}

/*
 * mul_res -- multiply two residues mod p^2 (used only to combine the threads' partial products).
 *   In:     x = a*p + b and y = a'*p + b' with all digits < p.
 *   Out:    x*y mod p^2 = ((a*b' + a'*b + c) mod p)*p + d, where b*b' = c*p + d; the a*a'*p^2 term
 *           vanishes modulo p^2.
 */
static res_t mul_res(res_t x, res_t y) {         /* general product mod p^2 */
    u64 d, c = divmod_p((u128)x.b * y.b, &d);
    u128 t = (u128)x.a * y.b + (u128)x.b * y.a + c;   /* < 2*2^78 + 2^39 */
    u64 f; divmod_p(t % P, &f);  /* t % P is exact 128-bit mod (rare, cheap here) */
    return (res_t){ f, d };
}

/*
 * job_t / worker -- thread body: the product of the integers i in [lo, hi), modulo p^2.
 *   In:     *arg = job with 1 <= lo <= hi <= p; the globals P and Pd are already set.
 *   Out:    j->out = that product as (a, b) (the residue 1 for an empty range); returns NULL.
 */
typedef struct { u64 lo, hi; res_t out; } job_t;
static void *worker(void *arg) { job_t *j = arg; res_t r = {0, 1}; for (u64 i = j->lo; i < j->hi; i++) r = mul_small(r, i); j->out = r; return NULL; }

/*
 * main -- for each p on the command line: split [1, p-1] over nt threads, combine the partial
 * products, and print (p-1)! mod p^2 and k (see OUTPUT).  The globals P and Pd are set per p
 * before any thread starts, and the threads are joined before the next p.
 *   Out:    exit status 0, or 1 at the first p >= 2^39 (earlier p have already been printed).
 */
int main(int argc, char **argv) {
    int nt = 8, a = 1;
    if (argc > 2 && !strcmp(argv[1], "-t")) { nt = atoi(argv[2]); a = 3; }
    for (; a < argc; a++) {
        P = strtoull(argv[a], 0, 10); Pd = (double)P;
        if (P >= (1ULL << 39)) { fprintf(stderr, "p too large (need p < 2^39)\n"); return 1; }
        pthread_t th[256]; job_t jb[256]; u64 n = P - 1;      /* i = 1..p-1 */
        for (int t = 0; t < nt; t++) { jb[t].lo = 1 + n * t / nt; jb[t].hi = 1 + n * (t + 1) / nt; pthread_create(&th[t], 0, worker, &jb[t]); }
        res_t r = {0, 1};
        for (int t = 0; t < nt; t++) { pthread_join(th[t], 0); r = mul_res(r, jb[t].out); }
        u128 R = (u128)r.a * P + r.b;                          /* (p-1)! mod p^2 */
        /* R = -1 + k p  =>  k = (R+1)/p mod p  */
        u128 R1 = (R + 1) % ((u128)P * P); long long k = (long long)(R1 / P); if (k > (long long)(P / 2)) k -= (long long)P;
        /* print R (up to 2^78 < 10^24) as a high part and an 18-digit low part */
        unsigned long long hi = (unsigned long long)(R / 1000000000000000000ULL), lo = (unsigned long long)(R % 1000000000000000000ULL);
        if (hi) printf("%llu %llu%018llu  k=%lld\n", P, hi, lo, k); else printf("%llu %llu  k=%lld\n", P, lo, k);
        fflush(stdout);
    }
    return 0;
}
