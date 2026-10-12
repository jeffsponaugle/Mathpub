/*
 * wilsonnum.c  --  search for Wilson numbers (OEIS A157250) up to N.
 *
 * n is a Wilson number if  P(n) + e(n) == 0 (mod n^2),  where P(n) is the product of
 * the units mod n in [1,n] (the "n-phi-torial", A001783) and e(n) = +1 if n has a
 * primitive root (n = 1,2,4, p^k, 2p^k with p odd), e(n) = -1 otherwise (Gauss).
 *
 * Reduction (derived and verified against direct products for all n <= 3000, see
 * NEXT-A157250-A197632.md): for p^k || n write n = p^k * m,  G_k = prod_{j<p^k, p!|j} j
 * (the Gauss factorial, == eps mod p^k with eps = -1 for odd p and for 2^1, 2^2, eps = +1
 * for 2^k, k >= 3), and  A(m) = prod_{l | m prime} l^(-phi(m)/(l-1)).  Then
 *
 *     P(n) == A(m)^phi(p^k) * G_k^phi(m)        (mod p^2k)     except p^k = 2^1,
 *     P(2m) == A(m) * (-1)^(phi(m)/2)           (mod 4)        for odd m >= 3.
 *
 * Writing G_k = eps + p^k w  (w = generalised Wilson quotient of p^k, = ((p-1)!+1)/p mod p
 * for k = 1) and using that the generalised Fermat quotient
 * q(x) = (x^phi(p^k) - 1)/p^k mod p^k is additive, this becomes
 *
 *     P(n) == eps^phi(m) * (1 + p^k * (eps*phi(m)*w - t))   (mod p^2k),
 *     t    =  sum_{l | m} (phi(m)/(l-1)) * q(l)              (mod p^k),
 *
 * so the p-component of the Wilson condition is a congruence mod p^k that needs only
 * w(p^k), phi(m), the prime factors l of m and the Fermat quotients q(l).  n is a Wilson
 * number iff every prime-power component passes.
 *
 * Search organisation: every n <= N is examined exactly once, at its largest prime power
 * P^k (P the largest prime factor of n): for each P and each m <= N/P^k with P !| m and all
 * prime factors of m below P, evaluate the P-component (probability ~1/P of passing); the
 * survivors get all their components tested.  Wilson quotients w_P for primes come from a
 * binary table of (p, w_p) uint64 pairs produced by the patched pw13 (Costa-Gerbicz-Harvey
 * search code); w(p^k) for k >= 2 are computed here directly.
 *
 * Limits of this version: N < 2^32 so that every modulus p^2k <= N^2 fits in 64 bits
 * (fixed-modulus Montgomery arithmetic); smallest-prime-factor table up to N/3 in RAM.
 *
 * usage: wilsonnum -N <N> -t table.bin [-t more.bin ...] [-j threads] [-b N0]
 *        -b N0 : only report n > N0 as new (n <= N0 are still computed and printed as "known")
 * output: one line per Wilson number found: "WILSON n  e=.. factors=.." plus a summary.
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

/* ---------- 64-bit Montgomery arithmetic, odd modulus ---------- */
typedef struct { u64 n, ninv, r2; } mont_t;       /* ninv = -n^{-1} mod 2^64, r2 = 2^128 mod n */
static void mont_init(mont_t *M, u64 n) {
    u64 inv = 1; for (int i = 0; i < 6; i++) inv *= 2 - n * inv;   /* n^{-1} mod 2^64 */
    M->n = n; M->ninv = (u64)0 - inv;
    u128 r = ((u128)1 << 64) % n; M->r2 = (u64)((r * r) % n);
}
static inline u64 mont_redc(const mont_t *M, u128 t) {   /* valid for any odd n < 2^64 and t < n * 2^64 */
    u64 m = (u64)t * M->ninv;
    u128 mn = (u128)m * M->n;
    /* (t + mn) >> 64 without forming the possibly-overflowing sum: low words cancel to 0 mod 2^64,
       with a carry iff the low word of t is nonzero */
    u128 u = (t >> 64) + (mn >> 64) + ((u64)t != 0);
    if (u >= M->n) u -= M->n;
    return (u64)u;
}
static inline u64 mont_mul(const mont_t *M, u64 a, u64 b) { return mont_redc(M, (u128)a * b); }
static inline u64 mont_to(const mont_t *M, u64 a) { return mont_mul(M, a % M->n, M->r2); }
static inline u64 mont_from(const mont_t *M, u64 a) { return mont_redc(M, a); }
static u64 mont_pow(const mont_t *M, u64 a_m, u64 e) {   /* a_m in Montgomery form; returns Montgomery form */
    u64 r = mont_to(M, 1);
    while (e) { if (e & 1) r = mont_mul(M, r, a_m); a_m = mont_mul(M, a_m, a_m); e >>= 1; }
    return r;
}
/* generalised Fermat quotient q(x) = (x^phi(p^k) - 1) / p^k  mod p^k, for odd p, modulus p^2k = M->n, pk = p^k */
static inline u64 fermat_q(const mont_t *M, u64 pk, u64 phipk, u64 x) {
    u64 y = mont_from(M, mont_pow(M, mont_to(M, x), phipk));      /* x^phi(p^k) mod p^2k, == 1 mod p^k */
    u64 d = (y >= 1) ? y - 1 : y + M->n - 1;
    return (d / pk) % pk;                                            /* exact: d is a multiple of pk */
}
/* same for p = 2, modulus 2^(2k): q(x) = (x^(2^(k-1)) - 1)/2^k mod 2^k, x odd, k >= 2 */
static inline u64 fermat_q2(int k, u64 x) {
    u64 mask = (2 * k >= 64) ? ~(u64)0 : (((u64)1 << (2 * k)) - 1);
    u64 y; x &= mask;
    y = x; for (int i = 0; i < k - 1; i++) y = (y * y) & mask;      /* x^(2^(k-1)) by repeated squaring */
    u64 d = (y - 1) & mask;
    return (d >> k) & (((u64)1 << k) - 1);
}

/* ---------- global data ---------- */
static u64 N;                 /* search bound */
static u64 debug_n = 0, ncheck = 0;
static u64 N0 = 500000000ULL; /* terms <= N0 are "known" */
static int nthreads = 8;
static u32 *spf;              /* smallest prime factor, index up to spf_max */
static u64 spf_max;
static u32 *primes; static u64 nprimes;      /* all primes <= N/2 */
static u64 *tab_p, *tab_w; static u64 tab_n; /* Wilson quotient table, sorted by p */
static u32 *pidx_small; static u64 pidx_max; /* prime -> index among primes <= sqrtN (for q caches) */
static u64 sqrtN;

/* generalised Wilson quotients for prime powers p^k, k >= 2, p^k <= N: sorted array of (p^k, w) */
typedef struct { u64 pk, w; u32 p; u32 k; } wpk_t;
static wpk_t *wpk; static u64 nwpk;

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
static u64 wilson_quotient_pk(u64 p, u32 k) {    /* w(p^k) for k >= 2 from precomputed array */
    u64 pk = 1; for (u32 i = 0; i < k; i++) pk *= p;
    u64 lo = 0, hi = nwpk;
    while (lo < hi) { u64 mid = (lo + hi) / 2; if (wpk[mid].pk < pk) lo = mid + 1; else hi = mid; }
    if (lo < nwpk && wpk[lo].pk == pk) return wpk[lo].w;
    fprintf(stderr, "FATAL: w(%llu^%u) not precomputed\n", (unsigned long long)p, k); exit(2);
}
/*
 * O(p) computation of w(p^2) for odd p >= 5:  G_2 = prod_{j<p^2, p!|j} j  (mod p^4).
 * Block decomposition: prod_{i=1}^{p-1} (cp+i) = (p-1)! * sum_r (cp)^r e_r, with e_r the elementary
 * symmetric polynomials of {1/i : 1<=i<=p-1} (p-integral).  Mod p^4 only r <= 3 survive, so
 *   G_2 = ((p-1)!)^p * prod_{c=0}^{p-1} (1 + c p e1 + c^2 p^2 e2 + c^3 p^3 e3)   (mod p^4),
 * e1 = S1, e2 = (S1^2 - S2)/2, e3 = (S1^3 - 3 S1 S2 + 2 S3)/6 with power sums S_r = sum 1/i^r (Newton),
 * inverses of 1..p-1 mod p^4 by the linear recurrence inv[i] = -floor(M/i) * inv[M mod i].
 * Requires p >= 5 (division by 6) and p^4 < 2^64 (p <= 65535) in this 64-bit version.
 */
static inline u64 addmod(u64 a, u64 b, u64 n) { return (a >= n - b) ? a - (n - b) : a + b; }   /* a,b < n, no overflow */
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
    { u64 i6 = 1; for (int t = 0; t < 6; t++) { } /* compute 6^{-1} mod p^4 via extended Euclid-free trick: 6 * x == 1 */
      u64 g = 6 % M4; u128 a = 1; /* find inverse by pow: 6^(phi(p^4)-1) */ u64 phi = M4 / p * (p - 1); a = mont_from(&M, mont_pow(&M, mont_to(&M, g), phi - 1)); inv6 = mont_to(&M, (u64)a); (void)i6; }
    u64 e1 = S1;
    u64 e2 = mont_mul(&M, addmod(mont_mul(&M, S1, S1), M4 - S2, M4), inv2);
    u64 s1s2 = mont_mul(&M, S1, S2), m3 = addmod(addmod(M4 - s1s2, M4 - s1s2, M4), M4 - s1s2, M4);   /* -3 S1 S2 */
    u64 t3 = addmod(addmod(mont_mul(&M, mont_mul(&M, S1, S1), S1), m3, M4), addmod(S3, S3, M4), M4);
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
/* direct Gauss factorial: G_k = prod_{j<p^k, p!|j} j mod p^2k; returns w with G_k = eps + p^k w (mod p^2k) */
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

/* ---------- factorisation of m <= spf_max via spf ---------- */
typedef struct { u32 l[16]; u32 e[16]; int cnt; u64 phi; u32 maxp; } fac_t;
static inline void factor_spf(u64 m, fac_t *f) {
    f->cnt = 0; f->phi = 1; f->maxp = 1;
    while (m > 1) {
        u32 l = spf[m]; u32 e = 0; u64 pe = 1;
        while (m % l == 0) { m /= l; e++; pe *= l; }
        f->l[f->cnt] = l; f->e[f->cnt] = e; f->cnt++;
        f->phi *= pe / l * (l - 1); if (l > f->maxp) f->maxp = l;
    }
}
static inline int e_of_n(const fac_t *fn) {       /* Gauss sign e(n) from the factorisation of n */
    if (fn->cnt == 0) return 1;                                      /* n = 1 */
    if (fn->cnt == 1) return (fn->l[0] == 2) ? (fn->e[0] <= 2 ? 1 : -1) : 1;   /* 2,4 -> +1; 8,16.. -> -1; p^k -> +1 */
    if (fn->cnt == 2 && fn->l[0] == 2 && fn->e[0] == 1) return 1;              /* 2 p^k */
    return -1;
}

/* ---------- per-thread Fermat-quotient cache for the current prime power ---------- */
typedef struct {
    mont_t M; u64 p, pk, phipk; u32 k; u64 stamp;
    u64 *q; u64 *qstamp;          /* indexed by small-prime index; valid if qstamp[i] == stamp */
} qcache_t;
static void qcache_set(qcache_t *c, u64 p, u32 k) {
    c->p = p; c->k = k; c->pk = 1; for (u32 i = 0; i < k; i++) c->pk *= p;
    c->phipk = c->pk / p * (p - 1);
    if (p != 2) mont_init(&c->M, c->pk * c->pk);
    c->stamp++;
}
static inline u64 qcache_get(qcache_t *c, u32 l) {   /* q_{p^k}(l) mod p^k, l prime < p (or any l coprime to p) */
    u32 i = (l <= pidx_max) ? pidx_small[l] : 0xffffffffu;
    if (i != 0xffffffffu && c->qstamp[i] == c->stamp) return c->q[i];
    u64 v = (c->p == 2) ? fermat_q2(c->k, l) : fermat_q(&c->M, c->pk, c->phipk, l);
    if (i != 0xffffffffu) { c->q[i] = v; c->qstamp[i] = c->stamp; }
    return v;
}

/*
 * Component test at p^k for n = p^k * m, given m's factorisation (p !| m) and w = w(p^k).
 * Returns 1 if P(n) == -e_n (mod p^2k).
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
    /* eps */
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

/* full test of n = (P^k) * m with factorisation of m known; P^k is the largest prime power */
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
        u64 mm = n / pk; fac_t fmm; fmm.cnt = 0; fmm.phi = 1; fmm.maxp = 1;
        for (int j = 0; j < fs.cnt; j++) if (j != i) { fmm.l[fmm.cnt] = fs.l[j]; fmm.e[fmm.cnt] = fs.e[j]; fmm.cnt++; u64 pe = 1; for (u32 q = 0; q < fs.e[j]; q++) pe *= fs.l[j]; fmm.phi *= pe / fs.l[j] * (fs.l[j] - 1); if (fs.l[j] > fmm.maxp) fmm.maxp = fs.l[j]; }
        u64 w = (k == 1) ? wilson_quotient_prime(p) : wilson_quotient_pk(p, k);
        qcache_t cc = *c; cc.q = c->q; cc.qstamp = c->qstamp; qcache_set(&cc, p, k);
        int ok = component_ok(&cc, &fmm, mm, w, e_n);
        c->stamp = cc.stamp;                        /* keep stamps monotone */
        if (verbose) fprintf(stderr, "   n=%llu component %llu^%u (m=%llu, w=%llu): %s\n", (unsigned long long)n, (unsigned long long)p, k, (unsigned long long)mm, (unsigned long long)w, ok ? "pass" : "FAIL");
        if (!ok) { c->stamp++; return 0; }          /* fresh stamp: cache entries written here belong to other prime powers */
    }
    c->stamp++;
    return 1;
}
/* direct definition check for small n (n^2 < 2^64): P(n) mod n^2 and e(n) */
static int direct_wilson(u64 n) {
    u64 n2 = n * n, r = 1;
    for (u64 a = 1; a <= n; a++) { u64 x = a, y = n; while (y) { u64 t = x % y; x = y; y = t; } if (x == 1) r = (u64)(((u128)r * a) % n2); }
    /* e(n) */
    u64 m = n; int two = 0; while (m % 2 == 0) { m /= 2; two++; }
    int e; if (n == 1 || n == 2 || n == 4) e = 1; else if (two >= 2) e = -1; else { u64 q = m, d = 3; while (d * d <= q && q % d) d += 2; u64 pp = (d * d <= q) ? d : q; while (q % pp == 0) q /= pp; e = (q == 1) ? 1 : -1; }
    return ((r + (e > 0 ? 1 : n2 - 1)) % n2) == 0;
}
/* formula test of an arbitrary n (factored by trial division), verbose; returns 1 if Wilson */
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

/* ---------- results ---------- */
static pthread_mutex_t out_lock = PTHREAD_MUTEX_INITIALIZER;
static u64 *found; static u64 nfound, cap_found;
static atomic_ullong n_tested, n_fulltests;
static void report(u64 n, int known) {
    pthread_mutex_lock(&out_lock);
    if (nfound == cap_found) { cap_found = cap_found ? 2 * cap_found : 64; found = realloc(found, cap_found * sizeof(u64)); }
    found[nfound++] = n;
    printf("WILSON %llu%s\n", (unsigned long long)n, known ? "  (known)" : "  *** NEW ***");
    fflush(stdout);
    pthread_mutex_unlock(&out_lock);
}

/* ---------- worker: primes P (largest prime factor), k = 1 ---------- */
static atomic_ullong next_chunk;
#define CHUNK 256
static void *worker(void *arg) {
    (void)arg;
    qcache_t c; memset(&c, 0, sizeof c);
    c.q = malloc((pidx_max + 2) * sizeof(u64)); c.qstamp = calloc(pidx_max + 2, sizeof(u64)); c.stamp = 1;
    /* reuse: pidx_small has size for primes <= sqrtN, count = number of such primes; allocate by count */
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

/* ---------- prime powers P^k, k >= 2, as the largest prime power (P <= sqrt N) ---------- */
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

static int slow_wpk = 0;
static u64 compute_wpk(u64 p, u32 k) {
    if (!slow_wpk && k == 2 && p >= 5 && p <= 65535) return compute_wp2_fast(p);
    return compute_wpk_direct(p, k);
}
static atomic_ullong wpk_next;
static void *wpk_worker(void *a) { (void)a; for (;;) { u64 i = atomic_fetch_add(&wpk_next, 1); if (i >= nwpk) break; wpk[i].w = compute_wpk(wpk[i].p, wpk[i].k); } return NULL; }
static int cmp_u64(const void *a, const void *b) { u64 x = *(const u64 *)a, y = *(const u64 *)b; return x < y ? -1 : x > y; }
static int cmp_wpk(const void *a, const void *b) { u64 x = ((const wpk_t *)a)->pk, y = ((const wpk_t *)b)->pk; return x < y ? -1 : x > y; }

int main(int argc, char **argv) {
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

    /* --- load Wilson quotient table --- */
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

    /* --- sieve primes <= N/2, spf <= N/3 --- */
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

    /* table completeness check: every prime 7 <= p <= N/2 must be present */
    { u64 missing = 0, ti = 0;
      for (u64 i = 0; i < nprimes; i++) { u64 p = primes[i]; if (p < 7) continue; while (ti < tab_n && tab_p[ti] < p) ti++; if (ti == tab_n || tab_p[ti] != p) { if (missing < 5) fprintf(stderr, "  table missing p=%llu\n", (unsigned long long)p); missing++; } }
      if (missing) { fprintf(stderr, "FATAL: %llu primes missing from table\n", (unsigned long long)missing); return 2; }
      /* and every table entry must be a prime with w < p */
      for (u64 i = 0; i < tab_n; i++) if ((tab_p[i] <= spf_max && spf[tab_p[i]] != tab_p[i])) { fprintf(stderr, "FATAL: bad table entry p=%llu w=%llu\n", (unsigned long long)tab_p[i], (unsigned long long)tab_w[i]); return 2; }
      fprintf(stderr, "table complete for all primes in [7, %llu]\n", (unsigned long long)pmax);
    }

    /* --- prime index table for l <= sqrtN --- */
    pidx_max = sqrtN + 1; pidx_small = malloc((pidx_max + 1) * sizeof(u32));
    { u32 idx = 0; for (u64 l = 0; l <= pidx_max; l++) { if (l >= 2 && spf[l] == l) pidx_small[l] = idx++; else pidx_small[l] = 0xffffffffu; } pidx_max = pidx_max;
      /* caches are allocated with pidx_max+2 entries indexed by idx (<= number of primes <= sqrtN <= pidx_max) */ }

    /* --- self-check of the O(p) w(p^2) method against direct products --- */
    { u64 bad = 0, cnt = 0; static const u64 extra[] = {4099, 16411, 60013, 65521};
      for (u64 i = 0; i < nprimes && (primes[i] <= 1500 || cnt < 4); i++) {
          u64 p = primes[i]; if (p < 5) continue; if (p > 1500) { int hit = 0; for (int j = 0; j < 4; j++) if (extra[j] == p) hit = 1; if (!hit) continue; cnt++; }
          u64 a = compute_wp2_fast(p), b = compute_wpk_direct(p, 2);
          if (a != b) { bad++; fprintf(stderr, "w(p^2) SELF-CHECK MISMATCH p=%llu fast=%llu direct=%llu\n", (unsigned long long)p, (unsigned long long)a, (unsigned long long)b); }
      }
      if (bad) { fprintf(stderr, "FATAL: w(p^2) fast method disagrees with direct products\n"); return 2; }
      fprintf(stderr, "w(p^2) fast method agrees with direct products for all 5 <= p <= 1500 and p = 4099, 16411, 60013, 65521  (%.1f s)\n", difftime(time(NULL), t0)); }
    /* --- generalised Wilson quotients for prime powers p^k <= N, k >= 2 (parallel over p) --- */
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

    if (debug_n) { int ok = formula_test_n(debug_n, 1); printf("formula: %llu is %sa Wilson number; direct: %s\n", (unsigned long long)debug_n, ok ? "" : "NOT ", debug_n < 50000000 ? (direct_wilson(debug_n) ? "WILSON" : "not Wilson") : "(skipped, too large)"); return 0; }
    if (ncheck) {   /* randomized formula-vs-definition cross-check (also forces the known Wilson numbers) */
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
    /* --- main search --- */
    { pthread_t th[256]; atomic_store(&next_chunk, 0);
      for (int t = 0; t < nthreads; t++) pthread_create(&th[t], 0, worker, 0);
      for (int t = 0; t < nthreads; t++) pthread_join(th[t], 0); }
    prime_power_pass();
    /* n = 1 is a Wilson number by convention (P(1)=1, e(1)=+1: 1+1=2 == 0 mod 1) */
    qsort(found, nfound, sizeof(u64), cmp_u64);
    printf("SUMMARY N=%llu: %llu n tested at their largest prime power, %llu full tests, %llu Wilson numbers > 1 found  (%.0f s, %d threads)\n",
           (unsigned long long)N, (unsigned long long)atomic_load(&n_tested), (unsigned long long)atomic_load(&n_fulltests), (unsigned long long)nfound, difftime(time(NULL), t0), nthreads);
    printf("LIST 1");
    for (u64 i = 0; i < nfound; i++) printf(" %llu", (unsigned long long)found[i]);
    printf("\n");
    return 0;
}
