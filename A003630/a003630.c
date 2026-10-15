/*
 * a003630.c -- check the "sum of three earlier terms" conjecture proposed
 *              for OEIS A003630 (inert rational primes in Q(sqrt(3)))
 * ========================================================================
 *
 * A003630: 5, 7, 17, 19, 29, 31, 41, 43, 53, 67, 79, 89, 101, 103, 113, ...
 * are the primes p with Kronecker(12, p) = -1, i.e. the primes p == 5 or 7
 * (mod 12) (comment by Jianing Song in the entry).
 *
 * THE CONJECTURE (draft edits #77-#82 by Yves Debeuret, Oct 2026)
 * ---------------------------------------------------------------
 *   "Conjecture: For n > 3, a(n) = 2 + a(n-1) (when a(n) is the larger
 *    member of a twin prime pair), or a(n) = a(i) + a(j) + a(k) for some
 *    distinct indices i, j, k < n, verified up to 9941."
 *
 * The sequence is increasing, so "distinct indices i, j, k < n" means
 * "three distinct terms below a(n)".  For a term N write
 *
 *     C3(N)    N = x + y + z with x < y < z, all three terms of A003630;
 *     TWIN(N)  N - 2 is also a term (and then a(n-1) = N - 2).
 *
 * The conjecture says that C3(N) or TWIN(N) holds for every term N > 17.
 * This program checks C3 for EVERY term in a range (and TWIN wherever C3
 * fails), so it verifies the conjecture and also the stronger statement
 * without the twin clause.
 *
 * USAGE
 * -----
 *   a003630 check [START] END [-t T] [-c CHUNK] [-S STATE] [-q]
 *       Check every term N with START <= N <= END (START defaults to 0).
 *       Prints each term where C3 fails, the records of s(N) (see 2. below)
 *       and a summary.  If START = 0 and END = 10^k, the number of primes
 *       seen is compared with the known pi(10^k).  -S FILE appends one line
 *       per finished chunk; rerunning the same command, or one with a
 *       larger END and the same CHUNK, skips the chunks already in FILE.
 *       -t threads (default: all CPUs), -c chunk size (default 10^10 with
 *       -S, else 10^10 shrunk by powers of ten to give >= 8 chunks per
 *       thread), -q no progress line.
 *   a003630 rep A [B]
 *       For each term N in [A, B] (B defaults to A) print the representation
 *       N = x + y + z with the largest possible z.
 *   a003630 selftest [-t T]
 *
 * Numbers may be written as decimal, 1e12, 10^12, 2^40, or sums and
 * differences of those (1e12+1e9).  Results go to stdout, progress to stderr.
 * Exit status: 0 success, 1 failure or counterexample, 2 usage error.
 *
 * Build:  make   (needs primesieve >= 11: brew install primesieve /
 *                 apt install libprimesieve-dev)
 *
 * THEORY OF OPERATION
 * -------------------
 * 1. Residues.  Every term is 5 or 7 (mod 12), so a sum of three terms is
 *    3, 5, 7 or 9 (mod 12) according to how many summands are 7 (mod 12).
 *    A term N == 5 can only be (5)+(5)+(7) and a term N == 7 only
 *    (5)+(7)+(7).  The residue pattern is forced, so C3(N) only asks for
 *    three distinct terms that add up to N; no class conditions are needed.
 *
 * 2. The witness s(N).  Among the representations of N take the one whose
 *    largest part z is as large as possible and put s(N) = N - z = x + y.
 *    With P2 = { x + y : x < y, both terms } and, for s in P2, ymin(s) =
 *    the least possible y,
 *
 *        s(N) = min { N - q : q a term, N - q in P2, ymin(N - q) < q },
 *
 *    and C3(N) holds iff that set is nonempty.  The candidates for q are
 *    the terms a(n-1), a(n-2), ... in decreasing order, and the first one
 *    that works gives s(N).
 *
 * 3. P2 contains every value that can occur, except 2, 10 and 14.  If q < N
 *    are terms, N - q is 0 or 10 (mod 12) when N == 5 and 0 or 2 (mod 12)
 *    when N == 7.  At startup the program tabulates ymin(s) for s <= SMAX
 *    (65536) and records which of these s are not in P2; the self-test
 *    asserts that they are exactly 2, 10 = 5+5 and 14 = 7+7, which can only
 *    be written with a repeated term.  ymin(s) < s, so ymin(s) < q holds
 *    automatically when s <= N/2.  Hence for every term N >= 2*SMAX
 *
 *        N == 5 (mod 12): s(N) = N - a(n-1), or N - a(n-2) if N - a(n-1) = 10;
 *        N == 7 (mod 12): s(N) = the first of N - a(n-1), N - a(n-2) and
 *                         N - a(n-3) that is neither 2 nor 14,
 *
 *    provided that value is at most SMAX.  So for large N the conjecture is
 *    really a statement about gaps between consecutive terms: it holds as
 *    long as no three consecutive gaps add up to more than SMAX (and SMAX
 *    could be raised as far as P2 is known), and the records of s(N) follow
 *    the record gaps of the sequence.
 *
 * 4. Enumeration.  Nearly all of the work is enumerating the primes, which
 *    primesieve does.  [START, END] is cut into chunks (default 10^10) on the
 *    fixed grid of multiples of CHUNK; threads take chunks from an atomic
 *    counter.  Each chunk starts sieving MARGIN = 65536 below its start, so
 *    the terms just before the chunk are known.  Primes arrive from
 *    primesieve_generate_next_primes() in batches of about a thousand; each
 *    batch is compacted to its terms (p % 12 in {5, 7}) behind the last NBACK
 *    terms of the previous batch, so a(n-1), a(n-2), ... are simply the
 *    preceding array entries and the check in 3. is a table lookup or two.
 *
 * 5. Exact fallback.  For N < 2*SMAX, and whenever the last NBACK terms do
 *    not give a witness <= SMAX, slow_witness() searches exactly: it walks s
 *    upward through the admissible residues, so q = N - s runs through all
 *    candidates in decreasing order, tests q with a deterministic
 *    Miller-Rabin (bases 2..37, valid for every 64-bit n) and computes ymin(s)
 *    directly for s > SMAX.  It stops with "C3 fails" once q <= N/3 (then
 *    x < y < q forces x + y < 2q <= N - q).  The summary reports how many
 *    terms needed it.
 *
 * 6. Checks.  `selftest` compares the first 55 terms and a(100), a(500),
 *    a(1000) with the OEIS; checks that {2, 10, 14} is the exceptional set; compares s(N)
 *    from a brute-force triple search for all terms below 20000 with
 *    slow_witness() and with the chunked scan; compares the fast path with
 *    slow_witness() term by term just above 2*SMAX, above 10^12 and above
 *    10^18 with chunk boundaries every few 10^4; compares chunked and
 *    unchunked scans with 1 and T threads; and counts pi(10^9).  A `check`
 *    from 0 to a power of ten compares its prime count with pi(10^k).
 *    verify_a003630.py recomputes everything below 10^8 independently.
 */

#define _GNU_SOURCE
#include <primesieve.h>
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef unsigned __int128 u128;

#define SMAX      65536u            /* ymin(s) is tabulated for s <= SMAX     */
#define NBACK     8                 /* previous terms the fast path looks at  */
#define FASTMIN   (2ull * SMAX)     /* the fast path is used for N >= FASTMIN */
#define SMALLT    (1u << 22)        /* termtab[] covers m < SMALLT            */
#define MARGIN    65536ull          /* a chunk starts sieving this far below  */
#define SLOW_CAP  100000000ull      /* slow_witness() gives up beyond this s  */
#define DEF_CHUNK 10000000000ull    /* default chunk size, 10^10              */

typedef unsigned long long ull;

static void die(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fputs("a003630: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    exit(1);
}

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

/* ---------------------------------------------------------------------- */
/* Number parsing: 123, 1e12, 10^12, 2^40, and +/- combinations of those.  */

static int parse_atom(const char **sp, uint64_t *out)
{
    const char *s = *sp;
    char *end, *e2;
    if (!isdigit((unsigned char)*s))
        return 0;
    errno = 0;
    ull a = strtoull(s, &end, 10);
    if (errno)
        return 0;
    if (*end == 'e' || *end == 'E' || *end == '^') {
        int pow10 = (*end != '^');
        if (!isdigit((unsigned char)end[1]))
            return 0;
        ull k = strtoull(end + 1, &e2, 10);
        u128 v = pow10 ? a : 1;
        for (ull i = 0; i < k; i++) {
            v *= pow10 ? 10 : a;
            if (v > UINT64_MAX)
                return 0;
        }
        a = (ull)v;
        end = e2;
    }
    *out = a;
    *sp = end;
    return 1;
}

static uint64_t parse_num(const char *s)
{
    const char *p = s;
    uint64_t total = 0, t;
    int neg = 0;
    for (;;) {
        if (!parse_atom(&p, &t))
            die("bad number '%s'", s);
        if (neg) {
            if (t > total)
                die("'%s' is negative", s);
            total -= t;
        } else {
            if (total + t < total)
                die("'%s' does not fit in 64 bits", s);
            total += t;
        }
        if (*p == 0)
            return total;
        if (*p != '+' && *p != '-')
            die("bad number '%s'", s);
        neg = (*p++ == '-');
    }
}

/* ---------------------------------------------------------------------- */
/* Deterministic Miller-Rabin for 64-bit n (bases 2..37 are enough below
 * 3.3e24, Sorenson & Webster 2017).                                        */

static inline uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m)
{
    return (uint64_t)((u128)a * b % m);
}

static uint64_t powmod(uint64_t b, uint64_t e, uint64_t m)
{
    uint64_t r = 1;
    b %= m;
    while (e) {
        if (e & 1)
            r = mulmod(r, b, m);
        b = mulmod(b, b, m);
        e >>= 1;
    }
    return r;
}

static int is_prime_u64(uint64_t n)
{
    static const uint64_t B[12] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    if (n < 2)
        return 0;
    for (int i = 0; i < 12; i++)
        if (n % B[i] == 0)
            return n == B[i];
    uint64_t d = n - 1;
    int r = 0;
    while (!(d & 1))
        d >>= 1, r++;
    for (int i = 0; i < 12; i++) {
        uint64_t x = powmod(B[i], d, n);
        if (x == 1 || x == n - 1)
            continue;
        int j;
        for (j = 1; j < r; j++) {
            x = mulmod(x, x, n);
            if (x == n - 1)
                break;
        }
        if (j == r)
            return 0;
    }
    return 1;
}

/* ---------------------------------------------------------------------- */
/* Small tables: the terms below SMALLT and ymin(s) for s <= SMAX.          */

static uint8_t *termtab;              /* termtab[m] = 1 iff m < SMALLT is a term */
static uint32_t ymin_tab[SMAX + 1];   /* least y in s = x + y (x < y terms), 0 = s not in P2 */
static uint32_t nonp2[64];            /* admissible s <= SMAX that are not in P2 */
static int n_nonp2;

static inline int term_class(uint64_t m)
{
    unsigned r = (unsigned)(m % 12);
    return r == 5 || r == 7;
}

static int is_term(uint64_t m)
{
    if (m < SMALLT)
        return termtab[m];
    return term_class(m) && is_prime_u64(m);
}

static void init_tables(void)
{
    uint8_t *comp = calloc(SMALLT, 1);
    termtab = calloc(SMALLT, 1);
    if (!comp || !termtab)
        die("out of memory");
    for (uint64_t i = 2; i * i < SMALLT; i++)
        if (!comp[i])
            for (uint64_t j = i * i; j < SMALLT; j += i)
                comp[j] = 1;
    for (uint64_t m = 5; m < SMALLT; m++)
        termtab[m] = !comp[m] && term_class(m);
    free(comp);

    /* A sum of two terms is 10, 0 or 2 (mod 12) (5+5, 5+7, 7+7), so only
     * those s can be in P2.  The largest x < s/2 that works gives ymin.     */
    for (uint32_t s = 1; s <= SMAX; s++) {
        unsigned r = s % 12;
        if (r != 0 && r != 2 && r != 10)
            continue;
        for (uint32_t x = (s - 1) / 2; x >= 5; x--)
            if (termtab[x] && termtab[s - x]) {
                ymin_tab[s] = s - x;
                break;
            }
        if (!ymin_tab[s] && n_nonp2 < 64)
            nonp2[n_nonp2++] = s;
    }
}

/* If s = x + y with x < y < below and x, y terms, return the least such y,
 * else 0.                                                                  */
static uint64_t pair_ymin(uint64_t s, uint64_t below)
{
    if (s <= SMAX) {
        uint64_t y = ymin_tab[s];
        return y && y < below ? y : 0;
    }
    for (uint64_t x = (s - 1) / 2; x >= 5; x--) {
        uint64_t y = s - x;
        if (y >= below)
            return 0;
        if (is_term(x) && is_term(y))
            return y;
    }
    return 0;
}

/* Exact s(N) for a term N (section 5).  Returns 1 and sets *sout if C3(N)
 * holds, 0 if C3(N) is false, -1 if it gave up because s passed SLOW_CAP.
 * The admissible s are 10, 12, 22, 24, 34, ... (N == 5 mod 12) and
 * 2, 12, 14, 24, 26, ... (N == 7 mod 12), i.e. steps alternating 2 and 10. */
static int slow_witness(uint64_t N, uint64_t *sout)
{
    int five = (N % 12 == 5);
    uint64_t s = five ? 10 : 2;
    int step10 = !five;
    for (;;) {
        if (s + 5 > N)
            return 0;
        uint64_t q = N - s;
        if (q <= N / 3)              /* 3q <= N: x < y < q cannot reach N - q */
            return 0;
        if (s > SLOW_CAP)
            return -1;
        if (is_term(q) && pair_ymin(s, q)) {
            *sout = s;
            return 1;
        }
        s += step10 ? 10 : 2;
        step10 = !step10;
    }
}

/* number of terms m with q <= m < N, i.e. k such that q = a(n-k) when N = a(n) */
static uint64_t terms_between(uint64_t q, uint64_t N)
{
    uint64_t k = 0;
    for (uint64_t m = q; m < N; m++)
        k += term_class(m) && is_term(m);
    return k;
}

/* ---------------------------------------------------------------------- */
/* Growable list of (n, v) pairs.                                           */

typedef struct { uint64_t n, v; } pair_t;
typedef struct { pair_t *a; size_t len, cap; } pvec;

static void pv_push(pvec *p, uint64_t n, uint64_t v)
{
    if (p->len == p->cap) {
        p->cap = p->cap ? 2 * p->cap : 16;
        p->a = realloc(p->a, p->cap * sizeof *p->a);
        if (!p->a)
            die("out of memory");
    }
    p->a[p->len++] = (pair_t){n, v};
}

static void pv_free(pvec *p)
{
    free(p->a);
    memset(p, 0, sizeof *p);
}

static int pv_equal(const pvec *a, const pvec *b)
{
    return a->len == b->len &&
           (a->len == 0 || !memcmp(a->a, b->a, a->len * sizeof *a->a));
}

/* ---------------------------------------------------------------------- */
/* One chunk [lo, hi) and its statistics.                                   */

typedef struct {
    uint64_t lo, hi;
    int done;
    uint64_t nprimes;            /* primes in [lo, hi), any residue          */
    uint64_t nterm[2];           /* terms == 5 / == 7 (mod 12)               */
    uint64_t ntwin;              /* terms N with N - 2 a term                */
    uint64_t nslow;              /* terms that needed slow_witness()         */
    uint64_t sum_s;              /* sum of s(N) over the terms with C3(N)    */
    uint64_t maxgap, maxgap_at;  /* largest a(n) - a(n-1), a(n) in [lo, hi) */
    uint64_t kcnt[NBACK + 2];    /* kcnt[k]: terms whose largest part z is a(n-k);
                                    kcnt[NBACK+1]: k > NBACK (kcnt[0] unused) */
    pvec rec;                   /* (N, s(N)) whenever s(N) beats the chunk's max so far */
    pvec fail;                   /* (N, TWIN(N)) for each N where C3 fails   */
    pvec unres;                  /* (N, 0) where slow_witness() gave up      */
} chunk_t;

static void chunk_free(chunk_t *c)
{
    pv_free(&c->rec);
    pv_free(&c->fail);
    pv_free(&c->unres);
}

/* first index i < n with a[i] >= x (n if none) */
static size_t lbound(const uint64_t *a, size_t n, uint64_t x)
{
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (a[mid] < x)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

typedef struct { uint64_t *buf; size_t cap; } scratch_t;

/* Check every term in [c->lo, c->hi) (sections 3-5).  If dump is non-NULL,
 * (N, s(N)) is appended for every term (s = 0 where C3 fails).             */
static void process_chunk(primesieve_iterator *it, chunk_t *c, scratch_t *sc, pvec *dump)
{
    const uint64_t lo = c->lo, hi = c->hi;
    uint64_t back[NBACK] = {0};      /* last NBACK terms seen, oldest first; 0 = none */
    uint64_t runmax = 0;

    primesieve_jump_to(it, lo > MARGIN ? lo - MARGIN : 0, hi);
    for (;;) {
        primesieve_generate_next_primes(it);
        if (it->is_error)
            die("primesieve error near %llu", (ull)lo);
        const uint64_t *P = it->primes;
        size_t n = it->size, end = n;
        int last = 0;
        if (P[n - 1] >= hi) {
            last = 1;
            end = lbound(P, n, hi);
        }
        c->nprimes += end - (P[0] >= lo ? 0 : lbound(P, end, lo));

        if (sc->cap < end + NBACK) {
            sc->cap = 2 * (end + NBACK);
            free(sc->buf);
            sc->buf = malloc(sc->cap * sizeof *sc->buf);
            if (!sc->buf)
                die("out of memory");
        }
        uint64_t *t = sc->buf;
        memcpy(t, back, sizeof back);
        size_t nt = NBACK;
        for (size_t j = 0; j < end; j++) {      /* branch-free compaction */
            uint64_t p = P[j];
            t[nt] = p;
            nt += (0xA0u >> (p % 12)) & 1;      /* bits 5 and 7 */
        }

        for (size_t i = NBACK; i < nt; i++) {
            const uint64_t N = t[i];
            if (N < lo)
                continue;
            c->nterm[N % 12 == 7]++;
            if (t[i - 1]) {
                uint64_t g = N - t[i - 1];
                c->ntwin += (g == 2);
                if (g > c->maxgap) {
                    c->maxgap = g;
                    c->maxgap_at = N;
                }
            }
            uint64_t s = 0, kk = 0;
            int ok = 0;
            if (N >= FASTMIN)
                for (int k = 1; k <= NBACK; k++) {
                    uint64_t d = N - t[i - k];  /* t = 0 gives d = N > SMAX */
                    if (d > SMAX)
                        break;
                    if (ymin_tab[d]) {
                        s = d;
                        kk = k;
                        ok = 1;
                        break;
                    }
                }
            if (!ok) {
                c->nslow++;
                int v = slow_witness(N, &s);
                if (v > 0) {
                    ok = 1;
                    kk = terms_between(N - s, N);
                } else if (v == 0)
                    pv_push(&c->fail, N, is_term(N - 2));
                else
                    pv_push(&c->unres, N, 0);
            }
            if (ok) {
                c->kcnt[kk <= NBACK ? kk : NBACK + 1]++;
                c->sum_s += s;
                if (s > runmax) {
                    runmax = s;
                    pv_push(&c->rec, N, s);
                }
            }
            if (dump)
                pv_push(dump, N, ok ? s : 0);
        }
        memcpy(back, t + nt - NBACK, sizeof back);
        if (last)
            break;
    }
}

/* ---------------------------------------------------------------------- */
/* State file: one line per finished chunk.                                 */

static void write_chunk_line(FILE *f, const chunk_t *c)
{
    fprintf(f, "chunk %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu",
            (ull)c->lo, (ull)c->hi, (ull)c->nprimes, (ull)c->nterm[0], (ull)c->nterm[1],
            (ull)c->ntwin, (ull)c->nslow, (ull)c->sum_s, (ull)c->maxgap, (ull)c->maxgap_at);
    fputs(" k", f);
    for (int k = 1; k <= NBACK + 1; k++)
        fprintf(f, " %llu", (ull)c->kcnt[k]);
    const pvec *lists[3] = {&c->rec, &c->fail, &c->unres};
    const char *names[3] = {"rec", "fail", "unres"};
    for (int l = 0; l < 3; l++) {
        fprintf(f, " %s %zu", names[l], lists[l]->len);
        for (size_t i = 0; i < lists[l]->len; i++)
            fprintf(f, " %llu %llu", (ull)lists[l]->a[i].n, (ull)lists[l]->a[i].v);
    }
    fputs(" end\n", f);
}

/* Parse one state line into *c (which must be zeroed).  Returns 1 if the
 * line is complete and well formed.                                        */
static int read_chunk_line(char *line, chunk_t *c)
{
    char *save = NULL, *tok = strtok_r(line, " \n", &save);
    uint64_t v[10];
    if (!tok || strcmp(tok, "chunk"))
        return 0;
    for (int i = 0; i < 10; i++) {
        if (!(tok = strtok_r(NULL, " \n", &save)) || !isdigit((unsigned char)*tok))
            return 0;
        v[i] = strtoull(tok, NULL, 10);
    }
    c->lo = v[0], c->hi = v[1], c->nprimes = v[2], c->nterm[0] = v[3], c->nterm[1] = v[4];
    c->ntwin = v[5], c->nslow = v[6], c->sum_s = v[7], c->maxgap = v[8], c->maxgap_at = v[9];
    if (!(tok = strtok_r(NULL, " \n", &save)) || strcmp(tok, "k"))
        return 0;
    for (int k = 1; k <= NBACK + 1; k++) {
        if (!(tok = strtok_r(NULL, " \n", &save)) || !isdigit((unsigned char)*tok))
            return 0;
        c->kcnt[k] = strtoull(tok, NULL, 10);
    }
    pvec *lists[3] = {&c->rec, &c->fail, &c->unres};
    const char *names[3] = {"rec", "fail", "unres"};
    for (int l = 0; l < 3; l++) {
        if (!(tok = strtok_r(NULL, " \n", &save)) || strcmp(tok, names[l]))
            return 0;
        if (!(tok = strtok_r(NULL, " \n", &save)))
            return 0;
        size_t m = strtoull(tok, NULL, 10);
        for (size_t i = 0; i < m; i++) {
            char *a = strtok_r(NULL, " \n", &save), *b = a ? strtok_r(NULL, " \n", &save) : NULL;
            if (!b)
                return 0;
            pv_push(lists[l], strtoull(a, NULL, 10), strtoull(b, NULL, 10));
        }
    }
    tok = strtok_r(NULL, " \n", &save);
    return tok && !strcmp(tok, "end");
}

/* ---------------------------------------------------------------------- */
/* Running a scan over [start, end] (inclusive) with T threads.             */

typedef struct {
    chunk_t *c;
    size_t nc;
    atomic_size_t next;
    atomic_size_t done_chunks;
    _Atomic uint64_t done_ints;
    FILE *sf;
    pthread_mutex_t sf_lock;
} run_t;

typedef struct {
    uint64_t start, end;         /* inclusive */
    uint64_t nprimes, nterm[2], ntwin, nslow, sum_s, maxgap, maxgap_at;
    uint64_t kcnt[NBACK + 2];
    pvec rec, fail, unres;
} summary_t;

static void summary_free(summary_t *S)
{
    pv_free(&S->rec);
    pv_free(&S->fail);
    pv_free(&S->unres);
}

static void *worker(void *arg)
{
    run_t *R = arg;
    primesieve_iterator it;
    scratch_t sc = {0};
    primesieve_init(&it);
    for (;;) {
        size_t k = atomic_fetch_add(&R->next, 1);
        if (k >= R->nc)
            break;
        chunk_t *c = &R->c[k];
        if (c->done)
            continue;
        process_chunk(&it, c, &sc, NULL);
        c->done = 1;
        if (R->sf) {
            pthread_mutex_lock(&R->sf_lock);
            write_chunk_line(R->sf, c);
            fflush(R->sf);
            pthread_mutex_unlock(&R->sf_lock);
        }
        atomic_fetch_add(&R->done_ints, c->hi - c->lo);
        atomic_fetch_add(&R->done_chunks, 1);
    }
    free(sc.buf);
    primesieve_free_iterator(&it);
    return NULL;
}

static void fmt_secs(double s, char *buf, size_t n)
{
    if (s < 120)
        snprintf(buf, n, "%.0fs", s);
    else if (s < 7200)
        snprintf(buf, n, "%.1fm", s / 60);
    else
        snprintf(buf, n, "%.1fh", s / 3600);
}

/* Scan [start, end] in chunks of `chunk` (on the grid of multiples of chunk)
 * and merge the results into *S.                                           */
static void run_scan(uint64_t start, uint64_t end, uint64_t chunk, int threads,
                     const char *state, int quiet, summary_t *S)
{
    if (end == UINT64_MAX)
        die("END must be below 2^64 - 1");
    const uint64_t hi = end + 1;
    run_t R;
    memset(&R, 0, sizeof R);
    R.nc = (size_t)((hi - 1) / chunk - start / chunk + 1);
    R.c = calloc(R.nc, sizeof *R.c);
    if (!R.c)
        die("out of memory");
    for (size_t k = 0; k < R.nc; k++) {
        uint64_t g = (start / chunk + k) * chunk;
        R.c[k].lo = g > start ? g : start;
        R.c[k].hi = hi - g > chunk ? g + chunk : hi;
    }

    size_t resumed = 0;
    uint64_t resumed_ints = 0;
    if (state) {
        FILE *f = fopen(state, "r");
        if (f) {
            char *line = NULL;
            size_t cap = 0;
            while (getline(&line, &cap, f) > 0) {
                chunk_t c;
                memset(&c, 0, sizeof c);
                if (!read_chunk_line(line, &c) || c.lo < start || c.lo >= hi) {
                    chunk_free(&c);
                    continue;
                }
                size_t k = (size_t)(c.lo / chunk - start / chunk);
                if (k < R.nc && R.c[k].lo == c.lo && R.c[k].hi == c.hi && !R.c[k].done) {
                    R.c[k] = c;
                    R.c[k].done = 1;
                    resumed++;
                    resumed_ints += c.hi - c.lo;
                } else
                    chunk_free(&c);
            }
            free(line);
            fclose(f);
        }
        if (!(R.sf = fopen(state, "a")))
            die("cannot open %s: %s", state, strerror(errno));
        if (resumed && !quiet)
            fprintf(stderr, "resuming: %zu of %zu chunks already in %s\n", resumed, R.nc, state);
    }

    pthread_mutex_init(&R.sf_lock, NULL);
    atomic_store(&R.done_ints, resumed_ints);
    atomic_store(&R.done_chunks, resumed);
    pthread_t *tid = malloc(threads * sizeof *tid);
    double t0 = now(), last = t0;
    int tty = isatty(2);
    for (int i = 0; i < threads; i++)
        if (pthread_create(&tid[i], NULL, worker, &R))
            die("pthread_create failed");
    while (!quiet && atomic_load(&R.done_chunks) < R.nc) {
        usleep(200000);
        double t = now();
        if (t - last < (tty ? 1.0 : 60.0))
            continue;
        last = t;
        uint64_t d = atomic_load(&R.done_ints);
        double frac = (double)d / (double)(hi - start);
        double rate = (double)(d - resumed_ints) / (t - t0);
        char el[32], eta[32];
        fmt_secs(t - t0, el, sizeof el);
        fmt_secs(rate > 0 ? (double)(hi - start - d) / rate : 0, eta, sizeof eta);
        fprintf(stderr, "%s[%5.1f%%] %zu/%zu chunks  %.3g numbers/s  elapsed %s  eta %s%s",
                tty ? "\r" : "", 100 * frac, atomic_load(&R.done_chunks), R.nc, rate, el, eta,
                tty ? "   " : "\n");
    }
    for (int i = 0; i < threads; i++)
        pthread_join(tid[i], NULL);
    if (!quiet && tty)
        fputc('\n', stderr);
    free(tid);
    if (R.sf)
        fclose(R.sf);
    pthread_mutex_destroy(&R.sf_lock);

    /* merge in chunk order: a global record is necessarily a chunk record */
    memset(S, 0, sizeof *S);
    S->start = start;
    S->end = end;
    uint64_t gmax = 0;
    for (size_t k = 0; k < R.nc; k++) {
        chunk_t *c = &R.c[k];
        S->nprimes += c->nprimes;
        S->nterm[0] += c->nterm[0];
        S->nterm[1] += c->nterm[1];
        S->ntwin += c->ntwin;
        S->nslow += c->nslow;
        S->sum_s += c->sum_s;
        for (int k = 1; k <= NBACK + 1; k++)
            S->kcnt[k] += c->kcnt[k];
        if (c->maxgap > S->maxgap) {
            S->maxgap = c->maxgap;
            S->maxgap_at = c->maxgap_at;
        }
        for (size_t i = 0; i < c->rec.len; i++)
            if (c->rec.a[i].v > gmax) {
                gmax = c->rec.a[i].v;
                pv_push(&S->rec, c->rec.a[i].n, c->rec.a[i].v);
            }
        for (size_t i = 0; i < c->fail.len; i++)
            pv_push(&S->fail, c->fail.a[i].n, c->fail.a[i].v);
        for (size_t i = 0; i < c->unres.len; i++)
            pv_push(&S->unres, c->unres.a[i].n, c->unres.a[i].v);
        chunk_free(c);
    }
    free(R.c);
}

static int summary_equal(const summary_t *a, const summary_t *b)
{
    return a->nprimes == b->nprimes && a->nterm[0] == b->nterm[0] && a->nterm[1] == b->nterm[1] &&
           a->ntwin == b->ntwin && a->nslow == b->nslow && a->sum_s == b->sum_s &&
           a->maxgap == b->maxgap && a->maxgap_at == b->maxgap_at &&
           !memcmp(a->kcnt, b->kcnt, sizeof a->kcnt) &&
           pv_equal(&a->rec, &b->rec) && pv_equal(&a->fail, &b->fail) && pv_equal(&a->unres, &b->unres);
}

/* ---------------------------------------------------------------------- */
/* Output.                                                                  */

/* pi(10^k), k = 0..19 (A006880) */
static const uint64_t PI10[20] = {
    0, 4, 25, 168, 1229, 9592, 78498, 664579, 5761455, 50847534, 455052511,
    4118054813ull, 37607912018ull, 346065536839ull, 3204941750802ull,
    29844570422669ull, 279238341033925ull, 2623557157654233ull,
    24739954287740860ull, 234057667276344607ull};

/* index n of a small term N (0 if N is too large for the table) */
static uint64_t term_index(uint64_t N)
{
    if (N >= SMALLT)
        return 0;
    uint64_t n = 0;
    for (uint64_t m = 5; m <= N; m++)
        n += termtab[m];
    return n;
}

/* print "x + y + z" for the canonical representation with witness s */
static void print_rep(FILE *f, uint64_t N, uint64_t s)
{
    uint64_t z = N - s, y = pair_ymin(s, z), x = s - y;
    fprintf(f, "%llu + %llu + %llu", (ull)x, (ull)y, (ull)z);
}

/* Print the summary; returns the number of counterexamples to the draft
 * conjecture plus unresolved terms (0 = everything verified).              */
static int print_summary(const summary_t *S, double secs)
{
    uint64_t nterms = S->nterm[0] + S->nterm[1];
    int bad = 0;
    printf("A003630 conjecture check: all terms N with %llu <= N <= %llu\n", (ull)S->start, (ull)S->end);
    printf("  primes in range          %llu", (ull)S->nprimes);
    if (S->start == 0) {
        for (int k = 1; k < 20; k++) {
            uint64_t p10 = 1;
            for (int i = 0; i < k; i++)
                p10 *= 10;
            if (p10 == S->end) {
                printf("   (pi(10^%d) = %llu: %s)", k, (ull)PI10[k], PI10[k] == S->nprimes ? "OK" : "MISMATCH");
                bad += PI10[k] != S->nprimes;
            }
        }
    }
    printf("\n  terms (5 / 7 mod 12)     %llu + %llu = %llu%s\n", (ull)S->nterm[0], (ull)S->nterm[1],
           (ull)nterms, S->start == 0 ? "   (so n <= this count)" : "");
    printf("  terms with N - 2 a term  %llu\n", (ull)S->ntwin);
    printf("  exact search needed      %llu terms\n", (ull)S->nslow);
    printf("  largest part = a(n-k)   ");
    for (int k = 1; k <= NBACK + 1; k++)
        if (S->kcnt[k])
            printf(" k%s%d: %llu", k > NBACK ? ">" : "=", k > NBACK ? NBACK : k, (ull)S->kcnt[k]);
    printf("\n");
    printf("  largest gap a(n)-a(n-1)  %llu (a(n) = %llu)\n", (ull)S->maxgap, (ull)S->maxgap_at);
    if (nterms > S->fail.len)
        printf("  mean s(N)                %.3f\n", (double)S->sum_s / (double)(nterms - S->fail.len));
    printf("  C3 fails for             %zu term%s", S->fail.len, S->fail.len == 1 ? "" : "s");
    for (size_t i = 0; i < S->fail.len && i < 20; i++) {
        uint64_t N = S->fail.a[i].n, n = term_index(N);
        printf("%s%llu", i ? ", " : ": ", (ull)N);
        if (n)
            printf(" (n=%llu%s)", (ull)n, S->fail.a[i].v ? ", twin" : "");
        else if (S->fail.a[i].v)
            printf(" (twin)");
        /* counterexample to the draft: n > 3 and no twin clause */
        if ((n == 0 || n > 3) && !S->fail.a[i].v)
            bad++;
    }
    printf("\n");
    if (S->unres.len) {
        printf("  UNRESOLVED (search gave up) %zu terms:", S->unres.len);
        for (size_t i = 0; i < S->unres.len && i < 20; i++)
            printf(" %llu", (ull)S->unres.a[i].n);
        printf("\n");
        bad += (int)S->unres.len;
    }
    printf("  records of s(N) = N - (largest part):\n");
    printf("  %22s %8s   representation\n", "N", "s(N)");
    for (size_t i = 0; i < S->rec.len; i++) {
        printf("  %22llu %8llu   ", (ull)S->rec.a[i].n, (ull)S->rec.a[i].v);
        print_rep(stdout, S->rec.a[i].n, S->rec.a[i].v);
        printf("\n");
    }
    if (bad)
        printf("RESULT: PROBLEM -- %d counterexample(s), unresolved term(s) or count mismatch above\n", bad);
    else {
        printf("RESULT: verified.  Every term N in the range is a sum of three distinct smaller terms");
        if (S->fail.len)
            printf(",\n        except %zu term%s with n <= 3 or N - 2 a term (listed above)", S->fail.len,
                   S->fail.len == 1 ? "" : "s");
        printf(".\n");
    }
    if (secs >= 0)
        printf("  time                     %.1f s\n", secs);
    fflush(stdout);
    return bad;
}

/* ---------------------------------------------------------------------- */
/* Self-test.                                                               */

static int nfail_tests;

static void expect(int cond, const char *what)
{
    printf("  %-66s %s\n", what, cond ? "ok" : "FAILED");
    fflush(stdout);
    if (!cond)
        nfail_tests++;
}

/* s(N) by brute force over triples x < y < z (z largest first) */
static uint64_t brute_s(uint64_t N, const uint64_t *T, size_t nT)
{
    for (size_t c = nT; c-- > 0;) {
        uint64_t z = T[c];
        if (z >= N)
            continue;
        uint64_t s = N - z;
        if (s >= 2 * z)
            return 0;
        for (size_t a = 0; a < nT && T[a] < s - T[a]; a++) {
            uint64_t y = s - T[a];
            if (y < z && termtab[y])
                return s;
        }
    }
    return 0;
}

/* scan [lo, hi) sequentially in chunks of `chunk`, dumping (N, s(N)) */
static void scan_dump(uint64_t lo, uint64_t hi, uint64_t chunk, pvec *dump, uint64_t *nslow)
{
    primesieve_iterator it;
    scratch_t sc = {0};
    primesieve_init(&it);
    *nslow = 0;
    for (uint64_t a = lo; a < hi; a += chunk) {
        chunk_t c;
        memset(&c, 0, sizeof c);
        c.lo = a;
        c.hi = hi - a > chunk ? a + chunk : hi;
        process_chunk(&it, &c, &sc, dump);
        *nslow += c.nslow;
        chunk_free(&c);
    }
    free(sc.buf);
    primesieve_free_iterator(&it);
}

/* fast path (chunked scan) against slow_witness() and an MR enumeration */
static int fast_vs_slow(uint64_t lo, uint64_t len, uint64_t chunk, uint64_t *nterms, uint64_t *nslow)
{
    pvec d = {0};
    scan_dump(lo, lo + len, chunk, &d, nslow);
    int ok = 1;
    size_t k = 0;
    for (uint64_t m = lo; m < lo + len && ok; m++) {
        if (!is_term(m))
            continue;
        uint64_t s;
        int v = slow_witness(m, &s);
        if (k >= d.len || d.a[k].n != m || (v > 0 ? s : 0) != d.a[k].v || v < 0)
            ok = 0;
        k++;
    }
    ok &= (k == d.len);
    *nterms = k;
    pv_free(&d);
    return ok;
}

static int selftest(int threads)
{
    static const uint64_t DATA[55] = {
        5, 7, 17, 19, 29, 31, 41, 43, 53, 67, 79, 89, 101, 103, 113, 127, 137, 139, 149,
        151, 163, 173, 197, 199, 211, 223, 233, 257, 269, 271, 281, 283, 293, 307, 317, 331,
        353, 367, 379, 389, 401, 439, 449, 461, 463, 487, 499, 509, 521, 523, 547, 557, 569,
        571, 593};
    char msg[160];
    double t0 = now();

    printf("selftest (%d threads)\n", threads);

    /* 1. terms against the OEIS DATA section and the b-file */
    {
        primesieve_iterator it;
        primesieve_init(&it);
        int ok = 1;
        uint64_t n = 0, p, a[3] = {0};
        while (n < 1000 && (p = primesieve_next_prime(&it))) {
            if (!term_class(p))
                continue;
            if (n < 55 && p != DATA[n])
                ok = 0;
            n++;
            if (n == 100 || n == 500 || n == 1000)
                a[n == 100 ? 0 : n == 500 ? 1 : 2] = p;
        }
        primesieve_free_iterator(&it);
        expect(ok, "a(1..55) = DATA section of A003630");
        snprintf(msg, sizeof msg, "a(100), a(500), a(1000) = %llu, %llu, %llu (b-file: 1181, 7867, 17299)",
                 (ull)a[0], (ull)a[1], (ull)a[2]);
        expect(a[0] == 1181 && a[1] == 7867 && a[2] == 17299, msg);
    }

    /* 2. admissible s <= SMAX that are not sums of two distinct terms */
    {
        snprintf(msg, sizeof msg, "s <= %u not in P2: exactly 2, 10, 14 (found %d)", SMAX, n_nonp2);
        expect(n_nonp2 == 3 && nonp2[0] == 2 && nonp2[1] == 10 && nonp2[2] == 14, msg);
    }

    /* 3. brute force below BF: triple search vs slow_witness vs chunked scans */
    {
        const uint64_t BF = 20000;
        uint64_t *T = malloc(BF * sizeof *T);
        size_t nT = 0;
        for (uint64_t m = 5; m < BF; m++)
            if (termtab[m])
                T[nT++] = m;
        int ok_slow = 1, nf = 0;
        uint64_t fails[8];
        for (size_t i = 0; i < nT; i++) {
            uint64_t b = brute_s(T[i], T, nT), s = 0;
            int v = slow_witness(T[i], &s);
            if ((v > 0 ? s : 0) != b || v < 0)
                ok_slow = 0;
            if (!b && nf < 8)
                fails[nf++] = T[i];
        }
        snprintf(msg, sizeof msg, "brute force = slow_witness for all %zu terms < %llu", nT, (ull)BF);
        expect(ok_slow, msg);
        expect(nf == 4 && fails[0] == 5 && fails[1] == 7 && fails[2] == 17 && fails[3] == 19,
               "C3 fails below 20000 exactly for 5, 7, 17, 19");
        uint64_t chunks[3] = {BF, 997, 64}, nslow;
        for (int j = 0; j < 3; j++) {
            pvec d = {0};
            scan_dump(0, BF, chunks[j], &d, &nslow);
            int ok = (d.len == nT);
            for (size_t i = 0; ok && i < nT; i++)
                ok = (d.a[i].n == T[i] && d.a[i].v == brute_s(T[i], T, nT));
            snprintf(msg, sizeof msg, "chunked scan (chunk %llu) = brute force below %llu", (ull)chunks[j], (ull)BF);
            expect(ok, msg);
            pv_free(&d);
        }
        free(T);
    }

    /* 4. fast path against slow_witness at several heights, many chunk starts */
    {
        struct { uint64_t lo, len, chunk; } R[4] = {
            {FASTMIN - 3000, 2000000, 50000},
            {1000000000000ull, 2000000, 30011},
            {1000000000000000000ull, 300000, 20000},
            {18446744073709000000ull, 400000, 70000},
        };
        for (int j = 0; j < 4; j++) {
            uint64_t nterms, nslow;
            int ok = fast_vs_slow(R[j].lo, R[j].len, R[j].chunk, &nterms, &nslow);
            snprintf(msg, sizeof msg, "fast = exact on [%llu, +%llu): %llu terms, %llu exact", (ull)R[j].lo,
                     (ull)R[j].len, (ull)nterms, (ull)nslow);
            expect(ok && nterms > 0, msg);
        }
    }

    /* 5. chunking and threading do not change the merged result */
    {
        summary_t A, B, C;
        run_scan(0, 30000000, 30000001, 1, NULL, 1, &A);
        run_scan(0, 30000000, 1000007, threads, NULL, 1, &B);
        run_scan(0, 30000000, 65537, threads, NULL, 1, &C);
        expect(summary_equal(&A, &B) && summary_equal(&A, &C),
               "[0, 3e7]: 1 chunk / 1 thread = 30 chunks = 458 chunks, T threads");
        expect(A.fail.len == 4 && A.unres.len == 0, "  ... and C3 fails only for 5, 7, 17, 19");
        summary_free(&A);
        summary_free(&B);
        summary_free(&C);
    }

    /* 6. state file round trip: half the chunks from a file, then all */
    {
        char path[1024];
        const char *tmp = getenv("TMPDIR");
        snprintf(path, sizeof path, "%s/a003630_selftest_XXXXXX", tmp && *tmp ? tmp : "/tmp");
        int fd = mkstemp(path);
        if (fd < 0)
            die("mkstemp failed");
        close(fd);
        summary_t A, B, C;
        run_scan(0, 20000000, 1000000, threads, path, 1, &A);   /* writes 20 chunks */
        run_scan(0, 40000000, 1000000, threads, path, 1, &B);   /* resumes 20, adds 20 */
        run_scan(0, 40000000, 40000001, 1, NULL, 1, &C);
        expect(summary_equal(&B, &C), "state file: resumed + extended run = fresh run");
        unlink(path);
        summary_free(&A);
        summary_free(&B);
        summary_free(&C);
    }

    /* 7. pi(10^9) */
    {
        summary_t A;
        run_scan(0, 1000000000, 100000000, threads, NULL, 1, &A);
        snprintf(msg, sizeof msg, "pi(10^9) = %llu (expected 50847534), %llu terms", (ull)A.nprimes,
                 (ull)(A.nterm[0] + A.nterm[1]));
        expect(A.nprimes == 50847534 && A.fail.len == 4 && A.unres.len == 0, msg);
        summary_free(&A);
    }

    printf("selftest %s (%d failure%s, %.1f s)\n", nfail_tests ? "FAILED" : "passed", nfail_tests,
           nfail_tests == 1 ? "" : "s", now() - t0);
    return nfail_tests ? 1 : 0;
}

/* ---------------------------------------------------------------------- */

static void usage(void)
{
    fputs("usage: a003630 check [START] END [-t T] [-c CHUNK] [-S STATE] [-q]\n"
          "       a003630 rep A [B]\n"
          "       a003630 selftest [-t T]\n", stderr);
    exit(2);
}

int main(int argc, char **argv)
{
    if (argc < 2)
        usage();
    const char *cmd = argv[1];
    int threads = (int)sysconf(_SC_NPROCESSORS_ONLN), quiet = 0;
    uint64_t chunk = DEF_CHUNK;
    int chunk_set = 0;
    const char *state = NULL;
    uint64_t pos[2];
    int npos = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "-t") && i + 1 < argc)
            threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c") && i + 1 < argc)
            chunk = parse_num(argv[++i]), chunk_set = 1;
        else if (!strcmp(argv[i], "-S") && i + 1 < argc)
            state = argv[++i];
        else if (!strcmp(argv[i], "-q"))
            quiet = 1;
        else if (argv[i][0] != '-' && npos < 2)
            pos[npos++] = parse_num(argv[i]);
        else
            usage();
    }
    if (threads < 1 || chunk < 1)
        usage();
    init_tables();

    if (!strcmp(cmd, "selftest"))
        return selftest(threads);

    if (!strcmp(cmd, "rep")) {
        if (npos < 1)
            usage();
        uint64_t a = pos[0], b = npos > 1 ? pos[1] : pos[0];
        for (uint64_t N = a; N <= b && N >= a; N++) {
            if (!is_term(N))
                continue;
            uint64_t s;
            int v = slow_witness(N, &s);
            printf("%llu = ", (ull)N);
            if (v > 0)
                print_rep(stdout, N, s);
            else if (v == 0)
                printf("(no three distinct smaller terms)%s", is_term(N - 2) ? "   [N - 2 is a term]" : "");
            else
                printf("(search gave up)");
            printf("\n");
        }
        return 0;
    }

    if (!strcmp(cmd, "check")) {
        if (npos < 1)
            usage();
        uint64_t start = npos > 1 ? pos[0] : 0, end = npos > 1 ? pos[1] : pos[0];
        if (start > end)
            usage();
        /* Without -c and -S, shrink the chunk (by powers of ten, down to 10^6)
         * until there are at least 8 chunks per thread.  With -S the grid stays
         * at 10^10 so that a later run with a larger END can resume.          */
        if (!chunk_set && !state)
            while (chunk > 1000000 && (end - start) / chunk < 8 * (uint64_t)threads)
                chunk /= 10;
        if (!quiet)
            fprintf(stderr, "checking [%llu, %llu], %d threads, chunk %llu%s%s\n", (ull)start, (ull)end,
                    threads, (ull)chunk, state ? ", state " : "", state ? state : "");
        summary_t S;
        double t0 = now();
        run_scan(start, end, chunk, threads, state, quiet, &S);
        int bad = print_summary(&S, now() - t0);
        summary_free(&S);
        return bad ? 1 : 0;
    }
    usage();
}
