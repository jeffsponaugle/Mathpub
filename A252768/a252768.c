/*
 * a252768.c -- compute and extend OEIS A252768
 * ============================================
 *
 *   "Primes p with property that the sum of the k-th powers of the successive
 *    gaps between primes <= p are prime numbers for k = 1 to n."
 *
 * Terms: a(1..7) = 5, 5, 13, 14593, 372313, 2315773, 541613713 (OEIS, 2014);
 *        a(8) = 7952072743 and a(9) = 21814967833 were found with this program
 *        (Sep 16 2026, in the OEIS since Sep 17 2026); a(10) > 10^12.
 *
 * DEFINITIONS
 * -----------
 * Let 2 = q_1 < q_2 < ... < q_m = p be the primes up to p and g_i = q_(i+1) - q_i
 * their gaps (g_1 = 1, every other gap is even).  Write
 *
 *     S_k(p) = sum_i g_i^k             (k = 1, 2, ...).
 *
 * The DEPTH of p is the largest n such that S_1(p), ..., S_n(p) are all prime;
 * a(n) is the smallest prime of depth >= n.  Since S_1(p) = p - 2, every term is
 * the larger member of a twin prime pair (A006512), and a(n) is non-decreasing.
 *
 * Example: p = 13, gaps 1,2,2,4,2: S_1 = 11, S_2 = 29, S_3 = 89 are prime and
 * S_4 = 1+16+16+256+16 = 305 = 5*61 is not, so 13 has depth 3 and a(3) = 13.
 *
 * USAGE
 * -----
 *   a252768 scan [START] END [-t T] [-c CHUNK] [-k KMAX] [-r DEPTH] [-S STATE] [-i SECS] [-q]
 *       Exhaustive scan of the primes in [START, END] (START defaults to 0; with
 *       START > 0 the sums up to START are first computed in a totals-only pass).
 *       Prints every prime of depth >= DEPTH (default 7) as it is found, then a
 *       table of a(n).  -S FILE keeps a checkpoint (every SECS seconds, default
 *       60, and on Ctrl-C); rerunning the same command resumes from it.
 *   a252768 verify P [-k KMAX] [-t T]
 *       Recompute S_1..S_KMAX at the prime P from scratch and test each one
 *       (also with GMP when compiled in).  -t 1 (default) is a plain
 *       single-threaded accumulation; -t T uses the parallel totals pipeline.
 *   a252768 selftest [-t T]
 *
 * Numbers may be written as decimal, 2^k, 10^k, 1e12, or X+Y / X-Y of those.
 * Results go to stdout (one line per hit, then a summary), progress and
 * diagnostics to stderr.  Exit status: 0 success, 1 fatal error, 2 usage error.
 *
 * Build:  cc -O2 -std=gnu11 -pthread -I/opt/homebrew/include a252768.c \
 *            -L/opt/homebrew/lib -lprimesieve -lm [-DHAVE_GMP -lgmp] -o a252768
 *
 * THEORY OF OPERATION
 * -------------------
 *
 * 1. What is tracked.  The scan keeps S_k for k = 1..KMAX (default 12, at most
 *    KLIM = 16) as unsigned 128-bit integers.  S_1(p) = p - 2 exactly, which
 *    gives both the cheapest possible first filter (p can only qualify if the
 *    gap into p is 2) and a built-in consistency check of the bookkeeping
 *    (check_chunk verifies the published S_1 against the actual previous prime).
 *
 * 2. Making a sequential sum parallel.  S_k(p) depends on every gap below p, so
 *    the obvious loop "for each prime: add the powers of its gap, test" is one
 *    long sequential pass.  The tool cuts [start, end] into chunks of `chunk`
 *    integers; chunk i covers [start + i*chunk, start + (i+1)*chunk).  Every
 *    prime q except 2 has exactly one INCOMING gap q - prevprime(q), and that gap
 *    is attributed to the chunk that contains q.  Consequently the chunk total
 *    T_k(i) = sum of g^k over the incoming gaps of chunk i depends on the chunk
 *    alone, and the running sum at the start of chunk i is the prefix sum
 *    O_k(i) = O_k(0) + T_k(0) + ... + T_k(i-1), where O_k(0) is the sum at `start`
 *    (0 for a scan from 0, otherwise taken from a checkpoint or a totals pass).
 *
 *    Each worker thread (worker_main) repeats three steps per chunk:
 *      a) SIEVE (sieve_chunk): primesieve enumerates the primes of the chunk,
 *         plus the prime just below it, which is found by starting MARGIN below
 *         lo.  The incoming gaps are stored one byte each (gap/2, with an escape
 *         for the gap 1 and gaps >= 510) and counted in a histogram cnt[g], from
 *         which T_k(i) = sum_g cnt[g]*g^k is formed (chunk_totals).
 *      b) HAND-OFF: under the scan mutex the worker waits until sc->pub == i,
 *         i.e. until sc->O holds O_k(i).  That happens as soon as chunk i-1 has
 *         finished its own SIEVE (not its check), so the wait is short.  The
 *         worker copies O_k(i), adds its totals so that sc->O becomes O_k(i+1),
 *         records O_k(i) in the ring slot of chunk i for checkpoints, sets
 *         pub = i+1 and wakes the other threads.  Chunk indices come from an
 *         atomic counter in increasing order and nobody blocks while sieving,
 *         so the chain always advances: no deadlock, no barrier.
 *      c) CHECK (check_chunk): with O_k(i) known, the worker walks its stored
 *         gaps once more, now knowing the exact S_k at every prime of the chunk,
 *         and tests the candidates.  This needs no synchronisation, so the
 *         expensive part runs fully in parallel.
 *    The range is sieved exactly once and memory is one chunk of gap bytes per
 *    thread (about 40 MB at chunk = 10^9).
 *
 * 3. Bookkeeping ring and completion frontier.  sc->cf is the lowest chunk
 *    index whose check has not finished; everything below it is provably done,
 *    so checkpoints are written at cf and "a(n) > X" claims refer to it.  A ring
 *    of RING slots indexed by chunk mod RING holds each chunk's done flag and
 *    its O_k(i) record.  A worker may not take chunk i before i < cf + RING, so
 *    a slot is never reused while its chunk is live.  A stale record is
 *    recognised by its idx field.
 *
 * 4. Testing a prime (check_chunk).  Only primes whose incoming gap is 2 are
 *    tested (S_1 = p - 2 prime).  S_2, S_3, S_4 are maintained incrementally
 *    from the power tables PW[k][g] (three 128-bit adds per prime) and tested in
 *    turn; most candidates fail at S_2.  For the roughly 1 in 10^5 primes with
 *    S_2..S_4 all prime, S_5, S_6, ... are evaluated on demand as
 *    O_k(i) + sum_g hist[g]*g^k from a running histogram of the chunk's gaps so
 *    far: a few hundred multiply-adds, instead of KMAX adds at every prime.
 *    Per chunk the worker records the number of primes of each exact depth, the
 *    smallest prime of each depth, and every hit of depth >= report; these are
 *    merged into the shared results when the chunk completes.
 *
 * 5. Arithmetic limits.  All sums are unsigned __int128 and every addition and
 *    multiplication is checked with __builtin_*_overflow.  If a chunk total or
 *    a running sum for some k would exceed 2^128, that k is marked overflowed
 *    for the rest of the scan (sticky, position in ovf_at[k]); check_chunk then
 *    stops at k-1 and prints "[S_k exceeds 2^128, not tested]" on hits that get
 *    that far.  The power tables carry their own overflow flags (PWOVF).  Gaps
 *    must be < MAXG = 4096; the largest gap below 2^64 is 1550.
 *
 * 6. Primality (is_prime128).  Below 2^64: trial division by the primes up to
 *    47 through two 32-bit remainders, then Miller-Rabin in Montgomery
 *    arithmetic to the bases {2, 325, 9375, 28178, 450775, 9780504,
 *    1795265022}, deterministic for every n < 2^64.  From 2^64 on: trial
 *    division by the primes up to 97, then 128-bit Montgomery Miller-Rabin to
 *    the 24 prime bases 2..89; the first 13 are deterministic below 3.3e24
 *    (Sorenson-Webster), beyond that the verdict is a strong probable-prime
 *    test.  The verify command shows GMP's mpz_probab_prime_p beside it.
 *
 * 7. Checkpoints (state_t).  A plain-text state file holds the position lo
 *    (first integer not yet scanned), the running sums O_k at lo, the prime
 *    count below lo and the smallest prime of every depth found below lo.  The
 *    monitor loop in scan_run writes it atomically (temp file + rename) every
 *    ckpt_secs and once more after the workers stop.  SIGINT/SIGTERM only stop
 *    the workers from taking new chunks, so chunks in flight complete and the
 *    final checkpoint is exact.  Resuming reads the file back as O_k(0).
 *
 * 8. Concurrency map.  sc->next (atomic) hands out chunk indices; sc->finished
 *    (atomic) counts exited workers.  sc->mu + sc->cv protect pub, O, Oovf,
 *    primes_pub, cf, done[], rec[] and the merged results.  out_mu serialises
 *    stdout/stderr so hit lines, notes and the status line do not interleave.
 *    The power tables and the configuration globals are written before the
 *    threads start and read-only afterwards.  Each worker owns its primesieve
 *    iterator, chunk buffers, histogram and per-chunk result fields.
 *
 * 9. Errors.  Anything that would make the results untrustworthy -- bad
 *    arguments, allocation failure, a primesieve error, an inconsistent S_1
 *    offset, a gap >= MAXG, an unreadable state file -- calls die(), which
 *    prints "a252768: <message>" and exits with status 1.  A checkpoint that
 *    cannot be written is reported with note() and the scan goes on.
 *    Primality code is validated by the selftest against GMP and a list of
 *    strong pseudoprimes.
 *
 * SOURCE MAP
 * ----------
 *   generic helpers ....... timing, fatal errors, allocation, 128-bit printing, number parsing
 *   power tables .......... PW[k][g] = g^k with overflow flags
 *   primality ............. 64-bit and 128-bit Montgomery Miller-Rabin, optional GMP bridge
 *   chunks ................ chunk_t, chunk_push, sieve_chunk, chunk_totals
 *   scan state ............ worker_t, struct scan, report_hit, check_chunk, worker_main
 *   checkpoints ........... state_t, write_state, read_state, checkpoint
 *   running a scan ........ print_status, scan_run (threads + monitor), result helpers
 *   direct computation .... direct_hist, sums_from_hist (used by verify and selftest)
 *   commands .............. usage, cmd_scan, cmd_verify, cmd_selftest, main
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdarg.h>
#include <ctype.h>
#include <math.h>
#include <signal.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>
#include <time.h>
#include <inttypes.h>
#include <primesieve.h>
#ifdef HAVE_GMP
#include <gmp.h>
#endif

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint8_t  u8;
typedef unsigned __int128 u128;

#define KLIM    16      /* largest k the power tables support (-k may not exceed it) */
#define MAXG    4096    /* every prime gap must be < MAXG (the largest gap below 2^64 is 1550) */
#define MARGIN  4096    /* sieve this far below a chunk to find the prime preceding it */
#define RING    4096    /* chunk bookkeeping ring; bounds how far threads may run ahead of cf */
#define NKNOWN  7       /* terms listed in the OEIS entry before this work */

/* a(1..7) from the OEIS; the selftest and the summary compare against these */
static const u64 KNOWN[NKNOWN + 1] = { 0, 5, 5, 13, 14593, 372313, 2315773, 541613713 };

/* configuration, set from the command line before any thread starts */
static int  g_kmax   = 12;          /* highest k tested (-k) */
static int  g_report = 7;           /* print/record primes of depth >= this (-r) */
static bool g_quiet  = false;       /* -q: no status line */
static bool stderr_tty;             /* stderr is a terminal: use \r-rewritten status lines */
static volatile sig_atomic_t g_stop = 0;   /* set by SIGINT/SIGTERM: workers take no new chunks */
static pthread_mutex_t out_mu = PTHREAD_MUTEX_INITIALIZER;   /* serialises stdout/stderr output */

/* ------------------------------------------------------------------ */
/* Generic helpers                                                     */
/* ------------------------------------------------------------------ */

/*
 * now -- monotonic clock for timings, rates and the ETA.
 * Returns: seconds (double) since an arbitrary fixed point; unaffected by clock changes.
 * Errors: none (clock_gettime cannot fail for CLOCK_MONOTONIC).
 */
static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

/*
 * die -- report a fatal error and exit.
 * Inputs:  printf-style format and arguments describing the problem.
 * Effect:  prints "a252768: <message>" on stderr (first erasing a live status line
 *          when stderr is a terminal) and calls exit(1).  Never returns.  Used for
 *          every condition after which the results could not be trusted; there is
 *          deliberately no recovery path.
 */
__attribute__((noreturn)) static void die(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fputs(stderr_tty ? "\r\033[Ka252768: " : "a252768: ", stderr);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(1);
}

/*
 * note -- print one diagnostic line on stderr without garbling the status line.
 * Inputs:  printf-style format and arguments.
 * Effect:  takes out_mu, erases the status line if stderr is a terminal, prints the
 *          message plus newline.  Thread-safe; may be called from workers.
 * Errors:  none.
 */
static void note(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    pthread_mutex_lock(&out_mu);
    if (stderr_tty) fputs("\r\033[K", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    pthread_mutex_unlock(&out_mu);
    va_end(ap);
}

/*
 * xmalloc / xcalloc / xrealloc -- allocation that cannot fail silently.
 * Inputs:  as malloc / calloc / realloc.  A zero size is rounded up to one byte so
 *          that the result is never NULL.
 * Returns: the new block.
 * Errors:  out of memory is fatal (die).
 */
static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static void *xcalloc(size_t n, size_t sz)
{
    void *p = calloc(n ? n : 1, sz);
    if (!p) die("out of memory");
    return p;
}

static void *xrealloc(void *p, size_t n)
{
    p = realloc(p, n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

/*
 * u128_str -- decimal rendering of a 128-bit value (printf has no u128 conversion).
 * Inputs:  v; buf with room for at least 40 characters (2^128 has 39 digits).
 * Returns: buf, NUL-terminated.
 * Errors:  none.
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
 * bitlen128 -- number of significant bits of v (0 for v == 0); used in the verify output.
 */
static int bitlen128(u128 v)
{
    u64 hi = (u64)(v >> 64), lo = (u64)v;
    if (hi) return 128 - __builtin_clzll(hi);
    return lo ? 64 - __builtin_clzll(lo) : 0;
}

/*
 * parse_dec -- parse exactly len decimal digits of s into *out.
 * Returns: true on success; false for an empty string, a non-digit, or a value that
 *          does not fit in 128 bits.  *out is untouched on failure.
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
 * parse_u128 -- parse the number syntax accepted on the command line.
 * Accepts: plain decimal ("123"), a power ("2^64", "10^12"), a mantissa with exponent
 *          ("1e15", "25e9", "2.5e10" -- the result must be an integer), and X+Y or X-Y
 *          of any of those ("2^54-32").
 *          The sum/difference form is handled first by splitting at the first + or -
 *          that is not the leading character and parsing both halves recursively.
 * Inputs:  s (NUL-terminated), out.
 * Returns: true and *out on success; false on any syntax error, an exponent beyond
 *          400 (power) or 40 (e-notation), or a result outside 0..2^128-1.
 */
static bool parse_u128(const char *s, u128 *out)
{
    const char *pm = strpbrk(s, "+-");
    if (pm && pm != s) {
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
        /* mantissa may contain a decimal point: "2.5e10" = 25 * 10^(10-1) */
        u128 m, e, frac = 0;
        const char *dot = memchr(s, '.', (size_t)(ee - s));
        if (dot) {
            size_t nfrac = (size_t)(ee - dot - 1);
            u128 ip = 0, fp = 0;
            if (dot != s && !parse_dec(s, (size_t)(dot - s), &ip)) return false;
            if (nfrac == 0 || !parse_dec(dot + 1, nfrac, &fp)) return false;
            if (nfrac > 30) return false;
            m = ip;
            for (size_t i = 0; i < nfrac; i++) {
                if (m > ((u128)0 - 1) / 10) return false;
                m *= 10;
            }
            if (m > ((u128)0 - 1) - fp) return false;
            m += fp;
            frac = nfrac;
        } else if (!parse_dec(s, (size_t)(ee - s), &m)) {
            return false;
        }
        if (!parse_dec(ee + 1, strlen(ee + 1), &e) || e > 40) return false;
        if (e < frac) return false;                 /* would not be an integer */
        for (unsigned i = 0; i < (unsigned)(e - frac); i++) {
            if (m > ((u128)0 - 1) / 10) return false;
            m *= 10;
        }
        *out = m;
        return true;
    }
    return parse_dec(s, strlen(s), out);
}

/*
 * arg_u64 -- parse_u128 restricted to 64 bits for command-line arguments.
 * Inputs:  s, and `what` (e.g. "chunk size") for the error message.
 * Returns: the value.  Errors: fatal (die) with a message naming `what`.
 */
static u64 arg_u64(const char *s, const char *what)
{
    u128 v;
    if (!parse_u128(s, &v) || v > UINT64_MAX) die("bad %s: '%s'", what, s);
    return (u64)v;
}

/*
 * arg_int -- parse a plain decimal integer argument and check it lies in [lo, hi].
 * Returns: the value.  Errors: fatal (die) on junk, empty input or an out-of-range value.
 */
static int arg_int(const char *s, const char *what, int lo, int hi)
{
    char *end;
    long v = strtol(s, &end, 10);
    if (*s == 0 || *end != 0 || v < lo || v > hi) die("bad %s: '%s' (allowed %d..%d)", what, s, lo, hi);
    return (int)v;
}

/*
 * fmt_eng -- format a count for the status line: plain below 10^6, otherwise three
 * significant digits in exponent notation ("4.52e+11").  buf needs 32 bytes.  Returns buf.
 */
static const char *fmt_eng(double v, char *buf)
{
    if (v < 1e6) sprintf(buf, "%.0f", v);
    else sprintf(buf, "%.3g", v);
    return buf;
}

/*
 * fmt_hms -- format a duration in seconds as HH:MM:SS, or as "DdHHh" from 100 hours on.
 * Negative or NaN input prints as zero.  buf needs 32 bytes.  Returns buf.
 */
static const char *fmt_hms(double s, char *buf)
{
    long t = s > 0 ? (long)(s + 0.5) : 0;
    if (t >= 360000L) sprintf(buf, "%ldd%02ldh", t / 86400, (t / 3600) % 24);
    else sprintf(buf, "%02ld:%02ld:%02ld", t / 3600, (t / 60) % 60, t % 60);
    return buf;
}

/* ------------------------------------------------------------------ */
/* Power tables: PW[k][g] = g^k (PWOVF set where g^k >= 2^128)         */
/* ------------------------------------------------------------------ */

/*
 * The scan never computes a power at run time: every g^k it needs is looked up here.
 * Rows 2..4 (the incrementally maintained sums) are the hot ones, about 10 KB each
 * for the gap sizes that actually occur.  Entries that do not fit in 128 bits are 0
 * with the matching PWOVF flag set; the summing code treats such an entry as an
 * overflow of the whole sum.
 */
static u128 PW[KLIM + 1][MAXG];
static u8   PWOVF[KLIM + 1][MAXG];

/*
 * init_pow_tables -- fill PW and PWOVF for 0 <= k <= KLIM and 0 <= g < MAXG.
 * Must be called once, before any scan (main does it).  No inputs, no errors.
 */
static void init_pow_tables(void)
{
    for (unsigned g = 0; g < MAXG; g++) {
        PW[0][g] = 1;
        PWOVF[0][g] = 0;
        for (int k = 1; k <= KLIM; k++) {
            if (PWOVF[k - 1][g] || __builtin_mul_overflow(PW[k - 1][g], (u128)g, &PW[k][g])) {
                PWOVF[k][g] = 1;
                PW[k][g] = 0;
            } else {
                PWOVF[k][g] = 0;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Primality: Montgomery Miller-Rabin, 64-bit deterministic            */
/* ------------------------------------------------------------------ */

/* trial-division primes: all of them for the 128-bit path, the first 15 (<= 47) for 64-bit */
static const u32 SMALLP[] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                              53, 59, 61, 67, 71, 73, 79, 83, 89, 97 };
#define NSMALLP ((int)(sizeof SMALLP / sizeof SMALLP[0]))

/*
 * Montgomery arithmetic modulo an odd n with R = 2^64.  A value x is represented as
 * x*R mod n; the product of two representations, reduced by REDC, is again a
 * representation.  This turns every modular multiplication into two 64x64->128
 * multiplies and a shift instead of a 128-by-64 division, which has no hardware
 * instruction on ARM and is 10x slower.
 *   n    the modulus (odd, >= 3)
 *   ninv -n^-1 mod 2^64
 *   one  the representation of 1, i.e. 2^64 mod n
 */
typedef struct { u64 n, ninv, one; } mont64;

/*
 * m64_init -- prepare Montgomery constants for the odd modulus n.
 * Inputs:  m (output), n odd and >= 3.
 * Effect:  ninv by Newton iteration (x <- x(2 - n x) doubles the correct bits each
 *          round: n is its own inverse mod 8, so 6 rounds give 192 > 64 bits); one =
 *          2^64 mod n computed as (2^64 - n) mod n in 64-bit arithmetic.
 * Errors:  none (an even n would silently give nonsense; callers reject even n first).
 */
static inline void m64_init(mont64 *m, u64 n)
{
    u64 x = n;                              /* Newton iteration: n^-1 mod 2^64, n odd */
    for (int i = 0; i < 6; i++) x *= 2 - n * x;
    m->n = n;
    m->ninv = (u64)0 - x;
    m->one = ((u64)0 - n) % n;              /* 2^64 mod n */
}

/*
 * m64_redc -- Montgomery reduction: returns T * 2^-64 mod n.
 * Inputs:  m; T < n * 2^64 (true for any product of two values below n).
 * Method:  mm = T * ninv mod 2^64 makes T + mm*n divisible by 2^64, so the result is
 *          (T + mm*n) >> 64.  The low halves of T and mm*n add up to exactly 0 or
 *          2^64, hence the carry into the high half is 1 iff (u64)T != 0.  The sum
 *          of the high halves is < 2n, so one conditional subtraction finishes.
 * Returns: a value in [0, n).  Errors: none.
 */
static inline u64 m64_redc(const mont64 *m, u128 T)   /* T < n * 2^64 */
{
    u64 mm = (u64)T * m->ninv;
    u128 mn = (u128)mm * m->n;
    u128 s = (T >> 64) + (mn >> 64) + ((u64)T != 0);
    if (s >= m->n) s -= m->n;
    return (u64)s;
}

/*
 * m64_mul -- product of two Montgomery representations a, b < n (result in [0, n)).
 * m64_add -- modular sum of a, b < n; the carry check handles n close to 2^64.
 * m64_to  -- convert an ordinary residue x < n to Montgomery form (x * 2^64 mod n); uses
 *            a 128-by-64 division, so it is only called a handful of times per test.
 */
static inline u64 m64_mul(const mont64 *m, u64 a, u64 b) { return m64_redc(m, (u128)a * b); }

static inline u64 m64_add(const mont64 *m, u64 a, u64 b)
{
    u64 s = a + b;
    if (s < a || s >= m->n) s -= m->n;
    return s;
}

static inline u64 m64_to(const mont64 *m, u64 x) { return (u64)(((u128)x << 64) % m->n); }

/*
 * m64_sprp -- strong probable-prime (Miller-Rabin) test of m->n to one base.
 * Inputs:  m (initialised for n), am = the base in Montgomery form (nonzero mod n),
 *          d and s with n - 1 = d * 2^s, d odd.
 * Method:  x = base^d (right-to-left binary exponentiation); n passes if x == 1 or
 *          x == n-1, or if squaring x up to s-1 times reaches n-1 before it reaches 1.
 * Returns: true if n is a strong probable prime to this base (every prime passes;
 *          a composite passes with probability <= 1/4), false if n is proven composite.
 */
static bool m64_sprp(const mont64 *m, u64 am, u64 d, int s)
{
    u64 x = m->one, b = am, nm1 = m->n - m->one;
    for (u64 e = d; e; e >>= 1) {
        if (e & 1) x = m64_mul(m, x, b);
        b = m64_mul(m, b, b);
    }
    if (x == m->one || x == nm1) return true;
    for (int i = 1; i < s; i++) {
        x = m64_mul(m, x, x);
        if (x == nm1) return true;
        if (x == m->one) return false;
    }
    return false;
}

/*
 * m64_sprp2 -- the same test specialised to base 2, which is the one that rejects
 * almost every composite and is therefore run first.  Left-to-right exponentiation:
 * squaring is a Montgomery multiply but multiplying by the base is a modular doubling,
 * which saves about a third of the work.  Inputs and result as for m64_sprp.
 */
static bool m64_sprp2(const mont64 *m, u64 d, int s)
{
    u64 two = m64_add(m, m->one, m->one), nm1 = m->n - m->one;
    u64 x = two;
    for (int i = 62 - __builtin_clzll(d); i >= 0; i--) {
        x = m64_mul(m, x, x);
        if ((d >> i) & 1) x = m64_add(m, x, x);
    }
    if (x == m->one || x == nm1) return true;
    for (int i = 1; i < s; i++) {
        x = m64_mul(m, x, x);
        if (x == nm1) return true;
        if (x == m->one) return false;
    }
    return false;
}

/*
 * is_prime64 -- deterministic primality test for any 64-bit n.
 * Steps:   small n and even n directly; n < 1000 by trial division; otherwise two
 *          32-bit remainders (modulo 3*5*...*29 and 31*37*41*43*47) weed out the two
 *          thirds of candidates with a factor <= 47 at a cost of a few nanoseconds;
 *          then Miller-Rabin to base 2, and -- for n >= 2047, the first base-2
 *          pseudoprime -- to the six further bases of Jim Sinclair's set, which
 *          together are deterministic for every n < 2^64.  A base that is 0 mod n is
 *          skipped, as the set requires.
 * Returns: true iff n is prime.  Errors: none.
 * Cost:    this is the hot primality test (every S_2 of every twin prime goes through
 *          it); the trial division and the base-2 shortcut are what keep it cheap.
 */
static bool is_prime64(u64 n)
{
    if (n < 2) return false;
    if (n < 4) return true;
    if (!(n & 1)) return false;
    if (n < 1000) {
        for (int i = 1; i < NSMALLP; i++) {
            if (n == SMALLP[i]) return true;
            if (n % SMALLP[i] == 0) return false;
        }
        return true;
    }
    u32 r = (u32)(n % 3234846615u);         /* 3*5*7*11*13*17*19*23*29 */
    if (!(r % 3) || !(r % 5) || !(r % 7) || !(r % 11) || !(r % 13) ||
        !(r % 17) || !(r % 19) || !(r % 23) || !(r % 29)) return false;
    r = (u32)(n % 95041567u);               /* 31*37*41*43*47 */
    if (!(r % 31) || !(r % 37) || !(r % 41) || !(r % 43) || !(r % 47)) return false;

    mont64 m;
    m64_init(&m, n);
    u64 d = n - 1;
    int s = __builtin_ctzll(d);
    d >>= s;
    if (!m64_sprp2(&m, d, s)) return false;
    if (n < 2047) return true;
    /* with base 2 these bases are deterministic for all n < 2^64 */
    static const u64 B[6] = { 325, 9375, 28178, 450775, 9780504, 1795265022 };
    for (int i = 0; i < 6; i++) {
        u64 a = B[i] % n;
        if (a == 0) continue;
        if (!m64_sprp(&m, m64_to(&m, a), d, s)) return false;
    }
    return true;
}

/* ------------------------------------------------------------------ */
/* Primality: 128-bit Montgomery Miller-Rabin                          */
/* ------------------------------------------------------------------ */

/*
 * Montgomery arithmetic with R = 2^128 for moduli between 2^64 and 2^128.  Same idea
 * as mont64; the extra field r2 = 2^256 mod n lets m128_to convert to Montgomery form
 * with one multiplication instead of a 256-by-128 division.
 */
typedef struct { u128 n, ninv, one, r2; } mont128;

/*
 * mul128_full -- full 256-bit product of a and b as (hi, lo).
 * Schoolbook with four 64x64->128 multiplies; `mid` collects the two cross terms plus
 * the carry out of p00 and is below 3*2^64, so it cannot overflow.  Errors: none.
 */
static inline void mul128_full(u128 a, u128 b, u128 *hi, u128 *lo)
{
    u64 a0 = (u64)a, a1 = (u64)(a >> 64), b0 = (u64)b, b1 = (u64)(b >> 64);
    u128 p00 = (u128)a0 * b0, p01 = (u128)a0 * b1, p10 = (u128)a1 * b0, p11 = (u128)a1 * b1;
    u128 mid = (p00 >> 64) + (u64)p01 + (u64)p10;       /* < 3 * 2^64 */
    *lo = (mid << 64) | (u64)p00;
    *hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
}

/*
 * m128_add -- modular sum of a, b < n (the carry test covers n near 2^128).
 */
static inline u128 m128_add(const mont128 *m, u128 a, u128 b)
{
    u128 s = a + b;
    if (s < a || s >= m->n) s -= m->n;
    return s;
}

/*
 * m128_redc -- Montgomery reduction of the 256-bit value hi:lo; returns hi:lo * 2^-128 mod n.
 * Inputs:  hi < n (true for products of values below n), any lo.
 * Method:  as m64_redc one level up.  mm*n is formed in full; its low half l2 is the
 *          two's complement of lo, so lo + l2 is 0 or 2^128 and the carry into the
 *          high half is (lo != 0).  hi + h2 + carry can exceed 2^128 when n is close
 *          to it; __builtin_add_overflow catches that, and subtracting n (mod 2^128)
 *          then yields the right value because the true sum is below 2n.
 * Returns: a value in [0, n).  Errors: none.
 */
static inline u128 m128_redc(const mont128 *m, u128 hi, u128 lo)   /* hi < n */
{
    u128 mm = lo * m->ninv;
    u128 h2, l2, t;
    mul128_full(mm, m->n, &h2, &l2);
    (void)l2;                                           /* lo + l2 is 0 or 2^128 */
    bool c1 = __builtin_add_overflow(hi, h2, &t);
    bool c2 = __builtin_add_overflow(t, (u128)(lo != 0), &t);
    if (c1 || c2 || t >= m->n) t -= m->n;
    return t;
}

/*
 * m128_mul -- product of two Montgomery representations a, b < n, reduced to [0, n).
 */
static inline u128 m128_mul(const mont128 *m, u128 a, u128 b)
{
    u128 hi, lo;
    mul128_full(a, b, &hi, &lo);
    return m128_redc(m, hi, lo);
}

/*
 * m128_init -- Montgomery constants for an odd n in (2^64, 2^128).
 * Effect:  ninv by Newton iteration (7 rounds: 3 -> 384 bits); one = 2^128 mod n as
 *          (2^128 - n) mod n; r2 = 2^256 mod n by doubling `one` 128 times, which is
 *          cheaper than a 256-bit division and runs only once per candidate.
 * Errors:  none (even n is rejected by the caller).
 */
static void m128_init(mont128 *m, u128 n)
{
    u128 x = n;
    for (int i = 0; i < 7; i++) x *= 2 - n * x;
    m->n = n;
    m->ninv = (u128)0 - x;
    m->one = ((u128)0 - n) % n;             /* 2^128 mod n */
    u128 r = m->one;
    for (int i = 0; i < 128; i++) r = m128_add(m, r, r);
    m->r2 = r;                              /* 2^256 mod n */
}

/*
 * m128_to  -- ordinary residue x < n to Montgomery form: x * 2^128 mod n = REDC(x * r2).
 * ctz128   -- number of trailing zero bits of a nonzero 128-bit value.
 */
static inline u128 m128_to(const mont128 *m, u128 x) { return m128_mul(m, x, m->r2); }   /* x < n */

static inline int ctz128(u128 v)
{
    u64 lo = (u64)v;
    return lo ? __builtin_ctzll(lo) : 64 + __builtin_ctzll((u64)(v >> 64));
}

/*
 * m128_sprp -- strong probable-prime test of m->n to the base am (Montgomery form),
 * with n - 1 = d * 2^s.  Same algorithm and contract as m64_sprp.
 */
static bool m128_sprp(const mont128 *m, u128 am, u128 d, int s)
{
    u128 x = m->one, b = am, nm1 = m->n - m->one;
    for (u128 e = d; e; e >>= 1) {
        if (e & 1) x = m128_mul(m, x, b);
        b = m128_mul(m, b, b);
    }
    if (x == m->one || x == nm1) return true;
    for (int i = 1; i < s; i++) {
        x = m128_mul(m, x, x);
        if (x == nm1) return true;
        if (x == m->one) return false;
    }
    return false;
}

/* the first 13 primes are deterministic below 3.3e24 (Sorenson-Webster); 11 more beyond */
static const u32 BASES128[24] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
                                  43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89 };

/*
 * is_prime128 -- primality test for any 128-bit n; the only test the scan calls.
 * Dispatch: n < 2^64 goes to the deterministic is_prime64.  Otherwise: even -> false;
 *          trial division by the primes up to 97 (a u128 remainder each, cheap next to
 *          an exponentiation); then Miller-Rabin to the 24 bases of BASES128, stopping
 *          at the first failure.  Every base is < n here, so none is 0 mod n.
 * Returns: true iff n passes.  Proven for n < 3.3e24 (13 bases suffice there); above
 *          that a composite can pass only if it is a strong pseudoprime to all 24
 *          bases, which no number below 2^128 is known to be.
 * Errors:  none.
 */
static bool is_prime128(u128 n)
{
    if (n < ((u128)1 << 64)) return is_prime64((u64)n);
    if (!(n & 1)) return false;
    for (int i = 1; i < NSMALLP; i++)
        if (n % SMALLP[i] == 0) return false;
    mont128 m;
    m128_init(&m, n);
    u128 d = n - 1;
    int s = ctz128(d);
    d >>= s;
    for (int i = 0; i < 24; i++)
        if (!m128_sprp(&m, m128_to(&m, BASES128[i]), d, s)) return false;
    return true;
}

#ifdef HAVE_GMP
/*
 * mpz_set_u128 -- store a 128-bit value in a GMP integer (two 64-bit halves).
 */
static void mpz_set_u128(mpz_t z, u128 v)
{
    mpz_set_ui(z, (unsigned long)(u64)(v >> 64));
    mpz_mul_2exp(z, z, 64);
    mpz_add_ui(z, z, (unsigned long)(u64)v);
}

/*
 * gmp_isprime128 -- GMP's verdict on n, as an independent check of is_prime128.
 * Returns: mpz_probab_prime_p with 40 rounds: 2 = proven prime (small n), 1 = probable
 *          prime (BPSW + Miller-Rabin), 0 = composite.
 */
static int gmp_isprime128(u128 n)   /* 2 prime, 1 probable prime, 0 composite */
{
    mpz_t z;
    mpz_init(z);
    mpz_set_u128(z, n);
    int r = mpz_probab_prime_p(z, 40);
    mpz_clear(z);
    return r;
}
#endif

/* ------------------------------------------------------------------ */
/* Chunks: sieve a range, keep its gaps and gap histogram              */
/* ------------------------------------------------------------------ */

/*
 * chunk_t -- one chunk of the range as seen by one worker: the primes of [lo, hi) are
 * represented by the sequence of their incoming gaps, in order, so that the check pass
 * can replay them without sieving again.  Owned by a single worker and reused for
 * every chunk it processes (buffers only grow).
 */
typedef struct {
    u8  *gb;  size_t gb_cap, n;       /* gap bytes: g/2 for even g <= 508, 255 = escaped; n = count */
    u32 *esc; size_t esc_cap, esc_n;  /* escaped gaps (the gap 1, or gaps >= 510) in order of occurrence */
    u64  cnt[MAXG];                   /* gap histogram of the chunk: cnt[g] = number of incoming gaps g */
    u64  lo, hi;                      /* the chunk covers the primes in [lo, hi) */
    u64  p_prev;                      /* largest prime below lo (valid iff has_prev) */
    u64  p_last;                      /* largest prime below hi (p_prev if the chunk has no prime) */
    u64  nprimes;                     /* number of primes in [lo, hi) */
    unsigned maxg;                    /* largest gap seen (bounds the histogram loops) */
    bool has_prev;                    /* false only for the chunk that contains 2 (no prime below it) */
} chunk_t;

/*
 * chunk_push -- append the incoming gap g of the prime q to the chunk.
 * Inputs:  c, q (only for the error message), g = q - previous prime.
 * Effect:  stores g as a byte (g/2 for even g <= 508) or as the escape byte 255 plus
 *          an entry in esc[]; counts it in cnt[g]; updates maxg.  Both buffers grow
 *          by doubling when full.
 * Errors:  g >= MAXG is fatal (cannot happen below 2^64); allocation failure is fatal.
 */
static inline void chunk_push(chunk_t *c, u64 q, u64 g)
{
    if (g >= MAXG) die("prime gap %" PRIu64 " before %" PRIu64 " exceeds MAXG", g, q);
    if (c->n == c->gb_cap) {
        c->gb_cap = c->gb_cap * 2 + 65536;
        c->gb = xrealloc(c->gb, c->gb_cap);
    }
    if (!(g & 1) && g <= 508) {
        c->gb[c->n++] = (u8)(g >> 1);
    } else {
        if (c->esc_n == c->esc_cap) {
            c->esc_cap = c->esc_cap * 2 + 64;
            c->esc = xrealloc(c->esc, c->esc_cap * sizeof(u32));
        }
        c->esc[c->esc_n++] = (u32)g;
        c->gb[c->n++] = 255;
    }
    c->cnt[g]++;
    if (g > c->maxg) c->maxg = (unsigned)g;
}

/*
 * sieve_chunk -- step (a) of the pipeline: enumerate the primes of [lo, hi) and record
 * the incoming gap of each one.
 * Inputs:  it = this worker's primesieve iterator; c = this worker's chunk buffer;
 *          lo < hi, hi <= end + 1.
 * Effect:  resets c, then starts the iterator MARGIN below lo so that the prime just
 *          before the chunk (p_prev) is seen first; the first prime >= lo gets the gap
 *          from p_prev, every later prime the gap from its predecessor.  If no prime
 *          lies within MARGIN below lo (impossible below 2^64, where gaps are < 1551)
 *          the margin is quadrupled and the sieve restarted.  When lo <= 2 there is no
 *          prime below the chunk: has_prev is false and the prime 2 gets no gap.
 *          On return: gb[0..n) and esc[] hold the gaps, cnt[] the histogram, and
 *          nprimes, p_prev, p_last, maxg are set.  The gap buffer is pre-sized from the
 *          prime density (hi - lo) / ln(midpoint) * 1.25 and grows if that was short.
 * Errors:  fatal on a primesieve error, a gap >= MAXG, or no prime within 2^24 below lo.
 */
static void sieve_chunk(primesieve_iterator *it, chunk_t *c, u64 lo, u64 hi)
{
    c->lo = lo; c->hi = hi;
    c->n = 0; c->esc_n = 0; c->nprimes = 0; c->maxg = 0;
    memset(c->cnt, 0, sizeof c->cnt);

    /* expected number of primes in the chunk, with 25% headroom; chunk_push grows if needed */
    double lnx = log((double)lo + (double)(hi - lo) * 0.5 + 3.0);
    size_t need = (size_t)((double)(hi - lo) / (lnx > 1.0 ? lnx : 1.0) * 1.25) + 65536;
    if (c->gb_cap < need) {
        c->gb_cap = need;
        c->gb = xrealloc(c->gb, need);
    }

    u64 margin = MARGIN;
    for (;;) {
        u64 start = lo > margin ? lo - margin : 0;
        primesieve_jump_to(it, start, hi);
        u64 pp = 0, q;
        bool have = false;
        while ((q = primesieve_next_prime(it)) < lo) { pp = q; have = true; }
        if (!have && lo > 2) {
            margin *= 4;
            if (margin > ((u64)1 << 24)) die("no prime within 2^24 below %" PRIu64 "?!", lo);
            continue;
        }
        c->has_prev = have;
        c->p_prev = pp;
        while (q < hi) {
            if (have) chunk_push(c, q, q - pp);
            else have = true;                   /* q = 2, the first prime, has no gap */
            pp = q;
            c->nprimes++;
            q = primesieve_next_prime(it);
        }
        if (it->is_error) die("primesieve error in [%" PRIu64 ", %" PRIu64 ")", lo, hi);
        c->p_last = pp;
        return;
    }
}

/*
 * chunk_totals -- the chunk's contribution to every running sum.
 * Inputs:  c after sieve_chunk; T, Tovf with room for indices 1..g_kmax.
 * Effect:  T[k] = sum_g cnt[g] * g^k for k = 1..g_kmax, formed from the histogram (a
 *          few hundred multiply-adds per k).  Tovf[k] is set, and T[k] zeroed, when
 *          a power or the sum would exceed 2^128.
 * Errors:  none.
 */
static void chunk_totals(const chunk_t *c, u128 T[], bool Tovf[])
{
    for (int k = 1; k <= g_kmax; k++) {
        u128 s = 0;
        bool ovf = false;
        for (unsigned g = 1; g <= c->maxg; g++) {
            u64 n = c->cnt[g];
            u128 prod;
            if (!n) continue;
            if (PWOVF[k][g] || __builtin_mul_overflow((u128)n, PW[k][g], &prod) ||
                __builtin_add_overflow(s, prod, &s)) { ovf = true; break; }
        }
        T[k] = ovf ? 0 : s;
        Tovf[k] = ovf;
    }
}

/* ------------------------------------------------------------------ */
/* Scan state shared by the worker threads                             */
/* ------------------------------------------------------------------ */

/* a recorded hit: prime p of exact depth `depth` (>= the report threshold) */
typedef struct { u64 p; int depth; } hit_t;

/*
 * rec_t -- the ring record of a chunk, written at hand-off time: the running sums at
 * the START of the chunk.  checkpoint() reads the record of the completion frontier.
 */
typedef struct {
    u64  idx;                         /* chunk index the record belongs to (UINT64_MAX = empty) */
    u64  primes;                      /* number of primes below this chunk (origin 0 or checkpoint) */
    u128 O[KLIM + 1];                 /* O_k(idx), k = 1..g_kmax */
    bool ovf[KLIM + 1];               /* O_k already overflowed 2^128 at this point */
} rec_t;

typedef struct scan scan_t;

/*
 * worker_t -- everything private to one worker thread.  The result fields (best,
 * depth_count, twins, hits) hold the current chunk only; worker_main resets them
 * before check_chunk and merges them into the scan under the mutex afterwards.
 */
typedef struct {
    int id;                           /* 0..nthreads-1 (unused except for debugging) */
    pthread_t th;
    scan_t *sc;                       /* the shared scan */
    primesieve_iterator it;           /* each thread has its own sieve */
    chunk_t c;                        /* the chunk being processed */
    u64 hist[MAXG];                   /* running gap histogram during the check pass */
    u64 best[KLIM + 1];               /* per chunk: smallest p of exact depth d (0 = none) */
    u64 depth_count[KLIM + 1];        /* per chunk: number of twin primes of exact depth d */
    u64 twins;                        /* per chunk: twin primes examined (= primes of depth >= 1) */
    hit_t *hits; size_t nhits, hits_cap;   /* per chunk: hits of depth >= report */
} worker_t;

/*
 * struct scan -- one scan of [start, end]: its parameters, the shared pipeline state
 * (protected by mu unless marked atomic) and the merged results.  Allocated with
 * scan_new; large (the ring is about 1.2 MB), so never put on the stack.
 */
struct scan {
    /* parameters, set by the caller before scan_run */
    u64  origin;                      /* where the whole computation began (0 = exhaustive from 0) */
    u64  start, end;                  /* this run scans the primes in [start, end] */
    u64  chunk, nchunks;              /* chunk size (0 = choose automatically) and resulting count */
    bool check;                       /* false: totals only, no primality tests (used for START > 0) */
    bool print_hits;                  /* print hit lines on stdout as they are found */
    bool verbose;                     /* status line on stderr */
    int  report;                      /* record/print primes of depth >= report */
    u128 O0[KLIM + 1]; bool O0ovf[KLIM + 1];   /* running sums at `start` (S_k of the largest prime below it) */
    u64  primes0;                     /* primes below start */
    /* runtime, shared by the workers */
    atomic_uint_fast64_t next;        /* next chunk index to hand out */
    atomic_int finished;              /* workers that have exited */
    pthread_mutex_t mu;               /* protects everything below down to the results */
    pthread_cond_t cv;                /* signalled whenever pub or cf advances */
    u64  pub;                         /* chunk whose start offsets are currently in O */
    u128 O[KLIM + 1];                 /* O_k(pub) */
    bool Oovf[KLIM + 1];              /* O_k has exceeded 2^128 (sticky) */
    u64  ovf_at[KLIM + 1];            /* chunk start where that happened */
    u64  primes_pub;                  /* primes below chunk pub */
    u64  cf;                          /* completion frontier: lowest chunk whose check is unfinished */
    u8   done[RING];                  /* done[i % RING]: chunk i (>= cf) is checked but cf has not passed it */
    rec_t rec[RING];                  /* rec[i % RING]: start offsets of chunk i */
    /* results, merged as chunks complete */
    u64  best[KLIM + 1];              /* smallest p of exact depth d found so far (0 = none) */
    u64  depth_count[KLIM + 1];       /* number of twin primes of exact depth d */
    u64  primes, twins, chunks_done;  /* totals over completed chunks */
    hit_t *hits; size_t nhits, hits_cap;   /* all hits of depth >= report (unordered) */
    u64 *ghist;                       /* when non-NULL: gap histogram of all completed chunks (MAXG entries) */
    double t0, t1;                    /* start and end time of scan_run */
};

/*
 * report_hit -- print one hit line on stdout.
 * Inputs:  p and its depth; S[1..known] = the power sums that were computed (known is
 *          depth + 1 when the first composite sum is available, else depth); capped =
 *          S_(depth+1) could not be computed because it exceeds 2^128.
 * Effect:  "p = ...  n = depth  S_1..S_n = ...  S_(n+1) = ... (composite)" or a note
 *          why nothing further was tested.  Thread-safe (out_mu); flushes stdout so
 *          the line survives a later crash or kill.
 * Errors:  none.
 */
static void report_hit(u64 p, int depth, const u128 S[], int known, bool capped)
{
    char buf[48];
    pthread_mutex_lock(&out_mu);
    if (stderr_tty) fputs("\r\033[K", stderr);
    printf("p = %" PRIu64 "  n = %d  S_1..S_%d =", p, depth, depth);
    for (int k = 1; k <= depth; k++) printf(" %s", u128_str(S[k], buf));
    if (known > depth) printf("  S_%d = %s (composite)", depth + 1, u128_str(S[depth + 1], buf));
    else if (capped) printf("  [S_%d exceeds 2^128, not tested]", depth + 1);
    else printf("  [k_max = %d reached]", depth);
    printf("\n");
    fflush(stdout);
    pthread_mutex_unlock(&out_mu);
}

/*
 * check_chunk -- step (c) of the pipeline: replay the chunk's gaps with the exact
 * running sums and test every candidate.
 * Inputs:  w (its chunk c filled by sieve_chunk; hist as scratch; best, depth_count,
 *          twins, hits already reset by the caller); O[k], Oovf[k] = the running sums
 *          O_k at the start of the chunk and their overflow flags, k = 1..g_kmax.
 * Method:  keff = largest k such that no sum up to k has overflowed.  p starts at
 *          p_prev (or at 2 for the first chunk) and advances gap by gap; S2, S3, S4
 *          are updated from the power tables and the gap is counted in hist.  A
 *          prime is a candidate only when its incoming gap is 2 (then S_1 = p - 2 is
 *          prime).  S2, S3, S4 are tested in turn; after that S_k for k = 5..keff is
 *          formed as O[k] + sum_g hist[g]*g^k and tested, until the first composite
 *          one or an overflow.  The exact depth is counted, the smallest p per depth
 *          kept, and hits of depth >= sc->report recorded (and printed if print_hits).
 * Errors:  fatal if O[1] != p_prev - 2 (the hand-off bookkeeping is wrong) or if the
 *          sums for k <= 4 are marked overflowed (impossible below 2^64).
 */
static void check_chunk(worker_t *w, const u128 O[], const bool Oovf[])
{
    scan_t *sc = w->sc;
    chunk_t *c = &w->c;
    char b1[48];
    int keff = g_kmax;
    for (int k = 1; k <= g_kmax; k++) if (Oovf[k]) { keff = k - 1; break; }
    if (keff < 4) die("running sums overflowed 128 bits already for k <= 4 (unexpected)");
    if (c->has_prev && O[1] != (u128)(c->p_prev - 2))
        die("internal error: S_1 offset %s at chunk start %" PRIu64 " but previous prime is %" PRIu64,
            u128_str(O[1], b1), c->lo, c->p_prev);

    memset(w->hist, 0, sizeof w->hist);
    u128 S2 = O[2], S3 = O[3], S4 = O[4];
    u64 p = c->has_prev ? c->p_prev : 2;    /* the gaps start at the prime 2 */
    size_t ei = 0;
    unsigned maxg = 0;
    const u8 *gb = c->gb;
    const u32 *esc = c->esc;

    for (size_t i = 0, n = c->n; i < n; i++) {
        u64 g = gb[i] == 255 ? (u64)esc[ei++] : ((u64)gb[i] << 1);
        p += g;
        S2 += PW[2][g];
        S3 += PW[3][g];
        S4 += PW[4][g];
        w->hist[g]++;
        if (g > maxg) maxg = (unsigned)g;
        if (g != 2) continue;               /* S_1 = p - 2 is prime iff the gap into p is 2 */
        w->twins++;
        if (!is_prime128(S2)) {
            w->depth_count[1]++;
            if (!w->best[1]) w->best[1] = p;
            continue;
        }

        u128 S[KLIM + 2];
        int depth = 2, known = 3;
        bool capped = false;
        S[1] = p - 2; S[2] = S2; S[3] = S3;
        if (is_prime128(S3)) {
            depth = 3; S[4] = S4; known = 4;
            if (is_prime128(S4)) {
                depth = 4;
                /* rare survivor: evaluate the higher sums from the histogram on demand */
                for (int k = 5; k <= keff; k++) {
                    u128 s = O[k];
                    bool ovf = false;
                    for (unsigned gg = 1; gg <= maxg; gg++) {
                        u64 cnt = w->hist[gg];
                        u128 prod;
                        if (!cnt) continue;
                        if (PWOVF[k][gg] || __builtin_mul_overflow((u128)cnt, PW[k][gg], &prod) ||
                            __builtin_add_overflow(s, prod, &s)) { ovf = true; break; }
                    }
                    if (ovf) { capped = true; break; }
                    S[k] = s; known = k;
                    if (!is_prime128(s)) break;
                    depth = k;
                }
                if (depth == keff && keff < g_kmax) capped = true;
            }
        }
        w->depth_count[depth]++;
        if (!w->best[depth] || p < w->best[depth]) w->best[depth] = p;
        if (depth >= sc->report) {
            if (sc->print_hits) report_hit(p, depth, S, known, capped);
            if (w->nhits == w->hits_cap) {
                w->hits_cap = w->hits_cap * 2 + 64;
                w->hits = xrealloc(w->hits, w->hits_cap * sizeof(hit_t));
            }
            w->hits[w->nhits].p = p;
            w->hits[w->nhits].depth = depth;
            w->nhits++;
        }
    }
}

/*
 * worker_main -- body of each worker thread (see THEORY OF OPERATION, 2 and 3).
 * Inputs:  arg = its worker_t.
 * Loop:    take the next chunk index i from the atomic counter (stop when the range
 *          is exhausted or g_stop is set); wait until i < cf + RING so the ring slot
 *          is free; sieve the chunk and form its totals; hand-off: wait for pub == i,
 *          copy O_k(i), record it in rec[i % RING], add the totals into sc->O
 *          (marking sticky overflows), publish pub = i + 1; run the check pass
 *          (unless this is a totals-only scan); then under the mutex mark the chunk
 *          done, advance the completion frontier over consecutive done chunks, and
 *          merge the chunk's statistics, best primes and hits into the scan.
 * Returns: NULL after incrementing sc->finished.
 * Errors:  anything fatal inside the steps calls die (process exits).
 */
static void *worker_main(void *arg)
{
    worker_t *w = arg;
    scan_t *sc = w->sc;
    chunk_t *c = &w->c;

    for (;;) {
        if (g_stop) break;
        u64 i = atomic_fetch_add(&sc->next, 1);
        if (i >= sc->nchunks) break;

        /* back-pressure: the ring slot i % RING must not belong to a live chunk */
        pthread_mutex_lock(&sc->mu);
        while (i >= sc->cf + RING) pthread_cond_wait(&sc->cv, &sc->mu);
        pthread_mutex_unlock(&sc->mu);

        u64 lo = sc->start + i * sc->chunk;
        u64 hi = (i + 1 == sc->nchunks) ? sc->end + 1 : lo + sc->chunk;
        /* step (a): sieve, store the gaps, form the chunk totals */
        sieve_chunk(&w->it, c, lo, hi);
        u128 T[KLIM + 1];
        bool Tovf[KLIM + 1];
        chunk_totals(c, T, Tovf);

        /* step (b): ordered hand-off -- wait for the offsets of this chunk, publish the next */
        u128 O[KLIM + 1];
        bool Oovf[KLIM + 1];
        pthread_mutex_lock(&sc->mu);
        while (sc->pub != i) pthread_cond_wait(&sc->cv, &sc->mu);
        memcpy(O, sc->O, sizeof O);
        memcpy(Oovf, sc->Oovf, sizeof Oovf);
        rec_t *r = &sc->rec[i % RING];
        r->idx = i;
        r->primes = sc->primes_pub;
        memcpy(r->O, O, sizeof O);
        memcpy(r->ovf, Oovf, sizeof Oovf);
        for (int k = 1; k <= g_kmax; k++) {
            if (sc->Oovf[k]) continue;
            if (Tovf[k] || __builtin_add_overflow(sc->O[k], T[k], &sc->O[k])) {
                sc->Oovf[k] = true;
                sc->ovf_at[k] = lo;
                if (sc->check)
                    note("note: S_%d exceeds 2^128 from %" PRIu64 " on; depths >= %d are no longer tested", k, lo, k);
            }
        }
        sc->primes_pub += c->nprimes;
        sc->pub = i + 1;
        pthread_cond_broadcast(&sc->cv);
        pthread_mutex_unlock(&sc->mu);

        /* step (c): the check pass, with this chunk's results collected in w */
        w->nhits = 0;
        w->twins = 0;
        memset(w->best, 0, sizeof w->best);
        memset(w->depth_count, 0, sizeof w->depth_count);
        if (sc->check) check_chunk(w, O, Oovf);

        /* completion: advance the frontier over every consecutive finished chunk, merge results */
        pthread_mutex_lock(&sc->mu);
        sc->done[i % RING] = 1;
        while (sc->cf < sc->nchunks && sc->done[sc->cf % RING]) {
            sc->done[sc->cf % RING] = 0;
            sc->cf++;
        }
        sc->primes += c->nprimes;
        sc->twins += w->twins;
        sc->chunks_done++;
        if (sc->ghist)
            for (unsigned g = 1; g <= c->maxg; g++) sc->ghist[g] += c->cnt[g];
        for (int d = 1; d <= KLIM; d++) {
            sc->depth_count[d] += w->depth_count[d];
            if (w->best[d] && (!sc->best[d] || w->best[d] < sc->best[d])) sc->best[d] = w->best[d];
        }
        if (w->nhits) {
            if (sc->nhits + w->nhits > sc->hits_cap) {
                sc->hits_cap = (sc->nhits + w->nhits) * 2 + 256;
                sc->hits = xrealloc(sc->hits, sc->hits_cap * sizeof(hit_t));
            }
            memcpy(sc->hits + sc->nhits, w->hits, w->nhits * sizeof(hit_t));
            sc->nhits += w->nhits;
        }
        pthread_cond_broadcast(&sc->cv);
        pthread_mutex_unlock(&sc->mu);
    }
    atomic_fetch_add(&sc->finished, 1);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Checkpoint files                                                    */
/* ------------------------------------------------------------------ */

/*
 * state_t -- contents of a checkpoint file.  Text format, one item per line:
 *     A252768-state 1            magic and format version
 *     kmax K                     -k the scan was run with (must match on resume)
 *     origin X                   where the whole computation began (0 = from scratch)
 *     end X                      END of the run that wrote the file
 *     lo X                       first integer not yet scanned
 *     primes N                   number of primes below lo
 *     O k VALUE OVF              running sum S_k at lo (decimal) and its overflow flag, k = 1..kmax
 *     best d P                   smallest prime of exact depth d found below lo (only d with a value)
 */
typedef struct {
    int  kmax;
    u64  origin, end, lo, primes;
    u128 O[KLIM + 1];
    bool ovf[KLIM + 1];
    u64  best[KLIM + 1];
} state_t;

/*
 * write_state -- write a checkpoint atomically.
 * Inputs:  path; st.
 * Effect:  writes path.tmp and renames it over path, so a crash mid-write leaves the
 *          previous checkpoint intact.
 * Errors:  not fatal: a failure to create, close or rename the file is reported with
 *          note() and the scan continues (the next checkpoint will try again).
 */
static void write_state(const char *path, const state_t *st)
{
    char tmp[4096], b[48];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "w");
    if (!f) { note("warning: cannot write %s: %s", tmp, strerror(errno)); return; }
    fprintf(f, "A252768-state 1\nkmax %d\norigin %" PRIu64 "\nend %" PRIu64 "\nlo %" PRIu64 "\nprimes %" PRIu64 "\n",
            st->kmax, st->origin, st->end, st->lo, st->primes);
    for (int k = 1; k <= st->kmax; k++)
        fprintf(f, "O %d %s %d\n", k, u128_str(st->O[k], b), st->ovf[k] ? 1 : 0);
    for (int d = 1; d <= KLIM; d++)
        if (st->best[d]) fprintf(f, "best %d %" PRIu64 "\n", d, st->best[d]);
    if (fclose(f) != 0 || rename(tmp, path) != 0)
        note("warning: cannot finish writing %s: %s", path, strerror(errno));
}

/*
 * read_state -- read a checkpoint written by write_state.
 * Inputs:  path; st (overwritten).
 * Returns: false if the file does not exist (a fresh start); true with st filled in
 *          otherwise.  Unknown keys are ignored, missing ones stay zero.
 * Errors:  fatal if the file exists but is not a valid state file (bad magic or
 *          version, unparsable value, k or d out of range, kmax outside 4..KLIM).
 */
static bool read_state(const char *path, state_t *st)
{
    FILE *f = fopen(path, "r");
    if (!f) return false;
    memset(st, 0, sizeof *st);
    char line[512];
    int ver = 0;
    if (!fgets(line, sizeof line, f) || sscanf(line, "A252768-state %d", &ver) != 1 || ver != 1)
        die("%s is not an a252768 state file", path);
    while (fgets(line, sizeof line, f)) {
        char key[32], v1[64], v2[64], v3[64];
        int n = sscanf(line, "%31s %63s %63s %63s", key, v1, v2, v3);
        u128 v;
        if (n < 2) continue;
        if (!strcmp(key, "kmax")) st->kmax = atoi(v1);
        else if (!strcmp(key, "origin")) { if (!parse_u128(v1, &v)) die("bad state"); st->origin = (u64)v; }
        else if (!strcmp(key, "end"))    { if (!parse_u128(v1, &v)) die("bad state"); st->end = (u64)v; }
        else if (!strcmp(key, "lo"))     { if (!parse_u128(v1, &v)) die("bad state"); st->lo = (u64)v; }
        else if (!strcmp(key, "primes")) { if (!parse_u128(v1, &v)) die("bad state"); st->primes = (u64)v; }
        else if (!strcmp(key, "O") && n >= 4) {
            int k = atoi(v1);
            if (k < 1 || k > KLIM || !parse_u128(v2, &st->O[k])) die("bad O line in %s", path);
            st->ovf[k] = atoi(v3) != 0;
        } else if (!strcmp(key, "best") && n >= 3) {
            int d = atoi(v1);
            if (d < 1 || d > KLIM || !parse_u128(v2, &v)) die("bad best line in %s", path);
            st->best[d] = (u64)v;
        }
    }
    fclose(f);
    if (st->kmax < 4 || st->kmax > KLIM) die("bad kmax in %s", path);
    return true;
}

/*
 * checkpoint -- write the state at the completion frontier.
 * Inputs:  sc (locked internally), path.
 * Method:  everything below chunk cf is checked, so the file records lo = start of
 *          chunk cf and the running sums there.  Those sums come from one of four
 *          places: the final sums when every chunk is done (lo = end + 1); the initial
 *          sums when cf == 0; the ring record of chunk cf when that chunk has had its
 *          hand-off; or sc->O itself when chunk cf was never started (pub == cf, e.g.
 *          after Ctrl-C).  If none applies (chunk cf started but not yet handed off)
 *          this round is skipped -- the previous file stays valid.  Only best[] values
 *          below lo are saved; later ones will be found again after a resume.
 * Errors:  see write_state.
 */
static void checkpoint(scan_t *sc, const char *path)
{
    state_t st;
    memset(&st, 0, sizeof st);
    bool ok = true;
    pthread_mutex_lock(&sc->mu);
    u64 cf = sc->cf;
    if (cf >= sc->nchunks) {                  /* everything done: the final sums */
        st.lo = sc->end + 1;
        memcpy(st.O, sc->O, sizeof st.O);
        memcpy(st.ovf, sc->Oovf, sizeof st.ovf);
        st.primes = sc->primes_pub;
    } else if (cf == 0) {                     /* nothing completed yet: the initial sums */
        st.lo = sc->start;
        memcpy(st.O, sc->O0, sizeof st.O);
        memcpy(st.ovf, sc->O0ovf, sizeof st.ovf);
        st.primes = sc->primes0;
    } else if (sc->rec[cf % RING].idx == cf) {   /* chunk cf has had its hand-off: its record */
        st.lo = sc->start + cf * sc->chunk;
        memcpy(st.O, sc->rec[cf % RING].O, sizeof st.O);
        memcpy(st.ovf, sc->rec[cf % RING].ovf, sizeof st.ovf);
        st.primes = sc->rec[cf % RING].primes;
    } else if (sc->pub == cf) {               /* chunk cf never started: O holds its offsets */
        st.lo = sc->start + cf * sc->chunk;
        memcpy(st.O, sc->O, sizeof st.O);
        memcpy(st.ovf, sc->Oovf, sizeof st.ovf);
        st.primes = sc->primes_pub;
    } else {
        ok = false;
    }
    if (ok) {
        st.kmax = g_kmax;
        st.origin = sc->origin;
        st.end = sc->end;
        for (int d = 1; d <= KLIM; d++)
            if (sc->best[d] && sc->best[d] < st.lo) st.best[d] = sc->best[d];
    }
    pthread_mutex_unlock(&sc->mu);
    if (ok) write_state(path, &st);
}

/* ------------------------------------------------------------------ */
/* Running a scan                                                      */
/* ------------------------------------------------------------------ */

/*
 * print_status -- one status line on stderr: elapsed, percent done, frontier
 * position, rate, ETA, and (for a checking scan) primes, twins, hits and the best
 * depth so far.  Progress is measured at the completion frontier.
 * Inputs:  sc (locked briefly to read the counters), t = current time, final = end
 *          of the run (always finish the line with a newline).
 * Effect:  on a terminal the line is rewritten in place (\r); otherwise a full line.
 * Errors:  none.
 */
static void print_status(scan_t *sc, double t, bool final)
{
    pthread_mutex_lock(&sc->mu);
    u64 cf = sc->cf;
    u64 lo = cf >= sc->nchunks ? sc->end + 1 : sc->start + cf * sc->chunk;
    u64 primes = sc->primes, twins = sc->twins;
    size_t nh = sc->nhits;
    int bd = 0;
    u64 bp = 0;
    for (int d = KLIM; d >= 1; d--) if (sc->best[d]) { bd = d; bp = sc->best[d]; break; }
    pthread_mutex_unlock(&sc->mu);

    double done = (double)(lo - sc->start), total = (double)(sc->end - sc->start) + 1.0;
    double el = t - sc->t0, rate = el > 0 ? done / el : 0, eta = rate > 0 ? (total - done) / rate : 0;
    char b1[32], b2[32], b3[32], b4[32], b5[32], b6[32];
    pthread_mutex_lock(&out_mu);
    if (stderr_tty) fputs("\r\033[K", stderr);
    fprintf(stderr, "[%s] %5.1f%% at %s  %s/s  ETA %s", fmt_hms(el, b1), 100.0 * done / total,
            fmt_eng((double)lo, b2), fmt_eng(rate, b3), fmt_hms(eta, b4));
    if (sc->check)
        fprintf(stderr, "  primes %s  twins %s  hits %zu  best n=%d @ %" PRIu64,
                fmt_eng((double)primes, b5), fmt_eng((double)twins, b6), nh, bd, bp);
    else
        fprintf(stderr, "  (sums only)");
    if (!stderr_tty || final) fputc('\n', stderr);
    fflush(stderr);
    pthread_mutex_unlock(&out_mu);
}

/*
 * scan_run -- execute a scan: start the workers, monitor them, collect the results.
 * Inputs:  sc with the parameter fields set (origin, start, end, chunk or 0, check,
 *          print_hits, verbose, report, O0/O0ovf/primes0, best[] may be preloaded
 *          from a checkpoint); nthreads; statefile (NULL = no checkpoints);
 *          ckpt_secs = checkpoint interval.
 * Effect:  chooses the chunk size if none was given (range / (32 * threads), clamped
 *          to 10^6..10^9), resets the runtime state, starts nthreads workers, then
 *          loops every 0.1 s printing the status line (every 0.5 s on a terminal,
 *          every 60 s otherwise) and writing checkpoints until all workers have
 *          exited; joins them and frees their buffers.  On return sc->O holds the
 *          running sums at the end of the scanned prefix (= S_k at end for a complete
 *          run), and best, depth_count, primes, twins, hits hold the results.
 * Errors:  fatal if end is beyond primesieve's limit or a thread cannot be created.
 */
static void scan_run(scan_t *sc, int nthreads, const char *statefile, double ckpt_secs)
{
    if (sc->end >= primesieve_get_max_stop()) die("END too large for primesieve");
    if (nthreads < 1) nthreads = 1;
    u64 range = sc->end >= sc->start ? sc->end - sc->start + 1 : 0;
    if (!sc->chunk) {
        u64 c = range / ((u64)nthreads * 32);
        if (c < 1000000) c = 1000000;
        if (c > 1000000000) c = 1000000000;
        sc->chunk = c;
    }
    sc->nchunks = range ? (range + sc->chunk - 1) / sc->chunk : 0;

    atomic_store(&sc->next, 0);
    atomic_store(&sc->finished, 0);
    pthread_mutex_init(&sc->mu, NULL);
    pthread_cond_init(&sc->cv, NULL);
    sc->pub = 0;
    sc->cf = 0;
    memcpy(sc->O, sc->O0, sizeof sc->O);
    memcpy(sc->Oovf, sc->O0ovf, sizeof sc->Oovf);
    memset(sc->ovf_at, 0, sizeof sc->ovf_at);
    sc->primes_pub = sc->primes0;
    memset(sc->done, 0, sizeof sc->done);
    for (int i = 0; i < RING; i++) sc->rec[i].idx = UINT64_MAX;
    sc->primes = sc->twins = sc->chunks_done = 0;
    sc->nhits = 0;
    sc->t0 = now();

    worker_t *ws = xcalloc((size_t)nthreads, sizeof(worker_t));
    for (int i = 0; i < nthreads; i++) {
        ws[i].id = i;
        ws[i].sc = sc;
        primesieve_init(&ws[i].it);
        if (pthread_create(&ws[i].th, NULL, worker_main, &ws[i]) != 0) die("pthread_create failed");
    }

    double last_status = sc->t0, last_ckpt = sc->t0;
    while (atomic_load(&sc->finished) < nthreads) {
        usleep(100000);
        double t = now();
        if (sc->verbose && ((stderr_tty && t - last_status >= 0.5) || (!stderr_tty && t - last_status >= 60))) {
            print_status(sc, t, false);
            last_status = t;
        }
        if (statefile && t - last_ckpt >= ckpt_secs) {
            checkpoint(sc, statefile);
            last_ckpt = t;
        }
    }
    for (int i = 0; i < nthreads; i++) {
        pthread_join(ws[i].th, NULL);
        primesieve_free_iterator(&ws[i].it);
        free(ws[i].c.gb);
        free(ws[i].c.esc);
        free(ws[i].hits);
    }
    free(ws);
    sc->t1 = now();
    if (sc->verbose) print_status(sc, sc->t1, true);
    if (statefile) checkpoint(sc, statefile);
}

/*
 * scan_new / scan_free -- allocate a zeroed scan (report preset from -r) and release it.
 */
static scan_t *scan_new(void)
{
    scan_t *sc = xcalloc(1, sizeof *sc);
    sc->report = g_report;
    return sc;
}

static void scan_free(scan_t *sc)
{
    free(sc->hits);
    free(sc->ghist);
    free(sc);
}

/*
 * a_from_best -- a(n) from the smallest-prime-per-exact-depth table: the minimum of
 * best[d] over d >= n (a prime of depth d counts for every n <= d).
 * Returns: a(n), or 0 if no prime of depth >= n was found.
 */
static u64 a_from_best(const u64 best[], int n)
{
    u64 a = 0;
    for (int d = n; d <= KLIM; d++)
        if (best[d] && (!a || best[d] < a)) a = best[d];
    return a;
}

/*
 * cmp_hit -- qsort comparator ordering hits by p.
 */
static int cmp_hit(const void *x, const void *y)
{
    const hit_t *a = x, *b = y;
    return a->p < b->p ? -1 : a->p > b->p;
}

/* ------------------------------------------------------------------ */
/* Direct (single-threaded) computation of the gap histogram up to P   */
/* ------------------------------------------------------------------ */

/*
 * direct_hist -- the gap histogram of all primes <= P by a plain single-threaded pass
 * (no chunks, no offsets): the reference implementation used by verify and selftest.
 * Inputs:  P (must be prime), cnt (overwritten).
 * Returns: the number of primes <= P; cnt[g] = number of gaps g among them.
 * Errors:  fatal if P is not prime, on a primesieve error, or a gap >= MAXG.
 */
static u64 direct_hist(u64 P, u64 cnt[MAXG])
{
    primesieve_iterator it;
    primesieve_init(&it);
    primesieve_jump_to(&it, 0, P);
    memset(cnt, 0, sizeof(u64) * MAXG);
    u64 pp = 0, q, nprimes = 0;
    while ((q = primesieve_next_prime(&it)) <= P) {
        if (pp) {
            u64 g = q - pp;
            if (g >= MAXG) die("prime gap %" PRIu64 " exceeds MAXG", g);
            cnt[g]++;
        }
        pp = q;
        nprimes++;
    }
    if (it.is_error) die("primesieve error");
    primesieve_free_iterator(&it);
    if (pp != P) die("%" PRIu64 " is not prime (largest prime <= it is %" PRIu64 ")", P, pp);
    return nprimes;
}

/*
 * sums_from_hist -- S[k] = sum_g cnt[g] * g^k for k = 1..kmax (kmax <= KLIM), with
 * ovf[k] set and S[k] = 0 where the value would exceed 2^128.  Errors: none.
 */
static void sums_from_hist(const u64 cnt[MAXG], int kmax, u128 S[], bool ovf[])
{
    for (int k = 1; k <= kmax; k++) {
        u128 s = 0;
        bool o = false;
        for (unsigned g = 1; g < MAXG; g++) {
            u128 prod;
            if (!cnt[g]) continue;
            if (PWOVF[k][g] || __builtin_mul_overflow((u128)cnt[g], PW[k][g], &prod) ||
                __builtin_add_overflow(s, prod, &s)) { o = true; break; }
        }
        S[k] = o ? 0 : s;
        ovf[k] = o;
    }
}

/* ------------------------------------------------------------------ */
/* Commands                                                            */
/* ------------------------------------------------------------------ */

/*
 * usage -- print the command summary on stderr and exit with status 2.
 */
static void usage(void)
{
    fputs(
"usage: a252768 scan [START] END [-t T] [-c CHUNK] [-k KMAX] [-r DEPTH] [-S STATE] [-i SECS] [-q]\n"
"       a252768 verify P [-k KMAX] [-t T]\n"
"       a252768 selftest [-t T]\n"
"\n"
"  scan     exhaustive search of the primes in [START, END] (START defaults to 0);\n"
"           prints every prime p whose first DEPTH power sums S_1(p)..S_DEPTH(p) are\n"
"           prime (default DEPTH 7) and the table of a(n) at the end.  -k sets how\n"
"           many sums are tested (default 12, max 16).  -S FILE writes a checkpoint\n"
"           every SECS seconds (default 60) and on Ctrl-C; running the same command\n"
"           again resumes from it.\n"
"  verify   recompute S_1..S_KMAX at the prime P from scratch and test each one\n"
"           (independently with GMP when compiled in).  -t T > 1 uses the parallel\n"
"           pipeline (totals + gap histogram) instead of a plain single-threaded\n"
"           accumulation; with GMP both modes handle KMAX up to 64.\n"
"  numbers  may be written as 123, 2^40, 10^12, 1e12, 2^40-1\n", stderr);
    exit(2);
}

/*
 * on_signal -- SIGINT/SIGTERM handler: ask the workers to stop taking chunks.  Installed
 * with SA_RESETHAND, so a second signal terminates the process immediately.
 */
static void on_signal(int s)
{
    (void)s;
    g_stop = 1;
}

/*
 * print_summary -- the final report on stdout after a scan.
 * Inputs:  sc after scan_run; exhaustive_from_zero = the computation started at 0, so
 *          a missing depth means a(n) > the scanned bound (otherwise it only means no
 *          such prime in this range).
 * Effect:  prints the range, timing and rate, the counts of primes by exact depth,
 *          then a(n) for n = 1..g_kmax -- marked "(matches OEIS)" / "** DIFFERS **"
 *          for n <= 7 and "NEW" beyond -- or the bound / "not tested" (overflow) line,
 *          and finally the sorted list of hits when there are at most 500.  When the
 *          run was interrupted the bound is the completion frontier.
 * Errors:  none.
 */
static void print_summary(scan_t *sc, bool exhaustive_from_zero)
{
    char b1[32];
    double secs = sc->t1 - sc->t0;
    u64 total_primes = sc->primes0 + sc->primes;
    printf("# scanned [%" PRIu64 ", %" PRIu64 "]%s in %.1f s (%s numbers/s): %" PRIu64 " primes in range"
           " (%" PRIu64 " below %" PRIu64 "), %" PRIu64 " twin primes, %zu hits with n >= %d\n",
           sc->start, g_stop ? sc->start + sc->cf * sc->chunk - 1 : sc->end, g_stop ? " (interrupted)" : "",
           secs, fmt_eng((double)(sc->end - sc->start + 1) / (secs > 0 ? secs : 1), b1),
           sc->primes, sc->primes0, sc->start, sc->twins, sc->nhits, sc->report);
    printf("# primes in range by depth n:");
    for (int d = 1; d <= KLIM; d++)
        if (sc->depth_count[d]) printf("  n=%d: %" PRIu64, d, sc->depth_count[d]);
    printf("\n");
    (void)total_primes;

    int keff = g_kmax;
    for (int k = 1; k <= g_kmax; k++) if (sc->Oovf[k]) { keff = k - 1; break; }
    u64 scanned_to = g_stop ? sc->start + sc->cf * sc->chunk - 1 : sc->end;
    for (int n = 1; n <= g_kmax; n++) {
        u64 a = a_from_best(sc->best, n);
        if (a) {
            const char *tag = "";
            if (n <= NKNOWN) tag = a == KNOWN[n] ? "   (matches OEIS)" : "   ** DIFFERS FROM OEIS **";
            else tag = "   NEW";
            printf("a(%d) = %" PRIu64 "%s\n", n, a, tag);
        } else if (n > keff) {
            printf("a(%d): not tested (S_%d exceeds 2^128 from %" PRIu64 " on)\n", n, keff + 1, sc->ovf_at[keff + 1]);
        } else if (exhaustive_from_zero) {
            printf("a(%d) > %" PRIu64 "\n", n, scanned_to);
        } else {
            printf("a(%d): no prime of depth >= %d in [%" PRIu64 ", %" PRIu64 "]\n", n, n, sc->start, scanned_to);
        }
    }
    if (sc->nhits && sc->nhits <= 500) {
        qsort(sc->hits, sc->nhits, sizeof(hit_t), cmp_hit);
        printf("# hits (p, n) in increasing order:");
        for (size_t i = 0; i < sc->nhits; i++) printf(" (%" PRIu64 ", %d)", sc->hits[i].p, sc->hits[i].depth);
        printf("\n");
    }
    fflush(stdout);
}

/*
 * cmd_scan -- the `scan` command.
 * Inputs:  argv after the command word: one or two bounds plus options
 *          -t threads, -c chunk, -k kmax, -r report depth, -S state file,
 *          -i checkpoint seconds, -q quiet.
 * Flow:    parse; if -S names an existing checkpoint, resume from it (its lo replaces
 *          START, its sums become O0, kmax must match; END may be larger than the
 *          END it was written with); else if START > 2, run a totals-only scan of
 *          [0, START-1] to obtain the sums at START; print the header lines; install
 *          the signal handler; scan_run; print the summary.
 * Returns: 0.  Errors: fatal on bad arguments, an incompatible checkpoint, or
 *          END below the checkpoint position (unless the checkpoint shows that exact
 *          END already complete, which just prints a note).
 */
static int cmd_scan(int argc, char **argv)
{
    u64 pos[2] = { 0, 0 };
    int npos = 0, threads = 0;
    u64 chunk = 0;
    const char *statefile = NULL;
    double ckpt = 60;

    for (int i = 0; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-t") && i + 1 < argc) threads = arg_int(argv[++i], "thread count", 1, 1024);
        else if (!strcmp(a, "-c") && i + 1 < argc) chunk = arg_u64(argv[++i], "chunk size");
        else if (!strcmp(a, "-k") && i + 1 < argc) g_kmax = arg_int(argv[++i], "KMAX", 4, KLIM);
        else if (!strcmp(a, "-r") && i + 1 < argc) g_report = arg_int(argv[++i], "report depth", 1, KLIM);
        else if (!strcmp(a, "-S") && i + 1 < argc) statefile = argv[++i];
        else if (!strcmp(a, "-i") && i + 1 < argc) ckpt = atof(argv[++i]);
        else if (!strcmp(a, "-q")) g_quiet = true;
        else if (a[0] == '-' && !isdigit((unsigned char)a[1])) die("unknown option '%s'", a);
        else if (npos < 2) pos[npos++] = arg_u64(a, "bound");
        else usage();
    }
    if (npos == 0) usage();
    if (chunk && chunk < 1000) die("chunk size too small");
    if (ckpt < 1) ckpt = 1;
    if (threads <= 0) {
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        threads = n > 0 ? (int)n : 4;
    }

    u64 start = npos == 2 ? pos[0] : 0, end = pos[npos - 1], origin = start;
    scan_t *sc = scan_new();
    sc->report = g_report;
    bool resumed = false;
    state_t st;

    if (statefile && read_state(statefile, &st)) {
        if (st.kmax != g_kmax) die("%s was written with -k %d; use the same value", statefile, st.kmax);
        if (npos == 2 && pos[0] != st.lo)
            note("note: START %" PRIu64 " ignored, resuming from checkpoint at %" PRIu64, pos[0], st.lo);
        if (end < st.lo) {
            if (end == st.end) { note("%s: scan of [%" PRIu64 ", %" PRIu64 "] already complete", statefile, st.origin, st.end); return 0; }
            die("END %" PRIu64 " is below the checkpoint position %" PRIu64, end, st.lo);
        }
        start = st.lo;
        origin = st.origin;
        memcpy(sc->O0, st.O, sizeof sc->O0);
        memcpy(sc->O0ovf, st.ovf, sizeof sc->O0ovf);
        memcpy(sc->best, st.best, sizeof sc->best);
        sc->primes0 = st.primes;
        resumed = true;
        note("resuming from %s at %" PRIu64 " (%" PRIu64 " primes done)", statefile, start, st.primes);
    } else if (start > 2) {
        /* no checkpoint: a totals-only pass over [0, START-1] gives the power sums at START */
        scan_t *pre = scan_new();
        pre->origin = 0; pre->start = 0; pre->end = start - 1;
        pre->check = false; pre->verbose = !g_quiet; pre->print_hits = false;
        pre->chunk = chunk;
        if (!g_quiet) fprintf(stderr, "computing the power sums below %" PRIu64 " ...\n", start);
        scan_run(pre, threads, NULL, 0);
        if (g_stop) die("interrupted");
        memcpy(sc->O0, pre->O, sizeof sc->O0);
        memcpy(sc->O0ovf, pre->Oovf, sizeof sc->O0ovf);
        sc->primes0 = pre->primes_pub;
        scan_free(pre);
    }

    if (!chunk) {
        u64 range = end >= start ? end - start + 1 : 0;
        chunk = range / ((u64)threads * 32);
        if (chunk < 1000000) chunk = 1000000;
        if (chunk > 1000000000) chunk = 1000000000;
    }
    sc->origin = origin;
    sc->start = start;
    sc->end = end;
    sc->chunk = chunk;
    sc->check = true;
    sc->verbose = !g_quiet;
    sc->print_hits = true;

    char b[48];
    printf("# A252768: a(n) = least prime p such that S_1(p)..S_n(p) are prime, S_k(p) = sum of k-th powers of the prime gaps up to p\n");
    printf("# scanning primes in [%" PRIu64 ", %" PRIu64 "] with %d threads, chunk %" PRIu64 ", testing k <= %d, reporting depth >= %d%s\n",
           start, end, threads, chunk, g_kmax, g_report, resumed ? " (resumed)" : "");
    if (start > 2) {
        printf("# power sums at start:");
        for (int k = 1; k <= g_kmax; k++) printf(" S_%d=%s", k, sc->O0ovf[k] ? "overflow" : u128_str(sc->O0[k], b));
        printf("\n");
    }
    fflush(stdout);

    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_signal;
    sa.sa_flags = SA_RESETHAND;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    scan_run(sc, threads, statefile, ckpt);
    if (g_stop) note("interrupted; all chunks below %" PRIu64 " are complete%s", start + sc->cf * sc->chunk,
                     statefile ? ", checkpoint written" : "");
    print_summary(sc, origin == 0);
    scan_free(sc);
    return 0;
}

/*
 * cmd_verify -- the `verify` (alias `depth`) command: recompute and test S_1..S_kmax
 * at one prime P independently of the scan.
 * Inputs:  argv after the command word: P, -k kmax (up to 64 with GMP, else <= KLIM),
 *          -t threads (1 = direct single-threaded histogram, > 1 = the parallel totals
 *          pipeline with a merged gap histogram; it shares the chunk code with the scan,
 *          and with GMP it too handles any k, cross-checking GMP's sums against the
 *          128-bit ones for k <= KLIM).
 * Effect:  prints P, the prime count, then one line per k with S_k, its bit length,
 *          the tool's verdict ([mr]) and GMP's ([gmp: ...]), flagging any disagreement;
 *          with GMP the sums are also formed in GMP and compared with the 128-bit ones.
 *          Ends with the depth of P.  Sums beyond 128 bits are reported as not
 *          computed unless GMP is available (direct mode only).
 * Returns: 0.  Errors: fatal if P is not prime or an option is invalid; a mismatch
 *          between the 128-bit and GMP sums is treated as an internal error.
 */
static int cmd_verify(int argc, char **argv)
{
    u64 P = 0;
    int kmax = g_kmax, threads = 1;
    bool have_p = false;
    for (int i = 0; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-k") && i + 1 < argc) kmax = arg_int(argv[++i], "KMAX", 1, 64);
        else if (!strcmp(a, "-t") && i + 1 < argc) threads = arg_int(argv[++i], "thread count", 1, 1024);
        else if (a[0] == '-' && !isdigit((unsigned char)a[1])) die("unknown option '%s'", a);
        else { P = arg_u64(a, "prime"); have_p = true; }
    }
    if (!have_p) usage();
    if (!is_prime64(P)) die("%" PRIu64 " is not prime", P);
#ifndef HAVE_GMP
    if (kmax > KLIM) die("without GMP at most k = %d is supported", KLIM);
#endif

    double t0 = now();
    u64 cnt[MAXG];
    u64 nprimes;
    u128 S[KLIM + 1];
    bool ovf[KLIM + 1];
    bool have_cnt = false;
    if (threads <= 1) {
        nprimes = direct_hist(P, cnt);
        have_cnt = true;
        sums_from_hist(cnt, kmax <= KLIM ? kmax : KLIM, S, ovf);
    } else {
        /* parallel: the totals pipeline gives S_k for k <= KLIM in 128 bits, and the merged
         * gap histogram lets GMP form S_k exactly for any k (and cross-check the 128-bit sums) */
        int save = g_kmax;
        g_kmax = kmax <= KLIM ? kmax : KLIM;
        scan_t *sc = scan_new();
        sc->start = 0; sc->end = P; sc->check = false; sc->verbose = !g_quiet;
        sc->ghist = xcalloc(MAXG, sizeof(u64));
        scan_run(sc, threads, NULL, 0);
        nprimes = sc->primes_pub;
        memcpy(S, sc->O, sizeof S);
        memcpy(ovf, sc->Oovf, sizeof ovf);
        memcpy(cnt, sc->ghist, sizeof cnt);
        have_cnt = true;
        scan_free(sc);
        g_kmax = save;
    }
    double secs = now() - t0;

    char b[48];
    printf("p = %" PRIu64 " is prime; %" PRIu64 " primes <= p, %" PRIu64 " gaps (%.1f s)\n", P, nprimes, nprimes - 1, secs);
    int depth = 0;
    bool stopped = false;
#ifdef HAVE_GMP
    mpz_t z, t;
    mpz_init(z); mpz_init(t);
#endif
    for (int k = 1; k <= kmax; k++) {
        bool own_known = k <= KLIM && !ovf[k];
        bool own_prime = own_known && is_prime128(S[k]);
        int gmp_res = -1;
        char *dec = NULL;
#ifdef HAVE_GMP
        if (have_cnt) {
            mpz_set_ui(z, 0);
            for (unsigned g = 1; g < MAXG; g++) {
                if (!cnt[g]) continue;
                mpz_ui_pow_ui(t, g, (unsigned long)k);
                mpz_mul_ui(t, t, (unsigned long)cnt[g]);
                mpz_add(z, z, t);
            }
        } else if (own_known) {
            mpz_set_u128(z, S[k]);
        } else {
            mpz_set_ui(z, 0);
        }
        if (have_cnt || own_known) {
            gmp_res = mpz_probab_prime_p(z, 40);
            dec = mpz_get_str(NULL, 10, z);
            if (have_cnt && own_known) {
                mpz_t w;
                mpz_init(w);
                mpz_set_u128(w, S[k]);
                if (mpz_cmp(w, z) != 0) die("internal error: 128-bit and GMP sums differ for k = %d", k);
                mpz_clear(w);
            }
        }
#endif
        if (!dec && own_known) { dec = xmalloc(48); u128_str(S[k], dec); }
        bool prime = own_known ? own_prime : gmp_res > 0;
        if (!dec) {
            printf("S_%-2d exceeds 2^128 (not computed)\n", k);
            stopped = true;
            break;
        }
        int bits = own_known ? bitlen128(S[k]) : 0;
#ifdef HAVE_GMP
        if (!own_known) bits = (int)mpz_sizeinbase(z, 2);
#endif
        printf("S_%-2d = %s  (%d bits)  %s", k, dec, bits, prime ? "prime" : "composite");
        if (own_known) printf("  [mr]");
        if (gmp_res >= 0) printf("  [gmp: %s]", gmp_res == 2 ? "prime" : gmp_res == 1 ? "probable prime" : "composite");
        if (own_known && gmp_res >= 0 && own_prime != (gmp_res > 0)) printf("  ** DISAGREEMENT **");
        printf("\n");
        free(dec);
        if (prime && depth == k - 1) depth = k;
    }
#ifdef HAVE_GMP
    mpz_clear(z); mpz_clear(t);
#endif
    printf("depth(%" PRIu64 ") = %d: S_1..S_%d are prime%s\n", P, depth, depth,
           depth == kmax ? (stopped ? "" : " (all tested k)") : "");
    (void)b;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Selftest                                                            */
/* ------------------------------------------------------------------ */

static int st_fail;     /* number of failed selftest checks */

/*
 * expect -- record one selftest check: prints "ok" or "FAIL" with the description and
 * counts failures.  Inputs: ok, printf-style description.
 */
static void expect(bool ok, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    printf("  %s  ", ok ? "ok  " : "FAIL");
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
    if (!ok) st_fail++;
}

/*
 * rng -- xorshift64* pseudo-random generator with a fixed seed, so the selftest is
 * reproducible.  Returns 64 random bits.
 */
static u64 rng_state = 0x9E3779B97F4A7C15ull;
static u64 rng(void)
{
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return rng_state * 0x2545F4914F6CDD1Dull;
}

/*
 * quiet_scan -- run a checking scan of [start, end] with no output, for the selftest.
 * Inputs:  chunk (0 = automatic), threads, report threshold, and optionally the
 *          running sums O0/O0ovf and prime count at start (NULL = start from 0).
 * Returns: the scan with its results; the caller frees it with scan_free.
 */
static scan_t *quiet_scan(u64 start, u64 end, u64 chunk, int threads, int report, const u128 *O0, const bool *O0ovf, u64 primes0)
{
    scan_t *sc = scan_new();
    sc->origin = 0; sc->start = start; sc->end = end; sc->chunk = chunk;
    sc->check = true; sc->verbose = false; sc->print_hits = false; sc->report = report;
    if (O0) { memcpy(sc->O0, O0, sizeof sc->O0); memcpy(sc->O0ovf, O0ovf, sizeof sc->O0ovf); sc->primes0 = primes0; }
    scan_run(sc, threads, NULL, 0);
    return sc;
}

/*
 * quiet_totals -- run a totals-only scan of [0, end] with no output; on return sc->O
 * holds S_k at the largest prime <= end.  Caller frees.
 */
static scan_t *quiet_totals(u64 end, u64 chunk, int threads)
{
    scan_t *sc = scan_new();
    sc->origin = 0; sc->start = 0; sc->end = end; sc->chunk = chunk;
    sc->check = false; sc->verbose = false; sc->print_hits = false;
    sc->ghist = xcalloc(MAXG, sizeof(u64));
    scan_run(sc, threads, NULL, 0);
    return sc;
}

/*
 * same_hits -- true iff two scans recorded exactly the same (p, depth) hits.  Sorts
 * both hit lists as a side effect.
 */
static bool same_hits(scan_t *a, scan_t *b)
{
    if (a->nhits != b->nhits) return false;
    qsort(a->hits, a->nhits, sizeof(hit_t), cmp_hit);
    qsort(b->hits, b->nhits, sizeof(hit_t), cmp_hit);
    for (size_t i = 0; i < a->nhits; i++)
        if (a->hits[i].p != b->hits[i].p || a->hits[i].depth != b->hits[i].depth) return false;
    return true;
}

/*
 * cmd_selftest -- the `selftest` command (option: -t threads).
 * Checks, in order: the power tables; Miller-Rabin on known primes, Carmichael
 * numbers and strong pseudoprimes (including the ones that fool the first 12 and 13
 * prime bases); with GMP, agreement on 20000 random 64-bit and 3000 random 65..127-bit
 * numbers plus 300 products of two primes; the OEIS example p = 13; the depth of
 * a(7) by direct accumulation and by the parallel totals pipeline; reproduction of
 * a(1..7) by a chunked scan to 6e8; identical hits and sums for three different chunk
 * sizes; a scan starting at 3e8 after a totals pass; and a checkpoint round trip.
 * Returns: 0 if everything passed, 1 otherwise (so `make test` fails loudly).
 */
static int cmd_selftest(int argc, char **argv)
{
    int threads = 0;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "-t") && i + 1 < argc) threads = arg_int(argv[++i], "thread count", 1, 1024);
        else die("unknown option '%s'", argv[i]);
    }
    if (threads <= 0) {
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        threads = n > 0 ? (int)n : 4;
    }
    g_kmax = 12;
    char b1[48], b2[48];
    double t0 = now();
    printf("a252768 selftest (%d threads)\n", threads);

    /* 1. power tables */
    expect(PW[3][2] == 8 && PW[12][2] == 4096 && PW[1][MAXG - 1] == MAXG - 1 &&
           !PWOVF[12][1550] && PWOVF[13][1550] && PWOVF[11][MAXG - 1], "power tables");

    /* 1b. command-line number syntax */
    {
        static const struct { const char *s; const char *want; } NT[] = {
            { "123", "123" }, { "2^40", "1099511627776" }, { "10^12", "1000000000000" },
            { "1e12", "1000000000000" }, { "25e9", "25000000000" }, { "2.5e10", "25000000000" },
            { "0.5e3", "500" }, { "2^40-1", "1099511627775" }, { "1e3+7", "1007" },
        };
        static const char *BAD[] = { "2.5e0", "1.e3", ".e3", "abc", "1e41", "2^", "-5", "" };
        bool ok = true;
        for (size_t i = 0; i < sizeof NT / sizeof NT[0]; i++) {
            u128 v, w;
            if (!parse_u128(NT[i].s, &v) || !parse_dec(NT[i].want, strlen(NT[i].want), &w) || v != w) {
                ok = false; printf("      parse '%s' wrong\n", NT[i].s);
            }
        }
        for (size_t i = 0; i < sizeof BAD / sizeof BAD[0]; i++) {
            u128 v;
            if (parse_u128(BAD[i], &v)) { ok = false; printf("      parse '%s' should fail\n", BAD[i]); }
        }
        expect(ok, "number syntax: 2^40, 10^12, 1e12, 25e9, 2.5e10, 2^40-1 and rejection of bad forms");
    }

    /* 2. primality on known values */
    static const struct { const char *s; bool prime; } PT[] = {
        { "2", true }, { "3", true }, { "4", false }, { "5", true }, { "9", false }, { "561", false },
        { "997", true }, { "1009", true }, { "2047", false }, { "1000003", true }, { "4294967291", true },
        { "4294967297", false }, { "3215031751", false }, { "3825123056546413051", false },
        { "18446744073709551557", true }, { "18446744073709551615", false },
        { "18446744073709551629", true },                                    /* 2^64 + 13 */
        { "318665857834031151167461", false },                               /* spsp to bases 2..37 */
        { "3317044064679887385961981", false },                              /* spsp to bases 2..41 */
        { "618970019642690137449562111", true },                             /* 2^89 - 1 */
        { "1856910058928070412348686333", false },                           /* 3 (2^89 - 1) */
        { "162259276829213363391578010288127", true },                       /* 2^107 - 1 */
        { "5316911983139663487003542222693990401", false },                  /* (2^61 - 1)^2 */
        { "170141183460469231731687303715884105727", true },                 /* 2^127 - 1 */
    };
    bool okp = true;
    for (size_t i = 0; i < sizeof PT / sizeof PT[0]; i++) {
        u128 n;
        if (!parse_u128(PT[i].s, &n)) die("selftest: bad literal");
        if (is_prime128(n) != PT[i].prime) { okp = false; printf("      wrong verdict for %s\n", PT[i].s); }
    }
    expect(okp, "Miller-Rabin on %zu known primes/composites", sizeof PT / sizeof PT[0]);

#ifdef HAVE_GMP
    {
        int bad = 0, nprime = 0;
        for (int i = 0; i < 20000; i++) {
            u64 n = rng() | 1;
            n >>= (rng() % 40);
            bool a = is_prime64(n), g = gmp_isprime128(n) > 0;
            if (a != g) bad++;
            nprime += a;
        }
        expect(bad == 0, "64-bit Miller-Rabin agrees with GMP on 20000 random odd numbers (%d primes)", nprime);
        bad = 0; nprime = 0;
        for (int i = 0; i < 3000; i++) {
            int bits = 65 + (int)(rng() % 63);
            u128 n = ((u128)rng() << 64) | rng();
            n >>= (128 - bits);
            n |= ((u128)1 << (bits - 1)) | 1;
            bool a = is_prime128(n), g = gmp_isprime128(n) > 0;
            if (a != g) bad++;
            nprime += a;
        }
        expect(bad == 0, "128-bit Miller-Rabin agrees with GMP on 3000 random odd numbers of 65..127 bits (%d primes)", nprime);
        bad = 0;
        for (int i = 0; i < 300; i++) {
            u64 p = (rng() >> 20) | 1, q = (rng() >> 24) | 1;
            while (!is_prime64(p)) p += 2;
            while (!is_prime64(q)) q += 2;
            u128 n = (u128)p * q;
            if (is_prime128(n) || gmp_isprime128(n) > 0) bad++;
        }
        expect(bad == 0, "products of two random primes (40-44 bits each) are recognised as composite");
    }
#endif

    /* 3. the OEIS example and a(7) by direct accumulation */
    {
        u64 cnt[MAXG];
        u128 S[KLIM + 1];
        bool ovf[KLIM + 1];
        direct_hist(13, cnt);
        sums_from_hist(cnt, 4, S, ovf);
        expect(S[1] == 11 && S[2] == 29 && S[3] == 89 && S[4] == 305, "p = 13: S_1..S_4 = 11, 29, 89, 305");
        direct_hist(541613713, cnt);
        sums_from_hist(cnt, 12, S, ovf);
        int depth = 0;
        for (int k = 1; k <= 12; k++) { if (ovf[k] || !is_prime128(S[k])) break; depth = k; }
        expect(depth == 7, "p = 541613713 = a(7) has depth exactly 7 (S_8 = %s is composite)", u128_str(S[8], b1));
        printf("        S_1..S_8 =");
        for (int k = 1; k <= 8; k++) printf(" %s", u128_str(S[k], b1));
        printf("\n");

        /* the parallel totals pipeline must reproduce these sums */
        scan_t *tot = quiet_totals(541613713, 777777, threads);
        bool same = tot->primes_pub == direct_hist(541613713, cnt);
        for (int k = 1; k <= 12; k++) if (tot->Oovf[k] || tot->O[k] != S[k]) same = false;
        for (unsigned g = 0; g < MAXG; g++) if (tot->ghist[g] != cnt[g]) same = false;
        expect(same, "parallel totals pipeline (chunk 777777) reproduces S_1..S_12, the gap histogram and the prime count %" PRIu64 " at 541613713", tot->primes_pub);
        scan_free(tot);
    }

    /* 4. the sequence itself */
    {
        scan_t *sc = quiet_scan(0, 600000000, 1000000, threads, 3, NULL, NULL, 0);
        bool ok = true;
        for (int n = 1; n <= NKNOWN; n++) {
            u64 a = a_from_best(sc->best, n);
            if (a != KNOWN[n]) { ok = false; printf("      a(%d) = %" PRIu64 ", OEIS has %" PRIu64 "\n", n, a, KNOWN[n]); }
        }
        expect(ok, "scan [0, 6e8] (chunk 1e6) reproduces a(1..7) = 5, 5, 13, 14593, 372313, 2315773, 541613713");
        printf("        %" PRIu64 " primes, %" PRIu64 " twins, %zu primes of depth >= 3, %.1f s\n", sc->primes, sc->twins, sc->nhits, sc->t1 - sc->t0);
        u64 a8 = a_from_best(sc->best, 8);
        printf("        a(8) %s 6e8\n", a8 ? "<=" : ">");
        if (a8) printf("        a(8) = %" PRIu64 "\n", a8);

        /* 5. chunking independence */
        scan_t *s2 = quiet_scan(0, 300000000, 300000000, 1, 3, NULL, NULL, 0);
        scan_t *s3 = quiet_scan(0, 300000000, 777777, threads, 3, NULL, NULL, 0);
        scan_t *s4 = quiet_scan(0, 300000000, 1000000, threads, 3, NULL, NULL, 0);
        bool same = same_hits(s2, s3) && same_hits(s2, s4) && s2->primes == s3->primes && s3->primes == s4->primes;
        for (int k = 1; k <= 12; k++) if (s2->O[k] != s3->O[k] || s3->O[k] != s4->O[k]) same = false;
        expect(same, "scan [0, 3e8] gives identical hits and sums with one chunk, chunk 777777 and chunk 1e6 (%zu hits of depth >= 3)", s2->nhits);

        /* 6. a scan that starts in the middle (offsets from a totals pass) */
        scan_t *tot = quiet_totals(299999999, 0, threads);
        scan_t *s5 = quiet_scan(300000000, 600000000, 1000000, threads, 3, tot->O, tot->Oovf, tot->primes_pub);
        size_t nabove = 0;
        bool sub = true;
        qsort(sc->hits, sc->nhits, sizeof(hit_t), cmp_hit);
        qsort(s5->hits, s5->nhits, sizeof(hit_t), cmp_hit);
        for (size_t i = 0; i < sc->nhits; i++) {
            if (sc->hits[i].p < 300000000) continue;
            if (nabove >= s5->nhits || s5->hits[nabove].p != sc->hits[i].p || s5->hits[nabove].depth != sc->hits[i].depth) sub = false;
            nabove++;
        }
        if (nabove != s5->nhits) sub = false;
        expect(sub && a_from_best(s5->best, 7) == KNOWN[7], "scan [3e8, 6e8] after a totals pass finds exactly the hits above 3e8 (%zu) including a(7)", s5->nhits);
        scan_free(sc); scan_free(s2); scan_free(s3); scan_free(s4); scan_free(tot); scan_free(s5);
    }

    /* 7. checkpoint round trip */
    {
        state_t st, st2;
        memset(&st, 0, sizeof st);
        st.kmax = 12; st.origin = 0; st.end = 12345678901ull; st.lo = 4000000000ull; st.primes = 189961812;
        for (int k = 1; k <= 12; k++) { st.O[k] = ((u128)rng() << 64) | rng(); st.ovf[k] = k == 12; }
        st.best[7] = 541613713; st.best[3] = 13;
        const char *path = "/tmp/a252768-selftest.state";
        write_state(path, &st);
        bool ok = read_state(path, &st2) && st2.kmax == 12 && st2.end == st.end && st2.lo == st.lo && st2.primes == st.primes &&
                  st2.best[7] == 541613713 && st2.best[3] == 13 && st2.best[5] == 0;
        for (int k = 1; k <= 12; k++) if (st2.O[k] != st.O[k] || st2.ovf[k] != st.ovf[k]) ok = false;
        unlink(path);
        expect(ok, "checkpoint file round trip");
    }

    (void)b2;
    printf("%s (%.1f s)\n", st_fail ? "SELFTEST FAILED" : "all tests passed", now() - t0);
    return st_fail ? 1 : 0;
}

/* ------------------------------------------------------------------ */

/*
 * main -- line-buffer stdout (so `tee` and logs see hits immediately), detect whether
 * stderr is a terminal, build the power tables, and dispatch on the command word.
 * Returns: the command's status; 2 for a usage error.
 */
int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    stderr_tty = isatty(2);
    init_pow_tables();
    if (argc < 2) usage();
    const char *cmd = argv[1];
    if (!strcmp(cmd, "scan")) return cmd_scan(argc - 2, argv + 2);
    if (!strcmp(cmd, "verify") || !strcmp(cmd, "depth")) return cmd_verify(argc - 2, argv + 2);
    if (!strcmp(cmd, "selftest")) return cmd_selftest(argc - 2, argv + 2);
    usage();
    return 2;
}
