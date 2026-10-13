/*
 * a359636.c  --  compute and extend OEIS A359636 (CPU version)
 *
 *   "a(n) is the least odd prime not in A001359 such that all subsequent
 *    composites in the gap up to the next prime have at least n distinct
 *    prime factors."
 *
 * Status (October 2026)
 *   a(1..8) = 7, 19, 643, 51427, 8083633, 1077940147, 75582271489,
 *             34710483181813 (OEIS; all reproduced by this program).
 *   a(9)    = 31610535900218923, found in Sep 2026 by the CUDA port of this
 *             program (cuda/a359636_cuda.cu); in the OEIS entry since
 *             Sep 18 2026.
 *   a(10)   > 408697968217030657 (level-10 GPU scan, paused Sep 23 2026);
 *             a(10) <= 225096507194749219819 (David A. Corneth, 2023).
 *   See README.md for the full record and verify_a359636.py for an
 *   independent Python check.
 *
 * ==========================================================================
 * THEORY OF OPERATION
 * ==========================================================================
 *
 * 1. Definitions
 *
 *    omega(x) is the number of distinct prime factors of x (A001221).  Let
 *    p < q be consecutive primes.  The gap (p, q) QUALIFIES AT LEVEL n when
 *    q - p >= 4 (so p is not the smaller member of a twin pair, A001359) and
 *    omega(x) >= n for every x with p < x < q.  a(n) is the smallest prime p
 *    whose gap qualifies.  A gap that qualifies at level n + 1 also qualifies
 *    at level n, so a(n) is non-decreasing.  Every known term is a gap of 4,
 *    e.g. a(8) = 34710483181813:
 *      34710483181814 = 2*11*17*29*47*229*409*727
 *      34710483181815 = 3^2*5*7*19*31*101*263*7043
 *      34710483181816 = 2^3*13*41*43*61*83*139*269,   34710483181817 prime.
 *
 * 2. Key reduction: every qualifying gap contains a "triple" around an odd
 *    multiple of 3
 *
 *    For a prime p > 3 let m0(p) be the smallest odd multiple of 3 above p:
 *        p = 1 (mod 6)  ->  m0 = p + 2,        p = 5 (mod 6)  ->  m0 = p + 4.
 *    In every gap of length >= 4 the numbers m0-1, m0, m0+1 lie strictly
 *    inside the gap (a gap of exactly 4 forces p = 1 (mod 6) because p + 4
 *    is prime; a longer gap contains p+1 .. p+5).  So a qualifying gap
 *    contains an m = 3 (mod 6) with
 *
 *        omega(m-1) >= n,   omega(m) >= n,   omega(m+1) >= n.          (*)
 *
 *    The program enumerates m = 6k + 3 (k = 1, 2, ...), keeps the m that
 *    satisfy (*) ("triples"), and for each triple checks the whole gap that
 *    starts at p = prevprime(m-1).  For n >= 2 the numbers m-1, m, m+1 are
 *    composite, so that is exactly the gap containing m.  (For n = 1, m-1 may
 *    be prime; the gap checked is then the one ending at m-1, which is still
 *    a genuine gap, so a(1) = 7 is unaffected.)
 *
 *    Coverage lemma.  Suppose every m = 3 (mod 6) with m <= M has been
 *    examined (M = 3 (mod 6), M >= 9).  Then every qualifying gap with
 *    p <= M - 2 has been found: for p = 1 (mod 6), m0 = p + 2 <= M; for
 *    p = 5 (mod 6) even p <= M - 4 holds, so m0 = p + 4 <= M.  M - 1 and
 *    M + 1 are even and 3 divides M, so no prime lies in (M - 2, M + 2) and
 *    the statement holds for p <= M + 1 as well.  The same argument shows
 *    that the smallest solution found is a(n) as soon as every m below its
 *    own m has been examined: a smaller qualifying gap ends before p, so its
 *    m0 < p.  A scan "to END" examines m <= END + 2; by the lemma that covers
 *    every p <= END except a prime p = END (END = 5 (mod 6)) or p = END - 1
 *    (END = 0 (mod 6)), whose m0 = p + 4 lies just beyond.  run_scan() states
 *    its final bound accordingly.
 *
 * 3. Sieving the three targets at once
 *
 *    Index by k = (m - 3) / 6.  The targets are N_t(k) = 6k + 3 + t for
 *    t = -1, 0, +1 (that is m-1, m, m+1).  For a prime p >= 5, p divides
 *    N_t(k) exactly when k = r_t(p) (mod p), r_t(p) = -(3 + t) * 6^-1 mod p
 *    (target_residue()).  Every k owns one 32-bit counter word whose bytes
 *    0, 1, 2 count the sieve primes dividing m-1, m, m+1:
 *
 *        word(k) = cnt(m-1) + (cnt(m) << 8) + (cnt(m+1) << 16).
 *
 *    No 64-bit number has 16 distinct prime factors, so a byte never exceeds
 *    15 and bytes never carry into each other.  The forced small factors are
 *    not sieved: m-1 and m+1 are even (and prime to 3), m is a multiple of 3
 *    (and odd).  Each target therefore has exactly one prime <= T that its
 *    counter does not see, and the threshold below is c - 1, not c.
 *
 *    The primes 5..37 are applied by copying/adding three precomputed
 *    periodic patterns (periods 5*7*11*13 = 5005, 17*19*23 = 7429 and
 *    29*31*37 = 33263; setup_sieve()).  The primes 41..T are marked one
 *    arithmetic progression at a time; the three progressions of a prime
 *    share the stride p and are marked in one interleaved loop.  The work
 *    proceeds in segments of SEG consecutive k whose SEG counter words stay
 *    in the L1/L2 cache.
 *
 * 4. Branch-free threshold test
 *
 *    With thr = c - 1 (section 5), adding 128 - thr to a byte sets its top
 *    bit exactly when the count is >= thr (counts are <= 15: no carry).  A k
 *    "survives" when the top bits of all three bytes are set:
 *
 *        (~(word + 0x010101 * (128 - thr))) & 0x808080 == 0.
 *
 *    worker() first tests blocks of 16 words with a min() reduction (which
 *    the compiler vectorizes) and examines single words only in blocks that
 *    contain a survivor.  thr <= 0 (tiny levels) makes every k a survivor.
 *
 * 5. How many small prime factors may be demanded (compute_c())
 *
 *    Let Bmax be the largest number examined (END + 3).  If N <= Bmax has n
 *    distinct prime factors, r of them above T, then N >= prod(r) =
 *    (product of the first n-r primes) * (product of the first r primes
 *    above T).  With rmax the largest r such that prod(r) <= Bmax, every
 *    candidate has at least c = n - rmax distinct prime factors <= T, in
 *    each of the three targets.  (prod(r+1)/prod(r) = q_(r+1)/p_(n-r) grows
 *    with r, so prod is log-convex in r: after the first r with
 *    prod(r) > Bmax every larger r fails too, and compute_c() stops there.)
 *    Example: n = 9, Bmax = 7.7e16, T = 1000 gives c = 5; near 7e16 only
 *    about 2e-5 of all k survive, so the sieve (about 2 counter increments
 *    and ~2.2 cycles per k on an M1 core) is essentially the whole cost.
 *    Without -T the program picks the smallest T from a fixed list that
 *    gives c >= min(n, 5).
 *
 * 6. Exact verification of survivors
 *
 *    omega_at_least() decides omega(N) >= n by trial division with a size
 *    bound (a cofactor that still needs r distinct primes, all >= p, must be
 *    >= p^r) and settles the last two primes with Miller-Rabin plus a
 *    perfect-power test.  check_survivor() applies it to m-1, m, m+1; for a
 *    triple it prints a "triple" line, finds p = prevprime(m-1) with
 *    deterministic Miller-Rabin and checks every number in the gap
 *    (gap_qualifies()).  A qualifying gap is printed as a SOLUTION block
 *    with the full factorization (trial division + Pollard rho) of each
 *    composite.
 *
 * 7. Threads, frontier, early stop, checkpoint
 *
 *    The k range [k_lo, k_hi) is cut into chunks of CHUNK k (default 2^26,
 *    a multiple of SEG).  Worker threads take chunk indices from an atomic
 *    counter, sieve and verify each chunk on their own, and report it to
 *    chunk_done(), which records completion in a ring of flags and advances
 *    the FRONTIER: the lowest chunk not yet complete.  Everything below the
 *    frontier is final, so the frontier drives
 *      - the status line (progress()),
 *      - the checkpoint (-S FILE, written by the main thread every -i
 *        seconds and at exit; format in save_state()), and
 *      - the early stop: once the frontier has passed the chunk holding the
 *        smallest solution, that solution is certain (Coverage lemma) and
 *        the scan ends, unless -a asks for every solution in the range.
 *    A worker never runs more than RING-1 chunks ahead of the frontier, so a
 *    completion flag is never overwritten before it has been consumed.
 *    SIGINT/SIGTERM set g_stop: the workers finish the chunk in hand and stop,
 *    the main thread writes the checkpoint and the summary, and rerunning the
 *    same command resumes at the frontier.  Chunks finished beyond the
 *    frontier are redone after a resume, so their triple lines can appear
 *    twice and the survivor/triple counts are then slightly too high.
 *
 * 8. Hunt mode (cmd_hunt())
 *
 *    A cheap, non-exhaustive search for upper bounds.  It enumerates only
 *    m = 3^e * p_1 * ... * p_(n-1) (e = 1 or 2; distinct primes
 *    5 <= p_i <= L; with -Q the last one may be any prime <= Q), which have
 *    omega(m) = n by construction, and tests m-1, m+1 and the gap as above.
 *    Every SOLUTION it prints is a genuine qualifying gap, hence an upper
 *    bound for a(n), but a(n) itself may be missed (a(9)'s m has a square
 *    factor 5^2, which no hunt variant enumerates).
 *
 * 9. Output
 *
 *    stdout  "triple ..." lines (every m satisfying (*)), "SOLUTION ..."
 *            blocks, and one result line at the end of a scan:
 *              a(N) = P                              scan from 0, P certain
 *              smallest qualifying p with m in [A, B] is P (partial range...)
 *              a(N) <= P (scan incomplete below it)
 *              a(N) > B (no qualifying gap with p <= B)        complete scan
 *              a(N) > B (scan stopped early; ...)
 *              no qualifying gap with m in [A, B] (partial range...)
 *            A scan with START > 9 is "partial": it speaks only about the m
 *            in its own range.
 *    stderr  start-up parameters, a status line (redrawn every second on a
 *            terminal, one line per minute otherwise) and the final
 *            "scanned m <= M: ..." summary.
 *
 * 10. Limits, error handling, known imperfections
 *
 *    - All arithmetic is 64-bit with 128-bit intermediate products.
 *      run_scan() rejects END > 2^64 - 65536 so that m+1 and the next prime
 *      stay below 2^64 (no prime gap below 2^64 exceeds 1550).  'scan'
 *      accepts N = 1..20; N >= 16 can never succeed because no 64-bit number
 *      has 16 distinct prime factors.
 *    - Fatal conditions (bad arguments, out of memory, primesieve failure, an
 *      unreadable or mismatched checkpoint, a failed pthread_create) go
 *      through die(): message on stderr, exit status 1.  usage() exits with
 *      status 2.  'verify' exits 0 when the gap qualifies and 1 otherwise;
 *      'selftest' exits 0 when every check passes.
 *    - A checkpoint write that fails (e.g. disk full) is reported on stderr
 *      and the previous checkpoint is kept.
 *    - Benign data races: workers read S->stop_now and progress() reads
 *      S->frontier without fr_mu, and chunk_done() reads best_p/best_m
 *      (written under sol_mu) while holding fr_mu.  These are aligned
 *      word-sized loads on the supported 64-bit targets; a stale value only
 *      delays the stop by a chunk or shows an old number, and since best_m
 *      only decreases, a stale value can never stop a scan early.
 *    - Numeric options go through atoi() or parse_num() and are only partly
 *      validated (e.g. -T is truncated to 32 bits; -t 0 becomes 1).
 *
 * Usage
 * -----
 *   a359636 scan N [START] END [-t T] [-c CHUNK] [-T SIEVEMAX] [-S STATE] [-i SECS] [-a] [-q]
 *       Search m = 3 (mod 6) with START <= m <= END+2 for level N; prints every
 *       triple satisfying (*), every qualifying gap, and a result line.
 *       -t threads (default: all CPUs), -c k per chunk, -T sieve bound (default
 *       automatic), -S FILE keeps a checkpoint (every SECS seconds, default 60,
 *       and on Ctrl-C; rerunning the same command resumes from it), -a reports
 *       every solution instead of stopping at the first certain one, -q quiet.
 *   a359636 hunt N END [-L L] [-Q Q] [-t T] [-q]
 *       Structured search for upper bounds: tests every m = 3^e * (N-1 distinct
 *       primes in [5, L]) <= END+2 (default L = 1000); with -Q the largest of
 *       those primes may instead be any prime <= Q.  Cheap, not exhaustive.
 *   a359636 verify N P
 *       Check directly that the gap after the prime P qualifies for level N,
 *       printing the factorization of every composite in the gap.
 *   a359636 omega X ...
 *       Factor the given numbers (trial division + Pollard rho) and print omega.
 *   a359636 selftest [-t T] [-8] [-q]
 *       Reproduce a(1..7) (a few seconds), and with -8 also a(8) (~10 min).
 *
 * Numbers may be written as decimal, 2^k, 10^k, 1e12, or sums and differences
 * of those (10^12-1, 7e16+12e9), evaluated left to right.
 *
 * Build
 * -----
 *   make        (= cc -O2 -std=gnu11 -Wall -Wextra -pthread -I/opt/homebrew/include
 *                    a359636.c -L/opt/homebrew/lib -lprimesieve -lm -o a359636)
 *   If cc ever fails to link with a "tapi ... unknown architecture" error (an Xcode
 *   older than the Command Line Tools, as on the M1 Pro in early Oct 2026), build with
 *       DEVELOPER_DIR=/Library/Developer/CommandLineTools make
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
#ifdef __APPLE__
#include <pthread/qos.h>
#endif

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint8_t  u8;
typedef unsigned __int128 u128;           /* GCC/Clang extension, for exact products */

/* a(1..8) from the OEIS (index 0 unused); 'selftest' must reproduce them. */
#define NKNOWN 8
static const u64 KNOWN[NKNOWN + 1] = { 0, 7, 19, 643, 51427, 8083633, 1077940147,
                                       75582271489ULL, 34710483181813ULL };

/*
 * Sieve geometry (THEORY sections 3 and 7).
 *   SEG            k values per segment; the segment's SEG counter words (64 KB) stay in
 *                  L1/L2.  Chunks are rounded up to whole segments.  16384 was the fastest
 *                  choice on the M1 Pro (8192 and 32768 were 4-5% slower).  -DSEG=... to tune.
 *   PAT_A/B/C      periods of the three patterns = products of their primes.
 *   PATC           1: the primes 29, 31, 37 come from the PAT_C pattern; 0: they are
 *                  strided like the larger primes.  In Sep 2026 tests on the M1 Pro, PATC=0
 *                  was about 10% faster (2.10 vs 2.35 cycles per k); -DPATC=0 to use it.
 *   FIRST_STRIDED  smallest prime marked by strided progressions.
 *   RING           size of the chunk-completion ring; a worker may run at most RING-1
 *                  chunks ahead of the frontier.
 *   MAXSOL         solutions remembered for de-duplication and the checkpoint (further
 *                  ones are still printed and counted).
 */
#ifndef SEG
#define SEG      16384          /* k per sieve segment: 64 KB of u32 counters (fits L1) */
#endif
#define PAT_A    5005           /* 5*7*11*13 */
#define PAT_B    7429           /* 17*19*23  */
#define PAT_C    33263          /* 29*31*37  */
#ifndef PATC
#define PATC 1                  /* apply 29,31,37 as a pattern too (else strided) */
#endif
#define FIRST_STRIDED (PATC ? 41 : 29)
#define RING     65536          /* chunk completion ring; bounds how far threads run ahead */
#define MAXSOL   4096

/*
 * Process-wide state.
 *   g_quiet     -q: no status line and no start-up messages on stderr (stdout is unaffected).
 *   stderr_tty  stderr is a terminal: the status line is redrawn in place ("\r\033[K")
 *               every second; otherwise one line is printed per minute.
 *   g_stop      set by SIGINT/SIGTERM (on_sigint); workers and the main loop poll it and
 *               wind down at the next chunk boundary.
 *   out_mu      serializes the stdout blocks (triple / SOLUTION) of different threads and
 *               keeps them from interleaving with the status line.
 */
static bool g_quiet = false;
static bool stderr_tty;
static volatile sig_atomic_t g_stop = 0;
static pthread_mutex_t out_mu = PTHREAD_MUTEX_INITIALIZER;

/* ------------------------------------------------------------------ */
/* Generic helpers                                                     */
/* ------------------------------------------------------------------ */

/*
 * now: monotonic wall-clock time, for rates, ETAs and checkpoint intervals.
 *   In:     nothing.
 *   Out:    seconds since an arbitrary fixed point (CLOCK_MONOTONIC), as a double.
 *   Errors: none (CLOCK_MONOTONIC cannot fail on the supported systems).
 */
static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

/*
 * die: report a fatal error and end the process.
 *   In:     printf-style format and arguments.
 *   Out:    does not return; prints "a359636: <message>" on stderr and exits with
 *           status 1.  Used for every unrecoverable condition.
 *   Errors: none.  It does not erase a status line drawn with "\r", so on a terminal
 *           the message may start on the status line's row.  Called from a worker
 *           (out of memory) it ends the whole process at once.
 */
static void die(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fputs("a359636: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    exit(1);
}

/*
 * xmalloc: malloc that never returns NULL.
 *   In:     n = number of bytes (0 is treated as 1, so the result is always a valid
 *           pointer for free()).
 *   Out:    pointer to n uninitialized bytes; release with free().
 *   Errors: out of memory -> die().
 */
static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory (%zu bytes)", n);
    return p;
}

/*
 * parse_num: read a number from the command line.
 *   In:     s = text: decimal digits, optionally followed by ^E (a power: 2^40, 10^12)
 *           or eE (times 10^E: 1e12, 7e16), or a sum/difference of such terms
 *           (10^12-1, 7e16+12e9), evaluated left to right (the split is at the last
 *           '+' or '-', recursively).
 *   Out:    the value.  0^0 is taken as 1; 0^E and 1^E are computed without a loop.
 *   Errors: die() for an empty string, a leading sign or blank, a missing exponent
 *           ("1e"), any other trailing character, a value or partial sum that does
 *           not fit in 64 bits, and a negative difference.
 */
static u64 parse_num(const char *s)
{
    if (!*s) die("bad number ''");
    const char *op = NULL;
    for (const char *t = s + 1; *t; t++)
        if (*t == '+' || *t == '-') op = t;
    if (op) {
        char *a = strndup(s, (size_t)(op - s));
        if (!a) die("out of memory");
        u64 x = parse_num(a), y = parse_num(op + 1);
        free(a);
        if (*op == '+') {
            if (x > UINT64_MAX - y) die("number '%s' does not fit in 64 bits", s);
            return x + y;
        }
        if (y > x) die("number '%s' is negative", s);
        return x - y;
    }
    if (!isdigit((unsigned char)s[0])) die("bad number '%s'", s);
    char *end;
    errno = 0;
    unsigned long long base = strtoull(s, &end, 10);
    if (errno == ERANGE) die("number '%s' does not fit in 64 bits", s);
    if (*end == '\0') return base;
    if ((*end == '^' || *end == 'e' || *end == 'E') && isdigit((unsigned char)end[1])) {
        bool pw = (*end == '^');
        errno = 0;
        unsigned long long e = strtoull(end + 1, &end, 10);
        if (*end != '\0' || errno == ERANGE) die("bad number '%s'", s);
        u128 r = 1, b = pw ? base : 10;
        if (b < 2)
            r = (e == 0 || b == 1) ? 1 : 0;
        else
            for (unsigned long long i = 0; i < e; i++) {
                r *= b;
                if (r >> 64) die("number '%s' does not fit in 64 bits", s);
            }
        if (!pw) r *= base;
        if (r >> 64) die("number '%s' does not fit in 64 bits", s);
        return (u64)r;
    }
    die("bad number '%s'", s);
    return 0;
}

/*
 * fmt_commas: format v with thousands separators ("1,234,567").
 *   In:     out = caller's buffer of at least 27 bytes (20 digits, 6 commas, NUL;
 *           every caller uses char[32]); v = value.
 *   Out:    out holds the NUL-terminated text.
 *   Errors: none; a smaller buffer would overflow (not checked).
 */
static void fmt_commas(char *out, u64 v)
{
    char tmp[32];
    int n = snprintf(tmp, sizeof tmp, "%" PRIu64, v), o = 0;
    for (int i = 0; i < n; i++) {
        if (i && (n - i) % 3 == 0) out[o++] = ',';
        out[o++] = tmp[i];
    }
    out[o] = 0;
}

/*
 * fmt_time: format a duration for the status line: "3d04h05m", "1h02m03s", "4m05s", "12s".
 *   In:     out = caller's buffer (16 bytes suffice; callers use char[32]); s = seconds.
 *   Out:    out holds the text.  Negative or absurd (> 1e9 s) values give "?", which is
 *           how an unknown ETA is shown.
 *   Errors: none.  NaN is not filtered out (no caller produces one).
 */
static void fmt_time(char *out, double s)
{
    if (s < 0 || s > 1e9) { strcpy(out, "?"); return; }
    int t = (int)s;
    if (t >= 86400) sprintf(out, "%dd%02dh%02dm", t / 86400, t % 86400 / 3600, t % 3600 / 60);
    else if (t >= 3600) sprintf(out, "%dh%02dm%02ds", t / 3600, t % 3600 / 60, t % 60);
    else if (t >= 60) sprintf(out, "%dm%02ds", t / 60, t % 60);
    else sprintf(out, "%ds", t);
}

/*
 * on_sigint: handler for SIGINT and SIGTERM.
 *   In:     sig (unused).
 *   Out:    sets g_stop.  A scan then finishes the chunks in hand, writes its checkpoint
 *           and prints its summary; a hunt stops taking tasks.  Async-signal-safe (a
 *           single store to a volatile sig_atomic_t).
 *   Errors: none.
 */
static void on_sigint(int sig) { (void)sig; g_stop = 1; }

/* ------------------------------------------------------------------ */
/* Arithmetic: mulmod, Miller-Rabin, roots, factoring                  */
/* ------------------------------------------------------------------ */

/*
 * mulmod: a*b mod m without overflow, via the exact 128-bit product.
 *   In:     a, b (need not be reduced), m >= 1.
 *   Out:    (a*b) mod m.
 *   Errors: m = 0 divides by zero (never called that way).
 */
static inline u64 mulmod(u64 a, u64 b, u64 m) { return (u64)((u128)a * b % m); }

/*
 * powmod: a^e mod m by binary exponentiation.
 *   In:     a, e, m >= 2.
 *   Out:    a^e mod m (1 for e = 0).
 *   Errors: none for m >= 2.  (m = 1 with e = 0 would return 1 instead of 0; m = 0 would
 *           divide by zero.  Callers pass primes or odd n >= 41*41.)
 */
static u64 powmod(u64 a, u64 e, u64 m)
{
    u64 r = 1;
    a %= m;
    while (e) {
        if (e & 1) r = mulmod(r, a, m);
        a = mulmod(a, a, m);
        e >>= 1;
    }
    return r;
}

/*
 * is_prime: deterministic primality test for any 64-bit n.
 *   In:     n.
 *   Out:    true iff n is prime.  Trial division by the 12 primes 2..37 (any n < 41^2
 *           that survives it is prime), then strong Miller-Rabin to the same 12 bases,
 *           which is deterministic for n < 3.18e23 and therefore for every 64-bit n.
 *   Errors: none.
 */
static bool is_prime(u64 n)
{
    static const u64 bases[] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 };
    if (n < 2) return false;
    for (int i = 0; i < 12; i++) {
        if (n % bases[i] == 0) return n == bases[i];
    }
    if (n < 41 * 41) return true;
    u64 d = n - 1;
    int s = 0;
    while (!(d & 1)) { d >>= 1; s++; }
    for (int i = 0; i < 12; i++) {
        u64 x = powmod(bases[i], d, n);
        if (x == 1 || x == n - 1) continue;
        bool comp = true;
        for (int r = 1; r < s && comp; r++) {
            x = mulmod(x, x, n);
            if (x == n - 1) comp = false;
        }
        if (comp) return false;
    }
    return true;
}

/*
 * iroot: integer r-th root floor(n^(1/r)).
 *   In:     n, r >= 1.
 *   Out:    the largest x with x^r <= n (n itself for r = 1 or n < 2).  A floating-point
 *           estimate is corrected in +-1 steps using exact 128-bit powers (which cannot
 *           overflow: the loops stop as soon as a partial power exceeds n).
 *   Errors: r <= 0 is invalid (1.0/0 and an out-of-range float-to-integer conversion);
 *           no caller passes it.
 */
static u64 iroot(u64 n, int r)
{
    if (r == 1) return n;
    if (n < 2) return n;
    u64 x = (u64)pow((double)n, 1.0 / r);
    if (x == 0) x = 1;
    for (;;) {                         /* adjust so that x^r <= n < (x+1)^r */
        u128 p = 1;
        bool over = false;
        for (int i = 0; i < r; i++) { p *= x; if (p > n) { over = true; break; } }
        if (over) { x--; continue; }
        u128 q = 1;
        over = false;
        for (int i = 0; i < r; i++) { q *= (x + 1); if (q > n) { over = true; break; } }
        if (!over) { x++; continue; }
        return x;
    }
}

/*
 * is_perfect_power: is n = a^j for some integers a >= 2, j >= 2?
 *   In:     n.
 *   Out:    true or false (false for n < 4).  Tries j = 2..63 and stops as soon as the
 *           j-th root falls below 2.
 *   Errors: none.
 */
static bool is_perfect_power(u64 n)
{
    for (int j = 2; j <= 63; j++) {
        u64 x = iroot(n, j);
        if (x < 2) return false;
        u128 p = 1;
        for (int i = 0; i < j; i++) p *= x;
        if (p == n) return true;
    }
    return false;
}

/*
 * The prime table: all primes <= g_plimit in ascending order, used by trial division
 * (omega_at_least, factor_full), by the sieve set-up and by compute_c().  Built by
 * build_primes(), rebuilt larger when a scan needs it, read-only while threads run.
 */
static u32 *g_primes;          /* all primes up to g_plimit */
static size_t g_nprimes;
static u64 g_plimit;

/*
 * build_primes: (re)build the prime table with primesieve.
 *   In:     limit = largest number to include (below 2^32: the table holds u32).
 *   Out:    g_primes, g_nprimes, g_plimit describe all primes <= limit; the previous
 *           table, if any, is released with primesieve_free().
 *   Errors: primesieve failure (e.g. out of memory) -> die().  Must not run while worker
 *           threads read the table.
 */
static void build_primes(u64 limit)
{
    size_t n;
    u32 *pr = (u32 *)primesieve_generate_primes(2, limit, &n, UINT32_PRIMES);
    if (!pr) die("primesieve failed");
    if (g_primes) primesieve_free(g_primes);
    g_primes = pr;
    g_nprimes = n;
    g_plimit = limit;
}

/*
 * omega_at_least: does N have at least n distinct prime factors?  (THEORY section 6)
 *   In:     N, n.  The prime table must reach cbrt(N) for an exact answer: every caller
 *           builds it to at least cbrt(Bmax) + 1000 (2^21 for the selftest spot checks).
 *   Out:    true iff omega(N) >= n (n <= 0: always true; N < 2: false).
 *   Method: trial division in increasing order with a size bound.  When r more distinct
 *           primes are needed and every remaining factor is >= p, the cofactor must be
 *           >= p^r, so the search fails as soon as p exceeds floor(cofactor^(1/r)).
 *           With r = 1 any cofactor > 1 succeeds.  With r = 2 and p^3 > cofactor, the
 *           cofactor is a prime, a prime square or a product of two distinct primes,
 *           told apart by is_prime() and is_perfect_power().  Sieve survivors are
 *           usually rejected after a few hundred divisions.
 *   Errors: none.  Only if the table were smaller than cbrt(N) could the fallback after
 *           the loop undercount (it counts a composite cofactor as 1 or 2 primes, so a
 *           product of three large primes would be missed).  p*p*p wraps for
 *           p > 2642245; that merely skips the r = 2 shortcut, and the trial division
 *           that follows is still exact.
 */
static bool omega_at_least(u64 N, int n)
{
    if (n <= 0) return true;
    if (N < 2) return false;
    int cnt = 0;
    u64 lim = iroot(N, n);            /* smallest prime factor must be <= lim */
    for (size_t i = 0; i < g_nprimes; i++) {
        u64 p = g_primes[i];
        int r = n - cnt;               /* distinct primes still needed */
        if (r == 1) return N > 1;      /* any cofactor > 1 supplies one more prime */
        if (p > lim) return false;     /* cofactor too small for r primes above p */
        if (r == 2 && p * p * p > N) { /* cofactor is q, q^2 or q1*q2 territory */
            /* N > 1 here; N has all prime factors >= p and N < p^3, so N is a
               prime, a prime square or a product of two distinct primes. */
            if (is_prime(N)) return false;
            return !is_perfect_power(N);
        }
        if (N % p == 0) {
            cnt++;
            N /= p;
            while (N % p == 0) N /= p;
            r = n - cnt;
            if (r <= 0) return true;
            if (r == 1) return N > 1;
            lim = iroot(N, r);
        }
    }
    /* Prime table exhausted (only possible for huge N): fall back. */
    if (N == 1) return cnt >= n;
    if (is_prime(N)) return cnt + 1 >= n;
    /* composite cofactor with all prime factors > g_plimit: at least 1 prime,
       at least 2 unless a perfect power */
    return cnt + (is_perfect_power(N) ? 1 : 2) >= n;
}

/*
 * rho: a nontrivial factor of a composite n (Pollard rho with Floyd cycle detection).
 *   In:     n composite (callers test is_prime() first).
 *   Out:    a divisor d with 1 < d < n (2 for even n).  Iterates x -> x^2 + c and takes
 *           gcd(|x - y|, n); if the gcd is n itself it retries with the next c.
 *   Errors: never returns for prime n (not passed).  For n within c of 2^64 the "+ c"
 *           could wrap and spoil the iteration; such n do not occur here.
 */
static u64 rho(u64 n)
{
    if (!(n & 1)) return 2;
    for (u64 c = 1;; c++) {
        u64 x = 2, y = 2, d = 1;
        while (d == 1) {
            x = (mulmod(x, x, n) + c) % n;
            y = (mulmod(y, y, n) + c) % n;
            y = (mulmod(y, y, n) + c) % n;
            u64 diff = x > y ? x - y : y - x;
            /* gcd */
            u64 a = diff, b = n;
            while (b) { u64 t = a % b; a = b; b = t; }
            d = a;
        }
        if (d != n) return d;
    }
}

/*
 * factor_full: complete factorization, used for printing and by omega_exact().
 *   In:     n; pr, ex = caller arrays with room for at least 15 entries (callers use 64).
 *   Out:    returns k = omega(n); pr[0..k-1] = the distinct primes in ascending order,
 *           ex[0..k-1] = their exponents.  n = 0 or 1 gives k = 0.  Trial division with
 *           the table up to sqrt(n), then Pollard rho on what is left (a stack of 64
 *           pending factors is enough for any 64-bit n).
 *   Errors: none for 64-bit n (at most 15 distinct primes, at most 63 prime factors).
 */
static int factor_full(u64 n, u64 *pr, int *ex)   /* returns number of distinct primes */
{
    int k = 0;
    for (size_t i = 0; i < g_nprimes && (u64)g_primes[i] * g_primes[i] <= n; i++) {
        u64 p = g_primes[i];
        if (n % p == 0) {
            pr[k] = p; ex[k] = 0;
            while (n % p == 0) { n /= p; ex[k]++; }
            k++;
        }
    }
    /* remaining n has no prime factor <= min(g_plimit, sqrt) */
    u64 stack[64];
    int sp = 0;
    if (n > 1) stack[sp++] = n;
    while (sp) {
        u64 x = stack[--sp];
        if (x == 1) continue;
        if (is_prime(x)) {
            int j;
            for (j = 0; j < k; j++) if (pr[j] == x) { ex[j]++; break; }
            if (j == k) { pr[k] = x; ex[k] = 1; k++; }
            continue;
        }
        u64 d = rho(x);
        stack[sp++] = d;
        stack[sp++] = x / d;
    }
    /* sort by prime */
    for (int i = 1; i < k; i++)
        for (int j = i; j > 0 && pr[j - 1] > pr[j]; j--) {
            u64 tp = pr[j]; pr[j] = pr[j - 1]; pr[j - 1] = tp;
            int te = ex[j]; ex[j] = ex[j - 1]; ex[j - 1] = te;
        }
    return k;
}

/*
 * fmt_factorization: text like "2^3*13*41*43" for n, as printed in SOLUTION blocks.
 *   In:     out = caller's buffer (about 100 bytes are needed at most for 64-bit n;
 *           callers use char[256]); n.
 *   Out:    out holds the text; n = 0 or 1 is printed as itself.
 *   Errors: none (sprintf; the buffer size is the caller's responsibility).
 */
static void fmt_factorization(char *out, u64 n)
{
    u64 pr[64]; int ex[64];
    int k = factor_full(n, pr, ex);
    int o = 0;
    for (int i = 0; i < k; i++) {
        o += sprintf(out + o, "%s%" PRIu64, i ? "*" : "", pr[i]);
        if (ex[i] > 1) o += sprintf(out + o, "^%d", ex[i]);
    }
    if (k == 0) sprintf(out + o, "%" PRIu64, n);
}

/*
 * omega_exact: omega(n) by complete factorization (exact for any n, slower than
 *   omega_at_least; used only for printing and in the selftest).
 *   In: n.   Out: number of distinct prime factors (0 for n < 2).   Errors: none.
 */
static int omega_exact(u64 n)
{
    u64 pr[64]; int ex[64];
    return factor_full(n, pr, ex);
}

/*
 * prev_prime: largest prime below x.
 *   In:     x > 3.
 *   Out:    the prime (steps down over odd numbers with is_prime()).
 *   Errors: x <= 3 would step below 2 and wrap around; callers pass m - 1 >= 8.
 */
static u64 prev_prime(u64 x)           /* largest prime < x, x > 3 */
{
    x--;
    if (!(x & 1)) x--;
    while (!is_prime(x)) x -= 2;
    return x;
}

/*
 * next_prime: smallest prime above x.
 *   In:     2 <= x < 2^64 - 59 (the largest 64-bit prime).
 *   Out:    the prime.
 *   Errors: x < 2 returns 3 (2 is skipped); x >= 2^64 - 59 would wrap.  Neither occurs:
 *           run_scan() keeps END at least 65536 below 2^64.
 */
static u64 next_prime(u64 x)           /* smallest prime > x */
{
    x++;
    if (!(x & 1)) x++;
    while (!is_prime(x)) x += 2;
    return x;
}

/* ------------------------------------------------------------------ */
/* Gap check                                                           */
/* ------------------------------------------------------------------ */

/*
 * gap_qualifies: does the prime gap that starts at p qualify at level n?
 *   In:     p = a prime (not checked; callers pass prev_prime() results); n = level;
 *           q_out = where to store the next prime.
 *   Out:    *q_out = nextprime(p); true iff q - p >= 4 and omega(x) >= n for every
 *           x in (p, q).  Uses omega_at_least(), so the table must cover cbrt(q).
 *   Errors: none.
 */
static bool gap_qualifies(u64 p, int n, u64 *q_out)
{
    u64 q = next_prime(p);
    *q_out = q;
    if (q - p < 4) return false;
    for (u64 x = p + 1; x < q; x++)
        if (!omega_at_least(x, n)) return false;
    return true;
}

/* ------------------------------------------------------------------ */
/* Scan                                                                */
/* ------------------------------------------------------------------ */

/* A qualifying gap: consecutive primes p < q and the m (= 3 mod 6) of the triple that
   led to it. */
typedef struct {
    u64 p, q, m;
} solution_t;

/*
 * scan_t: the state of one 'scan' (one run_scan() call), heap-allocated because its
 * arrays take about 210 KB.  Locking:
 *   fr_mu   protects frontier and done[], is held whenever stop_now is written, and is
 *           the mutex of fr_cv (broadcast whenever the frontier may have moved);
 *   sol_mu  protects sol[], nsolrec, best_p and best_m;
 *   the statistics and next_chunk are atomics; everything else is set before the
 *   workers start and read-only afterwards.
 */
typedef struct {
    int      n;              /* level */
    u64      k_lo, k_hi;     /* k range [k_lo, k_hi), m = 6k+3 */
    u64      chunk;          /* k per chunk */
    u64      nchunks;        /* chunks in [k_lo, k_hi); the last one may be short */
    u32      T;              /* sieve prime bound */
    int      c;              /* min # prime factors <= T in each target */
    int      thr;            /* sieved-count threshold c-1 (may be <= 0) */
    /* strided sieve primes FIRST_STRIDED..T */
    u32     *sp;             /* primes */
    u32     *sr;             /* residues, 3 per prime: k = sr[3i+t] (mod p) marks target t */
    size_t   nsp;            /* number of strided primes */
    u32      patA[PAT_A], patB[PAT_B];   /* packed counts of 5..13 / 17..23, by k mod period */
    u32     *patC;                       /* same for 29, 31, 37 (all zero unless PATC) */
    /* statistics */
    atomic_ullong survivors, triples, nsol;
    /* solutions */
    solution_t sol[MAXSOL];
    int      nsolrec;        /* entries used in sol[] */
    u64      best_p, best_m; /* smallest solution so far and its m (0 = none yet) */
    pthread_mutex_t sol_mu;
    /* chunk frontier */
    atomic_ullong next_chunk;   /* next chunk index to hand out */
    u64      frontier;          /* all chunks < frontier are complete */
    u8       done[RING];        /* done[i % RING]: chunk i complete, frontier not yet past it */
    pthread_mutex_t fr_mu;
    pthread_cond_t  fr_cv;
    bool     scan_all;          /* -a: keep going after a certain solution */
    bool     stop_now;          /* set by chunk_done() for the early stop */
    double   t0;                /* start time (now()) */
    const char *state_file;     /* -S FILE, or NULL */
    u64      resume_frontier;   /* frontier read from the checkpoint (for the rate) */
} scan_t;

/*
 * inv_mod: inverse of a modulo a prime p, by Fermat (a^(p-2) mod p).
 *   In:     p prime, p not dividing a.
 *   Out:    a^-1 mod p.
 *   Errors: meaningless for composite p, 0 when p divides a.  Only used with a = 6 and
 *           p >= 5.
 */
static u64 inv_mod(u64 a, u64 p)       /* a^-1 mod p, p prime */
{
    return powmod(a % p, p - 2, p);
}

/*
 * target_residue: the residue class of k in which p divides a target.
 *   In:     p prime >= 5; t = -1, 0, +1 selects the target m-1, m, m+1 (m = 6k+3).
 *   Out:    r in [0, p) with  p | 6k + 3 + t  <=>  k = r (mod p);
 *           r = -(3 + t) * 6^-1 mod p.
 *   Errors: p = 2 or 3 (6 not invertible) gives nonsense; never passed.
 */
static u32 target_residue(u64 p, int t)
{
    u64 inv6 = inv_mod(6, p);
    u64 c = (u64)(3 + t) % p;               /* 6k = -c  (mod p) */
    return (u32)(((p - c) % p) * inv6 % p);
}

/*
 * setup_sieve: build the periodic patterns and the strided prime list for S->T.
 *   In:     S with T set; the prime table must hold every prime <= T.
 *   Out:    S->patA (primes 5..13, period 5005), S->patB (17..23, period 7429) and
 *           S->patC (29..37, period 33263; allocated here, all zero if PATC is 0):
 *           entry j holds the packed counts (bytes 0..2 = m-1, m, m+1) of the pattern
 *           primes dividing the targets of every k = j (mod period).  Pattern primes
 *           above T are left out, so a small -T is honored.
 *           S->sp[0..nsp-1] = the primes FIRST_STRIDED..T and S->sr[3i+t] their target
 *           residues.
 *   Errors: out of memory -> die().
 */
static void setup_sieve(scan_t *S)
{
    /* patterns for 5,7,11,13 and 17,19,23 */
    memset(S->patA, 0, sizeof S->patA);
    memset(S->patB, 0, sizeof S->patB);
    static const u32 pa[4] = { 5, 7, 11, 13 }, pb[3] = { 17, 19, 23 };
    for (int i = 0; i < 4; i++) {
        if (pa[i] > S->T) continue;
        for (int t = -1; t <= 1; t++) {
            u32 r = target_residue(pa[i], t), inc = 1u << (8 * (t + 1));
            for (u32 k = r; k < PAT_A; k += pa[i]) S->patA[k] += inc;
        }
    }
    for (int i = 0; i < 3; i++) {
        if (pb[i] > S->T) continue;
        for (int t = -1; t <= 1; t++) {
            u32 r = target_residue(pb[i], t), inc = 1u << (8 * (t + 1));
            for (u32 k = r; k < PAT_B; k += pb[i]) S->patB[k] += inc;
        }
    }
    S->patC = xmalloc(PAT_C * sizeof(u32));
    memset(S->patC, 0, PAT_C * sizeof(u32));
    if (PATC) {
        static const u32 pc[3] = { 29, 31, 37 };
        for (int i = 0; i < 3; i++) {
            if (pc[i] > S->T) continue;
            for (int t = -1; t <= 1; t++) {
                u32 r = target_residue(pc[i], t), inc = 1u << (8 * (t + 1));
                for (u32 k = r; k < PAT_C; k += pc[i]) S->patC[k] += inc;
            }
        }
    }
    /* strided primes FIRST_STRIDED..T */
    size_t cnt = 0;
    for (size_t i = 0; i < g_nprimes && g_primes[i] <= S->T; i++)
        if (g_primes[i] >= FIRST_STRIDED) cnt++;
    S->sp = xmalloc(cnt * sizeof(u32) + 1);
    S->sr = xmalloc(3 * cnt * sizeof(u32) + 1);
    S->nsp = 0;
    for (size_t i = 0; i < g_nprimes && g_primes[i] <= S->T; i++) {
        u32 p = g_primes[i];
        if (p < FIRST_STRIDED) continue;
        S->sp[S->nsp] = p;
        for (int t = -1; t <= 1; t++) S->sr[3 * S->nsp + (t + 1)] = target_residue(p, t);
        S->nsp++;
    }
}

/*
 * compute_c: the number of distinct prime factors <= T that every candidate must have.
 *   In:     n = level; T = sieve bound; Bmax = largest number that will be examined.
 *   Out:    c = n - rmax, where rmax is the largest r with
 *             prod(r) = (product of the first n-r primes) * (first r primes above T)
 *                     <= Bmax
 *           (THEORY section 5): every N <= Bmax with omega(N) >= n has at least c
 *           distinct prime factors <= T.  When even the product of the first n primes
 *           exceeds Bmax the result is n (nothing can qualify then anyway).
 *   Errors: die() if the prime table has too few primes above T, i.e. for a -T beyond
 *           the table (about max(2^20, cbrt(Bmax))).
 */
static int compute_c(int n, u32 T, u64 Bmax)
{
    int rmax = 0;
    for (int r = 0; r <= n; r++) {
        u128 prod = 1;
        bool over = false;
        for (int i = 0; i < n - r; i++) { prod *= g_primes[i]; if (prod > Bmax) { over = true; break; } }
        size_t j = 0;
        while (j < g_nprimes && g_primes[j] <= T) j++;
        for (int i = 0; i < r && !over; i++) {
            if (j + i >= g_nprimes) die("prime table too small in compute_c");
            prod *= g_primes[j + i];
            if (prod > Bmax) over = true;
        }
        if (over) break;
        rmax = r;
    }
    return n - rmax;
}

/*
 * record_solution: remember and print a qualifying gap.
 *   In:     S; p < q consecutive primes whose gap qualifies; m = the triple's m.
 *   Out:    unless p is already known (a gap of 8 or more can hold several triples):
 *           stores it in S->sol (up to MAXSOL), counts it, makes it best_p/best_m when
 *           p is the smallest so far, and prints the SOLUTION block with the
 *           factorization of every composite in the gap.
 *   Errors: none.  Past MAXSOL solutions are still printed and counted but no longer
 *           de-duplicated or checkpointed.  Lock order: sol_mu, then out_mu.
 */
static void record_solution(scan_t *S, u64 p, u64 q, u64 m)
{
    pthread_mutex_lock(&S->sol_mu);
    bool dup = false;
    for (int i = 0; i < S->nsolrec; i++) if (S->sol[i].p == p) dup = true;
    if (!dup) {
        if (S->nsolrec < MAXSOL) {
            S->sol[S->nsolrec].p = p;
            S->sol[S->nsolrec].q = q;
            S->sol[S->nsolrec].m = m;
            S->nsolrec++;
        }
        atomic_fetch_add(&S->nsol, 1);
        if (!S->best_p || p < S->best_p) { S->best_p = p; S->best_m = m; }
        pthread_mutex_lock(&out_mu);
        if (stderr_tty && !g_quiet) fputs("\r\033[K", stderr);
        printf("SOLUTION n=%d p=%" PRIu64 " q=%" PRIu64 " gap=%" PRIu64 " (m=%" PRIu64 ")\n",
               S->n, p, q, q - p, m);
        char buf[256];
        for (u64 x = p + 1; x < q; x++) {
            fmt_factorization(buf, x);
            printf("    %" PRIu64 " = %s  omega=%d\n", x, buf, omega_exact(x));
        }
        fflush(stdout);
        pthread_mutex_unlock(&out_mu);
    }
    pthread_mutex_unlock(&S->sol_mu);
}

/*
 * check_survivor: exact test of one sieve survivor (THEORY section 6).
 *   In:     S; k = index of m = 6k + 3.
 *   Out:    if omega(m-1), omega(m), omega(m+1) are all >= n: counts the triple, prints
 *           the triple line (exact omegas and the offset of prevprime(m-1)), and records
 *           the gap that starts at p = prevprime(m-1) if it qualifies.
 *   Errors: none.  Runs concurrently in all workers; shares only the atomic counters,
 *           out_mu and record_solution()'s locks.
 */
static void check_survivor(scan_t *S, u64 k)
{
    u64 m = 6 * k + 3;
    int n = S->n;
    if (!omega_at_least(m - 1, n)) return;
    if (!omega_at_least(m, n)) return;
    if (!omega_at_least(m + 1, n)) return;
    atomic_fetch_add(&S->triples, 1);
    u64 p = prev_prime(m - 1), q;
    pthread_mutex_lock(&out_mu);
    if (stderr_tty && !g_quiet) fputs("\r\033[K", stderr);
    printf("triple   n=%d m=%" PRIu64 "  omega(m-1,m,m+1) = %d %d %d  prevprime=%" PRIu64 " (%+" PRId64 ")\n",
           n, m, omega_exact(m - 1), omega_exact(m), omega_exact(m + 1), p, (int64_t)(p - m));
    fflush(stdout);
    pthread_mutex_unlock(&out_mu);
    if (gap_qualifies(p, n, &q)) record_solution(S, p, q, m);
}

static void save_state(scan_t *S);       /* checkpoint writer, defined further down */

/*
 * progress: the status line on stderr.
 *   In:     S; final = true for the closing call (always printed, ends with a newline).
 *   Out:    position (largest m below the frontier), percentage, rate in Gm/s (10^9 m
 *           values per second = 6 * 10^9 k per second), survivor/triple/solution counts,
 *           best solution, elapsed time and ETA.  At most once a second on a terminal
 *           (redrawn in place), once a minute otherwise.
 *   Errors: none.  Called only by worker 0 and, after the workers are joined, by the
 *           main thread, so the static timestamp needs no lock; reads S->frontier
 *           without fr_mu (benign).  After resuming a scan that had already finished,
 *           the rate is garbage (the subtraction underflows); cosmetic only.
 */
static void progress(scan_t *S, bool final)
{
    static double last = 0;
    double t = now();
    if (!final && t - last < (stderr_tty ? 1.0 : 60.0)) return;
    last = t;
    u64 fr = S->frontier;
    u64 kdone = fr * S->chunk;
    if (kdone > S->k_hi - S->k_lo) kdone = S->k_hi - S->k_lo;
    u64 mnow = 6 * (S->k_lo + kdone) + 3;
    double frac = (double)kdone / (double)(S->k_hi - S->k_lo);
    double el = t - S->t0;
    double rate = el > 0 ? (double)(kdone - (S->resume_frontier * S->chunk)) / el : 0;
    char c1[32], c2[32], eta[32], elap[32];
    fmt_commas(c1, mnow);
    fmt_time(eta, rate > 0 ? (double)((S->k_hi - S->k_lo) - kdone) / rate : -1);
    fmt_time(elap, el);
    snprintf(c2, sizeof c2, "%.2f", rate * 6 / 1e9);
    char best[48] = "";
    if (S->best_p) snprintf(best, sizeof best, " best=%" PRIu64, S->best_p);
    fprintf(stderr, "%sm=%s (%.2f%%)  %s Gm/s  surv=%llu trip=%llu sol=%llu%s  %s elapsed, ETA %s%s",
            stderr_tty ? "\r\033[K" : "", c1, frac * 100, c2,
            (unsigned long long)atomic_load(&S->survivors),
            (unsigned long long)atomic_load(&S->triples),
            (unsigned long long)atomic_load(&S->nsol),
            best, elap, eta,
            stderr_tty && !final ? "" : "\n");
    fflush(stderr);
}

/*
 * chunk_done: mark chunk idx complete and advance the frontier.
 *   In:     S; idx = index of the chunk just finished.
 *   Out:    sets done[idx % RING], moves S->frontier over every completed chunk (clearing
 *           their flags), sets S->stop_now once the frontier has passed the chunk that
 *           holds best_m (unless -a), and wakes everything waiting on fr_cv (workers
 *           held back by the ring, the main loop).
 *   Errors: none.  Relies on workers staying less than RING chunks ahead.  Reads
 *           best_p/best_m without sol_mu (benign; see THEORY section 10).
 */
static void chunk_done(scan_t *S, u64 idx)
{
    pthread_mutex_lock(&S->fr_mu);
    S->done[idx % RING] = 1;
    while (S->frontier < S->nchunks && S->done[S->frontier % RING]) {
        S->done[S->frontier % RING] = 0;
        S->frontier++;
    }
    /* early stop: frontier past the chunk holding the best m */
    if (S->best_p && !S->scan_all) {
        u64 kbest = (S->best_m - 3) / 6;
        u64 cbest = (kbest - S->k_lo) / S->chunk;
        if (S->frontier > cbest) S->stop_now = true;
    }
    pthread_cond_broadcast(&S->fr_cv);
    pthread_mutex_unlock(&S->fr_mu);
}

/* Argument of worker(): the scan and the thread number (thread 0 draws the status line). */
typedef struct { scan_t *S; int id; } worker_arg;

/*
 * worker: one sieve thread.  Takes chunks from S->next_chunk until none are left (or
 * g_stop / S->stop_now), sieves each chunk segment by segment and verifies its
 * survivors.
 *   In:     arg = worker_arg *.
 *   Out:    NULL.  Effects: triples and solutions via check_survivor(), completion via
 *           chunk_done(), the status line (thread 0).
 *   Errors: out of memory -> die() (ends the process).  A chunk that has been started
 *           is always finished and reported, so the frontier stays exact.
 * Per chunk:
 *   1. for every strided prime and target, the distance from k0 to the first k >= k0 in
 *      its residue class (nextj);
 *   2. per segment of at most SEG k: counters := patterns (rotated to the segment's phase),
 *      add the strided progressions, then the threshold scan (THEORY section 4).
 */

static void *worker(void *arg)
{
    scan_t *S = ((worker_arg *)arg)->S;
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);   /* prefer the performance cores */
#endif
    u32 *W = xmalloc(SEG * sizeof(u32) + 64);
    u32 *nextj = xmalloc(3 * S->nsp * sizeof(u32) + 4);
    /* threshold test of THEORY section 4 (thradd is unused when thr <= 0);
       INC1..INC3 add one to the counter byte of m-1, m, m+1 */
    const u32 thrmask = 0x808080u;
    const u32 thradd = (S->thr > 0) ? 0x010101u * (u32)(128 - S->thr) : 0x808080u;
    enum { INC1 = 0x000001u, INC2 = 0x000100u, INC3 = 0x010000u };

    for (;;) {
        u64 idx = atomic_fetch_add(&S->next_chunk, 1);
        if (idx >= S->nchunks || g_stop || S->stop_now) break;
        /* don't run more than RING-1 chunks ahead of the frontier */
        pthread_mutex_lock(&S->fr_mu);
        while (idx >= S->frontier + RING - 1 && !g_stop && !S->stop_now)
            pthread_cond_wait(&S->fr_cv, &S->fr_mu);
        pthread_mutex_unlock(&S->fr_mu);

        /* this chunk: k in [k0, k1) */
        u64 k0 = S->k_lo + idx * S->chunk;
        u64 k1 = k0 + S->chunk;
        if (k1 > S->k_hi) k1 = S->k_hi;

        /* first offsets for the strided primes: nextj[3i+t] = distance from k0 to the
           first k >= k0 with k = sr[3i+t] (mod p) */
        for (size_t i = 0; i < S->nsp; i++) {
            u32 p = S->sp[i];
            u32 k0m = (u32)(k0 % p);
            for (int t = 0; t < 3; t++) {
                u32 r = S->sr[3 * i + t];
                nextj[3 * i + t] = r >= k0m ? r - k0m : r + p - k0m;
            }
        }
        /* phase of the current k in each pattern period, carried from segment to segment */
        u32 a = (u32)(k0 % PAT_A), b = (u32)(k0 % PAT_B), c = (u32)(k0 % PAT_C);

        for (u64 ks = k0; ks < k1; ks += SEG) {
            u32 len = (u32)(k1 - ks < SEG ? k1 - ks : SEG);
            /* pattern init: W = patA + patB (rotated), one fused pass */
            for (u32 j = 0; j < len;) {
                u32 run = PAT_A - a;
                if (run > PAT_B - b) run = PAT_B - b;
                if (run > len - j) run = len - j;
                const u32 *pa = S->patA + a, *pb = S->patB + b;
                u32 *w = W + j;
                for (u32 i = 0; i < run; i++) w[i] = pa[i] + pb[i];
                j += run; a += run; b += run;
                if (a == PAT_A) a = 0;
                if (b == PAT_B) b = 0;
            }
            if (PATC) {
                for (u32 j = 0; j < len;) {
                    u32 run = PAT_C - c; if (run > len - j) run = len - j;
                    const u32 *pc = S->patC + c;
                    u32 *w = W + j;
                    for (u32 i = 0; i < run; i++) w[i] += pc[i];
                    j += run; c += run; if (c == PAT_C) c = 0;
                }
            }
            /* strided marks: the three targets of a prime share the stride p, so after
               aligning their phases they are marked in one loop.  j1, j2, j3 are the next
               positions (< p) of the three progressions.  If j2 or j3 lies before j1 it is
               marked once and moved up by p, so that j1 <= j2, j3 < j1 + p; then the
               progressions differ by the fixed offsets d2, d3, and one loop (unrolled four
               times) marks all three while the farthest stays below len.  The plain loops
               finish each progression separately.  Segments shorter than 2p (the last one
               of a chunk, or very large p) use the plain loops only. */
            for (size_t i = 0; i < S->nsp; i++) {
                const u32 p = S->sp[i];
                u32 j1 = nextj[3 * i], j2 = nextj[3 * i + 1], j3 = nextj[3 * i + 2];
                if (len >= 2 * p) {
                    if (j2 < j1) { W[j2] += INC2; j2 += p; }
                    if (j3 < j1) { W[j3] += INC3; j3 += p; }
                    const u32 d2 = j2 - j1, d3 = j3 - j1;
                    const u32 end = len - (d2 > d3 ? d2 : d3);
                    u32 j = j1;
                    for (; j + 3 * p < end; j += 4 * p) {
                        W[j] += INC1;             W[j + d2] += INC2;             W[j + d3] += INC3;
                        W[j + p] += INC1;         W[j + p + d2] += INC2;         W[j + p + d3] += INC3;
                        W[j + 2 * p] += INC1;     W[j + 2 * p + d2] += INC2;     W[j + 2 * p + d3] += INC3;
                        W[j + 3 * p] += INC1;     W[j + 3 * p + d2] += INC2;     W[j + 3 * p + d3] += INC3;
                    }
                    for (; j < end; j += p) { W[j] += INC1; W[j + d2] += INC2; W[j + d3] += INC3; }
                    j1 = j; j2 = j + d2; j3 = j + d3;
                }
                for (; j1 < len; j1 += p) W[j1] += INC1;
                for (; j2 < len; j2 += p) W[j2] += INC2;
                for (; j3 < len; j3 += p) W[j3] += INC3;
                /* carry the offsets into the next segment */
                nextj[3 * i] = j1 - len; nextj[3 * i + 1] = j2 - len; nextj[3 * i + 2] = j3 - len;
            }
            /* scan: a k survives when all three counters are >= thr, i.e. when
               (~(W + thradd)) & 0x808080 == 0; blocks are tested with a min-reduction */
            if (S->thr <= 0) {
                for (u32 j = 0; j < len; j++) { atomic_fetch_add(&S->survivors, 1); check_survivor(S, ks + j); }
            } else {
                u32 j = 0;
                for (; j + 16 <= len; j += 16) {
                    u32 mn = 0xffffffffu;
                    for (u32 i = 0; i < 16; i++) {
                        u32 t = (~(W[j + i] + thradd)) & thrmask;
                        mn = t < mn ? t : mn;
                    }
                    if (mn == 0) {
                        for (u32 i = 0; i < 16; i++)
                            if ((((~(W[j + i] + thradd)) & thrmask)) == 0) {
                                atomic_fetch_add(&S->survivors, 1);
                                check_survivor(S, ks + j + i);
                            }
                    }
                }
                for (; j < len; j++) {
                    if ((((~(W[j] + thradd)) & thrmask)) == 0) {
                        atomic_fetch_add(&S->survivors, 1);
                        check_survivor(S, ks + j);
                    }
                }
            }
        }
        /* chunk complete: advance the frontier; thread 0 also refreshes the status line */
        chunk_done(S, idx);
        if (((worker_arg *)arg)->id == 0 && !g_quiet) progress(S, false);
    }
    free(W);
    free(nextj);
    return NULL;
}

/* ---- checkpoint ---- */

/*
 * save_state: write the checkpoint, if -S was given.
 *   In:     S.
 *   Out:    FILE is replaced atomically (write FILE.tmp, then rename) by
 *               A359636 state v1
 *               n <level>
 *               k_lo <first k>
 *               k_hi <k end, exclusive>
 *               chunk <k per chunk>
 *               frontier <number of complete chunks>
 *               survivors <count>
 *               triples <count>
 *               solution <p> <q> <m>          (one line per recorded solution)
 *   Errors: FILE.tmp cannot be created, a write fails (e.g. disk full) or the rename
 *           fails: message on stderr, the previous checkpoint stays in place and the
 *           scan carries on.  The counters can include chunks beyond the frontier
 *           (statistics only).
 */
static void save_state(scan_t *S)
{
    if (!S->state_file) return;
    char tmp[4096];
    snprintf(tmp, sizeof tmp, "%s.tmp", S->state_file);
    FILE *f = fopen(tmp, "w");
    if (!f) { fprintf(stderr, "a359636: cannot write %s\n", tmp); return; }
    pthread_mutex_lock(&S->fr_mu);
    u64 fr = S->frontier;
    pthread_mutex_unlock(&S->fr_mu);
    fprintf(f, "A359636 state v1\n");
    fprintf(f, "n %d\nk_lo %" PRIu64 "\nk_hi %" PRIu64 "\nchunk %" PRIu64 "\nfrontier %" PRIu64 "\n",
            S->n, S->k_lo, S->k_hi, S->chunk, fr);
    fprintf(f, "survivors %llu\ntriples %llu\n",
            (unsigned long long)atomic_load(&S->survivors), (unsigned long long)atomic_load(&S->triples));
    pthread_mutex_lock(&S->sol_mu);
    for (int i = 0; i < S->nsolrec; i++)
        fprintf(f, "solution %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", S->sol[i].p, S->sol[i].q, S->sol[i].m);
    pthread_mutex_unlock(&S->sol_mu);
    int werr = ferror(f);
    if (fclose(f) != 0 || werr) {
        fprintf(stderr, "a359636: write error on %s; previous checkpoint kept\n", tmp);
        remove(tmp);
        return;
    }
    if (rename(tmp, S->state_file) != 0)
        fprintf(stderr, "a359636: cannot rename %s to %s: %s\n", tmp, S->state_file, strerror(errno));
}

/*
 * load_state: resume from the checkpoint, if FILE exists.
 *   In:     S with n, k_lo, k_hi and chunk already set from the command line.
 *   Out:    S->frontier, next_chunk, resume_frontier, the counters and the recorded
 *           solutions (with best_p/best_m) restored.  No file: nothing happens (fresh
 *           start).
 *   Errors: die() if the first line is not "A359636 state v1" (an empty file included)
 *           or if n, k_lo, k_hi or chunk differ from this run -- resume with the same
 *           N, START, END and -c.  Unknown lines are ignored.
 */
static void load_state(scan_t *S)
{
    FILE *f = fopen(S->state_file, "r");
    if (!f) return;
    char line[512];
    if (!fgets(line, sizeof line, f) || strncmp(line, "A359636 state v1", 16)) die("bad state file");
    int n = -1; u64 klo = 0, khi = 0, chunk = 0, fr = 0;
    unsigned long long surv = 0, trip = 0;
    while (fgets(line, sizeof line, f)) {
        u64 p, q, m;
        if (sscanf(line, "n %d", &n) == 1) continue;
        if (sscanf(line, "k_lo %" SCNu64, &klo) == 1) continue;
        if (sscanf(line, "k_hi %" SCNu64, &khi) == 1) continue;
        if (sscanf(line, "chunk %" SCNu64, &chunk) == 1) continue;
        if (sscanf(line, "frontier %" SCNu64, &fr) == 1) continue;
        if (sscanf(line, "survivors %llu", &surv) == 1) continue;
        if (sscanf(line, "triples %llu", &trip) == 1) continue;
        if (sscanf(line, "solution %" SCNu64 " %" SCNu64 " %" SCNu64, &p, &q, &m) == 3) {
            if (S->nsolrec < MAXSOL) {
                S->sol[S->nsolrec].p = p; S->sol[S->nsolrec].q = q; S->sol[S->nsolrec].m = m; S->nsolrec++;
            }
            atomic_fetch_add(&S->nsol, 1);
            if (!S->best_p || p < S->best_p) { S->best_p = p; S->best_m = m; }
        }
    }
    fclose(f);
    if (n != S->n || klo != S->k_lo || khi != S->k_hi || chunk != S->chunk)
        die("state file %s belongs to a different scan (n=%d k_lo=%" PRIu64 " k_hi=%" PRIu64 " chunk=%" PRIu64 ")",
            S->state_file, n, klo, khi, chunk);
    S->frontier = fr;
    S->resume_frontier = fr;
    atomic_store(&S->next_chunk, fr);
    atomic_store(&S->survivors, surv);
    atomic_store(&S->triples, trip);
    if (!g_quiet) {
        char c1[32];
        fmt_commas(c1, 6 * (klo + fr * chunk) + 3);
        fprintf(stderr, "resuming from %s: m = %s (%.2f%%), %d solution(s) on file\n",
                S->state_file, c1, 100.0 * (double)(fr * chunk) / (double)(khi - klo), S->nsolrec);
    }
}

/*
 * run_scan: the 'scan' command (also the work horse of 'selftest').
 *   In:     n = level; start, end: every m = 3 (mod 6) with start <= m <= end + 2 is
 *           examined; nthreads >= 1; chunk = k per chunk (0 = 2^26; rounded up to whole
 *           segments); T = sieve bound (0 = automatic); state_file = -S FILE or NULL;
 *           interval = seconds between checkpoints; scan_all = -a; certain = output flag.
 *   Out:    returns the smallest solution found and sets *certain when every m below
 *           its m has been examined (with start <= 9 that is a(n)); otherwise returns
 *           0 / the smallest solution with *certain false.  Prints the start-up lines,
 *           the status line and the summary on stderr; triples, solutions and one
 *           result line on stdout (THEORY section 9).
 *   Errors: die() for END within 65536 of 2^64, START > END, an empty m range, out of
 *           memory, primesieve failure, a -T beyond the prime table, a mismatched
 *           checkpoint or a failed pthread_create.
 */
static u64 run_scan(int n, u64 start, u64 end, int nthreads, u64 chunk, u32 T, const char *state_file,
                    int interval, bool scan_all, bool *certain)
{
    scan_t *S = xmalloc(sizeof *S);
    memset(S, 0, sizeof *S);
    S->n = n;
    S->scan_all = scan_all;
    S->state_file = state_file;
    pthread_mutex_init(&S->sol_mu, NULL);
    pthread_mutex_init(&S->fr_mu, NULL);
    pthread_cond_init(&S->fr_cv, NULL);

    /* m = 3 (mod 6), start <= m <= end + 2 */
    if (end > UINT64_MAX - 65536) die("range too large for 64-bit arithmetic");
    if (start > end) die("empty range (START > END)");
    u64 m_lo = start < 9 ? 9 : start;
    while (m_lo % 6 != 3) m_lo++;
    u64 m_hi = end + 2;                   /* inclusive */
    if (m_hi < m_lo) die("empty range");
    S->k_lo = (m_lo - 3) / 6;
    S->k_hi = (m_hi - 3) / 6 + 1;
    u64 Bmax = m_hi + 1;

    /* prime table: sieve primes up to T, trial division up to cbrt(Bmax) with margin
       (at least up to 2^20, which covers every automatic T and the primes above T
       that compute_c() needs) */
    u64 plim = (u64)iroot(Bmax, 3) + 1000;
    if (plim < 1u << 20) plim = 1u << 20;
    if (plim < 60000) plim = 60000;
    if (!g_primes || g_plimit < plim) build_primes(plim);

    /* T and c: with T = 0 pick the smallest sieve bound that still forces
       min(n,5) prime factors <= T into every target (survivors ~1e-5 then) */
    if (T == 0) {
        static const u32 cand[] = { 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 1000, 1250, 1500,
                                    2000, 2500, 3000, 4000, 5000, 7000, 10000, 15000, 20000, 30000, 50000 };
        int want = n < 5 ? n : 5;
        T = cand[sizeof cand / sizeof *cand - 1];
        for (size_t i = 0; i < sizeof cand / sizeof *cand; i++)
            if (compute_c(n, cand[i], Bmax) >= want) { T = cand[i]; break; }
    }
    S->T = T;
    S->c = compute_c(n, T, Bmax);
    S->thr = S->c - 1;
    setup_sieve(S);

    /* chunk: default 2^26 k (a fraction of a second per thread), whole segments only */
    if (chunk == 0) chunk = 1u << 26;
    if (chunk % SEG) chunk += SEG - chunk % SEG;
    S->chunk = chunk;
    S->nchunks = (S->k_hi - S->k_lo + chunk - 1) / chunk;

    /* resume: frontier, counters and known solutions from the checkpoint, if any */
    if (state_file) load_state(S);

    /* start-up lines on stderr */
    if (!g_quiet) {
        char c1[32], c2[32];
        fmt_commas(c1, m_lo); fmt_commas(c2, m_hi);
        double marks = 0;
        for (size_t i = 0; i < S->nsp; i++) marks += 3.0 / S->sp[i];
        fprintf(stderr, "A359636 level n=%d: m in [%s, %s] (m = 3 mod 6, %" PRIu64 " values), "
                "%d threads, chunk %" PRIu64 "\n", n, c1, c2, S->k_hi - S->k_lo, nthreads, chunk);
        fprintf(stderr, "sieve primes 5..%u (%zu strided), each target needs >= %d prime factors <= %u "
                "(threshold %d after the forced 2/3), %.2f marks per k\n",
                S->T, S->nsp, S->c, S->T, S->thr, marks);
    }

    /* start the workers */
    S->t0 = now();
    pthread_t *th = xmalloc(nthreads * sizeof *th);
    worker_arg *wa = xmalloc(nthreads * sizeof *wa);
    for (int i = 0; i < nthreads; i++) {
        wa[i].S = S; wa[i].id = i;
        if (pthread_create(&th[i], NULL, worker, &wa[i])) die("pthread_create");
    }
    /* main thread: wait on fr_cv in steps of at most 0.2 s, write a checkpoint every
       `interval` seconds, and leave when every chunk is done, the early stop has been
       triggered, or g_stop is set */
    double last_save = now();
    for (;;) {
        bool all_done;
        pthread_mutex_lock(&S->fr_mu);
        all_done = S->frontier >= S->nchunks || S->stop_now;
        if (!all_done && !g_stop) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += 200000000L;
            if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
            pthread_cond_timedwait(&S->fr_cv, &S->fr_mu, &ts);
        }
        pthread_mutex_unlock(&S->fr_mu);
        if (all_done || g_stop) break;
        if (state_file && now() - last_save >= interval) { save_state(S); last_save = now(); }
    }
    /* wake workers blocked on the ring so that they see g_stop, then wait for all */
    if (g_stop) { pthread_mutex_lock(&S->fr_mu); pthread_cond_broadcast(&S->fr_cv); pthread_mutex_unlock(&S->fr_mu); }
    for (int i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
    if (state_file) save_state(S);
    if (!g_quiet) progress(S, true);

    /* covered_m: every m = 3 (mod 6) up to it has been examined (THEORY section 2) */
    u64 covered_k = S->frontier >= S->nchunks ? S->k_hi : S->k_lo + S->frontier * S->chunk;
    u64 covered_m = covered_k ? 6 * covered_k - 3 : 0;   /* all m < 6*covered_k+3 done */
    u64 result = 0;
    *certain = false;
    if (S->best_p && S->best_m <= covered_m) { result = S->best_p; *certain = true; }
    if (!g_quiet) {
        char c1[32];
        fmt_commas(c1, covered_m);
        fprintf(stderr, "scanned m <= %s: %llu survivors, %llu triples, %llu solutions\n", c1,
                (unsigned long long)atomic_load(&S->survivors),
                (unsigned long long)atomic_load(&S->triples),
                (unsigned long long)atomic_load(&S->nsol));
    }
    /* The result line.  A scan with START > 9 ("partial") speaks only about its own m
       range.  For a complete scan from 0 the Coverage lemma gives "no qualifying gap
       with p <= covered_m + 1"; capped at END this is END itself unless END = 5 or
       0 (mod 6), where the prime END resp. END - 1 is not covered (its m is END + 4
       resp. END + 3).  A scan stopped early reports covered_m - 2, the form used in
       the published bounds (equivalent, as no prime lies between the two). */
    bool partial = m_lo > 9;
    bool complete = S->frontier >= S->nchunks;
    if (*certain && !partial) {
        printf("a(%d) = %" PRIu64 "\n", n, result);
    } else if (*certain) {
        printf("smallest qualifying p with m in [%" PRIu64 ", %" PRIu64 "] is %" PRIu64 " (partial range: a(%d) <= this)\n",
               m_lo, covered_m, result, n);
    } else if (S->best_p) {
        printf("a(%d) <= %" PRIu64 " (scan incomplete below it)\n", n, S->best_p);
    } else if (!partial && complete) {
        u64 B = covered_m + 1 < end ? covered_m + 1 : end;
        printf("a(%d) > %" PRIu64 " (no qualifying gap with p <= %" PRIu64 ")\n", n, B, B);
    } else if (!partial) {
        printf("a(%d) > %" PRIu64 " (scan stopped early; no qualifying gap with p below that)\n", n,
               covered_m > 2 ? covered_m - 2 : 0);
    } else {
        printf("no qualifying gap with m in [%" PRIu64 ", %" PRIu64 "] (partial range%s)\n",
               m_lo, covered_m, complete ? "" : ", stopped early");
    }
    fflush(stdout);
    free(S->sp); free(S->sr); free(S->patC); free(th); free(wa); free(S);
    return result;
}


/* ------------------------------------------------------------------ */
/* Hunt: structured search for upper bounds                            */
/* ------------------------------------------------------------------ */
/*
 * Enumerate m = 3^e * p_1 * ... * p_(n-1) (e = 1, 2; distinct primes
 * 5 <= p_i <= L) with m <= END+2 and test omega(m-1) >= n, omega(m+1) >= n
 * and then the gap, exactly as in the scan.  This covers only the m whose
 * own prime factors are all small (and, apart from 3^2, squarefree), so it can
 * miss solutions, but it visits about 1.5e11 candidates for n = 9, L = 1000
 * below Corneth's bound instead of 1.3e16 sieve positions, and any hit is an
 * upper bound for a(n).  Work is split over threads by e and the two smallest
 * primes.  (History: hunts lowered the a(9) bound to 33955545649252303 before
 * the exhaustive GPU scan found a(9) = 31610535900218923, whose m contains 5^2.)
 */
/*
 * hunt_t: the state of one 'hunt'.  mu protects best_p/best_m; the counters are
 * atomics; everything else is read-only while the threads run.
 */
typedef struct {
    int      n;                /* level */
    u64      mmax;             /* largest m considered (END + 2) */
    u32     *pr; int npr;      /* primes 5..L */
    u64      Q;                /* if > 0: the last prime is any prime <= Q (not just <= L) */
    u64      ntasks;           /* 2 * npr*(npr-1)/2: e = 1, 2 times the pair of smallest primes */
    atomic_ullong next_task, tested, triples, nsol;   /* task counter and statistics */
    u64      best_p, best_m;   /* smallest solution so far and its m (0 = none) */
    pthread_mutex_t mu;
    double   t0;               /* start time (now()) */
} hunt_t;

/*
 * hunt_leaf: test one hunt candidate m (omega(m) = n by construction).
 *   In:     H; m = 3^e * (n-1 distinct primes >= 5) <= mmax; tested = the calling
 *           thread's candidate counter.
 *   Out:    counts the candidate; if omega(m-1) and omega(m+1) are >= n, prints the
 *           triple line, checks the gap at prevprime(m-1) and, if it qualifies, records
 *           it and prints the SOLUTION block.  No de-duplication: a gap containing two
 *           enumerated m would be printed twice.
 *   Errors: none.  Called concurrently; uses out_mu and H->mu (in that nesting order:
 *           H->mu, then out_mu).
 */
static void hunt_leaf(hunt_t *H, u64 m, u64 *tested)
{
    (*tested)++;
    if (!omega_at_least(m - 1, H->n)) return;
    if (!omega_at_least(m + 1, H->n)) return;
    atomic_fetch_add(&H->triples, 1);
    u64 p = prev_prime(m - 1), q;
    pthread_mutex_lock(&out_mu);
    if (stderr_tty && !g_quiet) fputs("\r\033[K", stderr);
    printf("triple   n=%d m=%" PRIu64 "  omega(m-1,m,m+1) = %d %d %d  prevprime=%" PRIu64 " (%+" PRId64 ")\n",
           H->n, m, omega_exact(m - 1), omega_exact(m), omega_exact(m + 1), p, (int64_t)(p - m));
    fflush(stdout);
    pthread_mutex_unlock(&out_mu);
    if (!gap_qualifies(p, H->n, &q)) return;
    pthread_mutex_lock(&H->mu);
    atomic_fetch_add(&H->nsol, 1);
    if (!H->best_p || p < H->best_p) { H->best_p = p; H->best_m = m; }
    pthread_mutex_lock(&out_mu);
    if (stderr_tty && !g_quiet) fputs("\r\033[K", stderr);
    printf("SOLUTION n=%d p=%" PRIu64 " q=%" PRIu64 " gap=%" PRIu64 " (m=%" PRIu64 ")\n", H->n, p, q, q - p, m);
    char buf[256];
    for (u64 x = p + 1; x < q; x++) {
        fmt_factorization(buf, x);
        printf("    %" PRIu64 " = %s  omega=%d\n", x, buf, omega_exact(x));
    }
    fflush(stdout);
    pthread_mutex_unlock(&out_mu);
    pthread_mutex_unlock(&H->mu);
}

/*
 * hunt_dfs: depth-first enumeration of the remaining primes of m.
 *   In:     H; start = index in H->pr of the smallest prime still allowed (primes are
 *           chosen in increasing order); depth = primes chosen so far; prod = 3^e times
 *           those primes; tested = counter.
 *   Out:    calls hunt_leaf() for every completion with prod * (rest) <= mmax.  Prunes
 *           with the cheapest completion (the next need-1 consecutive primes of H->pr).
 *           With -Q the last prime comes from the whole prime table: every prime above
 *           the previous one and <= min(Q, mmax / prod).
 *   Errors: none; recursion depth <= n.  Known gap with -Q: the pruning still assumes
 *           all but the last prime come from H->pr, so tuples whose next-to-last prime
 *           is the largest prime <= L are skipped (harmless for a heuristic search).
 */
static void hunt_dfs(hunt_t *H, int start, int depth, u64 prod, u64 *tested)
{
    int need = H->n - 1 - depth;
    if (need == 0) { hunt_leaf(H, prod, tested); return; }
    if (need == 1 && H->Q) {
        /* last prime free: every prime q with (last chosen prime) < q <= min(Q, mmax/prod) */
        u64 last = start > 0 ? H->pr[start - 1] : 4;
        u64 qmax = H->mmax / prod;
        if (qmax > H->Q) qmax = H->Q;
        size_t lo = 0, hi = g_nprimes;                 /* first prime > last */
        while (lo < hi) { size_t mid = (lo + hi) / 2; if (g_primes[mid] <= last) lo = mid + 1; else hi = mid; }
        for (size_t i = lo; i < g_nprimes && g_primes[i] <= qmax; i++) hunt_leaf(H, prod * g_primes[i], tested);
        return;
    }
    for (int i = start; i < H->npr; i++) {
        u128 q = (u128)prod * H->pr[i];
        u128 mn = q;                                   /* cheapest completion */
        for (int j = 1; j < need; j++) {
            if (i + j >= H->npr) { mn = (u128)H->mmax + 1; break; }
            mn *= H->pr[i + j];
        }
        if (mn > H->mmax) break;                       /* primes ascend: later i cost more */
        hunt_dfs(H, i + 1, depth + 1, (u64)q, tested);
    }
}

/*
 * hunt_progress: the hunt's status line on stderr (tasks done, candidates, triples,
 *   solutions, best, elapsed), same cadence and caveats as progress().
 *   In: H; final = closing call.   Out: the line.   Errors: none.
 */
static void hunt_progress(hunt_t *H, bool final)
{
    static double last = 0;
    double t = now();
    if (!final && t - last < (stderr_tty ? 1.0 : 60.0)) return;
    last = t;
    u64 done = atomic_load(&H->next_task);
    if (done > H->ntasks) done = H->ntasks;
    char el[32], best[48] = "";
    fmt_time(el, t - H->t0);
    if (H->best_p) snprintf(best, sizeof best, " best=%" PRIu64, H->best_p);
    fprintf(stderr, "%stasks %" PRIu64 "/%" PRIu64 " (%.1f%%)  tested=%llu trip=%llu sol=%llu%s  %s elapsed%s",
            stderr_tty ? "\r\033[K" : "", done, H->ntasks, 100.0 * (double)done / (double)H->ntasks,
            (unsigned long long)atomic_load(&H->tested), (unsigned long long)atomic_load(&H->triples),
            (unsigned long long)atomic_load(&H->nsol), best, el, stderr_tty && !final ? "" : "\n");
    fflush(stderr);
}

/* Argument of hunt_worker(): the hunt and the thread number (thread 0 draws the status line). */
typedef struct { hunt_t *H; int id; } hunt_arg;

/*
 * hunt_worker: one hunt thread.  Task t fixes e = t / npairs (m = 3*... or 9*...) and the
 * two smallest primes pr[i] < pr[j] (pair number t % npairs, lexicographic order), then
 * enumerates the rest with hunt_dfs().
 *   In:     arg = hunt_arg *.
 *   Out:    NULL; candidates, triples and solutions via hunt_leaf().
 *   Errors: none; stops taking tasks when g_stop is set.
 */

static void *hunt_worker(void *arg)
{
    hunt_t *H = ((hunt_arg *)arg)->H;
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    u64 npairs = (u64)H->npr * (H->npr - 1) / 2;
    for (;;) {
        u64 t = atomic_fetch_add(&H->next_task, 1);
        if (t >= H->ntasks || g_stop) break;
        int e = (int)(t / npairs);                    /* 0: 3^1, 1: 3^2 */
        u64 pair = t % npairs;
        /* pair -> (i, j), i < j, lexicographic */
        int i = 0;
        u64 acc = 0;
        while (acc + (u64)(H->npr - 1 - i) <= pair) { acc += (u64)(H->npr - 1 - i); i++; }
        int j = i + 1 + (int)(pair - acc);
        u64 base = e ? 9 : 3;
        u128 prod = (u128)base * H->pr[i] * H->pr[j];
        u64 tested = 0;
        if (prod <= H->mmax) hunt_dfs(H, j + 1, 2, (u64)prod, &tested);
        atomic_fetch_add(&H->tested, tested);
        if (((hunt_arg *)arg)->id == 0 && !g_quiet) hunt_progress(H, false);
    }
    return NULL;
}

/*
 * cmd_hunt: the 'hunt' command (THEORY section 8).
 *   In:     n >= 3; end (m <= end + 2); L = largest small prime; Q = 0 or the bound for the
 *           free last prime; nthreads >= 1.
 *   Out:    returns 0.  Prints triples, SOLUTION blocks and either "a(N) <= P (hunt; m = M)"
 *           or "hunt found no qualifying gap ...".  Results are upper bounds only.
 *   Errors: die() for n < 3, fewer than n-1 primes in [5, L], out of memory, primesieve
 *           failure or a failed pthread_create.  There is no checkpoint, and an interrupted
 *           hunt prints the same summary for the tasks it finished (check the task count on
 *           the last status line).  With n = 3 both primes come from the task's pair, so -Q
 *           has no effect.
 */
static int cmd_hunt(int n, u64 end, u32 L, u64 Q, int nthreads)
{
    if (n < 3) die("hunt needs N >= 3");
    hunt_t *H = xmalloc(sizeof *H);
    memset(H, 0, sizeof *H);
    H->n = n;
    H->mmax = end + 2;
    H->Q = Q;
    pthread_mutex_init(&H->mu, NULL);
    u64 plim = (u64)iroot(H->mmax, 3) + 1000;
    if (plim < 1u << 20) plim = 1u << 20;
    if (plim < L + 1000) plim = L + 1000;
    if (plim < Q + 1000) plim = Q + 1000;
    build_primes(plim);
    H->pr = xmalloc((g_nprimes + 1) * sizeof(u32));
    for (size_t i = 0; i < g_nprimes && g_primes[i] <= L; i++)
        if (g_primes[i] >= 5) H->pr[H->npr++] = g_primes[i];
    if (H->npr < n - 1) die("too few primes below L=%u for n=%d", L, n);
    H->ntasks = 2 * ((u64)H->npr * (H->npr - 1) / 2);
    if (!g_quiet) {
        char c1[32];
        fmt_commas(c1, H->mmax);
        fprintf(stderr, "A359636 hunt n=%d: m = 3^e * (%d primes in [5,%u])%s <= %s, %d threads, %" PRIu64 " tasks\n",
                n, Q ? n - 2 : n - 1, L, Q ? " * (one prime <= Q)" : "", c1, nthreads, H->ntasks);
        if (Q) fprintf(stderr, "Q = %" PRIu64 "\n", Q);
    }
    H->t0 = now();
    pthread_t *th = xmalloc(nthreads * sizeof *th);
    hunt_arg *ha = xmalloc(nthreads * sizeof *ha);
    for (int i = 0; i < nthreads; i++) {
        ha[i].H = H; ha[i].id = i;
        if (pthread_create(&th[i], NULL, hunt_worker, &ha[i])) die("pthread_create");
    }
    for (int i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
    if (!g_quiet) hunt_progress(H, true);
    if (!g_quiet)
        fprintf(stderr, "hunt tested %llu candidates: %llu triples, %llu solutions\n",
                (unsigned long long)atomic_load(&H->tested), (unsigned long long)atomic_load(&H->triples),
                (unsigned long long)atomic_load(&H->nsol));
    if (H->best_p) printf("a(%d) <= %" PRIu64 " (hunt; m = %" PRIu64 ")\n", n, H->best_p, H->best_m);
    else printf("hunt found no qualifying gap with m <= %" PRIu64 " (n=%d, L=%u)\n", H->mmax, n, L);
    fflush(stdout);
    free(H->pr); free(th); free(ha); free(H);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Commands                                                            */
/* ------------------------------------------------------------------ */

/* default_threads: number of online CPUs (at least 1), the default for -t. */
static int default_threads(void)
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n < 1 ? 1 : (int)n;
}

/*
 * cmd_verify: the 'verify' command: check one gap from scratch and show it.
 *   In:     n = level; p = the prime that starts the gap.
 *   Out:    prints the gap and the factorization and omega of every number in it, marking
 *           the failures; returns 0 if the gap qualifies, 1 otherwise (also 1 if p is not
 *           prime).  It uses omega_exact() (Pollard rho), so it is independent of the sieve
 *           and of omega_at_least().
 *   Errors: none.  For p = 2 the "twin prime" note is printed although that gap is 1.
 */
static int cmd_verify(int n, u64 p)
{
    build_primes(1u << 21);
    char buf[256];
    if (!is_prime(p)) { printf("%" PRIu64 " is not prime\n", p); return 1; }
    u64 q = next_prime(p);
    printf("p = %" PRIu64 ", next prime q = %" PRIu64 ", gap %" PRIu64 "\n", p, q, q - p);
    bool ok = q - p >= 4;
    if (!ok) printf("p is in A001359 (twin prime), excluded\n");
    for (u64 x = p + 1; x < q; x++) {
        fmt_factorization(buf, x);
        int w = omega_exact(x);
        printf("  %" PRIu64 " = %s  omega=%d%s\n", x, buf, w, w >= n ? "" : "  <-- fails");
        if (w < n) ok = false;
    }
    printf("gap after %" PRIu64 " %s for level %d\n", p, ok ? "QUALIFIES" : "does not qualify", n);
    return ok ? 0 : 1;
}

/*
 * cmd_selftest: the 'selftest' command.
 *   In:     nthreads; with8 = also reproduce a(8) (about 10 minutes on the M1 Pro).
 *   Out:    returns 0 if every check passes, 1 otherwise; prints one line per check.
 *           Checks: run_scan() from 0 reproduces a(1..7) (and a(8)) with certainty, and
 *           omega_at_least() agrees with exact factoring for n = 1..12 on numbers that
 *           include the gaps of known terms, primorials, products of large primes and a
 *           prime square.
 *   Errors: none.
 */
static int cmd_selftest(int nthreads, bool with8)
{
    int fails = 0;
    int top = with8 ? 8 : 7;
    for (int n = 1; n <= top; n++) {
        bool certain;
        double t = now();
        u64 r = run_scan(n, 0, KNOWN[n], nthreads, n <= 6 ? 1u << 20 : 1u << 26, 0, NULL, 60, false, &certain);
        bool ok = certain && r == KNOWN[n];
        printf("selftest n=%d: got %" PRIu64 "%s expected %" PRIu64 " -> %s  (%.1fs)\n", n, r,
               certain ? "" : " (uncertain)", KNOWN[n], ok ? "ok" : "FAIL", now() - t);
        fflush(stdout);
        if (!ok) fails++;
    }
    /* spot checks of omega_at_least against exact factoring */
    build_primes(1u << 21);
    u64 tests[] = { 8, 9, 10, 20, 21, 22, 644, 645, 646, 51428, 51429, 51430, 8083634, 8083635,
                    1077940148ULL, 75582271490ULL, 34710483181814ULL, 34710483181815ULL, 34710483181816ULL,
                    76340177205657728ULL, 76340177205657729ULL, 76340177205657730ULL,
                    223092870ULL, 6469693230ULL, 200560490130ULL, 1000000007ULL * 998244353ULL,
                    (u64)1000003 * 1000003, 6ULL * 1000003 * 1000033 };
    for (size_t i = 0; i < sizeof tests / sizeof *tests; i++) {
        int w = omega_exact(tests[i]);
        for (int n = 1; n <= 12; n++) {
            bool got = omega_at_least(tests[i], n), exp = w >= n;
            if (got != exp) { printf("omega_at_least(%" PRIu64 ", %d) = %d, expected %d FAIL\n", tests[i], n, got, exp); fails++; }
        }
    }
    printf("selftest %s\n", fails ? "FAILED" : "passed");
    return fails ? 1 : 0;
}

/* usage: print the command summary on stderr and exit with status 2. */
static void usage(void)
{
    fputs("usage:\n"
          "  a359636 scan N [START] END [-t THREADS] [-c CHUNK] [-T SIEVEMAX] [-S STATE] [-i SECS] [-a] [-q]\n"
          "  a359636 hunt N END [-L L] [-Q Q] [-t THREADS] [-q]\n"
          "  a359636 verify N P\n"
          "  a359636 omega X ...\n"
          "  a359636 selftest [-t THREADS] [-8]\n", stderr);
    exit(2);
}

/*
 * main: parse the command line and dispatch.
 *   In:     argv[1] = scan | hunt | verify | omega | selftest, then that command's arguments
 *           (header, Usage).  Options and positional numbers may be mixed; a token that
 *           starts with '-' followed by a non-digit is taken as an option.
 *   Out:    the command's exit status (THEORY section 10).
 *   Errors: unknown command or option, missing or surplus arguments -> usage() (exit 2);
 *           malformed numbers -> die() (exit 1).  The atoi() fields (N, -t, -i) are only
 *           checked where noted (scan: 1 <= N <= 20; hunt: N >= 3; -t < 1 becomes 1).
 */
int main(int argc, char **argv)
{
    stderr_tty = isatty(2);
    if (argc < 2) usage();
    /* Ctrl-C and SIGTERM request an orderly stop (checkpoint and summary) */
    signal(SIGINT, on_sigint);
    signal(SIGTERM, on_sigint);
    const char *cmd = argv[1];

    /* selftest [-t T] [-8] [-q] */
    if (!strcmp(cmd, "selftest")) {
        int nthreads = default_threads();
        bool with8 = false;
        for (int i = 2; i < argc; i++) {
            if (!strcmp(argv[i], "-t") && i + 1 < argc) nthreads = atoi(argv[++i]);
            else if (!strcmp(argv[i], "-8")) with8 = true;
            else if (!strcmp(argv[i], "-q")) g_quiet = true;
            else usage();
        }
        return cmd_selftest(nthreads, with8);
    }
    /* hunt N END [-L L] [-Q Q] [-t T] [-q] */
    if (!strcmp(cmd, "hunt")) {
        if (argc < 4) usage();
        int n = atoi(argv[2]);
        u64 end = 0; u32 L = 1000; u64 Q = 0; int nthreads = default_threads(); int npos = 0;
        for (int i = 3; i < argc; i++) {
            if (!strcmp(argv[i], "-L") && i + 1 < argc) L = (u32)parse_num(argv[++i]);
            else if (!strcmp(argv[i], "-Q") && i + 1 < argc) Q = parse_num(argv[++i]);
            else if (!strcmp(argv[i], "-t") && i + 1 < argc) nthreads = atoi(argv[++i]);
            else if (!strcmp(argv[i], "-q")) g_quiet = true;
            else if (argv[i][0] == '-' && !isdigit((unsigned char)argv[i][1])) usage();
            else if (npos == 0) { end = parse_num(argv[i]); npos++; }
            else usage();
        }
        if (!npos) usage();
        if (nthreads < 1) nthreads = 1;
        return cmd_hunt(n, end, L, Q, nthreads);
    }
    /* verify N P */
    if (!strcmp(cmd, "verify")) {
        if (argc != 4) usage();
        return cmd_verify(atoi(argv[2]), parse_num(argv[3]));
    }
    /* omega X ... : factor each number, print omega and whether it is prime */
    if (!strcmp(cmd, "omega")) {
        build_primes(1u << 21);
        char buf[256];
        for (int i = 2; i < argc; i++) {
            u64 x = parse_num(argv[i]);
            fmt_factorization(buf, x);
            printf("%" PRIu64 " = %s  omega=%d%s\n", x, buf, omega_exact(x), is_prime(x) ? " (prime)" : "");
        }
        return 0;
    }
    /* scan N [START] END [-t T] [-c CHUNK] [-T SIEVEMAX] [-S STATE] [-i SECS] [-a] [-q] */
    if (!strcmp(cmd, "scan")) {
        if (argc < 4) usage();
        int n = atoi(argv[2]);
        if (n < 1 || n > 20) die("N must be 1..20");
        u64 pos[2]; int npos = 0;
        int nthreads = default_threads();
        u64 chunk = 0; u32 T = 0; const char *state = NULL; int interval = 60; bool all = false;
        for (int i = 3; i < argc; i++) {
            if (!strcmp(argv[i], "-t") && i + 1 < argc) nthreads = atoi(argv[++i]);
            else if (!strcmp(argv[i], "-c") && i + 1 < argc) chunk = parse_num(argv[++i]);
            else if (!strcmp(argv[i], "-T") && i + 1 < argc) T = (u32)parse_num(argv[++i]);
            else if (!strcmp(argv[i], "-S") && i + 1 < argc) state = argv[++i];
            else if (!strcmp(argv[i], "-i") && i + 1 < argc) interval = atoi(argv[++i]);
            else if (!strcmp(argv[i], "-a")) all = true;
            else if (!strcmp(argv[i], "-q")) g_quiet = true;
            else if (argv[i][0] == '-' && !isdigit((unsigned char)argv[i][1])) usage();
            else if (npos < 2) pos[npos++] = parse_num(argv[i]);
            else usage();
        }
        if (npos == 0) usage();
        u64 start = npos == 2 ? pos[0] : 0, end = pos[npos - 1];
        if (nthreads < 1) nthreads = 1;
        if (T && T < 30) T = 30;         /* a tiny -T only weakens the sieve; keep 5..29 */
        bool certain;
        run_scan(n, start, end, nthreads, chunk, T, state, interval, all, &certain);
        return 0;
    }
    usage();
    return 2;
}
