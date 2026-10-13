/*
 * a248701.c
 *
 * Compute and extend OEIS A248701 and its three companions A248702-A248704:
 *
 *   A248701  "Smallest prime such that the preceding n prime gaps are increasing
 *             and the following n prime gaps are decreasing."
 *             Known terms (n = 1..7): 3, 7, 359, 7853, 96401, 2812099, 294276293.
 *   A248702  the same with decreasing gaps before and increasing gaps after (a valley);
 *             offset 0, a(0) = 2.  Known (n = 0..6): 2, 3, 19, 43, 2687, 179819, 1107791,
 *             "a(7) >= 8960453 if it exists".
 *   A248703  strict peak: strictly increasing gaps before, strictly decreasing after.
 *             The entry originally had offset 1 with the index counting strict steps
 *             (a(1) = 23 has two gaps on each side); in Sep 2026 the OEIS editors changed
 *             the offset to 2 so that a(n) has n gaps on each side, which is the
 *             convention used here.  Known (n = 2..7): 23, 1439, 21433, 1130863,
 *             19881311, 331542583.
 *   A248704  strict valley, n gaps each side.  Known (n = 1..6): 3, 19, 1429, 25243,
 *             340577, 1107791.
 *
 * (The "known terms" above are the ones the OEIS had when this program was
 * written; they are what the selftest checks against.  The program has since
 * extended all four sequences, see README.md.)
 *
 * ======================================================================
 * THEORY OF OPERATION
 * ======================================================================
 *
 * 1. Definitions
 * --------------
 * Let p_1 = 2 < p_2 < ... be the primes and g_i = p_(i+1) - p_i their gaps.
 * The prime p_k sits between the gap g_(k-1) INTO it and the gap g_k OUT of it.
 * Its peak depth is min(L, M), where
 *
 *     L = length of the longest non-decreasing run of gaps that ENDS with g_(k-1)
 *     M = length of the longest non-increasing run of gaps that STARTS with g_k
 *
 * "Increasing" in the OEIS names is non-strict (a(2) = 7 has gaps 2, 2 | 4, 2)
 * and nothing is required of g_(k-1) against g_k; this is what the PARI and
 * Maple programs in the OEIS entry test.  A248701(n) is the smallest prime of
 * peak depth >= n.  Example: 359 lies in 337, 347, 349, 353, 359, 367, 373,
 * 379, 383 with gaps 10, 2, 4, 6 | 8, 6, 6, 4: L = 3 (2 <= 4 <= 6, the 10
 * before them breaks the run), M = 4 (8 >= 6 >= 6 >= 4), depth 3, a(3) = 359.
 *
 * The three companions use the same gap stream with the directions or the
 * strictness changed.  Throughout the code a "mode" is one of
 *
 *     PEAK     (A248701)  non-decreasing in,        non-increasing out
 *     VALLEY   (A248702)  non-increasing in,        non-decreasing out
 *     SPEAK    (A248703)  strictly increasing in,   strictly decreasing out
 *     SVALLEY  (A248704)  strictly decreasing in,   strictly increasing out
 *
 * and the "depth" of a prime in a mode is min(run in, run out) for that mode.
 * The sequence value a(n) is the smallest prime of depth >= n (A248702 has
 * offset 0 with a(0) = 2 by convention, A248703 has offset 2).  Each a(n) is
 * non-decreasing in n, because depth >= n+1 implies depth >= n.
 *
 * 2. The per-gap state machine
 * ----------------------------
 * A straightforward test ("for each prime, walk left while the gaps keep
 * rising, walk right while they keep falling") costs O(depth) per prime and
 * revisits every gap many times.  Instead the scanner walks the gaps once and
 * keeps four running lengths, each describing the run that ENDS at the gap
 * just consumed:
 *
 *     nd  non-decreasing run    (g_j >= g_(j-1) extends it, otherwise it restarts at 1)
 *     ni  non-increasing run    (g_j <= g_(j-1))
 *     si  strictly increasing   (g_j >  g_(j-1))
 *     sd  strictly decreasing   (g_j <  g_(j-1))
 *
 * together with rings (circular buffers of 256 entries indexed by gap number
 * mod 256) holding those four lengths for the last 256 gaps and the prime at
 * the start of each of the last 256 gaps.
 *
 * The trick that makes every (prime, depth) pair cost O(1): suppose the
 * non-increasing run ending at gap j has length r.  Then for every n <= r the
 * gaps j-n+1 .. j form a non-increasing run of length n, i.e. the prime at the
 * start of gap j-n+1 (call it the "centre" k = j-n+1) is followed by n
 * non-increasing gaps.  It has peak depth >= n exactly when the non-decreasing
 * run ending at gap k-1 (read from the ring) has length >= n.  Since each
 * (centre, n) pair is reached from exactly one (j, n), the counts of centres
 * with depth >= n are exact, and because j, hence k for fixed n, increases
 * monotonically, the first hit for a given (mode, n) within a chunk is the
 * smallest centre in that chunk.  The same argument with the roles of the
 * four run kinds exchanged gives the other three modes.
 *
 * Only runs of length >= DMIN (default 4) are examined above 2^30.  Depths 1-3
 * are known for all four sequences, and examining them would add about as much
 * work per gap as the sieve itself (roughly half of all gaps start a run of
 * length >= 2); runs of length >= 4 occur at about 5 % of the gaps, so the
 * default costs a few percent.  Chunks that start below 2^30 (which hold
 * every term known when this was written) examine every depth, so a scan
 * from 0 still reproduces the small terms.
 *
 * 3. Parallel chunked scan
 * ------------------------
 * The range [start, end] is cut into chunks of `chunk` numbers, numbered from
 * 0.  Worker threads take chunk indices from an atomic counter, so the load
 * balances itself across cores of different speeds.
 *
 * Each chunk is scanned independently: the thread sieves it with primesieve's
 * iterator, starting a margin BELOW the chunk and running WARM primes PAST its
 * end, so that the run lengths are exact for every centre inside the chunk
 * (a run is at most RUNCAP long and the margin holds at least WARM = 256
 * primes; if it does not, the margin is enlarged and the chunk redone).  Only
 * centres whose prime lies in [lo, hi) are counted, so the overlapping
 * margins never double-count.  The output of a chunk is a result_t: for each
 * mode and depth the number of centres and the smallest centre, plus the
 * prime count and the largest gap.
 *
 * Chunk results are folded into the global totals IN INCREASING CHUNK ORDER.
 * Finished results wait in a ring of RING slots until every lower-numbered
 * chunk has been folded ("completion frontier" cf: chunks [0, cf) are folded).
 * This has three consequences:
 *   - the first fold that sets first[mode][d] is the global smallest prime of
 *     depth >= d in the scanned range, so records can be printed as they are
 *     folded and are final;
 *   - the global totals always describe an initial segment [origin, frontier),
 *     so a checkpoint is just the frontier plus the totals, and resuming means
 *     starting a new chunk grid at the frontier;
 *   - a thread can never run more than RING chunks ahead of the frontier
 *     (it waits on a condition variable), which bounds memory.
 *
 * -n N (early stop): when a chunk finds a prime of peak depth >= N, no chunk
 * with a higher index is started.  All lower chunks are still finished and
 * folded, so the record is exact; chunks that were already in flight simply
 * complete and extend the folded range a little.
 *
 * 4. Checkpoints
 * --------------
 * Every -i seconds (default 60) and at the end (including after Ctrl-C) the
 * main thread writes a small text file: the frontier position `lo` (every
 * prime below it has been examined), the cumulative prime count, the largest
 * gap, the elapsed time and all first/count entries.  Running the same command
 * again (with the same or a larger END) reads it and continues from `lo`.
 * Work in flight at the time of a checkpoint is simply redone on resume.
 *
 * 5. Verification built into the program
 * --------------------------------------
 * `selftest` compares the chunk scanner with reference_scan(), a deliberately
 * naive array-based implementation, on several ranges; checks that chunking
 * (one chunk / many chunks / split chunk / scan starting mid-range) does not
 * change any count or record; checks -d 4 against -d 1 above 2^30; checks the
 * early stop and the checkpoint round trip; and reproduces all 26 terms the
 * OEIS had plus the depth histogram below 4e8 computed independently in numpy
 * (verify_a248701.py, which also re-verifies every new term's window).
 *
 * 6. Error handling and limits
 * ----------------------------
 * Unrecoverable problems (bad arguments, out of memory, primesieve errors,
 * corrupt state file, thread creation failure) call die(), which prints a
 * message and exits with status 1; nothing is left half-written because the
 * state file is written to a temporary name and renamed.  Everything is
 * 64-bit: END must stay below primesieve's limit minus 2^33 (the sieving
 * overshoot), depths above NMAX = 63 are never tracked (63 monotone gaps in a
 * row do not occur below 2^64), and run lengths saturate at RUNCAP = 250 so
 * that they fit in a byte.  Gaps themselves are u64, so no gap can overflow.
 *
 * 7. Threads and locking
 * ----------------------
 *   sc->next, sc->finished   atomics, no lock
 *   sc->mu                   protects cf, stop_chunk, done[], res[], the folded
 *                            totals (first, count, primes, maxgap, chunks_done)
 *   sc->cv                   signalled whenever the frontier moves or g_stop is set
 *   out_mu                   serialises writes to stdout/stderr so that the
 *                            status line and record lines do not interleave
 *   g_stop                   set by the SIGINT/SIGTERM handler, polled by the
 *                            workers and the main loop
 * Each worker owns its primesieve iterator, its rings and its result_t.
 *
 * Heuristics: for independent continuous gaps P(depth >= d) = 1/(d!)^2.  The
 * observed ratio count(>= d) / count(>= d+1) is about 0.85 d^2 for the two
 * non-strict modes (ties help) and about 1.25 d^2 for the strict ones (ties
 * hurt).  Below 4e8 there are two centres of peak depth >= 7, so one expects
 * about one centre of depth >= 8 per 5e8 primes, >= 9 per 3e10, >= 10 per
 * 2e12 and >= 11 per 2e14: a(8) ~ 1e10, a(9) ~ 1e12, a(10) ~ 1e14, a(11) ~
 * 1e16, each uncertain by a factor of several either way.
 *
 * Usage
 * -----
 *   a248701 scan [START] END [-t T] [-c CHUNK] [-n N] [-d DMIN] [-r DEPTH] [-S STATE] [-i SECS] [-q]
 *       Exhaustive scan of the primes in [START, END] (START defaults to 0).
 *       Prints every new record of depth >= DEPTH (default 6) in any of the four
 *       modes as it is found, then the tables of a(n) and of the depth counts.
 *       -n N stops as soon as A248701(N) is known.  -S FILE keeps a checkpoint
 *       (every SECS seconds, default 60, and on Ctrl-C); rerunning the same
 *       command resumes from it.
 *   a248701 show P [-n N]
 *       The depths of the prime P in the four modes and the primes and gaps
 *       around it.
 *   a248701 selftest [-t T]
 *
 * Numbers may be written as decimal, 2^k, 10^k, 1e12, or X+Y / X-Y of those.
 *
 * Build:  cc -O2 -std=gnu11 -pthread -I/opt/homebrew/include a248701.c \
 *            -L/opt/homebrew/lib -lprimesieve -lm -o a248701
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

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint8_t  u8;
typedef unsigned __int128 u128;

/* ------------------------------------------------------------------ */
/* Compile-time limits                                                 */
/* ------------------------------------------------------------------ */

#define NMAX    63              /* largest depth tracked (63 monotone gaps in a row never happen below 2^64) */
#define RUNCAP  250             /* run lengths saturate here; > NMAX and fits a byte */
#define RMASK   255             /* rings of the last 256 gaps */
#define WARM    256             /* primes sieved before and after a chunk so that its edge runs are exact */
#define SMALL   (1ull << 30)    /* chunks starting below this track every depth whatever -d says; holds all known terms */
#define RING    1024            /* chunk bookkeeping ring; bounds how far threads may run ahead of the frontier */
#define NMODE   4

/* The four modes; the order is used as an array index everywhere below. */
enum { PEAK, VALLEY, SPEAK, SVALLEY };
static const char *SEQ[NMODE]    = { "A248701", "A248702", "A248703", "A248704" };
static const char *MODE[NMODE]   = { "peak", "valley", "strict peak", "strict valley" };
static const int   SHIFT[NMODE]  = { 0, 0, 0, 0 };     /* OEIS index n <-> depth n + SHIFT */
static const int   FIRSTN[NMODE] = { 1, 0, 2, 1 };     /* OEIS offsets (A248703 has offset 2 since Sep 2026) */
#define KDEPTH 7
/* Terms the OEIS had when the program was written, indexed [mode][depth];
 * 0 marks "no term at this depth" (depth 0 of VALLEY is the conventional a(0) = 2).
 * Used by the summary ("matches OEIS") and by the selftest. */
static const u64 KNOWN[NMODE][KDEPTH + 1] = {           /* known terms indexed by depth */
    { 0, 3, 7, 359, 7853, 96401, 2812099, 294276293 },
    { 2, 3, 19, 43, 2687, 179819, 1107791, 0 },
    { 0, 0, 23, 1439, 21433, 1130863, 19881311, 331542583 },
    { 0, 3, 19, 1429, 25243, 340577, 1107791, 0 },
};

/* ------------------------------------------------------------------ */
/* Process-wide state                                                  */
/* ------------------------------------------------------------------ */

static bool g_quiet  = false;                   /* -q: no status line on stderr */
static bool stderr_tty;                         /* stderr is a terminal: status line is rewritten in place */
static volatile sig_atomic_t g_stop = 0;        /* set by the SIGINT/SIGTERM handler; workers stop taking chunks */
static pthread_mutex_t out_mu = PTHREAD_MUTEX_INITIALIZER;   /* serialises stdout/stderr output */

/* ------------------------------------------------------------------ */
/* Generic helpers                                                     */
/* ------------------------------------------------------------------ */

/*
 * now()
 *
 * Function:  wall-clock time from a monotonic clock, for rates and ETAs.
 * Inputs:    none.
 * Returns:   seconds as a double (arbitrary origin; only differences matter).
 * Errors:    none.
 */
static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

/*
 * die()
 *
 * Function:  print "a248701: <message>" on stderr and exit the process with
 *            status 1.  Used for every unrecoverable condition.  If stderr is a
 *            terminal the in-place status line is erased first so that the
 *            message starts on a clean line.
 * Inputs:    printf-style format and arguments.
 * Returns:   never.
 * Errors:    none (this is the error path).  Note that it is called from worker
 *            threads too; exit() then tears the whole process down, which is
 *            intended.
 */
__attribute__((noreturn)) static void die(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fputs(stderr_tty ? "\r\033[Ka248701: " : "a248701: ", stderr);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(1);
}

/*
 * note()
 *
 * Function:  print a one-line informational message on stderr, erasing the
 *            in-place status line first when stderr is a terminal.  Thread safe
 *            (takes out_mu).
 * Inputs:    printf-style format and arguments.
 * Returns:   nothing.
 * Errors:    none.
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
 * xcalloc()
 *
 * Function:  calloc() that never returns NULL.
 * Inputs:    n elements of sz bytes (n = 0 is treated as 1 so the result is
 *            always a valid pointer).
 * Returns:   zero-filled block; the caller frees it with free().
 * Errors:    calls die() when memory is exhausted.
 */
static void *xcalloc(size_t n, size_t sz)
{
    void *p = calloc(n ? n : 1, sz);
    if (!p) die("out of memory");
    return p;
}

/*
 * parse_dec()
 *
 * Function:  parse an unsigned decimal integer of exactly len characters.
 * Inputs:    s, len  the characters (not necessarily NUL-terminated at len).
 * Outputs:   *out    the value, only written on success.
 * Returns:   true on success; false if len is 0, a character is not a digit,
 *            or the value does not fit in 128 bits.
 * Errors:    none beyond the false return.
 */
static bool parse_dec(const char *s, size_t len, u128 *out)
{
    u128 r = 0;
    if (len == 0) return false;
    for (size_t i = 0; i < len; i++) {
        if (s[i] < '0' || s[i] > '9') return false;
        unsigned dgt = (unsigned)(s[i] - '0');
        if (r > (((u128)0 - 1) - dgt) / 10) return false;      /* r*10 + dgt would overflow */
        r = r * 10 + dgt;
    }
    *out = r;
    return true;
}

/*
 * parse_u128()
 *
 * Function:  parse a number in any of the command-line forms: "123", "2^64",
 *            "10^12", "1e15", and X+Y or X-Y of those ("2^54-32").  The
 *            grammar is: an optional single + or - splits the string into two
 *            halves that are parsed recursively (the left half must be
 *            non-empty, so a leading sign is not accepted); otherwise a "^"
 *            means base^exponent, an "e"/"E" means mantissa*10^exponent, and
 *            plain digits are decimal.  Exponents are limited to 400 (for ^)
 *            and 40 (for e); mantissas and bases must be integers.
 * Inputs:    s     NUL-terminated string.
 * Outputs:   *out  the value, only written on success.
 * Returns:   true on success, false on any syntax error or overflow of 128 bits
 *            (including a negative result of X-Y).
 * Errors:    none beyond the false return.
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
            if (b != 0 && r > ((u128)0 - 1) / b) return false;  /* r*b would overflow */
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
 * arg_u64()
 *
 * Function:  parse a command-line number (parse_u128 syntax) that must fit in
 *            64 bits.
 * Inputs:    s     the argument text; what  its name for the error message.
 * Returns:   the value.
 * Errors:    die() with "bad <what>: '<s>'" on a syntax error or a value
 *            above 2^64-1.
 */
static u64 arg_u64(const char *s, const char *what)
{
    u128 v;
    if (!parse_u128(s, &v) || v > UINT64_MAX) die("bad %s: '%s'", what, s);
    return (u64)v;
}

/*
 * arg_int()
 *
 * Function:  parse a small decimal command-line integer and range-check it.
 * Inputs:    s        the argument text (plain decimal only, sign allowed by
 *                     strtol but rejected by the range check for our callers);
 *            what     its name for the error message;
 *            lo, hi   inclusive bounds.
 * Returns:   the value.
 * Errors:    die() if s is empty, has trailing junk, or is outside [lo, hi].
 */
static int arg_int(const char *s, const char *what, int lo, int hi)
{
    char *end;
    long v = strtol(s, &end, 10);
    if (*s == 0 || *end != 0 || v < lo || v > hi) die("bad %s: '%s' (allowed %d..%d)", what, s, lo, hi);
    return (int)v;
}

/*
 * fmt_eng()
 *
 * Function:  format a number for the status line: plain integer below 10^6,
 *            otherwise 3 significant digits in %g style ("1.23e+13").
 * Inputs:    v    the value; buf  at least 32 bytes.
 * Returns:   buf.
 * Errors:    none.
 */
static const char *fmt_eng(double v, char *buf)
{
    if (v < 1e6) sprintf(buf, "%.0f", v);
    else sprintf(buf, "%.3g", v);
    return buf;
}

/*
 * fmt_hms()
 *
 * Function:  format a duration as hh:mm:ss, or as "DdHHh" from 100 hours on.
 *            Negative or NaN inputs are shown as zero.
 * Inputs:    s    seconds; buf  at least 32 bytes.
 * Returns:   buf.
 * Errors:    none.
 */
static const char *fmt_hms(double s, char *buf)
{
    long t = s > 0 ? (long)(s + 0.5) : 0;
    if (t >= 360000L) sprintf(buf, "%ldd%02ldh", t / 86400, (t / 3600) % 24);
    else sprintf(buf, "%02ld:%02ld:%02ld", t / 3600, (t / 60) % 60, t % 60);
    return buf;
}

/* ------------------------------------------------------------------ */
/* Scanning one chunk                                                  */
/* ------------------------------------------------------------------ */

/* Everything a chunk (or, after folding, a whole range) tells us.  "Centre"
 * means a prime P in [lo, hi) regarded as the middle of its gap pattern. */
typedef struct {
    u64 first[NMODE][NMAX + 1];     /* smallest centre in the chunk of depth >= d, 0 if none */
    u64 count[NMODE][NMAX + 1];     /* centres in the chunk of depth >= d (d >= dmin) */
    u64 nprimes;                    /* primes in [lo, hi) */
    u64 maxgap;                     /* largest gap starting in [lo, hi) */
} result_t;

/* Circular history of the last 256 gaps, indexed by gap number & RMASK.
 * Entry x describes gap j (with j & RMASK == x): the four run lengths of the
 * runs ending at gap j, and the prime at the START of gap j (so the prime
 * between gap j-1 and gap j, i.e. the centre whose gap out is gap j). */
typedef struct {
    u8  nd[RMASK + 1], ni[RMASK + 1], si[RMASK + 1], sd[RMASK + 1];   /* run lengths ending at gap j */
    u64 p[RMASK + 1];                                                 /* prime at the start of gap j */
} rings_t;

/*
 * examine()
 *
 * Function:  the heart of the state machine (see THEORY OF OPERATION, 2).
 *            The run of the kind that must FOLLOW a centre in this mode ends
 *            at gap j and has length `run`.  For every n in [dmin, min(run,
 *            NMAX)] the centre k = j - n + 1 is followed by n gaps of that run,
 *            so it has depth >= n iff the opposite run ending at gap k - 1 has
 *            length >= n; that length is read from the ring `bwd`.  Centres
 *            outside [lo, hi) belong to the warm-up margins and are skipped.
 *            Each qualifying (centre, n) increments count[mode][n] and sets
 *            first[mode][n] if it is the first such centre of the chunk (which
 *            is the smallest, because j and hence k increase monotonically).
 * Inputs:    r      chunk result being accumulated;
 *            R      the rings (for the centre primes);
 *            mode   PEAK, VALLEY, SPEAK or SVALLEY;
 *            bwd    the ring of the run kind required BEFORE a centre in this
 *                   mode (R->nd for PEAK, R->ni for VALLEY, R->si for SPEAK,
 *                   R->sd for SVALLEY);
 *            run    length of the run ending at gap j (already >= dmin);
 *            j      the gap number (0 = the first gap seen by this chunk);
 *            dmin   smallest depth to examine;
 *            lo, hi the chunk bounds.
 * Outputs:   r->count[mode][n], r->first[mode][n] updated.
 * Errors:    none.  The `n > j` test only matters in the first 63 gaps of the
 *            chunk that starts at 0 (there is no gap before the prime 2).
 */
static inline void examine(result_t *r, const rings_t *R, int mode, const u8 *bwd,
                           unsigned run, u64 j, unsigned dmin, u64 lo, u64 hi)
{
    unsigned top = run < NMAX ? run : NMAX;
    for (unsigned n = dmin; n <= top; n++) {
        if (n > j) break;                       /* the centre needs a gap before it */
        u64 k = j - n + 1;
        if (bwd[(k - 1) & RMASK] < n) continue;
        u64 P = R->p[k & RMASK];
        if (P < lo || P >= hi) continue;
        r->count[mode][n]++;
        if (!r->first[mode][n]) r->first[mode][n] = P;
    }
}

/*
 * scan_chunk()
 *
 * Function:  scan every prime in [lo, hi) as a centre and fill *r.
 *
 *            Sieving starts `margin` numbers below lo (initially 2^16) and
 *            continues until WARM primes beyond hi have been consumed, so that
 *            (a) the run lengths are exact by the time the first centre >= lo
 *            is reached (a run longer than the warm-up would have to exceed
 *            RUNCAP, which never happens), and (b) every centre < hi has seen
 *            all the gaps after it that any depth <= NMAX needs.  If fewer than
 *            WARM primes were found below lo (only conceivable for absurdly
 *            large lo), the margin is quadrupled and the chunk is redone.
 *
 *            Per gap the four run lengths are updated (saturating at RUNCAP),
 *            stored in the rings together with the prime at the gap's start,
 *            and examine() is called for each mode whose "run out" kind has
 *            reached dmin: PEAK needs a non-increasing run out (ni) and reads
 *            the non-decreasing ring (nd) backwards, VALLEY the reverse, and
 *            the strict modes likewise with si/sd.
 * Inputs:    it    a primesieve iterator owned by the calling thread;
 *            R     scratch rings owned by the calling thread (contents
 *                  irrelevant on entry);
 *            lo, hi  chunk bounds, hi > lo;
 *            dmin  smallest depth examined (1..8).
 * Outputs:   *r    zeroed, then filled: first/count per mode and depth, the
 *                  number of primes in [lo, hi) and the largest gap that starts
 *                  in [lo, hi).
 * Errors:    die() if primesieve reports an error (it->is_error, e.g. a stop
 *            beyond its limit) or if no WARM primes exist within 2^32 below lo.
 */
static void scan_chunk(primesieve_iterator *it, rings_t *R, u64 lo, u64 hi, unsigned dmin, result_t *r)
{
    u64 margin = 1ull << 16;
    for (;;) {
        memset(r, 0, sizeof *r);
        u64 start = lo > margin ? lo - margin : 0;
        primesieve_jump_to(it, start, hi + margin);     /* second argument is only a sizing hint */
        u64 prev = primesieve_next_prime(it);           /* the first prime >= start */
        u64 nbefore = 0, nafter = 0, j = 0, pg = 0;     /* j = gap number, pg = previous gap */
        unsigned nd = 1, ni = 1, si = 1, sd = 1;        /* a single gap is a run of length 1 of every kind */
        if (prev < lo) nbefore++;
        else if (prev < hi) r->nprimes++;
        for (;;) {
            u64 q = primesieve_next_prime(it);
            u64 g = q - prev;                           /* gap j runs from prev to q */
            if (j) {
                nd = g >= pg ? nd + 1 : 1;
                ni = g <= pg ? ni + 1 : 1;
                si = g >  pg ? si + 1 : 1;
                sd = g <  pg ? sd + 1 : 1;
                if (nd > RUNCAP) nd = RUNCAP;
                if (ni > RUNCAP) ni = RUNCAP;
                if (si > RUNCAP) si = RUNCAP;
                if (sd > RUNCAP) sd = RUNCAP;
            }
            unsigned x = (unsigned)j & RMASK;
            R->nd[x] = (u8)nd; R->ni[x] = (u8)ni; R->si[x] = (u8)si; R->sd[x] = (u8)sd;
            R->p[x] = prev;
            if (ni >= dmin) examine(r, R, PEAK,    R->nd, ni, j, dmin, lo, hi);
            if (nd >= dmin) examine(r, R, VALLEY,  R->ni, nd, j, dmin, lo, hi);
            if (sd >= dmin) examine(r, R, SPEAK,   R->si, sd, j, dmin, lo, hi);
            if (si >= dmin) examine(r, R, SVALLEY, R->sd, si, j, dmin, lo, hi);
            if (g > r->maxgap && prev >= lo && prev < hi) r->maxgap = g;
            if (q < lo) nbefore++;
            else if (q < hi) r->nprimes++;
            else if (++nafter > WARM) break;            /* enough look-ahead past hi */
            pg = g; prev = q; j++;
        }
        if (it->is_error) die("primesieve error in [%" PRIu64 ", %" PRIu64 ")", lo, hi);
        if (start > 0 && nbefore < WARM) {
            margin *= 4;
            if (margin > (1ull << 32)) die("fewer than %d primes within 2^32 below %" PRIu64 "?!", WARM, lo);
            continue;
        }
        return;
    }
}

/*
 * reference_scan()
 *
 * Function:  the same computation as scan_chunk() written the obvious way,
 *            for the selftest only: generate all primes of the range (with a
 *            fixed margin of 2^16 on each side) into an array and, for every
 *            centre, walk backwards while each of the four run kinds continues
 *            and forwards likewise, then take min(L, M) per mode.  O(depth)
 *            per centre and no shared state with the fast path except the
 *            definitions, so agreement between the two is meaningful.
 * Inputs:    lo, hi  range of centres.
 * Outputs:   *r      as for scan_chunk() with dmin = 1 (every depth counted).
 * Errors:    none of its own; primesieve_generate_primes() aborts the process
 *            on failure.  Ranges with fewer than 3 primes yield an empty result.
 */
static void reference_scan(u64 lo, u64 hi, result_t *r)
{
    size_t n;
    u64 margin = 1ull << 16;
    u64 *pr = primesieve_generate_primes(lo > margin ? lo - margin : 0, hi + margin, &n, UINT64_PRIMES);
    memset(r, 0, sizeof *r);
    if (n < 3) { primesieve_free(pr); return; }
    for (size_t i = 0; i < n; i++)
        if (pr[i] >= lo && pr[i] < hi) r->nprimes++;
    for (size_t i = 0; i + 1 < n; i++)
        if (pr[i] >= lo && pr[i] < hi && pr[i + 1] - pr[i] > r->maxgap) r->maxgap = pr[i + 1] - pr[i];
    for (size_t k = 1; k + 1 < n; k++) {
        u64 P = pr[k];
        if (P < lo || P >= hi) continue;
        /* runs ending at gap k-1 (gaps g_i = pr[i+1] - pr[i]): walk back while the run continues */
        int L[4] = { 1, 1, 1, 1 }, M[4] = { 1, 1, 1, 1 };
        for (size_t i = k - 1; i >= 1 && i + NMAX >= k; i--) {
            u64 a = pr[i] - pr[i - 1], b = pr[i + 1] - pr[i];   /* a = g_(i-1), b = g_i */
            bool cont[4] = { a <= b, a >= b, a < b, a > b };     /* nd, ni, si, sd */
            bool any = false;
            /* a run kind only continues if it has not been broken yet (L[m] == k - i) */
            for (int m = 0; m < 4; m++) if (L[m] == (int)(k - i) && cont[m]) { L[m]++; any = true; }
            if (!any) break;
        }
        /* runs starting at gap k: walk forward */
        for (size_t i = k; i + 2 < n && i < k + NMAX; i++) {
            u64 a = pr[i + 1] - pr[i], b = pr[i + 2] - pr[i + 1]; /* a = g_i, b = g_(i+1) */
            bool cont[4] = { b <= a, b >= a, b < a, b > a };     /* ni, nd, sd, si */
            bool any = false;
            for (int m = 0; m < 4; m++) if (M[m] == (int)(i - k + 1) && cont[m]) { M[m]++; any = true; }
            if (!any) break;
        }
        /* L[m] and M[m] are paired so that index m is the mode: PEAK = (nd in, ni out), ... */
        int depth[NMODE] = { L[0] < M[0] ? L[0] : M[0], L[1] < M[1] ? L[1] : M[1],
                             L[2] < M[2] ? L[2] : M[2], L[3] < M[3] ? L[3] : M[3] };
        for (int m = 0; m < NMODE; m++)
            for (int d = 1; d <= depth[m] && d <= NMAX; d++) {
                r->count[m][d]++;
                if (!r->first[m][d]) r->first[m][d] = P;
            }
    }
    primesieve_free(pr);
}

/* ------------------------------------------------------------------ */
/* Windows of consecutive primes around P                              */
/* ------------------------------------------------------------------ */

/*
 * prime_window()
 *
 * Function:  collect the nb consecutive primes before P, P itself and the na
 *            consecutive primes after it, using a private primesieve iterator
 *            over a span of about (nb + na + 2) * 2000 numbers on each side
 *            (quadrupled and retried if the span held fewer than nb primes
 *            below P; near the start of the primes fewer are returned).
 * Inputs:    P        the candidate prime;
 *            nb, na   how many primes to fetch on each side;
 *            w        array of at least nb + na + 1 entries.
 * Outputs:   w[0..nb'-1] the primes before P in increasing order, w[nb'] = P,
 *            w[nb'+1..nb'+na] the primes after P.
 * Returns:   nb' (= nb unless P is close to 2), or -1 if P is not prime (in
 *            which case w is untouched).
 * Errors:    die() on a primesieve error (e.g. P + span beyond its limit) or
 *            out of memory.
 */
static int prime_window(u64 P, int nb, int na, u64 *w)
{
    primesieve_iterator it;
    primesieve_init(&it);
    u64 *ring = xcalloc((size_t)nb + 1, sizeof(u64));    /* last nb+1 primes seen below P */
    u64 span = (u64)(nb + na + 2) * 2000;
    int got;
    for (;;) {
        u64 start = P > span ? P - span : 0;
        primesieve_jump_to(&it, start, P + span);
        u64 q;
        int cnt = 0;
        while ((q = primesieve_next_prime(&it)) < P) { ring[cnt % (nb + 1)] = q; cnt++; }
        if (it.is_error) die("primesieve error near %" PRIu64, P);
        if (q != P) { got = -1; break; }                /* the first prime >= P is not P: P is composite */
        if (cnt < nb && start > 0) { span *= 4; continue; }
        int have = cnt < nb ? cnt : nb;
        for (int i = 0; i < have; i++) w[i] = ring[(cnt - have + i) % (nb + 1)];
        w[have] = P;
        for (int i = 0; i < na; i++) {
            q = primesieve_next_prime(&it);
            if (it.is_error) die("primesieve error near %" PRIu64, P);
            w[have + 1 + i] = q;
        }
        got = have;
        break;
    }
    free(ring);
    primesieve_free_iterator(&it);
    return got;
}

/*
 * depths_at()
 *
 * Function:  the depth of a single prime P in all four modes, computed
 *            directly from a window of RUNCAP + 2 primes on each side (so any
 *            depth up to the cap is found).  Independent of the scanner; used
 *            by `show` and by the selftest.
 * Inputs:    P        the prime.
 * Outputs:   depth[PEAK..SVALLEY]; all 0 for P = 2 (no gap into it).
 * Returns:   true, or false if P is not prime (depth[] untouched).
 * Errors:    as prime_window().
 */
static bool depths_at(u64 P, int depth[NMODE])
{
    enum { W = RUNCAP + 2 };
    u64 *w = xcalloc(2 * W + 1, sizeof(u64));
    int nb = prime_window(P, W, W, w);
    if (nb < 0) { free(w); return false; }
    u64 bg[W], ag[W];                           /* bg[0] is the gap into P, ag[0] the gap out of it */
    for (int i = 0; i < nb; i++) bg[i] = w[nb - i] - w[nb - i - 1];
    for (int i = 0; i < W; i++) ag[i] = w[nb + 1 + i] - w[nb + i];
    /* L[m]: run into P, M[m]: run out of P, with m ordered as the modes (PEAK = nd in / ni out, ...).
     * A run is extended only while it is unbroken (L[m] == i + 1 means it still reaches gap i). */
    int L[4] = { nb > 0, nb > 0, nb > 0, nb > 0 }, M[4] = { 1, 1, 1, 1 };
    for (int i = 0; i + 1 < nb; i++) {          /* going back: bg[i+1] is the earlier gap */
        if (L[0] == i + 1 && bg[i + 1] <= bg[i]) L[0]++;   /* non-decreasing into P */
        if (L[1] == i + 1 && bg[i + 1] >= bg[i]) L[1]++;   /* non-increasing into P */
        if (L[2] == i + 1 && bg[i + 1] <  bg[i]) L[2]++;   /* strictly increasing into P */
        if (L[3] == i + 1 && bg[i + 1] >  bg[i]) L[3]++;   /* strictly decreasing into P */
    }
    for (int i = 0; i + 1 < W; i++) {
        if (M[0] == i + 1 && ag[i + 1] <= ag[i]) M[0]++;   /* non-increasing out of P */
        if (M[1] == i + 1 && ag[i + 1] >= ag[i]) M[1]++;   /* non-decreasing out of P */
        if (M[2] == i + 1 && ag[i + 1] <  ag[i]) M[2]++;   /* strictly decreasing out of P */
        if (M[3] == i + 1 && ag[i + 1] >  ag[i]) M[3]++;   /* strictly increasing out of P */
    }
    for (int m = 0; m < NMODE; m++) depth[m] = L[m] < M[m] ? L[m] : M[m];
    free(w);
    return true;
}

/*
 * print_gap_window()
 *
 * Function:  print, on the current stdout line, the primes and gaps around P
 *            in the form
 *                primes 337 .. 359 .. 389  gaps [10] 2 4 6 | 8 6 6 [4]
 *            with n gaps on each side of the "|" (the position of P) and the
 *            next gap outside in brackets, so the reader sees where the runs
 *            stop.  Near the start of the primes fewer gaps are shown on the
 *            left and the bracket is omitted.  No newline is written.
 * Inputs:    P   the prime; n   gaps per side (capped at NMAX).
 * Outputs:   text on stdout; "(P is not prime)" if P is composite.
 * Errors:    as prime_window().
 */
static void print_gap_window(u64 P, int n)
{
    if (n > NMAX) n = NMAX;
    u64 *w = xcalloc(2 * (size_t)n + 5, sizeof(u64));
    int nb = prime_window(P, n + 1, n + 1, w);
    if (nb < 0) { printf("  (%" PRIu64 " is not prime)", P); free(w); return; }
    printf("  primes %" PRIu64 " .. %" PRIu64 " .. %" PRIu64 "  gaps", w[0], P, w[nb + n + 1]);
    for (int i = 0; i < nb; i++) {
        u64 g = w[i + 1] - w[i];
        printf(i == 0 && nb == n + 1 ? " [%" PRIu64 "]" : " %" PRIu64, g);
    }
    printf(" |");
    for (int i = 0; i <= n; i++) {
        u64 g = w[nb + 1 + i] - w[nb + i];
        printf(i == n ? " [%" PRIu64 "]" : " %" PRIu64, g);
    }
    free(w);
}

/* ------------------------------------------------------------------ */
/* Scan state shared by the worker threads                             */
/* ------------------------------------------------------------------ */

/* A new record discovered while folding: mode, depth and the prime. */
typedef struct { int mode, d; u64 P; } rec_t;

typedef struct scan scan_t;

/* Per-thread state; everything in here is private to the thread. */
typedef struct {
    int id;
    pthread_t th;
    scan_t *sc;                     /* the shared scan */
    primesieve_iterator it;         /* reused for every chunk this thread scans */
    rings_t R;
    result_t r;                     /* result of the chunk being scanned */
} worker_t;

/* One scan over [start, end].  See THEORY OF OPERATION, 3 and 7. */
struct scan {
    /* parameters, read-only once the workers run */
    u64  origin, start, end, chunk, nchunks;   /* origin: where the totals begin (start of the first run when resumed) */
    unsigned dmin;                  /* -d */
    int  report, target;            /* print records of depth >= report; stop once PEAK depth >= target is found (0: never) */
    bool verbose, print_records;    /* status line on stderr; record lines on stdout */
    /* runtime */
    atomic_uint_fast64_t next;      /* next chunk index to hand out */
    atomic_int finished;            /* workers that have exited */
    pthread_mutex_t mu;             /* protects everything below */
    pthread_cond_t cv;              /* broadcast when cf advances or g_stop is set */
    u64  cf;                        /* completion frontier: chunks [0, cf) are folded */
    u64  stop_chunk;                /* no chunk above this index is started (-n) */
    u8   done[RING];                /* done[i % RING]: result of chunk i is in res[] and awaits folding */
    result_t res[RING];
    /* folded totals: everything in [origin, start + cf*chunk) */
    u64  first[NMODE][NMAX + 1];
    u64  count[NMODE][NMAX + 1];
    u64  primes, maxgap, chunks_done;
    double t0, t1, elapsed0;        /* elapsed0: time spent by earlier runs (checkpoint) */
};

/*
 * print_record()
 *
 * Function:  print one record line on stdout, e.g.
 *              A248701(8) = 12579905251  [peak depth >= 8]  primes ... gaps ...
 *            The OEIS index is the depth minus SHIFT[mode].  Holds out_mu so
 *            that the line is not interleaved with the status line.
 * Inputs:    mode, d   mode and depth of the record; P   the prime.
 * Outputs:   one line on stdout, flushed.
 * Errors:    as print_gap_window().
 */
static void print_record(int mode, int d, u64 P)
{
    pthread_mutex_lock(&out_mu);
    if (stderr_tty) fputs("\r\033[K", stderr);
    printf("%s(%d) = %" PRIu64 "  [%s depth >= %d]", SEQ[mode], d - SHIFT[mode], P, MODE[mode], d);
    print_gap_window(P, d);
    printf("\n");
    fflush(stdout);
    pthread_mutex_unlock(&out_mu);
}

/*
 * fold()
 *
 * Function:  add one chunk's result to the global totals.  Must be called
 *            with sc->mu held and in increasing chunk order; under that rule
 *            the first chunk that supplies first[m][d] holds the smallest
 *            prime of depth >= d in the whole folded range, so a newly set
 *            entry with d >= sc->report is a record worth printing and is
 *            appended to nr[] (the caller prints after releasing the lock).
 * Inputs:    sc   the scan; r   the chunk result; nr, *nn   output array of
 *            records and its fill count (nr must have room for NMODE*(NMAX+1)).
 * Outputs:   sc->count/first/primes/maxgap updated; nr[], *nn extended.
 * Errors:    none.
 */
static void fold(scan_t *sc, const result_t *r, rec_t *nr, int *nn)
{
    for (int m = 0; m < NMODE; m++)
        for (int d = 1; d <= NMAX; d++) {
            sc->count[m][d] += r->count[m][d];
            if (r->first[m][d] && !sc->first[m][d]) {
                sc->first[m][d] = r->first[m][d];
                if (d >= sc->report) { nr[*nn].mode = m; nr[*nn].d = d; nr[*nn].P = r->first[m][d]; (*nn)++; }
            }
        }
    sc->primes += r->nprimes;
    if (r->maxgap > sc->maxgap) sc->maxgap = r->maxgap;
}

/*
 * worker_main()
 *
 * Function:  thread body.  Loop: take the next chunk index from the atomic
 *            counter; wait (on sc->cv) while that index is RING or more ahead
 *            of the frontier; give up if g_stop is set or the index is beyond
 *            stop_chunk (-n); scan the chunk; then, under sc->mu, lower
 *            stop_chunk if the chunk found a PEAK centre of depth >= target,
 *            park the result in the ring, and fold every consecutive finished
 *            chunk from the frontier upwards, collecting new records; finally
 *            print the records outside the lock.
 *            Chunks starting below SMALL examine every depth (dmin = 1) so the
 *            known small terms are reproduced; the last chunk is cut at end+1.
 * Inputs:    arg   the worker_t.
 * Returns:   NULL; increments sc->finished on exit so scan_run() can tell when
 *            all workers are done.
 * Errors:    scan_chunk() may die(); nothing else fails.
 */
static void *worker_main(void *arg)
{
    worker_t *w = arg;
    scan_t *sc = w->sc;

    for (;;) {
        if (g_stop) break;
        u64 i = atomic_fetch_add(&sc->next, 1);
        if (i >= sc->nchunks) break;

        pthread_mutex_lock(&sc->mu);
        while (i >= sc->cf + RING && !g_stop) pthread_cond_wait(&sc->cv, &sc->mu);
        bool skip = g_stop || i > sc->stop_chunk;
        pthread_mutex_unlock(&sc->mu);
        if (skip) break;

        u64 lo = sc->start + i * sc->chunk;
        u64 hi = (i + 1 == sc->nchunks) ? sc->end + 1 : lo + sc->chunk;
        unsigned dmin = lo < SMALL ? 1 : sc->dmin;
        scan_chunk(&w->it, &w->R, lo, hi, dmin, &w->r);

        rec_t newrec[NMODE * (NMAX + 1)];
        int nnew = 0;
        pthread_mutex_lock(&sc->mu);
        if (sc->target && w->r.first[PEAK][sc->target] && i < sc->stop_chunk) sc->stop_chunk = i;
        sc->res[i % RING] = w->r;
        sc->done[i % RING] = 1;
        while (sc->cf < sc->nchunks && sc->done[sc->cf % RING]) {    /* advance the frontier as far as possible */
            fold(sc, &sc->res[sc->cf % RING], newrec, &nnew);
            sc->done[sc->cf % RING] = 0;
            sc->cf++;
        }
        sc->chunks_done++;
        pthread_cond_broadcast(&sc->cv);
        pthread_mutex_unlock(&sc->mu);
        if (sc->print_records)
            for (int k = 0; k < nnew; k++) print_record(newrec[k].mode, newrec[k].d, newrec[k].P);
    }
    atomic_fetch_add(&sc->finished, 1);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Checkpoint files                                                    */
/* ------------------------------------------------------------------ */

/* Contents of a checkpoint.  File format (text, one item per line):
 *     A248701-state 1
 *     dmin D
 *     origin X          where the totals begin
 *     end X             END of the run that wrote the file (informational)
 *     lo X              first number NOT yet examined; all primes below lo are done
 *     primes N          primes in [origin, lo)
 *     maxgap G          largest gap in [origin, lo)
 *     elapsed S         seconds of scanning so far, all runs
 *     first M D P       smallest prime of depth >= D in mode M (only non-zero entries)
 *     count M D N       centres of depth >= D in mode M (only non-zero entries)
 */
typedef struct {
    unsigned dmin;
    u64  origin, end, lo, primes, maxgap;
    double elapsed;
    u64  first[NMODE][NMAX + 1];
    u64  count[NMODE][NMAX + 1];
} state_t;

/*
 * write_state()
 *
 * Function:  write a checkpoint atomically: the text goes to "<path>.tmp",
 *            which is then renamed over path, so a crash mid-write leaves the
 *            previous checkpoint intact.
 * Inputs:    path   file name; st   the state.
 * Outputs:   the file.
 * Errors:    failures are reported with note() as warnings and otherwise
 *            ignored (a scan should not die because a checkpoint could not be
 *            written); the previous file, if any, survives.
 */
static void write_state(const char *path, const state_t *st)
{
    char tmp[4096];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "w");
    if (!f) { note("warning: cannot write %s: %s", tmp, strerror(errno)); return; }
    fprintf(f, "A248701-state 1\ndmin %u\norigin %" PRIu64 "\nend %" PRIu64 "\nlo %" PRIu64 "\nprimes %" PRIu64
               "\nmaxgap %" PRIu64 "\nelapsed %.1f\n", st->dmin, st->origin, st->end, st->lo, st->primes, st->maxgap, st->elapsed);
    for (int m = 0; m < NMODE; m++)
        for (int d = 1; d <= NMAX; d++) {
            if (st->first[m][d]) fprintf(f, "first %d %d %" PRIu64 "\n", m, d, st->first[m][d]);
            if (st->count[m][d]) fprintf(f, "count %d %d %" PRIu64 "\n", m, d, st->count[m][d]);
        }
    if (fclose(f) != 0 || rename(tmp, path) != 0)
        note("warning: cannot finish writing %s: %s", path, strerror(errno));
}

/*
 * read_state()
 *
 * Function:  parse a checkpoint written by write_state().  Unknown keys are
 *            ignored (forward compatibility); missing keys stay zero.
 * Inputs:    path   file name.
 * Outputs:   *st    the state (zeroed first).
 * Returns:   true if the file was read; false if it does not exist (a fresh
 *            scan then starts from scratch).
 * Errors:    die() if the file exists but is not a version-1 a248701 state
 *            file, has an unparsable number, a mode or depth out of range, or
 *            a dmin outside 1..8.
 */
static bool read_state(const char *path, state_t *st)
{
    FILE *f = fopen(path, "r");
    if (!f) return false;
    memset(st, 0, sizeof *st);
    char line[512];
    int ver = 0;
    if (!fgets(line, sizeof line, f) || sscanf(line, "A248701-state %d", &ver) != 1 || ver != 1)
        die("%s is not an a248701 state file", path);
    while (fgets(line, sizeof line, f)) {
        char key[32], v1[64], v2[64], v3[64];
        int n = sscanf(line, "%31s %63s %63s %63s", key, v1, v2, v3);
        u128 v;
        if (n < 2) continue;
        if (!strcmp(key, "dmin")) st->dmin = (unsigned)atoi(v1);
        else if (!strcmp(key, "origin")) { if (!parse_u128(v1, &v)) die("bad state"); st->origin = (u64)v; }
        else if (!strcmp(key, "end"))    { if (!parse_u128(v1, &v)) die("bad state"); st->end = (u64)v; }
        else if (!strcmp(key, "lo"))     { if (!parse_u128(v1, &v)) die("bad state"); st->lo = (u64)v; }
        else if (!strcmp(key, "primes")) { if (!parse_u128(v1, &v)) die("bad state"); st->primes = (u64)v; }
        else if (!strcmp(key, "maxgap")) { if (!parse_u128(v1, &v)) die("bad state"); st->maxgap = (u64)v; }
        else if (!strcmp(key, "elapsed")) st->elapsed = atof(v1);
        else if ((!strcmp(key, "first") || !strcmp(key, "count")) && n >= 4) {
            int m = atoi(v1), d = atoi(v2);
            if (m < 0 || m >= NMODE || d < 1 || d > NMAX || !parse_u128(v3, &v)) die("bad %s line in %s", key, path);
            if (key[0] == 'f') st->first[m][d] = (u64)v;
            else st->count[m][d] = (u64)v;
        }
    }
    fclose(f);
    if (st->dmin < 1 || st->dmin > 8) die("bad dmin in %s", path);
    return true;
}

/*
 * checkpoint()
 *
 * Function:  snapshot the folded totals of a running (or finished) scan under
 *            sc->mu and write them with write_state().  `lo` is the frontier
 *            position (start + cf*chunk), or end + 1 when every chunk is
 *            folded; the elapsed time includes earlier runs.
 * Inputs:    sc   the scan; path   checkpoint file name.
 * Outputs:   the file.
 * Errors:    as write_state() (warnings only).
 */
static void checkpoint(scan_t *sc, const char *path)
{
    state_t st;
    memset(&st, 0, sizeof st);
    pthread_mutex_lock(&sc->mu);
    st.dmin = sc->dmin;
    st.origin = sc->origin;
    st.end = sc->end;
    st.lo = sc->cf >= sc->nchunks ? sc->end + 1 : sc->start + sc->cf * sc->chunk;
    st.primes = sc->primes;
    st.maxgap = sc->maxgap;
    st.elapsed = sc->elapsed0 + now() - sc->t0;
    memcpy(st.first, sc->first, sizeof st.first);
    memcpy(st.count, sc->count, sizeof st.count);
    pthread_mutex_unlock(&sc->mu);
    write_state(path, &st);
}

/* ------------------------------------------------------------------ */
/* Running a scan                                                      */
/* ------------------------------------------------------------------ */

/*
 * scanned_to()
 *
 * Function:  the largest number whose primes have all been folded: end if the
 *            scan is complete, otherwise the number just below the frontier
 *            chunk.  Caller holds sc->mu or the scan has finished.
 * Inputs:    sc.
 * Returns:   that bound (inclusive).
 * Errors:    none.
 */
static u64 scanned_to(const scan_t *sc)
{
    return sc->cf >= sc->nchunks ? sc->end : sc->start + sc->cf * sc->chunk - 1;
}

/*
 * print_status()
 *
 * Function:  write the progress line on stderr: elapsed time, percent done,
 *            frontier position, rate (numbers per second over this run), ETA,
 *            folded prime count, largest gap, and the best depth found so far
 *            in each mode (with the prime for PEAK).  On a terminal the line
 *            is rewritten in place (no newline unless final); otherwise, or
 *            when final, it ends with a newline.
 * Inputs:    sc; t   current time; final   true for the last line.
 * Outputs:   text on stderr.
 * Errors:    none.
 */
static void print_status(scan_t *sc, double t, bool final)
{
    pthread_mutex_lock(&sc->mu);
    u64 pos = scanned_to(sc) + 1;
    u64 primes = sc->primes, maxgap = sc->maxgap;
    int best[NMODE] = { 0, 0, 0, 0 };
    u64 bestp = 0;
    for (int m = 0; m < NMODE; m++)
        for (int d = NMAX; d >= 1; d--) if (sc->first[m][d]) { best[m] = d; if (m == PEAK) bestp = sc->first[m][d]; break; }
    pthread_mutex_unlock(&sc->mu);

    double done = (double)(pos - sc->start), total = (double)(sc->end - sc->start) + 1.0;
    double el = t - sc->t0, rate = el > 0 ? done / el : 0, eta = rate > 0 ? (total - done) / rate : 0;
    char b1[32], b2[32], b3[32], b4[32], b5[32];
    pthread_mutex_lock(&out_mu);
    if (stderr_tty) fputs("\r\033[K", stderr);
    fprintf(stderr, "[%s] %5.1f%% at %s  %s/s  ETA %s  primes %s  maxgap %" PRIu64 "  depth peak %d @ %" PRIu64 " valley %d strict %d/%d",
            fmt_hms(el, b1), 100.0 * done / total, fmt_eng((double)pos, b2), fmt_eng(rate, b3), fmt_hms(eta, b4),
            fmt_eng((double)primes, b5), maxgap, best[PEAK], bestp, best[VALLEY], best[SPEAK], best[SVALLEY]);
    if (!stderr_tty || final) fputc('\n', stderr);
    fflush(stderr);
    pthread_mutex_unlock(&out_mu);
}

/*
 * default_chunk()
 *
 * Function:  choose a chunk size when -c is not given: range / (64 * threads)
 *            so that there are plenty of chunks for load balancing, clamped to
 *            [2^20, 2^34].  The upper clamp keeps the work lost on an interrupt
 *            (the chunks in flight) to a few seconds per thread and the
 *            frontier moving smoothly; the lower clamp keeps primesieve's
 *            per-chunk setup cost negligible.
 * Inputs:    range   numbers to scan; threads   worker count.
 * Returns:   the chunk size.
 * Errors:    none.
 */
static u64 default_chunk(u64 range, int threads)
{
    u64 c = range / ((u64)threads * 64);
    if (c < (1ull << 20)) c = 1ull << 20;
    if (c > (1ull << 34)) c = 1ull << 34;
    return c;
}

/*
 * scan_run()
 *
 * Function:  run the scan described by sc to completion (or until g_stop or
 *            the -n stop): set up the chunk grid and the shared state, start
 *            nthreads workers, then loop in the calling thread every 100 ms to
 *            print the status line (every 0.5 s on a terminal, every 60 s
 *            otherwise), write a checkpoint every ckpt_secs, and wake sleeping
 *            workers when g_stop is set.  When all workers have exited, join
 *            them, free their iterators, print the final status and write the
 *            final checkpoint.
 * Inputs:    sc          parameters filled in (origin, start, end, chunk or 0,
 *                        dmin, report, target, verbose, print_records) and the
 *                        folded totals preloaded from a checkpoint if resuming;
 *            nthreads    worker count (>= 1);
 *            statefile   checkpoint path or NULL;
 *            ckpt_secs   checkpoint interval.
 * Outputs:   sc->first/count/primes/maxgap/cf/t0/t1 etc.
 * Errors:    die() if END is within 2^33 of primesieve's maximum (the sieving
 *            overshoot past the last chunk must fit) or if a thread cannot be
 *            created.
 */
static void scan_run(scan_t *sc, int nthreads, const char *statefile, double ckpt_secs)
{
    if (sc->end > primesieve_get_max_stop() - (1ull << 33)) die("END too large for primesieve");
    if (nthreads < 1) nthreads = 1;
    u64 range = sc->end >= sc->start ? sc->end - sc->start + 1 : 0;
    if (!sc->chunk) sc->chunk = default_chunk(range, nthreads);
    sc->nchunks = range ? (range + sc->chunk - 1) / sc->chunk : 0;
    if (sc->dmin < 1) sc->dmin = 1;

    atomic_store(&sc->next, 0);
    atomic_store(&sc->finished, 0);
    pthread_mutex_init(&sc->mu, NULL);
    pthread_cond_init(&sc->cv, NULL);
    sc->cf = 0;
    sc->stop_chunk = UINT64_MAX;
    memset(sc->done, 0, sizeof sc->done);
    sc->chunks_done = 0;
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
        if (g_stop) { pthread_mutex_lock(&sc->mu); pthread_cond_broadcast(&sc->cv); pthread_mutex_unlock(&sc->mu); }
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
    }
    free(ws);
    sc->t1 = now();
    if (sc->verbose) print_status(sc, sc->t1, true);
    if (statefile) checkpoint(sc, statefile);
}

/*
 * scan_new()
 *
 * Function:  allocate a zeroed scan_t with the default dmin (4) and report
 *            depth (6).  The caller fills in the range and frees it with
 *            free().
 * Returns:   the scan.
 * Errors:    die() when out of memory.
 */
static scan_t *scan_new(void)
{
    scan_t *sc = xcalloc(1, sizeof *sc);
    sc->dmin = 4;
    sc->report = 6;
    return sc;
}

/* ------------------------------------------------------------------ */
/* Commands                                                            */
/* ------------------------------------------------------------------ */

/*
 * usage()
 *
 * Function:  print the command synopsis on stderr and exit with status 2.
 * Returns:   never.
 */
static void usage(void)
{
    fputs(
"usage: a248701 scan [START] END [-t T] [-c CHUNK] [-n N] [-d DMIN] [-r DEPTH] [-S STATE] [-i SECS] [-q]\n"
"       a248701 show P [-n N]\n"
"       a248701 selftest [-t T]\n"
"\n"
"  scan     exhaustive search of the primes in [START, END] (START defaults to 0) for the\n"
"           smallest prime of every depth in the four modes: A248701 (peak), A248702\n"
"           (valley), A248703 (strict peak), A248704 (strict valley).  Prints each new\n"
"           record of depth >= DEPTH (default 6) as it is found, then the tables.\n"
"           -n N stops as soon as A248701(N) is known.  -d DMIN (default 4, max 8) is the\n"
"           smallest depth examined above 2^30; all depths are examined below 2^30, which\n"
"           holds every known term.  -S FILE writes a checkpoint every SECS seconds\n"
"           (default 60) and on Ctrl-C; running the same command again resumes from it.\n"
"  show     the depths of the prime P in the four modes and the gaps around it\n"
"           (-n N: show N gaps on each side).\n"
"  numbers  may be written as 123, 2^40, 10^12, 1e12, 2^40-1\n", stderr);
    exit(2);
}

/*
 * on_signal()
 *
 * Function:  SIGINT/SIGTERM handler: request a stop.  Workers finish the
 *            chunk they are on, the frontier is folded as far as possible,
 *            the checkpoint and summary are written, and the process exits
 *            normally.  Installed with SA_RESETHAND, so a second signal kills
 *            the process immediately.
 * Inputs:    s   the signal number (ignored).
 * Errors:    none.
 */
static void on_signal(int s)
{
    (void)s;
    g_stop = 1;
}

/*
 * print_summary()
 *
 * Function:  print the end-of-scan report on stdout:
 *              - a "# scanned [...]" line with the time, rate, cumulative prime
 *                count and largest gap (cumulative from origin, i.e. including
 *                earlier runs of a resumed scan), and whether the run was
 *                interrupted or stopped by -n;
 *              - the table of centres of depth >= d per mode, from d = dmin to
 *                the largest depth seen;
 *              - for each mode the list of a(n): "matches OEIS" or "DIFFERS"
 *                for the depths in KNOWN[], "NEW" beyond them, and for the first
 *                depth with no record either "a(n) > <scanned bound>" (scan from
 *                0, so this is a proven lower bound) or a note that no such
 *                prime lies in the scanned interval; depths below dmin that
 *                were never examined (scan starting above 2^30) are marked.
 * Inputs:    sc   a finished scan.
 * Outputs:   text on stdout, flushed.
 * Errors:    none.
 */
static void print_summary(scan_t *sc)
{
    char b1[32];
    double secs = sc->t1 - sc->t0;
    u64 to = scanned_to(sc);
    bool complete = sc->cf >= sc->nchunks;
    printf("# scanned [%" PRIu64 ", %" PRIu64 "]%s in %.1f s (%s numbers/s%s): %" PRIu64 " primes and largest gap %" PRIu64
           " in [%" PRIu64 ", %" PRIu64 "]\n",
           sc->start, to, complete ? "" : g_stop ? " (interrupted)" : " (stopped early, -n)", secs,
           fmt_eng((double)(to + 1 - sc->start) / (secs > 0 ? secs : 1), b1),
           sc->elapsed0 > 0 ? ", this run" : "", sc->primes, sc->maxgap, sc->origin, to);
    if (sc->elapsed0 > 0) printf("# total time including earlier runs %.1f s\n", sc->elapsed0 + secs);

    int maxd = 0;
    for (int m = 0; m < NMODE; m++)
        for (int d = NMAX; d >= 1; d--) if (sc->count[m][d]) { if (d > maxd) maxd = d; break; }
    unsigned dlo = sc->dmin;
    printf("# centres of depth >= d in [%" PRIu64 ", %" PRIu64 "]%s:\n", sc->origin, to,
           dlo > 1 ? " (depths below -d are only examined in chunks starting below 2^30)" : "");
    printf("#   d            :");
    for (int d = (int)dlo; d <= maxd; d++) printf(" %12d", d);
    printf("\n");
    for (int m = 0; m < NMODE; m++) {
        printf("#   %-13s:", MODE[m]);
        for (int d = (int)dlo; d <= maxd; d++) printf(" %12" PRIu64, sc->count[m][d]);
        printf("\n");
    }

    for (int m = 0; m < NMODE; m++) {
        printf("%s (%s depth, a(n) = depth >= %s):", SEQ[m], MODE[m], SHIFT[m] ? "n+1" : "n");
        if (m == VALLEY) printf("\n  a(0) = 2   (by definition)");
        for (int d = 1; d <= NMAX; d++) {
            int n = d - SHIFT[m];
            if (n < FIRSTN[m]) continue;
            u64 a = sc->first[m][d];
            if (!a && (unsigned)d < sc->dmin && sc->origin >= SMALL) {
                printf("\n  a(%d): depth %d not examined (-d %u)", n, d, sc->dmin);
                continue;
            }
            if (a) {
                const char *tag = "NEW";
                if (d <= KDEPTH && KNOWN[m][d]) tag = a == KNOWN[m][d] ? "(matches OEIS)" : "** DIFFERS FROM OEIS **";
                printf("\n  a(%d) = %" PRIu64 "   %s", n, a, tag);
            } else {
                if (sc->origin == 0) printf("\n  a(%d) > %" PRIu64, n, to);
                else printf("\n  a(%d): no prime of depth >= %d in [%" PRIu64 ", %" PRIu64 "]", n, d, sc->origin, to);
                break;                          /* a(n) is non-decreasing: later n are unknown too */
            }
        }
        printf("\n");
    }
    fflush(stdout);
}

/*
 * cmd_scan()
 *
 * Function:  the `scan` command.  Parses the options, resumes from the
 *            checkpoint if -S names an existing file (START is then taken from
 *            the file and END may be raised), prints the header lines,
 *            installs the signal handlers, runs the scan and prints the
 *            summary.
 * Inputs:    argc/argv   the arguments after "scan": [START] END and the
 *            options -t threads, -c chunk, -n target depth, -d dmin (1..8),
 *            -r report depth, -S state file, -i checkpoint seconds, -q quiet.
 * Returns:   0 (also when a resumed scan was already complete).
 * Errors:    die() on bad arguments, END < START, a checkpoint written with a
 *            different -d, or END below the checkpoint position (unless END
 *            equals the checkpoint's own END, which just means "already done");
 *            plus whatever scan_run() reports.
 */
static int cmd_scan(int argc, char **argv)
{
    u64 pos[2] = { 0, 0 };
    int npos = 0, threads = 0, target = 0, report = 6, dmin = 4;
    u64 chunk = 0;
    const char *statefile = NULL;
    double ckpt = 60;

    for (int i = 0; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-t") && i + 1 < argc) threads = arg_int(argv[++i], "thread count", 1, 1024);
        else if (!strcmp(a, "-c") && i + 1 < argc) chunk = arg_u64(argv[++i], "chunk size");
        else if (!strcmp(a, "-n") && i + 1 < argc) target = arg_int(argv[++i], "target depth", 1, NMAX);
        else if (!strcmp(a, "-d") && i + 1 < argc) dmin = arg_int(argv[++i], "DMIN", 1, 8);
        else if (!strcmp(a, "-r") && i + 1 < argc) report = arg_int(argv[++i], "report depth", 1, NMAX);
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
    if (end < start) die("END is below START");
    scan_t *sc = scan_new();
    sc->dmin = (unsigned)dmin;
    sc->report = report;
    sc->target = target;
    bool resumed = false;
    state_t st;

    if (statefile && read_state(statefile, &st)) {
        if (st.dmin != (unsigned)dmin) die("%s was written with -d %u; use the same value", statefile, st.dmin);
        if (npos == 2 && pos[0] != st.lo)
            note("note: START %" PRIu64 " ignored, resuming from checkpoint at %" PRIu64, pos[0], st.lo);
        if (end < st.lo) {
            if (end == st.end) { note("%s: scan of [%" PRIu64 ", %" PRIu64 "] already complete", statefile, st.origin, st.end); return 0; }
            die("END %" PRIu64 " is below the checkpoint position %" PRIu64, end, st.lo);
        }
        start = st.lo;
        origin = st.origin;
        memcpy(sc->first, st.first, sizeof sc->first);
        memcpy(sc->count, st.count, sizeof sc->count);
        sc->primes = st.primes;
        sc->maxgap = st.maxgap;
        sc->elapsed0 = st.elapsed;
        resumed = true;
        note("resuming from %s at %" PRIu64 " (%" PRIu64 " primes done, %.0f s spent)", statefile, start, st.primes, st.elapsed);
    }

    if (!chunk) chunk = default_chunk(end - start + 1, threads);
    sc->origin = origin;
    sc->start = start;
    sc->end = end;
    sc->chunk = chunk;
    sc->verbose = !g_quiet;
    sc->print_records = true;

    printf("# A248701 / A248702 / A248703 / A248704: smallest prime with n monotone prime gaps on each side (peak, valley, strict peak, strict valley)\n");
    printf("# scanning primes in [%" PRIu64 ", %" PRIu64 "] with %d threads, chunk %" PRIu64 ", -d %d, reporting depth >= %d%s%s\n",
           start, end, threads, chunk, dmin, report, resumed ? " (resumed)" : "", target ? ", stopping at A248701 depth" : "");
    if (target) printf("# stop target: A248701(%d)\n", target);
    fflush(stdout);

    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_signal;
    sa.sa_flags = SA_RESETHAND;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    scan_run(sc, threads, statefile, ckpt);
    if (g_stop) note("interrupted; all chunks below %" PRIu64 " are complete%s", scanned_to(sc) + 1,
                     statefile ? ", checkpoint written" : "");
    print_summary(sc);
    free(sc);
    return 0;
}

/*
 * cmd_show()
 *
 * Function:  the `show` command: print the depth of P in each mode with the
 *            corresponding OEIS index, then the gap window with -n N gaps per
 *            side (default: the largest depth, so the brackets show the gaps
 *            that break the longest runs).
 * Inputs:    argc/argv   the arguments after "show": P and optionally -n N.
 * Returns:   0.
 * Errors:    die() if P is missing, malformed or not prime.
 */
static int cmd_show(int argc, char **argv)
{
    u64 P = 0;
    int nshow = 0;
    bool have = false;
    for (int i = 0; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-n") && i + 1 < argc) nshow = arg_int(argv[++i], "N", 1, NMAX);
        else if (a[0] == '-' && !isdigit((unsigned char)a[1])) die("unknown option '%s'", a);
        else { P = arg_u64(a, "prime"); have = true; }
    }
    if (!have) usage();
    int depth[NMODE];
    if (!depths_at(P, depth)) die("%" PRIu64 " is not prime", P);
    printf("%" PRIu64 ":", P);
    int maxd = 0;
    for (int m = 0; m < NMODE; m++) {
        printf("%s %s depth %d (%s index %d)", m ? "," : "", MODE[m], depth[m], SEQ[m], depth[m] - SHIFT[m]);
        if (depth[m] > maxd) maxd = depth[m];
    }
    printf("\n");
    int n = nshow ? nshow : (maxd > 0 ? maxd : 1);
    print_gap_window(P, n);
    printf("\n");
    return 0;
}

/* ------------------------------------------------------------------ */
/* Selftest                                                            */
/* ------------------------------------------------------------------ */

static int st_fail;                             /* number of failed expectations */

/*
 * expect()
 *
 * Function:  record one selftest expectation: print "ok" or "FAIL" with a
 *            description and count the failures.
 * Inputs:    ok   the verdict; printf-style description.
 * Errors:    none.
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
 * quiet_scan()
 *
 * Function:  run a scan with no output (no status line, no record lines) and
 *            return it for inspection; used by the selftest.
 * Inputs:    start, end, chunk, threads, dmin, target   as for a scan; chunk
 *            must be non-zero.
 * Returns:   the finished scan_t (caller frees it).
 * Errors:    as scan_run().
 */
static scan_t *quiet_scan(u64 start, u64 end, u64 chunk, int threads, unsigned dmin, int target)
{
    scan_t *sc = scan_new();
    sc->origin = start; sc->start = start; sc->end = end; sc->chunk = chunk;
    sc->dmin = dmin; sc->target = target;
    sc->verbose = false; sc->print_records = false; sc->report = NMAX + 1;
    scan_run(sc, threads, NULL, 0);
    return sc;
}

/*
 * same_results()
 *
 * Function:  compare two results: prime count, largest gap and, for every
 *            mode and every depth >= dlo, both the record and the count.
 * Inputs:    a, b   the results; dlo   first depth compared.
 * Outputs:   *why_first   true if the mismatch was in a record (as opposed to
 *            a count or a total); only meaningful when false is returned.
 * Returns:   true if identical.
 * Errors:    none.
 */
static bool same_results(const result_t *a, const result_t *b, unsigned dlo, bool *why_first)
{
    *why_first = false;
    if (a->nprimes != b->nprimes || a->maxgap != b->maxgap) return false;
    for (int m = 0; m < NMODE; m++)
        for (unsigned d = dlo; d <= NMAX; d++) {
            if (a->first[m][d] != b->first[m][d]) { *why_first = true; return false; }
            if (a->count[m][d] != b->count[m][d]) return false;
        }
    return true;
}

/*
 * totals_of()
 *
 * Function:  copy the folded totals of a scan into a result_t so that whole
 *            scans can be compared with same_results().
 * Inputs:    sc.   Outputs: *r.   Errors: none.
 */
static void totals_of(const scan_t *sc, result_t *r)
{
    memset(r, 0, sizeof *r);
    memcpy(r->first, sc->first, sizeof r->first);
    memcpy(r->count, sc->count, sizeof r->count);
    r->nprimes = sc->primes;
    r->maxgap = sc->maxgap;
}

/*
 * cmd_selftest()
 *
 * Function:  the `selftest` command (see THEORY OF OPERATION, 5).  Seven
 *            groups of checks:
 *              1. depths_at() on the OEIS examples and edge cases (2, 3,
 *                 composites);
 *              2. scan_chunk() against reference_scan() on four ranges at
 *                 different magnitudes, and a chunk against the sum of its
 *                 halves;
 *              3. a full scan below 4e8 against the 26 known terms and against
 *                 depth counts computed independently with numpy;
 *              4. the same range as one chunk, as 777777-sized chunks and as
 *                 2^20 chunks, plus a scan starting at 3e8 that must add up
 *                 with one ending there and must find A248703's 331542583;
 *              5. -d 1 against -d 4 above 2^30 (all depths >= 4 identical,
 *                 depth 3 not counted with -d 4);
 *              6. the -n early stop lands just after a(7);
 *              7. a checkpoint written and read back is identical.
 * Inputs:    argc/argv   optionally -t threads.
 * Returns:   0 if every expectation held, 1 otherwise.
 * Errors:    die() on an unknown option; a failing expectation is reported,
 *            not fatal.
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
    double t0 = now();
    printf("a248701 selftest (%d threads)\n", threads);

    /* 1. the OEIS example and the small terms through depths_at */
    {
        int d[NMODE];
        bool ok = depths_at(359, d) && d[PEAK] == 3 && d[SPEAK] == 2 && d[VALLEY] == 1 && d[SVALLEY] == 1;
        expect(ok, "359: peak depth 3, strict peak depth 2, valley depths 1 (gaps 10, 2, 4, 6 | 8, 6, 6, 4)");
        ok = depths_at(7, d) && d[PEAK] == 2 && d[SPEAK] == 1 && d[VALLEY] == 1;
        expect(ok, "7: peak depth 2 but strict peak depth 1 (gaps 2, 2 | 4, 2)");
        ok = depths_at(43, d) && d[VALLEY] == 3 && d[SVALLEY] == 2;
        expect(ok, "43: valley depth 3, strict valley depth 2 (gaps 6, 4, 2 | 4, 6, 6)");
        ok = depths_at(1429, d) && d[SVALLEY] == 3 && depths_at(1439, d) && d[SPEAK] == 3;
        expect(ok, "1429: strict valley depth 3; 1439: strict peak depth 3");
        ok = depths_at(2, d) && d[PEAK] == 0 && d[VALLEY] == 0 && depths_at(3, d) && d[PEAK] == 1 && d[VALLEY] == 1;
        expect(ok, "2 has depth 0 (no gap before it), 3 has depth 1");
        ok = !depths_at(360, d) && !depths_at(1, d);
        expect(ok, "360 and 1 are rejected as non-primes");
    }

    /* 2. the chunk scanner against the reference implementation */
    {
        static const u64 LOS[] = { 0, 1000000000ull, 1000000000000ull, 123456789012345ull };
        primesieve_iterator it;
        primesieve_init(&it);
        rings_t *R = xcalloc(1, sizeof *R);
        result_t ra, rb;
        bool wf;
        for (size_t i = 0; i < sizeof LOS / sizeof LOS[0]; i++) {
            u64 lo = LOS[i], hi = lo + 3000000;
            reference_scan(lo, hi, &ra);
            scan_chunk(&it, R, lo, hi, 1, &rb);
            bool ok = same_results(&ra, &rb, 1, &wf);
            int maxd = 0;
            for (int d = NMAX; d >= 1; d--) if (ra.count[PEAK][d]) { maxd = d; break; }
            expect(ok, "chunk [%" PRIu64 ", %" PRIu64 ") agrees with the reference implementation (%" PRIu64 " primes, max peak depth %d)%s",
                   lo, hi, ra.nprimes, maxd, ok ? "" : wf ? " [records differ]" : " [counts differ]");
        }
        /* a chunk split in two must give the same totals as one piece */
        result_t r1, r2;
        scan_chunk(&it, R, 5000000000ull, 5003000000ull, 1, &ra);
        scan_chunk(&it, R, 5000000000ull, 5001500000ull, 1, &r1);
        scan_chunk(&it, R, 5001500000ull, 5003000000ull, 1, &r2);
        bool ok = ra.nprimes == r1.nprimes + r2.nprimes && ra.maxgap == (r1.maxgap > r2.maxgap ? r1.maxgap : r2.maxgap);
        for (int m = 0; m < NMODE; m++)
            for (int d = 1; d <= NMAX; d++) {
                u64 f = r1.first[m][d] ? r1.first[m][d] : r2.first[m][d];
                if (ra.count[m][d] != r1.count[m][d] + r2.count[m][d] || ra.first[m][d] != f) ok = false;
            }
        expect(ok, "chunk [5e9, 5e9+3e6) equals the sum of its two halves");
        primesieve_free_iterator(&it);
        free(R);
    }

    /* 3. the four sequences below 4e8 and the depth counts (checked against an independent numpy computation) */
    scan_t *sc = quiet_scan(0, 399999999ull, 1000000, threads, 1, 0);
    {
        bool ok = true;
        for (int m = 0; m < NMODE; m++)
            for (int d = 1; d <= KDEPTH; d++) {
                if (!KNOWN[m][d]) continue;
                if (sc->first[m][d] != KNOWN[m][d]) {
                    ok = false;
                    printf("      %s depth %d: %" PRIu64 ", OEIS has %" PRIu64 "\n", SEQ[m], d, sc->first[m][d], KNOWN[m][d]);
                }
            }
        expect(ok, "scan [0, 4e8) reproduces all 26 known terms of A248701-A248704 (%" PRIu64 " primes, %.1f s)",
               sc->primes, sc->t1 - sc->t0);
        /* centres of depth >= d below 4e8, d = 1..7, from verify_a248701.py (numpy) */
        static const u64 EXP_PEAK[8]  = { 0, 21336325, 5601745, 670557, 47070, 2154, 74, 2 };
        static const u64 EXP_VALL[8]  = { 0, 21336325, 5597514, 669692, 46218, 2078, 69, 0 };
        static const u64 EXP_SPK[8]   = { 0, 21336325, 5016394, 481047, 24392, 715, 21, 1 };
        static const u64 EXP_SVL[8]   = { 0, 21336325, 5020624, 483720, 24355, 755, 16, 0 };
        ok = true;
        for (int d = 1; d <= 7; d++)
            if (sc->count[PEAK][d] != EXP_PEAK[d] || sc->count[VALLEY][d] != EXP_VALL[d] ||
                sc->count[SPEAK][d] != EXP_SPK[d] || sc->count[SVALLEY][d] != EXP_SVL[d]) ok = false;
        expect(ok, "depth counts below 4e8 match numpy: peak %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64,
               sc->count[PEAK][1], sc->count[PEAK][2], sc->count[PEAK][3], sc->count[PEAK][4], sc->count[PEAK][5], sc->count[PEAK][6], sc->count[PEAK][7]);
        expect(sc->count[PEAK][8] == 0 && sc->count[VALLEY][8] == 0, "no centre of depth >= 8 below 4e8");
    }

    /* 4. chunking independence and a scan that starts in the middle */
    {
        result_t ta, tb, tc, td, te;
        bool wf;
        scan_t *s1 = quiet_scan(0, 299999999ull, 300000000ull, 1, 1, 0);
        scan_t *s2 = quiet_scan(0, 299999999ull, 777777, threads, 1, 0);
        scan_t *s3 = quiet_scan(0, 299999999ull, 1 << 20, threads, 1, 0);
        totals_of(s1, &ta); totals_of(s2, &tb); totals_of(s3, &tc);
        bool ok = same_results(&ta, &tb, 1, &wf) && same_results(&ta, &tc, 1, &wf);
        expect(ok, "scan [0, 3e8) gives identical records and counts as one chunk, chunk 777777 and chunk 2^20");

        scan_t *s4 = quiet_scan(300000000ull, 399999999ull, 1 << 20, threads, 1, 0);
        totals_of(s4, &td);
        totals_of(sc, &te);
        ok = te.nprimes == ta.nprimes + td.nprimes;
        for (int m = 0; m < NMODE && ok; m++)
            for (int d = 1; d <= NMAX; d++) {
                if (te.count[m][d] != ta.count[m][d] + td.count[m][d]) ok = false;
                u64 f = ta.first[m][d] ? ta.first[m][d] : td.first[m][d];
                if (te.first[m][d] != f) ok = false;
            }
        expect(ok && td.first[SPEAK][7] == 331542583ull, "scan [3e8, 4e8) adds up with [0, 3e8) and finds A248703(6) = 331542583 above 3e8");
        free(s1); free(s2); free(s3); free(s4);
    }

    /* 5. -d 4 above 2^30 only drops depths below 4 */
    {
        result_t ta, tb;
        bool wf;
        scan_t *s1 = quiet_scan(1ull << 30, (1ull << 30) + 199999999ull, 1 << 22, threads, 1, 0);
        scan_t *s2 = quiet_scan(1ull << 30, (1ull << 30) + 199999999ull, 1 << 22, threads, 4, 0);
        totals_of(s1, &ta); totals_of(s2, &tb);
        bool ok = same_results(&ta, &tb, 4, &wf) && tb.count[PEAK][3] == 0 && ta.count[PEAK][3] > 0;
        expect(ok, "scan [2^30, 2^30+2e8) with -d 1 and -d 4 agree on every depth >= 4 (%" PRIu64 " centres of depth >= 4, %" PRIu64 " of depth >= 5)",
               ta.count[PEAK][4], ta.count[PEAK][5]);
        free(s1); free(s2);
    }

    /* 6. early stop */
    {
        scan_t *s1 = quiet_scan(0, 999999999ull, 1000000, threads, 1, 7);
        bool ok = s1->first[PEAK][7] == 294276293ull && s1->cf < s1->nchunks && scanned_to(s1) >= 294276293ull &&
                  scanned_to(s1) < 294276293ull + (u64)(threads + RING) * 1000000ull;
        expect(ok, "-n 7 on [0, 1e9) stops soon after a(7) = 294276293 (scanned to %" PRIu64 ")", scanned_to(s1));
        free(s1);
    }

    /* 7. checkpoint round trip */
    {
        state_t st, st2;
        memset(&st, 0, sizeof st);
        st.dmin = 4; st.origin = 0; st.end = 12345678901ull; st.lo = 4000000000ull; st.primes = 189961812; st.maxgap = 354; st.elapsed = 123.5;
        st.first[PEAK][7] = 294276293; st.first[SPEAK][7] = 331542583; st.count[PEAK][4] = 47070; st.count[SVALLEY][7] = 1;
        const char *path = "/tmp/a248701-selftest.state";
        write_state(path, &st);
        bool ok = read_state(path, &st2) && st2.dmin == 4 && st2.end == st.end && st2.lo == st.lo && st2.primes == st.primes &&
                  st2.maxgap == 354 && st2.elapsed == 123.5 && st2.first[PEAK][7] == 294276293 && st2.first[SPEAK][7] == 331542583 &&
                  st2.first[PEAK][6] == 0 && st2.count[PEAK][4] == 47070 && st2.count[SVALLEY][7] == 1 && st2.count[PEAK][5] == 0;
        unlink(path);
        expect(ok, "checkpoint file round trip");
    }

    free(sc);
    printf("%s (%.1f s)\n", st_fail ? "SELFTEST FAILED" : "all tests passed", now() - t0);
    return st_fail ? 1 : 0;
}

/* ------------------------------------------------------------------ */

/*
 * main()
 *
 * Function:  dispatch on the first argument: scan, show (alias depth),
 *            selftest.  stdout is line-buffered so that record lines appear
 *            promptly even when redirected to a file.
 * Returns:   the command's status; 2 with a usage message otherwise.
 */
int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    stderr_tty = isatty(2);
    if (argc < 2) usage();
    const char *cmd = argv[1];
    if (!strcmp(cmd, "scan")) return cmd_scan(argc - 2, argv + 2);
    if (!strcmp(cmd, "show") || !strcmp(cmd, "depth")) return cmd_show(argc - 2, argv + 2);
    if (!strcmp(cmd, "selftest")) return cmd_selftest(argc - 2, argv + 2);
    usage();
    return 2;
}
