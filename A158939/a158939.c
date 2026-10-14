/*
 * a158939.c -- compute and extend OEIS A158939
 *
 *   "First primes followed by sequences of exactly n monotonic increasing
 *    prime gaps."
 *
 * Known terms (offset 1, n = 1..17):
 *   3, 2, 17, 347, 2903, 15373, 128981, 1319407, 17797517, 94097537,
 *   6927837557, 48486712783, 968068681511, 1472840004017, 129001208165717,
 *   17293451238695141, 52461866207504471
 * a(16) and a(17) were found in September 2026 with the CUDA version of this
 * program (cuda/a158939_cuda.cu, same bookkeeping on a DGX Spark GPU) and
 * verified with the `verify` command below; a(18) > 5.46e16 (October 2026).
 *
 * ======================================================================
 * 1. Definition
 * ======================================================================
 * For a prime p let p = q_0 < q_1 < q_2 < ... be the consecutive primes from p
 * on and g_i = q_i - q_(i-1) the gaps that follow p.  The run length of p is
 *
 *   L(p) = largest n such that g_1 < g_2 < ... < g_n,
 *
 * i.e. g_(n+1) <= g_n is the gap that ends the run (this is the PARI program
 * by Charles Greathouse in the entry).  a(n) = smallest prime with L(p) = n.
 * Examples: L(2) = 2 (gaps 1, 2, 2), L(3) = 1 (2, 2), L(17) = 3 (2, 4, 6, 2),
 * so a(2) = 2, a(1) = 3, a(3) = 17.  a(14) = 1472840004017 has the run
 * 2 4 6 8 10 12 14 28 30 38 48 64 66 74 ended by 22.
 *
 * No prime has L(p) = 0 under this definition, and the entry's offset is 1
 * (a(1) = 3, a(2) = 2).  Related entries: A229832(n) is the prime after
 * a(n+1) (the first of n consecutive weak primes) and A133697(n) is the index
 * of a(n+2) among the primes.  a(n) is not monotone in principle, so the scan
 * keeps the first prime of every run length instead of stopping at a target.
 *
 * Two facts drive the whole design:
 *   (F1) L(p) depends only on the gaps AFTER p.  A worker that starts at an
 *        arbitrary point needs no history, so the range can be cut into
 *        chunks that are scanned independently and in any order.
 *   (F2) When a strictly increasing run of L gaps ends at the prime q_k (the
 *        gap after q_k is <= the gap before it), the primes q_(k-1), q_(k-2),
 *        ..., q_(k-L) have run lengths exactly 1, 2, ..., L.  Run lengths are
 *        therefore assigned in batches at run ends, and the set of run
 *        lengths seen so far in a chunk is always {1, ..., M}: a run needs
 *        attention only if it is longer than M, i.e. one compare per run.
 *
 * ======================================================================
 * 2. Theory of operation
 * ======================================================================
 * "First prime with ..." means every prime up to the answer has to be looked
 * at, so the cost is one pass of a prime sieve over the range and nothing
 * cleverer applies.  primesieve does the sieving (~3e9 numbers/s per core
 * near 10^15 on an M1 Pro) and is the whole cost: the bookkeeping per prime
 * is a compare, an increment and a store.
 *
 * Data flow:
 *
 *   main() -> cmd_scan() / cmd_bench() / cmd_selftest() -> run_scan()
 *                                                            |
 *       +----------------------------------------------------+
 *       |  starts T worker() threads, then monitors them: status line,
 *       |  rate and ETA, periodic checkpoint, CONFIRMED lines, -n N stop
 *       v
 *   worker()      takes chunk indices from the atomic counter S.next
 *       |         (never more than RINGC-1 chunks ahead of the frontier)
 *       v
 *   scan_chunk()  sieves one chunk with a primesieve iterator, finds the
 *       |         first prime of every run length in it, the exact
 *       |         run-length histogram and the prime count, then merges
 *       |         them into S under the mutex g_mu
 *       v
 *   S             global scan state: best[n] = smallest prime with run
 *                 length n seen so far, the frontier, histogram, prime count
 *
 * Chunks.  [start, end) is cut into chunks of `chunk` numbers (default 10^10;
 * END is rounded up to a chunk boundary).  Chunk i is [start + i*chunk,
 * start + (i+1)*chunk).  Workers take indices from S.next, so T threads
 * process T consecutive chunks at a time, each with its own primesieve
 * iterator.
 *
 * Per-chunk bookkeeping (scan_chunk).  The worker reads primes straight out
 * of the iterator's buffer and keeps
 *   run       number of gaps in the strictly increasing run in progress,
 *   prevgap   the last gap,
 *   ring[]    the last RINGP = 256 primes, enough to hold every prime of a
 *             run that is tracked (NMAX-1 = 63 gaps; longer runs set
 *             S.overflow and are reported),
 *   count     number of primes seen in [lo, prev],
 *   M         longest run that has ended so far in the chunk,
 *   first[n], first_idx[n], gaps[n][]
 *             the first prime with run length n, its 1-based index among
 *             the primes of the chunk, and its gaps,
 *   runs[L]   number of runs of length L that ended below hi,
 *   extra[n]  run lengths handed out by the final run end to primes < hi.
 * By (F2) a run end of length L > M yields first[M+1..L] from the ring in one
 * go.  The worker keeps going past hi until the run in progress ends (the
 * iterator is told to sieve LOOKAHEAD numbers beyond hi for that), so every
 * prime below hi has its run length settled by the worker that owns it; the
 * primes >= hi of that last run belong to the next chunk, which by (F1)
 * recomputes them on its own.  The histogram also follows from (F2):
 * hist[n] = (number of run ends of length >= n below hi) + extra[n], and
 * sum_n hist[n] = number of primes in [lo, hi), a built-in consistency check
 * (for scans [0, 10^k) the total is compared with pi(10^k)).
 *
 * Merging and the frontier.  Under g_mu the chunk's first[n] are compared
 * with S.best[n] (the smaller prime wins) and the chunk is marked done in the
 * ring S.slot[idx % RINGC].  S.frontier is the number of leading chunks that
 * are complete; complete chunks are folded into S.primes and S.hist in
 * frontier order, which keeps the prime count and the histogram exact at the
 * frontier and gives pi(p) for a candidate (primes before its chunk plus its
 * index inside the chunk).  A candidate a(n) found in chunk c stays a
 * candidate until the frontier passes c -- every chunk below c has to be
 * scanned before we know that no smaller prime has run length n -- and is
 * then printed as CONFIRMED.  Workers never run more than RINGC-1 chunks
 * ahead of the frontier, so a slot is never reused before it is folded.
 *
 * Checkpoint / resume (-S FILE).  The checkpoint records start, chunk, END,
 * the frontier, the prime count and histogram at the frontier, and every
 * candidate with its chunk, index and gaps.  Rerunning the same command
 * resumes at the frontier; chunks above it are scanned again, which is safe
 * because a candidate is only ever replaced by a smaller prime and its pi(p)
 * is recomputed when the frontier passes its chunk.  END may be changed on
 * resume (only start and chunk must match).  The file is written to FILE.tmp
 * and renamed, so an interrupted write leaves the previous checkpoint
 * intact.  SIGINT/SIGTERM set g_stop: workers finish the chunk in hand, the
 * state is saved, the table is printed and the program exits normally.
 *
 * Verification built in.  `verify` and the selftest recompute run lengths
 * with a deterministic Miller-Rabin next-prime search that shares no code
 * with the sieve (plus GMP's mpz_nextprime when compiled in).  The selftest
 * also compares the chunked, multithreaded scan with a brute-force pass over
 * an array of primes (odd chunk sizes, so boundaries fall mid-run), checks
 * a(1..13) and pi(a(n)) = A133697(n-2) by sieving to 10^12, and every scan
 * from 0 to 10^k compares its prime count with pi(10^k).
 *
 * Heuristic.  Treating gaps as i.i.d. exponential, P(L(p) = n) = n/(n+1)!,
 * so the first run of length n appears near pi(x) = (n+1)!/n.  The measured
 * densities fall below that model by a factor that shrinks by about 0.86 per
 * step in n (0.25 at n = 14 and 0.21 at n = 15 below 2e16; ties between even
 * gaps make strict increase rarer), which puts a(18) near 2e18.  print_hist
 * prints the model next to the measured counts.
 *
 * Limits.  Run lengths up to NMAX-1 = 63 are tracked (a longer run sets
 * S.overflow and is reported at the end); numbers must stay below
 * primesieve's limit (about 2^64 - 2*LOOKAHEAD); the Miller-Rabin test is
 * deterministic for every n < 2^64.  Exit status: 0 success, 1 failure
 * (die(), a failed selftest, a GMP disagreement), 2 usage error.
 *
 * ======================================================================
 * 3. Usage
 * ======================================================================
 *   a158939 scan [START] END [-t T] [-c CHUNK] [-s KIB] [-n N] [-r NMIN] [-S FILE] [-i SECS] [-q]
 *       Exhaustive scan of the primes in [START, END) (START defaults to 0;
 *       END is rounded up to a chunk boundary).  Prints each new smallest
 *       prime with run length >= NMIN (default 14) as it is found, a
 *       "CONFIRMED" line once everything below it has been examined, and a
 *       table of a(n) at the end.  -n N stops as soon as a(N) is confirmed.
 *       -S FILE keeps a checkpoint (every SECS seconds, default 60, and on
 *       Ctrl-C); rerunning the same command resumes from it.  -t T threads
 *       (default: all cores), -c CHUNK chunk size (default 10^10), -s KIB
 *       primesieve sieve size (default: library choice), -q no status line.
 *   a158939 verify P
 *       Print the gaps after the prime P and its run length L(P), computed
 *       with a Miller-Rabin next-prime search (independent of primesieve;
 *       also cross-checked with GMP when compiled in).
 *   a158939 bench [N] [-d SPAN] [-t T] [-c CHUNK] [-s KIB]
 *       Time the scan of [N, N+SPAN) (defaults 10^15 and 10^11) and project
 *       the running time of full scans.
 *   a158939 selftest [LIMIT] [-t T]
 *       Scan [0, LIMIT) (default 10^12) and compare with the known terms.
 *
 * Numbers may be written as decimal, 2^k, 10^k, 1e12, 1.29e14, or X+Y / X-Y
 * of those.
 *
 * Build:  cc -O2 -std=gnu11 -pthread -I/opt/homebrew/include a158939.c \
 *            -L/opt/homebrew/lib -lprimesieve -lm [-DHAVE_GMP -lgmp] -o a158939
 *         (or `make`; the Makefile adds -DHAVE_GMP -lgmp when gmp.h is found)
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
#include <limits.h>
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
typedef uint16_t u16;
typedef unsigned __int128 u128;

/* Tunables and fixed limits.  None of them is a correctness knob except as
 * noted; the defaults were chosen for the M1 Pro and 10^10 chunks. */
#define NMAX      64        /* run lengths 1..NMAX-1 are tracked; a run of NMAX
                               or more gaps sets S.overflow (never seen: the
                               longest run below 5.5e16 has 17 gaps) */
#define RINGP     256       /* recent primes kept by a worker; a power of two
                               larger than NMAX+1, so that every prime of a
                               tracked run is still in the ring when the run
                               ends */
#define PMASK     (RINGP - 1)   /* ring index mask */
#define RINGC     4096      /* chunk completion ring: workers never run more
                               than RINGC-1 chunks ahead of the frontier, so
                               a slot is never overwritten before it is folded */
#define LOOKAHEAD 1000000   /* numbers sieved past a chunk end so the run in
                               progress at the boundary can be finished; the
                               worker reads on past it if needed, this is
                               only primesieve's stop hint */
#define NKNOWN    17        /* terms of A158939 in the OEIS: KNOWN[1..NKNOWN] */
#define SCANNED_TO 54557804097699840ULL  /* every prime below this has been
                               examined (GPU scan, Oct 2026): no run of more
                               than NKNOWN gaps exists below it */

/* KNOWN[n] = a(n) as published (KNOWN[0] unused).  Used by `verify` and
 * print_table to label results and by the selftest as the expected values. */
static const u64 KNOWN[NKNOWN + 1] = {
    0, 3, 2, 17, 347, 2903, 15373, 128981, 1319407, 17797517, 94097537,
    6927837557ULL, 48486712783ULL, 968068681511ULL, 1472840004017ULL,
    129001208165717ULL, 17293451238695141ULL, 52461866207504471ULL
};

static bool stderr_tty;                   /* stderr is a terminal: live status line */
static volatile sig_atomic_t g_stop = 0;  /* set by SIGINT/SIGTERM or by -n N: stop after the current chunks */

/* ------------------------------------------------------------------ */
/* Generic helpers                                                     */
/* ------------------------------------------------------------------ */

/*
 * now() -- monotonic wall-clock time.
 *
 * Inputs:  none.
 * Returns: seconds since an arbitrary fixed point (CLOCK_MONOTONIC) with
 *          nanosecond resolution; unaffected by clock adjustments, so
 *          differences are safe for rates and ETAs.
 * Errors:  none (CLOCK_MONOTONIC is always available where this builds).
 */
static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

/*
 * die() -- print a fatal error and exit.
 *
 * Inputs:  printf-style format and arguments.
 * Effects: clears the live status line first when stderr is a terminal,
 *          prints the message plus a newline to stderr and calls exit(1).
 *          Never returns.  Worker threads may call it too; exit() then ends
 *          the whole process, and no checkpoint is written in that case.
 * Errors:  this is the error path for everything that cannot continue.
 */
static void die(const char *fmt, ...)
{
    va_list ap;
    if (stderr_tty) fputs("\r\033[K", stderr);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(1);
}

/*
 * fmt_dur() -- format a duration for the status line and summaries.
 *
 * Inputs:  s       duration in seconds;
 *          buf, n  output buffer and its size (32 bytes is plenty).
 * Returns: buf, holding "1d02h03m", "1h02m03s", "5m03s" or "4.2s" depending
 *          on the magnitude, or "?" when s is negative, NaN or above 1e9 s
 *          (used for an ETA that is not known yet).
 * Errors:  none; snprintf truncates to n bytes.
 */
static const char *fmt_dur(double s, char *buf, size_t n)
{
    if (s < 0 || s != s || s > 1e9) { snprintf(buf, n, "?"); return buf; }
    long t = (long)s;
    if (t >= 86400) snprintf(buf, n, "%ldd%02ldh%02ldm", t / 86400, (t / 3600) % 24, (t / 60) % 60);
    else if (t >= 3600) snprintf(buf, n, "%ldh%02ldm%02lds", t / 3600, (t / 60) % 60, t % 60);
    else if (t >= 60) snprintf(buf, n, "%ldm%02lds", t / 60, t % 60);
    else snprintf(buf, n, "%.1fs", s);
    return buf;
}

/*
 * mul_ok() -- 64-bit multiplication with overflow detection.
 *
 * Inputs:  a, b  factors;  r  where to store the product.
 * Returns: true and *r = a*b if the product fits in 64 bits; false, with *r
 *          untouched, if it does not.
 * Errors:  none.
 */
static bool mul_ok(u64 a, u64 b, u64 *r)
{
    u128 x = (u128)a * b;
    if (x >> 64) return false;
    *r = (u64)x;
    return true;
}

/*
 * all_digits() -- is s[0..n) a non-empty string of decimal digits?
 *
 * Inputs:  s  characters (need not be NUL-terminated);  n  their count.
 * Returns: true iff n > 0 and every character is '0'..'9'.
 * Errors:  none.
 */
static bool all_digits(const char *s, size_t n)
{
    if (n == 0) return false;
    for (size_t i = 0; i < n; i++)
        if (!isdigit((unsigned char)s[i])) return false;
    return true;
}

/*
 * parse_term() -- parse one unsigned number term into a u64.
 *
 * Accepted forms (no sign, no spaces):
 *   decimal      "129001208165717"   up to 20 digits, must fit in 64 bits;
 *   power        "2^54", "10^16"     base and exponent decimal;
 *   scientific   "1e12", "1.29e14"   mantissa of up to 19 digits with an
 *                                    optional decimal point; the exponent
 *                                    must be large enough to make the value
 *                                    an integer ("1.5e0" is rejected).
 * Inputs:  s, n  the term's characters (not necessarily NUL-terminated);
 *          out   where to store the value.
 * Returns: true and *out on success; false on a syntax error, a value that
 *          does not fit in 64 bits, or a term of 64 characters or more.
 *          *out is untouched on failure.
 * Errors:  none raised here; parse_num() turns a false return into die().
 */
static bool parse_term(const char *s, size_t n, u64 *out)
{
    char buf[64], *c;
    if (n == 0 || n >= sizeof buf) return false;
    memcpy(buf, s, n);
    buf[n] = 0;
    if ((c = strchr(buf, '^'))) {
        *c = 0;
        if (!all_digits(buf, strlen(buf)) || !all_digits(c + 1, strlen(c + 1))) return false;
        u64 b = strtoull(buf, 0, 10), e = strtoull(c + 1, 0, 10), r = 1;
        while (e--) if (!mul_ok(r, b, &r)) return false;
        *out = r;
        return true;
    }
    if ((c = strpbrk(buf, "eE"))) {
        *c = 0;
        if (!all_digits(c + 1, strlen(c + 1))) return false;
        long e = strtol(c + 1, 0, 10);
        char mant[64];
        int md = 0, frac = -1;
        for (const char *q = buf; *q; q++) {
            if (*q == '.') { if (frac >= 0) return false; frac = 0; continue; }
            if (!isdigit((unsigned char)*q)) return false;
            mant[md++] = *q;
            if (frac >= 0) frac++;
        }
        if (md == 0 || md > 19) return false;
        mant[md] = 0;
        if (frac > 0) e -= frac;
        if (e < 0) return false;
        u64 m = strtoull(mant, 0, 10);
        while (e-- > 0) if (!mul_ok(m, 10, &m)) return false;
        *out = m;
        return true;
    }
    if (!all_digits(buf, n) || n > 20) return false;
    errno = 0;
    u64 v = strtoull(buf, 0, 10);
    if (errno) return false;
    *out = v;
    return true;
}

/*
 * parse_num() -- parse a number argument, optionally a sum or difference.
 *
 * Inputs:  s  NUL-terminated: a term (see parse_term) or TERM+TERM or
 *          TERM-TERM, split at the LAST '+' or '-' ("2^54-32", "10^15+7").
 *          A sign in the first position is not a split point, so negative
 *          numbers and "-5" style values are rejected.
 * Returns: the value as a u64.
 * Errors:  die() on a syntax error, on overflow of a sum, or on a negative
 *          difference.
 */
static u64 parse_num(const char *s)
{
    size_t n = strlen(s);
    for (size_t i = n; i-- > 1; ) {           /* split at the last top-level + or - */
        if (s[i] != '+' && s[i] != '-') continue;
        u64 a, b;
        if (!parse_term(s, i, &a) || !parse_term(s + i + 1, n - i - 1, &b)) die("bad number '%s'", s);
        if (s[i] == '+') {
            if (a + b < a) die("overflow in '%s'", s);
            return a + b;
        }
        if (b > a) die("negative result in '%s'", s);
        return a - b;
    }
    u64 v;
    if (!parse_term(s, n, &v)) die("bad number '%s'", s);
    return v;
}

/*
 * parse_int() -- parse an integer option value and range-check it.
 *
 * Inputs:  s       the argument text (any parse_num form);
 *          lo, hi  inclusive range;
 *          what    name of the option for the error message.
 * Returns: the value as an int.
 * Errors:  die() with "<what> must be between lo and hi" when out of range,
 *          or any parse_num() error.
 */
static int parse_int(const char *s, int lo, int hi, const char *what)
{
    u64 v = parse_num(s);
    if (v < (u64)lo || v > (u64)hi) die("%s must be between %d and %d", what, lo, hi);
    return (int)v;
}

/* ------------------------------------------------------------------ */
/* Independent primality: deterministic Miller-Rabin below 2^64         */
/* ------------------------------------------------------------------ */

/*
 * mulmod() -- (a * b) mod m without overflow, through a 128-bit product.
 *
 * Inputs:  a, b  any u64;  m  modulus, must be > 0.
 * Returns: a*b mod m.
 * Errors:  m == 0 would trap on the division; no caller passes it.
 */
static u64 mulmod(u64 a, u64 b, u64 m) { return (u64)(((u128)a * b) % m); }

/*
 * powmod() -- a^e mod m by square-and-multiply.
 *
 * Inputs:  a  base (any u64);  e  exponent;  m  modulus > 0.
 * Returns: a^e mod m.  (The result starts at 1 and is only reduced by the
 *          multiplications, so e == 0 yields 1 even for m == 1; no caller
 *          does that.)
 * Errors:  m == 0 traps, as in mulmod().
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
 * is_prime64() -- deterministic primality test for every n < 2^64.
 *
 * Trial division by the primes up to 37, then strong-probable-prime tests
 * to the seven bases 2, 325, 9375, 28178, 450775, 9780504, 1795265022, a set
 * (Jim Sinclair, 2011) that has been verified to admit no strong pseudoprime
 * below 2^64.  A base that is 0 mod n is skipped, which is the correct
 * treatment for the n > 37 that reach that point.
 *
 * Inputs:  n  any u64.
 * Returns: true iff n is prime (false for n < 2).
 * Errors:  none.  Shares no code with primesieve, which is the point: this
 *          is the independent check of what the sieve reports.
 */
static bool is_prime64(u64 n)
{
    static const u64 small[] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 };
    static const u64 bases[] = { 2, 325, 9375, 28178, 450775, 9780504, 1795265022 };
    if (n < 2) return false;
    for (size_t i = 0; i < sizeof small / sizeof *small; i++)
        if (n % small[i] == 0) return n == small[i];
    u64 d = n - 1;
    int r = 0;
    while (!(d & 1)) { d >>= 1; r++; }
    for (size_t i = 0; i < sizeof bases / sizeof *bases; i++) {
        u64 a = bases[i] % n;
        if (a == 0) continue;
        u64 x = powmod(a, d, n);
        if (x == 1 || x == n - 1) continue;
        bool composite = true;
        for (int j = 1; j < r && composite; j++) {
            x = mulmod(x, x, n);
            if (x == n - 1) composite = false;
        }
        if (composite) return false;
    }
    return true;
}

/*
 * next_prime64() -- smallest prime greater than n.
 *
 * Inputs:  n  any u64 below 18446744073709551557 (the largest 64-bit prime).
 *          Above that the candidate wraps around and the result is
 *          meaningless; the scan never gets near it (primesieve's limit is
 *          lower) and `verify` inputs are nowhere close in practice.
 * Returns: the next prime, found by testing odd candidates with is_prime64;
 *          about log(n)/2 tests per call.
 * Errors:  none.
 */
static u64 next_prime64(u64 n)
{
    if (n < 2) return 2;
    u64 m = n + 1;
    if (!(m & 1)) m++;
    while (!is_prime64(m)) m += 2;
    return m;
}

/*
 * run_length64() -- L(p) straight from the definition, by next-prime search.
 *
 * Inputs:  p        a prime (not checked; for a composite p the gaps are
 *                   simply measured from p);
 *          gaps     array with room for maxgaps entries;
 *          maxgaps  its capacity.
 * Returns: L, the run length, with gaps[0..L-1] the L increasing gaps and
 *          gaps[L] the gap that ends the run (L+1 values in total); or -1 if
 *          the run has maxgaps or more gaps (gaps[] then holds the first
 *          maxgaps of them).  Callers pass maxgaps = NMAX+1.
 * Errors:  none besides the -1 signal.  Independent of primesieve: this is
 *          the cross-check used by `verify` and by the selftest.
 */
static int run_length64(u64 p, u16 *gaps, int maxgaps)
{
    u64 prev = p, prevgap = 0;
    int L = 0;
    for (;;) {
        u64 q = next_prime64(prev), g = q - prev;
        if (L >= maxgaps) return -1;
        gaps[L] = (u16)g;
        if (g <= prevgap) return L;
        L++;
        prevgap = g;
        prev = q;
    }
}

/* ------------------------------------------------------------------ */
/* Scan engine                                                         */
/* ------------------------------------------------------------------ */

/*
 * cand_t -- the smallest prime found so far with a given run length n
 * (stored in S.best[n]).  It is only a *candidate* for a(n) until the
 * frontier passes its chunk, i.e. until every chunk below it is complete;
 * after that it is final and gets its pi value.
 */
typedef struct {
    u64  p;                 /* smallest prime seen with this run length, 0 = none */
    u64  chunk;             /* chunk it was found in */
    u64  idx;               /* number of primes in [chunk start, p] (1-based index in the chunk) */
    u64  pi;                /* number of primes in [START, p]; 0 until the frontier passes chunk */
    u16  gaps[NMAX + 1];    /* the n gaps of the run, then the gap that ends it */
    bool announced;         /* CONFIRMED line already printed */
} cand_t;

/* A133697(n) = pi(A158939(n+2)) for n = 0..NA133697-1, the prime index of
 * a(n+2) as published; print_table and the selftest check the scan's own
 * pi(p) bookkeeping against it. */
#define NA133697 16
static const u64 A133697[NA133697] = {
    1, 7, 69, 420, 1796, 12073, 101397, 1139211, 5440508, 320620306ULL,
    2058187481ULL, 36451609409ULL, 54594153615ULL, 4100904808215ULL,
    475618519121221ULL, 1400080864310974ULL
};
_Static_assert(NA133697 >= NKNOWN - 1, "A133697 table must cover every known term");

/*
 * S -- the global state of one scan.  run_scan() fills it in before the
 * threads start and resets it for every call, so consecutive scans (the
 * selftest runs several) are independent.  While threads run, `next` and
 * `overflow` are atomics and everything after the "protected by g_mu" line
 * is touched only with g_mu held (scan_chunk's merge step, the monitor loop
 * in run_scan, save_state, announce_locked).  print_table/print_hist read it
 * after the threads have been joined.
 */
static struct {
    u64 start, end, chunk, nchunks; /* the range [start, end), end a chunk boundary; chunk size; number of chunks */
    _Atomic u64 next;               /* next chunk index to hand out (workers fetch-and-add it) */
    _Atomic bool overflow;          /* a run of NMAX or more gaps was seen: results for n = NMAX-1 are unreliable */
    /* protected by g_mu: */
    u64 frontier;                   /* chunks [0, frontier) are complete and folded into primes/hist */
    struct {                        /* completed chunks waiting for the frontier, by chunk % RINGC */
        bool done;                  /* slot holds a completed chunk not yet folded */
        u64 primes;                 /* its prime count */
        u64 hist[NMAX];             /* its run-length histogram */
    } slot[RINGC];
    u64 primes;                     /* primes in [start, frontier position) */
    u64 hist[NMAX];                 /* hist[n] = primes with run length n in [start, frontier position) */
    cand_t best[NMAX];              /* best[n] = smallest prime with run length n found so far, any chunk */
} S;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static int g_report = 14;           /* print "found"/CONFIRMED lines for run lengths >= this (-r NMIN) */
static int g_sieve_kib = 0;         /* primesieve sieve size in KiB, 0 = library default (-s KIB) */

/* Per-thread state: a thread number (only for debugging) and the thread's
 * primesieve iterator, which owns the sieve buffers and is reused for every
 * chunk the thread processes. */
typedef struct {
    int id;
    primesieve_iterator it;
} worker_t;

/*
 * print_cand() -- print one "found" or CONFIRMED line on stdout.
 *
 * Inputs:  n    run length;  c  the candidate;
 *          tag  "found    " or "CONFIRMED" (same width so the lines align).
 * Output:  "<tag> a(n) = p  gaps: g1 ... gn | g_(n+1)  (chunk c[, pi(p) = k])";
 *          pi(p) appears only once it is known (frontier past the chunk).
 *          Clears the status line first when stderr is a terminal.
 * Errors:  none.  Called with g_mu held; stdout is line-buffered (main), so
 *          the line is visible at once even through a pipe.
 */
static void print_cand(int n, const cand_t *c, const char *tag)
{
    if (stderr_tty) fputs("\r\033[K", stderr);
    printf("%s a(%d) = %" PRIu64 "  gaps:", tag, n, c->p);
    for (int i = 0; i < n; i++) printf(" %u", c->gaps[i]);
    printf(" | %u  (chunk %" PRIu64, c->gaps[n], c->chunk);
    if (c->pi) printf(", pi(p) = %" PRIu64, c->pi);
    printf(")\n");
    fflush(stdout);
}

/*
 * scan_chunk() -- sieve one chunk and settle the run length of every prime
 * in it.  This is the inner loop of the program; "Theory of operation" in
 * the header explains the two facts (F1) and (F2) it relies on.
 *
 * Inputs:  w    the worker (its primesieve iterator is reused);
 *          idx  chunk index; the chunk is [lo, hi) with
 *               lo = S.start + idx*S.chunk and hi = min(lo + S.chunk, S.end).
 * Effects: under g_mu, replaces S.best[n] with any smaller first prime of
 *          run length n found in the chunk (printing a "found" line when
 *          n >= g_report), records the chunk's prime count and run-length
 *          histogram in S.slot[idx % RINGC], and folds every complete chunk
 *          at the frontier into S.primes, S.hist and S.frontier, assigning
 *          pi(p) to the candidates whose chunk just became complete.
 * Returns: nothing.
 * Errors:  die() if primesieve reports an error or returns no primes (it is
 *          asked for [lo, hi + LOOKAHEAD), which always contains primes).
 *          A run of NMAX or more gaps sets S.overflow and is clamped to
 *          NMAX-1 gaps, so S.best[NMAX-1] and hist[NMAX-1] are then
 *          unreliable; run_scan() reports this at the end.
 *
 * Algorithm.  Primes are read straight out of the iterator's buffer.  For
 * each prime p with gap g = p - prev:
 *   - if g > prevgap, the increasing run grows (run++);
 *   - otherwise the run of `run` gaps has just ended at prev = q_k.  By (F2)
 *     the primes ring[ri-1-n], n = 1..run, have run length exactly n.  The
 *     run end is counted in runs[run] (or, when prev >= hi, in extra[n] for
 *     the primes that are still below hi), and if run > M the first primes
 *     of the new lengths M+1..run are recorded with their gaps and their
 *     chunk-local indices.  The next run then starts with g (run = 1).
 * The loop continues past hi until a run ends at prev >= hi: at that point
 * every prime below hi has been assigned a run length, and the primes >= hi
 * of that final run belong to the next chunk, which recomputes them by (F1).
 * `count` is the number of primes appended to the ring, i.e. in [lo, prev];
 * the primes >= hi are taken out again at `finish`, so that afterwards
 * count = number of primes in [lo, hi).
 */
static void scan_chunk(worker_t *w, u64 idx)
{
    const u64 lo = S.start + idx * S.chunk;
    u64 hi = lo + S.chunk;
    if (hi > S.end || hi < lo) hi = S.end;       /* last chunk may be short; guard the 64-bit wrap */

    u64 first[NMAX], first_idx[NMAX];   /* first[n] = first prime in the chunk with run length n (0 = none),
                                           first_idx[n] = its 1-based index among the chunk's primes */
    u16 gaps[NMAX][NMAX + 1];           /* gaps[n][0..n] = the n gaps after first[n] and the gap that ends the run */
    u64 ring[RINGP];                    /* the last RINGP primes, ring[(ri-1) & PMASK] = prev */
    u64 runs[NMAX];                 /* runs[L] = run ends of length L with all L primes < hi */
    u64 extra[NMAX];                /* run lengths from the final run end, per prime < hi */
    unsigned ri = 0;                /* ring write position (grows without bound, masked on use) */
    int M = 0, run = 0;             /* M = longest run ended so far; run = gaps in the run in progress */
    u64 prev, prevgap = 0, count = 0;   /* previous prime, previous gap, primes in [lo, prev] */

    memset(first, 0, sizeof first);
    memset(runs, 0, sizeof runs);
    memset(extra, 0, sizeof extra);
    /* Start sieving at lo; hi + LOOKAHEAD is only a stop hint that lets
     * primesieve size its buffers for the overrun past hi (the iterator keeps
     * producing primes beyond the hint if a run is still open). */
    primesieve_jump_to(&w->it, lo, hi + LOOKAHEAD);
    /* The primes are read straight out of the iterator's buffer, which
     * primesieve_generate_next_primes() fills with primes[0..size).  (Mixing in
     * primesieve_next_prime() for the first prime would discard the rest of
     * that first buffer -- an early version of this program lost the first
     * ~1000 primes of every chunk that way.) */
    primesieve_generate_next_primes(&w->it);
    if (w->it.is_error || w->it.size == 0) die("primesieve error in chunk %" PRIu64, idx);
    const u64 *P = w->it.primes;
    size_t np = w->it.size, i = 1;
    prev = P[0];                                    /* the first prime >= lo */
    if (prev >= hi) goto finish;                    /* no prime in this chunk */
    ring[ri++ & PMASK] = prev;
    count = 1;

    /* Main loop: the outer loop refills the buffer, the inner one walks it.
     * prevgap = 0 at the start makes the first gap "increasing", which is
     * right: the first prime's own run length depends only on later gaps. */

    for (;;) {
        for (; i < np; i++) {
            const u64 p = P[i];
            const u64 g = p - prev;
            if (g > prevgap) {
                run++;
            } else {
                /* The run of `run` gaps ended at prev = q_k (= ring[ri-1]);
                 * prime ring[ri-1-n] has run length n for n = 1..run (F2). */
                if (run >= NMAX) { atomic_store(&S.overflow, true); run = NMAX - 1; }
                if (prev < hi) {
                    runs[run]++;                    /* all `run` primes of this run are < hi */
                } else {                            /* final run end: some primes belong to the next chunk */
                    for (int n = 1; n <= run; n++)
                        if (ring[(ri - 1 - n) & PMASK] < hi) extra[n]++;
                }
                if (run > M) {
                    /* New record length in this chunk: the primes with run
                     * lengths M+1..run are seen for the first time, and since
                     * the chunk is scanned in increasing order they are the
                     * smallest such primes in [lo, hi).  Lengths <= M were
                     * settled by an earlier (hence smaller) run. */
                    for (int n = M + 1; n <= run; n++) {
                        const u64 q = ring[(ri - 1 - n) & PMASK];
                        if (q >= hi) continue;          /* belongs to the next chunk */
                        first[n] = q;
                        first_idx[n] = count - (u64)n;  /* count = primes in [lo, prev] */
                        for (int m = 0; m < n; m++)     /* the n gaps after q, from the ring */
                            gaps[n][m] = (u16)(ring[(ri - n + m) & PMASK] - ring[(ri - 1 - n + m) & PMASK]);
                        gaps[n][n] = (u16)g;            /* and the gap that ended the run */
                    }
                    M = run;                            /* may exceed the lengths actually recorded
                                                           (primes >= hi skipped); harmless, first[] is 0 there */
                }
                run = 1;                            /* g starts the next run */
                if (prev >= hi) goto finish;        /* every prime < hi is settled */
            }
            prevgap = g;
            prev = p;
            ring[ri++ & PMASK] = p;
            count++;
        }
        primesieve_generate_next_primes(&w->it);
        if (w->it.is_error || w->it.size == 0) die("primesieve error in chunk %" PRIu64, idx);
        P = w->it.primes;
        np = w->it.size;
        i = 0;
    }

finish:
    /* count includes the primes >= hi of the final run (and the one prime
     * >= hi read when the chunk is empty); walk back over the ring and
     * remove them, so count = pi(hi - 1) - pi(lo - 1). */
    for (unsigned j = 1; j <= ri && j < RINGP; j++) {   /* don't count primes >= hi */
        if (ring[(ri - j) & PMASK] >= hi) count--;
        else break;
    }

    /* Histogram of the chunk: a run end of length L assigns the lengths
     * 1..L to L distinct primes, so the number of primes with run length
     * exactly n is the number of run ends of length >= n (suffix sum of
     * runs[]) plus the primes < hi served by the final run end (extra[n]). */
    u64 hist[NMAX];
    {
        u64 acc = 0;
        for (int n = NMAX - 1; n >= 1; n--) {
            acc += runs[n];
            hist[n] = acc + extra[n];
        }
        hist[0] = 0;
    }

    /* Merge into the global state.  Chunks finish out of order, so a
     * candidate from this chunk only replaces the current one if it is
     * smaller; a candidate from a lower chunk that arrives later wins the
     * same way.  pi is unknown until the frontier reaches this chunk, and
     * announced = false lets announce_locked() print CONFIRMED then. */
    pthread_mutex_lock(&g_mu);
    for (int n = 1; n <= M && n < NMAX; n++) {
        if (!first[n]) continue;
        cand_t *c = &S.best[n];
        if (c->p == 0 || first[n] < c->p) {
            c->p = first[n];
            c->chunk = idx;
            c->idx = first_idx[n];
            c->pi = 0;
            c->announced = false;
            memcpy(c->gaps, gaps[n], (size_t)(n + 1) * sizeof(u16));
            if (n >= g_report) print_cand(n, c, "found    ");
        }
    }
    {                                               /* park this chunk's totals until the frontier reaches it */
        unsigned s = (unsigned)(idx % RINGC);
        S.slot[s].done = true;
        S.slot[s].primes = count;
        memcpy(S.slot[s].hist, hist, sizeof hist);
    }
    /* Advance the frontier over every consecutive complete chunk.  Folding
     * in order is what keeps S.primes exact, which is what makes
     * pi(p) = (primes before the chunk) + (index inside the chunk) exact. */
    while (S.slot[S.frontier % RINGC].done) {       /* fold complete chunks in frontier order */
        unsigned s = (unsigned)(S.frontier % RINGC);
        S.slot[s].done = false;
        for (int n = 1; n < NMAX; n++)                  /* S.primes = primes in [start, chunk start) */
            if (S.best[n].p && S.best[n].chunk == S.frontier) S.best[n].pi = S.primes + S.best[n].idx;
        S.primes += S.slot[s].primes;
        for (int n = 1; n < NMAX; n++) S.hist[n] += S.slot[s].hist[n];
        S.frontier++;
    }
    pthread_mutex_unlock(&g_mu);
}

/*
 * worker() -- thread body: process chunks until none are left or g_stop.
 *
 * Inputs:  arg  the worker_t for this thread.
 * Effects: initializes the thread's primesieve iterator (and the sieve size
 *          when -s was given), takes chunk indices from S.next and calls
 *          scan_chunk() on each.  Before scanning chunk idx it waits, in
 *          2 ms polls, until idx < frontier + RINGC - 1, so the completion
 *          ring S.slot[] can never be overwritten before it is folded; the
 *          frontier normally moves far faster than that, so this never
 *          stalls in practice.
 * Returns: NULL (joined by run_scan).
 * Errors:  none of its own; scan_chunk() dies on primesieve errors.  A stop
 *          request is honoured between chunks, so the chunk in progress is
 *          finished first (a few seconds for the default 10^10 numbers).
 */
static void *worker(void *arg)
{
    worker_t *w = arg;
    if (g_sieve_kib) primesieve_set_sieve_size(g_sieve_kib);
    primesieve_init(&w->it);
    while (!g_stop) {
        u64 idx = atomic_fetch_add(&S.next, 1);     /* claim the next chunk */
        if (idx >= S.nchunks) break;                /* nothing left */
        for (;;) {                                   /* stay within RINGC of the frontier */
            pthread_mutex_lock(&g_mu);
            u64 f = S.frontier;
            pthread_mutex_unlock(&g_mu);
            if (idx < f + RINGC - 1 || g_stop) break;
            usleep(2000);
        }
        if (g_stop) break;
        scan_chunk(w, idx);
    }
    primesieve_free_iterator(&w->it);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Checkpoint file                                                     */
/* ------------------------------------------------------------------ */

/*
 * save_state() -- write the checkpoint file atomically.
 *
 * Inputs:  path  checkpoint file name (-S FILE).
 * Effects: writes path.tmp, with g_mu held, in this text format:
 *            A158939 checkpoint 2
 *            start N / chunk N / end N / frontier N / primes N
 *            hist N COUNT                      (every non-zero entry)
 *            cand N P CHUNK IDX PI G1 ... G(N+1)  (every candidate)
 *            end
 *          and renames it over path, so a crash in mid-write leaves the
 *          previous checkpoint intact.  frontier, primes and hist describe
 *          the position start + frontier*chunk exactly; candidates above the
 *          frontier are provisional and their chunks are rescanned on
 *          resume.
 * Returns: nothing.
 * Errors:  I/O failures print a warning on stderr and leave the old file in
 *          place; they do not stop the scan.
 */
static void save_state(const char *path)
{
    char tmp[PATH_MAX];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "w");
    if (!f) { fprintf(stderr, "\nwarning: cannot write %s: %s\n", tmp, strerror(errno)); return; }
    pthread_mutex_lock(&g_mu);
    fprintf(f, "A158939 checkpoint 2\nstart %" PRIu64 "\nchunk %" PRIu64 "\nend %" PRIu64
               "\nfrontier %" PRIu64 "\nprimes %" PRIu64 "\n",
            S.start, S.chunk, S.end, S.frontier, S.primes);
    for (int n = 1; n < NMAX; n++)
        if (S.hist[n]) fprintf(f, "hist %d %" PRIu64 "\n", n, S.hist[n]);
    for (int n = 1; n < NMAX; n++) {
        if (!S.best[n].p) continue;
        fprintf(f, "cand %d %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64, n, S.best[n].p, S.best[n].chunk,
                S.best[n].idx, S.best[n].pi);
        for (int i = 0; i <= n; i++) fprintf(f, " %u", S.best[n].gaps[i]);
        fputc('\n', f);
    }
    pthread_mutex_unlock(&g_mu);
    fputs("end\n", f);
    if (fclose(f) != 0 || rename(tmp, path) != 0)
        fprintf(stderr, "\nwarning: cannot update %s: %s\n", path, strerror(errno));
}

/*
 * load_state() -- read a checkpoint written by save_state().
 *
 * Inputs:  path  checkpoint file name.  S.start, S.chunk and S.nchunks must
 *          already describe the current command line.
 * Effects: on success sets S.frontier (clamped to S.nchunks), S.primes,
 *          S.hist[] and S.best[] (with announced = false, so confirmed terms
 *          are printed again on resume).  The file's `end` is ignored: END
 *          may differ from the original run, only start and chunk must
 *          match, which is what lets a finished scan be extended.
 * Returns: true if a checkpoint was loaded; false if the file does not
 *          exist (a fresh scan).
 * Errors:  die() if the file exists but cannot be read, lacks the header,
 *          has a malformed hist or cand line, is truncated (no "end" line)
 *          or was written for a different start or chunk size.
 */
static bool load_state(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        if (errno == ENOENT) return false;
        die("cannot read %s: %s", path, strerror(errno));
    }
    char line[2048];
    if (!fgets(line, sizeof line, f) || strncmp(line, "A158939 checkpoint 2", 20) != 0)
        die("%s is not an A158939 checkpoint", path);
    u64 start = 0, chunk = 0, end = 0, frontier = 0, primes = 0;
    bool complete = false;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "start %" SCNu64, &start) == 1) continue;
        if (sscanf(line, "chunk %" SCNu64, &chunk) == 1) continue;
        if (sscanf(line, "end %" SCNu64, &end) == 1) continue;
        if (sscanf(line, "frontier %" SCNu64, &frontier) == 1) continue;
        if (sscanf(line, "primes %" SCNu64, &primes) == 1) continue;
        if (strncmp(line, "hist ", 5) == 0) {
            int n = 0;
            u64 c = 0;
            if (sscanf(line, "hist %d %" SCNu64, &n, &c) != 2 || n < 1 || n >= NMAX) die("%s: bad hist line", path);
            S.hist[n] = c;
            continue;
        }
        if (strncmp(line, "cand ", 5) == 0) {
            char *s = line + 5, *e;
            long n = strtol(s, &e, 10);
            if (n < 1 || n >= NMAX) die("%s: bad candidate line", path);
            cand_t *c = &S.best[n];
            c->p = strtoull(e, &e, 10);
            c->chunk = strtoull(e, &e, 10);
            c->idx = strtoull(e, &e, 10);
            c->pi = strtoull(e, &e, 10);
            for (int i = 0; i <= n; i++) c->gaps[i] = (u16)strtoul(e, &e, 10);
            c->announced = false;
            continue;
        }
        if (strncmp(line, "end", 3) == 0) { complete = true; break; }
    }
    fclose(f);
    if (!complete) die("%s is truncated", path);
    if (start != S.start || chunk != S.chunk)
        die("%s was written for start=%" PRIu64 " chunk=%" PRIu64 ", not start=%" PRIu64 " chunk=%" PRIu64,
            path, start, chunk, S.start, S.chunk);
    if (frontier > S.nchunks) frontier = S.nchunks;
    S.frontier = frontier;
    S.primes = primes;
    return true;
}

/*
 * on_sigint() -- SIGINT/SIGTERM handler: request an orderly stop.
 *
 * Inputs:  sig  the signal number (ignored).
 * Effects: sets g_stop.  Workers stop taking chunks, the monitor loop in
 *          run_scan() exits, the checkpoint is written and the table is
 *          printed.  Only a flag is set, so the handler is async-signal-safe.
 * Errors:  none.
 */
static void on_sigint(int sig)
{
    (void)sig;
    g_stop = 1;
}

/*
 * announce_locked() -- print CONFIRMED lines for terms that became final.
 *
 * Inputs:  none; must be called with g_mu held (hence the name).
 * Effects: for every candidate whose chunk is below the frontier and that
 *          has not been announced yet, marks it announced and, when
 *          n >= g_report, prints it with print_cand("CONFIRMED").
 * Returns: the largest n with a confirmed candidate (0 if none), used for
 *          the "confirmed to n=" field of the status line and by -n N.
 * Errors:  none.
 */
static int announce_locked(void)
{
    int maxc = 0;
    for (int n = 1; n < NMAX; n++) {
        cand_t *c = &S.best[n];
        if (!c->p || c->chunk >= S.frontier) continue;
        if (n > maxc) maxc = n;
        if (!c->announced) {
            c->announced = true;
            if (n >= g_report) print_cand(n, c, "CONFIRMED");
        }
    }
    return maxc;
}

/* What run_scan() reports back. */
typedef struct {
    double seconds;     /* wall time of this run */
    u64 primes;         /* primes in [start, frontier position), including work resumed from a checkpoint */
    u64 numbers;        /* numbers scanned by this run: completed chunks only */
} scan_stats_t;

/*
 * run_scan() -- scan [start, end) with `threads` threads: the engine behind
 * the scan, bench and selftest commands.
 *
 * Inputs:  start, end  the range; end is rounded UP to a chunk boundary
 *                      (with a note on stderr);
 *          chunk       numbers per chunk (> 0);
 *          threads     worker count (>= 1);
 *          state       checkpoint file name, or NULL for none;
 *          interval    seconds between checkpoints, and between progress
 *                      lines when stderr is not a terminal;
 *          stop_n      if > 0, stop as soon as a(stop_n) is confirmed;
 *          quiet       suppress the status line and the start/stop notes.
 * Effects: resets S (so consecutive calls are independent), loads the
 *          checkpoint if there is one, installs the SIGINT/SIGTERM handler
 *          (left installed), starts the workers and then loops every 200 ms:
 *          announces confirmed terms, checks the -n stop condition, keeps a
 *          sliding window of up to 128 one-second samples of the frontier
 *          position for the rate and ETA, prints the status line (every
 *          second on a terminal, every `interval` seconds otherwise) and
 *          writes the checkpoint every `interval` seconds.  After the
 *          workers are joined it announces the last terms, writes the final
 *          checkpoint and warns about a run-length overflow or an early
 *          stop.  Results stay in S for print_table()/print_hist().
 * Returns: scan_stats_t with the wall time, S.primes and the numbers
 *          scanned by this run.
 * Errors:  die() if end <= start, chunk == 0, the range does not fit in
 *          64 bits, END is beyond primesieve's limit, memory or thread
 *          creation fails, or the checkpoint is inconsistent (load_state).
 *          After a stop request the frontier may be below nchunks; the
 *          table then marks candidates above it as unconfirmed.
 */
static scan_stats_t run_scan(u64 start, u64 end, u64 chunk, int threads, const char *state,
                             int interval, int stop_n, bool quiet)
{
    if (end <= start) die("END must be larger than START");
    if (chunk == 0) die("CHUNK must be positive");
    S.start = start;
    S.chunk = chunk;
    S.nchunks = (end - start + chunk - 1) / chunk;
    if (S.nchunks == 0 || (S.nchunks - 1) > (UINT64_MAX - start) / chunk) die("range too large");
    S.end = start + S.nchunks * chunk;
    if (S.end < start) die("range too large");
    if (S.end > primesieve_get_max_stop() - 2 * LOOKAHEAD) die("END beyond primesieve's limit");
    if (S.end != end)
        fprintf(stderr, "note: END rounded up to the chunk boundary %" PRIu64 "\n", S.end);
    S.frontier = 0;
    S.primes = 0;
    memset(S.slot, 0, sizeof S.slot);
    memset(S.hist, 0, sizeof S.hist);
    memset(S.best, 0, sizeof S.best);
    atomic_store(&S.overflow, false);

    if (state && load_state(state))
        fprintf(stderr, "resuming from %s: %" PRIu64 " of %" PRIu64 " chunks done (position %" PRIu64
                        ", %" PRIu64 " primes examined)\n",
                state, S.frontier, S.nchunks, S.start + S.frontier * S.chunk, S.primes);
    const u64 frontier0 = S.frontier;
    atomic_store(&S.next, S.frontier);

    if (!quiet)
        fprintf(stderr, "scanning [%" PRIu64 ", %" PRIu64 ") in %" PRIu64 " chunks of %" PRIu64
                        " with %d threads%s\n", S.start, S.end, S.nchunks, S.chunk, threads,
                stop_n ? " (stopping when a(N) is confirmed)" : "");

    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_sigint;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    worker_t *ws = calloc((size_t)threads, sizeof *ws);
    pthread_t *th = calloc((size_t)threads, sizeof *th);
    if (!ws || !th) die("out of memory");
    for (int i = 0; i < threads; i++) {
        ws[i].id = i;
        if (pthread_create(&th[i], NULL, worker, &ws[i]) != 0) die("pthread_create failed");
    }

    /* Rate window: one sample of the frontier position per second, up to
     * NS of them, so the rate and ETA average over the last ~2 minutes
     * rather than over the whole run (the sieve slows as numbers grow). */
    enum { NS = 128 };
    double ts[NS];
    u64 ps[NS];
    int ns = 0;
    const double t0 = now();
    const u64 pos0 = S.start + frontier0 * S.chunk;
    double last_status = 0, last_ckpt = t0, last_sample = 0;
    ts[0] = t0; ps[0] = pos0; ns = 1;
    char b1[32], b2[32];

    for (;;) {
        usleep(200000);
        double t = now();
        pthread_mutex_lock(&g_mu);
        u64 f = S.frontier, primes = S.primes;
        int maxc = announce_locked();
        /* -n N: a(N) is final once its chunk is below the frontier */
        if (stop_n > 0 && S.best[stop_n].p && S.best[stop_n].chunk < f) g_stop = 1;
        pthread_mutex_unlock(&g_mu);
        u64 pos = S.start + f * S.chunk;            /* everything below pos is done */
        if (f >= S.nchunks || g_stop) break;

        if (t - last_sample >= 1.0) {
            last_sample = t;
            if (ns < NS) { ts[ns] = t; ps[ns] = pos; ns++; }
            else { memmove(ts, ts + 1, (NS - 1) * sizeof *ts); memmove(ps, ps + 1, (NS - 1) * sizeof *ps); ts[NS - 1] = t; ps[NS - 1] = pos; }
        }
        double rate = (t - ts[0] > 0.5) ? (double)(pos - ps[0]) / (t - ts[0]) : 0;
        double eta = rate > 0 ? (double)(S.end - pos) / rate : -1;
        bool show = stderr_tty ? (t - last_status >= 1.0) : (t - last_status >= interval);
        if (show && !quiet) {
            last_status = t;
            fprintf(stderr, "%s%s  pos %.6g (%.2f%%)  %.3g/s  %.3g primes  ETA %s  confirmed to n=%d",
                    stderr_tty ? "\r\033[K" : "progress: ",
                    fmt_dur(t - t0, b1, sizeof b1), (double)pos,
                    100.0 * (double)(pos - S.start) / (double)(S.end - S.start),
                    rate, (double)primes, fmt_dur(eta, b2, sizeof b2), maxc);
            if (stop_n > 0) {
                if (S.best[stop_n].p) fprintf(stderr, "  a(%d) candidate %" PRIu64, stop_n, S.best[stop_n].p);
                else fprintf(stderr, "  a(%d) > %.4g", stop_n, (double)pos);
            }
            if (!stderr_tty) fputc('\n', stderr);
            fflush(stderr);
        }
        if (state && t - last_ckpt >= interval) {
            last_ckpt = t;
            save_state(state);
        }
    }
    for (int i = 0; i < threads; i++) pthread_join(th[i], NULL);
    double t1 = now();
    if (stderr_tty && !quiet) fputs("\r\033[K", stderr);

    pthread_mutex_lock(&g_mu);
    announce_locked();
    pthread_mutex_unlock(&g_mu);
    if (state) save_state(state);
    if (atomic_load(&S.overflow))
        fprintf(stderr, "warning: a run longer than %d gaps occurred; a(n) for n >= %d is not tracked\n", NMAX - 1, NMAX);
    if (g_stop && S.frontier < S.nchunks && !quiet)
        fprintf(stderr, "stopped at position %" PRIu64 " (%" PRIu64 " of %" PRIu64 " chunks complete)%s\n",
                S.start + S.frontier * S.chunk, S.frontier, S.nchunks,
                state ? ", checkpoint saved" : "");

    scan_stats_t st;
    st.seconds = t1 - t0;
    st.primes = S.primes;
    st.numbers = (S.frontier - frontier0) * S.chunk;
    free(ws);
    free(th);
    return st;
}

/* pi(10^k) for k = 6..16 (standard values), for checking the prime count
 * of a scan [0, 10^k): the scan's S.primes must equal PI10[k] exactly. */
static const u64 PI10[17] = {
    0, 0, 0, 0, 0, 0, 78498ULL, 664579ULL, 5761455ULL, 50847534ULL, 455052511ULL,
    4118054813ULL, 37607912018ULL, 346065536839ULL, 3204941750802ULL,
    29844570422669ULL, 279238341033925ULL
};

/*
 * print_hist() -- print the prime count and the run-length histogram of the
 * scanned range [S.start, frontier position).
 *
 * Inputs:  none (reads S after the workers have finished).
 * Output:  the prime count, compared with pi(10^k) when the scan started at
 *          0 and ended exactly at 10^k (k = 6..16); then one line per run
 *          length n with the count, the i.i.d.-exponential-gap model
 *          primes * n/(n+1)! and the ratio of the two.  Stops at the first
 *          n with no primes and a model expectation below 0.01.
 * Errors:  none.
 */
static void print_hist(void)
{
    const u64 pos = S.start + S.frontier * S.chunk;
    printf("\n%" PRIu64 " primes in [%" PRIu64 ", %" PRIu64 ")", S.primes, S.start, pos);
    if (S.start == 0)
        for (int k = 6; k <= 16; k++)
            if (pos == (u64)pow(10, k)) printf(S.primes == PI10[k] ? "  = pi(10^%d)" : "  BUT pi(10^%d) = %" PRIu64, k, PI10[k]);
    printf("\nrun-length distribution (model: gaps i.i.d. exponential, P(L = n) = n/(n+1)!):\n");
    printf("  n  primes with L(p) = n     model        ratio\n");
    double fact = 1;                                 /* becomes (n+1)! */
    for (int n = 1; n < NMAX; n++) {
        fact *= (n + 1);
        double model = (double)S.primes * n / fact;
        if (S.hist[n] == 0 && model < 0.01) break;
        printf(" %2d  %-24" PRIu64 " %-12.4g %.3f\n", n, S.hist[n], model, model > 0 ? (double)S.hist[n] / model : 0);
    }
    fflush(stdout);
}

/*
 * print_table() -- print the a(n) table after a scan, then the histogram.
 *
 * Inputs:  none (reads S after the workers have finished).
 * Output:  one line per n up to the largest run length found: a(n) (the
 *          smallest prime >= START with run length n when START > 0),
 *          pi(a(n)) if known, the gaps and the gap that ends the run, the
 *          comparison with the OEIS (KNOWN[] and A133697[]) when the scan
 *          started at 0, "NEW" for terms beyond the published ones, and an
 *          "[unconfirmed]" marker when earlier chunks are incomplete (after
 *          an interrupted run).  Then the lower bound for the next term and
 *          print_hist().
 * Errors:  none.
 */
static void print_table(void)
{
    const u64 pos = S.start + S.frontier * S.chunk;
    int maxn = 0;
    for (int n = 1; n < NMAX; n++) if (S.best[n].p) maxn = n;
    printf("\n %2s  %-24s %-21s %s\n", "n", S.start ? "a(n) (>= START)" : "a(n)",
           S.start ? "pi(a(n)) (from START)" : "pi(a(n))", "gaps | gap that ends the run");
    for (int n = 1; n <= maxn; n++) {
        const cand_t *c = &S.best[n];
        if (!c->p) { printf(" %2d  (none found)\n", n); continue; }
        bool confirmed = c->chunk < S.frontier;
        printf(" %2d  %-24" PRIu64, n, c->p);
        if (c->pi) printf(" %-21" PRIu64, c->pi); else printf(" %-21s", "?");
        for (int i = 0; i < n; i++) printf(" %u", c->gaps[i]);
        printf(" | %u", c->gaps[n]);
        if (S.start == 0 && n <= NKNOWN)
            printf(c->p == KNOWN[n] ? "  = OEIS" : "  DIFFERS from OEIS %" PRIu64, KNOWN[n]);
        else if (S.start == 0)
            printf("  NEW");
        if (S.start == 0 && c->pi && n >= 2 && n - 2 < NA133697)
            printf(c->pi == A133697[n - 2] ? ", pi = A133697(%d)" : ", pi DIFFERS from A133697(%d) = %" PRIu64,
                   n - 2, A133697[n - 2]);
        else if (S.start == 0 && c->pi && n >= 2)
            printf(", pi = new A133697(%d)", n - 2);
        if (!confirmed) printf("  [unconfirmed: earlier chunks incomplete]");
        putchar('\n');
    }
    if (S.frontier > 0)
        printf(" %2d  > %" PRIu64 "  (no%s prime below this position has run length %d)\n",
               maxn + 1, pos, S.start ? " such" : "", maxn + 1);
    print_hist();
}

/*
 * reference_first() -- brute-force reference for the selftest.
 *
 * Inputs:  limit  consider the primes below limit;
 *          first  array of NMAX entries: first[n] = first prime < limit with
 *                 run length n (0 if none);
 *          hist   array of NMAX entries: hist[n] = number of such primes.
 * Effects: generates every prime below limit + 10^6 into memory (the margin
 *          holds thousands of primes, far more than any run, so every run
 *          that starts below limit ends inside the array) and applies the
 *          definition of L(p) literally to each prime: walk the following
 *          gaps while they increase.  O(primes * run length), fine for the
 *          2*10^8 the selftest uses, not for real ranges.
 * Returns: nothing (results in first[] and hist[]).
 * Errors:  die() if primesieve fails or a run is not settled inside the
 *          array.
 */
static void reference_first(u64 limit, u64 *first, u64 *hist)
{
    size_t n;
    u64 *pr = primesieve_generate_primes(0, limit + 1000000, &n, UINT64_PRIMES);
    if (!pr) die("primesieve_generate_primes failed");
    memset(first, 0, NMAX * sizeof *first);
    memset(hist, 0, NMAX * sizeof *hist);
    for (size_t i = 0; i < n && pr[i] < limit; i++) {
        int L = 0;
        u64 prevgap = 0;
        size_t j;
        for (j = i + 1; j < n; j++) {
            u64 g = pr[j] - pr[j - 1];
            if (g <= prevgap) break;
            L++;
            prevgap = g;
        }
        if (j == n) die("reference_first: run at %" PRIu64 " not settled", pr[i]);
        if (L < NMAX && first[L] == 0) first[L] = pr[i];
        if (L < NMAX) hist[L]++;
    }
    primesieve_free(pr);
}

/* ------------------------------------------------------------------ */
/* Commands                                                            */
/* ------------------------------------------------------------------ */

/*
 * default_threads() -- number of online CPUs, clamped to [1, 1024].
 *
 * Inputs:  none.
 * Returns: the thread count used when -t is not given.
 * Errors:  none (a failed sysconf() yields 1).
 */
static int default_threads(void)
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n < 1 ? 1 : (n > 1024 ? 1024 : (int)n);
}

/*
 * usage() -- print the command summary to stderr and exit with status 2.
 * Never returns.
 */
static void usage(void)
{
    fprintf(stderr,
        "usage: a158939 scan [START] END [-t T] [-c CHUNK] [-s KIB] [-n N] [-r NMIN] [-S FILE] [-i SECS] [-q]\n"
        "       a158939 verify P\n"
        "       a158939 bench [N] [-d SPAN] [-t T] [-c CHUNK] [-s KIB]\n"
        "       a158939 selftest [LIMIT] [-t T]\n");
    exit(2);
}

/*
 * cmd_scan() -- the `scan` command: exhaustive scan of [START, END).
 *
 * Inputs:  argc/argv after the command word: one or two positional numbers
 *          (END, or START END) and the options -t T, -c CHUNK, -s KIB, -n N,
 *          -r NMIN, -S FILE, -i SECS, -q (see usage()).  Option values may
 *          use any parse_num form.
 * Effects: sets the globals g_report and g_sieve_kib, runs the scan, prints
 *          the table and histogram on stdout and a one-line timing summary
 *          on stderr.
 * Returns: 0 (errors leave through die()/usage()).
 * Errors:  die() if CHUNK < 10^6 (smaller chunks make the per-chunk
 *          overhead dominate) or on any run_scan() error; usage() on
 *          malformed arguments.
 */
static int cmd_scan(int argc, char **argv)
{
    u64 pos[2];
    int npos = 0, threads = default_threads(), interval = 60, stop_n = 0;
    u64 chunk = 10000000000ULL;
    const char *state = NULL;
    bool quiet = false;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "-t") && i + 1 < argc) threads = parse_int(argv[++i], 1, 1024, "threads");
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) chunk = parse_num(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) g_sieve_kib = parse_int(argv[++i], 16, 65536, "sieve KiB");
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) stop_n = parse_int(argv[++i], 1, NMAX - 1, "N");
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) g_report = parse_int(argv[++i], 1, 1000, "NMIN");
        else if (!strcmp(argv[i], "-S") && i + 1 < argc) state = argv[++i];
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) interval = parse_int(argv[++i], 1, 86400, "SECS");
        else if (!strcmp(argv[i], "-q")) quiet = true;
        else if (argv[i][0] == '-' && !isdigit((unsigned char)argv[i][1])) usage();
        else if (npos < 2) pos[npos++] = parse_num(argv[i]);
        else usage();
    }
    if (npos == 0) usage();
    u64 start = npos == 2 ? pos[0] : 0, end = pos[npos - 1];
    if (chunk < 1000000) die("CHUNK must be at least 10^6");
    scan_stats_t st = run_scan(start, end, chunk, threads, state, interval, stop_n, quiet);
    print_table();
    char b[32];
    fprintf(stderr, "%" PRIu64 " numbers, %" PRIu64 " primes in %s (%.3g numbers/s)\n",
            st.numbers, st.primes, fmt_dur(st.seconds, b, sizeof b),
            st.seconds > 0 ? (double)st.numbers / st.seconds : 0);
    return 0;
}

/*
 * cmd_verify() -- the `verify` command: L(P) for one prime, independently
 * of the sieve.
 *
 * Inputs:  argv[0] = P (any parse_num form).
 * Output:  "P: gaps g1 ... gL | g_(L+1)  ->  L(p) = L", labelled with the
 *          OEIS index when P is a known term (or MISMATCH if the run length
 *          differs from the published index); then the primes of the run;
 *          then, when built with GMP, whether mpz_nextprime reproduces the
 *          same primes.
 * Returns: 0, or 1 if GMP disagrees with the Miller-Rabin search.
 * Errors:  die() if P is not prime or the run exceeds NMAX gaps; usage()
 *          without an argument.
 */
static int cmd_verify(int argc, char **argv)
{
    if (argc < 1) usage();
    u64 p = parse_num(argv[0]);
    if (!is_prime64(p)) die("%" PRIu64 " is not prime", p);
    u16 gaps[NMAX + 1];
    int L = run_length64(p, gaps, NMAX + 1);
    if (L < 0) die("run longer than %d gaps", NMAX);
    printf("%" PRIu64 ": gaps", p);
    for (int i = 0; i < L; i++) printf(" %u", gaps[i]);
    printf(" | %u  ->  L(p) = %d", gaps[L], L);
    for (int n = 1; n <= NKNOWN; n++)
        if (KNOWN[n] == p) printf(L == n ? "  (= OEIS a(%d))" : "  (OEIS has this as a(%d): MISMATCH)", n);
    putchar('\n');
    printf("primes: %" PRIu64, p);
    u64 q = p;
    for (int i = 0; i <= L; i++) { q += gaps[i]; printf(" %" PRIu64, q); }
    putchar('\n');
#ifdef HAVE_GMP
    {
        mpz_t z;
        mpz_init_set_ui(z, p);
        bool ok = true;
        u64 r = p;
        for (int i = 0; i <= L && ok; i++) {
            mpz_nextprime(z, z);
            r += gaps[i];
            if (mpz_cmp_ui(z, r) != 0) ok = false;
        }
        mpz_clear(z);
        printf("GMP mpz_nextprime %s\n", ok ? "agrees" : "DISAGREES");
        if (!ok) return 1;
    }
#endif
    return 0;
}

/*
 * cmd_bench() -- the `bench` command: throughput at a chosen height.
 *
 * Inputs:  optional N (default 10^15) and -d SPAN (default 10^11), -t T,
 *          -c CHUNK, -s KIB.
 * Effects: scans [N, N+SPAN) quietly and without a checkpoint, then prints
 *          the rate in numbers/s and primes/s, the projected wall time of
 *          full scans [0, X) for X = 10^15..10^16 at that rate (the sieve
 *          is faster below N and slower above it, so this is a guide only),
 *          and the run-length histogram of the window, whose ratios to the
 *          model show how rare long runs are at that height.
 * Returns: 0.
 * Errors:  die() if CHUNK < 10^6 or on run_scan() errors; usage() on
 *          malformed arguments.
 */
static int cmd_bench(int argc, char **argv)
{
    u64 N = 1000000000000000ULL, span = 100000000000ULL, chunk = 10000000000ULL;
    int threads = default_threads(), npos = 0;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "-t") && i + 1 < argc) threads = parse_int(argv[++i], 1, 1024, "threads");
        else if (!strcmp(argv[i], "-d") && i + 1 < argc) span = parse_num(argv[++i]);
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) chunk = parse_num(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) g_sieve_kib = parse_int(argv[++i], 16, 65536, "sieve KiB");
        else if (argv[i][0] == '-' && !isdigit((unsigned char)argv[i][1])) usage();
        else if (npos++ == 0) N = parse_num(argv[i]);
        else usage();
    }
    if (chunk < 1000000) die("CHUNK must be at least 10^6");
    scan_stats_t st = run_scan(N, N + span, chunk, threads, NULL, 60, 0, true);
    double rate = (double)st.numbers / st.seconds;
    printf("bench: [%" PRIu64 ", %" PRIu64 ") with %d threads, chunk %" PRIu64 ": %.2f s\n",
           N, N + span, threads, chunk, st.seconds);
    printf("       %.4g numbers/s, %.4g primes/s (%" PRIu64 " primes)\n",
           rate, (double)st.primes / st.seconds, st.primes);
    printf("       time for a scan [0, X) at this rate (the sieve is faster below %.3g and slower above):\n", (double)N);
    static const double X[] = { 1e15, 2e15, 5e15, 1e16 };
    char b[32];
    for (size_t i = 0; i < sizeof X / sizeof *X; i++)
        printf("         X = %.0e: %s\n", X[i], fmt_dur(X[i] / rate, b, sizeof b));
    print_hist();
    return 0;
}

/*
 * cmd_selftest() -- the `selftest` command: check the program against
 * itself and against the OEIS.
 *
 * Inputs:  optional LIMIT (default 10^12) and -t T.
 * Checks (each prints "ok" or "FAIL" with a description):
 *   1. number parsing (parse_num);
 *   2. Miller-Rabin spot checks: small primes and composites, a strong
 *      pseudoprime to base 2, the largest 64-bit prime, a composite next to
 *      a(15);
 *   3. the run length of every known term by next-prime search;
 *   4. the chunked, multithreaded scan of [0, 2*10^8) with three chunk sizes
 *      (two of them odd, so chunk boundaries fall in the middle of runs)
 *      against reference_first(): the same first prime for every n and the
 *      same histogram, which must sum to pi(2*10^8) = 11078937;
 *   5. a full scan of [0, LIMIT): a(n) and pi(a(n)) = A133697(n-2) for every
 *      known term below LIMIT, no known term "found" below its value, and no
 *      run length beyond the known terms while LIMIT is within the
 *      exhaustively scanned bound SCANNED_TO (above it such a find is
 *      reported as NEW instead);
 *   6. the sieve's gaps for every a(n) found agree with the next-prime
 *      search.
 * Returns: 0 if everything passed, 1 otherwise ("SELFTEST FAILED (k)").
 * Errors:  die()/usage() as for the other commands.  g_report is raised to
 *          NMAX during the scans so that no "found" lines interleave with
 *          the check output, and restored afterwards.
 */
static int cmd_selftest(int argc, char **argv)
{
    u64 limit = 1000000000000ULL;
    int threads = default_threads(), npos = 0, fails = 0;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "-t") && i + 1 < argc) threads = parse_int(argv[++i], 1, 1024, "threads");
        else if (argv[i][0] == '-' && !isdigit((unsigned char)argv[i][1])) usage();
        else if (npos++ == 0) limit = parse_num(argv[i]);
        else usage();
    }
/* CHECK(cond, fmt, ...): print "ok    <description>" or "FAIL  <description>"
 * and count the failures. */
#define CHECK(cond, ...) do { if (cond) printf("ok    " __VA_ARGS__); else { printf("FAIL  " __VA_ARGS__); fails++; } putchar('\n'); } while (0)

    CHECK(parse_num("2^54-32") == 18014398509481952ULL && parse_num("1e12") == 1000000000000ULL &&
          parse_num("1.29e14") == 129000000000000ULL && parse_num("10^15+7") == 1000000000000007ULL,
          "number parsing");
    CHECK(is_prime64(2) && is_prime64(3) && !is_prime64(1) && !is_prime64(9) && is_prime64(1000000007ULL) &&
          !is_prime64(3215031751ULL) && is_prime64(18446744073709551557ULL) && !is_prime64(18446744073709551555ULL) &&
          is_prime64(129001208165717ULL) && !is_prime64(129001208165719ULL - 2 + 4),
          "Miller-Rabin spot checks");
    for (int n = 1; n <= NKNOWN; n++) {
        u16 gaps[NMAX + 1];
        int L = run_length64(KNOWN[n], gaps, NMAX + 1);
        CHECK(L == n, "run length of a(%d) = %" PRIu64 " by next-prime search is %d", n, KNOWN[n], L);
    }

    /* chunk boundaries: odd chunk sizes against a brute-force reference */
    int save_report = g_report;
    g_report = NMAX;
    {
        const u64 lim = 200000000ULL;
        u64 ref[NMAX], refhist[NMAX];
        static const u64 chunks[] = { 1234567ULL, 7000001ULL, 200000000ULL };
        for (size_t ci = 0; ci < sizeof chunks / sizeof *chunks; ci++) {
            run_scan(0, lim, chunks[ci], threads, NULL, 60, 0, true);
            reference_first(S.end, ref, refhist);       /* S.end = lim rounded up to a chunk boundary */
            bool same = true, samehist = true;
            int nref = 0;
            u64 tot = 0;
            for (int n = 1; n < NMAX; n++) {
                if (ref[n]) nref = n;
                if (S.best[n].p != ref[n]) same = false;
                if (S.hist[n] != refhist[n]) samehist = false;
                tot += S.hist[n];
            }
            CHECK(same, "chunked scan (chunk %" PRIu64 ", %d threads) matches brute force on [0, %" PRIu64 "), n = 1..%d",
                  chunks[ci], threads, S.end, nref);
            CHECK(samehist && tot == S.primes && (S.end != 200000000ULL || S.primes == 11078937ULL),
                  "run-length histogram matches brute force and sums to pi(%" PRIu64 ") = %" PRIu64, S.end, S.primes);
        }
    }

    printf("sieve scan of [0, %" PRIu64 ") with %d threads ...\n", limit, threads);
    fflush(stdout);
    u64 chunk = limit >= 100000000000ULL ? 10000000000ULL : (limit >= 10000000000ULL ? 1000000000ULL : 100000000ULL);
    scan_stats_t st = run_scan(0, limit, chunk, threads, NULL, 60, 0, true);
    g_report = save_report;
    char b[32];
    printf("      %" PRIu64 " primes in %s (%.3g numbers/s)\n", st.primes, fmt_dur(st.seconds, b, sizeof b),
           st.seconds > 0 ? (double)st.numbers / st.seconds : 0);
    for (int n = 1; n <= NKNOWN; n++) {
        if (KNOWN[n] < S.end) {
            CHECK(S.best[n].p == KNOWN[n], "a(%d) = %" PRIu64 " (sieve found %" PRIu64 ")", n, KNOWN[n], S.best[n].p);
            if (n >= 2)
                CHECK(S.best[n].pi == A133697[n - 2], "pi(a(%d)) = %" PRIu64 " = A133697(%d)", n, S.best[n].pi, n - 2);
        }
        else
            CHECK(S.best[n].p == 0, "a(%d) = %" PRIu64 " is above the limit and was not found below it", n, KNOWN[n]);
    }
    /* A run longer than every published term below the exhaustively scanned
     * bound would contradict the GPU scan, so it is a failure there; above
     * the bound it would be a discovery. */
    for (int n = NKNOWN + 1; n < NMAX; n++) {
        if (!S.best[n].p) continue;
        if (S.end <= SCANNED_TO)
            CHECK(false, "unexpected run length %d at %" PRIu64 " (none exist below %.3g)", n, S.best[n].p, (double)SCANNED_TO);
        else
            printf("NEW   a(%d) = %" PRIu64 " (not in OEIS; verify with 'a158939 verify')\n", n, S.best[n].p);
    }
    /* cross-check every sieve result with the independent next-prime search */
    {
        bool all = true;
        for (int n = 1; n < NMAX; n++) {
            if (!S.best[n].p) continue;
            u16 gaps[NMAX + 1];
            int L = run_length64(S.best[n].p, gaps, NMAX + 1);
            if (L != n || memcmp(gaps, S.best[n].gaps, (size_t)(n + 1) * sizeof(u16)) != 0) {
                all = false;
                printf("      sieve and next-prime search disagree at %" PRIu64 "\n", S.best[n].p);
            }
        }
        CHECK(all, "sieve gaps agree with next-prime search for every a(n) found");
    }
#undef CHECK
    printf(fails ? "SELFTEST FAILED (%d)\n" : "selftest passed\n", fails);
    return fails ? 1 : 0;
}

/*
 * main() -- command dispatch.
 *
 * Inputs:  argv[1] = scan | verify | bench | selftest, then that command's
 *          arguments.
 * Effects: records whether stderr is a terminal (live status line versus
 *          periodic progress lines) and makes stdout line-buffered so that
 *          result lines appear immediately even when piped through tee.
 * Returns: the command's exit status (0 ok, 1 failure); usage() exits 2.
 */
int main(int argc, char **argv)
{
    stderr_tty = isatty(2);
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 2) usage();
    if (!strcmp(argv[1], "scan")) return cmd_scan(argc - 2, argv + 2);
    if (!strcmp(argv[1], "verify")) return cmd_verify(argc - 2, argv + 2);
    if (!strcmp(argv[1], "bench")) return cmd_bench(argc - 2, argv + 2);
    if (!strcmp(argv[1], "selftest")) return cmd_selftest(argc - 2, argv + 2);
    usage();
    return 2;
}
