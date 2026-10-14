/*
 * wieferich.c  --  multi-base search for Wieferich primes to base b (CPU version)
 *
 *   "Primes p such that p^2 divides b^(p-1) - 1."
 *
 * One OEIS sequence per base: A001220 (b = 2), A014127 (3, Mirimanoff primes),
 * A123692 (5), A212583 (6), A123693 (7), A045616 (10), A111027 (12), A128667 (13),
 * A234810 (14), A242741 (15), A128668 (17), A244260 (18), A090968 (19), A242982 (20),
 * A298951 (22), A128669 (23), A306255 (26), A306256 (30); and A039951, the smallest
 * such p for each base n (a(47), a(72), ... are unknown).
 *
 * Status (October 2026)
 *   A306256(4) = 303632117562967 (base 30) was found on Oct 01 2026 by the CUDA port of
 *   this program (cuda/wieferich_cuda.cu), confirmed by an exhaustive base-30 scan of
 *   every prime below it, and is in the OEIS since Oct 02 2026.  Every prime in
 *   [2e14, 3.0848e14) has been tested for bases 6, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22,
 *   23, 26, 30, 47, 72 (search paused, resumable).  README.md has the record,
 *   submission.md the possible OEIS updates, verify_wieferich.py an independent Python
 *   check of everything this program computes.
 *
 * ==========================================================================
 * THEORY OF OPERATION
 * ==========================================================================
 *
 * 1. The problem
 *
 *    For a prime p that does not divide b, Fermat's little theorem gives
 *    b^(p-1) = 1 (mod p), so
 *
 *        b^(p-1) = 1 + q*p  (mod p^2)   with the FERMAT QUOTIENT q = q_p(b) in [0, p).
 *
 *    p is a Wieferich prime to base b (a "solution") iff q = 0, i.e. iff
 *    b^(p-1) mod p^2 = 1.  Two cases are special:
 *      * p | b: then b^(p-1) = 0 (mod p), never a solution; such pairs are skipped.
 *      * p = 2: 2^2 | b - 1 iff b = 1 (mod 4); decided directly, no arithmetic needed.
 *    Writing q as the symmetric residue A in (-p/2, p/2], p is a "near-Wieferich"
 *    prime when |A| <= T for a small T (option -n T); these are reported but are not
 *    solutions.
 *
 * 2. Where new terms can be found (heuristic)
 *
 *    If q behaves like a random residue mod p, a prime is a solution with
 *    probability 1/p, so a complete search of [x, y] finds on average
 *    sum 1/p ~ ln(ln y / ln x) solutions per base.  Base 2 is searched to 2^64
 *    (PrimeGrid, Dec 2022): even [2^64, 2^65] gives only 0.015.  Bases up to 150 were
 *    searched to 2e14 and bases 3, 5, 7 to 1.2e15 (R. Fischer, fermatquotient.com),
 *    where every test is ~10^5 times more likely to hit; that is why this program
 *    handles many bases at once.  Fischer's per-decade counts for bases <= 1052 agree
 *    with the heuristic to about 2 sigma.
 *
 * 3. What a scan proves (coverage)
 *
 *    "scan START END" tests every prime p in [START, END) for every selected base.
 *    The primes come from primesieve; the program never skips a prime of the range
 *    (the chunk log's prime counts can be checked against primecount: they equal
 *    pi(END) - pi(START)).  The range is cut into chunks; the FRONTIER is the number
 *    of leading chunks that are complete.  Hence
 *      * every solution in [START, START + frontier*CHUNK) has been found and printed
 *        (FOUND when computed, CONFIRMED once the frontier has passed its chunk);
 *      * a complete scan with no solution for base b proves there is none in
 *        [START, END); with an earlier search below START this extends the known bound.
 *    Solutions found in a chunk beyond the frontier (other threads still working
 *    below it) are printed as FOUND but are not yet claimed to be the next term.
 *
 * 4. Per-prime setup (pctx_init)
 *
 *    All arithmetic is mod N = p^2 for p < 2^63, so N < 2^126 fits in two 64-bit
 *    limbs with two bits to spare (the bounds below rely on that).  Once per prime,
 *    shared by every base:
 *      * pinv = p^-1 mod 2^64 by Newton iteration x <- x(2 - p x), started from
 *        x = 3p XOR 2, which is correct to 5 bits for any odd p (Montgomery's trick);
 *        four steps give 5 -> 10 -> 20 -> 40 -> 80 >= 64 bits;
 *      * ninv = -N^-1 mod 2^64 = -(pinv^2) mod 2^64, the Montgomery constant;
 *      * one = 2^128 mod N, the Montgomery form of 1 (R = 2^128);
 *      * invN = 1/N as a double, used for quotient estimates (section 6).
 *
 * 5. Montgomery arithmetic (msqr, redc)
 *
 *    A residue x is held as xR mod N (R = 2^128).  REDC(T) = T R^-1 mod N for
 *    T < N R: add multiples of N that clear T's low 64-bit words, one word at a time
 *    (q_i = t_i * ninv mod 2^64, T += q_i N 2^(64 i)), then shift right by 128 bits.
 *    The result is < 2N and one conditional subtraction makes it < N.  msqr(a)
 *    = REDC(a^2) uses a dedicated two-limb squaring: a0^2, 2 a0 a1, a1^2 (3 of the
 *    4 products of a general multiplication), then the two REDC steps (4 products),
 *    7 64x64->128 products in all.  With a < N < 2^126 no intermediate sum exceeds
 *    its 64- or 128-bit container (comments in msqr give each bound).
 *
 * 6. Multiplying by a small constant without a Montgomery step (msmall)
 *
 *    In the exponentiation the multiplier is always a plain integer c = b^d < 2^32.
 *    Since (xR) c = (xc) R, multiplying a Montgomery representative by c gives the
 *    representative of the product, so only x*c mod N is needed.  t = x*c has at most
 *    158 bits; its quotient Q = floor(t / N) < 2^32 is estimated as
 *    floor(double(t) * invN).  The relative error of that product is a few units of
 *    2^-53, so the absolute error is < 2^32 * 2^-50 < 1 and the estimate is Q - 1, Q or
 *    Q + 1.  r = t - estimate*N (computed mod 2^128) therefore lies in [-N, 2N), which
 *    fits in a signed 128-bit value since N < 2^126; one add or subtract of N finishes.
 *    This costs 4 products instead of the 8 of a Montgomery multiplication.
 *
 * 7. The exponentiation (run_group)
 *
 *    e = p - 1 is processed from the top in fixed windows of w bits:
 *        x = one * b^(top digit);  then per window:  w squarings, x = x * b^d.
 *    A digit d = 0 multiplies by b^0 = 1 (no branch).  w is chosen per base as the
 *    largest w <= 5 with b^(2^w - 1) < 2^32, so every multiplier fits msmall:
 *    w = 5 for b = 2, 4 for b = 3..4, 3 for b <= 23, 2 for b <= 1625, 1 up to 2^32 - 1.
 *    The powers b^0 .. b^(2^w - 1) are precomputed once per base (bases_setup).  At
 *    the end REDC leaves Montgomery form: r = b^(p-1) mod N in [0, N).
 *
 * 8. Lanes (instruction-level parallelism)
 *
 *    One exponentiation is a chain of ~50 dependent squarings, so a single chain is
 *    bound by multiply latency.  run_group runs LANES (default 8) independent
 *    exponentiations in lock step: every lane does the same w squarings and one msmall
 *    per window, which lets the out-of-order core overlap the chains (M1 Pro: 1.3e7
 *    tests/s with 1 lane, 3.1e7 with 8).  All lanes of a group share w; their
 *    exponents may differ (different primes) and the shorter ones simply get leading
 *    zero digits, which keep x = one.  Lanes are (prime, base) pairs taken from a
 *    batch of 64 primes times the bases of one window class (process_batch); the last
 *    group of a batch is padded with copies of its last lane, whose results are
 *    discarded.
 *
 * 9. Result checks (finish_lanes)
 *
 *    Because p divides r - 1, the quotient is an exact division and can be computed
 *    with a multiplication: q = (r - 1) * pinv mod 2^64 (only the low 64 bits of r
 *    matter).  If r were wrong (a bug or a hardware fault), q would be an essentially
 *    random 64-bit number, so the test "q < p" re-checks Fermat's little theorem and
 *    catches a wrong residue with probability 1 - p/2^64 (99.99% at 1e15).  A failure
 *    is reported as FLT-ERROR, counted, and the pair is left out of the checksum.
 *    q = 0 (equivalently r = 1) is a solution.
 *
 * 10. Checksums and the chunk log
 *
 *    Every tested pair adds mix64(q XOR p*0x9E3779B97F4A7C15) to its base's checksum,
 *    mod 2^64 (mix64 = the splitmix64 finalizer).  Addition does not depend on order,
 *    so the totals are the same for any chunk size, thread count or device.  Pairs
 *    with p | b and the prime 2 are not tested and do not enter the checksum; they do
 *    count as primes.  With -L FILE one line per completed chunk is appended in
 *    frontier order:
 *        C <lo> <hi> <primes> <b>:<checksum> <b>:<checksum> ...
 *    The CUDA tool and verify_wieferich.py write identical lines for the same chunk,
 *    which is how GPU runs are double-checked (dcheck.sh).
 *
 * 11. Parallel scan, frontier and checkpoints (run_scan, worker, scan_chunk)
 *
 *    Chunk i is [START + i*CHUNK, min(START + (i+1)*CHUNK, END)); END is rounded up to
 *    a chunk boundary.  Workers take chunk indices from an atomic counter and sieve
 *    their chunk with their own primesieve iterator.  A finished chunk's results go
 *    into a ring of RINGC slots; under g_mu, complete slots are folded into the totals
 *    in chunk order, advancing the frontier and writing the chunk log.  A worker never
 *    starts a chunk more than RINGC - 1 ahead of the frontier, so ring slots are never
 *    reused too early.  The main thread prints the status line, announces CONFIRMED
 *    solutions and writes the checkpoint (-S FILE) every SECS seconds and at the end.
 *    The checkpoint is a small text file holding the base list, START, CHUNK, END, the
 *    frontier, the totals, the checksums and the solutions below the frontier; it is
 *    written to FILE.tmp and renamed, so a crash leaves the previous checkpoint
 *    intact.  Rerunning the same command resumes from the frontier (the base list,
 *    START and CHUNK must match, END may change).  SIGINT/SIGTERM set g_stop: workers
 *    abandon their chunk within one batch of 64 primes (it is redone on resume), the
 *    checkpoint is saved and the summary printed.
 *
 * 12. Validation (selftest) and independent checks
 *
 *    selftest: (a) the Montgomery routines against a shift-and-add reference on
 *    200,000 random operands plus edge cases, and ~3,000 full Fermat quotients for
 *    random primes < 2^63 and 64 bases (all window sizes); (b) a scan of [0, 10^9) for
 *    all 23 non-power bases 2..30 must give pi(10^9) and exactly the 65 known solutions;
 *    (c) windows of +-10^6 around the 12 known solutions above 10^9 must re-find each
 *    one; (d) three checksum windows (at 1e15, 1e18 and 2^63) must match
 *    wieferich_cks.h, which verify_wieferich.py generates with Python integers and its
 *    own Miller-Rabin prime list.  check P recomputes single quotients with the
 *    reference arithmetic.
 *
 * 13. Limits
 *
 *    p < 2^63 (END <= 2^63 after rounding), 2 <= b < 2^32, at most 64 bases per run,
 *    at most 4096 recorded solutions per run (more are printed but not kept),
 *    CHUNK >= 1, 1..1024 threads.  Numbers on the command line must fit in 64 bits.
 *
 * Usage
 * -----
 *   wieferich scan [START] END [-b BASES] [-t T] [-c CHUNK] [-n T] [-S FILE] [-i SECS] [-L FILE] [-q]
 *       Test every prime in [START, END) (END <= 2^63, rounded up to a chunk
 *       boundary) for every base.  Prints FOUND / NEAR / FLT-ERROR lines as
 *       they occur and a summary with the checksums at the end.  -c CHUNK
 *       (default 10^9), -n T reports near-Wieferich primes with |A| <= T,
 *       -S FILE keeps a checkpoint (every SECS seconds, default 60, and on
 *       Ctrl-C; rerunning the same command resumes, END may change), -L FILE
 *       chunk log, -t T threads (default: all cores), -q no status line.
 *       Exit status 3 if any FLT error occurred.
 *   wieferich check P [-b BASES]
 *       Fermat quotients q_P(b) and A for one odd P, from the fast kernel and
 *       from an independent shift-and-add reference (default bases: "known").
 *   wieferich bench [N] [-d SPAN] [-b BASES] [-t T]
 *       Time [N, N+SPAN) (defaults 10^15, 10^10) and project longer scans.
 *   wieferich selftest [LIMIT] [-t T]
 *       The checks of section 12; LIMIT (default 10^9) is the end of part (b).
 *
 * BASES is a comma-separated list of bases and ranges (3,5,7 or 2-30); the keyword
 * "oeis" is the 17 bases with OEIS sequences other than 2 (the default), "known" is
 * every base 2..30 that is not a perfect power.  Duplicates are ignored.  Numbers may
 * be written as decimal, B^E, MeE with an optional decimal point (1e12, 1.2e15), or
 * X+Y / X-Y of those.
 *
 * Build:  make   (or: cc -O2 -std=gnu11 -pthread -I/opt/homebrew/include wieferich.c \
 *                     -L/opt/homebrew/lib -lprimesieve -lm -o wieferich)
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

typedef uint64_t u64;
typedef uint32_t u32;
typedef unsigned __int128 u128;

#ifndef LANES
#define LANES 8             /* exponentiations run in lock step per worker (section 8) */
#endif
#define BATCH   64          /* primes set up together; lanes are formed from BATCH x bases */
#define MAXB    64          /* bases per run */
#define MAXW    5           /* largest window (b = 2) */
#define RINGC   4096        /* chunk completion ring: bounds how far workers run ahead */
#define MAXHITS 4096        /* solutions kept per run (summary, checkpoint) */
#define GOLD    0x9E3779B97F4A7C15ULL   /* 2^64 / golden ratio, mixes p into the checksum */
#define PMAX    (1ULL << 63) /* p < 2^63 keeps N = p^2 < 2^126 */

static bool stderr_tty;                     /* stderr is a terminal: redraw the status line */
static volatile sig_atomic_t g_stop = 0;    /* set by SIGINT/SIGTERM; never reset */

/* ------------------------------------------------------------------ */
/* Generic helpers                                                     */
/* ------------------------------------------------------------------ */

/*
 * now: monotonic wall clock.
 *   In:     nothing.
 *   Out:    seconds since an arbitrary fixed point (CLOCK_MONOTONIC), as a double.
 *   Errors: none (clock_gettime on CLOCK_MONOTONIC does not fail on supported systems).
 */
static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

/*
 * die: print an error message and exit.
 *   In:     printf-style format and arguments.
 *   Out:    does not return; clears the status line on a terminal, prints the message
 *           and a newline on stderr, exits with status 1.
 *   Errors: none.  Running threads are not joined (exit ends them); a checkpoint is not
 *           written, so the last periodic checkpoint remains valid.
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
 * fmt_dur: format a duration for the status line.
 *   In:     s seconds; buf, n: output buffer.
 *   Out:    buf, e.g. "12.3s", "4m05s", "2h03m04s", "3d04h05m"; returns buf.
 *   Errors: "?" for negative, NaN or absurd (> 10^12 s) values, e.g. an ETA before
 *           any rate is known.  Output is truncated to n bytes by snprintf.
 */
static const char *fmt_dur(double s, char *buf, size_t n)
{
    if (s < 0 || s != s || s > 1e12) { snprintf(buf, n, "?"); return buf; }
    long t = (long)s;
    if (t >= 86400) snprintf(buf, n, "%ldd%02ldh%02ldm", t / 86400, (t / 3600) % 24, (t / 60) % 60);
    else if (t >= 3600) snprintf(buf, n, "%ldh%02ldm%02lds", t / 3600, (t / 60) % 60, t % 60);
    else if (t >= 60) snprintf(buf, n, "%ldm%02lds", t / 60, t % 60);
    else snprintf(buf, n, "%.1fs", s);
    return buf;
}

/*
 * mul_ok: 64-bit multiplication with overflow detection.
 *   In:     a, b.
 *   Out:    true and *r = a*b if the product fits in 64 bits; false otherwise
 *           (*r is then unchanged).
 *   Errors: none.
 */
static bool mul_ok(u64 a, u64 b, u64 *r)
{
    u128 x = (u128)a * b;
    if (x >> 64) return false;
    *r = (u64)x;
    return true;
}

/*
 * all_digits: is s[0..n) a non-empty run of decimal digits?
 *   In:     s, n.
 *   Out:    true iff n > 0 and every character is '0'..'9'.
 *   Errors: none.
 */
static bool all_digits(const char *s, size_t n)
{
    if (n == 0) return false;
    for (size_t i = 0; i < n; i++)
        if (!isdigit((unsigned char)s[i])) return false;
    return true;
}

/*
 * parse_term: parse one term of a command-line number (a part without + or -).
 *   In:     s, n: the term's characters (need not be NUL-terminated).  Accepted forms:
 *           decimal digits ("1000"), B^E with decimal B and E ("2^63"), and MeE where
 *           the mantissa M may contain one decimal point and E is decimal digits only
 *           ("1e15", "1.2e15", "5.e3"; "1e-5" and "1e+5" are not accepted).
 *   Out:    true and *out = the value on success.
 *   Errors: false for an empty or over-long term (>= 64 characters), stray characters,
 *           a mantissa without digits ("e5"), any value or intermediate that does not
 *           fit in 64 bits, and an MeE value that is not an integer ("1.5e0").  Every
 *           loop is short: B^E with B <= 1 and 0eE are answered without iterating,
 *           otherwise overflow ends the loops after at most 64 rounds.
 */
static bool parse_term(const char *s, size_t n, u64 *out)
{
    char buf[64], *c;
    if (n == 0 || n >= sizeof buf) return false;
    memcpy(buf, s, n);
    buf[n] = 0;
    if ((c = strchr(buf, '^'))) {                           /* B^E */
        *c = 0;
        if (!all_digits(buf, strlen(buf)) || !all_digits(c + 1, strlen(c + 1))) return false;
        errno = 0;
        u64 b = strtoull(buf, 0, 10), e = strtoull(c + 1, 0, 10), r = 1;
        if (errno) return false;                            /* B or E alone exceeds 64 bits */
        if (b <= 1) { *out = e == 0 ? 1 : b; return true; } /* 0^E, 1^E without a long loop */
        while (e--) if (!mul_ok(r, b, &r)) return false;    /* b >= 2: overflow within 64 rounds */
        *out = r;
        return true;
    }
    if ((c = strpbrk(buf, "eE"))) {                         /* MeE, M = digits[.digits] */
        *c = 0;
        char *dot = strchr(buf, '.');
        size_t ip = dot ? (size_t)(dot - buf) : strlen(buf);    /* digits before the point */
        const char *frac = dot ? dot + 1 : buf + ip;            /* digits after it ("" if none) */
        size_t fl = strlen(frac);
        if ((ip && !all_digits(buf, ip)) || (fl && !all_digits(frac, fl))) return false;
        if (ip + fl == 0) return false;                     /* no mantissa digits at all */
        if (!all_digits(c + 1, strlen(c + 1))) return false;    /* exponent: digits only */
        while (fl && frac[fl - 1] == '0') fl--;             /* 1.50e3 = 15e2: trailing zeros only cost range */
        errno = 0;
        long e = strtol(c + 1, 0, 10);
        if (errno) return false;
        e -= (long)fl;                                      /* value = digits(M) * 10^e */
        u64 m = 0;
        for (size_t i = 0; i < ip; i++) if (!mul_ok(m, 10, &m) || (m += (u64)(buf[i] - '0')) < (u64)(buf[i] - '0')) return false;
        for (size_t i = 0; i < fl; i++) if (!mul_ok(m, 10, &m) || (m += (u64)(frac[i] - '0')) < (u64)(frac[i] - '0')) return false;
        if (m == 0) { *out = 0; return true; }              /* 0eE without a long loop */
        for (; e > 0; e--) if (!mul_ok(m, 10, &m)) return false;    /* overflow within 20 rounds */
        for (; e < 0; e++) { if (m % 10) return false; m /= 10; }   /* digits left after the point */
        *out = m;
        return true;
    }
    if (!all_digits(buf, n)) return false;                  /* plain decimal */
    errno = 0;
    *out = strtoull(buf, 0, 10);
    return errno == 0;
}

/*
 * parse_num: parse a command-line number: a term (see parse_term) or TERM+TERM / TERM-TERM.
 *   In:     s, a NUL-terminated string such as "2e14", "2^63-25", "1e15+4e6".
 *   Out:    the value.
 *   Errors: dies with a message for anything parse_term rejects, for more than one
 *           operator ("1-2-3": the left part is not a term), an empty operand, a sum
 *           that overflows 64 bits, and a negative difference.  A '+' or '-' directly
 *           after 'e'/'E' is not treated as an operator, so "1e-5" is rejected as a
 *           term rather than split.  A leading '-' is never an operator ("-5" fails).
 */
static u64 parse_num(const char *s)
{
    size_t n = strlen(s);
    for (size_t i = n; i-- > 1; ) {           /* split at the last + or - (position >= 1) */
        if (s[i] != '+' && s[i] != '-') continue;
        if (s[i - 1] == 'e' || s[i - 1] == 'E') continue;
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
 * parse_int: parse a small integer option value within limits.
 *   In:     s (any parse_num syntax), the allowed range [lo, hi] (lo >= 0), and the
 *           option's name for the message.
 *   Out:    the value as an int.
 *   Errors: dies if s is not a number (parse_num) or lies outside [lo, hi].
 */
static int parse_int(const char *s, int lo, int hi, const char *what)
{
    u64 v = parse_num(s);
    if (v < (u64)lo || v > (u64)hi) die("%s must be between %d and %d", what, lo, hi);
    return (int)v;
}

/*
 * default_threads: number of online CPUs, the default worker count.
 *   In:     nothing.
 *   Out:    sysconf(_SC_NPROCESSORS_ONLN), or 1 if that fails.
 *   Errors: none.
 */
static int default_threads(void)
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
}

/*
 * mix64: the splitmix64 finalizer, a bijective 64-bit mixing function.
 *   In:     z.
 *   Out:    mix64(z).  The checksum term of a pair is mix64(q ^ p*GOLD) (section 10);
 *           the same function is used by the CUDA tool and verify_wieferich.py, and as
 *           the selftest's random number generator.
 *   Errors: none.
 */
static inline u64 mix64(u64 z)
{
    z += GOLD;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* ------------------------------------------------------------------ */
/* Independent primality: deterministic Miller-Rabin below 2^64         */
/* (used by check and the selftest only; scans get primes from primesieve) */
/* ------------------------------------------------------------------ */

/*
 * mulmod64: a*b mod m with a 128-bit intermediate.
 *   In:     a, b, m >= 1.
 *   Out:    (a*b) mod m.
 *   Errors: m = 0 divides by zero (callers never pass it).
 */
static u64 mulmod64(u64 a, u64 b, u64 m) { return (u64)(((u128)a * b) % m); }

/*
 * powmod64: a^e mod m by binary exponentiation.
 *   In:     a, e, m >= 1.
 *   Out:    a^e mod m (1 % m for e = 0, so 0 when m = 1).
 *   Errors: m = 0 divides by zero (callers never pass it).
 */
static u64 powmod64(u64 a, u64 e, u64 m)
{
    u64 r = 1 % m;
    a %= m;
    while (e) {
        if (e & 1) r = mulmod64(r, a, m);
        a = mulmod64(a, a, m);
        e >>= 1;
    }
    return r;
}

/*
 * is_prime64: deterministic primality test for 64-bit integers.
 *   In:     n.
 *   Out:    true iff n is prime.  Trial division by the primes up to 37, then strong
 *           probable-prime tests to the same 12 bases, which are correct for every
 *           n < 3.18e23 (the smallest strong pseudoprime to all of them), so for all
 *           64-bit n.
 *   Errors: none.
 */
static bool is_prime64(u64 n)
{
    static const u64 small[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    if (n < 2) return false;
    for (size_t i = 0; i < sizeof small / sizeof *small; i++) {
        if (n == small[i]) return true;
        if (n % small[i] == 0) return false;
    }
    u64 d = n - 1;
    int s = 0;
    while (!(d & 1)) { d >>= 1; s++; }                  /* n - 1 = d * 2^s, d odd */
    for (size_t i = 0; i < sizeof small / sizeof *small; i++) {
        u64 x = powmod64(small[i], d, n);
        if (x == 1 || x == n - 1) continue;
        bool comp = true;
        for (int r = 1; r < s && comp; r++) {
            x = mulmod64(x, x, n);
            if (x == n - 1) comp = false;
        }
        if (comp) return false;                         /* small[i] is a witness */
    }
    return true;
}

/* ------------------------------------------------------------------ */
/* Arithmetic mod N = p^2 (sections 4-6)                                */
/* ------------------------------------------------------------------ */

/* Per-prime constants, set up once and shared by all bases (section 4). */
typedef struct {
    u128 N, one;            /* p^2, 2^128 mod p^2 (Montgomery form of 1) */
    double invN;            /* 1/N, for quotient estimates in msmall */
    u64 p, pinv, ninv;      /* p, p^-1 mod 2^64, -N^-1 mod 2^64 */
} pctx_t;

/*
 * pctx_init: set up the constants for arithmetic mod N = p^2.
 *   In:     c (output), p odd with 3 <= p < 2^63.
 *   Out:    *c filled (section 4).  one = 2^128 mod N is computed as 2^128 - Q N with
 *           the quotient Q = floor(2^128 / N) estimated in double precision when
 *           N >= 2^80 (then Q < 2^48 and the estimate is off by at most 1, corrected
 *           as in msmall), and with a 128-bit '%' for smaller N (rare, slower).
 *   Errors: none detected.  An even p gives a wrong pinv/ninv (no inverse mod 2^64)
 *           and p >= 2^63 breaks the bounds of msqr; callers guarantee neither occurs
 *           (scans skip p = 2, check rejects even P, END <= 2^63).
 */
static inline void pctx_init(pctx_t *c, u64 p)
{
    u64 x = (3 * p) ^ 2;                        /* p * x = 1 mod 2^5 for odd p */
    x *= 2 - p * x; x *= 2 - p * x; x *= 2 - p * x; x *= 2 - p * x;   /* 10, 20, 40, 80 bits */
    c->p = p;
    c->pinv = x;
    c->N = (u128)p * p;
    c->ninv = 0 - x * x;                        /* N^-1 = (p^-1)^2 mod 2^64, negated */
    c->invN = 1.0 / (double)c->N;
    if (c->N >> 80) {                           /* 2^128 / N < 2^48: estimate the quotient */
        u64 q = (u64)(0x1p128 * c->invN);
        u128 r = (u128)0 - (u128)q * c->N;      /* 2^128 - q N, mod 2^128: true value in [-N, 2N) */
        if ((__int128)r < 0) r += c->N;
        else if (r >= c->N) r -= c->N;
        c->one = r;
    } else {
        c->one = ((u128)0 - c->N) % c->N;       /* (2^128 - N) mod N = 2^128 mod N */
    }
}

/*
 * msqr: Montgomery squaring, a^2 / 2^128 mod N (section 5).
 *   In:     a < N, m = the prime's constants (N < 2^126).
 *   Out:    a^2 R^-1 mod N, fully reduced (< N).
 *   Errors: none if a < N < 2^126; larger inputs overflow the bounds noted inline
 *           (callers only pass reduced values).
 */
static inline u128 msqr(u128 a, const pctx_t *m)
{
    const u64 a0 = (u64)a, a1 = (u64)(a >> 64);         /* a1 < 2^62 because a < 2^126 */
    const u64 n0 = (u64)m->N, n1 = (u64)(m->N >> 64);
    const u128 p00 = (u128)a0 * a0, p01 = (u128)a0 * a1, p11 = (u128)a1 * a1;
    const u128 d = p01 << 1;                    /* 2 a0 a1 < 2^127 since a1 < 2^62 */
    /* a^2 = p00 + d 2^64 + p11 2^128 as four words t0..t3 (t3 < 2^61) */
    const u64 t0 = (u64)p00;
    u128 acc = (u128)(u64)(p00 >> 64) + (u64)d;
    u64 t1 = (u64)acc;
    acc = (acc >> 64) + (u64)(d >> 64) + (u64)p11;
    u64 t2 = (u64)acc;
    u64 t3 = (u64)(acc >> 64) + (u64)(p11 >> 64);
    /* REDC, word 0: add q N with q = t0 * ninv so that the low word becomes 0 */
    u64 q = t0 * m->ninv;
    acc = (u128)q * n0 + t0;                    /* low 64 bits are 0 by construction */
    acc = (u128)q * n1 + t1 + (u64)(acc >> 64); /* q n1 < 2^126: no overflow */
    t1 = (u64)acc;
    acc = (u128)t2 + (u64)(acc >> 64);
    t2 = (u64)acc;
    t3 += (u64)(acc >> 64);
    /* REDC, word 1 */
    q = t1 * m->ninv;
    acc = (u128)q * n0 + t1;
    acc = (u128)q * n1 + t2 + (u64)(acc >> 64);
    const u64 r0 = (u64)acc;
    acc = (u128)t3 + (u64)(acc >> 64);
    const u128 r = ((u128)(u64)acc << 64) | r0; /* (a^2 + Q N) / 2^128 < 2N < 2^127 */
    return r >= m->N ? r - m->N : r;
}

/*
 * msmall: x * c mod N for a small plain constant c (section 6).
 *   In:     x < N < 2^126, c < 2^32, m = the prime's constants.  x may be a Montgomery
 *           representative: (x R) c is the representative of x c.
 *   Out:    x c mod N, fully reduced.
 *   Errors: none for the stated ranges; c >= 2^32 can make the quotient estimate off by
 *           more than 1 and the result wrong (bases_setup keeps every b^d < 2^32).
 */
static inline u128 msmall(u128 x, u64 c, const pctx_t *m)
{
    const u128 lo = (u128)(u64)x * c;
    const u128 hi = (u128)(u64)(x >> 64) * c;
    const u128 t = lo + (hi << 64);             /* x*c mod 2^128 */
    const u64 top = (u64)(hi >> 64) + (t < lo); /* x*c >> 128 (carry of the addition included) */
    const double td = (double)top * 0x1p128 + (double)(u64)(t >> 64) * 0x1p64 + (double)(u64)t;
    const u64 q = (u64)(td * m->invN);          /* floor(x*c / N) + {-1, 0, 1} */
    u128 r = t - (u128)q * m->N;                /* true value in [-N, 2N): fits in signed 128 */
    if ((__int128)r < 0) r += m->N;
    else if (r >= m->N) r -= m->N;
    return r;
}

/*
 * redc: leave Montgomery form, x / 2^128 mod N (section 5).
 *   In:     x < N, m = the prime's constants.
 *   Out:    x R^-1 mod N, fully reduced.  Applied to the representative of y this is y.
 *   Errors: none for x < N.
 */
static inline u128 redc(u128 x, const pctx_t *m)
{
    const u64 n0 = (u64)m->N, n1 = (u64)(m->N >> 64);
    u64 t0 = (u64)x, t1 = (u64)(x >> 64);
    u64 q = t0 * m->ninv;                       /* word 0 */
    u128 acc = (u128)q * n0 + t0;
    acc = (u128)q * n1 + t1 + (u64)(acc >> 64);
    t0 = (u64)acc;
    t1 = (u64)(acc >> 64);
    q = t0 * m->ninv;                           /* word 1 */
    acc = (u128)q * n0 + t0;
    acc = (u128)q * n1 + t1 + (u64)(acc >> 64);
    return acc >= m->N ? acc - m->N : acc;      /* (x + Q N) / R <= N */
}

/*
 * mulmod_ref: a*b mod m by shift and add, independent of the Montgomery code.
 *   In:     a, b, and 1 <= m < 2^127 (so 2a and r + a cannot overflow 128 bits).
 *   Out:    (a*b) mod m.  ~128 iterations; used only for checking (selftest, check).
 *   Errors: m = 0 divides by zero; m >= 2^127 can overflow silently.  All callers use
 *           m = p^2 < 2^126.
 */
static u128 mulmod_ref(u128 a, u128 b, u128 m)
{
    u128 r = 0;
    a %= m;
    b %= m;
    while (b) {
        if (b & 1) { r += a; if (r >= m) r -= m; }
        a <<= 1;
        if (a >= m) a -= m;
        b >>= 1;
    }
    return r;
}

/*
 * powmod_ref: b^e mod m with mulmod_ref (reference for checking only).
 *   In:     b, e, 1 <= m < 2^127.
 *   Out:    b^e mod m.
 *   Errors: as mulmod_ref.
 */
static u128 powmod_ref(u64 b, u64 e, u128 m)
{
    u128 r = 1 % m, x = b % m;
    while (e) {
        if (e & 1) r = mulmod_ref(r, x, m);
        x = mulmod_ref(x, x, m);
        e >>= 1;
    }
    return r;
}

/* ------------------------------------------------------------------ */
/* Bases                                                                */
/* ------------------------------------------------------------------ */

/* The bases with their own OEIS sequence (keyword "oeis"; base 2 is not included). */
static const u32 OEIS_BASES[] = {3, 5, 6, 7, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 26, 30};
static const char *const OEIS_ANUM[] = {"A014127", "A123692", "A212583", "A123693", "A045616",
    "A111027", "A128667", "A234810", "A242741", "A128668", "A244260", "A090968", "A242982",
    "A298951", "A128669", "A306255", "A306256"};

/* Known solutions (OEIS; R. Fischer's table) for the bases 2..30 that are not perfect
 * powers (keyword "known"; the selftest's reference data), including A306256(4) =
 * 303632117562967 found by this project in Oct 2026. */
typedef struct { u32 b; int n; u64 p[8]; } known_t;
static const known_t KNOWN[] = {
    { 2, 2, {1093, 3511}},
    { 3, 2, {11, 1006003}},
    { 5, 7, {2, 20771, 40487, 53471161, 1645333507ULL, 6692367337ULL, 188748146801ULL}},
    { 6, 3, {66161, 534851, 3152573}},
    { 7, 2, {5, 491531}},
    {10, 3, {3, 487, 56598313}},
    {11, 1, {71}},
    {12, 2, {2693, 123653}},
    {13, 3, {2, 863, 1747591}},
    {14, 3, {29, 353, 7596952219ULL}},
    {15, 2, {29131, 119327070011ULL}},
    {17, 5, {2, 3, 46021, 48947, 478225523351ULL}},
    {18, 6, {5, 7, 37, 331, 33923, 1284043}},
    {19, 6, {3, 7, 13, 43, 137, 63061489}},
    {20, 4, {281, 46457, 9377747, 122959073}},
    {21, 1, {2}},
    {22, 5, {13, 673, 1595813, 492366587, 9809862296159ULL}},
    {23, 5, {13, 2481757, 13703077, 15546404183ULL, 2549536629329ULL}},
    {24, 2, {5, 25633}},
    {26, 5, {3, 5, 71, 486999673, 6695256707ULL}},
    {28, 3, {3, 19, 23}},
    {29, 1, {2}},
    {30, 4, {7, 160541, 94727075783ULL, 303632117562967ULL}},
};
#define NKNOWN ((int)(sizeof KNOWN / sizeof *KNOWN))

/* The bases of the current run, sorted, with their windows and power tables, grouped
 * into window classes (bases with equal w can share a lane group, section 8). */
typedef struct {
    int nb;                                     /* number of bases */
    u32 b[MAXB];                                /* the bases, ascending */
    int w[MAXB];                                /* window size of each base */
    u64 pw[MAXB][1 << MAXW];                    /* b^d for d < 2^w, all < 2^32 */
    int ncls;                                   /* number of window classes */
    int cls_w[MAXW + 1], cls_n[MAXW + 1], cls_j[MAXW + 1][MAXB];  /* class: w, size, base indices */
} bases_t;

static bases_t B;

/*
 * window_for: the exponent window for base b (section 7).
 *   In:     b >= 2.
 *   Out:    the largest w <= MAXW with b^(2^w - 1) < 2^32 (at least 1).
 *   Errors: none; for b >= 2^32 it returns 1 although b itself does not fit msmall
 *           (parse_bases rejects such bases).
 */
static int window_for(u32 b)
{
    for (int w = MAXW; w > 1; w--) {
        u64 x = 1;
        bool ok = true;
        for (int i = 0; i < (1 << w) - 1 && ok; i++) {
            x *= b;                             /* x < 2^32 and b < 2^32: no overflow */
            if (x >> 32) ok = false;
        }
        if (ok) return w;
    }
    return 1;
}

/*
 * cmp_u32: qsort comparator for ascending u32.
 *   In:     pointers to two u32.
 *   Out:    -1, 0 or 1.
 *   Errors: none.
 */
static int cmp_u32(const void *a, const void *b)
{
    u32 x = *(const u32 *)a, y = *(const u32 *)b;
    return x < y ? -1 : x > y;
}

/*
 * bases_setup: make list[0..n) the bases of the run (global B).
 *   In:     list of bases (any order, duplicates allowed), 1 <= n <= MAXB.
 *   Out:    B: bases sorted and deduplicated, each with its window and power table,
 *           grouped into window classes ordered from large w to small.
 *   Errors: dies for n outside [1, MAXB] or a base < 2.  Bases must be < 2^32 (the
 *           type guarantees it; parse_bases checks the range on input).
 */
static void bases_setup(const u32 *list, int n)
{
    u32 tmp[MAXB];
    if (n < 1 || n > MAXB) die("between 1 and %d bases are supported", MAXB);
    memcpy(tmp, list, (size_t)n * sizeof *tmp);
    qsort(tmp, (size_t)n, sizeof *tmp, cmp_u32);
    memset(&B, 0, sizeof B);
    for (int i = 0; i < n; i++) {
        if (tmp[i] < 2) die("bases must be >= 2");
        if (B.nb && B.b[B.nb - 1] == tmp[i]) continue;      /* duplicate */
        int j = B.nb++;
        B.b[j] = tmp[i];
        B.w[j] = window_for(tmp[i]);
        u64 x = 1;
        for (int d = 0; d < (1 << B.w[j]); d++) { B.pw[j][d] = x; x *= tmp[i]; }
    }
    for (int w = MAXW; w >= 1; w--) {
        int c = B.ncls, k = 0;
        for (int j = 0; j < B.nb; j++) if (B.w[j] == w) B.cls_j[c][k++] = j;
        if (k) { B.cls_w[c] = w; B.cls_n[c] = k; B.ncls++; }
    }
}

/*
 * push_base: append b to list[0..*n) unless it is already there.
 *   In:     list, *n current length, b.
 *   Out:    list and *n updated.
 *   Errors: dies when a new base would exceed MAXB.
 */
static void push_base(u32 *list, int *n, u32 b)
{
    for (int i = 0; i < *n; i++) if (list[i] == b) return;
    if (*n >= MAXB) die("too many bases (max %d)", MAXB);
    list[(*n)++] = b;
}

/*
 * parse_bases: parse a -b argument and set up the bases (bases_setup).
 *   In:     s: comma-separated items, each a number (parse_num syntax), a range LO-HI
 *           (split at the first '-', so "2^3-10" is 8..10), "oeis" or "known".
 *   Out:    global B set; duplicates are ignored.
 *   Errors: dies for an over-long list (>= 1024 characters), bad numbers, a base < 2
 *           or >= 2^32, an empty or reversed range, more than MAXB distinct bases, or
 *           an empty list.
 */
static void parse_bases(const char *s)
{
    u32 list[MAXB];
    int n = 0;
    char buf[1024];
    if (strlen(s) >= sizeof buf) die("base list too long");
    strcpy(buf, s);
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        if (!strcmp(tok, "oeis")) {
            for (size_t i = 0; i < sizeof OEIS_BASES / sizeof *OEIS_BASES; i++) push_base(list, &n, OEIS_BASES[i]);
            continue;
        }
        if (!strcmp(tok, "known")) {
            for (int i = 0; i < NKNOWN; i++) push_base(list, &n, KNOWN[i].b);
            continue;
        }
        char *dash = strchr(tok, '-');
        u64 lo, hi;
        if (dash) { *dash = 0; lo = parse_num(tok); hi = parse_num(dash + 1); }
        else lo = hi = parse_num(tok);
        if (lo < 2 || hi < lo || hi >= (1ULL << 32)) die("bad base or range '%s'", tok);
        for (u64 b = lo; b <= hi; b++) push_base(list, &n, (u32)b);
    }
    bases_setup(list, n);
}

/*
 * anum_for: the OEIS A-number of base b's sequence.
 *   In:     b.
 *   Out:    "A001220" for b = 2, the entry of OEIS_ANUM for the "oeis" bases, else NULL.
 *   Errors: none.
 */
static const char *anum_for(u32 b)
{
    if (b == 2) return "A001220";
    for (size_t i = 0; i < sizeof OEIS_BASES / sizeof *OEIS_BASES; i++)
        if (OEIS_BASES[i] == b) return OEIS_ANUM[i];
    return NULL;
}

/*
 * print_bases: write the current bases as "b1,b2,..." (the checkpoint's format).
 *   In:     f, an open stream.
 *   Out:    the list written to f, no newline.
 *   Errors: none reported (write errors surface in the caller's ferror/fclose).
 */
static void print_bases(FILE *f)
{
    for (int j = 0; j < B.nb; j++) fprintf(f, "%s%u", j ? "," : "", B.b[j]);
}

/*
 * bases_str: the current bases as "b1,b2,..." in a buffer (same text as print_bases).
 *   In:     buf, n >= 1.
 *   Out:    buf, NUL-terminated.
 *   Errors: none; the text is truncated if n is too small (64 bases need < 720 bytes).
 */
static void bases_str(char *buf, size_t n)
{
    size_t k = 0;
    buf[0] = 0;
    for (int j = 0; j < B.nb && k < n; j++) {
        int w = snprintf(buf + k, n - k, "%s%u", j ? "," : "", B.b[j]);
        if (w < 0) break;
        k += (size_t)w;
    }
}

/* ------------------------------------------------------------------ */
/* The kernel (sections 7-8)                                            */
/* ------------------------------------------------------------------ */

/* One (prime, base) exponentiation: the prime's constants, the base's power table,
 * the exponent and the base's index (for the checksum). */
typedef struct {
    const pctx_t *c;
    const u64 *pw;
    u64 e;                  /* p - 1 */
    int j;                  /* base index */
} lane_t;

/*
 * run_group: LANES exponentiations in lock step (sections 7, 8).
 *   In:     L[0..LANES): lanes that all use window w (each L[g].pw has 2^w entries
 *           < 2^32, each L[g].e >= 2); w in 1..MAXW; r = output array.
 *   Out:    r[g] = b_g^(p_g - 1) mod p_g^2 in normal (non-Montgomery) form, < p_g^2.
 *   Errors: none for valid lanes.  Every lane is computed: callers pad incomplete
 *           groups with copies of a valid lane and ignore the extra results.
 */
static inline void run_group(const lane_t *L, int w, u128 *r)
{
    const u64 mask = ((u64)1 << w) - 1;
    u128 x[LANES];
    u64 emax = 0;
    for (int g = 0; g < LANES; g++) emax |= L[g].e;         /* bit length of the longest exponent */
    int sh = ((64 - __builtin_clzll(emax) + w - 1) / w - 1) * w;    /* shift of the top window */
    for (int g = 0; g < LANES; g++) x[g] = msmall(L[g].c->one, L[g].pw[(L[g].e >> sh) & mask], L[g].c);
    for (sh -= w; sh >= 0; sh -= w) {
        for (int s = 0; s < w; s++)
            for (int g = 0; g < LANES; g++) x[g] = msqr(x[g], L[g].c);
        for (int g = 0; g < LANES; g++) x[g] = msmall(x[g], L[g].pw[(L[g].e >> sh) & mask], L[g].c);
    }
    for (int g = 0; g < LANES; g++) r[g] = redc(x[g], L[g].c);
}

/*
 * fermat_quotient: q_p(b_j) through the production kernel (check, selftest).
 *   In:     p odd with 3 <= p < 2^63 and not dividing B.b[j]; j a valid base index;
 *           res = NULL or where to store the residue.
 *   Out:    q = (b^(p-1) mod p^2 - 1) / p mod 2^64, and *res = b^(p-1) mod p^2.  For
 *           prime p the quotient lies in [0, p); for composite p the value is the same
 *           formula but has no meaning.
 *   Errors: none detected here (callers compare with the reference).
 */
static u64 fermat_quotient(u64 p, int j, u128 *res)
{
    pctx_t c;
    pctx_init(&c, p);
    lane_t L[LANES];
    for (int g = 0; g < LANES; g++) { L[g].c = &c; L[g].pw = B.pw[j]; L[g].e = p - 1; L[g].j = j; }
    u128 r[LANES];
    run_group(L, B.w[j], r);                    /* all lanes identical; lane 0 is the answer */
    if (res) *res = r[0];
    return ((u64)r[0] - 1) * c.pinv;
}

/* ------------------------------------------------------------------ */
/* Scan state (section 11)                                              */
/* ------------------------------------------------------------------ */

/* Results of one completed chunk, waiting in the ring until the frontier reaches it. */
typedef struct {
    bool done;
    u64 primes, tests, near, flt;
    u64 cks[MAXB];
} slot_t;

/* A solution: base, prime, and the chunk it was found in (for CONFIRMED). */
typedef struct { u32 b; u64 p; u64 chunk; bool announced; } hit_t;

/* The scan.  Fields other than next and tests_live are protected by g_mu while
 * workers run. */
static struct {
    u64 start, chunk, end, nchunks;
    atomic_ullong next;             /* next chunk index to hand out */
    u64 frontier;                   /* chunks [0, frontier) are complete */
    u64 primes, tests, near, flt;   /* totals over [start, frontier position) */
    u64 cks[MAXB];
    slot_t slot[RINGC];
    int nhits;
    hit_t hits[MAXHITS];
    atomic_ullong tests_live;       /* tests done in all chunks, for the rate display */
} S;

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static long long g_near = 0;        /* report |A| <= g_near (0 = off) */
static bool g_print = true;         /* print FOUND/NEAR/CONFIRMED lines (off in selftest) */
static FILE *g_log = NULL;          /* chunk log (-L), or NULL */

/*
 * record_hit: remember and announce a solution.
 *   In:     base b, prime p, chunk index idx.  Takes g_mu (callers must not hold it).
 *   Out:    the pair is appended to S.hits (unless already present) and, if g_print, a
 *           "FOUND base b p P (Annnnnn)" line goes to stdout.
 *   Errors: a duplicate is ignored (a chunk redone after a resume finds its solutions
 *           again).  Beyond MAXHITS solutions per run, new ones are still printed but
 *           not kept for the summary and the checkpoint; a warning is printed once.
 */
static void record_hit(u32 b, u64 p, u64 idx)
{
    static bool warned = false;
    pthread_mutex_lock(&g_mu);
    bool dup = false;
    for (int i = 0; i < S.nhits; i++) if (S.hits[i].b == b && S.hits[i].p == p) dup = true;
    if (!dup && S.nhits < MAXHITS) {
        hit_t *h = &S.hits[S.nhits++];
        h->b = b; h->p = p; h->chunk = idx; h->announced = false;
    } else if (!dup && !warned) {
        warned = true;
        fprintf(stderr, "\nwarning: more than %d solutions; later ones are printed but not kept\n", MAXHITS);
    }
    if (g_print && !dup) {
        if (stderr_tty) fputs("\r\033[K", stderr);
        const char *a = anum_for(b);
        printf("FOUND base %u p %" PRIu64 "%s%s%s\n", b, p, a ? " (" : "", a ? a : "", a ? ")" : "");
        fflush(stdout);
    }
    pthread_mutex_unlock(&g_mu);
}

/*
 * report_line: print one result line on stdout without mixing with other threads.
 *   In:     printf-style format and arguments (should end with a newline).
 *           Takes g_mu (callers must not hold it).
 *   Out:    the line on stdout (flushed); the status line on a terminal is cleared first.
 *   Errors: none.
 */
static void report_line(const char *fmt, ...)
{
    va_list ap;
    pthread_mutex_lock(&g_mu);
    if (stderr_tty) fputs("\r\033[K", stderr);
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    fflush(stdout);
    pthread_mutex_unlock(&g_mu);
}

/* A worker's running totals for the chunk it is working on. */
typedef struct {
    u64 primes, tests, near, flt;
    u64 cks[MAXB];
    u64 idx;                        /* the chunk's index */
} acc_t;

/*
 * finish_lanes: check and account the results of the first n lanes of a group (section 9).
 *   In:     L, r from run_group; n <= LANES valid lanes (padding lanes are excluded);
 *           a = the chunk's totals.
 *   Out:    per lane: tests += 1; q = (r - 1) p^-1 mod 2^64; if q >= p an FLT-ERROR line
 *           is printed and flt += 1; otherwise the checksum term is added, q = 0 is
 *           recorded as a solution (record_hit) and 0 < |A| <= g_near as a NEAR line.
 *   Errors: an FLT error indicates a wrong residue (bug or hardware fault); the pair is
 *           left out of the checksum, so checksums disagree with any correct run.
 */
static inline void finish_lanes(const lane_t *L, const u128 *r, int n, acc_t *a)
{
    for (int g = 0; g < n; g++) {
        const u64 p = L[g].c->p;
        const u64 q = ((u64)r[g] - 1) * L[g].c->pinv;     /* exact division by p (mod 2^64) */
        const int j = L[g].j;
        a->tests++;
        if (q >= p) {                                       /* Fermat's little theorem failed */
            a->flt++;
            report_line("FLT-ERROR base %u p %" PRIu64 " r = %016" PRIx64 "%016" PRIx64 "\n",
                        B.b[j], p, (u64)(r[g] >> 64), (u64)r[g]);
            continue;
        }
        a->cks[j] += mix64(q ^ (p * GOLD));
        if (q == 0) {
            record_hit(B.b[j], p, a->idx);
        } else if (g_near > 0) {
            long long A = q <= p / 2 ? (long long)q : -(long long)(p - q);   /* symmetric residue */
            if (llabs(A) <= g_near) {
                a->near++;
                if (g_print) report_line("NEAR base %u p %" PRIu64 " A %lld\n", B.b[j], p, A);
            }
        }
    }
}

/*
 * process_batch: test np primes against every base (section 8).
 *   In:     ctx[0..np): set-up primes (odd, < 2^63); a = the chunk's totals.
 *   Out:    for each window class, the (prime, base) pairs with p not dividing b are
 *           packed into groups of LANES, run and accounted (finish_lanes); the last
 *           group of a class is padded with copies of its last lane.  tests_live grows
 *           by np * nb (display only; skipped pairs are counted there).
 *   Errors: none beyond those reported by finish_lanes.
 */
static void process_batch(const pctx_t *ctx, int np, acc_t *a)
{
    lane_t L[LANES];
    u128 r[LANES];
    for (int c = 0; c < B.ncls; c++) {
        const int w = B.cls_w[c];
        int nl = 0;
        for (int i = 0; i < np; i++) {
            const u64 p = ctx[i].p;
            for (int k = 0; k < B.cls_n[c]; k++) {
                const int j = B.cls_j[c][k];
                if (p <= B.b[j] && B.b[j] % p == 0) continue;   /* p | b: b^(p-1) = 0 mod p */
                L[nl].c = &ctx[i]; L[nl].pw = B.pw[j]; L[nl].e = p - 1; L[nl].j = j;
                if (++nl == LANES) {
                    run_group(L, w, r);
                    finish_lanes(L, r, LANES, a);
                    nl = 0;
                }
            }
        }
        if (nl) {
            for (int g = nl; g < LANES; g++) L[g] = L[nl - 1];     /* padding, results ignored */
            run_group(L, w, r);
            finish_lanes(L, r, nl, a);
        }
    }
    atomic_fetch_add_explicit(&S.tests_live, (u64)np * (u64)B.nb, memory_order_relaxed);
}

/*
 * scan_chunk: test every prime of chunk idx and fold completed chunks (section 11).
 *   In:     it = the worker's primesieve iterator; idx < S.nchunks.
 *   Out:    the chunk's totals are stored in its ring slot; then, under g_mu, every
 *           complete slot at the frontier is folded into S (totals, checksums, chunk
 *           log line) and the frontier advances.  p = 2 (only in the chunk containing
 *           it) is decided directly: a solution for every base b = 1 (mod 4).
 *   Errors: if g_stop is set, the chunk is abandoned after the current batch and its
 *           slot is not marked done (its solutions may already have been printed as
 *           FOUND; they are not CONFIRMED or saved, and the chunk is redone on
 *           resume).  primesieve reports its own errors (e.g. out of memory) on stderr
 *           and then returns UINT64_MAX, which ends the loop like the chunk's end; this
 *           cannot happen for valid ranges below 2^63.
 */
static void scan_chunk(primesieve_iterator *it, u64 idx)
{
    const u64 lo = S.start + idx * S.chunk;
    const u64 hi = lo + S.chunk < S.end ? lo + S.chunk : S.end;
    acc_t a;
    memset(&a, 0, sizeof a);
    a.idx = idx;
    pctx_t ctx[BATCH];
    int np = 0;
    primesieve_jump_to(it, lo, hi);             /* the next prime returned is the first >= lo */
    for (;;) {
        const u64 p = primesieve_next_prime(it);
        if (p >= hi) break;
        a.primes++;
        if (p == 2) {
            for (int j = 0; j < B.nb; j++) if (B.b[j] % 4 == 1) record_hit(B.b[j], 2, idx);
            continue;
        }
        pctx_init(&ctx[np], p);
        if (++np == BATCH) {
            process_batch(ctx, np, &a);
            np = 0;
            if (g_stop) return;                 /* chunk abandoned; redone on resume */
        }
    }
    if (np) process_batch(ctx, np, &a);

    pthread_mutex_lock(&g_mu);
    slot_t *s = &S.slot[idx % RINGC];
    s->done = true;
    s->primes = a.primes; s->tests = a.tests; s->near = a.near; s->flt = a.flt;
    memcpy(s->cks, a.cks, sizeof a.cks);
    while (S.slot[S.frontier % RINGC].done) {   /* fold complete chunks in frontier order */
        slot_t *f = &S.slot[S.frontier % RINGC];
        f->done = false;
        S.primes += f->primes; S.tests += f->tests; S.near += f->near; S.flt += f->flt;
        for (int j = 0; j < B.nb; j++) S.cks[j] += f->cks[j];
        if (g_log) {
            const u64 flo = S.start + S.frontier * S.chunk;
            const u64 fhi = flo + S.chunk < S.end ? flo + S.chunk : S.end;
            fprintf(g_log, "C %" PRIu64 " %" PRIu64 " %" PRIu64, flo, fhi, f->primes);
            for (int j = 0; j < B.nb; j++) fprintf(g_log, " %u:%016" PRIx64, B.b[j], f->cks[j]);
            fputc('\n', g_log);
            fflush(g_log);
        }
        S.frontier++;
    }
    pthread_mutex_unlock(&g_mu);
}

/*
 * worker: thread body; scans chunks until none are left or g_stop is set.
 *   In:     arg unused.
 *   Out:    NULL.  Chunk indices come from the atomic counter S.next; a chunk more than
 *           RINGC - 1 ahead of the frontier waits (polling every 2 ms) so that its ring
 *           slot is free.
 *   Errors: none of its own; see scan_chunk.
 */
static void *worker(void *arg)
{
    (void)arg;
    primesieve_iterator it;
    primesieve_init(&it);
    while (!g_stop) {
        u64 idx = atomic_fetch_add(&S.next, 1);
        if (idx >= S.nchunks) break;
        for (;;) {                                   /* stay within RINGC of the frontier */
            pthread_mutex_lock(&g_mu);
            u64 f = S.frontier;
            pthread_mutex_unlock(&g_mu);
            if (idx < f + RINGC - 1 || g_stop) break;
            usleep(2000);
        }
        if (g_stop) break;
        scan_chunk(&it, idx);
    }
    primesieve_free_iterator(&it);
    return NULL;
}

/* ---- checkpoint file ---- */

/*
 * save_state: write the checkpoint (section 11) atomically.
 *   In:     path of the checkpoint.  Takes g_mu while reading the scan state.
 *   Out:    path holds the text
 *               wieferich checkpoint 1 / bases b1,b2,... / start / chunk / end /
 *               frontier / primes / tests / near / flt / cks <b> <hex> (one per base) /
 *               hit <b> <p> <chunk> (solutions below the frontier) / end
 *           The file is written as path.tmp and renamed over path.
 *   Errors: if path.tmp cannot be created, or any write, the close or the rename fails,
 *           a warning is printed, path.tmp is removed and the previous checkpoint is
 *           kept unchanged.  The scan continues either way.
 */
static void save_state(const char *path)
{
    char tmp[PATH_MAX];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "w");
    if (!f) { fprintf(stderr, "\nwarning: cannot write %s: %s\n", tmp, strerror(errno)); return; }
    pthread_mutex_lock(&g_mu);
    fprintf(f, "wieferich checkpoint 1\nbases ");
    print_bases(f);
    fprintf(f, "\nstart %" PRIu64 "\nchunk %" PRIu64 "\nend %" PRIu64 "\nfrontier %" PRIu64
               "\nprimes %" PRIu64 "\ntests %" PRIu64 "\nnear %" PRIu64 "\nflt %" PRIu64 "\n",
            S.start, S.chunk, S.end, S.frontier, S.primes, S.tests, S.near, S.flt);
    for (int j = 0; j < B.nb; j++) fprintf(f, "cks %u %016" PRIx64 "\n", B.b[j], S.cks[j]);
    for (int i = 0; i < S.nhits; i++)
        if (S.hits[i].chunk < S.frontier)
            fprintf(f, "hit %u %" PRIu64 " %" PRIu64 "\n", S.hits[i].b, S.hits[i].p, S.hits[i].chunk);
    pthread_mutex_unlock(&g_mu);
    fputs("end\n", f);
    bool bad = ferror(f) != 0;                  /* a failed write (e.g. disk full) */
    if (fclose(f) != 0) bad = true;
    if (bad || rename(tmp, path) != 0) {
        fprintf(stderr, "\nwarning: cannot update %s: %s (previous checkpoint kept)\n", path, strerror(errno));
        remove(tmp);
    }
}

/*
 * load_state: resume from a checkpoint written by save_state (or by the CUDA tool).
 *   In:     path; S.start, S.chunk, S.nchunks and the bases (B) of the current command
 *           must already be set.
 *   Out:    false if path does not exist (a fresh scan).  Otherwise true, with
 *           S.frontier, the totals, the checksums and the solutions restored.
 *   Errors: dies if the file cannot be read (other than not existing), is not a
 *           wieferich checkpoint, is truncated (no "end" line), has a malformed cks or
 *           hit line, lacks a checksum for a base or has one for a base not in the run,
 *           or was written for different bases, START or CHUNK, or if its frontier lies
 *           beyond the new END.  Unknown lines are ignored.
 */
static bool load_state(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        if (errno == ENOENT) return false;
        die("cannot read %s: %s", path, strerror(errno));
    }
    char line[4096], bases[4096] = "";
    if (!fgets(line, sizeof line, f) || strncmp(line, "wieferich checkpoint 1", 22) != 0)
        die("%s is not a wieferich checkpoint", path);
    u64 start = 0, chunk = 0, end = 0, frontier = 0;
    bool complete = false, seen[MAXB] = {false};
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "bases %4000s", bases) == 1) continue;
        if (sscanf(line, "start %" SCNu64, &start) == 1) continue;
        if (sscanf(line, "chunk %" SCNu64, &chunk) == 1) continue;
        if (sscanf(line, "end %" SCNu64, &end) == 1) continue;
        if (sscanf(line, "frontier %" SCNu64, &frontier) == 1) continue;
        if (sscanf(line, "primes %" SCNu64, &S.primes) == 1) continue;
        if (sscanf(line, "tests %" SCNu64, &S.tests) == 1) continue;
        if (sscanf(line, "near %" SCNu64, &S.near) == 1) continue;
        if (sscanf(line, "flt %" SCNu64, &S.flt) == 1) continue;
        if (!strncmp(line, "cks ", 4)) {
            u32 b; u64 v;
            if (sscanf(line, "cks %u %" SCNx64, &b, &v) != 2) die("%s: bad cks line", path);
            int j = -1;
            for (int k = 0; k < B.nb; k++) if (B.b[k] == b) j = k;
            if (j < 0) die("%s: base %u is not in the current base list", path, b);
            S.cks[j] = v;
            seen[j] = true;
            continue;
        }
        if (!strncmp(line, "hit ", 4)) {
            hit_t h = {0, 0, 0, false};
            if (sscanf(line, "hit %u %" SCNu64 " %" SCNu64, &h.b, &h.p, &h.chunk) != 3) die("%s: bad hit line", path);
            if (S.nhits < MAXHITS) S.hits[S.nhits++] = h;
            continue;
        }
        if (!strncmp(line, "end", 3)) { complete = true; break; }
    }
    fclose(f);
    if (!complete) die("%s is truncated", path);
    char cur[4096];
    bases_str(cur, sizeof cur);
    if (strcmp(cur, bases) != 0) die("%s was written for bases %s, not %s", path, bases, cur);
    for (int j = 0; j < B.nb; j++)
        if (!seen[j]) die("%s has no checksum for base %u", path, B.b[j]);
    if (start != S.start || chunk != S.chunk)
        die("%s was written for start=%" PRIu64 " chunk=%" PRIu64 ", not start=%" PRIu64 " chunk=%" PRIu64,
            path, start, chunk, S.start, S.chunk);
    if (frontier > S.nchunks) die("%s is past the new END (frontier %" PRIu64 " > %" PRIu64 " chunks)",
                                  path, frontier, S.nchunks);
    (void)end;                                  /* END may change between runs */
    S.frontier = frontier;
    return true;
}

/*
 * on_sigint: SIGINT/SIGTERM handler.
 *   In:     the signal number (unused).
 *   Out:    g_stop = 1; workers stop after their current batch, run_scan saves the
 *           checkpoint and returns.
 *   Errors: none (only an async-signal-safe store).
 */
static void on_sigint(int sig)
{
    (void)sig;
    g_stop = 1;
}

/*
 * announce_locked: print CONFIRMED lines for solutions that are now below the frontier.
 *   In:     nothing; the caller holds g_mu.
 *   Out:    the number of solutions below the frontier.  Each is announced once
 *           (if g_print), with the range [START, frontier position) that is complete.
 *   Errors: none.
 */
static int announce_locked(void)
{
    int n = 0;
    for (int i = 0; i < S.nhits; i++) {
        hit_t *h = &S.hits[i];
        if (h->chunk >= S.frontier) continue;
        n++;
        if (!h->announced) {
            h->announced = true;
            if (g_print) {
                if (stderr_tty) fputs("\r\033[K", stderr);
                const char *a = anum_for(h->b);
                printf("CONFIRMED base %u p %" PRIu64 "%s%s%s (all primes in [%" PRIu64 ", %" PRIu64 ") tested)\n",
                       h->b, h->p, a ? " (" : "", a ? a : "", a ? ")" : "",
                       S.start, S.start + S.frontier * S.chunk);
                fflush(stdout);
            }
        }
    }
    return n;
}

/* What run_scan did in this session (a resumed part is not included). */
typedef struct {
    double seconds;
    u64 primes, tests, numbers;
} scan_stats_t;

/*
 * run_scan: test every prime in [start, end) for the bases in B (sections 3, 11).
 *   In:     start < end; chunk >= 1; threads >= 1; state = checkpoint path or NULL;
 *           interval = seconds between checkpoints (and status lines when stderr is not
 *           a terminal); quiet suppresses the status line and notes.  The bases must be
 *           set (parse_bases / bases_setup).
 *   Out:    the global S holds the complete result (totals, checksums, solutions,
 *           frontier); the checkpoint, if any, is saved at the end.  Returns this
 *           session's time, primes, tests and numbers covered.  END is rounded up to a
 *           chunk boundary (S.end).
 *   Errors: dies if start >= end, chunk = 0, the rounded END exceeds 2^63 or 64 bits,
 *           memory or a thread cannot be allocated, or the checkpoint is unusable
 *           (load_state).  SIGINT/SIGTERM end the scan early with a saved checkpoint;
 *           g_stop stays set, so later scans in the same process (selftest) return at
 *           once.
 */
static scan_stats_t run_scan(u64 start, u64 end, u64 chunk, int threads, const char *state,
                             int interval, bool quiet)
{
    if (end <= start) die("END must be larger than START");
    if (chunk == 0) die("CHUNK must be positive");
    S.start = start;
    S.chunk = chunk;
    S.nchunks = (end - start + chunk - 1) / chunk;          /* if the sum wraps, it is < chunk - 1 and nchunks = 0 */
    if (S.nchunks == 0 || (S.nchunks - 1) > (UINT64_MAX - start) / chunk) die("range too large");
    S.end = start + S.nchunks * chunk;
    if (S.end > PMAX || S.end < start) die("END must be <= 2^63 (after rounding up to the chunk size)");
    if (S.end != end && !quiet)
        fprintf(stderr, "note: END rounded up to the chunk boundary %" PRIu64 "\n", S.end);
    S.frontier = 0;
    S.primes = S.tests = S.near = S.flt = 0;
    memset(S.cks, 0, sizeof S.cks);
    memset(S.slot, 0, sizeof S.slot);
    S.nhits = 0;
    atomic_store(&S.tests_live, 0);

    if (state && load_state(state) && !quiet)
        fprintf(stderr, "resuming from %s: %" PRIu64 " of %" PRIu64 " chunks done (position %" PRIu64
                        ", %" PRIu64 " primes tested)\n",
                state, S.frontier, S.nchunks, S.start + S.frontier * S.chunk, S.primes);
    const u64 frontier0 = S.frontier, primes0 = S.primes, tests0 = S.tests;
    atomic_store(&S.next, S.frontier);
    if (!quiet) {
        fprintf(stderr, "scanning [%" PRIu64 ", %" PRIu64 ") in %" PRIu64 " chunks of %" PRIu64
                        " with %d threads, bases ", S.start, S.end, S.nchunks, S.chunk, threads);
        print_bases(stderr);
        fputc('\n', stderr);
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_sigint;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    pthread_t *th = calloc((size_t)threads, sizeof *th);
    if (!th) die("out of memory");
    for (int i = 0; i < threads; i++)
        if (pthread_create(&th[i], NULL, worker, NULL) != 0) die("pthread_create failed");

    enum { NS = 128 };      /* rate window: one sample per second, ~2 minutes deep */
    double ts[NS];
    u64 ps[NS], qs[NS];     /* position and live test count at each sample */
    int ns = 1;
    const double t0 = now();
    const u64 pos0 = S.start + frontier0 * S.chunk;
    double last_status = 0, last_ckpt = t0, last_sample = t0;
    ts[0] = t0; ps[0] = pos0; qs[0] = 0;
    char b1[32], b2[32];

    for (;;) {                                  /* monitor: status, CONFIRMED lines, checkpoints */
        usleep(200000);
        double t = now();
        pthread_mutex_lock(&g_mu);
        u64 f = S.frontier, primes = S.primes, flt = S.flt, near = S.near;
        int nconf = announce_locked(), nhits = S.nhits;
        pthread_mutex_unlock(&g_mu);
        u64 pos = S.start + f * S.chunk;
        if (pos > S.end) pos = S.end;
        if (f >= S.nchunks || g_stop) break;
        u64 tl = atomic_load(&S.tests_live);
        if (t - last_sample >= 1.0) {
            last_sample = t;
            if (ns < NS) { ts[ns] = t; ps[ns] = pos; qs[ns] = tl; ns++; }
            else {
                memmove(ts, ts + 1, (NS - 1) * sizeof *ts);
                memmove(ps, ps + 1, (NS - 1) * sizeof *ps);
                memmove(qs, qs + 1, (NS - 1) * sizeof *qs);
                ts[NS - 1] = t; ps[NS - 1] = pos; qs[NS - 1] = tl;
            }
        }
        double dt = t - ts[0];
        double rate = dt > 0.5 ? (double)(pos - ps[0]) / dt : 0;    /* numbers per second */
        double trate = dt > 0.5 ? (double)(tl - qs[0]) / dt : 0;    /* tests per second */
        double eta = rate > 0 ? (double)(S.end - pos) / rate : -1;
        bool show = stderr_tty ? (t - last_status >= 1.0) : (t - last_status >= interval);
        if (show && !quiet) {
            last_status = t;
            fprintf(stderr, "%s%s  pos %.6g (%.2f%%)  %.3g/s  %.3g tests/s  %.4g primes  ETA %s  found %d (%d confirmed)",
                    stderr_tty ? "\r\033[K" : "progress: ", fmt_dur(t - t0, b1, sizeof b1), (double)pos,
                    100.0 * (double)(pos - S.start) / (double)(S.end - S.start), rate, trate,
                    (double)primes, fmt_dur(eta, b2, sizeof b2), nhits, nconf);
            if (g_near > 0) fprintf(stderr, "  near %" PRIu64, near);
            if (flt) fprintf(stderr, "  FLT-ERRORS %" PRIu64, flt);
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
    if (g_stop && S.frontier < S.nchunks && !quiet)
        fprintf(stderr, "stopped at position %" PRIu64 " (%" PRIu64 " of %" PRIu64 " chunks complete)%s\n",
                S.start + S.frontier * S.chunk, S.frontier, S.nchunks, state ? ", checkpoint saved" : "");
    free(th);
    scan_stats_t st;
    st.seconds = t1 - t0;
    st.primes = S.primes - primes0;
    st.tests = S.tests - tests0;
    st.numbers = (S.frontier - frontier0) * S.chunk;
    return st;
}

/*
 * print_summary: the result of a scan on stdout.
 *   In:     nothing (reads S and B after run_scan).
 *   Out:    "SUMMARY [START, position) primes N tests N flt_errors N [near(...) N]" and
 *           one line per base with its A-number, checksum and the solutions below the
 *           frontier.  The position is END for a complete scan.
 *   Errors: none.
 */
static void print_summary(void)
{
    u64 pos = S.start + S.frontier * S.chunk;
    if (pos > S.end) pos = S.end;
    printf("SUMMARY [%" PRIu64 ", %" PRIu64 ") primes %" PRIu64 " tests %" PRIu64 " flt_errors %" PRIu64,
           S.start, pos, S.primes, S.tests, S.flt);
    if (g_near > 0) printf(" near(|A|<=%lld) %" PRIu64, g_near, S.near);
    putchar('\n');
    for (int j = 0; j < B.nb; j++) {
        const char *a = anum_for(B.b[j]);
        printf("base %3u %-8s cks %016" PRIx64 "  solutions:", B.b[j], a ? a : "", S.cks[j]);
        int n = 0;
        for (int i = 0; i < S.nhits; i++)
            if (S.hits[i].b == B.b[j] && S.hits[i].chunk < S.frontier) { printf(" %" PRIu64, S.hits[i].p); n++; }
        if (!n) printf(" none");
        putchar('\n');
    }
    fflush(stdout);
}

/* ------------------------------------------------------------------ */
/* Commands                                                             */
/* ------------------------------------------------------------------ */

/*
 * usage: print the command summary and exit.
 *   In:     nothing.
 *   Out:    does not return; exit status 2.
 *   Errors: none.
 */
static void usage(void)
{
    fputs("usage: wieferich scan [START] END [-b BASES] [-t T] [-c CHUNK] [-n T] [-S FILE] [-i SECS] [-L FILE] [-q]\n"
          "       wieferich check P [-b BASES]\n"
          "       wieferich bench [N] [-d SPAN] [-b BASES] [-t T]\n"
          "       wieferich selftest [LIMIT] [-t T]\n"
          "BASES: list/ranges like 3,5,7 or 2-30; 'oeis' (default) = 3,5,6,7,10,12,13,14,15,17,18,19,20,22,23,26,30;\n"
          "       'known' = all non-power bases 2..30\n", stderr);
    exit(2);
}

/*
 * cmd_scan: "scan [START] END [options]".
 *   In:     the arguments after "scan" (see the header's Usage).
 *   Out:    runs the scan, prints FOUND/NEAR/FLT-ERROR/CONFIRMED lines and the summary on
 *           stdout and a timing line on stderr.  Exit status 0, or 3 if any FLT error
 *           occurred (including errors recorded in a resumed checkpoint).
 *   Errors: usage() for unknown options, a missing END or more than two positional
 *           numbers; dies for bad numbers or bases, an unwritable chunk log, and every
 *           run_scan error.  -n values above 2^63 - 1 are treated as 2^63 - 1.
 */
static int cmd_scan(int argc, char **argv)
{
    u64 pos[2];
    int npos = 0, threads = default_threads(), interval = 60;
    u64 chunk = 1000000000ULL;
    const char *state = NULL, *logf = NULL, *bases = "oeis";
    bool quiet = false;
    for (int i = 0; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-t") && i + 1 < argc) threads = parse_int(argv[++i], 1, 1024, "threads");
        else if (!strcmp(a, "-c") && i + 1 < argc) chunk = parse_num(argv[++i]);
        else if (!strcmp(a, "-b") && i + 1 < argc) bases = argv[++i];
        else if (!strcmp(a, "-n") && i + 1 < argc) {
            u64 v = parse_num(argv[++i]);
            g_near = v > (u64)LLONG_MAX ? LLONG_MAX : (long long)v;
        }
        else if (!strcmp(a, "-S") && i + 1 < argc) state = argv[++i];
        else if (!strcmp(a, "-L") && i + 1 < argc) logf = argv[++i];
        else if (!strcmp(a, "-i") && i + 1 < argc) interval = parse_int(argv[++i], 1, 86400, "interval");
        else if (!strcmp(a, "-q")) quiet = true;
        else if (a[0] != '-' && npos < 2) pos[npos++] = parse_num(a);
        else usage();
    }
    if (npos == 0) usage();
    u64 start = npos == 2 ? pos[0] : 0, end = npos == 2 ? pos[1] : pos[0];
    parse_bases(bases);
    if (logf && !(g_log = fopen(logf, "a"))) die("cannot open %s: %s", logf, strerror(errno));
    scan_stats_t st = run_scan(start, end, chunk, threads, state, interval, quiet);
    print_summary();
    if (!quiet) {
        char b1[32];
        fprintf(stderr, "%s this session, %.3g primes/s, %.3g tests/s\n", fmt_dur(st.seconds, b1, sizeof b1),
                st.seconds > 0 ? (double)st.primes / st.seconds : 0, st.seconds > 0 ? (double)st.tests / st.seconds : 0);
    }
    if (g_log) fclose(g_log);
    return S.flt ? 3 : 0;
}

/*
 * cmd_check: "check P [-b BASES]": Fermat quotients of one number, two ways.
 *   In:     the arguments after "check"; P odd with 3 <= P < 2^63; bases default "known".
 *   Out:    "p = P (prime | NOT prime ...)" and per base: q, A, a WIEFERICH marker for
 *           q = 0, and whether the production kernel (fermat_quotient) agrees with the
 *           shift-and-add reference (powmod_ref): residue and quotient for prime P, the
 *           residue b^(P-1) mod P^2 only for composite P.  Bases divisible by P are
 *           reported as such.  Exit status 0 if all agree, 1 otherwise.
 *   Errors: usage() for a missing P or unknown options; dies for an even P or one
 *           outside [3, 2^63), and for bad bases.  For composite P the quotients are
 *           computed but meaningless (the program says so).
 */
static int cmd_check(int argc, char **argv)
{
    u64 p = 0;
    const char *bases = "known";
    bool havep = false;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "-b") && i + 1 < argc) bases = argv[++i];
        else if (argv[i][0] != '-' && !havep) { p = parse_num(argv[i]); havep = true; }
        else usage();
    }
    if (!havep) usage();
    if (p < 3 || p >= PMAX || !(p & 1)) die("P must be odd and in [3, 2^63)");
    parse_bases(bases);
    const bool prime = is_prime64(p);
    printf("p = %" PRIu64 " (%s)\n", p, prime ? "prime" : "NOT prime: quotients are meaningless");
    int bad = 0;
    for (int j = 0; j < B.nb; j++) {
        u32 b = B.b[j];
        if (b % p == 0) { printf("base %u: p divides b\n", b); continue; }
        u128 rf;
        u64 q = fermat_quotient(p, j, &rf);
        u128 rr = powmod_ref(b, p - 1, (u128)p * p);
        u64 qr = (u64)((rr - 1) / p);
        long long A = q <= p / 2 ? (long long)q : -(long long)(p - q);
        /* For composite p, r is not 1 mod p, so the exact-division quotient q and the
         * floor quotient qr differ legitimately; only the residues are compared then. */
        bool ok = rf == rr && (!prime || q == qr);
        if (!ok) bad++;
        printf("base %3u: q = %" PRIu64 "  A = %lld%s  [reference %s]\n", b, q, A,
               q == 0 ? "  <-- WIEFERICH" : "", ok ? "agrees" : "DISAGREES");
    }
    return bad ? 1 : 0;
}

/*
 * li_diff: the logarithmic-integral difference li(y) - li(x) = integral of dt / ln t.
 *   In:     1 < x < y.
 *   Out:    the integral by the midpoint rule on 2000 steps in log t; an estimate of the
 *           number of primes in [x, y] (bench projections).
 *   Errors: meaningless for x <= 1 (ln t <= 0); cmd_bench only calls it for x >= 1000.
 */
static double li_diff(double x, double y)
{
    const int n = 2000;
    double lx = log(x), ly = log(y), h = (ly - lx) / n, s = 0;
    for (int i = 0; i < n; i++) {
        double u = lx + (i + 0.5) * h;          /* t = e^u, dt / ln t = e^u / u du */
        s += exp(u) / u * h;
    }
    return s;
}

/*
 * cmd_bench: "bench [N] [-d SPAN] [-b BASES] [-t T]": time a short scan and project.
 *   In:     the arguments after "bench"; defaults N = 10^15, SPAN = 10^10, bases "oeis",
 *           all cores.
 *   Out:    the measured primes/s, tests/s (total and per thread) and numbers/s for
 *           [N, N+SPAN), and for [N, 2N), [N, 10N), [N, 100N) the number of tests, the
 *           projected time at this rate and the heuristic expected number of new
 *           solutions (section 2; assumes all bases start at N).  Exit status 0.
 *   Errors: usage() for unknown options; dies for bad numbers or bases and for every
 *           run_scan error (e.g. N + SPAN > 2^63, SPAN = 0).  Projections are printed
 *           only for N >= 1000.  The chunk size is SPAN / (8 T), at least 10^6.
 */
static int cmd_bench(int argc, char **argv)
{
    u64 n0 = 1000000000000000ULL, span = 10000000000ULL;
    int threads = default_threads();
    const char *bases = "oeis";
    bool haven = false;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "-d") && i + 1 < argc) span = parse_num(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) threads = parse_int(argv[++i], 1, 1024, "threads");
        else if (!strcmp(argv[i], "-b") && i + 1 < argc) bases = argv[++i];
        else if (argv[i][0] != '-' && !haven) { n0 = parse_num(argv[i]); haven = true; }
        else usage();
    }
    parse_bases(bases);
    g_print = true;
    u64 chunk = span / (u64)(threads * 8);
    if (chunk < 1000000) chunk = 1000000;
    if (n0 + span < n0) die("N + SPAN does not fit in 64 bits");
    scan_stats_t st = run_scan(n0, n0 + span, chunk, threads, NULL, 60, true);
    if (st.seconds <= 0) st.seconds = 1e-9;
    double tps = (double)st.tests / st.seconds, pps = (double)st.primes / st.seconds;
    printf("bench [%" PRIu64 ", +%" PRIu64 "): %" PRIu64 " primes x %d bases in %.2f s with %d threads\n",
           n0, span, st.primes, B.nb, st.seconds, threads);
    printf("  %.4g primes/s, %.4g tests/s (%.4g tests/s per thread), %.4g numbers/s\n",
           pps, tps, tps / threads, (double)st.numbers / st.seconds);
    if (n0 < 1000 || tps <= 0) return 0;
    const double targets[] = {2, 10, 100};
    for (int i = 0; i < 3; i++) {
        double x = (double)n0, y = x * targets[i];
        double tests = li_diff(x, y) * B.nb;
        double exp_new = B.nb * log(log(y) / log(x));
        char b1[32];
        printf("  [N, %gN): %.3g tests, ~%s at this rate; heuristic %.3f new solutions over %d bases\n",
               targets[i], tests, fmt_dur(tests / tps, b1, sizeof b1), exp_new, B.nb);
    }
    return 0;
}

/* ---- selftest (section 12) ---- */

static u64 rng_state = 0x243F6A8885A308D3ULL;  /* fixed seed: the selftest is reproducible */

/*
 * rnd: the selftest's pseudo-random numbers (splitmix64).
 *   In:     nothing (advances rng_state).
 *   Out:    the next 64-bit value.
 *   Errors: none.
 */
static u64 rnd(void) { return mix64(rng_state += GOLD); }

/*
 * test_arith: the arithmetic part of the selftest.
 *   In:     nothing; replaces the global bases (B) with 64 test bases.
 *   Out:    the number of failures (0 = pass), after a summary line on stderr:
 *           * 200,000 random odd moduli p < 2^63 (all bit lengths): pinv, ninv and one
 *             are checked, and msmall, msqr and redc on a random x < N are compared with
 *             mulmod_ref (Montgomery results are converted back by multiplying with
 *             one = R mod N);
 *           * edge cases x = N - 1 with c = 2^32 - 1 for small and near-2^63 moduli;
 *           * ~3,000 Fermat quotients of random primes < 2^63 with random bases,
 *             among them 2 (w = 5) and 4294967291 (w = 1), through fermat_quotient and
 *             powmod_ref.
 *   Errors: prints the first failures on stderr (stops each loop after 5 failures).
 */
static int test_arith(void)
{
    int bad = 0;
    for (int it = 0; it < 200000 && bad < 5; it++) {
        int bits = 2 + (int)(rnd() % 62);                 /* p in [2^(bits-1), 2^bits) */
        u64 p = (rnd() >> (64 - bits)) | (1ULL << (bits - 1)) | 1;
        if (p >= PMAX || p < 3) continue;
        pctx_t c;
        pctx_init(&c, p);
        if (p * c.pinv != 1 || (u64)c.N * c.ninv != UINT64_MAX) { bad++; fprintf(stderr, "inverse p=%" PRIu64 "\n", p); continue; }
        if (c.one != ((u128)0 - c.N) % c.N) { bad++; fprintf(stderr, "one p=%" PRIu64 "\n", p); continue; }
        u128 x = (((u128)rnd() << 64) | rnd()) % c.N;
        u64 k = rnd() >> (rnd() % 64);                    /* constants of every size */
        u64 cst = k & 0xFFFFFFFFULL;
        if (msmall(x, cst, &c) != mulmod_ref(x, cst, c.N)) { bad++; fprintf(stderr, "msmall p=%" PRIu64 "\n", p); }
        if (mulmod_ref(msqr(x, &c), c.one, c.N) != mulmod_ref(x, x, c.N)) { bad++; fprintf(stderr, "msqr p=%" PRIu64 "\n", p); }
        if (mulmod_ref(redc(x, &c), c.one, c.N) != x) { bad++; fprintf(stderr, "redc p=%" PRIu64 "\n", p); }
    }
    /* worst cases: x = N-1, largest constants, largest p */
    const u64 edge[] = {3, 5, 4294967291ULL, 4294967311ULL, 18446744073709551557ULL >> 1 | 1, PMAX - 25};
    for (size_t i = 0; i < sizeof edge / sizeof *edge; i++) {
        u64 p = edge[i];
        pctx_t c;
        pctx_init(&c, p);
        u128 x = c.N - 1;
        if (msmall(x, 0xFFFFFFFFULL, &c) != mulmod_ref(x, 0xFFFFFFFFULL, c.N)) { bad++; fprintf(stderr, "edge msmall %" PRIu64 "\n", p); }
        if (mulmod_ref(msqr(x, &c), c.one, c.N) != mulmod_ref(x, x, c.N)) { bad++; fprintf(stderr, "edge msqr %" PRIu64 "\n", p); }
    }
    /* full Fermat quotients for random primes and bases */
    u32 bl[MAXB];
    int nbl = 0;
    for (u32 b = 2; nbl < MAXB; b += 1 + (u32)(rnd() % 97)) bl[nbl++] = b;
    bl[nbl - 1] = 4294967291U;                          /* a base with w = 1 */
    bases_setup(bl, nbl);
    int nq = 0;
    for (int it = 0; it < 3000 && bad < 5; it++) {
        int bits = 3 + (int)(rnd() % 61);
        u64 p = (rnd() >> (64 - bits)) | (1ULL << (bits - 1)) | 1;
        if (p >= PMAX) continue;
        while (!is_prime64(p)) p += 2;
        if (p >= PMAX) continue;
        int j = (int)(rnd() % (u64)B.nb);
        if (B.b[j] % p == 0) continue;
        u128 rf;
        u64 q = fermat_quotient(p, j, &rf);
        u128 rr = powmod_ref(B.b[j], p - 1, (u128)p * p);
        if (rf != rr || q != (u64)((rr - 1) / p)) {
            bad++;
            fprintf(stderr, "quotient mismatch p=%" PRIu64 " b=%u\n", p, B.b[j]);
        }
        nq++;
    }
    fprintf(stderr, "arithmetic: 200000 random reductions + edge cases, %d Fermat quotients vs reference: %s\n",
            nq, bad ? "FAIL" : "ok");
    return bad;
}

/*
 * known_hit: is p a known solution for base b (table KNOWN)?
 *   In:     b, p.
 *   Out:    true iff listed.
 *   Errors: none.
 */
static bool known_hit(u32 b, u64 p)
{
    for (int i = 0; i < NKNOWN; i++)
        if (KNOWN[i].b == b)
            for (int k = 0; k < KNOWN[i].n; k++) if (KNOWN[i].p[k] == p) return true;
    return false;
}

/* Checksums from verify_wieferich.py (Python integers, Miller-Rabin primes), bases
 * 'oeis' in ascending order: {lo, hi, primes in [lo, hi), checksum per base}. */
typedef struct { u64 lo, hi, primes; u64 cks[17]; } ckwin_t;
static const ckwin_t CKWIN[] = {
#include "wieferich_cks.h"
};

/*
 * cmd_selftest: "selftest [LIMIT] [-t T]" (section 12).
 *   In:     the arguments after "selftest"; LIMIT default 10^9.
 *   Out:    one line per part on stderr and "selftest passed" or "FAILED"; exit status
 *           0 or 1.  The pi check is made only for LIMIT = 10^9 (pi(10^9) = 50847534).
 *   Errors: usage() for unknown options; dies for a bad LIMIT and every run_scan error
 *           (e.g. LIMIT = 0 or > 2^63).  Ctrl-C makes the remaining scans return at
 *           once and the selftest fail.
 */
static int cmd_selftest(int argc, char **argv)
{
    u64 limit = 1000000000ULL;
    int threads = default_threads();
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "-t") && i + 1 < argc) threads = parse_int(argv[++i], 1, 1024, "threads");
        else if (argv[i][0] != '-') limit = parse_num(argv[i]);
        else usage();
    }
    int bad = test_arith();
    g_print = false;

    /* 1. all bases 2..30 over [0, limit) against the known solutions */
    parse_bases("known");
    double t = now();
    u64 chunk = limit / (u64)(threads * 16);
    if (chunk < 1000000) chunk = 1000000;
    run_scan(0, limit, chunk, threads, NULL, 60, true);
    int found = 0, missing = 0, extra = 0;
    for (int i = 0; i < S.nhits; i++) {
        if (known_hit(S.hits[i].b, S.hits[i].p)) found++;
        else { extra++; fprintf(stderr, "  unexpected solution base %u p %" PRIu64 "\n", S.hits[i].b, S.hits[i].p); }
    }
    for (int i = 0; i < NKNOWN; i++)
        for (int k = 0; k < KNOWN[i].n; k++) {
            if (KNOWN[i].p[k] >= S.end) continue;
            bool f = false;
            for (int h = 0; h < S.nhits; h++) if (S.hits[h].b == KNOWN[i].b && S.hits[h].p == KNOWN[i].p[k]) f = true;
            if (!f) { missing++; fprintf(stderr, "  missing base %u p %" PRIu64 "\n", KNOWN[i].b, KNOWN[i].p[k]); }
        }
    bool pi_ok = limit != 1000000000ULL || S.primes == 50847534ULL;
    fprintf(stderr, "scan [0, %" PRIu64 "), %d bases: %" PRIu64 " primes%s, %d known solutions found, %d missing, %d unexpected, %" PRIu64 " FLT errors (%.1f s)\n",
            S.end, B.nb, S.primes, limit == 1000000000ULL ? (pi_ok ? " (= pi(10^9))" : " (WRONG, pi(10^9) = 50847534)") : "",
            found, missing, extra, S.flt, now() - t);
    bad += missing + extra + (int)S.flt + !pi_ok;

    /* 2. windows around the known solutions above the scan */
    t = now();
    const u64 scanned = S.end;                  /* run_scan below overwrites S.end */
    int wins = 0, wbad = 0;
    for (int i = 0; i < NKNOWN; i++)
        for (int k = 0; k < KNOWN[i].n; k++) {
            u64 p = KNOWN[i].p[k];
            if (p < scanned) continue;
            u32 b = KNOWN[i].b;
            bases_setup(&b, 1);
            run_scan(p > 1000000 ? p - 1000000 : 0, p + 1000000, 200000, threads, NULL, 60, true);
            bool ok = S.nhits == 1 && S.hits[0].p == p && S.flt == 0;
            if (!ok) { wbad++; fprintf(stderr, "  window around base %u p %" PRIu64 ": %d hits\n", b, p, S.nhits); }
            wins++;
        }
    fprintf(stderr, "windows: %d known solutions above %" PRIu64 " re-found through the scan: %s (%.1f s)\n",
            wins, limit, wbad ? "FAIL" : "ok", now() - t);
    bad += wbad;

    /* 3. checksums against verify_wieferich.py */
    t = now();
    parse_bases("oeis");
    int cbad = 0;
    for (size_t w = 0; w < sizeof CKWIN / sizeof *CKWIN; w++) {
        const ckwin_t *c = &CKWIN[w];
        run_scan(c->lo, c->hi, (c->hi - c->lo) / 16, threads, NULL, 60, true);
        bool ok = S.primes == c->primes && S.flt == 0 && S.end == c->hi;
        for (int j = 0; j < B.nb; j++) if (S.cks[j] != c->cks[j]) ok = false;
        if (!ok) {
            cbad++;
            fprintf(stderr, "  checksum window [%" PRIu64 ", %" PRIu64 "): primes %" PRIu64 " (expected %" PRIu64 ")\n",
                    c->lo, c->hi, S.primes, c->primes);
            for (int j = 0; j < B.nb; j++)
                if (S.cks[j] != c->cks[j]) fprintf(stderr, "    base %u: %016" PRIx64 " expected %016" PRIx64 "\n", B.b[j], S.cks[j], c->cks[j]);
        }
    }
    fprintf(stderr, "checksums: %zu windows x %d bases vs verify_wieferich.py: %s (%.1f s)\n",
            sizeof CKWIN / sizeof *CKWIN, B.nb, cbad ? "FAIL" : "ok", now() - t);
    bad += cbad;
    fprintf(stderr, "selftest %s\n", bad ? "FAILED" : "passed");
    return bad ? 1 : 0;
}

/*
 * main: dispatch to scan, check, bench or selftest.
 *   In:     argv[1] = command, the rest its arguments.
 *   Out:    the command's exit status.
 *   Errors: usage() (exit 2) for a missing or unknown command.
 */
int main(int argc, char **argv)
{
    stderr_tty = isatty(2);
    if (argc < 2) usage();
    const char *cmd = argv[1];
    if (!strcmp(cmd, "scan")) return cmd_scan(argc - 2, argv + 2);
    if (!strcmp(cmd, "check")) return cmd_check(argc - 2, argv + 2);
    if (!strcmp(cmd, "bench")) return cmd_bench(argc - 2, argv + 2);
    if (!strcmp(cmd, "selftest")) return cmd_selftest(argc - 2, argv + 2);
    usage();
    return 2;
}
