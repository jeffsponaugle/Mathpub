/*
 * a350826_cuda.cu - count prime sextuplets (p, p+4, p+6, p+10, p+12, p+16) on an NVIDIA GPU
 *
 * Purpose
 * =======
 * CUDA version of ../a350826.c for the OEIS sequences A350826 (sextuplets by number of digits
 * of the initial member p) and A063501 (pi_6(10^n)).  It uses the same wheel classes, chunks,
 * per-bin counts and checksums, chunk-log lines and checkpoint format as the CPU tool, so runs
 * can be compared chunk by chunk (same range, wheel, bound, chunk size and bins; see
 * `a350826 recheck`) and a checkpoint written by one tool can be resumed by the other.  On two
 * DGX Sparks (GB10, sm_121) it confirmed a(18) = 3439443854 and a(19) = 24711327817 and
 * computed a(20) = 180637585474 (Oct 2026; README.md).
 *
 * Theory of operation
 * ===================
 * The mathematics is that of a350826.c (sections 1-7 and 9 of its header): the admissible
 * residues, the wheel with its C(w) classes, one progression p = p0 + M k per class, the phase
 * phi = -p0 M^-1 mod q of each sieving prime with the class-independent offsets
 * K_j = -d_j M^-1 mod q, base-2 strong probable-prime tests followed by a strong Lucas test
 * (BPSW), why the count is exact, and the order-independent checksums.  This file differs in
 * how the work is mapped onto the GPU.
 *
 * 1. Launches.  A chunk (D.chunk consecutive classes, at most G.cap = 65536) is one launch.
 *    The host computes the residues of its classes (class_r) and copies them to the device.
 *    A launch is cut into sub-launches of G.sub classes, chosen so that their expected number
 *    of survivors (with a 1.3x margin) fits the survivor buffer of G.bcap = 2^25 entries.
 *    Each sub-launch runs three kinds of kernels on one stream:
 *      sext_kernel   one thread block (NT = 512 threads) per class: sieve the class and append
 *                    its survivors (initial members p) to the survivor buffer;
 *      round_kernel  six times, m = 0..5: keep the p whose member p + OFF[m] passes the base-2
 *                    strong test, moving them between two ping-pong buffers;
 *      cand_kernel   strong Lucas test of all six members of each remaining candidate, then
 *                    count and checksum it in its bin.
 *    All sub-launches of a launch add into one Res record (counts and checksums per bin,
 *    statistics, SPSP cases), which is copied back to the host and folded into the totals.
 *
 * 2. Sieving one class (sext_kernel).  The class is processed in segments of SEGB = 2^18 bits
 *    (the last one may be shorter), each held as a bitmap in shared memory (32 KB; 1 = killed).
 *    The sieving primes w < q <= B are split into three groups, each handled in the way that
 *    suits its size:
 *      pattern primes  q < pmax (-p, default 256; at most MAXPAT = 64 primes): the 64-bit word
 *                      patterns T_q[s] of the CPU tool (section 5 of its header).  Thread t
 *                      builds the WPT = 8 words [8t, 8t+8) of the segment in registers as the
 *                      OR of T_q[(64 j - phi) mod q] over these primes and stores them, which
 *                      also initialises the bitmap;
 *      medium primes   pmax <= q < QW (-Q, default 4096; at most 2048 primes): one warp per
 *                      prime at a time; lane l marks the rounds l-1, l+31, l+63, ... (a round is
 *                      the six positions phi + t q + {0, K_1, ..., K_5} for one t).  Each lane
 *                      keeps the phases of its own primes in registers, and __shfl_sync hands
 *                      them to the warp when that prime is marked;
 *      large primes    QW <= q <= B (at most NT * MAXLPT = 24576 primes): one thread per prime,
 *                      the round loop of the CPU tool's mark_prime(), phases in registers (the
 *                      template parameter LPT = primes per thread is 12, 24 or 48, whichever
 *                      is the smallest that suffices).
 *    Marks are 32-bit atomicOr operations on shared memory, since threads share words.  After
 *    a barrier every thread takes the clear bits of its WPT words (bits beyond the segment
 *    end masked off); a warp-wide prefix sum and one atomicAdd per warp reserve room in the
 *    survivor buffer, where each survivor is stored as its initial member p (two u64
 *    words).  For the next segment the phases move on by SEGB: phi = (phi - SEGB mod q) mod q
 *    for pattern and medium primes, the end position of the round loop for large primes.
 *
 * 3. Testing (round_kernel, cand_kernel).  Breadth first, as on the CPU: round m reads the
 *    survivors kept by round m - 1 and keeps those whose member p + OFF[m] is a base-2 strong
 *    probable prime, one thread per number, with one atomicAdd per warp for the output
 *    positions.  About half drop out per round, so few numbers reach cand_kernel, which
 *    runs the strong Lucas test on all six members.  A member that fails it is a base-2
 *    strong pseudoprime (SPSP): the candidate is counted in res->spsp, recorded (the first
 *    MAXSPSP per launch) and not counted as a sextuplet.  Otherwise the sextuplet is added
 *    to the count and checksum of its bin (binary search over the boundaries).  Testing in
 *    separate kernels with one thread per number, instead of inside the sieve kernel, made
 *    the tool about 2.7 times faster.
 *
 * 4. Arithmetic.  Montgomery multiplication throughout: R = 2^64 below 2^64 (d_mmul64, the
 *    "subtracting" form of a350826.c); for base-2 tests of members in [2^64, 2^96), R = 2^96
 *    with three 32-bit limbs (d_mmul96, CIOS method: the GPU's integer multiplier is 32 bits
 *    wide, and 64-bit products take several steps); for the Lucas tests above 2^64 and any
 *    test above 2^96, R = 2^128 with two 64-bit limbs (d_mmul128).  The Montgomery form of 1,
 *    R mod n, is computed without a slow wide division: a floating-point estimate of the
 *    quotient, lowered so that it cannot be too large, is corrected exactly (d_r64, d_r96,
 *    d_r128).
 *
 * 5. Survivor-buffer overflow.  If a sub-launch finds more survivors than the buffer holds,
 *    the extra ones are dropped and res->ovf is set.  The host then discards that launch's
 *    results and redoes the chunk in 8 pieces with 1/8 of the sub-launch size
 *    (redo_in_pieces); an overflow even then is a fatal error.  The sub-launch sizing makes
 *    this rare (it is not exercised by the selftest).
 *
 * 6. Host driver (run_count).  Two CUDA streams, each with its own buffers, keep two launches
 *    in flight while the host prepares the next.  The host waits for them in chunk order and
 *    folds each into the totals, so "chunks [0, frontier) are done" holds exactly as in the
 *    CPU tool, and the checkpoint (-S) and chunk log (-L) mean the same.  Initial members
 *    below 2^20 are tested on the host (small_region) and added to chunk 0.  SIGINT and
 *    SIGTERM stop the launching of new chunks; launches in flight finish and are folded.
 *
 * 7. Limits.  As in a350826.c (HI <= 2^80, B <= 2^20, 7 <= w <= 47, MAXBIN bins), and further:
 *    at most 64 pattern, 2048 medium and 24576 large primes (with the default -Q, B up to
 *    about 2.9e5; 2^18 always fits), chunks of at most 65536 classes, the survivors of one
 *    class must fit the buffer (2^25), at most MAXSPSP SPSP cases printed per chunk (all are
 *    counted).  Device memory: about 2 GiB, almost all of it the four survivor buffers
 *    (512 MiB each).
 *
 * 8. Errors and exit status.  As in a350826.c: die() (message, exit status 1) for invalid
 *    arguments, inconsistent checkpoints, failed allocations, any CUDA error (CK), a class too
 *    long for the wheel, too many medium or large primes, and an unrecoverable survivor-buffer
 *    overflow; usage() exits with status 2.  Exit status: count 0 if complete, 1 if stopped
 *    early (PARTIAL results); selftest 1 if any test fails; bench, num and mrbench 0.
 *
 * 9. Build-time geometry (-D...): NT threads per block (512), MINB blocks per SM for
 *    __launch_bounds__ (2), SEGB segment bits (2^18; SEGB/64 must be a multiple of NT).  The
 *    grids of the test kernels (48 * 8 and 48 * 4 blocks) suit the 48 SMs of GB10.
 *    -DNOATOMIC replaces the atomic ORs by plain ones, for timing experiments only (the
 *    results are then wrong).
 *
 * Code map
 * ========
 *   host helpers (parsing, hashing) -> reference arithmetic (host) -> wheel, sieving primes
 *   and patterns (host tables) -> device arithmetic (64-, 128- and 96-bit Montgomery, base-2
 *   and Lucas tests) -> kernels (sext_kernel, round_kernel, cand_kernel, test_kernel) -> host
 *   driver (device buffers, checkpoints, launches, folding, run_count) -> parameters and
 *   commands -> selftest and debug commands -> main.
 *
 * Usage
 * =====
 *   a350826_cuda count LO HI [-b B1,B2,..] [-w W] [-B B] [-p P] [-Q QW] [-c CH]
 *                            [-C C0:C1 | -P I/N] [-S FILE] [-i SECS] [-L FILE] [-q] [-d DEV]
 *   a350826_cuda bench LO HI [-n N] [-w W] [-B B] [-p P] [-Q QW] [-d DEV] [-X MASK]
 *   a350826_cuda selftest [N] [-d DEV]
 *   a350826_cuda num X [X ...]          (debug) GPU base-2 and Lucas tests of single numbers
 *   a350826_cuda mrbench X [N]          (debug) round_kernel throughput on N random odd
 *                                       numbers near X (N <= 2^25, default 2^24)
 * The options mean the same as for a350826.c (there is no -t: the GPU does all the work;
 * -s is replaced by the build-time SEGB).  -p pattern bound (default 256), -Q bound between
 * medium and large primes (4096), -d CUDA device, -n sample classes for bench (20000).
 * -X MASK (for bench only; with count the results are wrong) skips parts of the work to time
 * the rest: 1 tests, 2 patterns, 4 medium primes, 8 large primes, 16 survivor output.
 * The selftest checks the GPU arithmetic against the host reference, A350826(1..N) (default
 * N = 14) against the OEIS data, and the reference windows of verify_a350826.py.
 *
 * Build:  nvcc -O3 -std=c++17 -arch=sm_121 -o a350826_cuda a350826_cuda.cu   (or `make`)
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cstdarg>
#include <cctype>
#include <cmath>
#include <csignal>
#include <cerrno>
#include <climits>
#include <cinttypes>
#include <ctime>
#include <unistd.h>
#include <cuda_runtime.h>

typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int64_t i64;
typedef unsigned __int128 u128;

#define GOLD    0x9E3779B97F4A7C15ULL  /* 2^64 / golden ratio: odd constant of splitmix64 */
#define MAXBIN  256             /* most bins per run (-b gives at most MAXBIN - 1 boundaries) */
#define MAXSPSP 1024            /* SPSP cases recorded per launch (all are counted) */
#define P0      ((u64)1 << 20)  /* initial members below P0 are tested on the host */
#define BMAX    ((u32)1 << 20)  /* largest sieving bound accepted by -B (must not exceed P0) */
#define HIMAX   ((u128)1 << 80) /* largest HI (the host reference test is deterministic far beyond) */

/* kernel geometry (override with -D at build time) */
#ifndef NT
#define NT      512                 /* threads per block */
#endif
#ifndef MINB
#define MINB    2                   /* blocks per SM (launch bounds) */
#endif
#ifndef SEGB
#define SEGB    (1u << 18)          /* bits per segment (dynamic shared memory) */
#endif
#define NWARP   (NT / 32)
#define SEGW    (SEGB / 64)         /* u64 words per segment */
#define WPT     (SEGW / NT)         /* words per thread in the pattern pass */
#if SEGW % NT
#error "SEGB/64 must be a multiple of NT"
#endif
#define MAXPAT  64                  /* most pattern primes (their phases are kept in shared memory) */
#define MAXMPL  4                   /* medium primes per lane (NWARP * 32 * MAXMPL total) */
#define MAXLPT  48                  /* large primes per thread (NT * MAXLPT total) */

/* The six offsets of the pattern: p + OFF[m], m = 0..5, are the members of a sextuplet. */
static const u32 OFF[6] = {0, 4, 6, 10, 12, 16};

static bool stderr_tty;                     /* stderr is a terminal: rewrite the status line in place */
static volatile sig_atomic_t g_stop = 0;    /* set by SIGINT/SIGTERM: launch no more chunks */

/* CK: check the result of a CUDA runtime call; any error is fatal (die() with the CUDA message
   and the source line).  Errors inside kernels show up at the next synchronising call. */
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) die("CUDA error %s at %s:%d", cudaGetErrorString(e_), __FILE__, __LINE__); } while (0)

/* ------------------------------------------------------------------ */
/* Host helpers (as in a350826.c)                                      */
/* ------------------------------------------------------------------ */

/* now: seconds from a fixed point of the monotonic clock (status lines, rates, timings). */
static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

/*
 * die: report a fatal error and end the program.
 *   In:     printf-style format and arguments.
 *   Out:    does not return; clears the status line on a terminal, prints the message and a
 *           newline to stderr, exit(1).  A checkpoint on disk stays valid (it is replaced
 *           atomically).
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
 * fmt_dur: format a duration: "3d04h05m", "1h02m03s", "4m05s" or "12.3s"; negative, NaN or
 * absurd (> 1e12 s) values, such as an unknown ETA, give "?".
 *   In:     s = seconds; buf, n = caller's buffer and its size.   Out: buf.
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
 * u128s: decimal string of an unsigned 128-bit number (printf has no conversion for u128).
 *   Out:    pointer to a static buffer from a ring of 16, so up to 16 results can be used in one
 *           printf call; each buffer is overwritten 16 calls later.  Host only, single thread.
 */
static const char *u128s(u128 x)
{
    static char ring[16][44];
    static int k = 0;
    char *b = ring[k++ & 15], t[44];
    int n = 0;
    do { t[n++] = (char)('0' + (int)(x % 10)); x /= 10; } while (x);
    for (int i = 0; i < n; i++) b[i] = t[n - 1 - i];
    b[n] = 0;
    return b;
}

/* mul_ok: true and *r = a*b if the product fits in 128 bits, false (and *r unchanged) if not. */
static bool mul_ok(u128 a, u128 b, u128 *r)
{
    if (a && b > ~(u128)0 / a) return false;
    *r = a * b;
    return true;
}

/* all_digits: true iff s[0 .. n) is non-empty and consists of decimal digits only. */
static bool all_digits(const char *s, size_t n)
{
    if (n == 0) return false;
    for (size_t i = 0; i < n; i++)
        if (!isdigit((unsigned char)s[i])) return false;
    return true;
}

/*
 * parse_term: parse one term of a number given on the command line (as in a350826.c).
 *   In:     s[0 .. n) (need not be NUL-terminated).  Accepts plain decimal, powers "B^E"
 *           ("2^64") and scientific notation "MeE" with an optional fractional mantissa
 *           ("1.5e19") when the value is an integer.
 *   Out:    true and *out = the value.
 *   Errors: false for an empty or over-long term (80 characters or more), misplaced
 *           characters, a non-integer value or a value above 2^128 - 1.
 */
static bool parse_term(const char *s, size_t n, u128 *out)
{
    char buf[80], *c;
    if (n == 0 || n >= sizeof buf) return false;
    memcpy(buf, s, n);
    buf[n] = 0;
    if ((c = strchr(buf, '^'))) {
        *c = 0;
        if (!all_digits(buf, strlen(buf)) || !all_digits(c + 1, strlen(c + 1))) return false;
        u128 b = strtoull(buf, 0, 10), r = 1;
        u64 e = strtoull(c + 1, 0, 10);
        while (e--) if (!mul_ok(r, b, &r)) return false;
        *out = r;
        return true;
    }
    if ((c = strpbrk(buf, "eE"))) {
        *c = 0;
        char *dot = strchr(buf, '.');
        size_t ip = dot ? (size_t)(dot - buf) : strlen(buf);
        const char *frac = dot ? dot + 1 : "";
        if ((ip && !all_digits(buf, ip)) || (*frac && !all_digits(frac, strlen(frac)))) return false;
        if (!all_digits(c + 1, strlen(c + 1))) return false;
        long e = strtol(c + 1, 0, 10) - (long)strlen(frac);
        u128 m = 0;
        for (size_t i = 0; i < ip; i++) { if (!mul_ok(m, 10, &m)) return false; m += (u128)(buf[i] - '0'); }
        for (const char *f = frac; *f; f++) { if (!mul_ok(m, 10, &m)) return false; m += (u128)(*f - '0'); }
        for (; e > 0; e--) if (!mul_ok(m, 10, &m)) return false;
        for (; e < 0; e++) { if (m % 10) return false; m /= 10; }
        *out = m;
        return true;
    }
    if (!all_digits(buf, n)) return false;
    u128 m = 0;
    for (size_t i = 0; i < n; i++) { if (!mul_ok(m, 10, &m)) return false; m += (u128)(buf[i] - '0'); }
    *out = m;
    return true;
}

/*
 * parse_num: parse a number: one term, or "X+Y" / "X-Y" of two terms, split at the last '+' or
 * '-' that is not part of an exponent ("1e18+2e12", "2^64-1e12").
 *   Errors: die() for a malformed term, an overflowing sum or a negative difference.
 */
static u128 parse_num(const char *s)
{
    size_t n = strlen(s);
    for (size_t i = n; i-- > 1; ) {
        if (s[i] != '+' && s[i] != '-') continue;
        if (s[i - 1] == 'e' || s[i - 1] == 'E') continue;
        u128 a, b;
        if (!parse_term(s, i, &a) || !parse_term(s + i + 1, n - i - 1, &b)) die("bad number '%s'", s);
        if (s[i] == '+') {
            if (a + b < a) die("overflow in '%s'", s);
            return a + b;
        }
        if (b > a) die("negative result in '%s'", s);
        return a - b;
    }
    u128 v;
    if (!parse_term(s, n, &v)) die("bad number '%s'", s);
    return v;
}

/* parse_u64: parse_num() restricted to [lo, hi] for an option value; die() naming `what` if
   the number is malformed or out of range. */
static u64 parse_u64(const char *s, u64 lo, u64 hi, const char *what)
{
    u128 v = parse_num(s);
    if (v < lo || v > hi) die("%s must be between %" PRIu64 " and %" PRIu64, what, lo, hi);
    return (u64)v;
}

/*
 * mix64: one splitmix64 step (add GOLD, then the finalizer), a bijective 64-bit mixing
 * function.  The checksums depend on it: it must stay identical to d_mix64() below, to
 * a350826.c and to verify_a350826.py.
 */
static inline u64 mix64(u64 z)
{
    z += GOLD;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* hashp: checksum contribution of a sextuplet start p (added mod 2^64 per bin, so the checksum
   does not depend on the order in which sextuplets are found). */
static inline u64 hashp(u128 p) { return mix64((u64)p ^ ((u64)(p >> 64) * GOLD)); }

/* ---- reference primality (host; slow and simple, for the small region and the selftest) ---- */

/*
 * mulmod_ref: a*b mod n for 1 <= n < 2^127: one 128-bit product for n < 2^64, otherwise
 * double-and-add over the bits of b (a + a cannot overflow since n < 2^127).
 */
static u128 mulmod_ref(u128 a, u128 b, u128 n)
{
    if (!(n >> 64)) return ((a % n) * (b % n)) % n;
    u128 r = 0;
    a %= n;
    while (b) {
        if (b & 1) { r += a; if (r >= n) r -= n; }
        a += a; if (a >= n) a -= n;
        b >>= 1;
    }
    return r;
}

/* powmod_ref: a^e mod n (1 <= n < 2^127) by binary exponentiation with mulmod_ref. */
static u128 powmod_ref(u128 a, u128 e, u128 n)
{
    u128 r = 1 % n;
    a %= n;
    while (e) {
        if (e & 1) r = mulmod_ref(r, a, n);
        a = mulmod_ref(a, a, n);
        e >>= 1;
    }
    return r;
}

/*
 * mr_ref: strong probable-prime (Miller-Rabin) test of n to base a.
 *   In:     odd n > a, n < 2^127.
 *   Out:    true if n is a strong probable prime to base a (every prime is), false if a is a
 *           witness that n is composite.
 */
static bool mr_ref(u128 n, u64 a)
{
    u128 d = n - 1;
    int s = 0;
    while (!(d & 1)) { d >>= 1; s++; }
    u128 x = powmod_ref(a, d, n);
    if (x == 1 || x == n - 1) return true;
    for (int i = 1; i < s; i++) {
        x = mulmod_ref(x, x, n);
        if (x == n - 1) return true;
        if (x == 1) return false;
    }
    return false;
}

/*
 * is_prime_ref: deterministic primality test for n < 2^81: trial division by the primes up to
 * 41, then strong tests to the 13 prime bases 2..41, which prove primality below
 * 3317044064679887385961981 (Sorenson and Webster 2015).
 *   Out:    true iff n is prime.
 *   Errors: die() for n >= 2^81, where the test would no longer be a proof.
 */
static bool is_prime_ref(u128 n)
{
    static const u32 sp[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41};
    if (n < 2) return false;
    for (int i = 0; i < 13; i++) {
        if (n == sp[i]) return true;
        if (n % sp[i] == 0) return false;
    }
    if (n < 43 * 43) return true;
    if (n >= HIMAX * 2) die("is_prime_ref: argument too large");
    for (int i = 0; i < 13; i++) if (!mr_ref(n, sp[i])) return false;
    return true;
}

/* is_sext_ref: true iff p, p+4, p+6, p+10, p+12 and p+16 are all prime (p + 16 < 2^81). */
static bool is_sext_ref(u128 p)
{
    for (int i = 0; i < 6; i++) if (!is_prime_ref(p + OFF[i])) return false;
    return true;
}

/* ------------------------------------------------------------------ */
/* Wheel, sieving primes, patterns (host; as in a350826.c)             */
/* ------------------------------------------------------------------ */

/* The wheel (a350826.c, theory section 3).  SMALLP: possible wheel primes; MAXWD bounds the
   number of wheel primes >= 11 (11 of them for w = 47). */
#define MAXWD 12
static const u32 SMALLP[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47};

static struct {
    u32 w;                  /* largest wheel prime */
    u64 M, C;               /* modulus (product of the primes <= w), number of classes */
    int nd;                 /* wheel primes > 7 */
    u32 q[MAXWD], na[MAXWD];  /* wheel primes >= 11 and their numbers of allowed residues (q - 6) */
    u8 a[MAXWD][48];        /* allowed residues mod q, increasing */
    u64 E[MAXWD];           /* CRT idempotents: E = 1 (mod q), 0 (mod M/q) */
    u64 rad[MAXWD];         /* mixed-radix weights of the class index (last prime least significant) */
    u64 r0;                 /* 97 * E_210 mod M: every class is 97 mod 210 */
} W;

/* invmod: a^-1 mod m (2 <= m < 2^63) by the extended Euclidean algorithm; die() if
   gcd(a, m) != 1 (a bug: all callers pass coprime values). */
static u64 invmod(u64 a, u64 m)
{
    i64 t = 0, nt = 1;
    i64 r = (i64)m, nr = (i64)(a % m);
    while (nr) {
        i64 qq = r / nr, x;
        x = t - qq * nt; t = nt; nt = x;
        x = r - qq * nr; r = nr; nr = x;
    }
    if (r != 1) die("invmod: not invertible");
    return (u64)(t < 0 ? t + (i64)m : t);
}

/* wheel_ok: true iff w is one of the primes 7, 11, ..., 47 allowed as largest wheel prime. */
static bool wheel_ok(u32 w)
{
    if (w < 7 || w > 47) return false;
    for (int i = 0; i < 15; i++) if (SMALLP[i] == w) return true;
    return false;
}

/*
 * wheel_setup: build the wheel with largest prime w (global W), as in a350826.c: M, the allowed
 * residues mod each wheel prime >= 11, the CRT idempotents, r0 = 97 E_210 and the mixed-radix
 * weights of the class index.
 *   Errors: die() if w is not one of 7, 11, ..., 47.
 */
static void wheel_setup(u32 w)
{
    if (!wheel_ok(w)) die("wheel prime must be one of 7, 11, 13, ..., 47");
    memset(&W, 0, sizeof W);
    W.w = w;
    W.M = 1;
    for (int i = 0; i < 15 && SMALLP[i] <= w; i++) W.M *= SMALLP[i];
    W.C = 1;
    for (int i = 4; i < 15 && SMALLP[i] <= w; i++) {
        const u32 q = SMALLP[i];
        const int j = W.nd++;
        W.q[j] = q;
        for (u32 a = 0; a < q; a++) {
            bool ok = true;
            for (int k = 0; k < 6; k++) if ((a + OFF[k]) % q == 0) ok = false;
            if (ok) W.a[j][W.na[j]++] = (u8)a;
        }
        W.C *= W.na[j];
        const u64 Mq = W.M / q;
        W.E[j] = (u64)((u128)Mq * invmod(Mq % q, q) % W.M);
    }
    const u64 M210 = W.M / 210;
    const u64 E210 = (u64)((u128)M210 * invmod(M210 % 210, 210) % W.M);
    W.r0 = (u64)((u128)97 * E210 % W.M);
    u64 rad = 1;
    for (int j = W.nd - 1; j >= 0; j--) { W.rad[j] = rad; rad *= W.na[j]; }
}

/*
 * class_r: residue r mod M of class number c in [0, W.C): r = 97 E_210 + sum_j a_j E_j mod M,
 * where a_j is allowed residue number (c / rad_j) mod na_j of wheel prime j.  As c runs over
 * [0, C), r runs over all admissible residues once each.  (The u128 sum cannot overflow.)
 */
static inline u64 class_r(u64 c)
{
    u128 r = W.r0;
    for (int j = 0; j < W.nd; j++) r += (u128)W.a[j][(c / W.rad[j]) % W.na[j]] * W.E[j];
    return (u64)(r % W.M);
}

/* small_primes: malloc'ed array of the primes <= n (sieve of Eratosthenes), *count = their
   number; the caller frees it.  die() if memory runs out. */
static u32 *small_primes(u32 n, int *count)
{
    u8 *s = (u8 *)calloc(n + 1, 1);
    if (!s) die("out of memory");
    int c = 0;
    for (u32 i = 2; i <= n; i++) {
        if (s[i]) continue;
        c++;
        for (u64 j = (u64)i * i; j <= n; j += i) s[j] = 1;
    }
    u32 *p = (u32 *)malloc(sizeof *p * (size_t)(c ? c : 1));
    if (!p) die("out of memory");
    c = 0;
    for (u32 i = 2; i <= n; i++) if (!s[i]) p[c++] = i;
    free(s);
    *count = c;
    return p;
}

/* cmp_u32: qsort comparator for u32 values, ascending. */
static int cmp_u32(const void *a, const void *b)
{
    u32 x = *(const u32 *)a, y = *(const u32 *)b;
    return x < y ? -1 : x > y;
}

/* Host copies of the device tables (tables_setup; uploaded by dev_upload).  Per sieving prime i
   (w < q <= B, increasing): q[i]; k[j * np + i] = K_(j+1), j = 0..4 (structure of arrays, so
   neighbouring threads read neighbouring words); minv[i] = M^-1 mod q; bmod[i] = base mod q;
   segmod[i] = SEGB mod q (phase change per full segment).  Per pattern prime i: the q words of
   T_q at tab[poff[i]], pstep[i] = 64 mod q (change of s per word), pstept[i] = 64 WPT mod q
   (per thread). */
static struct {
    u32 B, patmax, qw;      /* sieving bound, pattern bound (-p), medium/large bound (-Q) */
    int np, npat, nmed, nlarge;   /* primes in all, and in the three groups (in this order) */
    u32 *q, *k, *minv, *bmod, *segmod, *pstep, *pstept, *poff;   /* k: 5 * np, SoA */
    u64 *tab;               /* pattern tables, ntab words */
    size_t ntab;
} SP;

/*
 * tables_setup: build all host tables for sieving bound B (global SP); needs wheel_setup.
 *   In:     B with W.w < B <= 2^20; patmax = pattern bound; qw = bound between medium and large
 *           primes; base = max(LO, 2^20), the first number handled by the GPU.
 *   Out:    SP as described above.  The primes are split into the three groups of the theory
 *           (section 2): pattern primes q < patmax (at most MAXPAT, the smallest ones), medium
 *           primes q < qw, large primes the rest.
 *   Errors: die() if B is out of range, memory runs out, there are more than
 *           NWARP * 32 * MAXMPL = 2048 medium primes (lower -Q) or more than NT * MAXLPT = 24576
 *           large primes (lower -B).
 *   Notes:  the pattern primes are among the first 64 primes above w (all below 410), so the
 *           local array bad[1024] is large enough for any -p.
 */
static void tables_setup(u32 B, u32 patmax, u32 qw, u128 base)
{
    if (B <= W.w || B > BMAX) die("sieving bound must be in (w, 2^20]");
    int n;
    u32 *pr = small_primes(B, &n);
    free(SP.q); free(SP.k); free(SP.minv); free(SP.bmod); free(SP.segmod); free(SP.pstep); free(SP.pstept);
    free(SP.poff); free(SP.tab);
    SP.B = B; SP.patmax = patmax; SP.qw = qw;
    SP.q = (u32 *)malloc(sizeof(u32) * n);
    SP.k = (u32 *)malloc(sizeof(u32) * 5 * n);
    SP.minv = (u32 *)malloc(sizeof(u32) * n);
    SP.bmod = (u32 *)malloc(sizeof(u32) * n);
    SP.segmod = (u32 *)malloc(sizeof(u32) * n);
    SP.pstep = (u32 *)malloc(sizeof(u32) * MAXPAT);
    SP.pstept = (u32 *)malloc(sizeof(u32) * MAXPAT);
    SP.poff = (u32 *)malloc(sizeof(u32) * MAXPAT);
    if (!SP.q || !SP.k || !SP.minv || !SP.bmod || !SP.segmod || !SP.pstep || !SP.pstept || !SP.poff) die("out of memory");
    int np = 0;
    for (int i = 0; i < n; i++) if (pr[i] > W.w) np++;
    SP.np = np;
    u32 *kt = (u32 *)malloc(sizeof(u32) * 5 * np);
    if (!kt) die("out of memory");
    np = 0;
    for (int i = 0; i < n; i++) {
        const u32 q = pr[i];
        if (q <= W.w) continue;
        const u32 mi = (u32)invmod(W.M % q, q);
        u32 c[5];                             /* K = -d M^-1 mod q for d = 4, 6, 10, 12, 16 */
        for (int k = 1; k < 6; k++) c[k - 1] = (u32)((q - (u64)OFF[k] * mi % q) % q);
        qsort(c, 5, sizeof *c, cmp_u32);
        SP.q[np] = q;
        for (int k = 0; k < 5; k++) kt[k * SP.np + np] = c[k];
        SP.minv[np] = mi;
        SP.bmod[np] = (u32)(base % q);
        SP.segmod[np] = SEGB % q;
        np++;
    }
    memcpy(SP.k, kt, sizeof(u32) * 5 * SP.np);
    free(kt);
    free(pr);
    /* classes of primes */
    SP.npat = 0;
    while (SP.npat < SP.np && SP.q[SP.npat] < patmax && SP.npat < MAXPAT) SP.npat++;
    int m = SP.npat;
    while (m < SP.np && SP.q[m] < qw) m++;
    SP.nmed = m - SP.npat;
    SP.nlarge = SP.np - m;
    if (SP.nmed > NWARP * 32 * MAXMPL) die("too many medium primes (raise MAXMPL or lower QW)");
    if (SP.nlarge > NT * MAXLPT) die("too many large primes for B = %u (max %d)", B, NT * MAXLPT);
    /* pattern tables: T_q[s] has bit i set iff (s + i) mod q is a bad offset {0, K_1..K_5} */
    size_t words = 0;
    for (int i = 0; i < SP.npat; i++) words += SP.q[i];
    SP.tab = (u64 *)malloc(sizeof(u64) * (words ? words : 1));
    if (!SP.tab) die("out of memory");
    SP.ntab = words;
    size_t off = 0;
    for (int i = 0; i < SP.npat; i++) {
        const u32 q = SP.q[i];
        u8 bad[1024] = {0};
        bad[0] = 1;
        for (int j = 0; j < 5; j++) bad[SP.k[j * SP.np + i]] = 1;
        for (u32 s = 0; s < q; s++) {
            u64 w = 0;
            for (u32 b = 0; b < 64; b++) if (bad[(s + b) % q]) w |= 1ULL << b;
            SP.tab[off + s] = w;
        }
        SP.poff[i] = (u32)off;
        SP.pstep[i] = 64 % q;
        SP.pstept[i] = (u32)(((u64)64 * WPT) % q);
        off += q;
    }
}

/* ------------------------------------------------------------------ */
/* Device code                                                         */
/* ------------------------------------------------------------------ */

/* Run parameters in constant memory (P, written by dev_upload).  With span = HI - 1 - base =
   Q0 M + rem0, a class whose first term is base + delta has Q0 + 1 terms if delta <= rem0 and
   Q0 terms otherwise, so the GPU needs no 128-bit division.  u128 values are split into 64-bit
   halves (lo, hi). */
struct DevP {
    u64 M, bmM, Q0, rem0, base_lo, base_hi;   /* modulus, base mod M, span / M, span % M, base */
    int nbins, npat, nmed, nlarge, np;   /* bins; sizes of the prime groups and in all */
    int skip;               /* bench ablation: 1 tests, 2 patterns, 4 medium, 8 large, 16 survivor output */
    u64 bnd_lo[MAXBIN + 1], bnd_hi[MAXBIN + 1];   /* bin boundaries bnd[0] = LO .. bnd[nbins] = HI */
};

struct Cnt {                    /* per sub-launch buffer counters */
    unsigned long long nb[7];   /* survivors in the buffer before round m (nb[0] from the sieve) */
};                              /* (nb[6]: candidates for cand_kernel; zeroed per sub-launch) */

/* Results of one launch (one chunk), accumulated by all its kernels with atomics: per bin the
   count and checksum, and the statistics of acc_t; SPSP cases (member n, candidate p, as
   low/high u64 halves) for the host's SPSP lines. */
struct Res {
    unsigned long long cnt[MAXBIN], cks[MAXBIN];
    unsigned long long surv, cand, spsp, bits;
    unsigned int ovf;           /* survivor buffer overflow: redo the launch in pieces */
    unsigned int nsp;           /* SPSP cases found (only the first MAXSPSP are recorded) */
    unsigned long long sp_n[MAXSPSP][2], sp_p[MAXSPSP][2];
};

/* Device pointers to the tables of SP (same layout), passed to sext_kernel by value. */
struct DevT {
    const u32 *q, *k, *minv, *bmod, *segmod, *pstep, *pstept, *poff;
    const u64 *tab;
};

__constant__ DevP P;                            /* run parameters */
__constant__ u32 OFFD[6] = {0, 4, 6, 10, 12, 16};   /* OFF on the device */

/* ---- device arithmetic: 64-bit Montgomery (R = 2^64), as in a350826.c ---- */

/* d_inv64: n^-1 mod 2^64 for odd n (Newton iteration from x = n, correct to 3 bits; five steps
   give 96 >= 64 bits). */
__device__ __forceinline__ u64 d_inv64(u64 n)
{
    u64 x = n;
    #pragma unroll
    for (int i = 0; i < 5; i++) x *= 2 - n * x;
    return x;
}

/*
 * d_mmul64: Montgomery product a*b/2^64 mod n.
 *   In:     a, b < n; odd n < 2^64; ni = n^-1 mod 2^64 (d_inv64, the positive inverse).
 *   Out:    a*b*2^-64 mod n in [0, n).  "Subtracting" form: with m = lo(ab) ni mod 2^64, the
 *           value hi(ab) - hi(mn) lies in (-n, n); a negative one is fixed by adding n.
 */
__device__ __forceinline__ u64 d_mmul64(u64 a, u64 b, u64 n, u64 ni)
{
    const u64 tl = a * b, th = __umul64hi(a, b);
    const u64 m = tl * ni;
    const u64 mh = __umul64hi(m, n);
    const u64 r = th - mh;
    return th < mh ? r + n : r;
}

/* d_madd64, d_msub64, d_mhalf64: (a + b), (a - b) and a/2 mod n for 0 <= a, b < n, odd n
   (d_madd64 also handles the wrap of a + b past 2^64; halving an odd a gives (a + n)/2). */
__device__ __forceinline__ u64 d_madd64(u64 a, u64 b, u64 n) { u64 s = a + b; return (s < a || s >= n) ? s - n : s; }
__device__ __forceinline__ u64 d_msub64(u64 a, u64 b, u64 n) { return a >= b ? a - b : a - b + n; }
__device__ __forceinline__ u64 d_mhalf64(u64 a, u64 n) { return (a & 1) ? (a >> 1) + (n >> 1) + 1 : a >> 1; }

/*
 * d_r64: 2^64 mod n, the Montgomery form of 1.
 *   In:     odd n > 1.   Out: 2^64 mod n, in [0, n).
 *   Method: for n >= 2^56, f = floor(2^64/n) < 256 is estimated in float (error < 1);
 *           k = max(1, f_est - 1) <= f, r = 2^64 - k n (computed mod 2^64), then n is
 *           subtracted until r < n (at most twice).  Smaller n use the slower 64-bit remainder
 *           (0 - n) % n.
 */
__device__ __forceinline__ u64 d_r64(u64 n)
{
    if (n >> 56) {
        const u64 ke = (u64)(1.8446744e19f / (float)n);
        const u64 k = ke > 2 ? ke - 1 : 1;
        u64 r = 0 - k * n;
        while (r >= n) r -= n;
        return r;
    }
    return (u64)(0 - n) % n;
}

/*
 * d_sprp2_64: base-2 strong probable-prime test for odd n, 2 < n < 2^64 (as sprp2_64 in
 * a350826.c).
 *   Out:    true for primes and base-2 strong pseudoprimes, false if n is certainly composite.
 *   Method: n - 1 = d 2^s; 2^d mod n left to right in Montgomery form (square, and double with
 *           an addition for a 1 bit), then up to s - 1 squarings looking for -1.
 */
__device__ bool d_sprp2_64(u64 n)
{
    const u64 ni = d_inv64(n);
    const u64 one = d_r64(n), mone = n - one;
    u64 d = n - 1;
    const int s = __ffsll((long long)d) - 1;
    d >>= s;
    int b = 63 - __clzll((long long)d);
    u64 x = d_madd64(one, one, n);
    while (b-- > 0) {
        x = d_mmul64(x, x, n, ni);
        if ((d >> b) & 1) x = d_madd64(x, x, n);
    }
    if (x == one || x == mone) return true;
    for (int i = 1; i < s; i++) {
        x = d_mmul64(x, x, n, ni);
        if (x == mone) return true;
        if (x == one) return false;
    }
    return false;
}

/* ---- strong Lucas test (Selfridge parameters), as in a350826.c ---- */

/* d_jacobi: Jacobi symbol (a/n) for odd n >= 1: -1, 0 or +1 (0 iff gcd(a, n) > 1 for n > 1). */
__device__ int d_jacobi(i64 a, u128 n)
{
    int t = 1;
    u128 x;
    if (a < 0) {
        x = (u128)(-(a + 1)) + 1;
        if ((n & 3) == 3) t = -t;
    } else x = (u128)a;
    x %= n;
    while (x) {
        while (!(x & 1)) {
            x >>= 1;
            unsigned r = (unsigned)(n & 7);
            if (r == 3 || r == 5) t = -t;
        }
        u128 tmp = x; x = n; n = tmp;
        if ((x & 3) == 3 && (n & 3) == 3) t = -t;
        x %= n;
    }
    return n == 1 ? t : 0;
}

/* d_is_square: true iff n (< 2^126) is a perfect square (double sqrt, corrected by +-1 steps). */
__device__ bool d_is_square(u128 n)
{
    u128 r = (u128)sqrt((double)n);
    while (r * r > n) r--;
    while ((r + 1) * (r + 1) <= n) r++;
    return r * r == n;
}

/*
 * d_selfridge: first D in 5, -7, 9, -11, ... with (D/n) = -1 (Selfridge's method A).
 *   In:     odd n > 1.
 *   Out:    1 and *Dout = D; or 0 if n is certainly composite: a perfect square (the search
 *           would not end), or gcd(D, n) > 1 for some D with |D| != n.
 */
__device__ int d_selfridge(u128 n, i64 *Dout)
{
    if (d_is_square(n)) return 0;
    i64 D = 5;
    for (;;) {
        int j = d_jacobi(D, n);
        if (j == -1) break;
        if (j == 0 && (u128)(D < 0 ? -D : D) != n) return 0;
        D = D > 0 ? -(D + 2) : -(D - 2);
    }
    *Dout = D;
    return 1;
}

/*
 * d_mulsmall64: v * x mod n by double-and-add over the bits of v.
 *   In:     x < n, n odd; any v >= 0 (callers pass the small |Q| and |D|).   Out: v x mod n.
 *   Makes the Montgomery forms v R mod n of Q and D (x = R mod n) without a wide division.
 */
__device__ u64 d_mulsmall64(u64 v, u64 x, u64 n)
{
    u64 r = 0;
    for (int b = 63 - __clzll((long long)(v | 1)); b >= 0; b--) {
        r = d_madd64(r, r, n);
        if ((v >> b) & 1) r = d_madd64(r, x, n);
    }
    return r;
}

/*
 * d_slprp64: strong Lucas probable-prime test with P = 1, Q = (1 - D)/4 (as slprp64 in
 * a350826.c), the Lucas half of BPSW.
 *   In:     odd n > 13, n < 2^64 (members of candidates: no small factors, passed base 2).
 *   Out:    true if n is a strong Lucas probable prime (every prime is), false if n is
 *           certainly composite (squares and numbers sharing a factor with D included).
 *   Method: n + 1 = d 2^s; U_d, V_d and Q^d left to right in Montgomery form (doubling, and for
 *           a 1 bit the step k -> k + 1 with halving mod n); n passes if U_d = 0 or
 *           V_(d 2^r) = 0 for some 0 <= r < s.
 */
__device__ bool d_slprp64(u64 n)
{
    i64 D;
    if (!d_selfridge(n, &D)) return false;
    const i64 Q = (1 - D) / 4;
    const u64 ni = d_inv64(n), one = d_r64(n);
    const u64 Qa = Q >= 0 ? d_mulsmall64((u64)Q, one, n) : n - d_mulsmall64((u64)(-Q), one, n);
    const u64 Qm = Qa == n ? 0 : Qa;
    const u64 Dm = D >= 0 ? d_mulsmall64((u64)D, one, n) : n - d_mulsmall64((u64)(-D), one, n);
    u64 d = n + 1;
    const int s = __ffsll((long long)d) - 1;
    d >>= s;
    u64 U = one, V = one, Qk = Qm;
    for (int b = 62 - __clzll((long long)d); b >= 0; b--) {
        U = d_mmul64(U, V, n, ni);
        V = d_msub64(d_mmul64(V, V, n, ni), d_madd64(Qk, Qk, n), n);
        Qk = d_mmul64(Qk, Qk, n, ni);
        if ((d >> b) & 1) {
            const u64 U2 = d_madd64(U, V, n);
            const u64 V2 = d_madd64(d_mmul64(Dm, U, n, ni), V, n);
            U = d_mhalf64(U2, n);
            V = d_mhalf64(V2, n);
            Qk = d_mmul64(Qk, Qm, n, ni);
        }
    }
    if (U == 0 || V == 0) return true;
    for (int r = 1; r < s; r++) {
        V = d_msub64(d_mmul64(V, V, n, ni), d_madd64(Qk, Qk, n), n);
        if (V == 0) return true;
        Qk = d_mmul64(Qk, Qk, n, ni);
    }
    return false;
}

/* ---- 128-bit Montgomery (R = 2^128): Lucas tests above 2^64, base-2 tests above 2^96 ---- */

/*
 * d_mmul128: Montgomery product a*b/2^128 mod n with two 64-bit limbs (as mmul128 in
 * a350826.c: schoolbook product, two word-by-word reduction steps, one final subtraction).
 *   In:     a, b < n; odd n < 2^127; ni = -n^-1 mod 2^64 (note the sign).
 *   Out:    a*b*2^-128 mod n, in [0, n).
 */
__device__ __forceinline__ u128 d_mmul128(u128 a, u128 b, u128 n, u64 ni)
{
    const u64 a0 = (u64)a, a1 = (u64)(a >> 64), b0 = (u64)b, b1 = (u64)(b >> 64);
    const u64 n0 = (u64)n, n1 = (u64)(n >> 64);
    const u128 p00 = (u128)a0 * b0, p01 = (u128)a0 * b1, p10 = (u128)a1 * b0, p11 = (u128)a1 * b1;
    const u64 t0 = (u64)p00;
    const u128 mid = (p00 >> 64) + (u64)p01 + (u64)p10;
    const u64 t1 = (u64)mid;
    const u128 hi = (mid >> 64) + (p01 >> 64) + (p10 >> 64) + p11;
    const u64 m = t0 * ni;
    const u128 q0 = (u128)m * n0, q1 = (u128)m * n1;
    const u128 s1 = (u128)t1 + (q0 >> 64) + (u64)q1 + (t0 != 0);
    const u64 u1 = (u64)s1;
    const u128 hi2 = hi + (s1 >> 64) + (q1 >> 64);
    const u64 m2 = u1 * ni;
    const u128 r0 = (u128)m2 * n0, r1 = (u128)m2 * n1;
    const u128 s2 = (u128)(u64)hi2 + (r0 >> 64) + (u64)r1 + (u1 != 0);
    const u128 top = (hi2 >> 64) + (s2 >> 64) + (r1 >> 64);
    const u128 res = (top << 64) | (u64)s2;
    return res >= n ? res - n : res;
}

/* d_madd128, d_msub128, d_mhalf128: as the 64-bit versions, for 0 <= a, b < n < 2^127. */
__device__ __forceinline__ u128 d_madd128(u128 a, u128 b, u128 n) { u128 s = a + b; return s >= n ? s - n : s; }
__device__ __forceinline__ u128 d_msub128(u128 a, u128 b, u128 n) { return a >= b ? a - b : a - b + n; }
__device__ __forceinline__ u128 d_mhalf128(u128 a, u128 n) { return (a & 1) ? (a >> 1) + (n >> 1) + 1 : a >> 1; }

/* d_r128: 2^128 mod n (the Montgomery form of 1 for R = 2^128), in [0, n), for odd n with
   2^64 < n < 2^100.  The quotient Q = floor(2^128/n) < 2^64 is estimated in
   double (error < 2^13) and lowered by 2^21, so q <= Q and r = 2^128 - q n lies in
   [0, (2^21 + 2^13 + 1) n); one more double estimate brings r below 4n. */
__device__ u128 d_r128(u128 n)
{
    const double inv = 1.0 / (double)n;
    const double qd = 3.402823669209385e38 * inv;
    u64 q = qd >= 1.8446744073709552e19 ? ~0ULL : (u64)qd;
    q = q > (1ULL << 21) ? q - (1ULL << 21) : 1;
    u128 r = (u128)0 - (u128)q * n;
    u64 k = (u64)((double)r * inv);
    k = k > 2 ? k - 2 : 0;
    r -= (u128)k * n;
    while (r >= n) r -= n;
    return r;
}

/* d_sprp2_128: base-2 strong probable-prime test (as d_sprp2_64) with R = 2^128, for odd n with
   2^64 < n < 2^100 (the range of d_r128).  Used by d_sprp2 only for n >= 2^96. */
__device__ bool d_sprp2_128(u128 n)
{
    const u64 ni = 0 - d_inv64((u64)n);
    const u128 one = d_r128(n), mone = n - one;
    u128 d = n - 1;
    int s = 0;
    while (!(d & 1)) { d >>= 1; s++; }
    int b = 127;
    while (!((d >> b) & 1)) b--;
    u128 x = d_madd128(one, one, n);
    while (b-- > 0) {
        x = d_mmul128(x, x, n, ni);
        if ((d >> b) & 1) x = d_madd128(x, x, n);
    }
    if (x == one || x == mone) return true;
    for (int i = 1; i < s; i++) {
        x = d_mmul128(x, x, n, ni);
        if (x == mone) return true;
        if (x == one) return false;
    }
    return false;
}

/* d_mulsmall128: v * x mod n by double-and-add (as d_mulsmall64), for x < n < 2^127. */
__device__ u128 d_mulsmall128(u64 v, u128 x, u128 n)
{
    u128 r = 0;
    for (int b = 63 - __clzll((long long)(v | 1)); b >= 0; b--) {
        r = d_madd128(r, r, n);
        if ((v >> b) & 1) r = d_madd128(r, x, n);
    }
    return r;
}

/* d_slprp128: strong Lucas probable-prime test (as d_slprp64) with R = 2^128, for odd n with
   2^64 < n < 2^100 (the range of d_r128); members of candidates are below 2^81. */
__device__ bool d_slprp128(u128 n)
{
    i64 D;
    if (!d_selfridge(n, &D)) return false;
    const i64 Q = (1 - D) / 4;
    const u64 ni = 0 - d_inv64((u64)n);
    const u128 one = d_r128(n);
    const u128 Qa = Q >= 0 ? d_mulsmall128((u64)Q, one, n) : n - d_mulsmall128((u64)(-Q), one, n);
    const u128 Qm = Qa == n ? 0 : Qa;
    const u128 Dm = D >= 0 ? d_mulsmall128((u64)D, one, n) : n - d_mulsmall128((u64)(-D), one, n);
    u128 d = n + 1;
    int s = 0;
    while (!(d & 1)) { d >>= 1; s++; }
    int top = 127;
    while (!((d >> top) & 1)) top--;
    u128 U = one, V = one, Qk = Qm;
    for (int b = top - 1; b >= 0; b--) {
        U = d_mmul128(U, V, n, ni);
        V = d_msub128(d_mmul128(V, V, n, ni), d_madd128(Qk, Qk, n), n);
        Qk = d_mmul128(Qk, Qk, n, ni);
        if ((d >> b) & 1) {
            const u128 U2 = d_madd128(U, V, n);
            const u128 V2 = d_madd128(d_mmul128(Dm, U, n, ni), V, n);
            U = d_mhalf128(U2, n);
            V = d_mhalf128(V2, n);
            Qk = d_mmul128(Qk, Qm, n, ni);
        }
    }
    if (U == 0 || V == 0) return true;
    for (int r = 1; r < s; r++) {
        V = d_msub128(d_mmul128(V, V, n, ni), d_madd128(Qk, Qk, n), n);
        if (V == 0) return true;
        Qk = d_mmul128(Qk, Qk, n, ni);
    }
    return false;
}

/* ---- 96-bit Montgomery (R = 2^96): base-2 tests of members in [2^64, 2^96) ---- */

/*
 * d_mmul96: Montgomery product a*b/2^96 mod n with three 32-bit limbs (CIOS: coarsely
 * integrated operand scanning; the GPU's native multiplier is 32 bits wide).
 *   In:     a, b < n; odd n with 2^64 <= n < 2^96; ni = -n^-1 mod 2^32.
 *   Out:    a*b*2^-96 mod n, in [0, n).
 *   Method: for each limb a_i of a, low to high (CIOS_STEP): t += a_i b; then m = t0 ni mod
 *           2^32 makes t + m n divisible by 2^32, and t = (t + m n) / 2^32.  The accumulator is
 *           t3:t2:t1:t0 plus a carry limb t4; it stays below 2n, so one subtraction at the end
 *           suffices.
 */
__device__ __forceinline__ u128 d_mmul96(u128 a, u128 b, u128 n, u32 ni)
{
    const u32 a0 = (u32)a, a1 = (u32)(a >> 32), a2 = (u32)(a >> 64);
    const u32 b0 = (u32)b, b1 = (u32)(b >> 32), b2 = (u32)(b >> 64);
    const u32 n0 = (u32)n, n1 = (u32)(n >> 32), n2 = (u32)(n >> 64);
    u32 t0 = 0, t1 = 0, t2 = 0, t3 = 0;
    u64 x;
    u32 c, m;
#define CIOS_STEP(ai)                                                                     \
    x = (u64)t0 + (u64)(ai) * b0;           t0 = (u32)x; c = (u32)(x >> 32);              \
    x = (u64)t1 + (u64)(ai) * b1 + c;       t1 = (u32)x; c = (u32)(x >> 32);              \
    x = (u64)t2 + (u64)(ai) * b2 + c;       t2 = (u32)x; c = (u32)(x >> 32);              \
    x = (u64)t3 + c;                        t3 = (u32)x; { const u32 t4 = (u32)(x >> 32); \
    m = t0 * ni;                                                                          \
    x = (u64)t0 + (u64)m * n0;              c = (u32)(x >> 32);                           \
    x = (u64)t1 + (u64)m * n1 + c;          t0 = (u32)x; c = (u32)(x >> 32);              \
    x = (u64)t2 + (u64)m * n2 + c;          t1 = (u32)x; c = (u32)(x >> 32);              \
    x = (u64)t3 + c;                        t2 = (u32)x; t3 = t4 + (u32)(x >> 32); }
    CIOS_STEP(a0)
    CIOS_STEP(a1)
    CIOS_STEP(a2)
#undef CIOS_STEP
    const u128 r = ((u128)t3 << 96) | ((u128)t2 << 64) | ((u128)t1 << 32) | t0;
    return r >= n ? r - n : r;
}

/* d_r96: 2^96 mod n (the Montgomery form of 1 for R = 2^96), in [0, n), for odd n with
   2^64 <= n < 2^96.  Q = floor(2^96/n) <= 2^32 is estimated in float (error
   < 2^9) and lowered by 2^10, so r = 2^96 - q n is in [0, 2^11 n); a second float estimate of
   r/n (error < 1) brings r below 3n. */
__device__ __forceinline__ u128 d_r96(u128 n)
{
    const float nf = (float)n;
    u64 q = (u64)__fdividef(7.9228163e28f, nf);
    q = q > 1024 ? q - 1024 : 0;
    u128 r = ((u128)1 << 96) - (u128)q * n;
    u64 k = (u64)__fdividef((float)r, nf);
    k = k > 1 ? k - 1 : 0;
    r -= (u128)k * n;
    while (r >= n) r -= n;
    return r;
}

/* d_sprp2_96: base-2 strong probable-prime test (as d_sprp2_64) with R = 2^96, for odd n with
   2^64 <= n < 2^96.  This is where a(20) spent half its time (members above 2^64). */
__device__ bool d_sprp2_96(u128 n)
{
    const u32 ni = (u32)(0 - d_inv64((u64)n));
    const u128 one = d_r96(n), mone = n - one;
    u128 d = n - 1;
    int s;
    if ((u64)d) s = __ffsll((long long)(u64)d) - 1;
    else s = 63 + __ffsll((long long)(u64)(d >> 64));
    d >>= s;
    int b = (d >> 64) ? 127 - __clzll((long long)(u64)(d >> 64)) : 63 - __clzll((long long)(u64)d);
    u128 x = d_madd128(one, one, n);
    while (b-- > 0) {
        x = d_mmul96(x, x, n, ni);
        if ((d >> b) & 1) x = d_madd128(x, x, n);
    }
    if (x == one || x == mone) return true;
    for (int i = 1; i < s; i++) {
        x = d_mmul96(x, x, n, ni);
        if (x == mone) return true;
        if (x == one) return false;
    }
    return false;
}

/* d_sprp2, d_slprp: base-2 and strong Lucas tests for any odd n > 41 below 2^100, choosing
   the width of the arithmetic by the size of n. */
__device__ __forceinline__ bool d_sprp2(u128 n)
{
    if (!(n >> 64)) return d_sprp2_64((u64)n);
    return (n >> 96) ? d_sprp2_128(n) : d_sprp2_96(n);
}
__device__ __forceinline__ bool d_slprp(u128 n) { return (n >> 64) ? d_slprp128(n) : d_slprp64((u64)n); }

/* d_mix64: device copy of mix64() (must stay identical). */
__device__ __forceinline__ u64 d_mix64(u64 z)
{
    z += GOLD;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/*
 * d_candidate: final test and counting of one candidate p, whose six members all passed the
 * base-2 test.
 *   In:     p in [LO, HI); res = Res of the launch.
 *   Out:    res->cand += 1.  If a member fails the strong Lucas test: res->spsp += 1 and, while
 *           there is room, the member and p are recorded (sp_n, sp_p) for the host's SPSP
 *           line.  Otherwise res->cnt[b] += 1 and res->cks[b] += hash of p for the bin b with
 *           bnd[b] <= p < bnd[b+1] (binary search).
 */
__device__ void d_candidate(u128 p, Res *res)
{
    atomicAdd(&res->cand, 1ULL);
    for (int m = 0; m < 6; m++) {
        const u128 n = p + OFFD[m];
        if (!d_slprp(n)) {
            atomicAdd(&res->spsp, 1ULL);
            const unsigned i = atomicAdd(&res->nsp, 1u);
            if (i < MAXSPSP) {
                res->sp_n[i][0] = (u64)n; res->sp_n[i][1] = (u64)(n >> 64);
                res->sp_p[i][0] = (u64)p; res->sp_p[i][1] = (u64)(p >> 64);
            }
            return;
        }
    }
    int lo = 0, hi = P.nbins;                 /* bnd[lo] <= p < bnd[hi] */
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (p >= (((u128)P.bnd_hi[mid] << 64) | P.bnd_lo[mid])) lo = mid; else hi = mid;
    }
    const int b = lo;
    atomicAdd(&res->cnt[b], 1ULL);
    atomicAdd(&res->cks[b], (unsigned long long)d_mix64((u64)p ^ ((u64)(p >> 64) * GOLD)));
}

/*
 * d_phase: phase of sieving prime i at the first term p0 = base + delta of a class:
 * phi = -(p0 mod q) M^-1 mod q, so that q divides the member p0 + M k + d_j iff
 * k = phi + K_j (mod q).
 *   In:     T = device tables; i = prime index; delta < M.   Out: phi in [0, q).
 */
__device__ __forceinline__ u32 d_phase(const DevT &T, int i, u64 delta)
{
    const u32 q = T.q[i];
    u32 u = (u32)(delta % q) + T.bmod[i];
    if (u >= q) u -= q;
    return u ? (u32)((u64)(q - u) * T.minv[i] % q) : 0;
}

#ifdef NOATOMIC                             /* timing experiment only: racy, wrong results */
#define AOR(a, v) (*(a) |= (v))
#else
#define AOR(a, v) atomicOr((a), (v))
#endif
/* Marking bit x of the segment, where bw is the shared bitmap seen as 32-bit words: AMARK
   for positions known to be inside, MARK checks x < len without a branch (an out-of-range
   position, including a negative one cast to u32, ORs 0 into word 0) */
#define AMARK(x) AOR(bw + ((u32)(x) >> 5), 1u << ((u32)(x) & 31))
#define MARK(x) do { const u32 x_ = (u32)(x); const bool in_ = x_ < len; \
                     AOR(bw + (in_ ? x_ >> 5 : 0u), in_ ? 1u << (x_ & 31) : 0u); } while (0)

/*
 * sext_kernel: sieve one class per thread block and collect its survivors (theory, section 2).
 *   In:     cls_r[blockIdx.x] = residue of the block's class (class_r); T = device tables;
 *           res = Res of the launch; cnt = counters of the sub-launch (cnt->nb[0] = entries
 *           used in buf); buf = survivor buffer of cap entries (two u64 words each).
 *           LPT = large-prime phases per thread (LPT * NT >= P.nlarge).  Launched with one
 *           block of NT threads per class and SEGW * 8 bytes of dynamic shared memory.
 *   Out:    the initial members p of all survivors appended to buf; res->surv += survivors,
 *           res->bits += L (terms of the class).  Survivors that do not fit (position >= cap)
 *           are dropped and set res->ovf, which makes the host redo the launch.
 *   Steps:  class setup (first term ps = base + delta, length L) and initial phases; per
 *           segment of SEGB bits: 1. pattern primes, 2. medium primes, 3. large primes,
 *           4. survivors to buf, then the phase updates for the next segment.
 *   Errors: none reported; the P.skip bits (bench -X) leave out parts of the work.
 */
template <int LPT>
__global__ void __launch_bounds__(NT, MINB) sext_kernel(const u64 *__restrict__ cls_r, DevT T, Res *__restrict__ res,
                                                        Cnt *__restrict__ cnt, u64 *__restrict__ buf, u64 cap)
{
    extern __shared__ u64 bm[];             /* SEGW words */
    __shared__ u32 phs[MAXPAT];
    u32 *const bw = (u32 *)bm;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    const u64 M = P.M;
    const u64 r = cls_r[blockIdx.x];
    /* delta = (r - base) mod M: ps = base + delta is the class's first term >= base; L = number
       of terms below HI (see DevP) */
    const u64 delta = r >= P.bmM ? r - P.bmM : r + (M - P.bmM);
    const u64 L = delta <= P.rem0 ? P.Q0 + 1 : P.Q0;
    if (L == 0) return;
    const u128 ps = (((u128)P.base_hi << 64) | P.base_lo) + delta;
    /* fits64 is not used below: survivors are stored with 128 bits either way */
    const bool fits64 = ps + (u128)M * (L - 1) + 16 < ((u128)1 << 64);

    /* initial phases: pattern primes in shared memory (phs[i]); medium primes in registers,
       spread over the lanes (mph[i] of lane l in warp w belongs to medium prime
       w + NWARP (l + 32 i)); large primes in registers (lph[i] of thread t belongs to large
       prime t + NT i) */
    for (int i = tid; i < P.npat; i += NT) phs[i] = d_phase(T, i, delta);
    u32 mph[MAXMPL];
    #pragma unroll
    for (int i = 0; i < MAXMPL; i++) {
        const int m = warp + NWARP * (lane + 32 * i);
        mph[i] = m < P.nmed ? d_phase(T, P.npat + m, delta) : 0;
    }
    const int lb = P.npat + P.nmed;
    u32 lph[LPT];
    #pragma unroll
    for (int i = 0; i < LPT; i++) {
        const int g = tid + NT * i;
        lph[i] = g < P.nlarge ? d_phase(T, lb + g, delta) : 0;
    }
    unsigned long long nsurv = 0;
    __syncthreads();

    for (u64 k0 = 0; k0 < L; k0 += SEGB) {     /* segment = terms k0 .. k0 + len - 1 */
        const u32 len = L - k0 < SEGB ? (u32)(L - k0) : SEGB;
        const bool last = k0 + SEGB >= L;

        /* 1. pattern primes, into registers: word j = tid WPT + w gets T_q[(64 j - phi) mod q];
              s starts at (-phi + 64 WPT tid) mod q and grows by 64 mod q per word.  Storing the
              words also initialises the bitmap (all zero without pattern primes). */
        {
            u64 acc[WPT];
            #pragma unroll
            for (int w = 0; w < WPT; w++) acc[w] = 0;
            for (int i = 0; i < ((P.skip & 2) ? 0 : P.npat); i++) {
                const u32 q = T.q[i], st = T.pstep[i];
                const u64 *tab = T.tab + T.poff[i];
                const u32 f = phs[i];
                const u32 s0 = f ? q - f : 0;
                u32 s = (u32)((s0 + (u64)tid * T.pstept[i]) % q);
                #pragma unroll
                for (int w = 0; w < WPT; w++) {
                    acc[w] |= __ldg(tab + s);
                    s += st;
                    s = s >= q ? s - q : s;
                }
            }
            #pragma unroll
            for (int w = 0; w < WPT; w++) bm[tid * WPT + w] = acc[w];
        }
        __syncthreads();

        /* 2. medium primes: one warp per prime, lanes take rounds.  Warp w marks the medium
              primes w + NWARP (jl + 32 i); lane l takes the rounds t = l - 1, l + 31, ...
              (b = phi + t q), so lane 0 starts with round -1, whose larger offsets may reach
              into the segment.  Full rounds (b + K_5 < len) need no range checks. */
        #pragma unroll
        for (int i = 0; i < MAXMPL; i++) {
            for (int jl = 0; jl < 32; jl++) {
                const int m = warp + NWARP * (jl + 32 * i);
                if (m >= P.nmed || (P.skip & 4)) break;
                const int pi = P.npat + m;
                const u32 phi = __shfl_sync(0xffffffffu, mph[i], jl);
                const int q = (int)T.q[pi];
                const int k1 = (int)T.k[pi], k2 = (int)T.k[P.np + pi], k3 = (int)T.k[2 * P.np + pi];
                const int k4 = (int)T.k[3 * P.np + pi], k5 = (int)T.k[4 * P.np + pi];
                int b = (int)phi + (lane - 1) * q;
                if (b < 0) {                   /* lane 0, round -1 */
                    MARK(b + k1); MARK(b + k2); MARK(b + k3); MARK(b + k4); MARK(b + k5);
                    b += 32 * q;
                }
                for (const int lim = (int)len - k5; b < lim; b += 32 * q) {
                    AMARK(b); AMARK(b + k1); AMARK(b + k2); AMARK(b + k3); AMARK(b + k4); AMARK(b + k5);
                }
                if (b < (int)len) {
                    MARK(b); MARK(b + k1); MARK(b + k2); MARK(b + k3); MARK(b + k4);
                }
            }
            if (!last) {
                const int m = warp + NWARP * (lane + 32 * i);
                if (m < P.nmed) {
                    const int pi = P.npat + m;
                    const u32 q = T.q[pi], sm = T.segmod[pi];
                    mph[i] = mph[i] >= sm ? mph[i] - sm : mph[i] + q - sm;
                }
            }
        }

        /* 3. large primes: one thread per prime, as mark_prime() in a350826.c (round -1, full
              rounds, the last partial round, next phase from the end position) */
        #pragma unroll
        for (int i = 0; i < LPT; i++) {
            const int g = tid + NT * i;
            if (g < P.nlarge && !(P.skip & 8)) {
                const int pi = lb + g;
                const u32 q = T.q[pi];
                const u32 k1 = T.k[pi], k2 = T.k[P.np + pi], k3 = T.k[2 * P.np + pi];
                const u32 k4 = T.k[3 * P.np + pi], k5 = T.k[4 * P.np + pi];
                const u32 phi = lph[i];
                const u32 t = q - phi;
                if (k5 >= t) {
                    if (k1 >= t) MARK(k1 - t);
                    if (k2 >= t) MARK(k2 - t);
                    if (k3 >= t) MARK(k3 - t);
                    if (k4 >= t) MARK(k4 - t);
                    MARK(k5 - t);
                }
                u32 b = phi;
                if (len > k5) {
                    const u32 lim = len - k5;
                    for (; b < lim; b += q) {
                        AMARK(b); AMARK(b + k1); AMARK(b + k2); AMARK(b + k3); AMARK(b + k4); AMARK(b + k5);
                    }
                }
                if (b < len) {
                    MARK(b); MARK(b + k1); MARK(b + k2); MARK(b + k3); MARK(b + k4);
                }
                lph[i] = b >= len ? b - len : b + q - len;
            }
        }
        __syncthreads();

        /* 4. survivors of my words into the global buffer (one atomic per warp): clear bits
              below len are survivors; an inclusive prefix sum over the lanes gives each thread
              its offset, lane 31 reserves the warp's total at cnt->nb[0]; entries at or beyond
              cap are dropped and set res->ovf */
        if (!(P.skip & 16)) {
            u64 x[WPT];
            u32 c = 0;
            #pragma unroll
            for (int w = 0; w < WPT; w++) {
                const u32 j = tid * WPT + w;
                u64 v = ~bm[j];
                if ((j + 1) * 64 > len) v = (j * 64 >= len) ? 0 : v & ((1ULL << (len - j * 64)) - 1);
                x[w] = v;
                c += __popcll(v);
            }
            u32 incl = c;
            #pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const u32 y = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += y;
            }
            const u32 wtot = __shfl_sync(0xffffffffu, incl, 31);
            unsigned long long wbase = 0;
            if (lane == 31 && wtot) wbase = atomicAdd(&cnt->nb[0], (unsigned long long)wtot);
            wbase = __shfl_sync(0xffffffffu, wbase, 31);
            u64 pos = wbase + incl - c;
            nsurv += c;
            const u128 pk = ps + (u128)M * k0;
            #pragma unroll
            for (int w = 0; w < WPT; w++) {
                u64 v = x[w];
                const u32 j = tid * WPT + w;
                while (v) {
                    const u32 kk = j * 64 + (u32)(__ffsll((long long)v) - 1);
                    v &= v - 1;
                    if (pos < cap) {
                        const u128 p = pk + (u128)M * kk;
                        buf[2 * pos] = (u64)p;
                        buf[2 * pos + 1] = (u64)(p >> 64);
                    } else res->ovf = 1;
                    pos++;
                }
            }
        }

        /* pattern phases for the next segment */
        if (!last)
            for (int i = tid; i < P.npat; i += NT) {
                const u32 q = T.q[i], sm = T.segmod[i];
                phs[i] = phs[i] >= sm ? phs[i] - sm : phs[i] + q - sm;
            }
        __syncthreads();
    }
    /* block totals */
    for (int o = 16; o; o >>= 1) nsurv += __shfl_down_sync(0xffffffffu, nsurv, o);
    if (lane == 0) atomicAdd(&res->surv, nsurv);
    if (tid == 0) atomicAdd(&res->bits, (unsigned long long)L);
}

/*
 * round_kernel: round m of the breadth-first base-2 tests (theory, section 3).
 *   In:     m = 0..5; in = the survivors kept by round m - 1 (for m = 0, the sieve's), their
 *           number cnt->nb[m] (capped at cap; an overflowed launch is redone anyway);
 *           out = output buffer of cap entries; cnt->nb[m + 1] = 0 on entry.  Any grid
 *           (grid-stride loop).
 *   Out:    out[0 .. cnt->nb[m + 1]) = the p whose member p + OFF[m] is a base-2 strong
 *           probable prime, in no particular order.
 *   Method: one thread per number (d_sprp2_96 for members >= 2^64, which are below 2^81);
 *           __ballot_sync collects the passing lanes, lane 0 reserves room with one atomicAdd
 *           per warp, and each passing lane writes at its rank among them.
 */
__global__ void __launch_bounds__(256, 4) round_kernel(int m, const u64 *__restrict__ in, u64 *__restrict__ out,
                                                       Cnt *__restrict__ cnt, u64 cap)
{
    const u64 n0 = cnt->nb[m];
    const u64 n = n0 < cap ? n0 : cap;
    const int lane = threadIdx.x & 31;
    const u64 stride = (u64)gridDim.x * blockDim.x;
    for (u64 base = (u64)blockIdx.x * blockDim.x; base < n; base += stride) {
        const u64 i = base + threadIdx.x;
        bool ok = false;
        u128 p = 0;
        if (i < n) {
            p = ((u128)in[2 * i + 1] << 64) | in[2 * i];
            const u128 v = p + OFFD[m];       /* v < 2^81 (HIMAX) */
            ok = (v >> 64) ? d_sprp2_96(v) : d_sprp2_64((u64)v);
        }
        const unsigned bal = __ballot_sync(0xffffffffu, ok);
        unsigned long long wb = 0;
        if (lane == 0 && bal) wb = atomicAdd(&cnt->nb[m + 1], (unsigned long long)__popc(bal));
        wb = __shfl_sync(0xffffffffu, wb, 0);
        if (ok) {
            const u64 j = wb + __popc(bal & ((1u << lane) - 1));
            out[2 * j] = (u64)p;
            out[2 * j + 1] = (u64)(p >> 64);
        }
    }
}

/* cand_kernel: the cnt->nb[6] candidates left after the six rounds (in `in`): strong Lucas
   tests of their members, counting and binning (d_candidate).  Any grid (grid-stride loop). */
__global__ void __launch_bounds__(256) cand_kernel(const u64 *__restrict__ in, const Cnt *__restrict__ cnt,
                                                   Res *__restrict__ res)
{
    const u64 n = cnt->nb[6];
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (u64)gridDim.x * blockDim.x)
        d_candidate(((u128)in[2 * i + 1] << 64) | in[2 * i], res);
}

/*
 * test_kernel: GPU tests of single numbers (selftest and the `num` command).
 *   In:     v = n numbers (two u64 words each, low first), odd, > 41 and below 2^100 (others
 *           give meaningless results); one thread per number.
 *   Out:    out[i] bit 0 = base-2 strong probable prime (d_sprp2), bit 1 = strong Lucas
 *           probable prime (d_slprp).
 */
__global__ void test_kernel(const u64 *v, int n, u8 *out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const u128 x = ((u128)v[2 * i + 1] << 64) | v[2 * i];
    out[i] = (u8)((d_sprp2(x) ? 1 : 0) | (d_slprp(x) ? 2 : 0));
}

/* ------------------------------------------------------------------ */
/* Host driver                                                         */
/* ------------------------------------------------------------------ */

/* Results of a chunk and the run totals (as in a350826.c): per bin count and checksum;
   survivors, candidates (six members passed base 2), SPSP rejections, terms sieved. */
typedef struct {
    u64 cnt[MAXBIN], cks[MAXBIN];
    u64 surv, cand, spsp, bits;
} acc_t;

/* The range of the run: initial members in [lo, hi); base = max(lo, P0), the first number
   sieved on the GPU; bmM = base mod M; bins bnd[b] .. bnd[b+1]. */
static struct {
    u128 lo, hi, base;
    u64 bmM;
    int nbins;
    u128 bnd[MAXBIN + 1];
} R;

/* Chunks of the run: chunk idx = classes c0 + idx*chunk .. (or entries of `sample` in bench
   mode); chunks [0, frontier) are folded into tot. */
static struct {
    u64 chunk, nchunks, frontier, c0, c1;
    acc_t tot;
    u64 *sample;
    u64 nsample;
} D;

static FILE *g_log = NULL;                   /* chunk log (-L), or NULL */
static bool g_print = true;                  /* print SPSP lines (off in bench and selftest) */
static int g_skip = 0;                       /* bench -X ablation mask */
static bool g_empty = false;                 /* nothing to sieve (HI <= 2^20) */
static u64 g_nspsp_printed = 0;              /* SPSP lines printed so far (information only) */

/* on_sigint: SIGINT/SIGTERM handler: only sets g_stop (async-signal-safe). */
static void on_sigint(int sig) { (void)sig; g_stop = 1; }

/* acc_add: t += a, field by field (the checksums add mod 2^64). */
static void acc_add(acc_t *t, const acc_t *a)
{
    for (int b = 0; b < MAXBIN; b++) { t->cnt[b] += a->cnt[b]; t->cks[b] += a->cks[b]; }
    t->surv += a->surv; t->cand += a->cand; t->spsp += a->spsp; t->bits += a->bits;
}

/*
 * log_chunk: append the line of one folded chunk to the chunk log: "C idx survivors
 * candidates spsp" and "count:cks" per bin (cks in hex) - the format of a350826.c, so
 * `a350826 recheck` can recompute GPU chunks on the CPU.  idx is absolute (c0/chunk + idx)
 * when the class range is chunk-aligned (-P parts), else relative.  Write errors are not
 * checked.
 */
static void log_chunk(u64 idx, const acc_t *a)
{
    /* absolute chunk index when the class range is chunk-aligned (-P parts), else relative */
    const u64 aidx = D.c0 % D.chunk == 0 ? D.c0 / D.chunk + idx : idx;
    fprintf(g_log, "C %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64, aidx, a->surv, a->cand, a->spsp);
    for (int b = 0; b < R.nbins; b++) fprintf(g_log, " %" PRIu64 ":%016" PRIx64, a->cnt[b], a->cks[b]);
    fputc('\n', g_log);
    fflush(g_log);
}

/*
 * small_region: sextuplets with initial member in [lo, min(hi, P0)), counted on the host by
 * direct testing of every p = 7 (mod 30) (the GPU sieve needs p > B, and the sextuplet at 7
 * is not 97 mod 210).  Added to chunk 0 of a run that starts at class 0 (fold).
 *   Out:    A->cnt and A->cks of the bins (linear search); nothing if lo >= P0.
 */
static void small_region(acc_t *A)
{
    if (R.lo >= P0) return;
    const u64 a = (u64)R.lo, b = R.hi < P0 ? (u64)R.hi : P0;
    for (u64 p = a; p < b; p++) {
        if (p % 30 != 7) continue;
        if (is_sext_ref(p)) {
            int k = 0;
            while (k + 1 < R.nbins && p >= R.bnd[k + 1]) k++;
            A->cnt[k]++;
            A->cks[k] += hashp(p);
        }
    }
}

/* Device state.  Index [2] = one set per stream; h* are pinned host buffers (needed for
   asynchronous copies), d* device buffers. */
static struct {
    DevT T;                 /* device pointers of the tables, passed to sext_kernel */
    u32 *dq, *dk, *dminv, *dbmod, *dsegmod, *dpstep, *dpstept, *dpoff;   /* device tables (dev_upload) */
    u64 *dtab;
    u64 *dcls[2], *hcls[2];   /* class residues of a launch (G.cap entries) */
    Res *dres[2], *hres[2];   /* results of a launch */
    u64 *dbuf[2][2];        /* survivor ping-pong buffers per stream (2 u64 per entry) */
    Cnt *dcnt[2];
    u64 bcap;               /* entries per survivor buffer */
    u64 sub;                /* classes per sub-launch (expected survivors fit the buffer) */
    cudaStream_t st[2];
    u64 cap;                /* classes per launch buffer */
} G;

/* dev_free: free the device tables of the previous configuration (the per-stream buffers of
   dev_init stay). */
static void dev_free(void)
{
    cudaFree(G.dq); cudaFree(G.dk); cudaFree(G.dminv); cudaFree(G.dbmod); cudaFree(G.dsegmod);
    cudaFree(G.dpstep); cudaFree(G.dpstept); cudaFree(G.dpoff); cudaFree(G.dtab);
    memset(&G.T, 0, sizeof G.T);
    G.dq = G.dk = G.dminv = G.dbmod = G.dsegmod = G.dpstep = G.dpstept = G.dpoff = NULL;
    G.dtab = NULL;
}

/* dev_copy: device array with a copy of h[0 .. n) (at least one element is allocated);
   die() on a CUDA error. */
template <class T> static T *dev_copy(const T *h, size_t n)
{
    T *d;
    CK(cudaMalloc(&d, sizeof(T) * (n ? n : 1)));
    if (n) CK(cudaMemcpy(d, h, sizeof(T) * n, cudaMemcpyHostToDevice));
    return d;
}

/*
 * dev_upload: copy the tables of SP to the device (replacing the previous ones) and set the
 * run parameters P in constant memory (see DevP), from SP, R, W and g_skip.
 *   Errors: die() on any CUDA error.
 */
static void dev_upload(void)
{
    dev_free();
    G.dq = dev_copy(SP.q, SP.np);
    G.dk = dev_copy(SP.k, 5 * (size_t)SP.np);
    G.dminv = dev_copy(SP.minv, SP.np);
    G.dbmod = dev_copy(SP.bmod, SP.np);
    G.dsegmod = dev_copy(SP.segmod, SP.np);
    G.dpstep = dev_copy(SP.pstep, MAXPAT);
    G.dpstept = dev_copy(SP.pstept, MAXPAT);
    G.dpoff = dev_copy(SP.poff, MAXPAT);
    G.dtab = dev_copy(SP.tab, SP.ntab);
    G.T.q = G.dq; G.T.k = G.dk; G.T.minv = G.dminv; G.T.bmod = G.dbmod; G.T.segmod = G.dsegmod;
    G.T.pstep = G.dpstep; G.T.pstept = G.dpstept; G.T.poff = G.dpoff; G.T.tab = G.dtab;
    DevP p;
    memset(&p, 0, sizeof p);
    p.M = W.M;
    p.bmM = R.bmM;
    const u128 span = R.hi > R.base ? R.hi - 1 - R.base : 0;
    p.Q0 = R.hi > R.base ? (u64)(span / W.M) : 0;
    p.rem0 = R.hi > R.base ? (u64)(span % W.M) : 0;
    if (R.hi <= R.base) { p.Q0 = 0; p.rem0 = 0; }
    p.base_lo = (u64)R.base; p.base_hi = (u64)(R.base >> 64);
    p.nbins = R.nbins;
    for (int b = 0; b <= R.nbins; b++) { p.bnd_lo[b] = (u64)R.bnd[b]; p.bnd_hi[b] = (u64)(R.bnd[b] >> 64); }
    p.npat = SP.npat; p.nmed = SP.nmed; p.nlarge = SP.nlarge; p.np = SP.np;
    p.skip = g_skip;
    CK(cudaMemcpyToSymbol(P, &p, sizeof p));
}

/*
 * dev_init: select CUDA device dev and allocate the per-stream buffers: two streams; class
 * residues for G.cap = 65536 classes (device and pinned host); one Res (device and pinned
 * host); two survivor ping-pong buffers of G.bcap = 2^25 entries (512 MiB each) and a Cnt.
 * Raises the dynamic shared memory limit of the three sext_kernel instances to one segment.
 *   Notes:  must be called before configure(), which uses G.bcap.
 *   Errors: die() on any CUDA error (no such device, not enough memory, ...).
 */
static void dev_init(int dev)
{
    CK(cudaSetDevice(dev));
    G.cap = 1 << 16;
    for (int i = 0; i < 2; i++) {
        CK(cudaStreamCreate(&G.st[i]));
        CK(cudaMalloc(&G.dcls[i], sizeof(u64) * G.cap));
        CK(cudaMallocHost(&G.hcls[i], sizeof(u64) * G.cap));
        CK(cudaMalloc(&G.dres[i], sizeof(Res)));
        CK(cudaMallocHost(&G.hres[i], sizeof(Res)));
    }
    G.bcap = 1ULL << 25;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) CK(cudaMalloc(&G.dbuf[i][j], sizeof(u64) * 2 * G.bcap));
        CK(cudaMalloc(&G.dcnt[i], sizeof(Cnt)));
    }
    CK(cudaFuncSetAttribute(sext_kernel<12>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(sizeof(u64) * SEGW)));
    CK(cudaFuncSetAttribute(sext_kernel<24>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(sizeof(u64) * SEGW)));
    CK(cudaFuncSetAttribute(sext_kernel<MAXLPT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(sizeof(u64) * SEGW)));
}

/* ---- checkpoint (same format as a350826.c) ---- */

/*
 * save_state: write the checkpoint (run parameters, frontier, totals of the folded chunks) to
 * path.tmp and rename it to path, so it is never half written.
 *   Errors: a file that cannot be written or renamed only gives a warning; the run goes on and
 *           the previous checkpoint stays valid.
 */
static void save_state(const char *path)
{
    char tmp[PATH_MAX];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "w");
    if (!f) { fprintf(stderr, "\nwarning: cannot write %s: %s\n", tmp, strerror(errno)); return; }
    fprintf(f, "a350826 checkpoint 1\nlo %s\nhi %s\nwheel %u\nbound %u\nchunk %" PRIu64 "\nclasses %" PRIu64
               " %" PRIu64 "\nbins %d\n", u128s(R.lo), u128s(R.hi), W.w, SP.B, D.chunk, D.c0, D.c1, R.nbins);
    for (int b = 0; b <= R.nbins; b++) fprintf(f, "bnd %s\n", u128s(R.bnd[b]));
    fprintf(f, "frontier %" PRIu64 "\nsurv %" PRIu64 "\ncand %" PRIu64 "\nspsp %" PRIu64 "\nbits %" PRIu64 "\n",
            D.frontier, D.tot.surv, D.tot.cand, D.tot.spsp, D.tot.bits);
    for (int b = 0; b < R.nbins; b++) fprintf(f, "bin %d %" PRIu64 " %016" PRIx64 "\n", b, D.tot.cnt[b], D.tot.cks[b]);
    fputs("end\n", f);
    if (fclose(f) != 0 || rename(tmp, path) != 0)
        fprintf(stderr, "\nwarning: cannot update %s: %s\n", path, strerror(errno));
}

/*
 * load_state: resume from a checkpoint (written by either tool).
 *   In:     path; the run must be configured and D.nchunks set.
 *   Out:    true with D.frontier and D.tot restored; false if the file does not exist.
 *   Errors: die() if the file cannot be read, is not an a350826 checkpoint, is truncated, has
 *           a malformed number or bin line, was written for other parameters (range, wheel,
 *           bound, chunk size, class range, bins) or has its frontier beyond the last chunk.
 */
static bool load_state(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        if (errno == ENOENT) return false;
        die("cannot read %s: %s", path, strerror(errno));
    }
    char line[512], slo[64] = "", shi[64] = "";
    if (!fgets(line, sizeof line, f) || strncmp(line, "a350826 checkpoint 1", 20) != 0)
        die("%s is not an a350826 checkpoint", path);
    u32 w = 0, B = 0;
    u64 chunk = 0, frontier = 0, c0 = 0, c1 = 0;
    int nbins = -1, nb = 0;
    u128 bnd[MAXBIN + 1];
    acc_t t;
    memset(&t, 0, sizeof t);
    bool complete = false;
    while (fgets(line, sizeof line, f)) {
        char sv[64];
        int b;
        unsigned long long c, k;
        if (sscanf(line, "lo %63s", slo) == 1) continue;
        if (sscanf(line, "hi %63s", shi) == 1) continue;
        if (sscanf(line, "wheel %u", &w) == 1) continue;
        if (sscanf(line, "bound %u", &B) == 1) continue;
        if (sscanf(line, "chunk %" SCNu64, &chunk) == 1) continue;
        if (sscanf(line, "classes %" SCNu64 " %" SCNu64, &c0, &c1) == 2) continue;
        if (sscanf(line, "bins %d", &nbins) == 1) continue;
        if (sscanf(line, "bnd %63s", sv) == 1) { if (nb <= MAXBIN) bnd[nb++] = parse_num(sv); continue; }
        if (sscanf(line, "frontier %" SCNu64, &frontier) == 1) continue;
        if (sscanf(line, "surv %" SCNu64, &t.surv) == 1) continue;
        if (sscanf(line, "cand %" SCNu64, &t.cand) == 1) continue;
        if (sscanf(line, "spsp %" SCNu64, &t.spsp) == 1) continue;
        if (sscanf(line, "bits %" SCNu64, &t.bits) == 1) continue;
        if (sscanf(line, "bin %d %llu %llx", &b, &c, &k) == 3) {
            if (b < 0 || b >= MAXBIN) die("%s: bad bin line", path);
            t.cnt[b] = c; t.cks[b] = k;
            continue;
        }
        if (!strncmp(line, "end", 3)) { complete = true; break; }
    }
    fclose(f);
    if (!complete) die("%s is truncated", path);
    if (parse_num(slo) != R.lo || parse_num(shi) != R.hi || w != W.w || B != SP.B || chunk != D.chunk ||
        c0 != D.c0 || c1 != D.c1)
        die("%s was written for lo=%s hi=%s w=%u B=%u chunk=%" PRIu64 " classes [%" PRIu64 ", %" PRIu64
            "); rerun with the same parameters", path, slo, shi, w, B, chunk, c0, c1);
    if (nbins != R.nbins || nb != nbins + 1) die("%s was written for different bins", path);
    for (int b = 0; b <= nbins; b++) if (bnd[b] != R.bnd[b]) die("%s was written for different bins", path);
    if (frontier > D.nchunks) die("%s: frontier beyond the last chunk", path);
    D.frontier = frontier;
    D.tot = t;
    return true;
}

/* fill_chunk: residues of the classes of chunk idx (D.c0 + c, or D.sample[c] in bench mode)
   into the host buffer h (at least D.chunk entries); returns their number. */
static u64 fill_chunk(u64 idx, u64 *h)
{
    const u64 n = D.sample ? D.nsample : D.c1 - D.c0;
    const u64 c0 = idx * D.chunk, c1 = c0 + D.chunk < n ? c0 + D.chunk : n;
    for (u64 c = c0; c < c1; c++) h[c - c0] = class_r(D.sample ? D.sample[c] : D.c0 + c);
    return c1 - c0;
}

/*
 * launch_classes: queue the kernels for ncls classes (residues at dcls on the device) on stream
 * s, in sub-launches of G.sub classes.  Per sub-launch: zero the counters; sext_kernel with the
 * smallest LPT that holds the large primes; the six round kernels (ping-pong between
 * dbuf[s][0] and dbuf[s][1], ending in dbuf[s][0]); cand_kernel.  All add into G.dres[s].
 *   Notes:  asynchronous (results are valid after the stream is synchronised); does nothing if
 *           the run has nothing to sieve (g_empty); with -X 1 (bench) the tests are left out.
 *   Errors: die() if a launch is rejected (errors during execution appear at the next sync).
 */
static void launch_classes(int s, const u64 *dcls, u64 ncls)
{
    if (!ncls || g_empty) return;
    const size_t smem = sizeof(u64) * SEGW;
    for (u64 a = 0; a < ncls; a += G.sub) {
        const u64 n = a + G.sub < ncls ? G.sub : ncls - a;
        CK(cudaMemsetAsync(G.dcnt[s], 0, sizeof(Cnt), G.st[s]));
        if (SP.nlarge <= NT * 12)
            sext_kernel<12><<<(unsigned)n, NT, smem, G.st[s]>>>(dcls + a, G.T, G.dres[s], G.dcnt[s], G.dbuf[s][0], G.bcap);
        else if (SP.nlarge <= NT * 24)
            sext_kernel<24><<<(unsigned)n, NT, smem, G.st[s]>>>(dcls + a, G.T, G.dres[s], G.dcnt[s], G.dbuf[s][0], G.bcap);
        else
            sext_kernel<MAXLPT><<<(unsigned)n, NT, smem, G.st[s]>>>(dcls + a, G.T, G.dres[s], G.dcnt[s], G.dbuf[s][0],
                                                                  G.bcap);
        CK(cudaGetLastError());
        if (g_skip & 1) continue;
        for (int m = 0; m < 6; m++)
            round_kernel<<<48 * 8, 256, 0, G.st[s]>>>(m, G.dbuf[s][m & 1], G.dbuf[s][(m + 1) & 1], G.dcnt[s], G.bcap);
        cand_kernel<<<48 * 4, 256, 0, G.st[s]>>>(G.dbuf[s][0], G.dcnt[s], G.dres[s]);
        CK(cudaGetLastError());
    }
}

/* launch: queue one launch on stream s: copy the ncls class residues from G.hcls[s], zero
   G.dres[s], launch_classes(), copy the Res back to G.hres[s].  Returns at once. */
static void launch(int s, u64 ncls)
{
    CK(cudaMemcpyAsync(G.dcls[s], G.hcls[s], sizeof(u64) * ncls, cudaMemcpyHostToDevice, G.st[s]));
    CK(cudaMemsetAsync(G.dres[s], 0, sizeof(Res), G.st[s]));
    launch_classes(s, G.dcls[s], ncls);
    CK(cudaMemcpyAsync(G.hres[s], G.dres[s], sizeof(Res), cudaMemcpyDeviceToHost, G.st[s]));
}

/*
 * redo_in_pieces: a launch overflowed its survivor buffer: redo its classes in pieces, summing
 * the results (theory, section 5).
 *   In:     s = stream; ncls classes, already in G.dcls[s]; out = where the result goes.
 *   Out:    *out = sum over 8 pieces of ncls/8 classes, each run with sub-launches of G.sub/8
 *           classes and synchronised; SPSP records merged (the first MAXSPSP).
 *   Errors: die() if a piece overflows again or memory runs out.  Not exercised by the
 *           selftest.
 */
static void redo_in_pieces(int s, u64 ncls, Res *out)
{
    Res *acc = (Res *)calloc(1, sizeof(Res));
    if (!acc) die("out of memory");
    const u64 sub0 = G.sub;
    G.sub = G.sub / 8 ? G.sub / 8 : 1;
    const u64 piece = ncls / 8 ? ncls / 8 : 1;
    for (u64 a = 0; a < ncls; a += piece) {
        const u64 n = a + piece < ncls ? piece : ncls - a;
        CK(cudaMemsetAsync(G.dres[s], 0, sizeof(Res), G.st[s]));
        launch_classes(s, G.dcls[s] + a, n);
        CK(cudaMemcpyAsync(G.hres[s], G.dres[s], sizeof(Res), cudaMemcpyDeviceToHost, G.st[s]));
        CK(cudaStreamSynchronize(G.st[s]));
        const Res *r = G.hres[s];
        if (r->ovf) die("survivor buffer overflow even for %" PRIu64 " classes", n);
        for (int b = 0; b < MAXBIN; b++) { acc->cnt[b] += r->cnt[b]; acc->cks[b] += r->cks[b]; }
        acc->surv += r->surv; acc->cand += r->cand; acc->spsp += r->spsp; acc->bits += r->bits;
        for (unsigned i = 0; i < r->nsp && i < MAXSPSP && acc->nsp < MAXSPSP; i++, acc->nsp++) {
            memcpy(acc->sp_n[acc->nsp], r->sp_n[i], sizeof r->sp_n[i]);
            memcpy(acc->sp_p[acc->nsp], r->sp_p[i], sizeof r->sp_p[i]);
        }
    }
    memcpy(out, acc, sizeof(Res));
    free(acc);
    G.sub = sub0;
}

/*
 * fold: add the results of chunk idx to the totals, in chunk order (called by run_count).
 *   In:     idx; r = its Res.
 *   Out:    D.tot += r (plus the small region for chunk 0 of a run that starts at class 0);
 *           SPSP lines printed (if g_print; at most MAXSPSP per chunk); chunk-log line (-L).
 */
static void fold(u64 idx, const Res *r)
{
    acc_t a;
    memset(&a, 0, sizeof a);
    for (int b = 0; b < MAXBIN; b++) { a.cnt[b] = r->cnt[b]; a.cks[b] = r->cks[b]; }
    a.surv = r->surv; a.cand = r->cand; a.spsp = r->spsp; a.bits = r->bits;
    if (idx == 0 && !D.sample && D.c0 == 0) small_region(&a);
    const unsigned nsp = r->nsp < MAXSPSP ? r->nsp : MAXSPSP;
    for (unsigned i = 0; i < nsp && g_print; i++) {
        const u128 n = ((u128)r->sp_n[i][1] << 64) | r->sp_n[i][0], p = ((u128)r->sp_p[i][1] << 64) | r->sp_p[i][0];
        if (stderr_tty) fputs("\r\033[K", stderr);
        printf("SPSP %s (base-2 strong pseudoprime in candidate p = %s)\n", u128s(n), u128s(p));
        g_nspsp_printed++;
    }
    acc_add(&D.tot, &a);
    if (g_log) log_chunk(idx, &a);
}

/* Wall time and number of terms sieved by one run_count call. */
typedef struct { double seconds; u64 bits; } run_stats_t;

/*
 * run_count: run the configured count until all chunks are folded or the run is stopped.
 *   In:     state = checkpoint path or NULL; interval = seconds between checkpoints (and
 *           between status lines when stderr is not a terminal); quiet = no status output.
 *           Needs dev_init() and configure() (bench also sets D.sample).
 *   Out:    D.tot = totals of all folded chunks (including those restored from the
 *           checkpoint), D.frontier = number of folded chunks; returns the wall time and
 *           the terms sieved by this call.  Prints a status line (rate and ETA since the
 *           start of this call) and writes the checkpoint every `interval` seconds and at the
 *           end.
 *   Method: two launches in flight on alternating streams; the older one is awaited, redone
 *           in pieces if it overflowed, and folded, so chunks are folded in order.
 *   Errors: die() for a chunk larger than the launch buffer, a bad checkpoint or a CUDA error.
 *           SIGINT/SIGTERM: no new launches; those in flight finish and are folded.
 */
static run_stats_t run_count(const char *state, int interval, bool quiet)
{
    D.nchunks = ((D.sample ? D.nsample : D.c1 - D.c0) + D.chunk - 1) / D.chunk;
    if (D.chunk > G.cap) die("chunk larger than the launch buffer (%" PRIu64 ")", G.cap);
    D.frontier = 0;
    memset(&D.tot, 0, sizeof D.tot);
    if (state && load_state(state) && !quiet)
        fprintf(stderr, "resuming from %s: %" PRIu64 " of %" PRIu64 " chunks done\n", state, D.frontier, D.nchunks);
    const u64 bits0 = D.tot.bits, f0 = D.frontier;
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_sigint;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    const double t0 = now();
    double last_status = 0, last_ckpt = t0;
    char b1[32], b2[32];
    u64 next = D.frontier;
    int inflight = 0;
    u64 idxs[2] = {0, 0};
    int cur = 0;
    /* two launches in flight on alternating streams; fold in order */
    while ((next < D.nchunks && !g_stop) || inflight) {
        while (inflight < 2 && next < D.nchunks && !g_stop) {
            const int s = (cur + inflight) & 1;
            const u64 n = fill_chunk(next, G.hcls[s]);
            launch(s, n);
            idxs[s] = next++;
            inflight++;
        }
        CK(cudaStreamSynchronize(G.st[cur]));
        if (G.hres[cur]->ovf) {                /* survivor buffer overflow: redo the chunk */
            const u64 n = fill_chunk(idxs[cur], G.hcls[cur]);
            CK(cudaMemcpy(G.dcls[cur], G.hcls[cur], sizeof(u64) * n, cudaMemcpyHostToDevice));
            redo_in_pieces(cur, n, G.hres[cur]);
        }
        fold(idxs[cur], G.hres[cur]);
        D.frontier = idxs[cur] + 1;
        inflight--;
        cur ^= 1;
        const double t = now();
        if (!quiet && (stderr_tty ? t - last_status >= 1.0 : t - last_status >= interval)) {
            last_status = t;
            const double el = t - t0;
            const double rate = el > 0 ? (double)(D.tot.bits - bits0) / el : 0;
            const double crate = el > 0 ? (double)(D.frontier - f0) / el : 0;
            u64 found = 0;
            for (int b = 0; b < R.nbins; b++) found += D.tot.cnt[b];
            fprintf(stderr, "%s%s  chunks %" PRIu64 "/%" PRIu64 " (%.3f%%)  %.3g bits/s  found %" PRIu64 "%s  ETA %s",
                    stderr_tty ? "\r\033[K" : "progress: ", fmt_dur(el, b1, sizeof b1), D.frontier, D.nchunks,
                    100.0 * (double)D.frontier / (double)D.nchunks, rate, found, D.tot.spsp ? "  SPSP!" : "",
                    fmt_dur(crate > 0 ? (double)(D.nchunks - D.frontier) / crate : -1, b2, sizeof b2));
            if (!stderr_tty) fputc('\n', stderr);
            fflush(stderr);
        }
        if (state && t - last_ckpt >= interval) { last_ckpt = t; save_state(state); }
    }
    const double t1 = now();
    if (stderr_tty && !quiet) fputs("\r\033[K", stderr);
    if (state) save_state(state);
    if (g_stop && D.frontier < D.nchunks && !quiet)
        fprintf(stderr, "stopped after %" PRIu64 " of %" PRIu64 " chunks%s\n", D.frontier, D.nchunks,
                state ? ", checkpoint saved" : "");
    run_stats_t st = {t1 - t0, D.tot.bits - bits0};
    return st;
}

/* ------------------------------------------------------------------ */
/* Parameters and commands                                             */
/* ------------------------------------------------------------------ */

/* rho_wheel: fraction of all integers in the classes of wheel w,
   1/210 * prod_{11 <= q <= w} (q - 6)/q. */
static double rho_wheel(u32 w)
{
    double r = 1.0 / 210;
    for (int i = 4; i < 15 && SMALLP[i] <= w; i++) r *= (double)(SMALLP[i] - 6) / SMALLP[i];
    return r;
}

/* auto_wheel: the wheel (largest prime w in 7..47) with the smallest estimated time for
   [lo, hi); 7 if nothing is sieved.  Wheels whose classes would hold less than one term are
   not considered.  die() if memory runs out.
   GPU cost model, measured on GB10 (B = 2^16): ~1.1 us per class (phases of all sieving primes,
   block start) plus ~5.8 ps per candidate bit for B = 2^16 (scaled by the marks per bit). */
static u32 auto_wheel(u128 lo, u128 hi, u32 B)
{
    const u128 base = lo > P0 ? lo : P0;
    if (hi <= base) return 7;
    int np;
    u32 *pr = small_primes(B, &np);
    const double len = (double)(hi - base);
    u32 best = 7;
    double bestc = 1e300;
    for (int i = 3; i < 15; i++) {
        const u32 w = SMALLP[i];
        double M = 1, C = 1, marks = 0;
        for (int j = 0; j < 15 && SMALLP[j] <= w; j++) M *= SMALLP[j];
        for (int j = 4; j < 15 && SMALLP[j] <= w; j++) C *= SMALLP[j] - 6;
        for (int j = 0; j < np; j++) if (pr[j] > w) marks += 6.0 / pr[j];
        if (len / M < 1 && w > 7) break;
        const double cost = C * 1.1e-6 * (np / 6542.0) + len * rho_wheel(w) * 5.8e-12 * (marks / 6.45);
        if (cost < bestc) { bestc = cost; best = w; }
    }
    free(pr);
    return best;
}

/* auto_chunk: default classes per chunk, as in a350826.c: the largest power of two <= C/4096,
   at most 16384 (so a chunk always fits the launch buffer of 65536 classes). */
static u64 auto_chunk(u64 C)
{
    u64 c = 1;
    while (c * 2 <= C / 4096 && c < 16384) c *= 2;
    return c;
}

/* Command-line options (parse_opts); 0 means "default" or "not given" for most fields. */
typedef struct {
    u32 w, B, pmax, qw;     /* -w (0 = auto), -B, -p, -Q */
    u64 chunk, nsample, cr0, cr1, part, nparts;   /* -c (0 = auto), -n, -C C0:C1, -P I/N */
    int interval, dev;      /* -i, -d */
    const char *state, *log;  /* -S, -L */
    bool quiet;             /* -q */
    int nb;                 /* number of -b boundaries */
    u128 bnd[MAXBIN];       /* -b boundaries (checked in configure) */
} opts_t;

/* usage: print the command summary to stderr and exit with status 2. */
static void usage(void)
{
    fputs("usage: a350826_cuda count LO HI [-b B1,B2,..] [-w W] [-B B] [-c CH] [-C C0:C1 | -P I/N] [-S FILE] [-i SECS]\n"
          "                          [-L FILE] [-q] [-d DEV]\n"
          "       a350826_cuda bench LO HI [-n N] [-w W] [-B B] [-d DEV]\n"
          "       a350826_cuda selftest [N] [-d DEV]\n", stderr);
    exit(2);
}

/*
 * parse_opts: parse the options and positional numbers of a command (as in a350826.c).
 *   Out:    *o = defaults (B = 2^16, -p 256, -Q 4096, interval 60 s, 20000 bench samples)
 *           overridden by the options; pos[0 .. npos) = the positional numbers.  -X sets the
 *           global g_skip.
 *   Errors: usage() (exit 2) for an unknown option, a missing value or the wrong number of
 *           positional arguments; die() for an invalid or out-of-range value.  A -b list
 *           longer than 16383 characters would be cut silently (255 boundaries fit easily).
 */
static void parse_opts(int argc, char **argv, int first, opts_t *o, int npos, u128 *pos)
{
    memset(o, 0, sizeof *o);
    o->B = 1u << 16;
    o->pmax = 256;
    o->qw = 4096;
    o->interval = 60;
    o->nsample = 20000;
    int np = 0;
    for (int i = first; i < argc; i++) {
        const char *a = argv[i];
        if (a[0] == '-' && a[1] && !isdigit((unsigned char)a[1]) && a[2] == 0) {
            const char c = a[1];
            if (c == 'q') { o->quiet = true; continue; }
            if (i + 1 >= argc) usage();
            const char *v = argv[++i];
            switch (c) {
            case 'b': {
                char buf[16384];
                snprintf(buf, sizeof buf, "%s", v);
                for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
                    if (o->nb >= MAXBIN - 1) die("too many bin boundaries");
                    o->bnd[o->nb++] = parse_num(tok);
                }
                break;
            }
            case 'w': o->w = (u32)parse_u64(v, 7, 47, "wheel prime"); break;
            case 'B': o->B = (u32)parse_u64(v, 11, BMAX, "sieving bound"); break;
            case 'p': o->pmax = (u32)parse_u64(v, 0, 1024, "pattern bound"); break;
            case 'Q': o->qw = (u32)parse_u64(v, 0, BMAX, "medium bound"); break;
            case 'c': o->chunk = parse_u64(v, 1, 1ULL << 16, "chunk"); break;
            case 'n': o->nsample = parse_u64(v, 1, 1ULL << 40, "sample"); break;
            case 'i': o->interval = (int)parse_u64(v, 1, 86400, "interval"); break;
            case 'd': o->dev = (int)parse_u64(v, 0, 64, "device"); break;
            case 'X': g_skip = (int)parse_u64(v, 0, 31, "skip mask"); break;
            case 'S': o->state = v; break;
            case 'L': o->log = v; break;
            case 'C': {
                const char *colon = strchr(v, ':');
                if (!colon) die("-C wants C0:C1");
                char tmp[64];
                snprintf(tmp, sizeof tmp, "%.*s", (int)(colon - v), v);
                o->cr0 = (u64)parse_num(tmp);
                o->cr1 = (u64)parse_num(colon + 1);
                if (o->cr1 <= o->cr0) die("-C wants C0 < C1");
                break;
            }
            case 'P': {
                const char *sl = strchr(v, '/');
                if (!sl) die("-P wants I/N");
                char tmp[64];
                snprintf(tmp, sizeof tmp, "%.*s", (int)(sl - v), v);
                o->part = (u64)parse_num(tmp);
                o->nparts = (u64)parse_num(sl + 1);
                if (o->nparts == 0 || o->part >= o->nparts) die("-P wants 0 <= I < N");
                break;
            }
            default: usage();
            }
            continue;
        }
        if (np >= npos) usage();
        pos[np++] = parse_num(a);
    }
    if (np != npos) usage();
}

/*
 * configure: set up everything for counting [lo, hi): range and bins (R), wheel (given or
 * auto_wheel), tables (tables_setup), sub-launch size G.sub, device tables and parameters
 * (dev_upload), chunk size and class range (all, -C or -P), as in a350826.c.
 *   In:     options o; lo < hi <= 2^80.  Needs dev_init() (G.bcap).
 *   Errors: die() for an empty or too large range, a -w that is not prime, B not above the
 *           wheel prime, bin boundaries that do not increase strictly inside (lo, hi), a class
 *           longer than 2^62 terms, too many medium or large primes, a -C range beyond the
 *           classes, an empty -P part, or a CUDA error.
 */
static void configure(opts_t *o, u128 lo, u128 hi)
{
    if (hi <= lo) die("HI must be larger than LO");
    if (hi > HIMAX) die("HI must be <= 2^80");
    R.lo = lo; R.hi = hi;
    R.base = lo > P0 ? lo : P0;
    const u32 w = o->w ? o->w : auto_wheel(lo, hi, o->B);
    wheel_setup(w);
    if (o->B <= W.w) die("sieving bound must exceed the wheel prime");
    R.bmM = (u64)(R.base % W.M);
    R.nbins = 0;
    R.bnd[R.nbins++] = lo;
    for (int i = 0; i < o->nb; i++) {
        if (o->bnd[i] <= R.bnd[R.nbins - 1] || o->bnd[i] >= hi) die("bin boundaries must increase inside (LO, HI)");
        R.bnd[R.nbins++] = o->bnd[i];
    }
    R.bnd[R.nbins] = hi;
    tables_setup(o->B, o->pmax, o->qw, R.base);
    const u128 span = R.hi > R.base ? (R.hi - 1 - R.base) / W.M : 0;
    if (span >> 62) die("class too long; use a larger wheel (-w)");
    g_empty = R.hi <= R.base;
    {                                         /* expected survivors per class -> classes per sub-launch */
        double sr = 1;
        for (int i = 0; i < SP.np; i++) sr *= 1.0 - 6.0 / SP.q[i];
        const double per = ((double)span + 1) * sr * 1.3 + 64;
        const double sub = (double)G.bcap / per;
        G.sub = sub < 1 ? 1 : sub > 65536 ? 65536 : (u64)sub;
    }
    dev_upload();
    D.chunk = o->chunk ? o->chunk : auto_chunk(W.C);
    D.sample = NULL;
    D.c0 = 0;
    D.c1 = W.C;
    if (o->cr1) {
        if (o->cr1 > W.C) die("-C range beyond the %" PRIu64 " classes of wheel %u", W.C, W.w);
        D.c0 = o->cr0;
        D.c1 = o->cr1;
    } else if (o->nparts) {
        const u64 nch = (W.C + D.chunk - 1) / D.chunk;
        const u64 a = (u64)((u128)nch * o->part / o->nparts), b = (u64)((u128)nch * (o->part + 1) / o->nparts);
        D.c0 = a * D.chunk;
        D.c1 = b * D.chunk < W.C ? b * D.chunk : W.C;
        if (D.c1 <= D.c0) die("part %" PRIu64 "/%" PRIu64 " is empty", o->part, o->nparts);
    }
}

/* decade_label: "A350826(n)" if [a, b) = [10^(n-1), 10^n) for some n <= 24, else NULL (static
   buffer). */
static const char *decade_label(u128 a, u128 b)
{
    static char buf[32];
    u128 x = 1;
    for (int n = 1; n <= 24; n++) {
        if (a == x && b == x * 10) { snprintf(buf, sizeof buf, "A350826(%d)", n); return buf; }
        x *= 10;
    }
    return NULL;
}

/* print_results: the result lines on stdout, as in a350826.c: "TAG [a, b) count N cks H" per
   bin (with the A350826(n) label for a decade), a total line for several bins, and the
   statistics line; tag = RESULT, PART (class range only) or PARTIAL (stopped early). */
static void print_results(const char *tag)
{
    u64 total = 0, cks = 0;
    for (int b = 0; b < R.nbins; b++) { total += D.tot.cnt[b]; cks += D.tot.cks[b]; }
    for (int b = 0; b < R.nbins; b++) {
        const char *lab = decade_label(R.bnd[b], R.bnd[b + 1]);
        printf("%s [%s, %s) count %" PRIu64 " cks %016" PRIx64 "%s%s\n", tag, u128s(R.bnd[b]), u128s(R.bnd[b + 1]),
               D.tot.cnt[b], D.tot.cks[b], lab ? "  " : "", lab ? lab : "");
    }
    if (R.nbins > 1)
        printf("%s [%s, %s) count %" PRIu64 " cks %016" PRIx64 "\n", tag, u128s(R.lo), u128s(R.hi), total, cks);
    printf("%s survivors %" PRIu64 " candidates %" PRIu64 " spsp %" PRIu64 " bits %" PRIu64 "\n", tag,
           D.tot.surv, D.tot.cand, D.tot.spsp, D.tot.bits);
    fflush(stdout);
}

/* print_banner: description of the run on stderr (device, wheel, expected candidates, sizes of
   the prime groups, chunks).  die() on a CUDA error. */
static void print_banner(const opts_t *o)
{
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, o->dev));
    fprintf(stderr, "[%s, %s) on %s: wheel %u (%" PRIu64 " classes, ~%.3g candidates), B = %u (%d pattern, %d medium, "
                    "%d large primes), %" PRIu64 " chunks of %" PRIu64 "\n",
            u128s(R.lo), u128s(R.hi), prop.name, W.w, W.C, (double)(R.hi - R.base) * rho_wheel(W.w), SP.B, SP.npat,
            SP.nmed, SP.nlarge, (D.c1 - D.c0 + D.chunk - 1) / D.chunk, D.chunk);
    if (D.c0 != 0 || D.c1 != W.C)
        fprintf(stderr, "classes [%" PRIu64 ", %" PRIu64 ") of %" PRIu64 " only (partial count)\n", D.c0, D.c1, W.C);
}

/*
 * cmd_count: `a350826_cuda count LO HI [options]`.
 *   Out:    RESULT or PART lines on stdout; returns 0 if all chunks were done, 1 if the run was
 *           stopped (PARTIAL lines; with -S the checkpoint is saved).
 *   Errors: die() for invalid arguments, a chunk log that cannot be opened or a CUDA error.
 */
static int cmd_count(int argc, char **argv)
{
    opts_t o;
    u128 pos[2];
    parse_opts(argc, argv, 2, &o, 2, pos);
    dev_init(o.dev);
    configure(&o, pos[0], pos[1]);
    if (o.log) {
        g_log = fopen(o.log, "a");
        if (!g_log) die("cannot open %s: %s", o.log, strerror(errno));
    }
    if (!o.quiet) print_banner(&o);
    run_stats_t st = run_count(o.state, o.interval, o.quiet);
    if (g_log) fclose(g_log);
    if (D.frontier < D.nchunks) { print_results("PARTIAL"); return 1; }
    print_results(D.c0 == 0 && D.c1 == W.C ? "RESULT" : "PART");
    char b1[32];
    if (!o.quiet)
        fprintf(stderr, "done in %s (%.3g bits/s)\n", fmt_dur(st.seconds, b1, sizeof b1),
                st.seconds > 0 ? (double)st.bits / st.seconds : 0);
    return 0;
}

/*
 * cmd_bench: `a350826_cuda bench LO HI [-n N] ...`: time N classes (default 20000) spread evenly
 * over all classes, each over the whole range, in chunks of up to 4096 classes after one
 * warm-up launch, and project the time of the full count.
 *   Out:    BENCH lines (rate, per-class statistics, projection); returns 0.  -X MASK leaves out
 *           parts of the work (see Usage in the header).
 *   Errors: die() for invalid arguments, out of memory or a CUDA error.
 */
static int cmd_bench(int argc, char **argv)
{
    opts_t o;
    u128 pos[2];
    parse_opts(argc, argv, 2, &o, 2, pos);
    dev_init(o.dev);
    configure(&o, pos[0], pos[1]);
    const u64 n = o.nsample < W.C ? o.nsample : W.C;
    u64 *smp = (u64 *)malloc(sizeof *smp * n);
    if (!smp) die("out of memory");
    for (u64 i = 0; i < n; i++) smp[i] = (u64)((u128)i * W.C / n);
    D.sample = smp;
    D.nsample = n;
    D.chunk = n < 4096 ? n : 4096;
    g_print = false;
    print_banner(&o);
    /* warm-up launch */
    {
        const u64 m = fill_chunk(0, G.hcls[0]);
        launch(0, m < 256 ? m : 256);
        CK(cudaStreamSynchronize(G.st[0]));
    }
    run_stats_t st = run_count(NULL, 60, true);
    const double scale = (double)W.C / (double)n;
    char b1[32];
    printf("BENCH w %u B %u: %.3f s for %" PRIu64 " classes, %.4g bits/s\n", W.w, SP.B, st.seconds, n,
           (double)st.bits / st.seconds);
    printf("BENCH per class: %.4g bits, %.4g survivors, %.4g candidates; survivor rate %.3e\n",
           (double)D.tot.bits / n, (double)D.tot.surv / n, (double)D.tot.cand / n,
           (double)D.tot.surv / (double)D.tot.bits);
    u64 found = 0;
    for (int b = 0; b < R.nbins; b++) found += D.tot.cnt[b];
    printf("BENCH projection for all %" PRIu64 " classes: %s (~%.4g sextuplets)\n", W.C,
           fmt_dur(st.seconds * scale, b1, sizeof b1), (double)found * scale);
    free(smp);
    D.sample = NULL;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Selftest                                                            */
/* ------------------------------------------------------------------ */

/* rnd: deterministic pseudo-random numbers for the selftest (fixed seed, reproducible). */
static u64 rng_state = 0x243F6A8885A308D3ULL;
static u64 rnd(void) { return mix64(rng_state += GOLD); }

/*
 * test_arith_gpu: the GPU's base-2 and Lucas tests against the host reference, on 65536
 * numbers: 12 base-2 strong pseudoprimes (A001262), 12 strong Lucas pseudoprimes (A217255),
 * the 3 largest primes below 2^64, and random odd numbers of 20 to 80 bits prime to 105.
 * The GPU base-2 test must agree with mr_ref(x, 2), the GPU Lucas test must accept every
 * prime and the 12 Lucas pseudoprimes, and no composite may pass both.
 *   Out:    number of failures, printed as FAIL lines (at most 10 of the comparisons).
 *   Debug:  A350826_NTEST = n tests only the first n numbers, A350826_FROM = f those from
 *           index f; not range-checked (n + f must not exceed 65536).
 */
static int test_arith_gpu(void)
{
    /* random odd numbers up to 2^80 (no factor <= 7), known pseudoprimes; compare with the host */
    enum { N = 1 << 16 };
    static u64 v[2 * N];
    static u8 out[N];
    static u128 xs[N];
    int n = 0;
    static const u64 fixed[] = {2047, 3277, 4033, 4681, 8321, 15841, 29341, 42799, 49141, 52633, 65281, 74665,
                                5459, 5777, 10877, 16109, 18971, 22499, 24569, 25199, 40309, 58519, 75077, 97439,
                                18446744073709551557ULL, 18446744073709551533ULL, 18446744073709551521ULL};
    for (size_t i = 0; i < sizeof fixed / sizeof *fixed; i++) xs[n++] = fixed[i];
    while (n < N) {
        const int bits = 20 + (int)(rnd() % 61);
        u128 x = ((u128)rnd() << 64 | rnd()) >> (128 - bits);
        x |= ((u128)1 << (bits - 1)) | 1;
        if (x % 3 == 0 || x % 5 == 0 || x % 7 == 0 || x < 50) continue;
        xs[n++] = x;
    }
    if (getenv("A350826_NTEST")) {
        n = atoi(getenv("A350826_NTEST"));
        if (getenv("A350826_FROM")) {
            const int f = atoi(getenv("A350826_FROM"));
            for (int i = 0; i < n; i++) xs[i] = xs[f + i];
        }
    }
    for (int i = 0; i < n; i++) { v[2 * i] = (u64)xs[i]; v[2 * i + 1] = (u64)(xs[i] >> 64); }
    u64 *dv;
    u8 *dout;
    CK(cudaMalloc(&dv, sizeof v));
    CK(cudaMalloc(&dout, sizeof out));
    CK(cudaMemcpy(dv, v, sizeof v, cudaMemcpyHostToDevice));
    test_kernel<<<(n + 255) / 256, 256>>>(dv, n, dout);
    CK(cudaGetLastError());
    CK(cudaMemcpy(out, dout, sizeof out, cudaMemcpyDeviceToHost));
    cudaFree(dv);
    cudaFree(dout);
    int bad = 0, primes = 0;
    for (int i = 0; i < n; i++) {
        const u128 x = xs[i];
        const bool p = is_prime_ref(x), s2 = mr_ref(x, 2);
        primes += p;
        const bool g2 = out[i] & 1, gl = (out[i] >> 1) & 1;
        if (g2 != s2) { if (bad < 10) printf("FAIL gpu sprp2(%s) = %d\n", u128s(x), g2); bad++; }
        if (p && !gl) { if (bad < 10) printf("FAIL gpu slprp(%s) rejects a prime\n", u128s(x)); bad++; }
        if (!p && g2 && gl) { if (bad < 10) printf("FAIL gpu BPSW accepts composite %s\n", u128s(x)); bad++; }
    }
    /* known strong Lucas pseudoprimes must pass slprp */
    for (int i = 12; i < 24; i++) if (!((out[i] >> 1) & 1)) { printf("FAIL gpu slprp(%s) should pass\n", u128s(xs[i])); bad++; }
    printf("gpu arithmetic: %s (%d numbers, %d primes)\n", bad ? "FAILED" : "ok", n, primes);
    return bad;
}

/* A350826(1..17) as in the OEIS in 2022 (index 0 unused). */
static const u64 A350826[] = {0, 1, 1, 0, 0, 3, 0, 13, 64, 235, 1296, 7013, 41782, 253420, 1607418, 10520883,
                              70785653, 488096844};

/* windows computed by verify_a350826.py (files in verify/): count, cks */
typedef struct { const char *lo, *hi; u64 count, cks; } window_t;
static const window_t WINDOWS[] = {
    {"1e15", "1e15+2e12", 20102, 0x698702d7f0805d7dULL},
    {"1e17-1e12", "1e17+1e12", 9407, 0xb42353cfea760f5fULL},
    {"1e18-1e12", "1e18+1e12", 6983, 0x67b3c4c4b0bfba36ULL},
    {"1e19-1e12", "1e19+1e12", 4881, 0x7b9d093b40d59fb3ULL},
    {"2^64-1e12", "2^64+1e12", 4488, 0xfb6f2691fb75e673ULL},
    {"1e20-1e12", "1e20+1e12", 3644, 0x398796b7104319cbULL},
    {"1e22", "1e22+2e12", 2059, 0x004f163c8720cbb2ULL},
    {NULL, NULL, 0, 0}
};

/* run_quiet: selftest helper: count [lo, hi) quietly with wheel w (0 = auto), bound B, the
   default -p 256 and -Q 4096, and class range [c0, c1) (c1 = 0: all); *count, *cks = totals. */
static void run_quiet(u128 lo, u128 hi, u32 w, u32 B, u64 c0, u64 c1, u64 *count, u64 *cks)
{
    opts_t o;
    memset(&o, 0, sizeof o);
    o.w = w; o.B = B; o.pmax = 256; o.qw = 4096; o.interval = 60;
    o.cr0 = c0; o.cr1 = c1;
    configure(&o, lo, hi);
    run_count(NULL, 60, true);
    *count = 0; *cks = 0;
    for (int b = 0; b < R.nbins; b++) { *count += D.tot.cnt[b]; *cks += D.tot.cks[b]; }
}

/*
 * cmd_selftest: `a350826_cuda selftest [N] [-d DEV]`: test_arith_gpu, A350826(1..N) against the
 * OEIS data (N <= 17, default 14; the larger decades use several segments per class), and the
 * windows of verify_a350826.py (members across 2^64 and up to 1e22: the 96- and 128-bit
 * paths).  All three groups of sieving primes take part with the default -p and -Q.
 *   Out:    one line per test; returns 0 if everything passed, else 1.
 *   Not covered: the survivor-buffer overflow path (redo_in_pieces), other -p/-Q/-B values,
 *           and survivor counts; those are compared with the CPU tool by `a350826 recheck` on
 *           chunk logs.
 *   Errors: die() for N outside 1..17 or a CUDA error.
 */
static int cmd_selftest(int argc, char **argv)
{
    opts_t o;
    u128 pos[1] = {14};
    int npos = argc > 2 && isdigit((unsigned char)argv[2][0]) ? 1 : 0;
    parse_opts(argc, argv, 2, &o, npos, pos);
    const int N = (int)pos[0];
    if (N < 1 || N > 17) die("N must be 1..17");
    dev_init(o.dev);
    g_print = false;
    const double t0 = now();
    int bad = test_arith_gpu();
    u128 lo = 1;
    for (int n = 1; n <= N; n++) {
        u64 c, k;
        run_quiet(lo, lo * 10, 0, 1u << 16, 0, 0, &c, &k);
        const bool ok = c == A350826[n];
        printf("A350826(%d) = %" PRIu64 " %s (cks %016" PRIx64 ", w %u, %.1fs)\n", n, c, ok ? "ok" : "FAIL", k, W.w,
               now() - t0);
        fflush(stdout);
        bad += !ok;
        lo *= 10;
    }
    for (size_t i = 0; WINDOWS[i].lo; i++) {
        u64 c, k;
        run_quiet(parse_num(WINDOWS[i].lo), parse_num(WINDOWS[i].hi), 0, 1u << 16, 0, 0, &c, &k);
        const bool ok = c == WINDOWS[i].count && k == WINDOWS[i].cks;
        printf("window [%s, %s): %" PRIu64 " %016" PRIx64 " %s\n", WINDOWS[i].lo, WINDOWS[i].hi, c, k, ok ? "ok" : "FAIL");
        fflush(stdout);
        bad += !ok;
    }
    char b1[32];
    printf("selftest %s (%s)\n", bad ? "FAILED" : "passed", fmt_dur(now() - t0, b1, sizeof b1));
    return bad ? 1 : 0;
}

/*
 * cmd_mrbench: `a350826_cuda mrbench X [N]` (debug): time round_kernel (round 0) on N random odd
 * numbers near X (default 2^24) with grids of 192 to 1536 blocks, on device 0.
 *   Out:    one timing line per grid size; returns 0.
 *   Errors: not checked: N must not exceed G.bcap = 2^25 (the device buffer would overflow), and
 *           the host allocation is not tested.
 */
static int cmd_mrbench(int argc, char **argv)      /* round-kernel throughput on random odd numbers near X */
{
    if (argc < 3) usage();
    const u128 x0 = parse_num(argv[2]);
    const u64 n = argc > 3 ? (u64)parse_num(argv[3]) : (1ULL << 24);
    dev_init(0);
    u64 *h = (u64 *)malloc(sizeof(u64) * 2 * n);
    for (u64 i = 0; i < n; i++) {
        const u128 v = (x0 + ((u128)(rnd() >> 8) << 1)) | 1;
        h[2 * i] = (u64)v; h[2 * i + 1] = (u64)(v >> 64);
    }
    CK(cudaMemcpy(G.dbuf[0][0], h, sizeof(u64) * 2 * n, cudaMemcpyHostToDevice));
    Cnt c;
    memset(&c, 0, sizeof c);
    c.nb[0] = n;
    for (int grid = 48 * 4; grid <= 48 * 32; grid *= 2) {
        CK(cudaMemcpy(G.dcnt[0], &c, sizeof c, cudaMemcpyHostToDevice));
        CK(cudaDeviceSynchronize());
        const double t0 = now();
        round_kernel<<<grid, 256>>>(0, G.dbuf[0][0], G.dbuf[0][1], G.dcnt[0], G.bcap);
        CK(cudaDeviceSynchronize());
        const double t = now() - t0;
        Cnt r;
        CK(cudaMemcpy(&r, G.dcnt[0], sizeof r, cudaMemcpyDeviceToHost));
        printf("grid %d: %.3f s, %.3g tests/s, %" PRIu64 " passed\n", grid, t, (double)n / t, (u64)r.nb[1]);
    }
    free(h);
    return 0;
}

/*
 * cmd_num: `a350826_cuda num X [X ...]` (debug): GPU base-2 and Lucas tests of each X
 * (test_kernel on device 0) next to the host's base-2 test and reference primality.
 *   Out:    one line per number; returns 0.
 *   Errors: die() for a malformed number, X >= 2^81 (is_prime_ref) or a CUDA error; X should
 *           be odd and > 41 (test_kernel).
 */
static int cmd_num(int argc, char **argv)          /* debug: GPU tests of single numbers */
{
    dev_init(0);
    for (int i = 2; i < argc; i++) {
        const u128 x = parse_num(argv[i]);
        u64 v[2] = {(u64)x, (u64)(x >> 64)}, *dv;
        u8 out = 0, *dout;
        CK(cudaMalloc(&dv, sizeof v));
        CK(cudaMalloc(&dout, 1));
        CK(cudaMemcpy(dv, v, sizeof v, cudaMemcpyHostToDevice));
        test_kernel<<<1, 1>>>(dv, 1, dout);
        CK(cudaGetLastError());
        CK(cudaMemcpy(&out, dout, 1, cudaMemcpyDeviceToHost));
        printf("%s: gpu sprp2 %d slprp %d | host mr2 %d prime %d\n", u128s(x), out & 1, (out >> 1) & 1,
               (int)mr_ref(x, 2), (int)is_prime_ref(x));
        fflush(stdout);
        cudaFree(dv); cudaFree(dout);
    }
    return 0;
}

/* main: dispatch on the command word (count, bench, selftest, num, mrbench); stdout is
   line-buffered so results appear at once when redirected to a file. */
int main(int argc, char **argv)
{
    stderr_tty = isatty(2);
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 2) usage();
    if (!strcmp(argv[1], "count")) return cmd_count(argc, argv);
    if (!strcmp(argv[1], "bench")) return cmd_bench(argc, argv);
    if (!strcmp(argv[1], "selftest")) return cmd_selftest(argc, argv);
    if (!strcmp(argv[1], "num")) return cmd_num(argc, argv);
    if (!strcmp(argv[1], "mrbench")) return cmd_mrbench(argc, argv);
    usage();
    return 2;
}
