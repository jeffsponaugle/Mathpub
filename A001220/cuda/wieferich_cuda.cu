/*
 * wieferich_cuda.cu  --  multi-base search for Wieferich primes, CUDA version
 *                        (DGX Spark / GB10, sm_121, CUDA 13)
 *
 * Same search as ../wieferich.c -- primes p with b^(p-1) == 1 (mod p^2) for many bases b --
 * with the same Fermat quotients, checksums, chunk log and checkpoint format.  The THEORY
 * OF OPERATION in ../wieferich.c (the problem, the heuristic, what a scan proves, the
 * result checks, checksums, frontier and checkpoints) applies unchanged and is not repeated
 * here.  A checkpoint written by either tool can be resumed by the other, and their chunk
 * logs agree line by line.  This program found A306256(4) = 303632117562967 (base 30) on
 * Oct 01 2026.
 *
 * ==========================================================================
 * THEORY OF OPERATION (GPU-specific parts)
 * ==========================================================================
 *
 * 1. Work decomposition
 *
 *    One GPU thread per (prime, base) pair.  A chunk is split into T sub-ranges, one per
 *    host sieve thread; each sub-range's primes are stored as 32-bit offsets from its start
 *    and tested by one launch of k_wieferich with grid (ceil(n / 256), number of bases).
 *    blockIdx.y selects the base, so all threads of a block share b, its window w and its
 *    power table (copied to shared memory); control flow is uniform except where exponent
 *    lengths differ (primes on both sides of a power of two).
 *
 * 2. Arithmetic mod N = p^2 in radix 2^26 (msqr26, redc26)
 *
 *    N < 2^126 is held as five 26-bit limbs and R = 2^130.  Limb products (< 2^53) are
 *    summed in 64-bit column accumulators, so each partial product is one IMAD.WIDE.U32
 *    (32x32 + 64 -> 64) and carries are propagated once per column (a shift and a mask).
 *    Montgomery REDC works word by word in radix 2^26 with n' = -N^-1 mod 2^26.  A squaring
 *    is 15 products (cross products use pre-doubled limbs) plus 25 for REDC: 40 IMAD.WIDE and
 *    ~83 instructions in all, with no register moves (checked in the SASS).  Measured on
 *    GB10: IMAD.WIDE issues at half the rate of a 32-bit IMAD, FFMA at twice it; this form
 *    beat a radix-2^32 version with per-product carry handling by 1.22x on an idle GPU, with
 *    bit-identical results on 8e8 values.
 *
 *    Lazy reduction: values are kept in [0, 3N).  REDC of the square of such a value is
 *    < (9 N^2 + R N) / R < 2N because 9N < R, so a squaring needs no final subtraction;
 *    only the conversion out of Montgomery form (redc26) reduces fully.
 *
 * 3. Multiplication by a small constant (mul_small26)
 *
 *    As in the CPU tool the multiplier c = b^d is a plain integer, here c < 2^16, and the
 *    quotient floor(x c / N) < 3 * 2^16 is estimated -- in single precision, because GB10
 *    runs FP64 at 1/64 of the FP32 rate -- from the top limbs of x c and of N, scaled by
 *    2^(-26 (L-1)) so that both fit the float range (L = the top nonzero limb of N).  The
 *    relative error, below 2^-21.7, keeps the estimate within 1 of the true quotient.
 *    Subtracting (estimate - 1) N (or 0 N when the estimate is 0) leaves a result in
 *    [0, 3N) with no sign test.  The window w is the largest <= 4 with b^(2^w - 1) < 2^16:
 *    4 for b = 2, 3 for b = 3, 2 for b <= 40, 1 for b <= 65535, the largest base accepted.
 *
 * 4. Per-thread setup (mod_init26, mod_one26)
 *
 *    N = p^2 from one 64x64 product, split into 26-bit limbs; pinv = p^-1 mod 2^64 by Newton
 *    iteration (as in the CPU tool); n' = -(pinv^2) mod 2^26; L; invD = the float reciprocal
 *    of N's top limbs at the scale of section 3.  The Montgomery form of 1, 2^130 mod N, is
 *    built from 2^(bits(N) - 1) < N by multiplications with 2^15 (2-3 of them in production,
 *    at most 9) instead of a division.
 *
 * 5. Results on the device
 *
 *    q = (r - 1) p^-1 mod 2^64, the FLT check q < p and the checksum term mix64(q ^ p*GOLD)
 *    are as in the CPU tool.  Per-thread checksum terms and test counts are summed with warp
 *    shuffles and shared memory, and each block adds its sums with one 64-bit atomicAdd per
 *    base.  Solutions, near-Wieferich primes and FLT errors are rare; they go to an event
 *    buffer (at most MAXEV per chunk) through an atomic counter, and the host sorts them by
 *    (p, base) before printing, so the output does not depend on thread scheduling.
 *
 * 6. Host pipeline (run_scan, sieve_chunk, enqueue_chunk, fold_chunk)
 *
 *    Chunks are processed strictly in order with two buffer sets, set = chunk index & 1:
 *        sieve chunk k+1 on the host (T threads, primesieve)   while the GPU runs chunk k;
 *        fold chunk k: wait for its event, copy back sums and events, print, log, frontier;
 *        enqueue chunk k+1: memset of its result block, memcpys and launches on one stream,
 *        then an event.
 *    A buffer set is refilled only after the chunk that used it two steps earlier has been
 *    folded, so host writes never race with device reads.  Since chunks complete in order,
 *    the frontier is the number of folded chunks and a solution is CONFIRMED as soon as its
 *    chunk is folded.  The checkpoint is written between chunks; after SIGINT/SIGTERM the
 *    chunk already on the GPU is still folded, then the scan stops.
 *    The CPU and the GPU of a GB10 share one power budget: heavy CPU load lowers the GPU clock
 *    (12 busy threads: 2366 -> 2060 MHz), so CPU work alongside a run (dcheck.sh) should be
 *    pinned to a few efficiency cores (CPUs 0-4 and 10-14 on a Spark).
 *
 * 7. Validation (selftest)
 *
 *    (0) 20,000 GPU Fermat quotients of random primes < 2^63 with 64 bases (all window sizes)
 *    against the host's shift-and-add reference; (1) [0, 10^9) for the 23 non-power bases
 *    2..30 must give pi(10^9) and exactly the 65 known solutions; (2) windows of +-10^6 around
 *    the 11 known solutions above 10^9 must re-find each one; (3) the checksum windows of
 *    ../wieferich_cks.h (verify_wieferich.py).  Production chunks are also re-checked on the
 *    CPU with ../dcheck.sh.
 *
 * Known issues
 *
 *    Deliberately not fixed here, so that this source stays code-identical to the binary the
 *    paused production queue on atom2 resumes with.  1-8 are fixed in ../wieferich.c (Oct 13
 *    2026) and can be ported when the search is restarted.
 *    1. parse_term: "1^E" or "0^E" and "0eE" with a huge E loop for a very long time; a
 *       mantissa without digits ("e5") is accepted as 0; an over-long B in "B^1" gives
 *       2^64 - 1; trailing zeros after a decimal point ("1.000...e15") can cause a false
 *       overflow.
 *    2. parse_bases counts duplicate bases against the limit of 64.
 *    3. save_state does not check ferror before fclose, so a write error in an intermediate
 *       flush could replace a good checkpoint with a partial file; a failed write leaves
 *       FILE.tmp behind.
 *    4. load_state calls fmemopen without a NULL check and does not verify that every base
 *       has a "cks" line (a missing line silently resumes that base's checksum from 0).
 *    5. record_hit silently drops solutions beyond MAXHITS = 4096 from the summary and the
 *       checkpoint (they are still printed).
 *    6. -n values above 2^63 - 1 become negative and silently disable NEAR reporting.
 *    7. check: an even P is not rejected (the arithmetic then uses an even modulus and the
 *       output is meaningless); for an odd composite P every base reports "DISAGREES",
 *       because the exact-division quotient differs from the floor quotient.
 *    8. bench: N < 2 gives NaN/inf projections, and a run of zero measured seconds divides
 *       by zero; run_scan's st.primes includes a resumed part (bench never resumes).
 *    9. More than MAXEV = 65536 events in one chunk (only possible with a large -n T or a
 *       flood of FLT errors) drops the excess events -- solutions included -- with only a
 *       warning at the end; S.flt and S.near then undercount.  Production runs use no -n.
 *   10. KNOWN lacks A306256(4) = 303632117562967, so this selftest does not re-find it (the
 *       CPU selftest does).
 *   11. Performance: every chunk creates fresh primesieve iterators per host thread, whose
 *       setup (sieving primes up to sqrt(hi)) dominates small chunks near 2^63; the
 *       selftest's checksum window at 2^63 takes ~25 s for this reason.
 *
 * Usage (as the CPU tool; host threads instead of CPU workers)
 *   wieferich_cuda scan [START] END [-b BASES] [-c CHUNK] [-n T] [-S FILE] [-i SECS] [-L FILE] [-T N] [-g DEV] [-q]
 *       Default CHUNK 10^10 (END is rounded up to a chunk boundary; CHUNK / T must be
 *       < 2^32); -T N (or -t N) host sieve threads, default all cores; -g DEV the CUDA
 *       device.  Exit status 3 if any FLT error occurred.
 *   wieferich_cuda check P [-b BASES]
 *       GPU Fermat quotients of P for each base, compared with the host reference.
 *   wieferich_cuda bench [N] [-d SPAN] [-b BASES] [-c CHUNK] [-T N]
 *       Time [N, N+SPAN) (defaults 10^15, 2*10^11) and project longer scans.
 *   wieferich_cuda selftest [LIMIT] [-T N]
 *       The checks of section 7.
 *   Bases must be in [2, 65535] (the windows of section 3 need b^1 < 2^16).
 *
 * Build: make  (nvcc -O3 -std=c++17 -arch=sm_121 wieferich_cuda.cu -lprimesieve -lpthread)
 */

#include <cstdio>
#include <cstddef>
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
#include <thread>
#include <vector>
#include <atomic>
#include <unistd.h>
#include <time.h>
#include <primesieve.h>
#include <cuda_runtime.h>

typedef uint64_t u64;
typedef uint32_t u32;
typedef unsigned __int128 u128;

#define MAXB    64
#define MAXW    4
#define BLOCK   256
#define MAXEV   65536
#define MAXHITS 4096
#define GOLD    0x9E3779B97F4A7C15ULL
#define PMAX    (1ULL << 63)

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) \
    die("CUDA error %s at %s:%d: %s", cudaGetErrorName(e_), __FILE__, __LINE__, cudaGetErrorString(e_)); } while (0)

static bool stderr_tty;
static volatile sig_atomic_t g_stop = 0;

/* ------------------------------------------------------------------ */
/* Generic helpers (as in ../wieferich.c)                              */
/* ------------------------------------------------------------------ */

/*
 * now: monotonic wall clock.
 *   In:     nothing.
 *   Out:    seconds since an arbitrary fixed point (CLOCK_MONOTONIC), as a double.
 *   Errors: none.
 */
static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

/*
 * die: print an error message and exit (also used by CUDA_CHECK).
 *   In:     printf-style format and arguments.
 *   Out:    does not return; clears the status line on a terminal, prints the message and a
 *           newline on stderr, exits with status 1.
 *   Errors: none.  Work on the GPU and in flight is lost; the last checkpoint stays valid.
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
 * fmt_dur: format a duration for the status line ("12.3s", "4m05s", "2h03m04s", "3d04h05m").
 *   In:     s seconds; buf, n: output buffer.
 *   Out:    buf (truncated to n bytes).
 *   Errors: "?" for negative, NaN or absurd (> 10^12 s) values.
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
 *   Out:    true and *r = a*b if the product fits in 64 bits, false otherwise.
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
 * parse_term: parse one term of a command-line number: decimal, B^E, or MeE with an optional
 * decimal point in M (1e15, 1.2e15).
 *   In:     s, n: the term's characters (need not be NUL-terminated).
 *   Out:    true and *out = the value on success.
 *   Errors: false for empty or over-long (>= 64 characters) terms, stray characters, values
 *           that do not fit in 64 bits, and non-integer MeE values.  See Known issues 1 for
 *           the inputs this copy still mishandles.
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
        char *dot = strchr(buf, '.');
        size_t ip = dot ? (size_t)(dot - buf) : strlen(buf);
        const char *frac = dot ? dot + 1 : "";
        if ((ip && !all_digits(buf, ip)) || (*frac && !all_digits(frac, strlen(frac)))) return false;
        if (!all_digits(c + 1, strlen(c + 1))) return false;
        long e = strtol(c + 1, 0, 10) - (long)strlen(frac);
        u64 m = 0;
        for (size_t i = 0; i < ip; i++) if (!mul_ok(m, 10, &m) || (m += (u64)(buf[i] - '0')) < (u64)(buf[i] - '0')) return false;
        for (const char *f = frac; *f; f++) if (!mul_ok(m, 10, &m) || (m += (u64)(*f - '0')) < (u64)(*f - '0')) return false;
        for (; e > 0; e--) if (!mul_ok(m, 10, &m)) return false;
        for (; e < 0; e++) { if (m % 10) return false; m /= 10; }
        *out = m;
        return true;
    }
    if (!all_digits(buf, n)) return false;
    errno = 0;
    *out = strtoull(buf, 0, 10);
    return errno == 0;
}

/*
 * parse_num: parse a command-line number: a term or TERM+TERM / TERM-TERM.
 *   In:     s, e.g. "2e14", "2^63-25", "1e15+4e6".
 *   Out:    the value.
 *   Errors: dies for a bad term, more than one operator, an overflowing sum and a negative
 *           difference; a '+' or '-' directly after 'e'/'E' is not an operator.
 */
static u64 parse_num(const char *s)
{
    size_t n = strlen(s);
    for (size_t i = n; i-- > 1; ) {
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
 * parse_int: parse a small integer option value within [lo, hi].
 *   In:     s, the range, the option's name for the message.
 *   Out:    the value.
 *   Errors: dies if s is not a number or lies outside [lo, hi].
 */
static int parse_int(const char *s, int lo, int hi, const char *what)
{
    u64 v = parse_num(s);
    if (v < (u64)lo || v > (u64)hi) die("%s must be between %d and %d", what, lo, hi);
    return (int)v;
}

/*
 * default_threads: number of online CPUs (the default host sieve thread count).
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
 * mix64: the splitmix64 finalizer (host and device): checksum terms mix64(q ^ p*GOLD) and the
 * selftest's random numbers.  Identical in ../wieferich.c and verify_wieferich.py.
 *   In:     z.
 *   Out:    mix64(z).
 *   Errors: none.
 */
static inline __host__ __device__ u64 mix64(u64 z)
{
    z += GOLD;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* ---- host reference arithmetic (independent of the GPU code) ---- */

/*
 * mulmod64 / powmod64: (a*b) mod m and a^e mod m with 128-bit intermediates (host only; used
 * by is_prime64).
 *   In:     m >= 1.
 *   Out:    the residue.
 *   Errors: m = 0 divides by zero (never passed).
 */
static u64 mulmod64(u64 a, u64 b, u64 m) { return (u64)(((u128)a * b) % m); }

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
 * is_prime64: deterministic primality test for 64-bit n (host only; check and selftest).
 *   In:     n.
 *   Out:    true iff n is prime: trial division by the primes up to 37, then strong
 *           probable-prime tests to those 12 bases, exact for all n < 3.18e23.
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
    while (!(d & 1)) { d >>= 1; s++; }
    for (size_t i = 0; i < sizeof small / sizeof *small; i++) {
        u64 x = powmod64(small[i], d, n);
        if (x == 1 || x == n - 1) continue;
        bool comp = true;
        for (int r = 1; r < s && comp; r++) {
            x = mulmod64(x, x, n);
            if (x == n - 1) comp = false;
        }
        if (comp) return false;
    }
    return true;
}

/*
 * mulmod_ref: a*b mod m by shift and add -- the host reference arithmetic, independent of the
 * GPU code.
 *   In:     a, b, 1 <= m < 2^127.
 *   Out:    (a*b) mod m.
 *   Errors: m = 0 divides by zero; m >= 2^127 can overflow.  Callers use m = p^2 < 2^126.
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
 * powmod_ref: b^e mod m with mulmod_ref (reference for check and selftest).
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

/* Known solutions (OEIS; R. Fischer's table) for the bases 2..30 that are not perfect powers
 * (keyword "known"; the selftest's reference data).  See Known issues 10. */
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
    {30, 3, {7, 160541, 94727075783ULL}},
};
#define NKNOWN ((int)(sizeof KNOWN / sizeof *KNOWN))

/* The bases of the run (host copy): sorted, deduplicated, with windows and power tables
 * b^d for d < 2^w (unused entries 0). */
static struct {
    int nb;
    u32 b[MAXB];
    int w[MAXB];
    u32 pw[MAXB][1 << MAXW];
} B;

/* Device copies of B (bases_setup uploads them): base, window and power table per base. */
__constant__ u32 c_b[MAXB];
__constant__ int c_w[MAXB];
__constant__ u32 c_pw[MAXB][1 << MAXW];

/*
 * window_for: the GPU exponent window for base b (theory section 3).
 *   In:     b >= 2.
 *   Out:    the largest w <= MAXW (4) with b^(2^w - 1) < 2^16, at least 1.
 *   Errors: none; for b >= 2^16 it returns 1 although b does not fit mul_small26 (bases_setup
 *           rejects such bases).
 */
static int window_for(u32 b)
{
    for (int w = MAXW; w > 1; w--) {
        u64 x = 1;
        bool ok = true;
        for (int i = 0; i < (1 << w) - 1 && ok; i++) {
            x *= b;
            if (x >> 16) ok = false;
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
 * bases_setup: make list[0..n) the bases of the run and upload them to constant memory.
 *   In:     list (any order, duplicates allowed), 1 <= n <= MAXB.  gpu_init must have run.
 *   Out:    B sorted and deduplicated, with windows and power tables (unused entries 0);
 *           c_b, c_w, c_pw on the device.
 *   Errors: dies for n outside [1, MAXB], a base outside [2, 65535], or a CUDA error.
 */
static void bases_setup(const u32 *list, int n)
{
    u32 tmp[MAXB];
    if (n < 1 || n > MAXB) die("between 1 and %d bases are supported", MAXB);
    memcpy(tmp, list, (size_t)n * sizeof *tmp);
    qsort(tmp, (size_t)n, sizeof *tmp, cmp_u32);
    memset(&B, 0, sizeof B);
    for (int i = 0; i < n; i++) {
        if (tmp[i] < 2 || tmp[i] >= 65536) die("bases must be in [2, 65535] for the GPU tool");
        if (B.nb && B.b[B.nb - 1] == tmp[i]) continue;
        int j = B.nb++;
        B.b[j] = tmp[i];
        B.w[j] = window_for(tmp[i]);
        u64 x = 1;
        for (int d = 0; d < (1 << MAXW); d++) {
            B.pw[j][d] = d < (1 << B.w[j]) ? (u32)x : 0;
            x *= tmp[i];
        }
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_b, B.b, sizeof B.b));
    CUDA_CHECK(cudaMemcpyToSymbol(c_w, B.w, sizeof B.w));
    CUDA_CHECK(cudaMemcpyToSymbol(c_pw, B.pw, sizeof B.pw));
}

/*
 * parse_bases: parse a -b argument (numbers, ranges LO-HI, "oeis", "known") and set up B.
 *   In:     s, comma-separated.
 *   Out:    B set (bases_setup).
 *   Errors: dies for an over-long list, bad numbers, bases outside [2, 65535], a reversed
 *           range, or more than MAXB entries counting duplicates (Known issues 2).
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
            for (size_t i = 0; i < sizeof OEIS_BASES / sizeof *OEIS_BASES; i++) {
                if (n >= MAXB) die("too many bases (max %d)", MAXB);
                list[n++] = OEIS_BASES[i];
            }
            continue;
        }
        if (!strcmp(tok, "known")) {
            for (int i = 0; i < NKNOWN; i++) {
                if (n >= MAXB) die("too many bases (max %d)", MAXB);
                list[n++] = KNOWN[i].b;
            }
            continue;
        }
        char *dash = strchr(tok, '-');
        u64 lo, hi;
        if (dash) { *dash = 0; lo = parse_num(tok); hi = parse_num(dash + 1); }
        else lo = hi = parse_num(tok);
        if (lo < 2 || hi < lo || hi >= 65536) die("bad base or range '%s'", tok);
        for (u64 b = lo; b <= hi; b++) {
            if (n >= MAXB) die("too many bases (max %d)", MAXB);
            list[n++] = (u32)b;
        }
    }
    bases_setup(list, n);
}

/*
 * anum_for: the OEIS A-number of base b's sequence, or NULL.
 *   In:     b.
 *   Out:    "A001220" for b = 2, the OEIS_ANUM entry for the "oeis" bases, else NULL.
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
 * print_bases: write the bases as "b1,b2,..." (the checkpoint's format).
 *   In:     f.
 *   Out:    the list on f, no newline.
 *   Errors: none reported.
 */
static void print_bases(FILE *f)
{
    for (int j = 0; j < B.nb; j++) fprintf(f, "%s%u", j ? "," : "", B.b[j]);
}

/* ------------------------------------------------------------------ */
/* Device arithmetic mod N = p^2 (5 x 26-bit limbs, R = 2^130)          */
/* ------------------------------------------------------------------ */

/* x = x0 + x1 2^26 + ... + x4 2^104, limbs < 2^26, N < 2^126.  Column sums are 64-bit
 * accumulators, so every limb product is one IMAD.WIDE.U32 and carries are handled once
 * per column.  Values are kept lazily in [0, 3N): Montgomery squaring maps inputs < 3N to
 * outputs < 2N because 9N < R, and the small multiplication returns [0, 3N) without a sign
 * fix-up.  Only the final conversion out of Montgomery form reduces fully.  (On GB10,
 * IMAD.WIDE issues at half the IMAD rate; this radix-2^26 form beat a radix-2^32 one with
 * per-product carry handling by 1.3x, with bit-identical results.) */

/* A rare per-pair result reported to the host: type 1 = solution, 2 = near-Wieferich prime,
 * 3 = FLT error; j = base index; q = the Fermat quotient (or the wrong value for type 3). */
struct Event { u32 type, j; u64 p, q; };           /* type 1 solution, 2 near, 3 FLT error */

/* Per-chunk results on the device: per-base checksum sums, number of tests, and the event
 * buffer with its counter (nev may exceed MAXEV; only the first MAXEV events are stored).
 * The header (everything before ev) is cleared before each chunk and copied back after it. */
struct ChunkRes {
    unsigned long long cks[MAXB];
    unsigned long long tests;
    unsigned int nev, pad;
    Event ev[MAXEV];
};
#define RES_HEADER (offsetof(ChunkRes, ev))

#define M26 0x3FFFFFFu

/* Per-thread constants for arithmetic mod N = p^2 in radix 2^26 (theory sections 2-4). */
struct Mod26 {
    u32 n[5];       /* limbs of N */
    u32 ninv;       /* -N^-1 mod 2^26 */
    int L;          /* index of the top nonzero limb of N (3 or 4 in production) */
    float invD;     /* 1 / (sum_{i >= L-1} n_i 2^(26(i-L+1))) */
};

/*
 * msqr26: Montgomery squaring, x = x^2 / 2^130 mod N, radix 2^26 (theory section 2).
 *   In:     x with normalized limbs (< 2^26) and value < 3N; m for N < 2^126.
 *   Out:    x = x^2 R^-1 mod N up to a multiple of N, value < 2N, limbs normalized.
 *   Errors: none for the stated ranges.  Column sums stay below 2^56: products are < 2^53
 *           (limbs < 2^26, doubled limbs < 2^27) and at most a few are summed per column.
 */
__device__ __forceinline__ void msqr26(u32 x[5], const Mod26 &m)
{
    const u32 d1 = x[1] << 1, d2 = x[2] << 1, d3 = x[3] << 1, d4 = x[4] << 1;
    u64 c0 = (u64)x[0] * x[0];
    u64 c1 = (u64)x[0] * d1;
    u64 c2 = (u64)x[0] * d2 + (u64)x[1] * x[1];
    u64 c3 = (u64)x[0] * d3 + (u64)x[1] * d2;
    u64 c4 = (u64)x[0] * d4 + (u64)x[1] * d3 + (u64)x[2] * x[2];
    u64 c5 = (u64)x[1] * d4 + (u64)x[2] * d3;
    u64 c6 = (u64)x[2] * d4 + (u64)x[3] * x[3];
    u64 c7 = (u64)x[3] * d4;
    u64 c8 = (u64)x[4] * x[4];
    u32 q;
    q = ((u32)c0 * m.ninv) & M26;
    c0 += (u64)q * m.n[0]; c1 += (u64)q * m.n[1]; c2 += (u64)q * m.n[2]; c3 += (u64)q * m.n[3]; c4 += (u64)q * m.n[4];
    c1 += c0 >> 26;
    q = ((u32)c1 * m.ninv) & M26;
    c1 += (u64)q * m.n[0]; c2 += (u64)q * m.n[1]; c3 += (u64)q * m.n[2]; c4 += (u64)q * m.n[3]; c5 += (u64)q * m.n[4];
    c2 += c1 >> 26;
    q = ((u32)c2 * m.ninv) & M26;
    c2 += (u64)q * m.n[0]; c3 += (u64)q * m.n[1]; c4 += (u64)q * m.n[2]; c5 += (u64)q * m.n[3]; c6 += (u64)q * m.n[4];
    c3 += c2 >> 26;
    q = ((u32)c3 * m.ninv) & M26;
    c3 += (u64)q * m.n[0]; c4 += (u64)q * m.n[1]; c5 += (u64)q * m.n[2]; c6 += (u64)q * m.n[3]; c7 += (u64)q * m.n[4];
    c4 += c3 >> 26;
    q = ((u32)c4 * m.ninv) & M26;
    c4 += (u64)q * m.n[0]; c5 += (u64)q * m.n[1]; c6 += (u64)q * m.n[2]; c7 += (u64)q * m.n[3]; c8 += (u64)q * m.n[4];
    c5 += c4 >> 26;
    x[0] = (u32)c5 & M26; c6 += c5 >> 26;
    x[1] = (u32)c6 & M26; c7 += c6 >> 26;
    x[2] = (u32)c7 & M26; c8 += c7 >> 26;
    x[3] = (u32)c8 & M26;
    x[4] = (u32)(c8 >> 26);
}

/*
 * mul_small26: x = x * c mod N up to a multiple of N, for a plain constant c (section 3).
 *   In:     x normalized with value < 3N; c < 2^16; m.
 *   Out:    x = x*c - q'N with q' = (estimate of floor(x*c/N)) - 1 (0 stays 0).  q' is at most
 *           the true quotient and at least it minus 2, so the value lies in [0, 3N) and no
 *           sign test is needed; limbs normalized (signed carries handle negative columns).
 *   Errors: none for the stated ranges; c >= 2^16 or x >= 3N can make the estimate off by
 *           more than 1 and the result wrong (window_for keeps every b^d < 2^16).
 */
__device__ __forceinline__ void mul_small26(u32 x[5], u32 c, const Mod26 &m)
{
    const u64 t0 = (u64)x[0] * c, t1 = (u64)x[1] * c, t2 = (u64)x[2] * c, t3 = (u64)x[3] * c, t4 = (u64)x[4] * c;
    float T;
    switch (m.L) {
    case 4:  T = __ull2float_rn(t4) * 0x1p26f + __ull2float_rn(t3) + __ull2float_rn(t2) * 0x1p-26f; break;
    case 3:  T = __ull2float_rn(t4) * 0x1p52f + __ull2float_rn(t3) * 0x1p26f + __ull2float_rn(t2) + __ull2float_rn(t1) * 0x1p-26f; break;
    case 2:  T = __ull2float_rn(t4) * 0x1p78f + __ull2float_rn(t3) * 0x1p52f + __ull2float_rn(t2) * 0x1p26f + __ull2float_rn(t1) + __ull2float_rn(t0) * 0x1p-26f; break;
    default: T = __ull2float_rn(t4) * 0x1p104f + __ull2float_rn(t3) * 0x1p78f + __ull2float_rn(t2) * 0x1p52f + __ull2float_rn(t1) * 0x1p26f + __ull2float_rn(t0); break;
    }
    u32 q = __float2uint_rz(T * m.invD);
    q -= q != 0;
    long long r0 = (long long)t0 - (long long)((u64)q * m.n[0]);
    long long r1 = (long long)t1 - (long long)((u64)q * m.n[1]);
    long long r2 = (long long)t2 - (long long)((u64)q * m.n[2]);
    long long r3 = (long long)t3 - (long long)((u64)q * m.n[3]);
    long long r4 = (long long)t4 - (long long)((u64)q * m.n[4]);
    r1 += r0 >> 26; x[0] = (u32)r0 & M26;
    r2 += r1 >> 26; x[1] = (u32)r1 & M26;
    r3 += r2 >> 26; x[2] = (u32)r2 & M26;
    r4 += r3 >> 26; x[3] = (u32)r3 & M26;
    x[4] = (u32)r4;
}

/*
 * redc26: leave Montgomery form and reduce fully, x = x / 2^130 mod N.
 *   In:     x normalized with value < 3N.
 *   Out:    x R^-1 mod N in [0, N), limbs normalized: the REDC result is at most N, and one
 *           conditional subtraction of N (done with signed carries) finishes.
 *   Errors: none for the stated ranges.
 */
__device__ __forceinline__ void redc26(u32 x[5], const Mod26 &m)
{
    u64 c0 = x[0], c1 = x[1], c2 = x[2], c3 = x[3], c4 = x[4], c5 = 0, c6 = 0, c7 = 0, c8 = 0;
    u32 q;
    q = ((u32)c0 * m.ninv) & M26;
    c0 += (u64)q * m.n[0]; c1 += (u64)q * m.n[1]; c2 += (u64)q * m.n[2]; c3 += (u64)q * m.n[3]; c4 += (u64)q * m.n[4];
    c1 += c0 >> 26;
    q = ((u32)c1 * m.ninv) & M26;
    c1 += (u64)q * m.n[0]; c2 += (u64)q * m.n[1]; c3 += (u64)q * m.n[2]; c4 += (u64)q * m.n[3]; c5 += (u64)q * m.n[4];
    c2 += c1 >> 26;
    q = ((u32)c2 * m.ninv) & M26;
    c2 += (u64)q * m.n[0]; c3 += (u64)q * m.n[1]; c4 += (u64)q * m.n[2]; c5 += (u64)q * m.n[3]; c6 += (u64)q * m.n[4];
    c3 += c2 >> 26;
    q = ((u32)c3 * m.ninv) & M26;
    c3 += (u64)q * m.n[0]; c4 += (u64)q * m.n[1]; c5 += (u64)q * m.n[2]; c6 += (u64)q * m.n[3]; c7 += (u64)q * m.n[4];
    c4 += c3 >> 26;
    q = ((u32)c4 * m.ninv) & M26;
    c4 += (u64)q * m.n[0]; c5 += (u64)q * m.n[1]; c6 += (u64)q * m.n[2]; c7 += (u64)q * m.n[3]; c8 += (u64)q * m.n[4];
    c5 += c4 >> 26;
    x[0] = (u32)c5 & M26; c6 += c5 >> 26;
    x[1] = (u32)c6 & M26; c7 += c6 >> 26;
    x[2] = (u32)c7 & M26; c8 += c7 >> 26;
    x[3] = (u32)c8 & M26;
    x[4] = (u32)(c8 >> 26);
    /* conditional subtraction of N (x <= N here) */
    long long s0 = (long long)x[0] - m.n[0];
    long long s1 = (long long)x[1] - m.n[1] + (s0 >> 26);
    long long s2 = (long long)x[2] - m.n[2] + (s1 >> 26);
    long long s3 = (long long)x[3] - m.n[3] + (s2 >> 26);
    long long s4 = (long long)x[4] - m.n[4] + (s3 >> 26);
    if (s4 >= 0) {
        x[0] = (u32)s0 & M26; x[1] = (u32)s1 & M26; x[2] = (u32)s2 & M26; x[3] = (u32)s3 & M26; x[4] = (u32)s4;
    }
}

/*
 * mod_init26: per-thread setup for p (theory section 4).
 *   In:     p odd with 3 <= p < 2^63.
 *   Out:    m (limbs of N = p^2, n' = -N^-1 mod 2^26, L, invD); returns p^-1 mod 2^64.
 *   Errors: none detected; an even p gives a meaningless n' (callers never pass one in scans;
 *           check does not reject it, Known issues 7).
 */
__device__ __forceinline__ u64 mod_init26(Mod26 &m, u64 p)
{
    /* N = p^2 = hi 2^64 + lo, split into 26-bit limbs n0..n4 (n4 < 2^22 since N < 2^126) */
    const u64 lo = p * p, hi = __umul64hi(p, p);
    m.n[0] = (u32)lo & M26;
    m.n[1] = (u32)(lo >> 26) & M26;
    m.n[2] = (u32)((lo >> 52) | (hi << 12)) & M26;
    m.n[3] = (u32)(hi >> 14) & M26;
    m.n[4] = (u32)(hi >> 40);
    u64 x = (3 * p) ^ 2;
    x *= 2 - p * x; x *= 2 - p * x; x *= 2 - p * x; x *= 2 - p * x;
    const u32 x32 = (u32)x;
    m.ninv = (0u - x32 * x32) & M26;
    m.L = m.n[4] ? 4 : m.n[3] ? 3 : m.n[2] ? 2 : m.n[1] ? 1 : 0;
    float D;
    switch (m.L) {
    case 4:  D = __uint2float_rn(m.n[4]) * 0x1p26f + __uint2float_rn(m.n[3]) + __uint2float_rn(m.n[2]) * 0x1p-26f; break;
    case 3:  D = __uint2float_rn(m.n[3]) * 0x1p26f + __uint2float_rn(m.n[2]) + __uint2float_rn(m.n[1]) * 0x1p-26f; break;
    case 2:  D = __uint2float_rn(m.n[2]) * 0x1p26f + __uint2float_rn(m.n[1]) + __uint2float_rn(m.n[0]) * 0x1p-26f; break;
    case 1:  D = __uint2float_rn(m.n[1]) * 0x1p26f + __uint2float_rn(m.n[0]); break;
    default: D = __uint2float_rn(m.n[0]); break;
    }
    /* D = N / 2^(26(L-1)) for L >= 2 and D = N for L <= 1, matching the scaling of T in mul_small26 */
    m.invD = 1.0f / D;
    return x;
}

/*
 * mod_one26: x = 2^130 mod N (up to a multiple of N, value < 3N): the Montgomery form of 1.
 *   In:     m from mod_init26.
 *   Out:    x, built from 2^(bits(N) - 1) < N by multiplications with 2^15 and one smaller
 *           power of two (mul_small26), 131 - bits(N) doublings in all.
 *   Errors: none.
 */
__device__ __forceinline__ void mod_one26(u32 x[5], const Mod26 &m)
{
    const u32 top = m.L == 4 ? m.n[4] : m.L == 3 ? m.n[3] : m.L == 2 ? m.n[2] : m.L == 1 ? m.n[1] : m.n[0];
    const int bits = 26 * m.L + 32 - __clz(top);
    const int lb = (bits - 1) / 26, bb = (bits - 1) % 26;
    x[0] = lb == 0 ? 1u << bb : 0;
    x[1] = lb == 1 ? 1u << bb : 0;
    x[2] = lb == 2 ? 1u << bb : 0;
    x[3] = lb == 3 ? 1u << bb : 0;
    x[4] = lb == 4 ? 1u << bb : 0;
    int k = 131 - bits;
    /* 2^(bits-1) * 2^(131-bits) = 2^130; multipliers stay below 2^16 as mul_small26 requires */
    while (k >= 15) { mul_small26(x, 1u << 15, m); k -= 15; }
    if (k) mul_small26(x, 1u << k, m);
}

/*
 * fermat_q26: the Fermat quotient of one (p, b) pair on the device.
 *   In:     p odd, 3 <= p < 2^63, not dividing b; w and pw = the base's window and its
 *           power table b^0..b^(2^w - 1) (each < 2^16, in shared or constant memory).
 *   Out:    q = (b^(p-1) mod p^2 - 1) * p^-1 mod 2^64; flt = (q >= p), i.e. Fermat's little
 *           theorem failed and the residue was wrong.  q = 0 means p is a solution.
 *   Errors: reported through flt only.
 */
__device__ __forceinline__ u64 fermat_q26(u64 p, int w, const u32 *pw, bool &flt)
{
    Mod26 m;
    const u64 pinv = mod_init26(m, p);
    u32 x[5];
    mod_one26(x, m);
    const u64 e = p - 1;
    const u32 mask = (1u << w) - 1;
    /* shift of the top window; then per window: w squarings and one multiplication by b^digit
       (b^0 = 1 for a zero digit), exactly as in ../wieferich.c's run_group */
    int sh = ((64 - __clzll((long long)e) + w - 1) / w - 1) * w;
    mul_small26(x, pw[(e >> sh) & mask], m);
    for (sh -= w; sh >= 0; sh -= w) {
        for (int s = 0; s < w; s++) msqr26(x, m);
        mul_small26(x, pw[(e >> sh) & mask], m);
    }
    redc26(x, m);
    /* only r mod 2^64 is needed: q = (r - 1) p^-1 mod 2^64 is an exact division by p */
    const u64 r = (u64)x[0] | ((u64)x[1] << 26) | ((u64)x[2] << 52);
    const u64 q = (r - 1) * pinv;
    flt = q >= p;
    return q;
}

/*
 * push_event: append an event to the chunk's buffer.
 *   In:     res, the event's fields.
 *   Out:    stored at the slot returned by an atomic increment of res->nev.
 *   Errors: beyond MAXEV events the event is dropped (nev still counts it; the host warns at
 *           the end of the scan, Known issues 9).
 */
__device__ __forceinline__ void push_event(ChunkRes *res, u32 type, u32 j, u64 p, u64 q)
{
    const u32 k = atomicAdd(&res->nev, 1u);
    if (k < MAXEV) { res->ev[k].type = type; res->ev[k].j = j; res->ev[k].p = p; res->ev[k].q = q; }
}

/*
 * k_wieferich: test one sub-range of primes against every base (theory sections 1, 5).
 *   In:     off[0..n): prime offsets from base (the sub-range start); res = the chunk's result
 *           block; nearT = the -n bound (0 = off).  Grid (ceil(n / BLOCK), number of bases),
 *           BLOCK threads; blockIdx.y is the base index.
 *   Out:    per thread i < n with p = base + off[i] not dividing b: one test.  The block's
 *           checksum terms and test count are added to res->cks[j] and res->tests;
 *           solutions, near-Wieferich primes (0 < |A| <= nearT) and FLT errors are pushed as
 *           events.  Threads with i >= n or p | b contribute nothing.
 *   Errors: FLT errors become events (type 3) and are left out of the checksum.  All threads
 *           take part in the shuffles and barriers, so no thread returns early.
 */
__global__ void __launch_bounds__(BLOCK)
k_wieferich(const u32 *__restrict__ off, u32 n, u64 base, ChunkRes *res, u64 nearT)
{
    const int j = blockIdx.y;
    __shared__ u32 s_pw[1 << MAXW];
    __shared__ unsigned long long s_sum[BLOCK / 32];
    __shared__ u32 s_cnt[BLOCK / 32];
    if (threadIdx.x < (1 << MAXW)) s_pw[threadIdx.x] = c_pw[j][threadIdx.x];
    __syncthreads();
    const u32 i = blockIdx.x * BLOCK + threadIdx.x;
    const u32 b = c_b[j];
    u64 term = 0;
    u32 cnt = 0;
    if (i < n) {
        const u64 p = base + off[i];
        /* pairs with p | b are skipped (b^(p-1) = 0 mod p); they count neither as tests nor in the checksum */
        if (!(p <= b && b % p == 0)) {
            bool flt;
            const u64 q = fermat_q26(p, c_w[j], s_pw, flt);
            cnt = 1;
            if (flt) {
                push_event(res, 3, j, p, q);
            } else {
                term = mix64(q ^ (p * GOLD));
                if (q == 0) push_event(res, 1, j, p, 0);
                else if (nearT) {
                    const u64 a = q <= p / 2 ? q : p - q;
                    if (a <= nearT) push_event(res, 2, j, p, q);
                }
            }
        }
    }
#pragma unroll
    /* block reduction: warp shuffles, then one partial sum per warp in shared memory */
    for (int o = 16; o; o >>= 1) {
        term += __shfl_down_sync(0xffffffffu, term, o);
        cnt += __shfl_down_sync(0xffffffffu, cnt, o);
    }
    if ((threadIdx.x & 31) == 0) { s_sum[threadIdx.x >> 5] = term; s_cnt[threadIdx.x >> 5] = cnt; }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long t = 0;
        u32 c = 0;
        for (int k = 0; k < BLOCK / 32; k++) { t += s_sum[k]; c += s_cnt[k]; }
        if (c) {
            atomicAdd(&res->cks[j], t);
            atomicAdd(&res->tests, (unsigned long long)c);
        }
    }
}

/*
 * k_quot: Fermat quotients for arbitrary (p, base index) pairs (check, selftest).
 *   In:     ps[i], js[i] for i < n (p odd, not dividing the base); the bases' tables in
 *           constant memory.
 *   Out:    qs[i] = q and flts[i] = FLT failure flag, as fermat_q26.
 *   Errors: none beyond the flag.
 */
__global__ void k_quot(const u64 *ps, const int *js, u64 *qs, u32 *flts, u32 n)
{
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    bool flt;
    qs[i] = fermat_q26(ps[i], c_w[js[i]], c_pw[js[i]], flt);
    flts[i] = flt;
}

/* ------------------------------------------------------------------ */
/* Host: sieve threads, GPU pipeline, scan state                        */
/* ------------------------------------------------------------------ */

/* A solution: base, prime, and the chunk it was found in. */
typedef struct { u32 b; u64 p; u64 chunk; bool announced; } hit_t;

/* The scan (host): range, chunk size, frontier (= number of folded chunks), totals over
 * the folded chunks, checksums, solutions, and the event-overflow flag (Known issues 9).
 * Only the main thread touches it, so no lock is needed. */
static struct {
    u64 start, chunk, end, nchunks;
    u64 frontier;
    u64 primes, tests, near, flt;
    u64 cks[MAXB];
    int nhits;
    hit_t hits[MAXHITS];
    bool ev_overflow;
} S;

/* Options: -n bound (0 = off), print result lines (off in selftest), chunk log, host threads. */
static long long g_near = 0;
static bool g_print = true;
static FILE *g_log = NULL;
static int g_hthreads = 0;

/* The double-buffered host/GPU pipeline (theory section 6).  Index [s] is the buffer set
 * s = chunk index & 1: T pinned host buffers and one device buffer of T * cap prime offsets,
 * a device result block, per-sub-range prime counts and start values, whether the chunk
 * contains the prime 2, the chunk's start and index, and its completion event. */
struct Pipe {
    int T = 0;                          /* host sieve threads = sub-ranges per chunk */
    u64 cap = 0;                        /* capacity per sub-range buffer (primes) */
    std::vector<u32 *> hbuf[2];         /* pinned, T per set */
    u32 *dbuf[2] = {NULL, NULL};        /* device, T * cap per set */
    ChunkRes *dres[2] = {NULL, NULL};
    std::vector<u64> cnt[2], sub_lo[2];
    bool has2[2] = {false, false};
    u64 lo[2] = {0, 0}, idx[2] = {0, 0};
    cudaStream_t st;
    cudaEvent_t done[2];
    std::vector<Event> evbuf;
    bool init = false;
} P;

/*
 * sub_cap: capacity of a sub-range buffer, in primes.
 *   In:     lo = the lowest number the buffers will hold, len = sub-range length.
 *   Out:    1.25 len / ln(max(lo, 10^4)) + 10^5, a safe upper bound on the number of primes in
 *           any interval of length len at or above lo (pi(x + y) - pi(x) ~ y / ln x).
 *   Errors: none; if a sub-range ever held more primes, sieve_chunk dies rather than truncate.
 */
static u64 sub_cap(u64 lo, u64 len)
{
    double l = log((double)(lo > 10000 ? lo : 10000));
    return (u64)(1.25 * (double)len / l) + 100000;
}

/*
 * pipe_alloc: (re)allocate the pipeline for T sub-ranges of capacity cap.
 *   In:     T >= 1 host threads, cap primes per sub-range.
 *   Out:    P with 2 x T pinned host buffers, 2 device buffers, 2 result blocks, the stream
 *           and the events.  Kept if T is unchanged and cap is large enough (repeated scans
 *           in selftest/bench); otherwise the prime buffers are freed and reallocated.
 *   Errors: dies on any CUDA allocation error (CUDA_CHECK).  Chunk 10^10 with T = 20 at 2e14
 *           needs ~3 GB of pinned host memory and ~3 GB on the device.
 */
static void pipe_alloc(int T, u64 cap)
{
    if (P.init && P.T == T && P.cap >= cap) return;
    if (P.init) {
        for (int s = 0; s < 2; s++) {
            for (u32 *h : P.hbuf[s]) cudaFreeHost(h);
            cudaFree(P.dbuf[s]);
        }
    } else {
        CUDA_CHECK(cudaStreamCreate(&P.st));
        for (int s = 0; s < 2; s++) {
            CUDA_CHECK(cudaEventCreateWithFlags(&P.done[s], cudaEventBlockingSync));
            CUDA_CHECK(cudaMalloc(&P.dres[s], sizeof(ChunkRes)));
        }
        P.evbuf.resize(MAXEV);
    }
    P.T = T;
    P.cap = cap;
    for (int s = 0; s < 2; s++) {
        P.hbuf[s].assign(T, NULL);
        for (int i = 0; i < T; i++) CUDA_CHECK(cudaMallocHost(&P.hbuf[s][i], cap * sizeof(u32)));
        CUDA_CHECK(cudaMalloc(&P.dbuf[s], (size_t)T * cap * sizeof(u32)));
        P.cnt[s].assign(T, 0);
        P.sub_lo[s].assign(T, 0);
    }
    P.init = true;
}

/*
 * sieve_chunk: generate the primes of chunk idx into buffer set s (host).
 *   In:     idx < S.nchunks; s = idx & 1, whose previous chunk has been folded.
 *   Out:    T threads each sieve one sub-range [a, b) of the chunk with their own primesieve
 *           iterator and store p - a as 32-bit offsets; P.cnt[s], P.sub_lo[s], P.has2[s]
 *           (the prime 2 is kept out of the buffers), P.lo[s], P.idx[s].
 *   Errors: dies if a sub-range holds more primes than its capacity (cannot happen for the
 *           sub_cap bound; reported, never truncated).  Offsets fit 32 bits because run_scan
 *           requires CHUNK / T < 2^32.  See Known issues 11 for the iterator setup cost.
 */
static void sieve_chunk(u64 idx, int s)
{
    const u64 lo = S.start + idx * S.chunk;
    const u64 hi = lo + S.chunk < S.end ? lo + S.chunk : S.end;
    const int T = P.T;
    std::atomic<bool> overflow(false), has2(false);
    std::vector<std::thread> th;
    for (int i = 0; i < T; i++) {
        const u64 a = lo + (u64)((u128)(hi - lo) * (u64)i / (u64)T);
        const u64 b = lo + (u64)((u128)(hi - lo) * (u64)(i + 1) / (u64)T);
        P.sub_lo[s][i] = a;
        th.emplace_back([=, &overflow, &has2]() {
            u64 k = 0;
            if (b > a) {
                primesieve_iterator it;
                primesieve_init(&it);
                primesieve_jump_to(&it, a, b);
                u32 *out = P.hbuf[s][i];
                const u64 cap = P.cap;
                for (;;) {
                    const u64 p = primesieve_next_prime(&it);
                    if (p >= b) break;
                    if (p == 2) { has2 = true; continue; }
                    if (k == cap) { overflow = true; break; }
                    out[k++] = (u32)(p - a);
                }
                primesieve_free_iterator(&it);
            }
            P.cnt[s][i] = k;
        });
    }
    for (auto &t : th) t.join();
    if (overflow) die("prime buffer overflow in chunk %" PRIu64 " (capacity %" PRIu64 ")", idx, P.cap);
    P.has2[s] = has2;
    P.lo[s] = lo;
    P.idx[s] = idx;
}

/*
 * enqueue_chunk: queue the GPU work for buffer set s on the stream.
 *   In:     s, filled by sieve_chunk.
 *   Out:    on P.st: clear the result header, then per non-empty sub-range one async copy of
 *           its offsets and one k_wieferich launch; finally record P.done[s].
 *   Errors: dies on a launch or API error (CUDA_CHECK); errors inside a kernel surface when
 *           fold_chunk waits for the event.
 */
static void enqueue_chunk(int s)
{
    CUDA_CHECK(cudaMemsetAsync(P.dres[s], 0, RES_HEADER, P.st));
    for (int i = 0; i < P.T; i++) {
        const u64 n = P.cnt[s][i];
        if (!n) continue;
        u32 *d = P.dbuf[s] + (size_t)i * P.cap;
        CUDA_CHECK(cudaMemcpyAsync(d, P.hbuf[s][i], n * sizeof(u32), cudaMemcpyHostToDevice, P.st));
        dim3 grid((unsigned)((n + BLOCK - 1) / BLOCK), (unsigned)B.nb);
        k_wieferich<<<grid, BLOCK, 0, P.st>>>(d, (u32)n, P.sub_lo[s][i], P.dres[s], (u64)g_near);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(P.done[s], P.st));
}

/*
 * record_hit: remember and print a solution (host, main thread only).
 *   In:     base b, prime p, chunk idx.
 *   Out:    appended to S.hits unless present; "FOUND base b p P (Annnnnn)" on stdout if
 *           g_print.
 *   Errors: duplicates are ignored; beyond MAXHITS solutions are printed but not kept
 *           (Known issues 5).
 */
static void record_hit(u32 b, u64 p, u64 idx)
{
    for (int i = 0; i < S.nhits; i++) if (S.hits[i].b == b && S.hits[i].p == p) return;
    if (S.nhits < MAXHITS) {
        hit_t *h = &S.hits[S.nhits++];
        h->b = b; h->p = p; h->chunk = idx; h->announced = false;
    }
    if (g_print) {
        if (stderr_tty) fputs("\r\033[K", stderr);
        const char *a = anum_for(b);
        printf("FOUND base %u p %" PRIu64 "%s%s%s\n", b, p, a ? " (" : "", a ? a : "", a ? ")" : "");
        fflush(stdout);
    }
}

/*
 * announce: print CONFIRMED lines for solutions below the frontier, once each.
 *   In:     nothing (reads S).
 *   Out:    the number of solutions below the frontier.
 *   Errors: none.
 */
static int announce(void)
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
                       h->b, h->p, a ? " (" : "", a ? a : "", a ? ")" : "", S.start, S.start + S.frontier * S.chunk);
                fflush(stdout);
            }
        }
    }
    return n;
}

/*
 * ev_cmp: qsort comparator ordering events by prime, then base index.
 *   In:     pointers to two Event.
 *   Out:    -1, 0 or 1.
 *   Errors: none.
 */
static int ev_cmp(const void *a, const void *b)
{
    const Event *x = (const Event *)a, *y = (const Event *)b;
    if (x->p != y->p) return x->p < y->p ? -1 : 1;
    return x->j < y->j ? -1 : x->j > y->j;
}

/*
 * fold_chunk: wait for the chunk in buffer set s and fold its results (theory section 6).
 *   In:     s; the chunk in it must be the next one at the frontier.
 *   Out:    waits for P.done[s], copies back the result header and the events, prints
 *           FOUND (also for the prime 2 when b = 1 mod 4), NEAR and FLT-ERROR lines sorted by
 *           (p, base), adds primes, tests, near, flt and checksums to S, writes the chunk-log
 *           line, and advances the frontier by one.
 *   Errors: dies on a CUDA error (including faults inside the kernels) and, as an internal
 *           consistency check, if the chunk is not the one at the frontier.  An event
 *           overflow sets S.ev_overflow (Known issues 9).
 */
static void fold_chunk(int s)
{
    CUDA_CHECK(cudaEventSynchronize(P.done[s]));
    ChunkRes h;
    CUDA_CHECK(cudaMemcpy(&h, P.dres[s], RES_HEADER, cudaMemcpyDeviceToHost));
    u32 nev = h.nev < MAXEV ? h.nev : MAXEV;
    if (h.nev > MAXEV) S.ev_overflow = true;
    if (nev) CUDA_CHECK(cudaMemcpy(P.evbuf.data(), (char *)P.dres[s] + RES_HEADER, nev * sizeof(Event), cudaMemcpyDeviceToHost));
    qsort(P.evbuf.data(), nev, sizeof(Event), ev_cmp);
    const u64 idx = P.idx[s];
    if (P.has2[s]) for (int j = 0; j < B.nb; j++) if (B.b[j] % 4 == 1) record_hit(B.b[j], 2, idx);
    u64 near = 0, flt = 0;
    for (u32 k = 0; k < nev; k++) {
        const Event &e = P.evbuf[k];
        if (e.type == 1) record_hit(B.b[e.j], e.p, idx);
        else if (e.type == 2) {
            near++;
            long long A = e.q <= e.p / 2 ? (long long)e.q : -(long long)(e.p - e.q);
            if (g_print) {
                if (stderr_tty) fputs("\r\033[K", stderr);
                printf("NEAR base %u p %" PRIu64 " A %lld\n", B.b[e.j], e.p, A);
            }
        } else {
            flt++;
            if (stderr_tty) fputs("\r\033[K", stderr);
            printf("FLT-ERROR base %u p %" PRIu64 " q %" PRIu64 "\n", B.b[e.j], e.p, e.q);
        }
    }
    fflush(stdout);
    u64 primes = P.has2[s] ? 1 : 0;
    for (int i = 0; i < P.T; i++) primes += P.cnt[s][i];
    S.primes += primes;
    S.tests += h.tests;
    S.near += near;
    S.flt += flt;
    for (int j = 0; j < B.nb; j++) S.cks[j] += h.cks[j];
    if (g_log) {
        const u64 flo = S.start + idx * S.chunk;
        const u64 fhi = flo + S.chunk < S.end ? flo + S.chunk : S.end;
        fprintf(g_log, "C %" PRIu64 " %" PRIu64 " %" PRIu64, flo, fhi, primes);
        for (int j = 0; j < B.nb; j++) fprintf(g_log, " %u:%016llx", B.b[j], h.cks[j]);
        fputc('\n', g_log);
        fflush(g_log);
    }
    if (idx != S.frontier) die("internal error: chunk %" PRIu64 " folded at frontier %" PRIu64, idx, S.frontier);
    S.frontier = idx + 1;
}

/* ---- checkpoint file (same format as the CPU tool) ---- */

/*
 * save_state: write the checkpoint (same format as ../wieferich.c) as path.tmp, then rename.
 *   In:     path.
 *   Out:    path holds bases, start, chunk, end, frontier, totals, checksums and the
 *           solutions below the frontier, ending with "end".
 *   Errors: a warning if path.tmp cannot be created, written, closed or renamed; the scan
 *           continues and the previous checkpoint normally survives (Known issues 3).
 */
static void save_state(const char *path)
{
    char tmp[PATH_MAX];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "w");
    if (!f) { fprintf(stderr, "\nwarning: cannot write %s: %s\n", tmp, strerror(errno)); return; }
    fprintf(f, "wieferich checkpoint 1\nbases ");
    print_bases(f);
    fprintf(f, "\nstart %" PRIu64 "\nchunk %" PRIu64 "\nend %" PRIu64 "\nfrontier %" PRIu64
               "\nprimes %" PRIu64 "\ntests %" PRIu64 "\nnear %" PRIu64 "\nflt %" PRIu64 "\n",
            S.start, S.chunk, S.end, S.frontier, S.primes, S.tests, S.near, S.flt);
    for (int j = 0; j < B.nb; j++) fprintf(f, "cks %u %016" PRIx64 "\n", B.b[j], S.cks[j]);
    for (int i = 0; i < S.nhits; i++)
        if (S.hits[i].chunk < S.frontier)
            fprintf(f, "hit %u %" PRIu64 " %" PRIu64 "\n", S.hits[i].b, S.hits[i].p, S.hits[i].chunk);
    fputs("end\n", f);
    if (fclose(f) != 0 || rename(tmp, path) != 0)
        fprintf(stderr, "\nwarning: cannot update %s: %s\n", path, strerror(errno));
}

/*
 * load_state: resume from a checkpoint written by either tool.
 *   In:     path; S.start, S.chunk, S.nchunks and B must be set.
 *   Out:    false if path does not exist; true with the frontier, totals, checksums and
 *           solutions restored.
 *   Errors: dies if the file cannot be read, is not a checkpoint, is truncated, has a bad
 *           cks/hit line or a checksum for a base not in the run, or was written for other
 *           bases, START or CHUNK, or if its frontier lies beyond the new END (Known
 *           issues 4).
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
    bool complete = false;
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
            u32 b;
            u64 v;
            if (sscanf(line, "cks %u %" SCNx64, &b, &v) != 2) die("%s: bad cks line", path);
            int j = -1;
            for (int k = 0; k < B.nb; k++) if (B.b[k] == b) j = k;
            if (j < 0) die("%s: base %u is not in the current base list", path, b);
            S.cks[j] = v;
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
    FILE *m = fmemopen(cur, sizeof cur, "w");
    print_bases(m);
    fclose(m);
    if (strcmp(cur, bases) != 0) die("%s was written for bases %s, not %s", path, bases, cur);
    if (start != S.start || chunk != S.chunk)
        die("%s was written for start=%" PRIu64 " chunk=%" PRIu64 ", not start=%" PRIu64 " chunk=%" PRIu64,
            path, start, chunk, S.start, S.chunk);
    if (frontier > S.nchunks) die("%s is past the new END", path);
    (void)end;
    S.frontier = frontier;
    return true;
}

/*
 * on_sigint: SIGINT/SIGTERM handler; sets g_stop so run_scan stops queueing chunks, folds the
 * one in flight, saves the checkpoint and returns.
 *   In:     signal number (unused).
 *   Out:    g_stop = 1.
 *   Errors: none.
 */
static void on_sigint(int sig)
{
    (void)sig;
    g_stop = 1;
}

/* What run_scan did: seconds and numbers of this session; tests of this session; primes
 * include a resumed part (Known issues 8). */
typedef struct { double seconds; u64 primes, tests, numbers; } scan_stats_t;

/*
 * run_scan: test every prime in [start, end) for the bases in B (theory section 6).
 *   In:     start < end; chunk >= 1; state = checkpoint path or NULL; interval = seconds
 *           between checkpoints (and status lines when stderr is not a terminal); quiet.
 *           gpu_init and parse_bases/bases_setup must have run; g_hthreads (0 = all cores).
 *   Out:    S holds the complete result; the checkpoint is saved at the end; statistics as
 *           scan_stats_t.  END is rounded up to a chunk boundary (S.end).
 *   Errors: dies if start >= end, chunk = 0, the rounded END exceeds 2^63 or 64 bits,
 *           CHUNK / T >= 2^32, on allocation or CUDA errors, and for an unusable checkpoint.
 *           SIGINT/SIGTERM stop the scan after the chunk in flight with a saved checkpoint;
 *           g_stop stays set for later scans in the same process.
 */
static scan_stats_t run_scan(u64 start, u64 end, u64 chunk, const char *state, int interval, bool quiet)
{
    if (end <= start) die("END must be larger than START");
    if (chunk == 0) die("CHUNK must be positive");
    S.start = start;
    S.chunk = chunk;
    S.nchunks = (end - start + chunk - 1) / chunk;
    if (S.nchunks == 0 || (S.nchunks - 1) > (UINT64_MAX - start) / chunk) die("range too large");
    S.end = start + S.nchunks * chunk;
    if (S.end > PMAX || S.end < start) die("END must be <= 2^63 (after rounding up to the chunk size)");
    if (S.end != end && !quiet) fprintf(stderr, "note: END rounded up to the chunk boundary %" PRIu64 "\n", S.end);
    S.frontier = 0;
    S.primes = S.tests = S.near = S.flt = 0;
    memset(S.cks, 0, sizeof S.cks);
    S.nhits = 0;
    S.ev_overflow = false;
    if (state && load_state(state) && !quiet)
        fprintf(stderr, "resuming from %s: %" PRIu64 " of %" PRIu64 " chunks done (position %" PRIu64
                        ", %" PRIu64 " primes tested)\n",
                state, S.frontier, S.nchunks, S.start + S.frontier * S.chunk, S.primes);
    const int T = g_hthreads > 0 ? g_hthreads : default_threads();
    if (chunk / (u64)T >= (1ULL << 32)) die("CHUNK / host threads must be < 2^32");
    const u64 lo0 = S.start + S.frontier * S.chunk;
    pipe_alloc(T, sub_cap(lo0, (chunk + T - 1) / T));
    if (!quiet) {
        fprintf(stderr, "scanning [%" PRIu64 ", %" PRIu64 ") in %" PRIu64 " chunks of %" PRIu64
                        " (%d sieve threads), bases ", S.start, S.end, S.nchunks, S.chunk, T);
        print_bases(stderr);
        fputc('\n', stderr);
    }
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_sigint;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    const u64 f0 = S.frontier, tests0 = S.tests;
    const double t0 = now();
    double last_status = 0, last_ckpt = t0;
    char b1[32], b2[32];
    u64 next = S.frontier;              /* next chunk to sieve */
    int inflight = 0;
    /* pipeline: keep one chunk on the GPU while the host sieves the next (theory section 6) */
    if (next < S.nchunks) { sieve_chunk(next, next & 1); enqueue_chunk(next & 1); next++; inflight = 1; }
    while (inflight) {
        const u64 cur = S.frontier;     /* chunk being computed */
        bool queued = false;
        if (next < S.nchunks && !g_stop) { sieve_chunk(next, next & 1); queued = true; }
        fold_chunk((int)(cur & 1));
        inflight--;
        if (queued) { enqueue_chunk(next & 1); next++; inflight++; }
        announce();
        const double t = now();
        const u64 pos = S.start + S.frontier * S.chunk < S.end ? S.start + S.frontier * S.chunk : S.end;
        const double dt = t - t0;
        const double rate = dt > 0 ? (double)(S.frontier - f0) * (double)S.chunk / dt : 0;
        const bool show = stderr_tty ? (t - last_status >= 1.0) : (t - last_status >= interval);
        if (show && !quiet) {
            last_status = t;
            fprintf(stderr, "%s%s  pos %.6g (%.2f%%)  %.3g/s  %.3g tests/s  %.4g primes  ETA %s  found %d",
                    stderr_tty ? "\r\033[K" : "progress: ", fmt_dur(dt, b1, sizeof b1), (double)pos,
                    100.0 * (double)(pos - S.start) / (double)(S.end - S.start), rate,
                    dt > 0 ? (double)(S.tests - tests0) / dt : 0, (double)S.primes,
                    fmt_dur(rate > 0 ? (double)(S.end - pos) / rate : -1, b2, sizeof b2), S.nhits);
            if (g_near > 0) fprintf(stderr, "  near %" PRIu64, S.near);
            if (S.flt) fprintf(stderr, "  FLT-ERRORS %" PRIu64, S.flt);
            if (!stderr_tty) fputc('\n', stderr);
            fflush(stderr);
        }
        if (state && t - last_ckpt >= interval) { last_ckpt = t; save_state(state); }
    }
    const double t1 = now();
    if (stderr_tty && !quiet) fputs("\r\033[K", stderr);
    announce();
    if (state) save_state(state);
    if (S.ev_overflow) fprintf(stderr, "warning: event buffer overflow (more than %d events in a chunk)\n", MAXEV);
    if (g_stop && S.frontier < S.nchunks && !quiet)
        fprintf(stderr, "stopped at position %" PRIu64 " (%" PRIu64 " of %" PRIu64 " chunks complete)%s\n",
                S.start + S.frontier * S.chunk, S.frontier, S.nchunks, state ? ", checkpoint saved" : "");
    scan_stats_t st;
    st.seconds = t1 - t0;
    st.primes = S.primes;
    st.tests = S.tests - tests0;
    st.numbers = (S.frontier - f0) * S.chunk;
    return st;
}

/*
 * print_summary: "SUMMARY [START, position) primes tests flt_errors [near]" and one line per
 * base with its A-number, checksum and solutions below the frontier (stdout).  The totals
 * include a resumed part; the position is END for a complete scan.
 *   In:     nothing (reads S, B).
 *   Out:    the summary.
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

/*
 * gpu_quotients: run k_quot on a list of (p, base index) pairs.
 *   In:     ps, js of equal non-zero length n < 2^32; B uploaded.
 *   Out:    qs[i], flts[i] as fermat_q26.
 *   Errors: dies on any CUDA error.
 */
static void gpu_quotients(const std::vector<u64> &ps, const std::vector<int> &js, std::vector<u64> &qs, std::vector<u32> &flts)
{
    const u32 n = (u32)ps.size();
    u64 *dp, *dq;
    int *dj;
    u32 *df;
    CUDA_CHECK(cudaMalloc(&dp, n * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&dq, n * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&dj, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&df, n * sizeof(u32)));
    CUDA_CHECK(cudaMemcpy(dp, ps.data(), n * sizeof(u64), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dj, js.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    k_quot<<<(n + 127) / 128, 128>>>(dp, dj, dq, df, n);
    CUDA_CHECK(cudaGetLastError());
    qs.resize(n);
    flts.resize(n);
    CUDA_CHECK(cudaMemcpy(qs.data(), dq, n * sizeof(u64), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(flts.data(), df, n * sizeof(u32), cudaMemcpyDeviceToHost));
    cudaFree(dp); cudaFree(dq); cudaFree(dj); cudaFree(df);
}

/* ------------------------------------------------------------------ */
/* Commands                                                             */
/* ------------------------------------------------------------------ */

/*
 * usage: print the command summary and exit with status 2.
 *   In:     nothing.
 *   Out:    does not return.
 *   Errors: none.
 */
static void usage(void)
{
    fputs("usage: wieferich_cuda scan [START] END [-b BASES] [-c CHUNK] [-n T] [-S FILE] [-i SECS] [-L FILE] [-T N] [-g DEV] [-q]\n"
          "       wieferich_cuda check P [-b BASES]\n"
          "       wieferich_cuda bench [N] [-d SPAN] [-b BASES]\n"
          "       wieferich_cuda selftest [LIMIT]\n"
          "BASES: list/ranges like 3,5,7 or 2-30 (bases < 65536); 'oeis' (default) or 'known'\n", stderr);
    exit(2);
}

/* CUDA device (-g). */
static int g_dev = 0;

/*
 * gpu_init: select device g_dev and print its name, SM count and compute capability (unless
 * g_print is off).
 *   In:     g_dev.
 *   Out:    the device is current for all later CUDA calls.
 *   Errors: dies if the device does not exist or the driver fails (CUDA_CHECK).
 */
static void gpu_init(void)
{
    CUDA_CHECK(cudaSetDevice(g_dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, g_dev));
    if (!g_print) return;
    fprintf(stderr, "GPU %d: %s, %d SMs, compute %d.%d\n", g_dev, prop.name, prop.multiProcessorCount, prop.major, prop.minor);
}

/*
 * cmd_scan: "scan [START] END [options]" (see the header's Usage).
 *   In:     the arguments after "scan".
 *   Out:    runs the scan; FOUND/NEAR/FLT-ERROR/CONFIRMED lines and the summary on stdout, a
 *           timing line on stderr.  Exit status 0, or 3 if any FLT error occurred.
 *   Errors: usage() for unknown options or a missing END; dies for bad numbers or bases, an
 *           unwritable chunk log and every run_scan error (Known issues 6 for -n).
 */
static int cmd_scan(int argc, char **argv)
{
    u64 pos[2];
    int npos = 0, interval = 60;
    u64 chunk = 10000000000ULL;
    const char *state = NULL, *logf = NULL, *bases = "oeis";
    bool quiet = false;
    for (int i = 0; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-c") && i + 1 < argc) chunk = parse_num(argv[++i]);
        else if (!strcmp(a, "-b") && i + 1 < argc) bases = argv[++i];
        else if (!strcmp(a, "-n") && i + 1 < argc) g_near = (long long)parse_num(argv[++i]);
        else if (!strcmp(a, "-S") && i + 1 < argc) state = argv[++i];
        else if (!strcmp(a, "-L") && i + 1 < argc) logf = argv[++i];
        else if (!strcmp(a, "-i") && i + 1 < argc) interval = parse_int(argv[++i], 1, 86400, "interval");
        else if (!strcmp(a, "-T") && i + 1 < argc) g_hthreads = parse_int(argv[++i], 1, 1024, "host threads");
        else if (!strcmp(a, "-g") && i + 1 < argc) g_dev = parse_int(argv[++i], 0, 64, "device");
        else if (!strcmp(a, "-t") && i + 1 < argc) g_hthreads = parse_int(argv[++i], 1, 1024, "host threads");
        else if (!strcmp(a, "-q")) quiet = true;
        else if (a[0] != '-' && npos < 2) pos[npos++] = parse_num(a);
        else usage();
    }
    if (npos == 0) usage();
    const u64 start = npos == 2 ? pos[0] : 0, end = npos == 2 ? pos[1] : pos[0];
    gpu_init();
    parse_bases(bases);
    if (logf && !(g_log = fopen(logf, "a"))) die("cannot open %s: %s", logf, strerror(errno));
    scan_stats_t st = run_scan(start, end, chunk, state, interval, quiet);
    print_summary();
    if (!quiet) {
        char b1[32];
        fprintf(stderr, "%s this session, %.3g numbers/s\n", fmt_dur(st.seconds, b1, sizeof b1),
                st.seconds > 0 ? (double)st.numbers / st.seconds : 0);
    }
    if (g_log) fclose(g_log);
    return S.flt ? 3 : 0;
}

/*
 * cmd_check: "check P [-b BASES]": GPU Fermat quotients of P for each base not divisible by P,
 * compared with the host reference.
 *   In:     the arguments after "check"; 3 <= P < 2^63; bases default "known".
 *   Out:    per base: q, A, a WIEFERICH marker for q = 0, agreement with the reference.  Exit
 *           status 0 if all agree, 1 otherwise.
 *   Errors: usage() for a missing P; dies for P outside [3, 2^63), bad bases or CUDA errors.
 *           Even or composite P give misleading output (Known issues 7).
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
    if (p < 3 || p >= PMAX) die("P must be in [3, 2^63)");
    gpu_init();
    parse_bases(bases);
    printf("p = %" PRIu64 " (%s)\n", p, is_prime64(p) ? "prime" : "NOT prime: quotients are meaningless");
    std::vector<u64> ps;
    std::vector<int> js;
    for (int j = 0; j < B.nb; j++) if (B.b[j] % p) { ps.push_back(p); js.push_back(j); }
    std::vector<u64> qs;
    std::vector<u32> fl;
    if (!ps.empty()) gpu_quotients(ps, js, qs, fl);
    int bad = 0;
    for (size_t k = 0; k < ps.size(); k++) {
        const u32 b = B.b[js[k]];
        const u128 rr = powmod_ref(b, p - 1, (u128)p * p);
        const u64 qr = (u64)((rr - 1) / p), q = qs[k];
        const long long A = q <= p / 2 ? (long long)q : -(long long)(p - q);
        const bool ok = q == qr && !fl[k];
        if (!ok) bad++;
        printf("base %3u: q = %" PRIu64 "  A = %lld%s  [reference %s]\n", b, q, A,
               q == 0 ? "  <-- WIEFERICH" : "", ok ? "agrees" : "DISAGREES");
    }
    return bad ? 1 : 0;
}

/*
 * li_diff: li(y) - li(x), the integral of dt / ln t, by the midpoint rule on 2000 steps in
 * log t (an estimate of the number of primes in [x, y]).
 *   In:     1 < x < y.
 *   Out:    the integral.
 *   Errors: meaningless for x <= 1 (Known issues 8).
 */
static double li_diff(double x, double y)
{
    const int n = 2000;
    double lx = log(x), ly = log(y), h = (ly - lx) / n, s = 0;
    for (int i = 0; i < n; i++) {
        double u = lx + (i + 0.5) * h;
        s += exp(u) / u * h;
    }
    return s;
}

/*
 * cmd_bench: "bench [N] [-d SPAN] [-b BASES] [-c CHUNK] [-T N]": time [N, N+SPAN) and project.
 *   In:     the arguments after "bench"; defaults N = 10^15, SPAN = 2*10^11, CHUNK =
 *           min(SPAN / 8, 10^10) (at least 10^6).
 *   Out:    primes/s, tests/s and numbers/s, and for [N, 2N), [N, 10N), [N, 100N) the tests,
 *           the projected time and the heuristic expected number of new solutions.
 *   Errors: usage() for unknown options; dies for bad numbers or bases and every run_scan
 *           error; see Known issues 8.
 */
static int cmd_bench(int argc, char **argv)
{
    u64 n0 = 1000000000000000ULL, span = 200000000000ULL, chunk = 0;
    const char *bases = "oeis";
    bool haven = false;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "-d") && i + 1 < argc) span = parse_num(argv[++i]);
        else if (!strcmp(argv[i], "-b") && i + 1 < argc) bases = argv[++i];
        else if (!strcmp(argv[i], "-T") && i + 1 < argc) g_hthreads = parse_int(argv[++i], 1, 1024, "host threads");
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) chunk = parse_num(argv[++i]);
        else if (argv[i][0] != '-' && !haven) { n0 = parse_num(argv[i]); haven = true; }
        else usage();
    }
    gpu_init();
    parse_bases(bases);
    if (!chunk) chunk = span / 8 > 10000000000ULL ? 10000000000ULL : span / 8;
    if (chunk < 1000000) chunk = 1000000;
    scan_stats_t st = run_scan(n0, n0 + span, chunk, NULL, 60, true);
    const double tps = (double)st.tests / st.seconds, pps = (double)st.primes / st.seconds;
    printf("bench [%" PRIu64 ", +%" PRIu64 "): %" PRIu64 " primes x %d bases in %.2f s\n", n0, span, st.primes, B.nb, st.seconds);
    printf("  %.4g primes/s, %.4g tests/s, %.4g numbers/s\n", pps, tps, (double)st.numbers / st.seconds);
    const double targets[] = {2, 10, 100};
    for (int i = 0; i < 3; i++) {
        const double x = (double)n0, y = x * targets[i];
        const double tests = li_diff(x, y) * B.nb;
        char b1[32];
        printf("  [N, %gN): %.3g tests, ~%s at this rate; heuristic %.3f new solutions over %d bases\n",
               targets[i], tests, fmt_dur(tests / tps, b1, sizeof b1), B.nb * log(log(y) / log(x)), B.nb);
    }
    return 0;
}

/*
 * rnd: the selftest's reproducible pseudo-random numbers (splitmix64 with a fixed seed).
 *   In:     nothing (advances rng_state).
 *   Out:    the next 64-bit value.
 *   Errors: none.
 */
static u64 rng_state = 0x243F6A8885A308D3ULL;
static u64 rnd(void) { return mix64(rng_state += GOLD); }

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

/* Checksum windows generated by verify_wieferich.py (Python integers, Miller-Rabin primes),
 * bases "oeis" ascending: {lo, hi, primes in [lo, hi), checksum per base}. */
typedef struct { u64 lo, hi, primes; u64 cks[17]; } ckwin_t;
static const ckwin_t CKWIN[] = {
#include "../wieferich_cks.h"
};

/*
 * cmd_selftest: "selftest [LIMIT] [-T N]" (theory section 7).
 *   In:     the arguments after "selftest"; LIMIT default 10^9 (end of part 1).
 *   Out:    one line per part on stderr and "selftest passed" or "FAILED"; exit status 0 or 1.
 *           The pi check is made only for LIMIT = 10^9.
 *   Errors: usage() for unknown options; dies for a bad LIMIT, CUDA errors and every run_scan
 *           error.
 */
static int cmd_selftest(int argc, char **argv)
{
    u64 limit = 1000000000ULL;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "-T") && i + 1 < argc) g_hthreads = parse_int(argv[++i], 1, 1024, "host threads");
        else if (argv[i][0] != '-') limit = parse_num(argv[i]);
        else usage();
    }
    gpu_init();
    g_print = false;
    int bad = 0;

    /* 0. random Fermat quotients against the host reference */
    {
        u32 bl[MAXB];
        int nbl = 0;
        for (u32 b = 2; nbl < MAXB; b += 1 + (u32)(rnd() % 97)) bl[nbl++] = b;
        bl[nbl - 1] = 65521;                                /* w = 1 */
        bases_setup(bl, nbl);
        std::vector<u64> ps;
        std::vector<int> js;
        const u64 edge[] = {3, 5, 7, 65537, 4294967291ULL, 4294967311ULL, (1ULL << 44) + 7, (1ULL << 48) + 21,
                            (1ULL << 62) + 135, PMAX - 25};
        for (u64 e : edge) {
            u64 p = e;
            while (!is_prime64(p)) p += 2;
            for (int j = 0; j < B.nb; j++) if (B.b[j] % p) { ps.push_back(p); js.push_back(j); }
        }
        while (ps.size() < 20000) {
            const int bits = 3 + (int)(rnd() % 61);
            u64 p = (rnd() >> (64 - bits)) | (1ULL << (bits - 1)) | 1;
            if (p >= PMAX) continue;
            while (!is_prime64(p)) p += 2;
            if (p >= PMAX) continue;
            const int j = (int)(rnd() % (u64)B.nb);
            if (B.b[j] % p == 0) continue;
            ps.push_back(p);
            js.push_back(j);
        }
        std::vector<u64> qs;
        std::vector<u32> fl;
        gpu_quotients(ps, js, qs, fl);
        int qbad = 0;
        for (size_t k = 0; k < ps.size(); k++) {
            const u128 rr = powmod_ref(B.b[js[k]], ps[k] - 1, (u128)ps[k] * ps[k]);
            if (fl[k] || qs[k] != (u64)((rr - 1) / ps[k])) {
                if (qbad < 5) fprintf(stderr, "  quotient mismatch p=%" PRIu64 " b=%u\n", ps[k], B.b[js[k]]);
                qbad++;
            }
        }
        fprintf(stderr, "arithmetic: %zu GPU Fermat quotients (random p < 2^63, 64 bases incl. w = 1..4) vs host reference: %s\n",
                ps.size(), qbad ? "FAIL" : "ok");
        bad += qbad;
    }

    /* 1. all bases 2..30 over [0, limit) against the known solutions */
    parse_bases("known");
    double t = now();
    u64 chunk = limit / 10;
    if (chunk < 1000000) chunk = 1000000;
    run_scan(0, limit, chunk, NULL, 60, true);
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
    const bool pi_ok = limit != 1000000000ULL || S.primes == 50847534ULL;
    fprintf(stderr, "scan [0, %" PRIu64 "), %d bases: %" PRIu64 " primes%s, %d known solutions found, %d missing, %d unexpected, %" PRIu64 " FLT errors (%.1f s)\n",
            S.end, B.nb, S.primes, limit == 1000000000ULL ? (pi_ok ? " (= pi(10^9))" : " (WRONG, pi(10^9) = 50847534)") : "",
            found, missing, extra, S.flt, now() - t);
    bad += missing + extra + (int)S.flt + !pi_ok;

    /* 2. windows around the known solutions above the scan */
    t = now();
    const u64 scanned = S.end;
    int wins = 0, wbad = 0;
    for (int i = 0; i < NKNOWN; i++)
        for (int k = 0; k < KNOWN[i].n; k++) {
            const u64 p = KNOWN[i].p[k];
            if (p < scanned) continue;
            u32 b = KNOWN[i].b;
            bases_setup(&b, 1);
            run_scan(p > 1000000 ? p - 1000000 : 0, p + 1000000, 200000, NULL, 60, true);
            const bool ok = S.nhits == 1 && S.hits[0].p == p && S.flt == 0;
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
        run_scan(c->lo, c->hi, (c->hi - c->lo) / 16, NULL, 60, true);
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
