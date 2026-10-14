/*
 * a158939_cuda.cu -- GPU (CUDA) version of the A158939 scan (see ../a158939.c)
 *
 * Same problem, bookkeeping and output as the CPU tool: for every prime p find
 * L(p) = largest n with g_1 < g_2 < ... < g_n (the gaps after p strictly
 * increasing) and report the first prime of every run length, the exact prime
 * count and the exact run-length histogram of the range scanned.  This is the
 * program that found a(16) = 17293451238695141 and a(17) = 52461866207504471
 * in September 2026 on a DGX Spark (NVIDIA GB10: 48 SMs, 24 MB L2, 128 GB of
 * unified memory at 273 GB/s).  The definition, the two facts the bookkeeping
 * rests on -- (F1) L(p) depends only on the gaps after p, so any range can be
 * scanned on its own; (F2) a run of L gaps ending at the prime q_k settles the
 * run lengths 1..L of the L primes before q_k -- and the OEIS context are in
 * the header of ../a158939.c and are not repeated here.
 *
 * ======================================================================
 * 1. Theory of operation
 * ======================================================================
 *
 * 1.1 Sieve layout (wheel 30)
 * ---------------------------
 * Only numbers coprime to 30 are represented: one byte per 30 numbers, bit j
 * of byte b standing for 30*b + R[j] with R = {1,7,11,13,17,19,23,29}.  A bit
 * that is still 0 after sieving is a prime.  For a sieving prime q >= 7 the
 * multiples q*m with gcd(m,30) = 1 fall into 8 arithmetic progressions of
 * byte index (stride q bytes) with a fixed bit each (the bit of q*m depends
 * only on q mod 30 and m mod 30, table c_BITIDX), so a mark is
 * "atomicOr(word[b >> 2], bit << 8*(b & 3))".  The primes 2, 3, 5 are not in
 * the wheel; the host adds them (L(2)=2, L(3)=1, L(5)=2) to a scan that
 * starts at 0.  The primes 7..19 are applied as a precomputed periodic byte
 * pattern (period 7*11*13*17*19 = 323323 bytes) instead of being sieved.
 *
 * 1.2 Geometry
 * ------------
 *   segment   SEG_WORDS = 21504 words = 84 KB = 2,580,480 numbers.  One block
 *             of BLOCKDIM = 1024 threads processes a segment in shared memory
 *             together with OVERLAP = 512 words (61,440 numbers) past its end,
 *             so that runs crossing the segment end can be finished there.
 *   chunk     CHUNK_SEGS = 96 segments = 247,726,080 numbers: the unit of the
 *             large-prime kernels, the host loop and the checkpoint.  The
 *             chunk bitmap (CHUNK_WORDS + OVERLAP words, 8 MB) lives in
 *             global memory but stays inside the GB10's 24 MB L2, pinned by a
 *             persisting access window unless -N is given; random atomics
 *             into it run at ~2e10/s there against ~1.3e9/s in DRAM.
 * The sieving primes are split at QSPLIT (default 4e6; 1e6 in production):
 *   7..19              presieve pattern;
 *   23..QSPLIT         "medium" primes, marked per segment in shared memory by
 *                      kernel_segment (one warp per prime below PWARP = 2048,
 *                      one thread per prime above);
 *   QSPLIT..sqrt(END)  "large" primes, marked per chunk into the chunk bitmap
 *                      by kernel_large / kernel_bucket (section 1.4).
 *
 * 1.3 Per-chunk pipeline (run_scan; one CUDA stream, chunks in order)
 * --------------------------------------------------------------------
 *   kernel_activate   primes whose square first falls into this chunk or its
 *                     overlap get their initial state;
 *   memset            the chunk bitmap is cleared;
 *   kernel_large      the ~2.5 million "dense" large primes (q < CHUNK_NUMS/6,
 *                     which hit every chunk) mark their multiples, one thread
 *                     per prime, state read and rewritten with streaming
 *                     loads and stores;
 *   kernel_bucket     the "sleeping" large primes due in this chunk, read from
 *                     the chunk's bucket list, mark their multiples and are
 *                     re-appended to the list of the chunk that needs their
 *                     next multiple;
 *   kernel_chunkmod   (chunk_lo/30) mod q for every medium prime: the one
 *                     64-bit division per medium prime per chunk;
 *   kernel_segment    one block per segment builds the finished segment bitmap
 *                     in shared memory and extracts primes, counts and run
 *                     lengths (section 1.5); one SegResult per segment goes
 *                     to global memory and is copied to pinned host memory;
 *   merge_chunk       the host folds the 96 results in order into the table,
 *                     the histogram and the prime count.
 * Chunks and segments complete in order, so every term is final the moment
 * it is recorded (no "frontier" as in the CPU tool), and pi(p) is exact:
 * primes before the segment plus the index inside it.  Two host
 * synchronizations per chunk (the bucket count before kernel_bucket, the
 * results after kernel_segment); the host merge takes a few percent.
 *
 * 1.4 Large-prime state and bucket lists
 * --------------------------------------
 * Every large prime keeps a 4-byte state: bits 31..26 = index (mod 64) of the
 * chunk holding its next multiple, bits 25..3 = that multiple's byte offset
 * inside the chunk (< 2^23), bits 2..0 = wheel index of the multiplier.
 * Consecutive multiples are at most 6q apart, so the chunk distance is below
 * 64 for every q below 2.6e9 (END below 6.8e18).  The dense primes keep their
 * state in a flat array.  The larger primes ("sleepers", about 48 million at
 * END = 1e18) mostly skip chunks; they live as 8-byte entries {state << 32 |
 * q} in a ring of SLOTS = 64 bucket lists, one per upcoming chunk, so a chunk
 * touches only the primes that hit it.  kernel_bucket stages each block's
 * outgoing entries in shared memory grouped by destination and reserves one
 * range per destination list with a single atomicAdd, so the appends are
 * coalesced.  The lists are sized from the expected hits per chunk at END
 * (twice that plus 2^20 entries each; 13 GB in all at END = 2e18).  A full
 * list or an entry found in the wrong list sets a flag that the host turns
 * into die().  A multiple in the overlap region of chunk k (the first OVERLAP
 * words of chunk k+1) has to be marked in chunk k's bitmap and again in chunk
 * k+1's, so the state after chunk k points at the first multiple >= chunk_hi
 * and that multiple is processed again in chunk k+1 (state_position's
 * "target k+1 inside the overlap" case; a sleeper is re-filed under k+1).
 * The state is not checkpointed: kernel_init_state recomputes it from the
 * first multiple of every active prime whenever a scan starts or resumes.
 *
 * 1.5 kernel_segment, the inner loop
 * ----------------------------------
 *   1. Every thread ORs the presieve pattern bytes and the large-prime bits of
 *      the chunk bitmap into the segment's SHARED_WORDS words of shared memory.
 *   2. The medium primes are marked with shared-memory atomics.  For a prime q
 *      and a residue class R[k], byte b (counted from the segment start) holds
 *      a multiple iff 30*(B0 + b) + R[k] = 0 (mod q), B0 = seg_lo/30, i.e.
 *      b = c + R[k]*beta (mod q) with c = -B0 mod q and beta = -30^-1 mod q.
 *      B0 mod q is assembled from (chunk_lo/30) mod q (kernel_chunkmod) and
 *      the segment number with 32-bit Barrett reductions; beta and the
 *      Barrett constant are per-prime tables from gpu_init.  A prime whose
 *      square lies inside the segment starts at q*q instead (early chunks).
 *      Warp-per-prime: the 32 lanes take the 8 classes times 4 sub-strides;
 *      thread-per-prime: a loop over the 8 classes.
 *   3. Prime count: each thread counts the primes of its slice of
 *      SEG_WORDS/BLOCKDIM = 21 words that lie inside [start, end) (popcount,
 *      bit by bit only in a boundary word); a block-wide exclusive scan gives
 *      every thread the number of primes before its slice and the block total.
 *   4. Run lengths: each thread walks its slice and keeps going into the
 *      following words, up to the end of the overlap, until the run in
 *      progress ends.  The bookkeeping is the CPU tool's, but in registers
 *      only: the run in progress is described by its first prime (run_start,
 *      with its index), its length, and how many of its primes lie at or
 *      beyond the slice end.  Run ends of length 1..7 are counted in
 *      registers and folded by warp reduction, longer ones with shared
 *      atomics.  A run longer than every earlier run of the slice re-walks
 *      the bitmap from run_start to record the new first occurrences
 *      (record_from; rare).  A run still open at the end of the shared bitmap
 *      is flagged (unsettled) and the host settles the rest of that segment
 *      on the CPU; that would need a run spanning the 61,440-number overlap
 *      and has never happened.
 *
 * 1.6 Checkpoint / resume (-S FILE)
 * ---------------------------------
 * The checkpoint records START, END, the position up to which the scan is
 * complete (a chunk boundary, or END when the last chunk is partial), the
 * prime count, the histogram and the table of first primes with their gaps.
 * Rerunning resumes at that position: the chunk containing it is rescanned
 * with the numbers below the position excluded from the counts (the kernel's
 * start bound), so a run that ended inside a chunk can be extended with a
 * larger END (only START must match).  An early version recorded a partial
 * last chunk as complete and skipped 41,615,360 numbers that way; selftest
 * (d) guards the fix.  SIGINT/SIGTERM stop after the current chunk and save.
 *
 * 1.7 Verification
 * ----------------
 * record_first re-derives every first occurrence with the host's Miller-Rabin
 * next-prime search and dies on a disagreement, so a wrong GPU run length
 * never reaches the output.  The selftest compares whole scans with exact
 * figures from the CPU tool (prime counts, first primes, prime indices and
 * histograms on [0, 1e11), on a window with bounds inside chunks, and on
 * [1e15, 1e15+2e12)) and checks that a resume across an END inside a chunk
 * equals a straight run.  Every version of the kernels was also diffed
 * against the CPU tool on 2e11-number windows at 3e16 and 1e18 (README.md).
 *
 * 1.8 Performance (GB10) and limits
 * ---------------------------------
 * 2.2-2.5e11 numbers/s at 1e15, 1.6e11 at 5e16, 1.3e11 at 1e18, i.e. 13x the
 * M1 Pro.  Above 1e16 the large-prime kernels dominate, and their cost is
 * state and entry traffic rather than the L2 atomics (~0.2 ms per chunk).
 * Raising QSPLIT above ~1e6 does not pay: every segment block re-reads the
 * medium-prime tables, and that bandwidth, not arithmetic, limits the
 * segment kernel.  Negative results (two streams, two pinned bitmaps, wheel
 * 210 = a158939_cuda_wheel210.cu, CPU alongside) are in README.md.
 * Limits: run lengths up to NMAX-1 = 63 (longer runs are clamped and
 * reported), PWARP <= QSPLIT <= 1e8, END below ~6.8e18 (chunk distance of
 * the state encoding), fewer than 2^32 entries per bucket list.
 * Exit status: 0 success, 1 failure (die(), failed selftest), 2 usage error.
 *
 * ======================================================================
 * 2. Usage
 * ======================================================================
 *   a158939_cuda scan [START] END [-n N] [-r NMIN] [-S FILE] [-i SECS] [-Q QSPLIT] [-N] [-q]
 *       Exhaustive scan of the primes in [START, END).  Prints each first
 *       prime with run length >= NMIN (default 14) as CONFIRMED when found
 *       and a table of a(n) at the end.  -n N stops once a(N) is found.
 *       -S FILE keeps a checkpoint every SECS seconds (default 60) and on
 *       Ctrl-C; rerunning resumes from it.  -Q QSPLIT sets the medium/large
 *       prime split (default 4e6), -N disables the L2 persisting window,
 *       -q suppresses the status line.  -1 and -L (stream and L2 options of
 *       earlier versions) are accepted and ignored.
 *   a158939_cuda bench [N] [-d SPAN] [-Q QSPLIT] [-N] [-p]
 *       Time the scan of [N, N+SPAN) (defaults 1e15 and 2e12); -p adds a
 *       per-kernel profile.
 *   a158939_cuda selftest [-Q QSPLIT]
 *       The checks of section 1.7.
 *
 * Build (DGX Spark / GB10, CUDA 13):
 *   nvcc -O3 -std=c++17 -arch=sm_121 a158939_cuda.cu -o a158939_cuda
 *   (or `make`; the geometry macros can be overridden with -D, and
 *   -DSKIP_MARK_WARP / -DSKIP_MARK_MED / -DSKIP_EXTRACT build ablation
 *   binaries for profiling whose results are wrong by design)
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
#include <ctime>
#include <cinttypes>
#include <climits>
#include <vector>
#include <algorithm>
#include <unistd.h>
#include <cuda_runtime.h>

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t  u8;
typedef unsigned __int128 u128;

/* ---- geometry (compile-time, overridable with -D) ----
 * The segment must fit in shared memory with its overlap (88 KB of the GB10's
 * 100 KB per SM), SEG_WORDS must be a multiple of BLOCKDIM (one slice of
 * SEG_WORDS/BLOCKDIM words per thread), and the chunk bitmap must fit in L2.
 * Chunk geometry changes were measured flat; see README.md. */
#ifndef SEG_WORDS
#define SEG_WORDS   21504           /* words owned by a segment: 84 KB, 2,580,480 numbers */
#endif
#ifndef OVERLAP
#define OVERLAP     512             /* extra words sieved past a segment: 61,440 numbers, so that
                                       a run crossing the segment end can be finished in the block */
#endif
#ifndef CHUNK_SEGS
#define CHUNK_SEGS  96              /* segments per chunk: 7.875 MB bitmap (must stay in L2) */
#endif
#ifndef BLOCKDIM
#define BLOCKDIM    1024            /* threads per segment block */
#endif
#ifndef PWARP
#define PWARP       2048            /* medium primes below this: one warp per prime (many marks
                                       per segment); above: one thread per prime */
#endif
#define SHARED_WORDS (SEG_WORDS + OVERLAP)  /* words of the segment bitmap in shared memory */
#define NUMS_PER_WORD 120ULL         /* 32 bits * 30/8: numbers covered by one bitmap word */
#define SEG_NUMS    (SEG_WORDS * NUMS_PER_WORD)
#define CHUNK_WORDS ((u64)SEG_WORDS * CHUNK_SEGS)
#define CHUNK_NUMS  (CHUNK_WORDS * NUMS_PER_WORD)   /* 247,726,080 */
#define PAT_PERIOD  323323          /* 7*11*13*17*19: period of the presieve pattern in bytes */
#define NMAX        64              /* run lengths 1..NMAX-1 tracked; a run of NMAX or more gaps is
                                       clamped and reported (the longest run below 5.5e16 has 17) */
#define NKNOWN      17              /* terms of A158939 in the OEIS: KNOWN[1..NKNOWN] */

/* KNOWN[n] = a(n) as published (KNOWN[0] unused): labels in print_table and
 * the expected values of the selftest. */
static const u64 KNOWN[NKNOWN + 1] = {
    0, 3, 2, 17, 347, 2903, 15373, 128981, 1319407, 17797517, 94097537,
    6927837557ULL, 48486712783ULL, 968068681511ULL, 1472840004017ULL,
    129001208165717ULL, 17293451238695141ULL, 52461866207504471ULL
};
/* A133697(n) = pi(A158939(n+2)) for n = 0..NA133697-1: the published prime
 * index of a(n+2), against which the scan's own pi(p) is checked. */
#define NA133697 16
static const u64 A133697[NA133697] = {
    1, 7, 69, 420, 1796, 12073, 101397, 1139211, 5440508, 320620306ULL,
    2058187481ULL, 36451609409ULL, 54594153615ULL, 4100904808215ULL,
    475618519121221ULL, 1400080864310974ULL
};
static_assert(NA133697 >= NKNOWN - 1, "A133697 table must cover every known term");
/* pi(10^k), k = 6..16: a scan [0, 10^k) must report exactly this many primes */
static const u64 PI10[17] = {
    0, 0, 0, 0, 0, 0, 78498ULL, 664579ULL, 5761455ULL, 50847534ULL, 455052511ULL,
    4118054813ULL, 37607912018ULL, 346065536839ULL, 3204941750802ULL,
    29844570422669ULL, 279238341033925ULL
};

static bool stderr_tty;                   /* stderr is a terminal: live status line instead of progress lines */
static volatile sig_atomic_t g_stop = 0;  /* set by SIGINT/SIGTERM or by -n N: stop after the current chunk */
static int g_report = 14;                 /* print CONFIRMED lines for run lengths >= this (-r NMIN) */
static bool g_l2_persist = true;       /* keep the bitmap resident in L2 with a persisting access window (-N disables) */
static bool g_quiet_init = false;         /* gpu_init prints nothing (bench and selftest) */

/* Every CUDA runtime call goes through CUDA_CHECK: any error is fatal and
 * reported with the call and the line number. */
#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) die("CUDA error %s at line %d: %s", #x, __LINE__, cudaGetErrorString(e_)); } while (0)

/* ------------------------------------------------------------------ */
/* Generic helpers (same conventions as a158939.c)                     */
/* ------------------------------------------------------------------ */

/*
 * now() -- monotonic wall-clock time in seconds (CLOCK_MONOTONIC), for rates
 * and ETAs.  Inputs: none.  Errors: none.
 */
static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

/*
 * die() -- print a fatal error ("a158939_cuda: ...") and exit(1).
 * Inputs:  printf-style format and arguments.
 * Effects: clears the status line when stderr is a terminal, prints the
 *          message and exits; never returns.  No checkpoint is written on
 *          this path (the last periodic checkpoint stays valid).
 */
static void die(const char *fmt, ...)
{
    va_list ap;
    if (stderr_tty) fputs("\r\033[K", stderr);
    fputs("a158939_cuda: ", stderr);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(1);
}

/*
 * fmt_dur() -- format a duration as "1d02h03m", "1h02m03s", "5m03s" or
 * "4.2s"; "?" for a negative, NaN or absurd value (an unknown ETA).
 * Inputs:  s  seconds;  buf, n  output buffer.   Returns: buf.
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
 * mul_ok() -- *r = a*b if it fits in 64 bits (returns true), else false with
 * *r untouched.
 */
static bool mul_ok(u64 a, u64 b, u64 *r)
{
    u128 x = (u128)a * b;
    if (x >> 64) return false;
    *r = (u64)x;
    return true;
}

/* all_digits() -- true iff s[0..n) is a non-empty string of decimal digits. */
static bool all_digits(const char *s, size_t n)
{
    if (n == 0) return false;
    for (size_t i = 0; i < n; i++)
        if (!isdigit((unsigned char)s[i])) return false;
    return true;
}

/*
 * parse_term() -- parse one unsigned number term: decimal ("129001208165717",
 * up to 20 digits), a power ("2^54", "10^16") or scientific notation with an
 * optional decimal point ("1e12", "1.29e14"; the value must be an integer).
 * Inputs:  s, n  the characters (not necessarily NUL-terminated);  out.
 * Returns: true and *out on success; false on a syntax error or a value that
 *          does not fit in 64 bits (*out untouched).
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
 * parse_num() -- a term, or TERM+TERM / TERM-TERM split at the last '+' or
 * '-' ("2^54-32", "10^15+7").  Negative values are rejected.
 * Returns: the value.   Errors: die() on a syntax error, overflow or a
 *          negative difference.
 */
static u64 parse_num(const char *s)
{
    size_t n = strlen(s);
    for (size_t i = n; i-- > 1; ) {
        if (s[i] != '+' && s[i] != '-') continue;
        u64 a, b;
        if (!parse_term(s, i, &a) || !parse_term(s + i + 1, n - i - 1, &b)) die("bad number '%s'", s);
        if (s[i] == '+') { if (a + b < a) die("overflow in '%s'", s); return a + b; }
        if (b > a) die("negative result in '%s'", s);
        return a - b;
    }
    u64 v;
    if (!parse_term(s, n, &v)) die("bad number '%s'", s);
    return v;
}

/*
 * parse_int() -- parse_num() plus an inclusive range check [lo, hi].
 * Errors:  die() with "<what> must be between lo and hi".
 */
static int parse_int(const char *s, int lo, int hi, const char *what)
{
    u64 v = parse_num(s);
    if (v < (u64)lo || v > (u64)hi) die("%s must be between %d and %d", what, lo, hi);
    return (int)v;
}

/* ---- Deterministic Miller-Rabin below 2^64, on the host.  Used to add the
 * wheel's missing small primes, to RE-DERIVE every first occurrence the GPU
 * reports (record_first) and for the CPU fallback (settle_cpu); it shares no
 * code with the sieve, which is the point. ---- */

/* mulmod() -- (a*b) mod m through a 128-bit product; m > 0. */
static u64 mulmod(u64 a, u64 b, u64 m) { return (u64)(((u128)a * b) % m); }
/* powmod() -- a^e mod m by square-and-multiply; m > 0. */
static u64 powmod(u64 a, u64 e, u64 m)
{
    u64 r = 1;
    a %= m;
    while (e) { if (e & 1) r = mulmod(r, a, m); a = mulmod(a, a, m); e >>= 1; }
    return r;
}
/*
 * is_prime64() -- deterministic primality test for every n < 2^64: trial
 * division by the primes up to 37, then strong-probable-prime tests to the
 * seven bases 2, 325, 9375, 28178, 450775, 9780504, 1795265022 (verified to
 * leave no strong pseudoprime below 2^64).  A base that is 0 mod n is
 * skipped, which is correct for the n > 37 that reach that point.
 * Inputs:  n.   Returns: true iff n is prime (false for n < 2).
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
        for (int j = 1; j < r && composite; j++) { x = mulmod(x, x, n); if (x == n - 1) composite = false; }
        if (composite) return false;
    }
    return true;
}
/*
 * next_prime64() -- smallest prime greater than n, by testing odd candidates
 * with is_prime64 (about log(n)/2 tests).  n must stay below the largest
 * 64-bit prime, 18446744073709551557, which no scan reaches.
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
 * run_length64() -- L(p) straight from the definition by next-prime search.
 * Inputs:  p  a prime;  gaps  room for maxgaps entries.
 * Returns: L, with gaps[0..L-1] the increasing gaps and gaps[L] the gap that
 *          ends the run; -1 if the run has maxgaps or more gaps.
 * Errors:  none besides the -1 signal.
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
/* Device side                                                         */
/* ------------------------------------------------------------------ */

/* Wheel tables in constant memory (8 entries: uniform accesses broadcast).
 *   c_R[j]         the residues coprime to 30, in order;
 *   c_GAP[j]       R[j+1] - R[j] (wrapping 29 -> 31), the step between the
 *                  multipliers of consecutive wheel multiples;
 *   c_RIDX[r]      index of residue r in R, 8 if r is not coprime to 30;
 *   c_BITIDX[a][j] bit (= residue index) of R[a]*R[j] mod 30: the bit a
 *                  multiple q*m occupies depends only on the classes of q
 *                  and m (filled by gpu_init);
 *   c_QINV30[r]    r^-1 mod 30 for the residues coprime to 30;
 *   wheelR(j)      R[j] from an immediate, for use in divergent code where a
 *                  constant-memory table lookup would serialize. */
__constant__ u32 c_R[8]       = { 1, 7, 11, 13, 17, 19, 23, 29 };
__constant__ u32 c_GAP[8]     = { 6, 4, 2, 4, 2, 4, 6, 2 };       /* R[j+1] - R[j] */
__constant__ u8  c_RIDX[30]   = { 8,0,8,8,8,8,8,1,8,8, 8,2,8,3,8,8,8,4,8,5, 8,8,8,6,8,8,8,8,8,7 };
__constant__ u8  c_BITIDX[8][8];                                  /* bit of (R[a]*R[j]) mod 30 */
__constant__ u8  c_QINV30[30]  = { 0,1,0,0,0,0,0,13,0,0, 0,11,0,7,0,0,0,23,0,19, 0,0,0,17,0,0,0,0,0,29 };  /* q^-1 mod 30 */
__device__ __forceinline__ u32 wheelR(u32 j) { return (u32)((0x1D1713110D0B0701ULL >> (8 * j)) & 0xFF); }   /* R[j] without a divergent table read */

/* Per-segment results written by kernel_segment and merged on the host, in
 * segment order.  Offsets are numbers counted from the segment start seg_lo;
 * "inside the bounds" means inside [start, end) as passed to the kernel. */
struct SegResult {
    u32 first[NMAX];        /* offset (from seg_lo) of the first prime with run length n, ~0u = none */
    u32 firstidx[NMAX];     /* its index among the segment's primes inside the bounds (1-based) */
    u32 runs[NMAX];         /* run ends of length L whose L primes are all inside the slice bounds */
    u32 extra[NMAX];        /* run lengths from final run ends, per prime inside the bounds */
    u32 count;              /* primes in [seg_lo, seg_hi) ∩ [start, end) */
    u32 unsettled;          /* offset of the first prime of a run still open at the bitmap end, ~0u = none */
    u32 overflow;           /* a run reached NMAX-1 gaps and was clamped */
    u32 pad;                /* keeps the struct 16-byte aligned */
};

/*
 * first_multiplier() -- smallest multiplier m with q*m >= lo, m >= q and
 * gcd(m, 30) = 1: multiples with a factor of the wheel are not in the
 * bitmap, and multiples below q*q have a smaller prime factor and are marked
 * by that prime.
 * Inputs:  q  a sieving prime not dividing the wheel;  lo  chunk start.
 * Returns: m, i.e. the first multiple of q to mark is q*m.
 * Errors:  none.
 */
__device__ __forceinline__ u64 first_multiplier(u32 q, u64 lo)
{
    u64 m = (lo + q - 1) / q;
    if (m < q) m = q;
    u32 r = (u32)(m % 30);
    while (c_RIDX[r] == 8) { m++; r = (r == 29) ? 0 : r + 1; }
    return m;
}

/* Large-prime state, one u32 per prime: bits 31..26 = index (mod 64) of the
 * chunk holding the prime's next multiple, bits 25..3 = that multiple's byte
 * offset inside the chunk, bits 2..0 = wheel index of its multiplier.
 * Consecutive multiples are at most 6q apart, so the chunk distance is below
 * 64 for every q below 2.6e9.
 *
 * Primes below CHUNK_NUMS/6 ("dense", ~2.5 million) hit every chunk and keep
 * their state in a flat array read and rewritten by kernel_large.  Larger
 * primes ("sleepers") mostly skip chunks; they live as 8-byte entries
 * (state << 32 | q) in a ring of SLOTS bucket lists, one list per upcoming
 * chunk, so a chunk touches only the primes that hit it (kernel_bucket), and
 * each entry is re-appended to the list of the chunk holding its next
 * multiple, block-staged so the appends are coalesced. */
#define SLOTS        64                              /* bucket lists = chunks of look-ahead (state field is 6 bits) */
#define OVERLAP_NUMS ((u64)OVERLAP * NUMS_PER_WORD)  /* numbers in the overlap region: 61,440 */
#define BLOCKDIM_B   1024                            /* threads per kernel_bucket block */

/*
 * pack_state() -- encode the multiple n of a large prime as a state word.
 * Inputs:  n  the multiple (coprime to 30);  j  wheel index of its
 *          multiplier n/q (so that the next multiple is n + q*c_GAP[j]).
 * Returns: (chunk of n mod SLOTS) << 26 | (byte offset of n in that chunk)
 *          << 3 | j.  The byte offset is below 2^23 because a chunk has
 *          CHUNK_WORDS*4 = 8,257,536 bytes.
 * Errors:  none.
 */
__device__ __forceinline__ u32 pack_state(u64 n, u32 j)
{
    u64 kc = n / CHUNK_NUMS;
    u32 b = (u32)((n - kc * CHUNK_NUMS) / 30);
    return ((u32)(kc & (SLOTS - 1)) << 26) | (b << 3) | j;
}

/*
 * slot_of() -- the chunk in which the multiple n must first be marked: the
 * chunk whose extended range [chunk_lo, chunk_hi + OVERLAP_NUMS) contains n.
 * Inputs:  n  any number.
 * Returns: that chunk index; a number in the overlap region of chunk k maps
 *          to k, not to k+1 (it is marked in both, k first).
 * Errors:  none.
 */
__device__ __forceinline__ u64 slot_of(u64 n)
{
    return n < OVERLAP_NUMS ? 0 : (n - OVERLAP_NUMS) / CHUNK_NUMS;
}

/*
 * append_entry() -- push a sleeper entry onto the bucket list of a chunk
 * (used by the initialization kernels; kernel_bucket has its own staged
 * version).
 * Inputs:  entries, counts, cap  the SLOTS lists of cap entries each and
 *          their fill counts;  err  error flag word;  slot  absolute index
 *          of the destination chunk (reduced mod SLOTS);  entry  state<<32|q.
 * Effects: one global atomicAdd on the list count; stores the entry at the
 *          reserved position, or sets bit 0 of *err when the list is full
 *          (the host dies at the next chunk).
 * Errors:  list overflow -> *err bit 0.
 */
__device__ __forceinline__ void append_entry(u64 *entries, u32 *counts, u32 cap, u32 *err, u64 slot, u64 entry)
{
    u32 s = (u32)(slot % SLOTS);
    u32 pos = atomicAdd(&counts[s], 1u);
    if (pos < cap) entries[(u64)s * cap + pos] = entry;
    else atomicOr(err, 1u);
}

/*
 * kernel_init_state -- initial state of every active large prime for a scan
 * that starts or resumes at chunk k0 (the state is not checkpointed).
 * Launch:  one thread per prime i < nactive, blocks of 256.
 * Inputs:  primes   large sieving primes, ascending;  state  flat state
 *          array of the dense primes;  nd  number of dense primes (the first
 *          nd of primes[]);  nactive  primes with q*q below the first chunk's
 *          extended end;  k0  first chunk;  entries/counts/cap/err  the
 *          bucket lists.
 * Effects: dense primes get the state of their first multiple >= k0's start
 *          (first_multiplier); sleepers are filed under the chunk that first
 *          needs that multiple, but never under a chunk below k0: a multiple
 *          in the overlap region of chunk k0-1 lies inside chunk k0, and
 *          filing it under k0-1, which is never processed, would lose it
 *          (the first bucket build lost 86,000 marks per 2e11-number window
 *          exactly that way; caught by the CPU-tool window diff).
 * Errors:  bucket overflow -> err bit 0.
 */
__global__ void kernel_init_state(const u32 *__restrict__ primes, u32 *__restrict__ state, u32 nd, u32 nactive, u64 k0,
                                  u64 *__restrict__ entries, u32 *__restrict__ counts, u32 cap, u32 *__restrict__ err)
{
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nactive) return;
    const u64 chunk_lo = k0 * CHUNK_NUMS;
    u32 q = primes[i];
    u64 m = first_multiplier(q, chunk_lo), n = (u64)q * m;
    u32 st = pack_state(n, c_RIDX[m % 30]);
    if (i < nd) { state[i] = st; return; }
    u64 sl = slot_of(n);
    if (sl < k0) sl = k0;                             /* the first chunk's overlap belongs to no earlier chunk here */
    append_entry(entries, counts, cap, err, sl, ((u64)st << 32) | q);
}

/*
 * kernel_activate -- state of the primes [nprev, nactive) that become active
 * in chunk k, i.e. whose square lies in the chunk or its overlap.
 * Launch:  one thread per new prime, blocks of 256; only when nactive > nprev.
 * Inputs:  as kernel_init_state; nprev = primes already active;  k  chunk.
 * Effects: the first multiple to mark is q*q (multiplier q): dense primes
 *          into state[], sleepers into the list of max(slot_of(q*q), k).
 * Errors:  bucket overflow -> err bit 0.
 */
__global__ void kernel_activate(const u32 *__restrict__ primes, u32 *__restrict__ state, u32 nd, u32 nprev, u32 nactive, u64 k,
                                u64 *__restrict__ entries, u32 *__restrict__ counts, u32 cap, u32 *__restrict__ err)
{
    u32 i = nprev + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nactive) return;
    u32 q = primes[i];
    u64 n = (u64)q * q;
    u32 st = pack_state(n, c_RIDX[q % 30]);
    if (i < nd) { state[i] = st; return; }
    u64 sl = slot_of(n);
    if (sl < k) sl = k;
    append_entry(entries, counts, cap, err, sl, ((u64)st << 32) | q);
}

/*
 * kernel_chunkmod -- rchunk[i] = (chunk_lo / 30) mod q for every medium
 * prime: the one 64-bit division per medium prime per chunk.  kernel_segment
 * derives every segment's start residue from it with 32-bit arithmetic.
 * Launch:  one thread per medium prime, blocks of 256, once per chunk.
 * Inputs:  sprimes/nsmall  the medium primes;  chunk_lo  chunk start.
 * Effects: fills rchunk[0..nsmall).
 * Errors:  none.
 */
__global__ void kernel_chunkmod(const u32 *__restrict__ sprimes, u32 *__restrict__ rchunk, u32 nsmall, u64 chunk_lo)
{
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nsmall) rchunk[i] = (u32)((chunk_lo / 30) % sprimes[i]);
}

/*
 * barrett32() -- x mod q for 32-bit x without a division.
 * Inputs:  x < 2^32;  q  the modulus (< 2^31);  inv = floor(2^32 / q), the
 *          per-prime table sinv built by gpu_init.
 * Returns: x mod q.  __umulhi(x, inv) = floor(x*inv / 2^32) is floor(x/q) or
 *          one less, so a single conditional subtraction corrects it.
 * Errors:  none; callers keep x < 2^32 (a residue plus at most 95 times a
 *          value below 2^17, or a residue plus 29q with q <= 1e8).
 */
__device__ __forceinline__ u32 barrett32(u32 x, u32 q, u32 inv)
{
    u32 r = x - __umulhi(x, inv) * q;
    return r >= q ? r - q : r;
}

/*
 * mark_prime() -- mark the multiples of a large prime q in the chunk bitmap,
 * from n up to hi_ext, and return the state of the first multiple beyond
 * the chunk.
 * Inputs:  q  the prime;  n  its first multiple to mark (>= chunk_lo), with
 *          j  the wheel index of n/q;  chunk_lo, chunk_hi  the chunk;
 *          hi_ext = chunk_hi + OVERLAP_NUMS;  bitmap  the chunk bitmap
 *          (CHUNK_WORDS + OVERLAP words).
 * Effects: one global atomicOr per multiple: byte (n - chunk_lo)/30, bit
 *          c_BITIDX[class of q][j].  n advances by q*c_GAP[j] and j by one,
 *          so the eight residue progressions are visited in number order.
 * Returns: pack_state() of the first multiple >= chunk_hi.  That multiple is
 *          either inside the overlap (then it has been marked here and will
 *          be marked again in the next chunk) or beyond hi_ext (then it is
 *          the first unmarked one).
 * Errors:  none.
 */
__device__ __forceinline__ u32 mark_prime(u32 q, u64 n, u32 j, u64 chunk_lo, u64 chunk_hi, u64 hi_ext, u32 *__restrict__ bitmap)
{
    const u32 a = c_RIDX[q % 30];
    u64 nsave = 0;
    u32 jsave = 0;
    bool saved = false;
    do {
        if (!saved && n >= chunk_hi) { nsave = n; jsave = j; saved = true; }
        u32 b = (u32)((n - chunk_lo) / 30);
        atomicOr(&bitmap[b >> 2], (1u << c_BITIDX[a][j]) << ((b & 3) << 3));
        n += (u64)q * c_GAP[j];
        j = (j + 1) & 7;
    } while (n < hi_ext);
    if (!saved) { nsave = n; jsave = j; }
    return pack_state(nsave, jsave);
}

/*
 * state_position() -- decode a state word relative to chunk k.
 * Inputs:  s  the state;  q  its prime;  k  the current chunk;  chunk_lo,
 *          chunk_hi  its bounds.
 * Returns: true, with *n = the multiple and *j = its wheel index, when the
 *          multiple is due now: inside chunk k (target field == k mod 64), or
 *          inside chunk k's overlap region (target == k+1 and byte offset
 *          within the first OVERLAP words).  false when the prime sleeps
 *          through this chunk.  The residue of the multiple is
 *          c_R[c_BITIDX[class of q][j]].
 * Errors:  none.
 */
__device__ __forceinline__ bool state_position(u32 s, u32 q, u64 k, u64 chunk_lo, u64 chunk_hi, u64 *n, u32 *j)
{
    const u32 tgt = s >> 26, kk = (u32)(k & (SLOTS - 1)), b = (s >> 3) & 0x7FFFFFu;
    u64 base;
    if (tgt == kk) base = chunk_lo;
    else if (tgt == ((kk + 1) & (SLOTS - 1)) && b < OVERLAP * 4) base = chunk_hi;
    else return false;
    *j = s & 7;
    *n = base + 30ULL * b + c_R[c_BITIDX[c_RIDX[q % 30]][*j]];
    return true;
}

/*
 * kernel_large -- the dense large primes (q below the dense threshold, which
 * hit every chunk): one thread per prime, every chunk.
 * Launch:  n_dense threads in blocks of 256 (launch bound 256).
 * Inputs:  primes, state  the flat arrays (the first n_dense entries);  k,
 *          chunk_lo, chunk_hi, hi_ext  the chunk;  bitmap  the chunk bitmap.
 * Effects: decodes the state (state_position); if the multiple is due, marks
 *          the chunk (mark_prime) and stores the new state, otherwise stores
 *          the old state unchanged.  Loads and stores use the streaming
 *          cache hint (__ldcs/__stcs) so the ~10 MB of prime and state
 *          arrays do not evict the bitmap from L2, and the unconditional
 *          store keeps the writes coalesced (the sparse write-back of only
 *          the ~18% of primes that hit was measured to be slower).
 * Errors:  none.
 */
__global__ void __launch_bounds__(256)
kernel_large(const u32 *__restrict__ primes, u32 *__restrict__ state, u32 n_dense,
             u64 k, u64 chunk_lo, u64 chunk_hi, u64 hi_ext, u32 *__restrict__ bitmap)
{
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dense) return;
    const u32 s = __ldcs(&state[i]);
    const u32 q = __ldcs(&primes[i]);
    u64 n;
    u32 j;
    if (!state_position(s, q, k, chunk_lo, chunk_hi, &n, &j)) { __stcs(&state[i], s); return; }
    __stcs(&state[i], mark_prime(q, n, j, chunk_lo, chunk_hi, hi_ext, bitmap));
}

/*
 * kernel_bucket -- the sleeping large primes due in chunk k.
 * Launch:  one thread per entry of list k mod SLOTS (n_in entries, read by
 *          the host before the launch), blocks of BLOCKDIM_B = 1024.
 * Inputs:  entries, counts, cap, err  the lists;  n_in  entries in this
 *          chunk's list;  k, chunk_lo, chunk_hi, hi_ext, bitmap  as in
 *          kernel_large.
 * Effects: each entry's multiples in the chunk are marked (mark_prime) and
 *          the entry, with its new state, is re-filed under the chunk that
 *          first needs the next multiple: slot_of() of that multiple, or
 *          k+1 when it lies in this chunk's overlap (it must be marked again
 *          there).  Outgoing entries are staged in shared memory grouped by
 *          destination (relative chunk 1..63): per-destination counts by
 *          shared atomics, an exclusive prefix over the 64 counts, one
 *          global atomicAdd per destination per block to reserve a range in
 *          the destination list, then a coalesced write-out.  The host
 *          clears this chunk's list count afterwards.
 * Errors:  err bit 0 on list overflow; err bit 1 if an entry's multiple is
 *          not due in this chunk, i.e. it was filed in the wrong list (a
 *          bug, never seen).  The host dies on either flag at the next chunk.
 */
__global__ void __launch_bounds__(BLOCKDIM_B)
kernel_bucket(u64 *__restrict__ entries, u32 *__restrict__ counts, u32 cap, u32 *__restrict__ err, u32 n_in,
              u64 k, u64 chunk_lo, u64 chunk_hi, u64 hi_ext, u32 *__restrict__ bitmap)
{
    __shared__ u32 s_cnt[SLOTS], s_off[SLOTS], s_base[SLOTS];
    __shared__ u64 s_stage[BLOCKDIM_B];
    __shared__ u8 s_rel[BLOCKDIM_B];
    const u32 tid = threadIdx.x;
    if (tid < SLOTS) s_cnt[tid] = 0;
    __syncthreads();

    const u32 slot = (u32)(k % SLOTS);
    const u32 i = blockIdx.x * BLOCKDIM_B + tid;
    const bool have = i < n_in;
    u32 rel = 0, rank = 0;
    u64 out = 0;
    if (have) {
        const u64 e = entries[(u64)slot * cap + i];     /* entry = state << 32 | q */
        const u32 q = (u32)e, s = (u32)(e >> 32);
        u64 n;
        u32 j;
        if (state_position(s, q, k, chunk_lo, chunk_hi, &n, &j)) {
            const u32 st = mark_prime(q, n, j, chunk_lo, chunk_hi, hi_ext, bitmap);
            /* next multiple: recover it from the packed state to find its slot */
            const u32 tgt = st >> 26, b = (st >> 3) & 0x7FFFFFu;
            u64 kc = k + (((tgt - (u32)(k % SLOTS)) & (SLOTS - 1)));        /* absolute chunk of the next multiple */
            u64 nn = kc * CHUNK_NUMS + 30ULL * b;                               /* (residue irrelevant for the slot) */
            u64 sl = slot_of(nn);
            if (sl <= k) sl = k + 1;                                            /* in this chunk's overlap: process again next chunk */
            rel = (u32)(sl - k);
            out = ((u64)st << 32) | q;
            rank = atomicAdd(&s_cnt[rel], 1u);
        } else {
            atomicOr(err, 2u);                                                  /* entry in the wrong list */
        }
    }
    __syncthreads();
    if (tid == 0) {                                   /* exclusive prefix of the per-destination counts */
        u32 acc = 0;
        for (u32 r = 0; r < SLOTS; r++) { s_off[r] = acc; acc += s_cnt[r]; }
    }
    if (tid < SLOTS) {                                /* reserve ranges in the destination lists */
        u32 c = s_cnt[tid];
        u32 base = c ? atomicAdd(&counts[(u32)((k + tid) % SLOTS)], c) : 0;
        s_base[tid] = base;
        if (c && base + c > cap) atomicOr(err, 1u);
    }
    __syncthreads();
    if (have && out) {
        u32 p = s_off[rel] + rank;
        s_stage[p] = out;
        s_rel[p] = (u8)rel;
    }
    __syncthreads();
    const u32 total = s_off[SLOTS - 1] + s_cnt[SLOTS - 1];
    if (tid < total) {                                /* coalesced write-out, one run per destination */
        const u32 r = s_rel[tid];
        const u32 pos = s_base[r] + (tid - s_off[r]);
        if (pos < cap) entries[(u64)((k + r) % SLOTS) * cap + pos] = s_stage[tid];
    }
}

/*
 * block_exscan() -- block-wide exclusive prefix sum of one u32 per thread.
 * Inputs:  v  this thread's value;  scratch  32 shared words;  blockDim.x a
 *          multiple of 32 and at most 1024; every thread must call it.
 * Returns: the sum of v over the lower thread indices; *tot = the block
 *          total, valid in every thread.  Shuffle scan inside each warp, the
 *          warp totals scanned by warp 0.
 * Errors:  none.
 */
__device__ __forceinline__ u32 block_exscan(u32 v, u32 *scratch, u32 *tot)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    u32 x = v;
    for (int o = 1; o < 32; o <<= 1) { u32 y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }
    if (lane == 31) scratch[warp] = x;
    __syncthreads();
    if (warp == 0) {
        u32 w = (lane < nwarps) ? scratch[lane] : 0;
        for (int o = 1; o < 32; o <<= 1) { u32 y = __shfl_up_sync(0xffffffffu, w, o); if (lane >= o) w += y; }
        if (lane < nwarps) scratch[lane] = w;
    }
    __syncthreads();
    u32 prefix = (warp ? scratch[warp - 1] : 0) + x - v;
    *tot = scratch[nwarps - 1];
    __syncthreads();
    return prefix;
}

/*
 * record_from() -- record first occurrences of new run lengths by re-walking
 * a run in the shared bitmap.
 * Inputs:  sh  the segment bitmap;  o  offset of the run's first prime;
 *          idx0  that prime's index among the counted primes;  count  how
 *          many primes to record;  run  the run's length;  slice_hi  the
 *          calling thread's slice end;  s_first  shared (offset << 32 |
 *          index) words, one per run length.
 * Effects: prime t of the walk (t = 0..count-1) has run length run - t and
 *          index idx0 + t; each is atomicMin'd into s_first[run - t], so the
 *          smallest offset wins across the block's threads.  Stops at
 *          slice_hi (primes beyond it belong to the next slice, which records
 *          them itself) or at the end of the bitmap.
 * Errors:  none.  Called only when a run is longer than every earlier run of
 *          the slice, so the re-walk is rare.
 */
__device__ __forceinline__ void record_from(const u32 *sh, u32 o, u32 idx0, u32 count, u32 run, u32 slice_hi,
                                            unsigned long long *s_first)
{
    u32 b = o / 30, w = b >> 2;
    u32 bi = ((b & 3) << 3) + c_RIDX[o % 30];
    u32 x = (~sh[w]) & (~0u << bi);
    for (u32 t = 0; t < count; t++) {
        while (!x) { if (++w >= SHARED_WORDS) return; x = ~sh[w]; }
        u32 bit = __ffs(x) - 1;
        x &= x - 1;
        u32 off = 120 * w + 30 * (bit >> 3) + wheelR(bit & 7);
        if (off >= slice_hi) return;
        atomicMin(&s_first[run - t], ((unsigned long long)off << 32) | (idx0 + t));
    }
}

/*
 * kernel_segment -- one block per segment: finish the segment's bitmap in
 * shared memory and extract primes, counts and run lengths (section 1.5 of
 * the header).
 * Launch:  CHUNK_SEGS blocks of BLOCKDIM threads with SHARED_WORDS*4 bytes
 *          of dynamic shared memory (88 KB; gpu_init raises the limit).
 * Inputs:  bitmap   chunk bitmap holding the large-prime marks;
 *          pattern  presieve pattern bytes;
 *          sprimes  medium primes 23..QSPLIT ascending, nsmall of them,
 *                   the first nwarpprimes below PWARP;
 *          sinv, ssegmod (and sbeta in the wheel-30 version)  per-prime
 *                   tables from gpu_init (Barrett constant, segment length
 *                   mod q, -30^-1 mod q);
 *          rchunk   chunk start mod q from kernel_chunkmod;
 *          chunk_lo chunk start;  start, end  bounds of the numbers to
 *                   account for: primes outside [start, end) are sieved but
 *                   neither counted nor recorded (START, END and the resume
 *                   position are enforced this way);
 *          results  one SegResult per segment.
 * Effects: results[seg] = prime count, first occurrences (offsets from the
 *          segment start and 1-based indices among the counted primes),
 *          the two histogram arrays, the unsettled flag and the overflow
 *          flag.
 * Errors:  none raised.  overflow (a run of NMAX or more gaps, clamped) and
 *          unsettled (a run still open at the bitmap end) are reported in
 *          the result and handled by the host.  Requires SEG_WORDS to be a
 *          multiple of BLOCKDIM and every segment start to be a multiple of
 *          the wheel size.
 */
__global__ void __launch_bounds__(BLOCKDIM)
kernel_segment(const u32 *__restrict__ bitmap, const u8 *__restrict__ pattern,
               const u32 *__restrict__ sprimes, const u32 *__restrict__ sinv, const u32 *__restrict__ ssegmod,
               const u32 *__restrict__ sbeta, const u32 *__restrict__ rchunk, u32 nsmall, u32 nwarpprimes,
               u64 chunk_lo, u64 start, u64 end, SegResult *__restrict__ results)
{
    extern __shared__ u32 sh[];                         /* SHARED_WORDS words of bitmap */
    __shared__ unsigned long long s_first[NMAX];       /* (offset << 32) | prime index */
    __shared__ u32 s_runs[NMAX], s_extra[NMAX];
    __shared__ u32 s_unsettled, s_overflow, s_scratch[32];

    const u32 seg = blockIdx.x;
    const u64 seg_lo = chunk_lo + (u64)seg * SEG_NUMS;
    const u32 word0 = seg * SEG_WORDS;
    const u32 tid = threadIdx.x;

    /* --- 1. presieve pattern (7, 11, 13, 17, 19) and large-prime bits ---
     * The pattern repeats every PAT_PERIOD bytes; pbase is the pattern byte
     * of the segment's first byte.  Each thread assembles four pattern bytes
     * per word and ORs in the chunk bitmap word (the large-prime marks) with
     * a streaming load, so the chunk bitmap is read exactly once. */
    {
        u32 pbase = (u32)((seg_lo / 30) % PAT_PERIOD);
        for (u32 w = tid; w < SHARED_WORDS; w += BLOCKDIM) {
            u32 k = pbase + 4 * w;
            if (k >= PAT_PERIOD) k -= PAT_PERIOD;
            u32 v = 0;
            #pragma unroll
            for (int t = 0; t < 4; t++) {
                v |= (u32)pattern[k] << (8 * t);
                k++;
                if (k == PAT_PERIOD) k = 0;
            }
            sh[w] = v | __ldcs(&bitmap[word0 + w]);
        }
        if (tid < NMAX) { s_first[tid] = ~0ull; s_runs[tid] = 0; s_extra[tid] = 0; }
        if (tid == 0) { s_unsettled = ~0u; s_overflow = 0; }
    }
    __syncthreads();

    /* --- 2. small primes: warp per prime (q < PWARP), then thread per prime ---
     * Byte offset b (from seg_lo) holds a multiple of q in residue class R[k]
     * iff 30*(B0 + b) + R[k] = 0 (mod q), B0 = seg_lo/30, i.e.
     * b = c + R[k]*beta (mod q) with c = -B0 mod q and beta = -30^-1 mod q.
     * B0 mod q comes from (chunk_lo/30) mod q (rchunk, per chunk) and
     * (SEG_NUMS/30) mod q (ssegmod) by 32-bit Barrett arithmetic; the bit of
     * class k is k.  Below q*q (early chunks only) the first multiple is q*q. */
    const u32 SHB = SHARED_WORDS * 4;                   /* bytes in the shared bitmap */
    {
        const u32 warp = tid >> 5, lane = tid & 31, nwarps = BLOCKDIM / 32;
#ifndef SKIP_MARK_WARP
        for (u32 i = warp; i < nwarpprimes; i += nwarps) {
            const u32 q = sprimes[i];
            /* lane -> residue class k (8 classes) x sub-stride sub (4 per class):
             * lane marks the bytes b, b+4q, b+8q, ... of class k from b + sub*q */
            const u32 k = lane & 7, sub = lane >> 3, mask = 1u << k, stride = 4 * q;
            u32 b;
            if ((u64)q * q > seg_lo) {              /* early chunks: start at q*q (q itself must stay unmarked) */
                u64 d = (u64)q * q - seg_lo;
                if (d >= (u64)SHB * 30) continue;
                const u32 D = (u32)d, qinv = c_QINV30[q % 30], Dm = D % 30;
                const u32 tk = ((wheelR(k) + 30 - Dm) * qinv) % 30;
                b = (D + q * tk) / 30;
            } else {
                const u32 inv = sinv[i];
                const u32 r = barrett32(rchunk[i] + seg * ssegmod[i], q, inv);
                const u32 c = r ? q - r : 0;
                b = barrett32(c + wheelR(k) * sbeta[i], q, inv);
            }
            for (b += sub * q; b < SHB; b += stride)
                atomicOr(&sh[b >> 2], mask << ((b & 3) << 3));
        }
#endif
#ifndef SKIP_MARK_MED
        for (u32 i = nwarpprimes + tid; i < nsmall; i += BLOCKDIM) {
            const u32 q = sprimes[i];
            if ((u64)q * q > seg_lo) {              /* early chunks: start at q*q */
                u64 d = (u64)q * q - seg_lo;
                if (d >= (u64)SHB * 30) continue;
                const u32 D = (u32)d, qinv = c_QINV30[q % 30], Dm = D % 30;
                #pragma unroll
                for (u32 k = 0; k < 8; k++) {
                    const u32 tk = ((wheelR(k) + 30 - Dm) * qinv) % 30;
                    const u32 mask = 1u << k;
                    for (u32 b = (D + q * tk) / 30; b < SHB; b += q)
                        atomicOr(&sh[b >> 2], mask << ((b & 3) << 3));
                }
            } else {
                const u32 inv = sinv[i], beta = sbeta[i];
                const u32 r = barrett32(rchunk[i] + seg * ssegmod[i], q, inv);
                const u32 c = r ? q - r : 0;
                #pragma unroll
                for (u32 k = 0; k < 8; k++) {
                    const u32 mask = 1u << k;
                    for (u32 b = barrett32(c + wheelR(k) * beta, q, inv); b < SHB; b += q)
                        atomicOr(&sh[b >> 2], mask << ((b & 3) << 3));
                }
            }
        }
#endif
    }
    __syncthreads();
    if (seg_lo == 0 && tid == 0) sh[0] = (sh[0] & ~0x3Eu) | 1u;   /* 1 is not prime; 7..19 are */
    __syncthreads();

    /* --- 3. prime count in the owned region ∩ [start, end) ---
     * Each thread counts its slice of WPT words; a block-wide exclusive scan
     * then gives every thread the number of counted primes before its slice
     * (needed for the prime indices of step 4) and thread 0 the total. */
    /* bounds as offsets from seg_lo */
    const u64 own_nums = SEG_NUMS;
    u32 lo_off = (start > seg_lo) ? (u32)(start - seg_lo) : 0;              /* first number accounted */
    u32 hi_off = (end < seg_lo + own_nums) ? (u32)(end - seg_lo) : (u32)own_nums;   /* exclusive */
    if (lo_off > hi_off) lo_off = hi_off;
    const u32 WPT = SEG_WORDS / BLOCKDIM;                  /* words per thread slice: 21 (SEG_WORDS is a multiple of BLOCKDIM) */
    const u32 w0 = tid * WPT;
    u32 prefix;                                            /* primes (inside the bounds) before this slice */
    {
        u32 c = 0;
        for (u32 w = w0; w < w0 + WPT; w++) {
            u32 x = ~sh[w];
            if (x) {
                /* word w covers offsets [120w, 120w+120) */
                u32 wlo = 120 * w;
                if (wlo >= lo_off && wlo + 120 <= hi_off) c += __popc(x);
                else if (wlo + 120 > lo_off && wlo < hi_off) {
                    while (x) {
                        u32 bit = __ffs(x) - 1;
                        x &= x - 1;
                        u32 off = wlo + 30 * (bit >> 3) + wheelR(bit & 7);
                        c += (off >= lo_off && off < hi_off);
                    }
                }
            }
        }
        u32 total;
        prefix = block_exscan(c, s_scratch, &total);
        if (tid == 0) results[seg].count = total;
    }

    /* --- 4. run lengths: thread per slice of words, continuing until settled ---
     * Registers only: the run in progress is described by its first prime
     * (run_start, with its index run_start_cnt), its length `run`, and
     * run_out = how many of its primes lie at or beyond slice_hi.  When a run
     * ends the histogram is updated from those (short runs in registers), and
     * the rare first-occurrence records re-walk the run from run_start. */
    u32 c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0, c6 = 0, c7 = 0;
    {
        u32 slice_lo = 120 * w0, slice_hi = 120 * (w0 + WPT);   /* offsets */
        if (slice_lo < lo_off) slice_lo = lo_off;
        if (slice_hi > hi_off) slice_hi = hi_off;
#ifdef SKIP_EXTRACT
        slice_hi = slice_lo;
#endif
        if (slice_lo < slice_hi) {
            u32 run = 0, M = 0, prev = 0, prevgap = 0, cnt = 0, run_out = 0, run_start = 0, run_start_cnt = 0;
            bool have_prev = false, done = false;
            for (u32 w = w0; w < SHARED_WORDS && !done; w++) {
                u32 x = ~sh[w];
                const u32 wlo = 120 * w;
                while (x) {
                    const u32 bit = __ffs(x) - 1;
                    x &= x - 1;
                    const u32 off = wlo + 30 * (bit >> 3) + wheelR(bit & 7);
                    if (!have_prev) {
                        if (off < slice_lo) continue;
                        if (off >= slice_hi) { done = true; break; }   /* no prime in the slice */
                        have_prev = true;
                        prev = off;
                        cnt = 1;
                        run_start = off;
                        run_start_cnt = 1;
                        continue;
                    }
                    const u32 g = off - prev;
                    if (g > prevgap) {
                        run++;                                   /* prev joins the run's primes */
                        run_out += (prev >= slice_hi);
                    } else {
                        /* run of `run` gaps ended at prev = q_k; its primes q_(k-run)..q_(k-1)
                         * (the first being run_start) have run lengths run..1 */
                        if (run > NMAX - 1) { s_overflow = 1; run = NMAX - 1; }
                        if (prev < slice_hi) {
                            switch (run) {
                            case 1: c1++; break; case 2: c2++; break; case 3: c3++; break; case 4: c4++; break;
                            case 5: c5++; break; case 6: c6++; break; case 7: c7++; break;
                            default: atomicAdd(&s_runs[run], 1u);
                            }
                        } else {
                            for (u32 n = run_out + 1; n <= run; n++) atomicAdd(&s_extra[n], 1u);
                        }
                        if (run > M) {
                            record_from(sh, run_start, prefix + run_start_cnt, run - M, run, slice_hi, s_first);
                            M = run;
                        }
                        run = 1;
                        run_out = (prev >= slice_hi);
                        run_start = prev;
                        run_start_cnt = cnt;
                        if (prev >= slice_hi) { done = true; break; }   /* every prime < slice_hi is settled */
                    }
                    prevgap = g;
                    prev = off;
                    if (off < slice_hi) cnt++;
                }
            }
            if (!done && have_prev) {
                /* run still open at the end of the shared bitmap: flag its first prime */
                u32 startp = run ? run_start : prev;
                if (startp < slice_hi) atomicMin(&s_unsettled, startp);
            }
        }
    }
    /* fold the short-run counts: warp reduce, one atomic per warp per length */
    {
        u32 v[7] = { c1, c2, c3, c4, c5, c6, c7 };
        #pragma unroll
        for (int n = 0; n < 7; n++) {
            u32 s = v[n];
            for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
            if ((tid & 31) == 0 && s) atomicAdd(&s_runs[n + 1], s);
        }
    }
    __syncthreads();
    /* write the segment's results (one SegResult per block) for the host merge */
    if (tid < NMAX) {
        results[seg].first[tid] = (u32)(s_first[tid] >> 32);
        results[seg].firstidx[tid] = (u32)s_first[tid];
        results[seg].runs[tid] = s_runs[tid];
        results[seg].extra[tid] = s_extra[tid];
    }
    if (tid == 0) { results[seg].unsettled = s_unsettled; results[seg].overflow = s_overflow; }
}

/* ------------------------------------------------------------------ */
/* Host side: sieving primes, pattern, GPU context                     */
/* ------------------------------------------------------------------ */

/* host copies of the wheel residues and the residue -> index map (8 = not coprime to 30) */
static const u32 H_R[8] = { 1, 7, 11, 13, 17, 19, 23, 29 };
static const u8  H_RIDX[30] = { 8,0,8,8,8,8,8,1,8,8, 8,2,8,3,8,8,8,4,8,5, 8,8,8,6,8,8,8,8,8,7 };

/*
 * primes_upto() -- all primes <= n, ascending, by a plain odd-only sieve of
 * Eratosthenes on the host.
 * Inputs:  n  (the sieving primes go up to sqrt(END): 2.6e9 at most, ~1.4e9
 *          for END = 2e18, which takes a few seconds and 0.7 GB).
 * Returns: the vector of primes.   Errors: none (std::bad_alloc if n is huge).
 */
static std::vector<u32> primes_upto(u32 n)
{
    std::vector<u32> out;
    if (n < 2) return out;
    std::vector<u8> comp((n >> 1) + 1, 0);           /* comp[i] <-> 2i+1 */
    for (u64 i = 1; (2 * i + 1) * (2 * i + 1) <= n; i++)
        if (!comp[i])
            for (u64 j = (2 * i + 1) * (2 * i + 1) >> 1; j <= (n >> 1); j += 2 * i + 1) comp[j] = 1;
    out.push_back(2);
    for (u64 i = 1; 2 * i + 1 <= n; i++) if (!comp[i]) out.push_back((u32)(2 * i + 1));
    return out;
}

/*
 * build_pattern() -- the presieve pattern: for every byte position b of the
 * period, the bits of the residues 30*b + R[j] (or 210*p + R[j]) that are
 * divisible by one of the pattern primes.  Sieving them once here replaces
 * the five (four) most expensive medium primes in every segment.
 * Inputs:  none.   Returns: PAT_PERIOD (PAT_BYTES) pattern bytes.
 * Errors:  none.
 */
static std::vector<u8> build_pattern(void)
{
    std::vector<u8> pat(PAT_PERIOD, 0);
    static const u32 P[5] = { 7, 11, 13, 17, 19 };
    for (u32 b = 0; b < PAT_PERIOD; b++)
        for (int j = 0; j < 8; j++) {
            u32 n = 30 * b + H_R[j];
            for (int k = 0; k < 5; k++) if (n % P[k] == 0) { pat[b] |= (u8)(1 << j); break; }
        }
    return pat;
}

/* The GPU context: device buffers, pinned host buffers and the sieving-prime
 * split, built by gpu_init() for a given END and freed by gpu_free(). */
struct Gpu {
    u32 *d_bitmap, *d_sprimes, *d_lprimes, *d_mstate;   /* chunk bitmap; medium primes; large primes; dense-prime state */
    u32 *d_sinv, *d_ssegmod, *d_sbeta, *d_rchunk;   /* per medium prime: floor(2^32/q), (SEG_NUMS/30) mod q, -30^-1 mod q, (chunk_lo/30) mod q */
    u8  *d_pattern;                 /* presieve pattern */
    u64 *d_entries;                 /* SLOTS bucket lists of cap entries for the sleeping large primes */
    u32 *d_counts, *d_err, *h_cnt;  /* per-list counts, error flags, pinned copy of {count, err} */
    u32 cap, nd;                    /* list capacity; number of dense large primes (q below the dense threshold) */
    SegResult *d_results, *h_results;   /* per-segment results, device and pinned host copy */
    std::vector<u32> lprimes;       /* large sieving primes (> qsplit), ascending */
    u32 nsmall, nwarpprimes, nlarge;    /* medium primes, those below PWARP, large primes */
    u32 qsplit;
    cudaStream_t sA;                /* the one stream everything runs on */
    cudaEvent_t ev_t0, ev_t1, ev_t2, ev_t3;     /* per-kernel timing for bench -p */
};

/*
 * gpu_init() -- build the sieving-prime tables and allocate the device.
 * Inputs:  g  the context to fill;  max_number  the largest number any chunk
 *          of the scan touches (the sieving primes go up to its square
 *          root);  qsplit  medium/large split.
 * Effects: uploads the wheel tables; sieves the primes up to
 *          sqrt(max_number) on the host and splits them into medium
 *          (23..qsplit, with the warp-per-prime count) and large (> qsplit);
 *          builds the presieve pattern; creates the stream, the chunk
 *          bitmap, the per-segment result buffers (device and pinned host),
 *          the timing events, the bucket lists (capacity from the expected
 *          hits per chunk of the sleepers at END, Sum over sleepers of
 *          CHUNK_NUMS * (wheel density) / q, doubled, plus 2^20), the pinned
 *          {count, err} pair, the per-medium-prime tables and the
 *          large-prime arrays; raises kernel_segment's dynamic shared memory
 *          limit; with g_l2_persist, sets the persisting-L2 limit and an
 *          access policy window over the bitmap on the stream (hit ratio 1,
 *          persisting; misses streaming), capped at the device maxima.
 * Returns: nothing; prints the large-prime split unless g_quiet_init.
 * Errors:  die() on any CUDA error, or if a bucket list would need 2^32 or
 *          more entries.  Memory: 4 bytes of state per large prime, 64 * cap
 *          * 8 bytes for the lists (13 GB at END = 2e18), 8 MB bitmap.
 */
static void gpu_init(Gpu &g, u64 max_number, u32 qsplit)
{
    u8 bitidx[8][8];
    for (int a = 0; a < 8; a++)
        for (int j = 0; j < 8; j++) bitidx[a][j] = H_RIDX[(H_R[a] * H_R[j]) % 30];
    CUDA_CHECK(cudaMemcpyToSymbol(c_BITIDX, bitidx, sizeof bitidx));

    u32 sq = (u32)sqrt((double)max_number);          /* integer square root, corrected below for rounding */
    while ((u64)sq * sq > max_number) sq--;
    while ((u64)(sq + 1) * (sq + 1) <= max_number) sq++;
    std::vector<u32> all = primes_upto(sq);
    std::vector<u32> sp;
    for (u32 q : all) {
        if (q < 23) continue;
        if (q <= qsplit) sp.push_back(q); else g.lprimes.push_back(q);
    }
    g.nsmall = (u32)sp.size();
    g.nwarpprimes = 0;
    while (g.nwarpprimes < g.nsmall && sp[g.nwarpprimes] < PWARP) g.nwarpprimes++;
    g.nlarge = (u32)g.lprimes.size();
    g.qsplit = qsplit;

    std::vector<u8> pat = build_pattern();
    CUDA_CHECK(cudaStreamCreate(&g.sA));
    CUDA_CHECK(cudaMalloc(&g.d_bitmap, (CHUNK_WORDS + OVERLAP) * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&g.d_results, CHUNK_SEGS * sizeof(SegResult)));
    CUDA_CHECK(cudaMallocHost(&g.h_results, CHUNK_SEGS * sizeof(SegResult)));
    CUDA_CHECK(cudaEventCreate(&g.ev_t0)); CUDA_CHECK(cudaEventCreate(&g.ev_t1));
    CUDA_CHECK(cudaEventCreate(&g.ev_t2)); CUDA_CHECK(cudaEventCreate(&g.ev_t3));
    /* sleeping large primes: q >= CHUNK_NUMS/6 may skip chunks; capacity from the expected hits per chunk at END */
    {
        const u32 qdense = (u32)(CHUNK_NUMS / 6);
        g.nd = (u32)(std::upper_bound(g.lprimes.begin(), g.lprimes.end(), qdense - 1) - g.lprimes.begin());
        double expect = 0;
        for (u32 i = g.nd; i < g.nlarge; i++) expect += (double)CHUNK_NUMS * 8.0 / 30.0 / g.lprimes[i];
        u64 cap = (u64)(2.0 * expect) + (1u << 20);
        if (cap > 0xFFFFFFF0ULL) die("bucket capacity too large");
        g.cap = (u32)cap;
        CUDA_CHECK(cudaMalloc(&g.d_entries, (u64)SLOTS * g.cap * sizeof(u64)));
        CUDA_CHECK(cudaMalloc(&g.d_counts, SLOTS * sizeof(u32)));
        CUDA_CHECK(cudaMalloc(&g.d_err, sizeof(u32)));
        CUDA_CHECK(cudaMemset(g.d_counts, 0, SLOTS * sizeof(u32)));
        CUDA_CHECK(cudaMemset(g.d_err, 0, sizeof(u32)));
        CUDA_CHECK(cudaMallocHost(&g.h_cnt, 2 * sizeof(u32)));
        if (!g_quiet_init)
            fprintf(stderr, "large primes: %u dense (< %u), %u sleepers in %d bucket lists of %u entries (%.1f GB)\n",
                    g.nd, qdense, g.nlarge - g.nd, SLOTS, g.cap, (double)SLOTS * g.cap * 8 / 1e9);
    }
    CUDA_CHECK(cudaMalloc(&g.d_pattern, PAT_PERIOD));
    CUDA_CHECK(cudaMemcpy(g.d_pattern, pat.data(), PAT_PERIOD, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&g.d_sprimes, (g.nsmall + 1) * sizeof(u32)));
    CUDA_CHECK(cudaMemcpy(g.d_sprimes, sp.data(), g.nsmall * sizeof(u32), cudaMemcpyHostToDevice));
    {
        std::vector<u32> inv(g.nsmall), segmod(g.nsmall), beta(g.nsmall);
        for (u32 i = 0; i < g.nsmall; i++) {
            const u32 q = sp[i];
            inv[i] = (u32)((1ULL << 32) / q);
            segmod[i] = (u32)((SEG_NUMS / 30) % q);
            u64 inv30 = powmod(30, q - 2, q);            /* 30^-1 mod q (q prime, q > 30) */
            beta[i] = (u32)((q - inv30) % q);
        }
        CUDA_CHECK(cudaMalloc(&g.d_sinv, (g.nsmall + 1) * sizeof(u32)));
        CUDA_CHECK(cudaMalloc(&g.d_ssegmod, (g.nsmall + 1) * sizeof(u32)));
        CUDA_CHECK(cudaMalloc(&g.d_sbeta, (g.nsmall + 1) * sizeof(u32)));
        CUDA_CHECK(cudaMalloc(&g.d_rchunk, (g.nsmall + 1) * sizeof(u32)));
        CUDA_CHECK(cudaMemcpy(g.d_sinv, inv.data(), g.nsmall * sizeof(u32), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g.d_ssegmod, segmod.data(), g.nsmall * sizeof(u32), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g.d_sbeta, beta.data(), g.nsmall * sizeof(u32), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&g.d_lprimes, (g.nlarge + 1) * sizeof(u32)));
    if (g.nlarge) CUDA_CHECK(cudaMemcpy(g.d_lprimes, g.lprimes.data(), g.nlarge * sizeof(u32), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&g.d_mstate, (g.nlarge + 1) * sizeof(u32)));
    CUDA_CHECK(cudaFuncSetAttribute(kernel_segment, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_WORDS * sizeof(u32)));
    /* Keep the chunk bitmap resident in L2: the large-prime atomics then run
     * at L2 speed.  Both limits are capped at what the device allows. */
    if (g_l2_persist) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        size_t bytes = (CHUNK_WORDS + OVERLAP) * sizeof(u32);
        size_t persist = bytes < (size_t)prop.persistingL2CacheMaxSize ? bytes : (size_t)prop.persistingL2CacheMaxSize;
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persist));
        size_t win = bytes;
        if (win > (size_t)prop.accessPolicyMaxWindowSize) win = prop.accessPolicyMaxWindowSize;
        cudaStreamAttrValue attr;
        memset(&attr, 0, sizeof attr);
        attr.accessPolicyWindow.base_ptr = g.d_bitmap;
        attr.accessPolicyWindow.num_bytes = win;
        attr.accessPolicyWindow.hitRatio = 1.0f;
        attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        CUDA_CHECK(cudaStreamSetAttribute(g.sA, cudaStreamAttributeAccessPolicyWindow, &attr));
    }
}

/*
 * active_count() -- number of large primes q with q*q < limit, i.e. the
 * primes that must be sieving by the time the scan reaches limit (they are
 * the first that many of g.lprimes).
 * Inputs:  g  context;  limit.   Returns: the count.   Errors: none.
 */
static u32 active_count(const Gpu &g, u64 limit)
{
    u32 sq = (u32)sqrt((double)limit);
    while ((u64)sq * sq >= limit && sq) sq--;
    while ((u64)(sq + 1) * (sq + 1) < limit) sq++;
    return (u32)(std::upper_bound(g.lprimes.begin(), g.lprimes.end(), sq) - g.lprimes.begin());
}

/* ------------------------------------------------------------------ */
/* Scan state and merge                                                */
/* ------------------------------------------------------------------ */

/* A first occurrence: the smallest prime with run length n.  Final as soon
 * as it is recorded, because chunks and segments are merged in order. */
struct Cand {
    u64 p;                  /* 0 = none */
    u64 pi;                 /* primes in [start, p] (exact: segments merge in order) */
    u16 gaps[NMAX + 1];     /* the n gaps of the run, then the gap that ends it (from run_length64) */
};

/* The scan state on the host.  Reset by run_scan() for every scan, written
 * by merge_chunk(), saved and restored by the checkpoint. */
static struct {
    u64 start, end;         /* requested bounds */
    u64 next_chunk;         /* next chunk index to process (chunks [k_lo, next) done) */
    u64 k_lo, k_hi;         /* chunk index range [k_lo, k_hi) */
    u64 primes;             /* primes in [start, pos) */
    u64 resume_pos;         /* numbers below this were scanned by an earlier run (= start for a fresh run) */
    u64 hist[NMAX];         /* hist[n] = primes with run length n in [start, pos) */
    Cand best[NMAX];        /* best[n] = first prime with run length n */
    bool overflow;          /* some run was clamped to NMAX-1 gaps */
    u64 cpu_settled;        /* primes settled by the CPU fallback */
} S;

/*
 * print_cand() -- print "<tag> a(n) = p  gaps: ... | g  (pi(p) = k)" on
 * stdout (line-buffered), clearing the status line first on a terminal.
 */
static void print_cand(int n, const Cand *c, const char *tag)
{
    if (stderr_tty) fputs("\r\033[K", stderr);
    printf("%s a(%d) = %" PRIu64 "  gaps:", tag, n, c->p);
    for (int i = 0; i < n; i++) printf(" %u", c->gaps[i]);
    printf(" | %u  (pi(p) = %" PRIu64 ")\n", c->gaps[n], c->pi);
    fflush(stdout);
}

/*
 * record_first() -- record prime p as the first with run length n.
 * Inputs:  n  run length;  p  the prime;  pi  its prime index (primes in
 *          [start, p]).
 * Effects: if S.best[n] is still empty: stores p and pi, RE-DERIVES the run
 *          and its gaps with the host's next-prime search (run_length64) and
 *          prints the CONFIRMED line when n >= g_report.  Later calls for the
 *          same n are ignored (segments arrive in order, so the first is the
 *          smallest).
 * Errors:  die() if the next-prime search finds a different run length: a
 *          GPU/CPU disagreement never reaches the output silently.
 */
static void record_first(int n, u64 p, u64 pi)
{
    if (n < 1 || n >= NMAX || S.best[n].p) return;
    Cand *c = &S.best[n];
    c->p = p;
    c->pi = pi;
    int L = run_length64(p, c->gaps, NMAX + 1);
    if (L != n) die("GPU/CPU disagreement: GPU says L(%" PRIu64 ") = %d, next-prime search says %d", p, n, L);
    if (n >= g_report) print_cand(n, c, "CONFIRMED");
}

/*
 * settle_cpu() -- CPU fallback: settle every prime in [lo, hi) directly.
 * Inputs:  lo, hi  the range;  pi_before  primes in [start, lo).
 * Effects: walks the primes with next_prime64, computes each run length by
 *          next-prime search, updates the histogram, S.cpu_settled and the
 *          first occurrences (S.primes is not touched: the segment's count
 *          already includes these primes).  Only used for the tail of a
 *          segment whose last run was still open at the end of the block's
 *          bitmap; never needed so far, and slow if the tail were long.
 * Errors:  die() if a run exceeds NMAX gaps.
 */
static void settle_cpu(u64 lo, u64 hi, u64 pi_before)
{
    u64 p = is_prime64(lo) ? lo : next_prime64(lo);
    u64 pi = pi_before;
    for (; p < hi; p = next_prime64(p)) {
        u16 gaps[NMAX + 1];
        pi++;
        int L = run_length64(p, gaps, NMAX + 1);
        if (L < 0) die("run longer than %d at %" PRIu64, NMAX, p);
        S.hist[L]++;
        S.cpu_settled++;
        record_first(L, p, pi);
    }
}

/*
 * merge_chunk() -- fold one chunk's segment results into the scan state.
 * Inputs:  results  CHUNK_SEGS SegResults (the pinned host copy);  chunk_lo.
 * Effects: for each segment in order, skipping segments entirely below the
 *          resume position or at/above END: records first occurrences with
 *          pi = primes before the segment + index inside it; adds the
 *          histogram (primes with run length n = run ends of length >= n
 *          plus the final-run extras, exactly as in the CPU tool); if a run
 *          was left unsettled, counts the primes from the segment's first
 *          counted number up to the flagged prime by next-prime search and
 *          settles the rest of the segment on the CPU; then adds the
 *          segment's prime count.  S.primes is exact after every segment,
 *          which is what makes pi(a(n)) exact.
 * Errors:  record_first() and settle_cpu() may die (see there).
 */
static void merge_chunk(const SegResult *results, u64 chunk_lo)
{
    for (u32 s = 0; s < CHUNK_SEGS; s++) {
        const SegResult *r = &results[s];
        const u64 seg_lo = chunk_lo + (u64)s * SEG_NUMS;
        const u64 seg_hi = seg_lo + SEG_NUMS;
        if (seg_hi <= S.resume_pos || seg_lo >= S.end) continue;
        if (r->overflow) S.overflow = true;
        /* first occurrences (segments and chunks arrive in order, so first seen = smallest) */
        for (int n = 1; n < NMAX; n++)
            if (r->first[n] != ~0u) record_first(n, seg_lo + r->first[n], S.primes + r->firstidx[n]);
        /* histogram: primes with run length n = run ends of length >= n, plus final-run extras */
        u64 acc = 0;
        for (int n = NMAX - 1; n >= 1; n--) { acc += r->runs[n]; S.hist[n] += acc + r->extra[n]; }
        if (r->unsettled != ~0u) {
            /* a run was still open at the end of the block's bitmap: settle [p0, hi) on the CPU */
            u64 p0 = seg_lo + r->unsettled, hi = seg_hi < S.end ? seg_hi : S.end;
            u64 before = S.primes;                  /* primes below seg_lo (∩ [start,·)) */
            u64 lo_n = seg_lo > S.resume_pos ? seg_lo : S.resume_pos;
            for (u64 p = is_prime64(lo_n) ? lo_n : next_prime64(lo_n); p < p0; p = next_prime64(p)) before++;
            settle_cpu(p0, hi, before);
        }
        S.primes += r->count;
    }
}

/* ------------------------------------------------------------------ */
/* Checkpoint                                                          */
/* ------------------------------------------------------------------ */

/*
 * save_state() -- write the checkpoint file atomically (path.tmp, then
 * rename).  Format:
 *     A158939 cuda checkpoint 2
 *     start N / end N / pos N / primes N / cpu_settled N
 *     hist N COUNT                 (every non-zero entry)
 *     cand N P PI G1 ... G(N+1)    (every first occurrence)
 *     end
 * pos is the position up to which the scan is complete: the next chunk
 * boundary, or END when the last chunk is partial (an early version wrote
 * the boundary in that case too and a later resume skipped the rest of the
 * chunk).
 * Errors:  I/O failures print a warning and leave the old file in place.
 */
static void save_state(const char *path)
{
    char tmp[PATH_MAX];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "w");
    if (!f) { fprintf(stderr, "\nwarning: cannot write %s: %s\n", tmp, strerror(errno)); return; }
    u64 pos = S.next_chunk * CHUNK_NUMS;
    if (pos > S.end) pos = S.end;                /* a partial last chunk is done only up to END */
    fprintf(f, "A158939 cuda checkpoint 2\nstart %" PRIu64 "\nend %" PRIu64 "\npos %" PRIu64
               "\nprimes %" PRIu64 "\ncpu_settled %" PRIu64 "\n", S.start, S.end, pos, S.primes, S.cpu_settled);
    for (int n = 1; n < NMAX; n++)
        if (S.hist[n]) fprintf(f, "hist %d %" PRIu64 "\n", n, S.hist[n]);
    for (int n = 1; n < NMAX; n++) {
        if (!S.best[n].p) continue;
        fprintf(f, "cand %d %" PRIu64 " %" PRIu64, n, S.best[n].p, S.best[n].pi);
        for (int i = 0; i <= n; i++) fprintf(f, " %u", S.best[n].gaps[i]);
        fputc('\n', f);
    }
    fputs("end\n", f);
    if (fclose(f) != 0 || rename(tmp, path) != 0)
        fprintf(stderr, "\nwarning: cannot update %s: %s\n", path, strerror(errno));
}

/*
 * load_state() -- read a checkpoint written by save_state().
 * Inputs:  path.  S.start and S.end must already hold the command line's
 *          values.
 * Effects: restores S.primes, S.cpu_settled, S.hist[], S.best[] and sets
 *          S.next_chunk to the chunk containing pos and S.resume_pos = pos,
 *          so that chunk is rescanned from pos.  A different END only
 *          produces a note (that is how a finished scan is extended).
 * Returns: true if loaded; false if the file does not exist.
 * Errors:  die() if the file cannot be read, has the wrong header, a bad
 *          hist/cand line, no "end" line, a different START, or a position
 *          below START.
 */
static bool load_state(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        if (errno == ENOENT) return false;
        die("cannot read %s: %s", path, strerror(errno));
    }
    char line[2048];
    if (!fgets(line, sizeof line, f) || strncmp(line, "A158939 cuda checkpoint 2", 25) != 0)
        die("%s is not an A158939 cuda checkpoint", path);
    u64 start = 0, end = 0, pos = 0, primes = 0, cpu = 0;
    bool complete = false;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "start %" SCNu64, &start) == 1) continue;
        if (sscanf(line, "end %" SCNu64, &end) == 1) continue;
        if (sscanf(line, "pos %" SCNu64, &pos) == 1) continue;
        if (sscanf(line, "primes %" SCNu64, &primes) == 1) continue;
        if (sscanf(line, "cpu_settled %" SCNu64, &cpu) == 1) continue;
        if (strncmp(line, "hist ", 5) == 0) {
            int n = 0; u64 c = 0;
            if (sscanf(line, "hist %d %" SCNu64, &n, &c) != 2 || n < 1 || n >= NMAX) die("%s: bad hist line", path);
            S.hist[n] = c;
            continue;
        }
        if (strncmp(line, "cand ", 5) == 0) {
            char *e;
            long n = strtol(line + 5, &e, 10);
            if (n < 1 || n >= NMAX) die("%s: bad cand line", path);
            S.best[n].p = strtoull(e, &e, 10);
            S.best[n].pi = strtoull(e, &e, 10);
            for (int i = 0; i <= n; i++) S.best[n].gaps[i] = (u16)strtoul(e, &e, 10);
            continue;
        }
        if (strncmp(line, "end", 3) == 0) { complete = true; break; }
    }
    fclose(f);
    if (!complete) die("%s is truncated", path);
    if (start != S.start) die("%s was written for START %" PRIu64 ", not %" PRIu64, path, start, S.start);
    if (end != S.end) fprintf(stderr, "note: checkpoint END was %" PRIu64 ", continuing to %" PRIu64 "\n", end, S.end);
    if (pos < S.start) die("%s: position %" PRIu64 " is below START", path, pos);
    S.next_chunk = pos / CHUNK_NUMS;             /* the chunk containing pos is rescanned from pos */
    S.resume_pos = pos;
    S.primes = primes;
    S.cpu_settled = cpu;
    return true;
}

/* on_sigint() -- SIGINT/SIGTERM: request a stop after the current chunk
 * (the loop then saves the checkpoint and prints the table).  Only a flag is
 * set, so the handler is async-signal-safe. */
static void on_sigint(int sig) { (void)sig; g_stop = 1; }

/* ------------------------------------------------------------------ */
/* Scan driver                                                         */
/* ------------------------------------------------------------------ */

/* What run_scan() reports: wall time, numbers scanned by this run (whole
 * chunks), total primes in [start, pos) including resumed work. */
struct ScanStats { double seconds; u64 numbers; u64 primes; };

static Gpu G;                       /* the one GPU context, rebuilt by every run_scan() */
static bool G_ready = false;
static bool g_profile = false;      /* bench -p: time the kernels with events */
static double prof_large, prof_segment, prof_memset, prof_host;   /* seconds */

/* gpu_free() -- release everything gpu_init() allocated (after a device
 * synchronize).  Errors are ignored: this runs between scans and at exit. */
static void gpu_free(Gpu &g)
{
    cudaDeviceSynchronize();
    cudaFree(g.d_bitmap); cudaFree(g.d_results); cudaFreeHost(g.h_results);
    cudaEventDestroy(g.ev_t0); cudaEventDestroy(g.ev_t1); cudaEventDestroy(g.ev_t2); cudaEventDestroy(g.ev_t3);
    cudaFree(g.d_pattern); cudaFree(g.d_sprimes); cudaFree(g.d_lprimes); cudaFree(g.d_mstate);
    cudaFree(g.d_sinv); cudaFree(g.d_ssegmod); cudaFree(g.d_sbeta); cudaFree(g.d_rchunk);
    cudaFree(g.d_entries); cudaFree(g.d_counts); cudaFree(g.d_err); cudaFreeHost(g.h_cnt);
    cudaStreamDestroy(g.sA);
    g.lprimes.clear();
}

/*
 * run_scan() -- scan [start, end) on the GPU: the driver behind the scan,
 * bench and selftest commands (section 1.3 of the header).
 * Inputs:  start, end  the range (any numbers: chunks are aligned to
 *                      CHUNK_NUMS and the kernels count only inside
 *                      [start, end));
 *          qsplit      medium/large prime split;
 *          state       checkpoint file name or NULL;
 *          interval    seconds between checkpoints and, without a terminal,
 *                      between progress lines;
 *          stop_n      stop once a(stop_n) is found (0 = never);
 *          quiet       no status line, no start/stop notes.
 * Effects: resets S, loads the checkpoint if any, (re)builds the GPU context
 *          for sqrt(end of the last chunk + overlap), adds the primes the
 *          wheel leaves out when the range contains them and the scan is
 *          not resumed, installs the SIGINT/SIGTERM handler, builds the
 *          initial large-prime state (kernel_init_state) and runs the
 *          per-chunk pipeline in order on one stream: activate, clear the
 *          bitmap, dense primes, read this chunk's bucket count and the
 *          error flags, sleepers, chunkmod, segment kernel, copy results,
 *          merge on the host.  Keeps a sliding window of chunk positions
 *          for the rate and ETA, prints the status line and saves the
 *          checkpoint every `interval` seconds.  After the loop saves the
 *          final checkpoint and warns about a run-length overflow, CPU-
 *          settled primes or an early stop.  Results stay in S for
 *          print_table()/print_hist().
 * Returns: ScanStats.
 * Errors:  die() if end <= start, on any CUDA error, on a bucket-list
 *          overflow or a misfiled entry, or on a GPU/CPU disagreement in
 *          record_first().
 */
static ScanStats run_scan(u64 start, u64 end, u32 qsplit, const char *state, int interval, int stop_n, bool quiet)
{
    if (end <= start) die("END must be larger than START");
    memset(&S, 0, sizeof S);
    S.start = start;
    S.end = end;
    S.k_lo = start / CHUNK_NUMS;
    S.k_hi = (end + CHUNK_NUMS - 1) / CHUNK_NUMS;
    S.next_chunk = S.k_lo;
    S.resume_pos = start;
    const bool resumed = state && load_state(state);
    if (resumed)
        fprintf(stderr, "resuming from %s at position %" PRIu64 " (%" PRIu64 " primes so far)\n",
                state, S.resume_pos, S.primes);
    const u64 max_number = S.k_hi * CHUNK_NUMS + (u64)OVERLAP * NUMS_PER_WORD;
    if (G_ready) gpu_free(G);
    g_quiet_init = quiet;
    gpu_init(G, max_number, qsplit);
    G_ready = true;

    if (!quiet)
        fprintf(stderr, "scanning [%" PRIu64 ", %" PRIu64 ") in %" PRIu64 " chunks of %" PRIu64
                        " (%u small + %u large sieving primes, QSPLIT %u)%s\n",
                start, end, S.k_hi - S.k_lo, (u64)CHUNK_NUMS, G.nsmall, G.nlarge, qsplit,
                stop_n ? ", stopping when a(N) is found" : "");

    /* the primes 2, 3, 5 are outside the wheel */
    if (!resumed) {
        static const u64 sp[3] = { 2, 3, 5 };
        static const int sL[3] = { 2, 1, 2 };
        for (int i = 0; i < 3; i++)
            if (sp[i] >= start && sp[i] < end) { S.primes++; S.hist[sL[i]]++; record_first(sL[i], sp[i], S.primes); }
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_sigint;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    /* state of the large primes for the first chunk: dense array + bucket lists */
    u64 k = S.next_chunk;
    u32 nprev = active_count(G, k * CHUNK_NUMS + CHUNK_NUMS + OVERLAP_NUMS);
    CUDA_CHECK(cudaMemsetAsync(G.d_counts, 0, SLOTS * sizeof(u32), G.sA));
    CUDA_CHECK(cudaMemsetAsync(G.d_err, 0, sizeof(u32), G.sA));
    if (nprev) {
        kernel_init_state<<<(nprev + 255) / 256, 256, 0, G.sA>>>(G.d_lprimes, G.d_mstate, G.nd, nprev, k,
                                                                 G.d_entries, G.d_counts, G.cap, G.d_err);
        CUDA_CHECK(cudaGetLastError());
    }

    const double t0 = now();
    const u64 k0 = k;
    double last_status = 0, last_ckpt = t0;
    enum { NS = 128 };
    double ts[NS]; u64 ks[NS]; int ns = 1;
    ts[0] = t0; ks[0] = k0;
    char b1[32], b2[32];

    /* The chunk loop.  Everything is issued on the one stream in order; the
     * host blocks twice per chunk (to read the bucket count and error flags
     * before kernel_bucket, and to receive the results before the merge). */
    for (; k < S.k_hi && !g_stop; k++) {
        const u64 chunk_lo = k * CHUNK_NUMS, chunk_hi = chunk_lo + CHUNK_NUMS;
        const u64 hi_ext = chunk_hi + OVERLAP_NUMS;
        const u32 nact = active_count(G, hi_ext);
        if (g_profile) CUDA_CHECK(cudaEventRecord(G.ev_t0, G.sA));
        if (nact > nprev)
            kernel_activate<<<(nact - nprev + 255) / 256, 256, 0, G.sA>>>(G.d_lprimes, G.d_mstate, G.nd, nprev, nact, k,
                                                                          G.d_entries, G.d_counts, G.cap, G.d_err);
        CUDA_CHECK(cudaMemsetAsync(G.d_bitmap, 0, (CHUNK_WORDS + OVERLAP) * sizeof(u32), G.sA));
        if (g_profile) CUDA_CHECK(cudaEventRecord(G.ev_t1, G.sA));
        const u32 n_dense = nact < G.nd ? nact : G.nd;
        if (n_dense)
            kernel_large<<<(n_dense + 255) / 256, 256, 0, G.sA>>>(G.d_lprimes, G.d_mstate, n_dense,
                                                                   k, chunk_lo, chunk_hi, hi_ext, G.d_bitmap);
        /* sleepers due in this chunk: read the list length, process, clear the list */
        CUDA_CHECK(cudaMemcpyAsync(G.h_cnt, G.d_counts + (k % SLOTS), sizeof(u32), cudaMemcpyDeviceToHost, G.sA));
        CUDA_CHECK(cudaMemcpyAsync(G.h_cnt + 1, G.d_err, sizeof(u32), cudaMemcpyDeviceToHost, G.sA));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(G.sA));
        if (G.h_cnt[1])
            die("bucket list error %u at chunk %" PRIu64 " (capacity %u entries)", G.h_cnt[1], k, G.cap);
        const u32 n_in = G.h_cnt[0];
        if (n_in)
            kernel_bucket<<<(n_in + BLOCKDIM_B - 1) / BLOCKDIM_B, BLOCKDIM_B, 0, G.sA>>>(
                G.d_entries, G.d_counts, G.cap, G.d_err, n_in, k, chunk_lo, chunk_hi, hi_ext, G.d_bitmap);
        CUDA_CHECK(cudaMemsetAsync(G.d_counts + (k % SLOTS), 0, sizeof(u32), G.sA));
        if (g_profile) CUDA_CHECK(cudaEventRecord(G.ev_t2, G.sA));
        kernel_chunkmod<<<(G.nsmall + 255) / 256, 256, 0, G.sA>>>(G.d_sprimes, G.d_rchunk, G.nsmall, chunk_lo);
        const u64 lo_bound = (chunk_lo < S.resume_pos) ? S.resume_pos : start;   /* rescanned chunk: count from resume_pos */
        kernel_segment<<<CHUNK_SEGS, BLOCKDIM, SHARED_WORDS * sizeof(u32), G.sA>>>(
            G.d_bitmap, G.d_pattern, G.d_sprimes, G.d_sinv, G.d_ssegmod, G.d_sbeta, G.d_rchunk, G.nsmall, G.nwarpprimes,
            chunk_lo, lo_bound, end, G.d_results);
        if (g_profile) CUDA_CHECK(cudaEventRecord(G.ev_t3, G.sA));
        CUDA_CHECK(cudaMemcpyAsync(G.h_results, G.d_results, CHUNK_SEGS * sizeof(SegResult), cudaMemcpyDeviceToHost, G.sA));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(G.sA));
        if (g_profile) {
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, G.ev_t0, G.ev_t1)); prof_memset += ms * 1e-3;
            CUDA_CHECK(cudaEventElapsedTime(&ms, G.ev_t1, G.ev_t2)); prof_large += ms * 1e-3;
            CUDA_CHECK(cudaEventElapsedTime(&ms, G.ev_t2, G.ev_t3)); prof_segment += ms * 1e-3;
        }
        double th0 = now();
        merge_chunk(G.h_results, chunk_lo);
        prof_host += now() - th0;
        nprev = nact;
        S.next_chunk = k + 1;
        if (stop_n > 0 && S.best[stop_n].p) g_stop = 1;    /* -n N: a(N) is final as soon as it is recorded */

        double t = now();
        if (t - ts[ns - 1] >= 1.0) {
            if (ns < NS) { ts[ns] = t; ks[ns] = S.next_chunk; ns++; }
            else { memmove(ts, ts + 1, (NS - 1) * sizeof *ts); memmove(ks, ks + 1, (NS - 1) * sizeof *ks); ts[NS - 1] = t; ks[NS - 1] = S.next_chunk; }
        }
        bool show = stderr_tty ? (t - last_status >= 1.0) : (t - last_status >= interval);
        if (show && !quiet) {
            last_status = t;
            double rate = (t - ts[0] > 0.5) ? (double)(S.next_chunk - ks[0]) * CHUNK_NUMS / (t - ts[0]) : 0;
            double eta = rate > 0 ? (double)(S.k_hi - S.next_chunk) * CHUNK_NUMS / rate : -1;
            int maxc = 0;
            for (int n = 1; n < NMAX; n++) if (S.best[n].p) maxc = n;
            fprintf(stderr, "%s%s  pos %.6g (%.2f%%)  %.3g/s  %.4g primes  ETA %s  found to n=%d",
                    stderr_tty ? "\r\033[K" : "progress: ", fmt_dur(t - t0, b1, sizeof b1),
                    (double)(S.next_chunk * CHUNK_NUMS), 100.0 * (double)(S.next_chunk - S.k_lo) / (double)(S.k_hi - S.k_lo),
                    rate, (double)S.primes, fmt_dur(eta, b2, sizeof b2), maxc);
            if (stop_n > 0 && !S.best[stop_n].p) fprintf(stderr, "  a(%d) > %.4g", stop_n, (double)(S.next_chunk * CHUNK_NUMS));
            if (!stderr_tty) fputc('\n', stderr);
            fflush(stderr);
        }
        if (state && t - last_ckpt >= interval) { last_ckpt = t; save_state(state); }
    }
    double t1 = now();
    if (stderr_tty && !quiet) fputs("\r\033[K", stderr);
    if (state) save_state(state);
    if (S.overflow) fprintf(stderr, "warning: a run exceeded %d gaps; longer run lengths were clamped\n", NMAX - 1);
    if (S.cpu_settled) fprintf(stderr, "note: %" PRIu64 " primes were settled by the CPU fallback\n", S.cpu_settled);
    if (g_stop && S.next_chunk < S.k_hi && !quiet)
        fprintf(stderr, "stopped at position %" PRIu64 "%s\n", (u64)(S.next_chunk * CHUNK_NUMS), state ? ", checkpoint saved" : "");
    ScanStats st;
    st.seconds = t1 - t0;
    st.numbers = (S.next_chunk - k0) * CHUNK_NUMS;
    st.primes = S.primes;
    return st;
}

/* scan_pos() -- position up to which the scan is complete (exclusive): the
 * next chunk boundary, or END inside the last chunk. */
static u64 scan_pos(void) { u64 p = S.next_chunk * CHUNK_NUMS; return p < S.end ? p : S.end; }

/*
 * print_hist() -- print the prime count of [S.start, scan_pos()), compared
 * with pi(10^k) when the scan started at 0 and ended exactly at 10^k, then
 * the run-length histogram next to the i.i.d.-exponential-gap model
 * primes * n/(n+1)! and the ratio of the two.
 */
static void print_hist(void)
{
    const u64 pos = scan_pos();
    printf("\n%" PRIu64 " primes in [%" PRIu64 ", %" PRIu64 ")", S.primes, S.start, pos);
    if (S.start == 0)
        for (int k = 6; k <= 16; k++)
            if (pos == (u64)pow(10, k)) printf(S.primes == PI10[k] ? "  = pi(10^%d)" : "  BUT pi(10^%d) = %" PRIu64, k, PI10[k]);
    printf("\nrun-length distribution (model: gaps i.i.d. exponential, P(L = n) = n/(n+1)!):\n");
    printf("  n  primes with L(p) = n     model        ratio\n");
    double fact = 1;
    for (int n = 1; n < NMAX; n++) {
        fact *= (n + 1);
        double model = (double)S.primes * n / fact;
        if (S.hist[n] == 0 && model < 0.01) break;
        printf(" %2d  %-24" PRIu64 " %-12.4g %.3f\n", n, S.hist[n], model, model > 0 ? (double)S.hist[n] / model : 0);
    }
    fflush(stdout);
}

/*
 * print_table() -- print the a(n) table after a scan: a(n) (or the first
 * prime >= START with run length n), pi(a(n)), the gaps and the gap that
 * ends the run, the comparison with the OEIS (KNOWN[], A133697[]) for scans
 * from 0 and "NEW" beyond the published terms; then the lower bound for the
 * next term and print_hist().
 */
static void print_table(void)
{
    int maxn = 0;
    for (int n = 1; n < NMAX; n++) if (S.best[n].p) maxn = n;
    printf("\n %2s  %-24s %-21s %s\n", "n", S.start ? "a(n) (>= START)" : "a(n)",
           S.start ? "pi(a(n)) (from START)" : "pi(a(n))", "gaps | gap that ends the run");
    for (int n = 1; n <= maxn; n++) {
        const Cand *c = &S.best[n];
        if (!c->p) { printf(" %2d  (none found)\n", n); continue; }
        printf(" %2d  %-24" PRIu64 " %-21" PRIu64, n, c->p, c->pi);
        for (int i = 0; i < n; i++) printf(" %u", c->gaps[i]);
        printf(" | %u", c->gaps[n]);
        if (S.start == 0 && n <= NKNOWN)
            printf(c->p == KNOWN[n] ? "  = OEIS" : "  DIFFERS from OEIS %" PRIu64, KNOWN[n]);
        else if (S.start == 0)
            printf("  NEW");
        if (S.start == 0 && n >= 2 && n - 2 < NA133697)
            printf(c->pi == A133697[n - 2] ? ", pi = A133697(%d)" : ", pi DIFFERS from A133697(%d) = %" PRIu64, n - 2, A133697[n - 2]);
        else if (S.start == 0 && n >= 2)
            printf(", pi = new A133697(%d)", n - 2);
        putchar('\n');
    }
    printf(" %2d  > %" PRIu64 "  (no%s prime below this position has run length %d)\n",
           maxn + 1, scan_pos(), S.start ? " such" : "", maxn + 1);
    print_hist();
}

/* ------------------------------------------------------------------ */
/* Commands                                                            */
/* ------------------------------------------------------------------ */

/* usage() -- print the command summary to stderr and exit(2); never returns. */
static void usage(void)
{
    fprintf(stderr,
        "usage: a158939_cuda scan [START] END [-n N] [-r NMIN] [-S FILE] [-i SECS] [-Q QSPLIT] [-N] [-q]\n"
        "       a158939_cuda bench [N] [-d SPAN] [-Q QSPLIT] [-N] [-p]\n"
        "       a158939_cuda selftest [-Q QSPLIT]\n");
    exit(2);
}

/*
 * parse_qsplit() -- parse -Q: the medium/large prime split, between PWARP
 * (below which primes would need the warp path anyway) and 10^8.
 * Errors:  die() outside that range.
 */
static u32 parse_qsplit(const char *s)
{
    u64 v = parse_num(s);
    if (v < PWARP || v > 100000000ULL) die("QSPLIT must be between %d and 10^8", PWARP);
    return (u32)v;
}

/*
 * cmd_scan() -- the `scan` command: exhaustive scan of [START, END).
 * Inputs:  one or two positional numbers (END, or START END) and the
 *          options -n N, -r NMIN, -S FILE, -i SECS, -Q QSPLIT, -N, -q
 *          (-1 and -L are accepted and ignored).
 * Effects: runs the scan, prints the table and histogram on stdout and a
 *          one-line timing summary on stderr.
 * Returns: 0.   Errors: die() from run_scan; usage() on bad arguments.
 */
static int cmd_scan(int argc, char **argv)
{
    u64 pos[2];
    int npos = 0, interval = 60, stop_n = 0;
    u32 qsplit = 4000000;
    const char *state = NULL;
    bool quiet = false;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) stop_n = parse_int(argv[++i], 1, NMAX - 1, "N");
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) g_report = parse_int(argv[++i], 1, 1000, "NMIN");
        else if (!strcmp(argv[i], "-S") && i + 1 < argc) state = argv[++i];
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) interval = parse_int(argv[++i], 1, 86400, "SECS");
        else if (!strcmp(argv[i], "-Q") && i + 1 < argc) qsplit = parse_qsplit(argv[++i]);
        else if (!strcmp(argv[i], "-1") || !strcmp(argv[i], "-L")) ;   /* accepted for compatibility */
        else if (!strcmp(argv[i], "-N")) g_l2_persist = false;
        else if (!strcmp(argv[i], "-q")) quiet = true;
        else if (argv[i][0] == '-' && !isdigit((unsigned char)argv[i][1])) usage();
        else if (npos < 2) pos[npos++] = parse_num(argv[i]);
        else usage();
    }
    if (npos == 0) usage();
    u64 start = npos == 2 ? pos[0] : 0, end = pos[npos - 1];
    ScanStats st = run_scan(start, end, qsplit, state, interval, stop_n, quiet);
    print_table();
    char b[32];
    fprintf(stderr, "%" PRIu64 " numbers, %" PRIu64 " primes in %s (%.3g numbers/s)\n",
            st.numbers, st.primes, fmt_dur(st.seconds, b, sizeof b), st.seconds > 0 ? (double)st.numbers / st.seconds : 0);
    return 0;
}

/*
 * cmd_bench() -- the `bench` command: throughput of [N, N+SPAN) (defaults
 * 1e15 and 2e12), quietly and without a checkpoint; prints numbers/s and
 * primes/s, with -p the per-kernel GPU times and the host merge time, the
 * projected wall time of full scans at that rate (a guide only: the sieve is
 * faster below N and slower above it) and the window's histogram.
 * Returns: 0.   Errors: die() from run_scan; usage() on bad arguments.
 */
static int cmd_bench(int argc, char **argv)
{
    u64 N = 1000000000000000ULL, span = 2000000000000ULL;
    u32 qsplit = 4000000;
    int npos = 0;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "-d") && i + 1 < argc) span = parse_num(argv[++i]);
        else if (!strcmp(argv[i], "-p")) g_profile = true;
        else if (!strcmp(argv[i], "-1") || !strcmp(argv[i], "-L")) ;   /* accepted for compatibility */
        else if (!strcmp(argv[i], "-N")) g_l2_persist = false;
        else if (!strcmp(argv[i], "-Q") && i + 1 < argc) qsplit = parse_qsplit(argv[++i]);
        else if (argv[i][0] == '-' && !isdigit((unsigned char)argv[i][1])) usage();
        else if (npos++ == 0) N = parse_num(argv[i]);
        else usage();
    }
    g_report = NMAX;
    ScanStats st = run_scan(N, N + span, qsplit, NULL, 60, 0, true);
    double rate = (double)st.numbers / st.seconds;
    printf("bench: [%" PRIu64 ", %" PRIu64 ") QSPLIT %u: %.2f s, %.4g numbers/s, %.4g primes/s\n",
           N, N + span, qsplit, st.seconds, rate, (double)st.primes / st.seconds);
    if (g_profile)
        printf("       GPU time: memset %.2f s, large %.2f s, segment %.2f s; host merge %.2f s (%.0f chunks, %u large primes active)\n",
               prof_memset, prof_large, prof_segment, prof_host, (double)st.numbers / CHUNK_NUMS,
               active_count(G, N + span));
    printf("       time for a scan [0, X) at this rate:");
    static const double X[] = { 1e15, 2e15, 5e15, 1e16 };
    char b[32];
    for (size_t i = 0; i < sizeof X / sizeof *X; i++) printf("  %.0e: %s", X[i], fmt_dur(X[i] / rate, b, sizeof b));
    putchar('\n');
    print_hist();
    return 0;
}

/*
 * cmd_selftest() -- compare the GPU scan with exact figures from the CPU
 * tool and check the resume logic.
 * Inputs:  optional -Q QSPLIT (default 4e6).
 * Checks:  number parsing; Miller-Rabin spot checks; the run length of every
 *          known term by next-prime search; (a) [0, 1e11): pi(10^11),
 *          a(1..12) with pi(a(n)) = A133697(n-2), no a(13), and the exact
 *          histogram; (b) [12345000000, 13345000000), bounds inside chunks
 *          and segments: prime count, the first prime >= START of each run
 *          length with its index, and the histogram; (c) [1e15, 1e15+2e12):
 *          prime count and histogram (the CPU tool's bench window); (d) a
 *          scan to 12345678901 with a checkpoint, resumed to 23456789012,
 *          must equal a straight scan to 23456789012 in prime count,
 *          histogram, first primes and indices.  All reference figures come
 *          from a158939.c.
 * Returns: 0 if every check passed, 1 otherwise.
 * Errors:  die() as for scan; usage() on unknown options.
 */
static int cmd_selftest(int argc, char **argv)
{
    u32 qsplit = 4000000;
    int fails = 0;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "-Q") && i + 1 < argc) qsplit = parse_qsplit(argv[++i]);
        else usage();
    }
/* CHECK(cond, fmt, ...): print "ok    <description>" or "FAIL  <description>"
 * and count the failures. */
#define CHECK(cond, ...) do { if (cond) printf("ok    " __VA_ARGS__); else { printf("FAIL  " __VA_ARGS__); fails++; } putchar('\n'); fflush(stdout); } while (0)
    CHECK(parse_num("2^54-32") == 18014398509481952ULL && parse_num("1e12") == 1000000000000ULL &&
          parse_num("1.29e14") == 129000000000000ULL && parse_num("10^15+7") == 1000000000000007ULL, "number parsing");
    CHECK(is_prime64(2) && !is_prime64(1) && !is_prime64(9) && is_prime64(1000000007ULL) && !is_prime64(3215031751ULL) &&
          is_prime64(18446744073709551557ULL) && is_prime64(129001208165717ULL), "Miller-Rabin spot checks");
    {
        bool ok = true;
        for (int n = 1; n <= NKNOWN; n++) { u16 g[NMAX + 1]; if (run_length64(KNOWN[n], g, NMAX + 1) != n) ok = false; }
        CHECK(ok, "run lengths of a(1..%d) by next-prime search", NKNOWN);
    }
    g_report = NMAX;
    char b[32];

    /* (a) [0, 1e11): exact values from the CPU tool (a158939.c) */
    {
        ScanStats st = run_scan(0, 100000000000ULL, qsplit, NULL, 60, 0, true);
        printf("      [0, 1e11): %s, %.3g numbers/s\n", fmt_dur(st.seconds, b, sizeof b), st.numbers / st.seconds);
        CHECK(S.primes == 4118054813ULL, "pi(10^11) = %" PRIu64 " (expected 4118054813)", S.primes);
        bool ok = true;
        for (int n = 1; n <= 12; n++) {
            if (S.best[n].p != KNOWN[n]) { ok = false; printf("      a(%d): GPU %" PRIu64 " expected %" PRIu64 "\n", n, S.best[n].p, KNOWN[n]); }
            if (n >= 2 && S.best[n].pi != A133697[n - 2]) { ok = false; printf("      pi(a(%d)): GPU %" PRIu64 " expected %" PRIu64 "\n", n, S.best[n].pi, A133697[n - 2]); }
        }
        CHECK(ok && S.best[13].p == 0, "a(1..12) and pi(a(n)) = A133697 below 10^11");
        static const u64 H[13] = { 0, 2103848608ULL, 1377895718ULL, 489399710ULL, 120501561ULL, 22556442ULL, 3381243ULL,
                                   421291ULL, 45508ULL, 4337ULL, 374ULL, 20ULL, 1ULL };
        ok = true;
        for (int n = 1; n <= 12; n++) if (S.hist[n] != H[n]) { ok = false; printf("      hist[%d]: GPU %" PRIu64 " expected %" PRIu64 "\n", n, S.hist[n], H[n]); }
        CHECK(ok && S.hist[13] == 0, "run-length histogram of [0, 1e11) matches the CPU tool exactly");
    }
    /* (b) odd bounds inside chunks: [12345000000, 13345000000) */
    {
        run_scan(12345000000ULL, 13345000000ULL, qsplit, NULL, 60, 0, true);
        CHECK(S.primes == 42962974ULL, "primes in [12345000000, 13345000000) = %" PRIu64 " (expected 42962974)", S.primes);
        static const u64 F[11] = { 0, 12345000011ULL, 12345000061ULL, 12345000403ULL, 12345000713ULL, 12345000709ULL,
                                   12345006277ULL, 12345463129ULL, 12345463127ULL, 12358263017ULL, 12525319013ULL };
        static const u64 FI[11] = { 0, 1, 2, 19, 35, 34, 278, 20093, 20092, 570629, 7758320 };
        static const u64 H[11] = { 0, 21967314ULL, 14376273ULL, 5095117ULL, 1250868ULL, 233575ULL, 34929ULL, 4355ULL, 494ULL, 48ULL, 1ULL };
        bool ok = true, okh = true;
        for (int n = 1; n <= 10; n++) {
            if (S.best[n].p != F[n] || S.best[n].pi != FI[n]) { ok = false; printf("      n=%d: GPU %" PRIu64 " (pi %" PRIu64 ") expected %" PRIu64 " (pi %" PRIu64 ")\n", n, S.best[n].p, S.best[n].pi, F[n], FI[n]); }
            if (S.hist[n] != H[n]) { okh = false; printf("      hist[%d]: GPU %" PRIu64 " expected %" PRIu64 "\n", n, S.hist[n], H[n]); }
        }
        CHECK(ok && S.best[11].p == 0, "first primes >= START and their indices match the CPU tool");
        CHECK(okh && S.hist[11] == 0, "run-length histogram matches the CPU tool exactly");
    }
    /* (c) [1e15, 1e15 + 2e12): the CPU tool's bench window */
    {
        ScanStats st = run_scan(1000000000000000ULL, 1002000000000000ULL, qsplit, NULL, 60, 0, true);
        printf("      [1e15, 1e15+2e12): %s, %.3g numbers/s\n", fmt_dur(st.seconds, b, sizeof b), st.numbers / st.seconds);
        CHECK(S.primes == 57904107084ULL, "primes in [1e15, 1e15+2e12) = %" PRIu64 " (expected 57904107084)", S.primes);
        static const u64 H[14] = { 0, 29414157759ULL, 19357439615ULL, 6979639719ULL, 1754686141ULL, 337953666ULL, 52549953ULL,
                                   6834766ULL, 764019ULL, 74339ULL, 6523ULL, 539ULL, 41ULL, 4ULL };
        bool ok = true;
        for (int n = 1; n <= 13; n++) if (S.hist[n] != H[n]) { ok = false; printf("      hist[%d]: GPU %" PRIu64 " expected %" PRIu64 "\n", n, S.hist[n], H[n]); }
        CHECK(ok && S.hist[14] == 0, "run-length histogram of [1e15, 1e15+2e12) matches the CPU tool exactly");
    }
    /* (d) a run that ends inside a chunk, then resumed with a larger END, must equal a straight run */
    {
        const char *st = "selftest_resume.state";
        remove(st);
        run_scan(0, 12345678901ULL, qsplit, st, 60, 0, true);
        run_scan(0, 23456789012ULL, qsplit, st, 60, 0, true);
        u64 primes_r = S.primes, hist_r[NMAX];
        Cand best_r[NMAX];
        memcpy(hist_r, S.hist, sizeof hist_r);
        memcpy(best_r, S.best, sizeof best_r);
        run_scan(0, 23456789012ULL, qsplit, NULL, 60, 0, true);
        bool ok = primes_r == S.primes;
        for (int n = 1; n < NMAX; n++)
            if (hist_r[n] != S.hist[n] || best_r[n].p != S.best[n].p || best_r[n].pi != S.best[n].pi) ok = false;
        remove(st);
        CHECK(ok, "resume across END=12345678901 (inside a chunk) to 23456789012 equals a straight run (%" PRIu64 " primes)", S.primes);
    }
#undef CHECK
    printf(fails ? "SELFTEST FAILED (%d)\n" : "selftest passed\n", fails);
    return fails ? 1 : 0;
}

/*
 * main() -- command dispatch: scan | bench | selftest.  Records whether
 * stderr is a terminal and makes stdout line-buffered so result lines show
 * up immediately through a pipe.  Returns the command's status; usage()
 * exits 2.
 */
int main(int argc, char **argv)
{
    stderr_tty = isatty(2);
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 2) usage();
    if (!strcmp(argv[1], "scan")) return cmd_scan(argc - 2, argv + 2);
    if (!strcmp(argv[1], "bench")) return cmd_bench(argc - 2, argv + 2);
    if (!strcmp(argv[1], "selftest")) return cmd_selftest(argc - 2, argv + 2);
    usage();
    return 2;
}
