/*
 * a359636_cuda.cu  --  GPU (CUDA) version of the exhaustive scan of a359636.c
 *
 * Same problem, same reduction, same output format as a359636.c; read that
 * file's THEORY OF OPERATION first (its sections 1-6 and 9 apply here
 * unchanged).  This program replaces the CPU sieve by a CUDA kernel and keeps
 * the exact verification on the CPU.  On a DGX Spark (GB10) it found
 * a(9) = 31610535900218923 in Sep 2026 (about 11 hours from m = 0) and ran
 * the level-10 scan behind a(10) > 408697968217030657.
 *
 * ==========================================================================
 * THEORY OF OPERATION (GPU specifics)
 * ==========================================================================
 *
 * G1. Work decomposition.  The k range [k_lo, k_hi) (m = 6k + 3) is processed
 *     in LAUNCHES of lsize = blocks * SEGK consecutive k (default
 *     65536 * 8192 = 2^29; the production runs used -b 262144, i.e. 2^31 k).
 *     One kernel launch sieves one launch range: thread block b handles the
 *     SEGK k starting at kbase + b * SEGK, with their packed counters (one
 *     32-bit word per k, bytes 0..2 for m-1, m, m+1, exactly as in the CPU
 *     tool) in shared memory.
 *
 * G2. Inside a block (sieve_kernel()):
 *       1. pattern init: W[j] = patA[(k0+j) mod 5005] + patB[(k0+j) mod 7429]
 *          applies the primes 5..23; consecutive threads take consecutive j,
 *          so the table reads are coalesced;
 *       2. primes 29..PSMALL (127): one warp per progression; lane L marks
 *          j + L*p, j + (L+32)*p, ...; p is odd, so the 32 lanes hit 32
 *          different shared-memory banks;
 *       3. primes above PSMALL: one thread per progression.  The progressions
 *          are sorted by p (so by decreasing work) and dealt out in "snake"
 *          order -- left to right, then right to left -- so that every thread
 *          gets a similar total;
 *       4. the threshold test of a359636.c (THEORY section 4) on every word;
 *          survivors are appended to a global list through an atomic counter.
 *     Marks use shared-memory atomicAdd because progressions handled by
 *     different threads can hit the same word.  The start of a progression
 *     needs k0 mod p, computed in 32 bits as (kbase mod p + b * (SEGK mod p))
 *     mod p with kbase mod p supplied by the host; hence run_scan() requires
 *     blocks * T < 2^32.  With the default T the GPU demands six small factors
 *     per target (T = 2000 at level 9, 1000 at level 10) instead of the CPU
 *     tool's five, which makes survivors about ten times rarer for about 13%
 *     more marks.
 *
 * G3. Host pipeline (run_scan()).  The main thread drives the GPU: for each
 *     launch it uploads kbase mod p for every progression, launches the
 *     kernel, downloads the survivor count and list (the copy synchronizes)
 *     and queues them as a Batch.  The -t verifier threads (default 8) take
 *     batches from the queue, verify every survivor exactly (check_survivor(),
 *     the same host code as the CPU tool) and report the batch to
 *     launch_done(), which advances the frontier -- counted in launches --
 *     exactly like chunk_done() in a359636.c.  At most 64 batches may wait,
 *     which bounds host memory and keeps the frontier well inside the RING.
 *     The frontier drives the status line, the checkpoint and the early stop
 *     as in the CPU tool.  Ctrl-C: no new launches; the queued batches are
 *     still verified, then the checkpoint and the summary are written.
 *
 * G4. Checkpoint ("A359636 cuda state v1", format in save_state()).  Unlike
 *     the CPU tool, a resume accepts a different END (k_hi; that is how a
 *     range was split between two machines) and a different -b (the frontier
 *     is converted to the new launch size, rounding down).  N and START (k_lo)
 *     must match.
 *
 * G5. Result line.  As in a359636.c (THEORY section 9): a scan with START > 9
 *     is "partial" and reports only on its own m range.
 *
 * Differences from a359636.c
 *   - Commands: scan, verify and selftest only (no hunt, no omega).  The
 *     selftest reproduces a(1..7) (-8: a(8)) but has no omega_at_least()
 *     spot checks.
 *   - No primesieve dependency: build_primes() is a plain sieve of
 *     Eratosthenes.
 *   - -t sets the number of CPU verifier threads (default 8); -b sets the
 *     number of blocks per launch.
 *
 * Known issues (fixed in a359636.c on Oct 12 2026, not yet here: this file
 * was changed only in its comments while its paused level-10 runs depend on
 * it)
 *   - parse_num() has no range checks: an over-large decimal becomes 2^64-1,
 *     X+Y and X-Y can wrap around, X-Y-Z is evaluated as X-(Y-Z), an empty
 *     argument is read past its end, and 1^E or 0^E loops E times.
 *   - After a complete scan from 0, "a(N) > END" is not quite proven when
 *     END = 5 or 0 (mod 6): the prime END resp. END-1 has its m just beyond
 *     the scanned range (a359636.c, THEORY section 2).  The bound
 *     covered_m - 2 printed by a scan that stopped early is always proven.
 *     The paused level-10 runs are not affected (atom1's END,
 *     664309293225476095, is 1 (mod 6), and atom2's range is partial).
 *   - save_state() does not check for write errors, so e.g. a full disk can
 *     replace a good checkpoint with an empty one.
 *   - run_scan() allocates an unused 128 MB host buffer (hostout).
 *   - After a launch-size or END change, load_state()'s "resuming from" line
 *     shows the position and percentage computed with the old values
 *     (cosmetic).
 *   - Benign data races as in a359636.c: the main loop reads S->stop_now
 *     without fr_mu, and launch_done() reads best_p/best_m without sol_mu.
 *
 * Usage
 *   a359636_cuda scan N [START] END [-t CPUTHREADS] [-T SIEVEMAX] [-b BLOCKS] [-S STATE] [-i SECS] [-a] [-q]
 *   a359636_cuda selftest [-t CPUTHREADS] [-8] [-q]
 *   a359636_cuda verify N P
 *
 * Build (DGX Spark / GB10, CUDA 13; no dependencies beyond CUDA):
 *   nvcc -O3 -std=c++17 -arch=sm_121 -Xcompiler -pthread a359636_cuda.cu -o a359636_cuda
 *   (cuda/Makefile does this; nvcc lives in /usr/local/cuda/bin, which is not on
 *   the PATH of a non-interactive ssh shell.)
 * Measured on the GB10: 810 Gm/s at level 9 (T = 2000), 904 Gm/s at level 10 (T = 1000),
 * versus 82 Gm/s for the CPU tool on the Spark's 20 Arm cores.
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
#include <atomic>
#include <vector>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <unistd.h>
#include <cuda_runtime.h>

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint8_t  u8;
typedef unsigned __int128 u128;           /* GCC/Clang extension (host code only) */

/*
 * Kernel geometry and limits.
 *   SEGK      k per thread block (= per shared-memory segment).  8192 (32 KB of counters)
 *             gave the best occupancy on the GB10; 16384 was 2x slower, 4096 ~20% slower.
 *   BLOCKDIM  threads per block; 256 was fastest with SEGK = 8192 (512: -5%, 128: -33%).
 *   PSMALL    progressions of primes <= PSMALL are marked by a whole warp each.
 *   PAT_A/B   pattern periods for the primes 5..13 and 17..23.
 *   RING      size of the launch-completion ring (far more than the 64 queued launches).
 *   MAXSOL    solutions remembered for de-duplication and the checkpoint.
 *   KNOWN     a(1..8) from the OEIS for 'selftest' (index 0 unused).
 */
#ifndef SEGK
#define SEGK     8192           /* k per segment (block): 32 KB of shared u32 (16384 is 2x slower: occupancy) */
#endif
#ifndef BLOCKDIM
#define BLOCKDIM 256
#endif
#define PSMALL   127            /* primes <= PSMALL: one warp per progression */
#define PAT_A    5005           /* 5*7*11*13 */
#define PAT_B    7429           /* 17*19*23  */
#define RING     (1 << 16)
#define MAXSOL   4096
#define NKNOWN   8

static const u64 KNOWN[NKNOWN + 1] = { 0, 7, 19, 643, 51427, 8083633, 1077940147ULL,
                                       75582271489ULL, 34710483181813ULL };

/* Process-wide state; same meaning as in a359636.c (g_quiet = -q, stderr_tty = stderr is a
   terminal, g_stop = set by SIGINT/SIGTERM, out_mu = serializes stdout blocks). */
static bool g_quiet = false;
static bool stderr_tty;
static volatile sig_atomic_t g_stop = 0;
static std::mutex out_mu;

/* CUDA_CHECK: evaluate a CUDA runtime call and die() with the call text, file, line and CUDA's
   error string unless it returned cudaSuccess. */
#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) die("CUDA error %s at %s:%d: %s", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); } while (0)

/* ------------------------------------------------------------------ */
/* Generic helpers                                                     */
/* ------------------------------------------------------------------ */

/*
 * now: monotonic seconds (CLOCK_MONOTONIC) for rates and checkpoint intervals.
 *   In: nothing.   Out: seconds as a double.   Errors: none.
 */
static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

/*
 * die: print "a359636_cuda: <message>" on stderr and exit with status 1.
 *   In: printf-style format and arguments.   Out: does not return.
 *   Errors: none (device memory is released by the process exit).
 */
static void die(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fputs("a359636_cuda: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    exit(1);
}

/*
 * xmalloc: malloc that never returns NULL.
 *   In: n bytes (0 is treated as 1).   Out: uninitialized memory for free().
 *   Errors: out of memory -> die().
 */
static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

/*
 * parse_num: read a command-line number: decimal, X^E, XeE, and sums/differences of those
 *   (same syntax as in a359636.c).
 *   In:     s.   Out: the value.
 *   Errors: die() for text that is not a number and for X^E / XeE above 2^64.  NOT checked
 *           here (fixed in a359636.c): decimal overflow (becomes 2^64-1), wrap-around of
 *           X+Y and of X-Y below zero, X-Y-Z evaluated as X-(Y-Z), an empty s (read past
 *           its end), and 1^E or 0^E with a huge E (loops E times).
 */
static u64 parse_num(const char *s)
{
    const char *plus = strchr(s + 1, '+'), *minus = strchr(s + 1, '-');
    if (plus || minus) {
        const char *op = plus ? plus : minus;
        char *a = strndup(s, (size_t)(op - s));
        u64 x = parse_num(a), y = parse_num(op + 1);
        free(a);
        return plus ? x + y : x - y;
    }
    char *end;
    errno = 0;
    unsigned long long base = strtoull(s, &end, 10);
    if (end == s) die("bad number '%s'", s);
    if (*end == '\0') return base;
    if (*end == '^' || *end == 'e' || *end == 'E') {
        bool pw = (*end == '^');
        unsigned long long e = strtoull(end + 1, &end, 10);
        if (*end != '\0') die("bad number '%s'", s);
        u128 r = 1, b = pw ? base : 10;
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
 * fmt_commas: v with thousands separators.
 *   In: out = buffer of at least 27 bytes (callers use char[32]); v.
 *   Out: NUL-terminated text in out.   Errors: none (buffer size not checked).
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
 * fmt_time: duration as "3d04h05m", "1h02m03s", "4m05s" or "12s"; "?" for negative or
 *   absurd (> 1e9 s) values, i.e. an unknown ETA.
 *   In: out = buffer (callers use char[32]); s = seconds.   Out: text in out.   Errors: none.
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
 * on_sigint: SIGINT/SIGTERM handler; sets g_stop (async-signal-safe).  run_scan() then stops
 *   launching, verifies the queued batches, writes the checkpoint and the summary.
 *   In: the signal number (unused).   Out: none.   Errors: none.
 */
static void on_sigint(int) { g_stop = 1; }

/* ------------------------------------------------------------------ */
/* Arithmetic (host)                                                   */
/* ------------------------------------------------------------------ */

/*
 * mulmod: (a*b) mod m via the exact 128-bit product.
 *   In: a, b, m >= 1.   Out: the residue.   Errors: m = 0 divides by zero (never passed).
 */
static inline u64 mulmod(u64 a, u64 b, u64 m) { return (u64)((u128)a * b % m); }

/*
 * powmod: a^e mod m by binary exponentiation.
 *   In: a, e, m >= 2.   Out: a^e mod m.   Errors: none for m >= 2 (callers pass primes
 *   or odd n >= 41*41).
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
 * is_prime: deterministic primality test for any 64-bit n (trial division by 2..37, then
 *   strong Miller-Rabin to the bases 2..37, deterministic below 3.18e23).
 *   In: n.   Out: true iff n is prime.   Errors: none.
 */
static bool is_prime(u64 n)
{
    static const u64 bases[] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 };
    if (n < 2) return false;
    for (int i = 0; i < 12; i++) if (n % bases[i] == 0) return n == bases[i];
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
 * iroot: floor(n^(1/r)), a floating-point estimate corrected with exact 128-bit powers.
 *   In: n, r >= 1.   Out: the largest x with x^r <= n.
 *   Errors: r <= 0 is invalid (never passed).
 */
static u64 iroot(u64 n, int r)
{
    if (r == 1) return n;
    if (n < 2) return n;
    u64 x = (u64)pow((double)n, 1.0 / r);
    if (x == 0) x = 1;
    for (;;) {
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
 * is_perfect_power: is n = a^j for some a >= 2, j >= 2 (j = 2..63 tried)?
 *   In: n.   Out: true or false (false for n < 4).   Errors: none.
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

/* The prime table: all primes <= g_plimit, ascending (trial division, compute_c(), the
   progressions).  Built by build_primes(); read-only while threads run. */
static u32 *g_primes;
static size_t g_nprimes;
static u64 g_plimit;

/*
 * build_primes: (re)build the prime table with a plain sieve of Eratosthenes.
 *   In:     limit = largest number to include (a scan needs a few million at most).
 *   Out:    g_primes, g_nprimes, g_plimit describe all primes <= limit; the old table is freed.
 *   Errors: limit > 2^31 or out of memory -> die() (an allocation failure inside
 *           std::vector throws std::bad_alloc instead, which also ends the program).
 */
static void build_primes(u64 limit)          /* plain Eratosthenes; limit is a few million at most */
{
    if (limit > (1ull << 31)) die("prime table limit too large");
    std::vector<u8> comp(limit + 1, 0);
    std::vector<u32> pr;
    for (u64 i = 2; i <= limit; i++) {
        if (comp[i]) continue;
        pr.push_back((u32)i);
        for (u64 j = i * i; j <= limit; j += i) comp[j] = 1;
    }
    free(g_primes);
    g_primes = (u32 *)xmalloc(pr.size() * sizeof(u32));
    memcpy(g_primes, pr.data(), pr.size() * sizeof(u32));
    g_nprimes = pr.size();
    g_plimit = limit;
}

/*
 * omega_at_least: does N have at least n distinct prime factors?  Identical to the
 *   function in a359636.c, which documents the method (trial division with the size bound
 *   cofactor >= p^r, Miller-Rabin and a perfect-power test for the last two primes).
 *   In:     N, n; the prime table must reach cbrt(N) for an exact answer (run_scan()
 *           builds it to cbrt(Bmax) + 1000 or more).
 *   Out:    true iff omega(N) >= n (n <= 0: true; N < 2: false).
 *   Errors: none.
 */
static bool omega_at_least(u64 N, int n)
{
    if (n <= 0) return true;
    if (N < 2) return false;
    int cnt = 0;
    u64 lim = iroot(N, n);
    for (size_t i = 0; i < g_nprimes; i++) {
        u64 p = g_primes[i];
        int r = n - cnt;
        if (r == 1) return N > 1;
        if (p > lim) return false;
        if (r == 2 && p * p * p > N) {
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
    if (N == 1) return cnt >= n;
    if (is_prime(N)) return cnt + 1 >= n;
    return cnt + (is_perfect_power(N) ? 1 : 2) >= n;
}

/*
 * rho: a nontrivial factor of a composite n (Pollard rho, Floyd cycle, c = 1, 2, ...).
 *   In: n composite.   Out: d with 1 < d < n (2 for even n).
 *   Errors: never returns for prime n (callers test is_prime() first).
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
            u64 a = diff, b = n;
            while (b) { u64 t = a % b; a = b; b = t; }
            d = a;
        }
        if (d != n) return d;
    }
}

/*
 * factor_full: complete factorization (trial division to sqrt(n), then Pollard rho).
 *   In: n; pr, ex = arrays with room for at least 15 entries (callers use 64).
 *   Out: k = omega(n); pr[0..k-1] ascending distinct primes, ex[] exponents (k = 0 for
 *        n < 2).
 *   Errors: none for 64-bit n.
 */
static int factor_full(u64 n, u64 *pr, int *ex)
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
    for (int i = 1; i < k; i++)
        for (int j = i; j > 0 && pr[j - 1] > pr[j]; j--) {
            u64 tp = pr[j]; pr[j] = pr[j - 1]; pr[j - 1] = tp;
            int te = ex[j]; ex[j] = ex[j - 1]; ex[j - 1] = te;
        }
    return k;
}

/*
 * fmt_factorization: text like "2^3*13*41" for n (n < 2 printed as itself).
 *   In: out = buffer (callers use char[256], ~100 bytes are needed at most); n.
 *   Out: text in out.   Errors: none.
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
 * omega_exact: number of distinct prime factors by complete factorization (for printing).
 *   In: n.   Out: omega(n), 0 for n < 2.   Errors: none.
 */
static int omega_exact(u64 n) { u64 pr[64]; int ex[64]; return factor_full(n, pr, ex); }

/*
 * prev_prime / next_prime: the largest prime below x / the smallest prime above x.
 *   In: prev_prime: x > 3;  next_prime: 2 <= x < 2^64 - 59.
 *   Out: the prime.
 *   Errors: prev_prime(x <= 3) and next_prime(x >= 2^64 - 59) wrap around; not reachable
 *           from scan ranges (callers pass m - 1 >= 8 and primes far below 2^64).
 */
static u64 prev_prime(u64 x) { x--; if (!(x & 1)) x--; while (!is_prime(x)) x -= 2; return x; }
static u64 next_prime(u64 x) { x++; if (!(x & 1)) x++; while (!is_prime(x)) x += 2; return x; }

/*
 * gap_qualifies: does the gap that starts at the prime p qualify at level n?
 *   In: p prime (not checked); n; q_out.
 *   Out: *q_out = nextprime(p); true iff q - p >= 4 and omega(x) >= n for all p < x < q.
 *   Errors: none.  Needs the prime table to reach cbrt(q).
 */
static bool gap_qualifies(u64 p, int n, u64 *q_out)
{
    u64 q = next_prime(p);
    *q_out = q;
    if (q - p < 4) return false;
    for (u64 x = p + 1; x < q; x++) if (!omega_at_least(x, n)) return false;
    return true;
}

/*
 * inv_mod: a^-1 mod a prime p, by Fermat (a^(p-2)).
 *   In: p prime, p not dividing a.   Out: the inverse.
 *   Errors: meaningless otherwise; only used with a = 6, p >= 5.
 */
static u64 inv_mod(u64 a, u64 p) { return powmod(a % p, p - 2, p); }

/*
 * target_residue: r in [0, p) with  p | 6k + 3 + t  <=>  k = r (mod p),  r = -(3+t) * 6^-1.
 *   In: p prime >= 5; t = -1, 0, +1 for the target m-1, m, m+1.   Out: r.
 *   Errors: p = 2, 3 give nonsense (never passed).
 */
static u32 target_residue(u64 p, int t)          /* k with p | 6k+3+t */
{
    u64 inv6 = inv_mod(6, p);
    u64 c = (u64)(3 + t) % p;
    return (u32)(((p - c) % p) * inv6 % p);
}

/*
 * compute_c: number of distinct prime factors <= T that every candidate must have
 *   (a359636.c, THEORY section 5).
 *   In:     n = level; T = sieve bound; Bmax = largest number examined.
 *   Out:    c = n - rmax, rmax = the largest r with (first n-r primes) * (first r primes
 *           above T) <= Bmax; n when even the product of the first n primes exceeds Bmax.
 *   Errors: die() if the prime table holds too few primes above T (a -T beyond the table).
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

/* ------------------------------------------------------------------ */
/* GPU sieve                                                           */
/* ------------------------------------------------------------------ */

/* One strided progression: every k with k = r (mod p) gets the packed increment inc
   (1 << 8t for the target t = 0, 1, 2, i.e. m-1, m, m+1). */
struct Prog { u32 p, r, inc; };            /* prime, residue of k, packed increment */

/*
 * sieve_kernel: sieve one launch range and collect its survivors (THEORY G1, G2).
 *   Grid:   nb blocks of BLOCKDIM threads with SEGK*4 bytes of dynamic shared memory; block b
 *           covers k in [kbase + b*SEGK, kbase + (b+1)*SEGK), clipped at klimit.
 *   In:     kbase, klimit = the launch range [kbase, klimit); prog[0..nprog-1] = the strided
 *           progressions, ascending by p, the first nsmall with p <= PSMALL; kb[q] = kbase
 *           mod prog[q].p; patA, patB = pattern tables for 5..13 and 17..23; thradd,
 *           thrmask = threshold-test constants (a359636.c, THEORY section 4); out, outcount,
 *           outcap = survivor list (k values), its counter (zeroed by the host) and capacity.
 *   Out:    every k of the range whose three counters reach the threshold is appended to
 *           out[] (in no particular order); *outcount = number of survivors, which can
 *           exceed outcap -- then only the first outcap were stored.
 *   Errors: none detected on the device.  The host must zero *outcount before the launch,
 *           check *outcount <= outcap afterwards (run_scan() dies otherwise) and keep
 *           blocks * T < 2^32 so that the 32-bit k0 mod p arithmetic cannot overflow.
 *           A block wholly beyond klimit returns at once; the test is the same for all
 *           threads of the block, so no thread skips a __syncthreads().
 */
__global__ void __launch_bounds__(BLOCKDIM)
sieve_kernel(u64 kbase, u64 klimit, const Prog *__restrict__ prog, const u32 *__restrict__ kb,
             int nprog, int nsmall, const u32 *__restrict__ patA, const u32 *__restrict__ patB,
             u32 thradd, u32 thrmask, u64 *__restrict__ out, unsigned int *__restrict__ outcount, unsigned int outcap)
{
    extern __shared__ u32 W[];
    const int tid = threadIdx.x;
    const u64 k0 = kbase + (u64)blockIdx.x * SEGK;
    if (k0 >= klimit) return;

    /* pattern init for the primes 5..23: W[j] = patA[(k0+j) % 5005] + patB[(k0+j) % 7429]
       (coalesced: consecutive threads take consecutive positions) */
#ifdef NOPAT
    for (int j = tid; j < SEGK; j += BLOCKDIM) W[j] = 0;
#else
    {
        /* running table indices: thread tid takes positions tid, tid+BLOCKDIM, ... */
        u32 a = (u32)(k0 % PAT_A) + tid, b = (u32)(k0 % PAT_B) + tid;
        if (a >= PAT_A) a -= PAT_A;
        if (b >= PAT_B) b -= PAT_B;
        for (int j = tid; j < SEGK; j += BLOCKDIM) {
            W[j] = __ldg(patA + a) + __ldg(patB + b);
            a += BLOCKDIM; if (a >= PAT_A) a -= PAT_A;
            b += BLOCKDIM; if (b >= PAT_B) b -= PAT_B;
        }
    }
#endif
    __syncthreads();

    /* MARK adds a packed increment to counter word j.  NOATOM, NOMARK and NOPAT are
       benchmark switches only: without atomics colliding marks are lost (the survivor
       counts change), NOMARK/NOPAT skip the marking / pattern phases. */
#ifdef NOATOM
#define MARK(j, inc) (W[j] += (inc))
#else
#define MARK(j, inc) atomicAdd(&W[j], (inc))
#endif
#ifndef NOMARK
    /* phase A: small primes, one warp per progression (lanes hit distinct banks: p odd) */
    {
        const int warp = tid >> 5, lane = tid & 31, nwarps = BLOCKDIM >> 5;
        for (int q = warp; q < nsmall; q += nwarps) {
            const u32 p = prog[q].p, inc = prog[q].inc, r = prog[q].r;
            const u32 k0m = (kb[q] + (u32)blockIdx.x * (SEGK % p)) % p;      /* < 2^32: blocks * T < 2^32 */
            u32 j = r >= k0m ? r - k0m : r + p - k0m;
            for (j += lane * p; j < SEGK; j += 32 * p) MARK(j, inc);
        }
    }
    /* phase B: larger primes, one thread per progression, snake order for balance */
    {
        const int nlarge = nprog - nsmall;
        for (int s = 0; s * BLOCKDIM < nlarge; s++) {
            const int idx = s * BLOCKDIM + ((s & 1) ? (BLOCKDIM - 1 - tid) : tid);
            if (idx >= nlarge) continue;
            const int q = nsmall + idx;
            const u32 p = prog[q].p, inc = prog[q].inc, r = prog[q].r;
            const u32 k0m = (kb[q] + (u32)blockIdx.x * (SEGK % p)) % p;
            u32 j = r >= k0m ? r - k0m : r + p - k0m;
            for (; j < SEGK; j += p) MARK(j, inc);
        }
    }
#endif
    __syncthreads();

    /* scan: all three counters >= thr  <=>  (~(W + thradd)) & 0x808080 == 0 */
    for (int j = tid; j < SEGK; j += BLOCKDIM) {
        if ((((~(W[j] + thradd)) & thrmask)) == 0) {
            u64 k = k0 + j;
            if (k < klimit) {
                unsigned int idx = atomicAdd(outcount, 1u);
                if (idx < outcap) out[idx] = k;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Scan driver                                                         */
/* ------------------------------------------------------------------ */

/* A qualifying gap: consecutive primes p < q and the m of the triple that led to it. */
struct Solution { u64 p, q, m; };

/* One launch's survivors (k values), handed from the GPU driver to the verifier threads;
   idx = launch number, reported to launch_done() once every survivor has been checked. */
struct Batch { u64 idx; std::vector<u64> ks; };

/*
 * Scan: the state of one 'scan' (one run_scan() call).  Locking: fr_mu protects frontier and
 * done[] and is held when stop_now is written; sol_mu protects sol, best_p, best_m; q_mu
 * protects queue, q_done, q_pending_k and is the mutex of q_cv; the statistics are atomics;
 * everything else is set before the verifier threads start.
 */
struct Scan {
    int n;                      /* level */
    u64 k_lo, k_hi;             /* k range [k_lo, k_hi), m = 6k+3 */
    u64 lsize;                  /* k per launch */
    u64 nlaunch;                /* launches in [k_lo, k_hi); the last one may be short */
    u32 T; int c, thr;          /* sieve bound, required small factors, counter threshold c-1 */
    std::vector<Prog> prog; int nsmall;   /* by ascending p; first nsmall: p <= PSMALL */
    /* device */
    Prog *d_prog; u32 *d_kb; u32 *d_patA, *d_patB; u64 *d_out; unsigned int *d_count;
    unsigned int outcap;        /* capacity of the survivor list d_out (2^24) */
    /* stats */
    std::atomic<unsigned long long> survivors{0}, triples{0}, nsol{0};
    std::vector<Solution> sol; std::mutex sol_mu;
    u64 best_p = 0, best_m = 0; /* smallest solution so far and its m (0 = none yet) */
    /* verification queue */
    std::deque<Batch> queue; std::mutex q_mu; std::condition_variable q_cv; bool q_done = false;
    size_t q_pending_k = 0;     /* survivors waiting in the queue (bookkeeping only) */
    /* ordered completion of launches */
    u64 frontier = 0; std::vector<u8> done; std::mutex fr_mu;   /* done: ring of RING flags */
    bool scan_all = false, stop_now = false;   /* -a; early stop requested by launch_done() */
    double t0 = 0;              /* start time (now()) */
    const char *state_file = nullptr;          /* -S FILE or nullptr */
    u64 resume_frontier = 0;    /* frontier read from the checkpoint (for the rate) */
};

/*
 * record_solution: remember and print a qualifying gap (as in a359636.c).
 *   In:     S; p < q consecutive primes whose gap qualifies; m = the triple's m.
 *   Out:    unless p is known already: stores it (up to MAXSOL), counts it, updates
 *           best_p/best_m, prints the SOLUTION block with factorizations.
 *   Errors: none.  Lock order sol_mu, then out_mu.
 */
static void record_solution(Scan *S, u64 p, u64 q, u64 m)
{
    std::lock_guard<std::mutex> lk(S->sol_mu);
    for (auto &s : S->sol) if (s.p == p) return;
    if (S->sol.size() < MAXSOL) S->sol.push_back({p, q, m});
    S->nsol++;
    if (!S->best_p || p < S->best_p) { S->best_p = p; S->best_m = m; }
    std::lock_guard<std::mutex> lo(out_mu);
    if (stderr_tty && !g_quiet) fputs("\r\033[K", stderr);
    printf("SOLUTION n=%d p=%" PRIu64 " q=%" PRIu64 " gap=%" PRIu64 " (m=%" PRIu64 ")\n", S->n, p, q, q - p, m);
    char buf[256];
    for (u64 x = p + 1; x < q; x++) {
        fmt_factorization(buf, x);
        printf("    %" PRIu64 " = %s  omega=%d\n", x, buf, omega_exact(x));
    }
    fflush(stdout);
}

/*
 * check_survivor: exact test of one survivor k (m = 6k + 3), as in a359636.c: if
 *   omega(m-1), omega(m), omega(m+1) >= n, count and print the triple and record the gap at
 *   prevprime(m-1) if it qualifies.
 *   In: S, k.   Out: none.   Errors: none; runs concurrently in the verifier threads.
 */
static void check_survivor(Scan *S, u64 k)
{
    u64 m = 6 * k + 3;
    int n = S->n;
    if (!omega_at_least(m - 1, n)) return;
    if (!omega_at_least(m, n)) return;
    if (!omega_at_least(m + 1, n)) return;
    S->triples++;
    u64 p = prev_prime(m - 1), q;
    {
        std::lock_guard<std::mutex> lo(out_mu);
        if (stderr_tty && !g_quiet) fputs("\r\033[K", stderr);
        printf("triple   n=%d m=%" PRIu64 "  omega(m-1,m,m+1) = %d %d %d  prevprime=%" PRIu64 " (%+" PRId64 ")\n",
               n, m, omega_exact(m - 1), omega_exact(m), omega_exact(m + 1), p, (int64_t)(p - m));
        fflush(stdout);
    }
    if (gap_qualifies(p, n, &q)) record_solution(S, p, q, m);
}

/*
 * launch_done: mark launch idx verified and advance the frontier (a359636.c's chunk_done()
 *   in units of launches); sets stop_now once the frontier has passed the launch that holds
 *   best_m (unless -a).
 *   In: S; idx.   Out: none.
 *   Errors: none.  Reads best_p/best_m without sol_mu (benign).
 */
static void launch_done(Scan *S, u64 idx)
{
    std::lock_guard<std::mutex> lk(S->fr_mu);
    S->done[idx % RING] = 1;
    while (S->frontier < S->nlaunch && S->done[S->frontier % RING]) {
        S->done[S->frontier % RING] = 0;
        S->frontier++;
    }
    if (S->best_p && !S->scan_all) {
        u64 kbest = (S->best_m - 3) / 6;
        u64 lbest = (kbest - S->k_lo) / S->lsize;
        if (S->frontier > lbest) S->stop_now = true;
    }
}

/*
 * verifier: CPU verification thread (THEORY G3).  Pops batches from S->queue, checks each
 *   survivor with check_survivor() and reports the batch with launch_done().
 *   In:     S.
 *   Out:    returns when the queue is empty and q_done is set.
 *   Errors: none.  Notifies q_cv after every pop, so a GPU driver waiting for room in the
 *           queue (64 batches) can launch again.
 */
static void verifier(Scan *S)
{
    for (;;) {
        Batch b;
        {
            std::unique_lock<std::mutex> lk(S->q_mu);
            S->q_cv.wait(lk, [&] { return !S->queue.empty() || S->q_done; });
            if (S->queue.empty()) return;
            b = std::move(S->queue.front());
            S->queue.pop_front();
            S->q_pending_k -= b.ks.size();
        }
        S->q_cv.notify_all();
        for (u64 k : b.ks) check_survivor(S, k);
        launch_done(S, b.idx);
    }
}

/*
 * save_state: write the checkpoint, if -S was given; FILE is replaced via FILE.tmp + rename:
 *               A359636 cuda state v1
 *               n <level>
 *               k_lo <first k>
 *               k_hi <k end, exclusive>
 *               lsize <k per launch>
 *               frontier <number of verified launches>
 *               survivors <count>
 *               triples <count>
 *               solution <p> <q> <m>          (one line per recorded solution)
 *   In: S.   Out: the file.
 *   Errors: FILE.tmp cannot be created -> message, previous checkpoint kept.  Write errors
 *           are NOT detected (Known issues): a full disk can leave a truncated checkpoint.
 */
static void save_state(Scan *S)
{
    if (!S->state_file) return;
    char tmp[4096];
    snprintf(tmp, sizeof tmp, "%s.tmp", S->state_file);
    FILE *f = fopen(tmp, "w");
    if (!f) { fprintf(stderr, "a359636_cuda: cannot write %s\n", tmp); return; }
    u64 fr;
    { std::lock_guard<std::mutex> lk(S->fr_mu); fr = S->frontier; }
    fprintf(f, "A359636 cuda state v1\n");
    fprintf(f, "n %d\nk_lo %" PRIu64 "\nk_hi %" PRIu64 "\nlsize %" PRIu64 "\nfrontier %" PRIu64 "\n",
            S->n, S->k_lo, S->k_hi, S->lsize, fr);
    fprintf(f, "survivors %llu\ntriples %llu\n", (unsigned long long)S->survivors.load(), (unsigned long long)S->triples.load());
    {
        std::lock_guard<std::mutex> lk(S->sol_mu);
        for (auto &s : S->sol) fprintf(f, "solution %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", s.p, s.q, s.m);
    }
    fclose(f);
    rename(tmp, S->state_file);
}

/*
 * load_state: resume from the checkpoint, if FILE exists (THEORY G4).
 *   In:     S with n, k_lo, k_hi, lsize set from the command line.
 *   Out:    frontier, resume_frontier, counters and recorded solutions restored.  A different
 *           k_hi (END) is accepted with a note; a different lsize (-b) converts the frontier
 *           to the new launch size, rounding down.  No file: fresh start.
 *   Errors: die() if the first line is not "A359636 cuda state v1" or if n or k_lo differ.
 *           The "resuming from" line uses the file's lsize and k_hi (cosmetic).
 */
static void load_state(Scan *S)
{
    FILE *f = fopen(S->state_file, "r");
    if (!f) return;
    char line[512];
    if (!fgets(line, sizeof line, f) || strncmp(line, "A359636 cuda state v1", 21)) die("bad state file");
    int n = -1; u64 klo = 0, khi = 0, lsize = 0, fr = 0;
    unsigned long long surv = 0, trip = 0;
    while (fgets(line, sizeof line, f)) {
        u64 p, q, m;
        if (sscanf(line, "n %d", &n) == 1) continue;
        if (sscanf(line, "k_lo %" SCNu64, &klo) == 1) continue;
        if (sscanf(line, "k_hi %" SCNu64, &khi) == 1) continue;
        if (sscanf(line, "lsize %" SCNu64, &lsize) == 1) continue;
        if (sscanf(line, "frontier %" SCNu64, &fr) == 1) continue;
        if (sscanf(line, "survivors %llu", &surv) == 1) continue;
        if (sscanf(line, "triples %llu", &trip) == 1) continue;
        if (sscanf(line, "solution %" SCNu64 " %" SCNu64 " %" SCNu64, &p, &q, &m) == 3) {
            S->sol.push_back({p, q, m});
            S->nsol++;
            if (!S->best_p || p < S->best_p) { S->best_p = p; S->best_m = m; }
        }
    }
    fclose(f);
    if (n != S->n || klo != S->k_lo)
        die("state file %s belongs to a different scan (n=%d, k_lo=%" PRIu64 ")", S->state_file, n, klo);
    if (khi != S->k_hi && !g_quiet)              /* END changed: fine, the frontier is relative to k_lo */
        fprintf(stderr, "state has k_hi %" PRIu64 ", now %" PRIu64 " (END changed on resume)\n", khi, S->k_hi);
    if (lsize != S->lsize) {                 /* different launch size: restart at or below the old k frontier */
        u64 kfr = fr * lsize;
        fr = kfr / S->lsize;
        if (!g_quiet) fprintf(stderr, "state has launch size %" PRIu64 ", now %" PRIu64 ": frontier converted\n", lsize, S->lsize);
    }
    S->frontier = fr;
    S->resume_frontier = fr;
    S->survivors = surv;
    S->triples = trip;
    if (!g_quiet) {
        char c1[32];
        fmt_commas(c1, 6 * (klo + fr * lsize) + 3);
        fprintf(stderr, "resuming from %s: m = %s (%.2f%%), %zu solution(s) on file\n",
                S->state_file, c1, 100.0 * (double)(fr * lsize) / (double)(khi - klo), S->sol.size());
    }
}

/*
 * progress: status line on stderr (position, percentage, Gm/s = 10^9 m per second, counts,
 *   best, elapsed, ETA), at most once a second on a terminal, once a minute otherwise.
 *   In: S; final_ = closing call (always printed, newline).   Out: the line.
 *   Errors: none.  Only the GPU driver (main) thread calls it.
 */
static void progress(Scan *S, bool final_)
{
    static double last = 0;
    double t = now();
    if (!final_ && t - last < (stderr_tty ? 1.0 : 60.0)) return;
    last = t;
    u64 fr;
    { std::lock_guard<std::mutex> lk(S->fr_mu); fr = S->frontier; }
    u64 kdone = fr * S->lsize;
    if (kdone > S->k_hi - S->k_lo) kdone = S->k_hi - S->k_lo;
    u64 mnow = 6 * (S->k_lo + kdone) + 3;
    double frac = (double)kdone / (double)(S->k_hi - S->k_lo);
    double el = t - S->t0;
    double rate = el > 0 ? (double)(kdone - S->resume_frontier * S->lsize) / el : 0;
    char c1[32], eta[32], elap[32], best[48] = "";
    fmt_commas(c1, mnow);
    fmt_time(eta, rate > 0 ? (double)((S->k_hi - S->k_lo) - kdone) / rate : -1);
    fmt_time(elap, el);
    if (S->best_p) snprintf(best, sizeof best, " best=%" PRIu64, S->best_p);
    fprintf(stderr, "%sm=%s (%.2f%%)  %.1f Gm/s  surv=%llu trip=%llu sol=%llu%s  %s elapsed, ETA %s%s",
            stderr_tty ? "\r\033[K" : "", c1, frac * 100, rate * 6 / 1e9,
            (unsigned long long)S->survivors.load(), (unsigned long long)S->triples.load(),
            (unsigned long long)S->nsol.load(), best, elap, eta, stderr_tty && !final_ ? "" : "\n");
    fflush(stderr);
}

/*
 * run_scan: the 'scan' command (and the selftest's work horse): set up the progressions and
 *   patterns, drive the GPU launch by launch, verify on cputhreads CPU threads, checkpoint,
 *   and print the result line (THEORY G1-G5).
 *   In:     n = level; m range start <= m <= end + 2 (m = 3 mod 6); cputhreads >= 1; T = sieve
 *           bound (0 = automatic); blocks per launch (0 = 65536); state_file = -S FILE or
 *           nullptr; interval = seconds between checkpoints; scan_all = -a; certain = output.
 *   Out:    returns the smallest solution found and sets *certain when every m below its m
 *           has been examined (for start <= 9 that is a(n)); otherwise 0 or an uncertain
 *           solution.  stdout: triples, solutions, result line; stderr: start-up lines, status
 *           line, summary.
 *   Errors: die() for an empty range, blocks * T >= 2^32, a -T beyond the prime table, more
 *           than 2^24 survivors in one launch (lower -b), a bad checkpoint, out of memory and
 *           any CUDA error (CUDA_CHECK).  There is no guard against END near 2^64
 *           (Known issues of a359636.c's version).
 */
static u64 run_scan(int n, u64 start, u64 end, int cputhreads, u32 T, int blocks, const char *state_file,
                    int interval, bool scan_all, bool *certain)
{
    Scan *S = new Scan();
    S->n = n;
    S->scan_all = scan_all;
    S->state_file = state_file;

    /* m = 3 (mod 6), start <= m <= end + 2 */
    u64 m_lo = start < 9 ? 9 : start;
    while (m_lo % 6 != 3) m_lo++;
    u64 m_hi = end + 2;
    if (m_hi < m_lo) die("empty range");
    S->k_lo = (m_lo - 3) / 6;
    S->k_hi = (m_hi - 3) / 6 + 1;
    u64 Bmax = m_hi + 1;

    /* prime table: trial division up to cbrt(Bmax) with margin, and at least up to 2^20 */
    u64 plim = (u64)iroot(Bmax, 3) + 1000;
    if (plim < 1u << 20) plim = 1u << 20;
    if (plim < 60000) plim = 60000;
    if (!g_primes || g_plimit < plim) build_primes(plim);

    /* T = 0: the smallest candidate T that forces min(n,6) small factors into every target,
       failing that min(n,5), failing that 1000 */
    if (T == 0) {
        static const u32 cand[] = { 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 8000, 10000, 15000, 20000 };
        int want = n < 6 ? n : 6;
        T = 0;
        for (u32 t : cand) if (compute_c(n, t, Bmax) >= want) { T = t; break; }
        if (!T) { want = n < 5 ? n : 5; for (u32 t : cand) if (compute_c(n, t, Bmax) >= want) { T = t; break; } }
        if (!T) T = 1000;
    }
    S->T = T;
    S->c = compute_c(n, T, Bmax);
    S->thr = S->c - 1;
    u32 thradd = (S->thr > 0) ? 0x010101u * (u32)(128 - S->thr) : 0x808080u, thrmask = 0x808080u;

    /* progressions for primes 29..T, ascending */
    for (size_t i = 0; i < g_nprimes && g_primes[i] <= T; i++) {
        u32 p = g_primes[i];
        if (p < 29) continue;
        for (int t = -1; t <= 1; t++) S->prog.push_back({p, target_residue(p, t), 1u << (8 * (t + 1))});
    }
    S->nsmall = 0;
    for (auto &pr : S->prog) if (pr.p <= PSMALL) S->nsmall++;
    /* patterns 5..13 and 17..23 */
    std::vector<u32> patA(PAT_A, 0), patB(PAT_B, 0);
    static const u32 pa[4] = { 5, 7, 11, 13 }, pb[3] = { 17, 19, 23 };
    for (u32 p : pa) if (p <= T) for (int t = -1; t <= 1; t++) { u32 r = target_residue(p, t), inc = 1u << (8 * (t + 1)); for (u32 k = r; k < PAT_A; k += p) patA[k] += inc; }
    for (u32 p : pb) if (p <= T) for (int t = -1; t <= 1; t++) { u32 r = target_residue(p, t), inc = 1u << (8 * (t + 1)); for (u32 k = r; k < PAT_B; k += p) patB[k] += inc; }

    /* launch geometry */
    if (blocks <= 0) blocks = 65536;
    if ((u64)blocks * T >= (1ull << 32))          /* kernel computes k0 mod p in 32 bits: blockIdx * (SEGK % p) */
        die("blocks * T must stay below 2^32 (have %d * %u); lower -b or -T", blocks, T);
    S->lsize = (u64)blocks * SEGK;
    S->nlaunch = (S->k_hi - S->k_lo + S->lsize - 1) / S->lsize;
    S->done.assign(RING, 0);
    S->outcap = 1u << 24;

    /* resume from the checkpoint, if any */
    if (state_file) load_state(S);

    /* device setup */
    CUDA_CHECK(cudaFuncSetAttribute(sieve_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SEGK * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&S->d_prog, S->prog.size() * sizeof(Prog)));
    CUDA_CHECK(cudaMalloc(&S->d_kb, S->prog.size() * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&S->d_patA, PAT_A * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&S->d_patB, PAT_B * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&S->d_out, (size_t)S->outcap * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&S->d_count, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(S->d_prog, S->prog.data(), S->prog.size() * sizeof(Prog), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(S->d_patA, patA.data(), PAT_A * sizeof(u32), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(S->d_patB, patB.data(), PAT_B * sizeof(u32), cudaMemcpyHostToDevice));
    /* kb[q] = kbase mod p for every progression, uploaded per launch; hostout is unused */
    std::vector<u32> kb(S->prog.size());
    std::vector<u64> hostout(S->outcap);

    if (!g_quiet) {
        char c1[32], c2[32];
        fmt_commas(c1, m_lo); fmt_commas(c2, m_hi);
        double marks = 0;
        for (auto &pr : S->prog) marks += 1.0 / pr.p;
        int dev; cudaDeviceProp prop; CUDA_CHECK(cudaGetDevice(&dev)); CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        fprintf(stderr, "A359636 level n=%d (CUDA on %s, %d SMs): m in [%s, %s] (m = 3 mod 6, %" PRIu64 " values), "
                "%d blocks x %d k per launch, %d CPU verifier threads\n", n, prop.name, prop.multiProcessorCount,
                c1, c2, S->k_hi - S->k_lo, blocks, SEGK, cputhreads);
        fprintf(stderr, "sieve primes 5..%u (%zu strided progressions, %d warp-level), each target needs >= %d prime "
                "factors <= %u (threshold %d after the forced 2/3), %.2f marks per k\n",
                S->T, S->prog.size(), S->nsmall, S->c, S->T, S->thr, marks);
    }

    /* start the verifier threads, then drive the GPU from this thread */
    S->t0 = now();
    std::vector<std::thread> vth;
    for (int i = 0; i < cputhreads; i++) vth.emplace_back(verifier, S);

    double last_save = now();
    u64 li = S->frontier;
    for (; li < S->nlaunch && !g_stop && !S->stop_now; li++) {
        /* bound how far the GPU may run ahead of verification */
        {
            std::unique_lock<std::mutex> lk(S->q_mu);
            S->q_cv.wait(lk, [&] { return S->queue.size() < 64 || g_stop; });
        }
        {
            std::lock_guard<std::mutex> lk(S->fr_mu);
            if (li >= S->frontier + RING - 2) { /* should not happen with queue bound */ }
        }
        /* launch li: k in [kbase, klimit) */
        u64 kbase = S->k_lo + li * S->lsize;
        u64 klimit = kbase + S->lsize;
        if (klimit > S->k_hi) klimit = S->k_hi;
        for (size_t i = 0; i < S->prog.size(); i++) kb[i] = (u32)(kbase % S->prog[i].p);
        CUDA_CHECK(cudaMemcpy(S->d_kb, kb.data(), kb.size() * sizeof(u32), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(S->d_count, 0, sizeof(unsigned int)));
        int nb = (int)((klimit - kbase + SEGK - 1) / SEGK);
        sieve_kernel<<<nb, BLOCKDIM, SEGK * sizeof(u32)>>>(kbase, klimit, S->d_prog, S->d_kb, (int)S->prog.size(),
                                                           S->nsmall, S->d_patA, S->d_patB, thradd, thrmask,
                                                           S->d_out, S->d_count, S->outcap);
        /* launch errors show here, execution errors at the synchronizing copy below */
        CUDA_CHECK(cudaGetLastError());
        unsigned int cnt;
        CUDA_CHECK(cudaMemcpy(&cnt, S->d_count, sizeof cnt, cudaMemcpyDeviceToHost));
        /* more survivors than d_out holds: stop rather than lose any */
        if (cnt > S->outcap) die("survivor buffer overflow (%u > %u): raise capacity or lower -b", cnt, S->outcap);
        Batch b; b.idx = li; b.ks.resize(cnt);
        if (cnt) CUDA_CHECK(cudaMemcpy(b.ks.data(), S->d_out, (size_t)cnt * sizeof(u64), cudaMemcpyDeviceToHost));
        S->survivors += cnt;
        {
            std::lock_guard<std::mutex> lk(S->q_mu);
            S->q_pending_k += cnt;
            S->queue.push_back(std::move(b));
        }
        S->q_cv.notify_all();
        if (!g_quiet) progress(S, false);
        if (state_file && now() - last_save >= interval) { save_state(S); last_save = now(); }
    }
    /* drain: no more launches; the verifiers finish every queued batch, then checkpoint */
    {
        std::lock_guard<std::mutex> lk(S->q_mu);
        S->q_done = true;
    }
    S->q_cv.notify_all();
    for (auto &t : vth) t.join();
    if (state_file) save_state(S);
    if (!g_quiet) progress(S, true);

    /* covered_m: every m = 3 (mod 6) up to it has been examined (a359636.c, THEORY section 2) */
    u64 covered_k = S->frontier >= S->nlaunch ? S->k_hi : S->k_lo + S->frontier * S->lsize;
    u64 covered_m = covered_k ? 6 * covered_k - 3 : 0;
    u64 result = 0;
    *certain = false;
    if (S->best_p && S->best_m <= covered_m) { result = S->best_p; *certain = true; }
    if (!g_quiet) {
        char c1[32];
        fmt_commas(c1, covered_m);
        fprintf(stderr, "scanned m <= %s: %llu survivors, %llu triples, %llu solutions\n", c1,
                (unsigned long long)S->survivors.load(), (unsigned long long)S->triples.load(),
                (unsigned long long)S->nsol.load());
    }
    /* The result line (THEORY G5).  Note: the "complete scan" bound below prints END; that is
       only proven when END is not 5 or 0 (mod 6) -- see Known issues in the header. */
    bool partial = m_lo > 9;                       /* START given: statements are about this range only */
    if (*certain && !partial) printf("a(%d) = %" PRIu64 "\n", n, result);
    else if (*certain) printf("smallest qualifying p with m in [%" PRIu64 ", %" PRIu64 "] is %" PRIu64 " (partial range: a(%d) <= this)\n", m_lo, covered_m, result, n);
    else if (S->best_p) printf("a(%d) <= %" PRIu64 " (scan incomplete below it)\n", n, S->best_p);
    else if (S->frontier >= S->nlaunch && !partial) printf("a(%d) > %" PRIu64 " (no qualifying gap with p <= %" PRIu64 ")\n", n, end, end);
    else if (S->frontier >= S->nlaunch) printf("no qualifying gap with m in [%" PRIu64 ", %" PRIu64 "] (partial range)\n", m_lo, m_hi);
    else if (!partial) printf("a(%d) > %" PRIu64 " (scan stopped early; no qualifying gap with p below that)\n", n, covered_m > 2 ? covered_m - 2 : 0);
    else printf("no qualifying gap with m in [%" PRIu64 ", %" PRIu64 "] (partial range, stopped early)\n", m_lo, covered_m);
    fflush(stdout);
    cudaFree(S->d_prog); cudaFree(S->d_kb); cudaFree(S->d_patA); cudaFree(S->d_patB); cudaFree(S->d_out); cudaFree(S->d_count);
    delete S;
    return result;
}

/* ------------------------------------------------------------------ */
/* Commands                                                            */
/* ------------------------------------------------------------------ */

/*
 * cmd_selftest: reproduce a(1..7) (with8: also a(8)) by scans from 0 that must end certain.
 *   Launches are small (256 blocks) for n <= 6, where nearly every k may survive and the
 *   2^24-entry survivor list must not overflow; 4096 blocks for n >= 7.
 *   In: cputhreads; with8.   Out: 0 if all pass, 1 otherwise; one line per level.
 *   Errors: none (a CUDA error dies in run_scan()).
 */
static int cmd_selftest(int cputhreads, bool with8)
{
    int fails = 0, top = with8 ? 8 : 7;
    for (int n = 1; n <= top; n++) {
        bool certain;
        double t = now();
        u64 r = run_scan(n, 0, KNOWN[n], cputhreads, 0, n <= 6 ? 256 : 4096, nullptr, 60, false, &certain);
        bool ok = certain && r == KNOWN[n];
        printf("selftest n=%d: got %" PRIu64 "%s expected %" PRIu64 " -> %s  (%.1fs)\n", n, r,
               certain ? "" : " (uncertain)", KNOWN[n], ok ? "ok" : "FAIL", now() - t);
        fflush(stdout);
        if (!ok) fails++;
    }
    printf("selftest %s\n", fails ? "FAILED" : "passed");
    return fails ? 1 : 0;
}

/*
 * cmd_verify: check the gap after the prime p at level n from scratch and print every number
 *   in it with its factorization (independent of the sieve: uses omega_exact()).
 *   In: n; p.   Out: 0 if the gap qualifies, 1 otherwise (also if p is not prime).
 *   Errors: none.
 */
static int cmd_verify(int n, u64 p)
{
    build_primes(1u << 21);
    char buf[256];
    if (!is_prime(p)) { printf("%" PRIu64 " is not prime\n", p); return 1; }
    u64 q = next_prime(p);
    printf("p = %" PRIu64 ", next prime q = %" PRIu64 ", gap %" PRIu64 "\n", p, q, q - p);
    bool ok = q - p >= 4;
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
 * usage: print the command summary on stderr and exit with status 2.
 */
static void usage(void)
{
    fputs("usage:\n"
          "  a359636_cuda scan N [START] END [-t CPUTHREADS] [-T SIEVEMAX] [-b BLOCKS] [-S STATE] [-i SECS] [-a] [-q]\n"
          "  a359636_cuda verify N P\n"
          "  a359636_cuda selftest [-t CPUTHREADS] [-8]\n", stderr);
    exit(2);
}

/*
 * main: parse the command line and dispatch to scan, verify or selftest.
 *   In:     argv (header, Usage).  A token starting with '-' and a non-digit is an option.
 *   Out:    the command's exit status (0 for scan; selftest/verify as documented above).
 *   Errors: unknown command/option or missing arguments -> usage() (exit 2); malformed
 *           numbers and scan's N outside 1..20 -> die() (exit 1).
 */
int main(int argc, char **argv)
{
    stderr_tty = isatty(2);
    if (argc < 2) usage();
    signal(SIGINT, on_sigint);
    signal(SIGTERM, on_sigint);
    const char *cmd = argv[1];
    /* selftest [-t CPUTHREADS] [-8] [-q] */
    if (!strcmp(cmd, "selftest")) {
        int th = 8; bool with8 = false;
        for (int i = 2; i < argc; i++) {
            if (!strcmp(argv[i], "-t") && i + 1 < argc) th = atoi(argv[++i]);
            else if (!strcmp(argv[i], "-8")) with8 = true;
            else if (!strcmp(argv[i], "-q")) g_quiet = true;
            else usage();
        }
        return cmd_selftest(th, with8);
    }
    /* verify N P */
    if (!strcmp(cmd, "verify")) {
        if (argc != 4) usage();
        return cmd_verify(atoi(argv[2]), parse_num(argv[3]));
    }
    /* scan N [START] END [options]  (see usage()) */
    if (!strcmp(cmd, "scan")) {
        if (argc < 4) usage();
        int n = atoi(argv[2]);
        if (n < 1 || n > 20) die("N must be 1..20");
        u64 pos[2]; int npos = 0;
        int th = 8, blocks = 0, interval = 60; u32 T = 0; const char *state = nullptr; bool all = false;
        for (int i = 3; i < argc; i++) {
            if (!strcmp(argv[i], "-t") && i + 1 < argc) th = atoi(argv[++i]);
            else if (!strcmp(argv[i], "-T") && i + 1 < argc) T = (u32)parse_num(argv[++i]);
            else if (!strcmp(argv[i], "-b") && i + 1 < argc) blocks = (int)parse_num(argv[++i]);
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
        if (th < 1) th = 1;
        if (T && T < 30) T = 30;
        bool certain;
        run_scan(n, start, end, th, T, blocks, state, interval, all, &certain);
        return 0;
    }
    usage();
    return 2;
}
