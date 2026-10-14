/*
 * a034822.c -- exhaustive search for palindromic squares of a given length L
 *
 * ===========================================================================
 * PURPOSE
 * ===========================================================================
 *
 * OEIS A034822: numbers k such that there are no palindromic squares of
 * length k.  Known: 2,4,8,10,14,18,20,24,30,38,40,46,54,56,62,64.  All lengths
 * <= 69 have been searched exhaustively (De Geest / R. Xiao, CUDA; the single
 * 68-digit square 7256171055736382499839982033184475^2 was found 2025-04-14),
 * so the next candidate term is L = 70.  All terms are even (odd lengths
 * always contain (10^k+1)^2), so only even L matter; the tool works for any L.
 *
 * This program lists every palindromic square with exactly L decimal digits
 * (L = 11..76, subject to the parameter limits in setup()).  If it finds none
 * for an even L, then L is a term of A034822.  It is the CPU reference
 * implementation; cuda/a034822_cuda.cu runs the same algorithm on a GPU and is
 * validated against it.
 *
 * ===========================================================================
 * THEORY OF OPERATION
 * ===========================================================================
 *
 * Notation used throughout the file
 * ---------------------------------
 *   L    number of digits of the square S = n^2 we are looking for
 *   D    ceil(L/2), the number of digits of the root n
 *   h    split point: n = t*10^h + B with 0 <= B < 10^h  ("B" = low h digits,
 *        "t" = high D-h digits).  Chosen near D/2 (h = 17 for L = 70).
 *   e    L - 3h  (exponent that appears in the window formula below)
 *   P    the top h digits of S, as an integer of exactly h digits
 *   Q    floor(B^2 / 10^h), the part of B^2 above the low h digits
 *   s_i  decimal digit i of S (s_0 = units digit)
 *
 * 1. Two-ended matching (the core idea, ~10^(L/4) work instead of 10^(L/2))
 * ------------------------------------------------------------------------
 *   The low h digits of S = (t*10^h + B)^2 depend only on B: they are
 *   B^2 mod 10^h.  If S is a palindrome, its TOP h digits are those same
 *   digits reversed, so B alone determines P = reverse(B^2 mod 10^h).
 *   Then S must lie in [P*10^(L-h), (P+1)*10^(L-h)), and since S ~ t^2*10^(2h),
 *
 *        isqrt(P*10^e) <= t <= isqrt((P+1)*10^e),     e = L - 3h.
 *
 *   (Proof sketch: n >= sqrt(P*10^(L-h)) and n = t*10^h + B with B < 10^h give
 *   t > sqrt(P*10^e) - 1, i.e. t >= floor(sqrt(P*10^e)); the upper bound is
 *   similar.  The window may contain one value too many at either end; those
 *   are rejected by the digit tests.)
 *
 *   The window holds about 10^(D-2h) values of t, i.e. O(1) when h ~ D/2.  So
 *   the search enumerates the ~10^h choices of B, derives P and the window,
 *   and tests the few t inside it.  Each test asks whether the next digits
 *   of S (just inside the already-matched outer h digits) also mirror.
 *
 * 2. Enumerating B: a digit DFS
 * -----------------------------
 *   B is built one digit at a time, from the units digit up (dfs()).  If B_i
 *   holds the low i digits and we append digit d at position i, then
 *
 *        B_{i+1}^2 = B_i^2 + 2*d*B_i*10^i + d^2*10^(2i),
 *
 *   so with Q_i = floor(B_i^2 / 10^i) (always < 10^i, fits in 64 bits):
 *
 *        v = Q_i + 2*d*B_i + d^2*10^i,   s_i = v mod 10,   Q_{i+1} = floor(v/10).
 *
 *   Each new digit of B fixes one more digit s_i of S, which is written into P
 *   at position h-1-i (P is the reverse).  After h levels B, P and Q are
 *   complete; that is a "leaf".  Along the way the DFS also carries
 *   f ~ B*2^64/10^h (B as a 64-bit binary fraction, used by the fast filter)
 *   and B mod 11.
 *
 * 3. Symmetries: one leaf serves up to 8 roots
 * --------------------------------------------
 *   Every root r with r^2 = B^2 (mod 10^h) gives the same P and hence the
 *   same t-window, so it is tested at the same leaf instead of having its own:
 *     a) -B mod 10^h                              (negation)
 *     b) B + 5*10^(h-1) mod 10^h                  ("H5" shift; for h >= 2,
 *        (B + 5*10^(h-1))^2 = B^2 + B*10^h + 25*10^(2h-2) = B^2 mod 10^h)
 *     c) B*E' mod 10^h, with E' = -1 (mod 2^h) and E' = 1 (mod 5^h)  ("sigma2")
 *        Since E'^2 = 1 (mod 10^h), (B E')^2 = B^2.  Multiplying by E' leaves
 *        digits 0..v2(B) unchanged and adds 5 (mod 10) to digit v2(B)+1, where
 *        v2(B) = number of trailing zero BITS of B.
 *   To enumerate exactly one representative per orbit, the DFS restricts:
 *     - the unit digit to 1..5 (units 6..9 are the negatives of 4..1; unit 0
 *       is impossible because S would end in 0);
 *     - for unit 5, the tens digit to 0..4 (negation flips it to 9 - d);
 *     - the top digit (position h-1) to 0..4 (H5 shift adds 5 there);
 *     - the digit sigma2 shifts (position v2(B)+1, if <= h-2 and unit != 5)
 *       to 0..4.  See s2_level().
 *   A leaf then tests 4 roots {B, B+H5, -B, H5-B}, plus the 4 sigma2 images
 *   when the sigma2 restriction was applied (these are distinct from the
 *   first 4 exactly when v2(B) <= h-3).  The sigma2 images are tracked
 *   incrementally in the DFS (dstep(), s2_roots_inc()).  Setting the
 *   environment variable A034822_NOS2 disables sigma2 (4 roots per leaf, about
 *   twice as many leaves); results must be identical.
 *
 * 4. Divisibility by 11 (even L only)
 * -----------------------------------
 *   Every palindrome with an even number of digits is divisible by 11
 *   (alternating digit sum is 0), and 11 is prime, so 11 | n.  Because
 *   10 = -1 (mod 11), n = t*10^h + r = (-1)^h t + r (mod 11), which fixes
 *   t mod 11 for each root r: only every 11th t in the window is a candidate.
 *
 * 5. Computing the window cheaply
 * -------------------------------
 *   An exact 128-bit isqrt per leaf would dominate the run time.  Instead,
 *   the DFS picks a "base" node at depth IB (setup()): below it, P varies only
 *   in its low h-IB digits, i.e. P = Pb + u with 0 <= u < 10^(h-IB).  The
 *   base computes R = sqrt(Pb*10^e) once (make_base(): integer part Rb plus
 *   fraction), and each leaf uses the Taylor expansion
 *
 *        sqrt((Pb+u)*10^e) ~= Rb + frac + u*s1 - u^2*s2,
 *        s1 = R/(2 Pb),  s2 = R/(8 Pb^2),
 *
 *   in double precision.  IB is chosen so the cubic remainder is < 1e-8, and
 *   the window is widened by EPS = 1e-5 on each side, so it always contains
 *   the exact window (verified by the PARANOID build).
 *
 * 6. Testing a candidate t for a root r (three stages)
 * ----------------------------------------------------
 *   Write S = (t*10^h + r)^2 = t^2*10^(2h) + C*10^h + (r^2 mod 10^h), where
 *   C = 2*t*r + floor(r^2/10^h).  The outer h digit pairs already match by
 *   construction.  The next x pairs (x = 2 normally, 1 if h = 18) are:
 *     low side:  S digits h, h+1   = C mod 100
 *     high side: the x digits of floor(S/10^(2h)) just below P
 *   Stage 1 (filter, every candidate): compute the high side approximately
 *     as D = t^2 + floor(2*t*f/2^64) - P*10^e, where f/2^64 ~ r/10^h.  The
 *     approximation error is a few units, while the digits we compare change
 *     every 10^(e-x) units, so a band of +-SLACK around the exact test makes
 *     the filter a strict superset of the exact test.  In the fast path
 *     (fast_root) everything is done modulo 2^64: a true hit has
 *     0 <= D < 10^e < 2^64, so D mod 2^64 is exact where it matters.
 *     About 1 candidate in 100 passes.
 *   Stage B (stageB(), fast path only): the same comparison for 4 digit
 *     pairs; about 1 in 100 again.
 *   Stage 3 (exact, test_cand()): 128-bit arithmetic for the x pairs, then
 *     the digits h..2h-1 (in c0 = C mod 10^h) against their mirror images in
 *     Y = floor(S/10^(2h)).  Survivors are almost always genuine; the final
 *     decimal conversion (square_digits + is_pal) decides.
 *
 * 7. Two leaf implementations
 * ---------------------------
 *   fast_leaf()/fast_root() are used when L is even, h >= 4 and every window
 *   is at most 22 wide (so each root has at most 2 candidates in its mod-11
 *   class): branch-free evaluation of both candidates, wrapping 64-bit
 *   arithmetic, stage B.  do_leaf()/do_root() handle everything else (odd L,
 *   small test lengths, wide windows) with a plain loop and 128-bit filter.
 *
 * 8. Parallelism and checkpointing
 * --------------------------------
 *   A task is a fixed prefix of B: the unit digit plus the next p-1 digits
 *   (5*10^(p-1) task numbers, some invalid because of the restrictions above).
 *   Worker threads take task numbers from an atomic counter, run the DFS
 *   below the prefix, and mark the task done in done_map.  A state file holds
 *   done_map plus counters; a run with the same L, h, p resumes from it.
 *   Tasks are committed atomically with respect to checkpoints, so after any
 *   number of interruptions every task is counted exactly once.
 *   -r a:b restricts the run to task numbers [a, b), so several machines can
 *   split one length.
 *
 * Arithmetic notes
 * ----------------
 *   - All quantities are unsigned 64-bit or 128-bit integers.  arm64 has no
 *     128/64 divide instruction (clang calls a slow library routine), so
 *     udiv128_64() uses a double-precision estimate plus exact correction.
 *   - Many expressions intentionally wrap modulo 2^64 (e.g. 0 - f for the
 *     negated root, D in fast_root); these are commented where they occur.
 *   - Limits: B < 10^h <= 10^18, t < 10^(D-h) <= 10^18, t^2 < 2^128,
 *     P*10^e < 2^127 (enforced by the h range in setup()).
 *
 * Validation builds
 * -----------------
 *   -DPARANOID  re-checks every window against exact isqrt, every filter
 *               rejection against the exact test, and every incremental sigma2
 *               root against a direct computation; aborts on any mismatch.
 *   -DNOROOT    skips the candidate tests in do_root() (timing experiments).
 *   selftest    compares the count for each L = 11..maxL with A263618 (KNOWN[]).
 *
 * ===========================================================================
 * USAGE
 * ===========================================================================
 *
 *   a034822 search L [-t threads] [-h H] [-p P] [-S state] [-i secs] [-r a:b]
 *       -t  worker threads (default: all online CPUs)
 *       -h  split point h (default D/2, clamped to the valid range)
 *       -p  task depth p (default h-6, capped at 7)
 *       -S  state file for checkpoint/resume; -i seconds between saves (300)
 *       -r  task-number range [a, b) for splitting a run across machines
 *   a034822 selftest [maxL [-t threads]]   counts vs A263618 for L = 11..maxL
 *                                           (maxL is capped at 69, KNOWN[] size)
 *   a034822 check n                        is n^2 a palindrome?
 *
 * Results: one line "FOUND L n n^2" per palindromic square on stdout, then a
 * final "DONE L count=..." (or "PARTIAL" if interrupted / range incomplete);
 * progress on stderr.  Exit status: 0 = ran (search/check), 1 = bad
 * parameters, 2 = usage error; selftest returns 1 on any mismatch.
 *
 * Build: cc -O3 -mcpu=native -o a034822 a034822.c -lpthread -lm
 *        (add -DPARANOID for the checking build).
 */
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef uint64_t u64;
typedef int64_t i64;
typedef unsigned __int128 u128;
typedef __int128 i128;

/* POW10[k] = 10^k for k = 0..19 (10^19 still fits in a u64).  Filled in main()
 * before anything else runs. */
static u64 POW10[20];

/* A263618 (number of palindromic squares with n digits), n = 1..69;
 * n = 68, 69 from De Geest's table (Nov 2025).  KNOWN[0] is unused.  Used only
 * by selftest(), which caps maxL at 69. */
static const int KNOWN[70] = {0,
    4,0,3,0,7,1,5,0,11,0,5,1,19,0,13,1,25,0,18,0,48,1,31,0,70,1,44,2,105,
    0,70,1,153,1,98,3,209,0,132,0,291,1,181,1,384,0,234,2,496,1,301,1,
    636,0,383,0,798,1,474,1,981,0,578,0,1199,2,701,1,1443};

/* ---------- 128-bit helpers (no 128/64 hardware divide on arm64) ---------- */

/*
 * u128_to_double -- convert an unsigned 128-bit integer to double.
 *   In:  x, any value.
 *   Out: x rounded to double (relative error <= ~2^-52).
 *   Why: written out by hand because the compiler's u128 -> double conversion
 *        is a library call on some targets.  No error conditions.
 */
static inline double u128_to_double(u128 x)
{
    return (double)(u64)(x >> 64) * 18446744073709551616.0 + (double)(u64)x;
}

/*
 * udiv128_64 -- exact floor(x / d) for a 128-bit dividend and 64-bit divisor.
 *   In:  x (128-bit), d (64-bit, must be > 0).
 *   Out: the exact quotient, which the CALLER GUARANTEES fits in 64 bits.
 *   How: estimate q with a double division (error up to ~2^13 in q), compute
 *        the signed remainder r = x - q*d exactly in 128 bits, correct q with a
 *        second double estimate of r/d, then fix the last +-1 with loops.
 *   Errors: d = 0 is undefined (division by zero in double -> inf -> garbage);
 *        if the true quotient is >= 2^64 the result is meaningless (q is
 *        clamped to UINT64_MAX before correction and the remainder overflows).
 *        Both preconditions hold at every call site; selftest() exercises the
 *        function on 2,000,000 random inputs.
 */
static inline u64 udiv128_64(u128 x, u64 d)
{
    double qd = u128_to_double(x) / (double)d;
    u64 q = qd >= 18446744073709549568.0 ? UINT64_MAX : (u64)qd;
    i128 r = (i128)(x - (u128)q * d);
    /* |r| <= ~2^13 * d: one more estimate brings it to within +-1 */
    double rd = r < 0 ? -u128_to_double((u128)(-r)) : u128_to_double((u128)r);
    i64 adj = (i64)(rd / (double)d);
    q += (u64)adj;
    r -= (i128)adj * (i128)d;
    while (r < 0) { q--; r += d; }
    while (r >= (i128)d) { q++; r -= d; }
    return q;
}

/*
 * isqrt128 -- exact integer square root floor(sqrt(x)).
 *   In:  x < 2^127.
 *   Out: r with r^2 <= x < (r+1)^2.
 *   How: double sqrt estimate (relative error ~1e-16), one integer Newton
 *        step r = (r + x/r)/2 via udiv128_64, then +-1 fix-up loops.
 *   Errors: for x >= 2^127 the final (r+1)^2 test can overflow u128 and the
 *        loop may not terminate correctly; callers pass at most
 *        (P+1)*10^e < 10^38 < 2^127 (setup() bounds h).  x = 0 returns 0.
 */
static inline u64 isqrt128(u128 x)
{
    if (x == 0) return 0;
    u64 r = (u64)sqrt(u128_to_double(x));
    if (r == 0) r = 1;
    /* one Newton step from a ~1e-16 relative estimate */
    u128 s = ((u128)r + udiv128_64(x, r)) >> 1;
    r = (u64)s;
    while ((u128)r * r > x) r--;
    while ((u128)(r + 1) * (r + 1) <= x) r++;
    return r;
}

/* ---------- full palindrome check of n^2, n < 10^37 ---------- */

/*
 * square_digits -- decimal string of n^2.
 *   In:  n (any u128; the square of up to 39 digits is up to 78 digits).
 *        out: buffer of at least 80 chars (callers use 128).
 *   Out: writes the decimal digits of n^2 (no leading zeros, "0" for n = 0)
 *        as a NUL-terminated string; returns the number of digits.
 *   How: schoolbook multiplication in base 10^9 (limbs fit u64 products plus
 *        carries), then the limbs are expanded to digits.  Slow (~1 us) but only
 *        called for final verification and the "check" command.
 *   Errors: none for u128 inputs (6 limbs of 10^9 hold up to 10^54 > 2^128).
 */
static int square_digits(u128 n, char *out) /* returns number of digits */
{
    u64 a[6] = {0}, s[12] = {0};
    int na = 0;
    while (n) { a[na++] = (u64)(n % 1000000000u); n /= 1000000000u; }
    for (int i = 0; i < na; i++) {
        u64 carry = 0;
        for (int j = 0; j < na; j++) {
            u64 v = s[i + j] + a[i] * a[j] + carry;
            s[i + j] = v % 1000000000u;
            carry = v / 1000000000u;
        }
        int k = i + na;
        while (carry) { u64 v = s[k] + carry; s[k] = v % 1000000000u; carry = v / 1000000000u; k++; }
    }
    int ns = 2 * na;
    while (ns > 0 && s[ns - 1] == 0) ns--;
    char buf[128];
    int len = 0;
    for (int i = 0; i < ns; i++) {
        u64 v = s[i];
        for (int k = 0; k < 9; k++) { buf[len++] = (char)('0' + v % 10); v /= 10; }
    }
    while (len > 1 && buf[len - 1] == '0') len--;
    for (int i = 0; i < len; i++) out[i] = buf[len - 1 - i];
    out[len] = 0;
    return len;
}

/*
 * is_pal -- is the string s[0..len-1] a palindrome?
 *   In:  s, len >= 0.   Out: 1 if s reads the same backwards, else 0.
 */
static int is_pal(const char *s, int len)
{
    for (int i = 0, j = len - 1; i < j; i++, j--)
        if (s[i] != s[j]) return 0;
    return 1;
}

/*
 * u128_str -- decimal string of n.
 *   In:  n; out: buffer of at least 40 chars (u128 has at most 39 digits).
 *   Out: NUL-terminated decimal representation in out.  No error conditions.
 */
static void u128_str(u128 n, char *out)
{
    char buf[64];
    int len = 0;
    do { buf[len++] = (char)('0' + (int)(n % 10)); n /= 10; } while (n);
    for (int i = 0; i < len; i++) out[i] = buf[len - 1 - i];
    out[len] = 0;
}

/*
 * parse_u128 -- parse a decimal string into a u128.
 *   In:  s, a string of digits '0'..'9' only.
 *   Out: 1 and *out set on success; 0 for an empty string, any non-digit
 *        (signs, spaces, etc. are rejected) or a value above 2^128 - 1.
 *   Only used by "check".
 */
static int parse_u128(const char *s, u128 *out)
{
    u128 v = 0;
    if (!*s) return 0;
    const u128 max = ~(u128)0;
    for (; *s; s++) {
        if (*s < '0' || *s > '9') return 0;
        u128 dgt = (u128)(*s - '0');
        if (v > (max - dgt) / 10) return 0; /* v*10 + dgt would exceed 2^128 - 1 */
        v = v * 10 + dgt;
    }
    *out = v;
    return 1;
}

/* ---------- search parameters and shared state ---------- */

/* Run parameters, filled in by setup() and read-only while threads run. */
typedef struct {
    int L, D, h, e, x; /* L digits of S, D = ceil(L/2) digits of n, split h, e = L-3h,
                          x = number of extra digit pairs checked cheaply (1 or 2) */
    int p;             /* task depth: low p digits of B fixed per task */
    int mod11;         /* even L: 11 | n */
    u64 ntasks;        /* number of task numbers, 5*10^(p-1) (some invalid) */
    u64 pe, pex;       /* 10^e (as needed), 10^(e-x) */
} params_t;

static params_t G;

/* Shared run state.
 * Each finished task is "committed" under commit_lock: its statistics are added
 * to the totals, found_done (squares in finished tasks) is increased and its
 * done_map byte is set, all in one critical section.  save_state() copies the
 * totals and done_map under the same lock, so a checkpoint is always a
 * consistent snapshot: every task marked done is counted exactly once, and
 * a task in progress is neither counted nor marked (it reruns after a resume).
 * found_count counts every report() immediately (for progress display);
 * found_done is the exact, checkpointed count.  done_map is only read without
 * the lock before the workers start and after they have been joined. */
static pthread_mutex_t out_lock = PTHREAD_MUTEX_INITIALIZER;    /* serializes FOUND output */
static pthread_mutex_t commit_lock = PTHREAD_MUTEX_INITIALIZER; /* task commit vs checkpoint */
static atomic_ullong next_task;   /* next task number to hand out */
static atomic_ullong found_count, nodes_total, cands_total, pass_total, filt_total;
static atomic_ullong found_done;  /* squares in committed (finished) tasks */
static atomic_ullong tasks_done;  /* tasks committed by this process */
static unsigned char *done_map; /* 1 byte per task */
static u64 task_lo, task_hi;      /* task range [task_lo, task_hi) of this run (-r) */
static volatile sig_atomic_t stop_flag = 0; /* set by SIGINT/SIGTERM */
static int quiet = 0;             /* selftest: suppress FOUND lines and progress */

/* Found squares collected for selftest (duplicate detection); only the first
 * MAXFOUND are kept, which is far more than any L <= 69 has. */
#define MAXFOUND 4096
static u128 found_list[MAXFOUND];

/* Per-thread statistics for one task, merged into the global atomics when the
 * task finishes.  nodes = leaves visited; cands = candidates (do_root) or
 * window widths (fast_leaf); filt = stage-1 passes; pass = exact x-pair passes;
 * pass3 = passes of the c0/Y digit check; found = palindromic squares. */
typedef struct {
    u64 nodes, cands, filt, pass, pass3, found;
} stats_t;

/*
 * report -- record and print one palindromic square.
 *   In:  n, a root whose square has been verified to be an L-digit palindrome.
 *   Out: prints "FOUND L n n^2" to stdout (unless quiet), increments
 *        found_count (live display; the exact count is found_done, updated when
 *        the task commits), stores n in found_list[] for the selftest.
 *   Thread-safe (out_lock).  If more than MAXFOUND squares are found, the
 *        extras are counted and printed but not stored (selftest only).
 */
static void report(u128 n)
{
    char ns[64], ss[128];
    u128_str(n, ns);
    int len = square_digits(n, ss);
    pthread_mutex_lock(&out_lock);
    u64 k = atomic_fetch_add(&found_count, 1);
    if (k < MAXFOUND) found_list[k] = n;
    if (!quiet) {
        printf("FOUND %d %s %s\n", len, ns, ss);
        fflush(stdout);
    }
    pthread_mutex_unlock(&out_lock);
}

/* Window base for a DFS subtree: every leaf below has P in [Pb, Pb + 10^(h-ib)),
 * and sqrt(P*10^e) = Rb + v(u), u = P - Pb, with
 * v(u) = frac + u*s1 - u^2*s2 (cubic remainder < 1e-7 by choice of ib).
 * See THEORY section 5. */
typedef struct {
    u64 Pb, Rb;          /* base P (low h-IB digits zero) and floor(sqrt(Pb*10^e)) */
    double frac, s1, s2; /* fractional part of the root and Taylor coefficients */
} base_t;

static int IB;          /* depth at which the base is computed */
static double EPS = 1e-5; /* window widening, >> Taylor + rounding error (~1e-7) */
#define SLACK 64        /* filter slack for the fixed-point top-digit estimate:
                           larger than the worst error of floor(2*t*f/2^64) as an
                           estimate of floor(C/10^h) (about 9h*2t/2^64 + 2 < 25 for
                           L = 70) and much smaller than 10^(e-x) */

/*
 * make_base -- precompute the window polynomial for a DFS subtree.
 *   In:  Pb, the P value at the base node (its low h-IB digits are still 0);
 *        b, output.
 *   Out: b->Rb = floor(sqrt(Pb*10^e)), b->frac = sqrt(Pb*10^e) - Rb (computed
 *        as (X - Rb^2)/(2 Rb), accurate to ~1e-17), and the Taylor
 *        coefficients s1 = R/(2 Pb), s2 = R/(8 Pb^2) in units of P.
 *   Errors: Pb must be > 0 (always true: P's leading digit is S's last digit,
 *        which is nonzero); the 1e-300 guards the R = 0 case of tiny tests.
 *        Pb*10^e must be < 2^127 (setup() limits h).
 */
static void make_base(u64 Pb, base_t *b)
{
    u128 X = (u128)Pb * G.pe;
    u64 r = isqrt128(X);
    double R = (double)r;
    b->Pb = Pb;
    b->Rb = r;
    b->frac = u128_to_double(X - (u128)r * r) / (2.0 * R + 1e-300);
    double Rreal = R + b->frac, Pd = (double)Pb;
    b->s1 = Rreal / (2.0 * Pd);
    b->s2 = Rreal / (8.0 * Pd * Pd);
}

/* umulh -- high 64 bits of the 128-bit product a*b (one instruction on arm64/x86). */
static inline u64 umulh(u64 a, u64 b) { return (u64)(((u128)a * b) >> 64); }

/*
 * exact_top_ok -- exact version of the stage-1 test (PARANOID builds only).
 *   In:  candidate t, root Bp, Bq = floor(Bp^2/10^h), want = the required top
 *        h+x digits of S (P followed by the mirrored digits).
 *   Out: 1 if floor(S/10^(L-h-x)) == want for S = (t*10^h + Bp)^2, else 0.
 *   Uses the identity floor(S/10^(2h)) = t^2 + floor(C/10^h), C = 2 t Bp + Bq.
 */
static inline int exact_top_ok(u64 t, u64 Bp, u64 Bq, u64 want)
{
    u128 C = (u128)2 * t * Bp + Bq;
    u64 X = udiv128_64(C, POW10[G.h]);
    u128 Y = (u128)t * t + X;
    return udiv128_64(Y, G.pex) == want;
}

/* Constants derived from h (set in setup()).  For the four roots of one group,
 * x, x + H5, 10^h - x, H5 - x, their Bq = floor(root^2/10^h) follow from
 * Q = floor(x^2/10^h) without a division:
 *   (x + H5)^2  = x^2 + x*10^h + 25*10^(2h-2)   -> Bq = Q + x + Q25
 *   (10^h - x)^2 = 10^(2h) - 2x*10^h + x^2      -> Bq = 10^h - 2x + Q
 *   (H5 - x)^2  = 25*10^(2h-2) - x*10^h + x^2   -> Bq = Q25 - x + Q
 * (exact, because the added terms are multiples of 10^h when h >= 2). */
static u64 H5, Q25, H5M11, PHM11; /* 5*10^(h-1), 25*10^(h-2), H5 mod 11, 10^h mod 11 */
static unsigned char REV2[100];    /* REV2[10a+b] = 10b+a  (reverses a 2-digit pair) */
static u64 ES;       /* E' mod 10^h: E' = -1 (mod 2^h), 1 (mod 5^h); (B E')^2 = B^2 (mod 10^h) */
static int USE_S2 = 1; /* 0 if A034822_NOS2 is set: disables the sigma2 symmetry */
static u64 CE[20]; /* 10^i * E' mod 10^h */

/* Everything do_root()/fast_root() need about one root: the root Bp itself,
 * Bq = floor(Bp^2/10^h), f ~ Bp*2^64/10^h, Bp mod 11, Bp mod 100, Bq mod 100. */
typedef struct { u64 Bp, Bq, f, b11, b100, q100; } rootp_t;

/*
 * root_params -- compute rootp_t for an arbitrary root directly (slow path).
 *   In:  Bp < 10^h.   Out: *r filled; f is exact floor(Bp*2^64/10^h).
 *   Used by s2_roots() (PARANOID cross-check of the incremental version).
 *   No error conditions for Bp < 10^h (all quotients fit in 64 bits).
 */
static inline void root_params(u64 Bp, rootp_t *r)
{
    const int h = G.h;
    r->Bp = Bp;
    r->Bq = udiv128_64((u128)Bp * Bp, POW10[h]);
    r->f = udiv128_64((u128)Bp << 64, POW10[h]);
    r->b11 = Bp % 11;
    r->b100 = Bp % 100;
    r->q100 = r->Bq % 100;
}
/* mulmod10h -- (a*b) mod 10^h for a, b < 10^h (exact, 128-bit product).  No error conditions. */
static inline u64 mulmod10h(u64 a, u64 b)
{
    u128 x = (u128)a * b;
    u64 q = udiv128_64(x, POW10[G.h]);
    return (u64)(x - (u128)q * POW10[G.h]);
}
/*
 * s2_roots -- the four sigma2 roots of B computed directly (reference version).
 *   In:  B < 10^h.   Out: r[0..3] = params of B2, B2 + H5, -B2, -B2 + H5
 *        (all mod 10^h), B2 = B*E' mod 10^h.  The order differs from
 *        s2_roots_inc(), which is why the PARANOID comparison matches as a set.
 *   Only used to cross-check s2_roots_inc() in PARANOID builds.
 */
static inline void s2_roots(u64 B, rootp_t r[4])
{
    const u64 M = POW10[G.h];
    u64 B2 = mulmod10h(B, ES);
    u64 H5v = M / 2;
    root_params(B2, &r[0]);
    root_params((B2 + H5v) % M, &r[1]);
    root_params(M - B2, &r[2]);
    root_params((M - B2 + H5v) % M, &r[3]);
}


/* DFS state after the low i digits of B are fixed (see dstep()):
 *   B    the prefix B_i (low i digits of B)
 *   Q    floor(B_i^2 / 10^i)
 *   P    S digits 0..i-1 placed at positions h-1 .. h-i (rest still zero)
 *   f    sum of digit * floor(2^64/10^(h-k)) ~ B_i * 2^64 / 10^h (from below,
 *        error < 9h units)
 *   b11  B_i mod 11
 *   B2f  B_i * E' mod 10^h (the full sigma2 image of the prefix)
 *   B2, Q2, f2, b211   the same quantities as B, Q, f, b11 for the low i
 *        digits of the sigma2 image (they are determined by B_i because
 *        multiplication by E' preserves digit prefixes). */
typedef struct { u64 B, Q, P, f, b11, B2f, B2, Q2, f2, b211; } dstate_t;

/*
 * s2_roots_inc -- the four sigma2 roots of a leaf from the DFS state, without
 * any division.
 *   In:  vp -> dstate_t of a complete leaf (i = h), passed as void* so the leaf
 *        functions can take it before dstate_t's helpers are defined.
 *   Out: r[0..3] = params of x, x + H5, 10^h - x, H5 - x, where x is the sigma2
 *        image B2 reduced below H5 (if B2 >= H5, subtract H5 and adjust Q, f,
 *        b11 with the H5 identities above).  These four are the sigma2 orbit
 *        half of the leaf's 8 roots.
 *   Notes: f values wrap modulo 2^64 on purpose: f(10^h - x) = 2^64 - f(x),
 *        f(x + H5) = f(x) + 2^63, because H5/10^h = 1/2.
 *   Errors: none at run time; PARANOID builds compare against s2_roots() and
 *        abort on mismatch.
 */
static inline void s2_roots_inc(const void *vp, rootp_t r[4])
{
    const dstate_t *n = (const dstate_t *)vp;
    const u64 M = POW10[G.h];
    u64 x = n->B2, Qx = n->Q2, fx = n->f2, bx = n->b211;
    if (x >= H5) { Qx = Qx - x + Q25; x -= H5; fx -= 1ull << 63; bx = (bx + 11 - H5M11) % 11; }
    r[0].Bp = x;          r[0].Bq = Qx;              r[0].f = fx;                  r[0].b11 = bx;
    r[1].Bp = x + H5;     r[1].Bq = Qx + x + Q25;    r[1].f = fx + (1ull << 63);   r[1].b11 = (bx + H5M11) % 11;
    r[2].Bp = M - x;      r[2].Bq = M - 2 * x + Qx;  r[2].f = 0 - fx;              r[2].b11 = (PHM11 + 11 - bx) % 11;
    r[3].Bp = H5 - x;     r[3].Bq = Q25 - x + Qx;    r[3].f = (1ull << 63) - fx;   r[3].b11 = (H5M11 + 11 - bx) % 11;
    for (int j = 0; j < 4; j++) { r[j].b100 = r[j].Bp % 100; r[j].q100 = r[j].Bq % 100; }
#ifdef PARANOID
    {
        rootp_t q[4];
        s2_roots(n->B, q);
        for (int j = 0; j < 4; j++) {
            int ok = 0;
            for (int k = 0; k < 4; k++)
                if (q[k].Bp == r[j].Bp && q[k].Bq == r[j].Bq && q[k].b11 == r[j].b11 &&
                    (q[k].f - r[j].f + 64) < 128) ok = 1;
            if (!ok) { fprintf(stderr, "PARANOID: s2 inc root %d mismatch B=%llu\n", j, (unsigned long long)n->B); abort(); }
        }
    }
#endif
}

/*
 * test_cand -- test one candidate n = t*10^h + Bp (THEORY section 6).
 *   In:  t        candidate high part (inside the window, right class mod 11)
 *        Bp, Bq   the root and floor(Bp^2/10^h)
 *        f        ~ Bp*2^64/10^h (for the stage-1 estimate)
 *        P        top h digits implied by the leaf
 *        W0       P*10^e (128-bit)
 *        low2     (Bq + 2*t*Bp) mod 100 = S digits h (units) and h+1 (tens)
 *        st       per-thread statistics
 *   Out: none; if n^2 is a palindrome of exactly L digits, report(n) is called
 *        and st->found incremented.  st->pass / st->pass3 count the exact stages.
 *   Stages: 1) 128-bit filter with SLACK (also run by do_root; fast_root has
 *        already applied a 64-bit version of it); 2) exact check of the x digit
 *        pairs after P; 3) exact check of S digits h+x .. min(2h-1, L-1-2h)
 *        against their mirrors in Y = floor(S/10^(2h)); 4) full decimal check.
 *   Errors: none.  Candidates just outside the exact window (the window is
 *        widened by EPS) simply fail stage 1 or 2.
 */
static inline void test_cand(u64 t, u64 Bp, u64 Bq, u64 f, u64 P, u128 W0, u64 low2, stats_t *st)
{
    /* y = the x digits that must follow P at the top of S: the low-side digits
     * s_h, s_{h+1} reversed (x = 2), or just s_h (x = 1). */
    u64 y = (G.x == 2) ? REV2[low2] : low2 % 10;
    /* top of S below 10^(2h) must lie in [W, W + 10^(e-x)) for a hit */
    u128 W = W0 + (u128)y * G.pex;
    /* V ~ floor(S/10^(2h)) = t^2 + floor(C/10^h), estimated with f */
    u128 V = (u128)t * t + umulh(2 * t, f);
#ifdef PARANOID
    {
        u64 want = P * POW10[G.x] + y;
        int ex = exact_top_ok(t, Bp, Bq, want);
        int fl = (V + SLACK >= W && V < W + G.pex + SLACK);
        if (ex && !fl) { fprintf(stderr, "PARANOID: filter rejected exact hit t=%llu B=%llu\n",
                                 (unsigned long long)t, (unsigned long long)Bp); abort(); }
    }
#endif
    /* stage 1: outside the band [W - SLACK, W + 10^(e-x) + SLACK) => not a hit */
    if (__builtin_expect(V + SLACK < W || V >= W + G.pex + SLACK, 1)) return;
    /* exact: S = Y*10^(2h) + c0*10^h + (Bp^2 mod 10^h), with
     * C = 2 t Bp + Bq = X*10^h + c0 and Y = t^2 + X.  Quotients fit in 64 bits
     * because C < 2*10^(D-h)*10^h and Y < 10^(L-2h) <= 10^38. */
    const int h = G.h, L = G.L;
    u128 C = (u128)2 * t * Bp + Bq;
    u64 X = udiv128_64(C, POW10[h]);
    u64 c0 = (u64)(C - (u128)X * POW10[h]);
    u128 Y = (u128)t * t + X;
    /* stage 2: exact top h+x digits of S == P followed by y */
    if (udiv128_64(Y, G.pex) != P * POW10[G.x] + y) return;
    st->pass++;
    /* stage 3: S digit p (h <= p < 2h, in c0) mirrors S digit L-1-p = Y digit L-1-p-2h.
     * Only the low 19 digits of Y (Ym) are needed; digit pairs that fall
     * outside Ym (q > 18) or inside c0 on both sides are left to stage 4. */
    {
        u64 yq = udiv128_64(Y, 10000000000000000000ull);
        u64 Ym = (u64)(Y - (u128)yq * 10000000000000000000ull);
        int pmax = 2 * h - 1;
        if (pmax > L - 1 - 2 * h) pmax = L - 1 - 2 * h;
        for (int p = h + G.x; p <= pmax; p++) {
            int q = L - 1 - p - 2 * h;
            if (q > 18) continue;
            if ((c0 / POW10[p - h]) % 10 != (Ym / POW10[q]) % 10) return;
        }
    }
    st->pass3++;
    (void)0;
    /* stage 4: full decimal check (also guards the length: S must have L digits) */
    u128 n = (u128)t * POW10[G.h] + Bp;
    char ss[128];
    int len = square_digits(n, ss);
    if (len == G.L && is_pal(ss, len)) { st->found++; report(n); }
}

/*
 * do_root -- generic path: test every candidate t of one root.
 *   In:  Bp, Bq, f   root parameters (see rootp_t)
 *        b100, q100  Bp mod 100 and Bq mod 100 (for the low digits)
 *        P, W0       leaf's top digits and P*10^e
 *        t0, thi     first candidate (first t >= tlo in the root's class mod 11,
 *                    or tlo for odd L) and the window's upper end (inclusive)
 *        step        11 for even L, 1 for odd L
 *        st          statistics
 *   Out: calls test_cand() for each t = t0, t0+step, ... <= thi.
 *   The low digits are updated incrementally: low2(t + step) =
 *        low2(t) + 2*Bp*step (mod 100), and 2*Bp*t mod 100 only depends on
 *        Bp mod 100 and t mod 100.  No error conditions.
 */
static inline void do_root(u64 Bp, u64 Bq, u64 f, u64 b100, u64 q100, u64 P, u128 W0,
                           u64 t0, u64 thi, u64 step, stats_t *st)
{
#ifdef NOROOT
    st->cands += Bp + Bq + f + b100 + q100 + P + (u64)W0 + t0 + thi + step; return;
#endif
    u64 tm = t0 % 100;
    u64 low2 = (q100 + 2 * b100 * tm) % 100;
    u64 dlow = (2 * b100 * step) % 100;
    for (u64 t = t0; t <= thi; t += step) {
        st->cands++;
        test_cand(t, Bp, Bq, f, P, W0, low2, st);
        low2 += dlow;
        if (low2 >= 100) low2 -= 100;
    }
}

/*
 * do_leaf -- generic leaf processing (odd L, small h, or wide windows).
 *   In:  B    complete canonical root (h digits, unit 1..5, top digit 0..4)
 *        P    top h digits of S implied by B (reverse of B^2 mod 10^h)
 *        Q    floor(B^2/10^h);  f ~ B*2^64/10^h;  b11 = B mod 11
 *        s2p  NULL, or the leaf's dstate_t if the sigma2 roots must be tested
 *        bs   window base of the enclosing subtree (make_base())
 *        st   statistics
 *   Out: computes the t-window [tlo, thi] from the Taylor polynomial, then
 *        runs do_root() for the 4 roots B, B+H5, 10^h-B, H5-B and, if s2p is
 *        set, the 4 sigma2 roots.
 *   Errors: none at run time.  PARANOID builds abort if the approximate window
 *        does not contain the exact isqrt window.  If the lower window end is
 *        negative (only possible in tiny test cases) it is clamped to 0.
 */
static inline void do_leaf(u64 B, u64 P, u64 Q, u64 f, u64 b11, const void *s2p, const base_t *bs, stats_t *st)
{
    const int h = G.h;
    st->nodes++;
    /* window: t in [Rb + floor(v(u) - EPS), Rb + floor(v(u+1) + EPS)] */
    double u = (double)(P - bs->Pb);
    double vlo = bs->frac + u * bs->s1 - u * u * bs->s2;
    double u1 = u + 1.0;
    double vhi = bs->frac + u1 * bs->s1 - u1 * u1 * bs->s2;
    i64 a = (i64)floor(vlo - EPS), b = (i64)floor(vhi + EPS);
    u64 tlo = bs->Rb + (u64)a, thi = bs->Rb + (u64)b;
    if (a < 0 && (u64)(-a) > bs->Rb) tlo = 0;
#ifdef PARANOID
    {
        u64 elo = isqrt128((u128)P * G.pe), ehi = isqrt128((u128)(P + 1) * G.pe);
        if (tlo > elo || thi < ehi) {
            fprintf(stderr, "PARANOID: window [%llu,%llu] misses exact [%llu,%llu] P=%llu\n",
                    (unsigned long long)tlo, (unsigned long long)thi, (unsigned long long)elo,
                    (unsigned long long)ehi, (unsigned long long)P);
            abort();
        }
    }
#endif
    u128 W0 = (u128)P * G.pe;
    u64 b100 = B % 100, q100 = Q % 100, nb100 = (100 - b100) % 100;
    /* roots: B, B + H5, 10^h - B, H5 - B; floor(root^2/10^h) mod 100 below
     * (Q25 and 10^h vanish mod 100 for h >= 4; general h handled by % 100). */
    u64 BqA = Q, BqB = Q + B + Q25, BqC = POW10[h] - 2 * B + Q, BqD = Q25 - B + Q;
    u64 rA = b11, rB = (b11 + H5M11) % 11, rC = (PHM11 + 11 - b11) % 11, rD = (H5M11 + 11 - b11) % 11;
    u64 step = 1, tA = tlo, tB = tlo, tC = tlo, tD = tlo;
    if (G.mod11) {
        /* t*10^h + root = 0 (mod 11), 10 = -1 (mod 11): t = -(-1)^h root */
        u64 tm = tlo % 11;
        step = 11;
        if (!(h & 1)) { rA = (11 - rA) % 11; rB = (11 - rB) % 11; rC = (11 - rC) % 11; rD = (11 - rD) % 11; }
        tA = tlo + (rA + 11 - tm) % 11;
        tB = tlo + (rB + 11 - tm) % 11;
        tC = tlo + (rC + 11 - tm) % 11;
        tD = tlo + (rD + 11 - tm) % 11;
    }
    /* f for the partner roots, using H5/10^h = 1/2 (wraps mod 2^64 on purpose):
     * f(B+H5) = f + 2^63, f(10^h-B) = -f, f(H5-B) = 2^63 - f */
    do_root(B, BqA, f, b100, BqA % 100, P, W0, tA, thi, step, st);
    do_root(B + H5, BqB, f + (1ull << 63), b100, BqB % 100, P, W0, tB, thi, step, st);
    do_root(POW10[h] - B, BqC, 0 - f, nb100, BqC % 100, P, W0, tC, thi, step, st);
    do_root(H5 - B, BqD, (1ull << 63) - f, nb100, BqD % 100, P, W0, tD, thi, step, st);
    (void)q100;
    /* the sigma2 half of the orbit (same window, same P) */
    if (s2p) {
        rootp_t r[4];
        s2_roots_inc(s2p, r);
        for (int j = 0; j < 4; j++) {
            u64 tj = tlo;
            if (G.mod11) {
                u64 rr = (h & 1) ? r[j].b11 : (11 - r[j].b11) % 11;
                tj = tlo + (rr + 11 - tlo % 11) % 11;
            }
            do_root(r[j].Bp, r[j].Bq, r[j].f, r[j].b100, r[j].q100, P, W0, tj, thi, step, st);
        }
    }
}

static int FAST; /* even L, h >= 4, window <= 22: branchless two-candidate path */

/* addmod/submod -- (a + b) mod m and (a - b) mod m for a, b already < m.
 * Cheaper than %; results are wrong if an input is >= m. */
static inline u64 addmod(u64 a, u64 b, u64 m) { u64 c = a + b; return c >= m ? c - m : c; }
static inline u64 submod(u64 a, u64 b, u64 m) { return a >= b ? a - b : a + m - b; }

static u64 P4; /* 10^(e-4), 0 if e < 4 or x != 2 */

/*
 * stageB -- second filter in the fast path: compare 4 digit pairs instead of 2.
 *   In:  t, Bp, Bq   candidate and root
 *        D           the 64-bit stage-1 value (top of S below P, approximate
 *                    by at most SLACK)
 *        st          statistics (st->filt counts calls)
 *   Out: 0 = certainly not a palindrome; 1 = pass on to test_cand().
 *   How: low4 = (Bq + 2 t Bp) mod 10^4 = S digits h..h+3; the matching top
 *        digits are floor(D / 10^(e-4)).  If D is within SLACK of a multiple of
 *        10^(e-4) the approximation could flip a digit, so it returns 1
 *        (undecided) and lets the exact stage decide.  Returns 1 when stage B
 *        is disabled (P4 = 0: small e or x = 1).  No error conditions.
 */
static inline int stageB(u64 t, u64 Bp, u64 Bq, u64 D, stats_t *st)
{
    st->filt++;
    if (!P4) return 1;
    u64 low4 = (Bq % 10000 + 2 * (Bp % 10000) * (t % 10000)) % 10000;
    u64 y4 = D / P4, rem = D - y4 * P4;
    if (rem < SLACK || rem > P4 - SLACK) return 1;
    return y4 == 100u * REV2[low4 % 100] + REV2[low4 / 100];
}

/* Fast path.  D = t^2 + floor(2 t f / 2^64) - P*10^e is the part of
 * floor(S/10^(2h)) below the P prefix; a true hit has D in [y pex, (y+1) pex)
 * with (y+1) pex <= 10^e < 2^64, so D mod 2^64 is exact there and the filter
 * runs in wrapping 64-bit arithmetic (false positives go to the exact test).
 *
 * fast_root -- evaluate the (at most two) candidates of one root, branch-free.
 *   In:  Bp, Bq, f, b100, q100   root parameters (rootp_t)
 *        P, W0                   leaf's top digits and P*10^e (128-bit)
 *        W0lo                    W0 mod 2^64
 *        tlo, thi                window [tlo, thi]
 *        T2                      tlo^2 mod 2^64
 *        k                       offset of the root's first candidate:
 *                                t0 = tlo + k is the first t >= tlo with the
 *                                right residue mod 11 (0 <= k <= 10)
 *        tm100                   tlo mod 100
 *        st                      statistics
 *   Out: for each of t0 and t1 = t0 + 11 that is <= thi and passes the 64-bit
 *        filter and stageB(), calls test_cand().  Relies on the window being
 *        at most 22 wide (setup() only enables FAST then), so no third
 *        candidate exists.
 *   Wrapping: sq0/sq1, D0/D1 and the comparison D + SLACK - y*pex <
 *        pex + 2*SLACK are all modulo 2^64; the comparison is the usual
 *        unsigned trick for "D in [y*pex - SLACK, (y+1)*pex + SLACK)".
 *   Errors: none at run time; PARANOID builds verify against the exact test.
 */
static inline void fast_root(u64 Bp, u64 Bq, u64 f, u64 b100, u64 q100, u64 P, u128 W0, u64 W0lo,
                             u64 tlo, u64 T2, u64 k, u64 tm100, u64 thi, stats_t *st)
{
    const u64 pex = G.pex;
    u64 t0 = tlo + k, t1 = t0 + 11;
    /* t^2 mod 2^64 from tlo^2: (tlo + k)^2 and then + 22 t0 + 121 for t0 + 11 */
    u64 sq0 = T2 + 2 * tlo * k + k * k;
    u64 sq1 = sq0 + 22 * t0 + 121;
    u64 D0 = sq0 + umulh(2 * t0, f) - W0lo;
    u64 D1 = sq1 + umulh(2 * t1, f) - W0lo;
    /* low digits: (Bq + 2 t Bp) mod 100 depends on Bp, t only mod 100 */
    u64 m = (2 * b100) % 100;
    u64 tt = tm100 + k;
    u64 l0 = (q100 + m * tt) % 100, l1 = (q100 + m * (tt + 11)) % 100;
    u64 y0 = (G.x == 2) ? REV2[l0] : l0 % 10, y1 = (G.x == 2) ? REV2[l1] : l1 % 10;
    /* h0/h1: candidate exists and passes the stage-1 band test */
    int h0 = (t0 <= thi) & (D0 + SLACK - y0 * pex < pex + 2 * SLACK);
    int h1 = (t1 <= thi) & (D1 + SLACK - y1 * pex < pex + 2 * SLACK);
#ifdef PARANOID
    {
        stats_t dummy = {0};
        for (u64 t = t0; t <= thi; t += 11) {
            u64 l = (q100 + m * ((tm100 + (t - tlo)) % 100)) % 100;
            u64 y = (G.x == 2) ? REV2[l] : l % 10;
            u128 W = W0 + (u128)y * pex;
            u128 C = (u128)2 * t * Bp + Bq;
            u64 X = udiv128_64(C, POW10[G.h]);
            u128 Y = (u128)t * t + X;
            int ex = (Y >= W && Y < W + pex);
            if (t > t1) { fprintf(stderr, "PARANOID: window wider than 22\n"); abort(); }
            int fl = (t == t0) ? h0 : h1;
            if (ex && !fl) { fprintf(stderr, "PARANOID: fast filter rejected exact hit\n"); abort(); }
        }
        (void)dummy;
    }
#endif
    if (__builtin_expect(h0 | h1, 0)) {
        if (h0 && stageB(t0, Bp, Bq, D0, st)) test_cand(t0, Bp, Bq, f, P, W0, l0, st);
        if (h1 && stageB(t1, Bp, Bq, D1, st)) test_cand(t1, Bp, Bq, f, P, W0, l1, st);
    }
}

/*
 * fast_leaf -- leaf processing for the fast path (even L, window <= 22).
 *   In/Out: same as do_leaf().  Computes the window once, then for each of the
 *        4 (or 8 with sigma2) roots derives its residue class offset k and its
 *        low-digit constants with add/subtract-mod helpers (no divisions except
 *        two mods of tlo), and calls fast_root().
 *   Notes: st->cands counts window width here (not candidates), so the
 *        "cands" totals of the two paths are not comparable.  Unlike do_leaf()
 *        there is no clamp on the lower window end: v(u) >= frac >= 0, so the
 *        offset floor(vlo - EPS) is >= -1 and Rb >= 1, hence tlo >= 0.
 *   Errors: none at run time; PARANOID builds check the window.
 */
static inline void fast_leaf(u64 B, u64 P, u64 Q, u64 f, u64 b11, const void *s2p, const base_t *bs, stats_t *st)
{
    const int h = G.h;
    st->nodes++;
    double u = (double)(P - bs->Pb);
    double vlo = bs->frac + u * bs->s1 - u * u * bs->s2;
    double u1 = u + 1.0;
    double vhi = bs->frac + u1 * bs->s1 - u1 * u1 * bs->s2;
    u64 tlo = bs->Rb + (u64)(i64)floor(vlo - EPS), thi = bs->Rb + (u64)(i64)floor(vhi + EPS);
#ifdef PARANOID
    {
        u64 elo = isqrt128((u128)P * G.pe), ehi = isqrt128((u128)(P + 1) * G.pe);
        if (tlo > elo || thi < ehi) { fprintf(stderr, "PARANOID: fast window\n"); abort(); }
    }
#endif
    st->cands += thi - tlo + 1; /* window size (candidates ~ 4/11 of this) */
    u128 W0 = (u128)P * G.pe;
    u64 W0lo = (u64)W0, T2 = tlo * tlo;
    u64 b100 = B % 100, nb100 = submod(0, b100, 100);
    u64 q100 = Q % 100;
    /* floor(root^2/10^h) mod 100 for B+H5, 10^h-B, H5-B (h >= 4) */
    u64 qB = addmod(q100, b100, 100);
    u64 qC = submod(q100, (2 * b100) % 100, 100);
    u64 qD = submod(q100, b100, 100);
    /* residue of each root mod 11, then the required t mod 11:
     * t*10^h + r = 0 (mod 11) and 10 = -1 give t = r (h odd) or -r (h even);
     * k = (that - tlo) mod 11 is the offset of the first candidate. */
    u64 tm11 = tlo % 11, tm100 = tlo % 100;
    u64 rA = b11, rB = addmod(b11, H5M11, 11), rC = submod(PHM11, b11, 11), rD = submod(H5M11, b11, 11);
    if (!(h & 1)) { rA = submod(0, rA, 11); rB = submod(0, rB, 11); rC = submod(0, rC, 11); rD = submod(0, rD, 11); }
    u64 kA = submod(rA, tm11, 11), kB = submod(rB, tm11, 11), kC = submod(rC, tm11, 11), kD = submod(rD, tm11, 11);
    fast_root(B, Q, f, b100, q100, P, W0, W0lo, tlo, T2, kA, tm100, thi, st);
    fast_root(B + H5, Q + B + Q25, f + (1ull << 63), b100, qB, P, W0, W0lo, tlo, T2, kB, tm100, thi, st);
    fast_root(POW10[h] - B, POW10[h] - 2 * B + Q, 0 - f, nb100, qC, P, W0, W0lo, tlo, T2, kC, tm100, thi, st);
    fast_root(H5 - B, Q25 - B + Q, (1ull << 63) - f, nb100, qD, P, W0, W0lo, tlo, T2, kD, tm100, thi, st);
    if (s2p) {
        rootp_t r[4];
        s2_roots_inc(s2p, r);
        for (int j = 0; j < 4; j++) {
            u64 rr = r[j].b11;
            if (!(h & 1)) rr = submod(0, rr, 11);
            u64 kk = submod(rr, tm11, 11);
            fast_root(r[j].Bp, r[j].Bq, r[j].f, r[j].b100, r[j].q100, P, W0, W0lo, tlo, T2, kk, tm100, thi, st);
        }
    }
}

static u64 F64[20];  /* floor(2^64 / 10^k); F64[0] = 2^64 - 1 (never used: k = h - i >= 1) */
static u64 P11[20];  /* 10^k mod 11 */

/*
 * s2_level -- does the sigma2 restriction apply to digit i?
 *   In:  i (DFS level = position of the digit about to be chosen),
 *        Bi = B mod 10^i (the digits already chosen), unit5 = unit digit is 5.
 *   Out: 1 if digit i must be restricted to 0..4, else 0.
 *   Why: sigma2 adds 5 to digit v2(B)+1 and leaves lower digits unchanged.
 *        v2(B) is determined by B mod 2^(v+1), hence by B mod 10^(v+1); so at
 *        level i the restriction applies exactly when ctz(Bi) == i-1 (Bi's
 *        low bits are B's low bits because 10^i is a multiple of 2^i).
 *        Not applied for unit digit 5 (its tens digit already serves the
 *        negation symmetry) or for i > h-2 (sigma2 would collide with the H5
 *        restriction on the top digit, or the images coincide), or when
 *        sigma2 is disabled (USE_S2 = 0).
 */
static inline int s2_level(int i, u64 Bi, int unit5)
{
    return USE_S2 && !unit5 && i >= 1 && i <= G.h - 2 && Bi != 0 && __builtin_ctzll(Bi) == i - 1;
}


/*
 * dstep -- append digit d at position i to the DFS state (THEORY section 2).
 *   In:  a     state for digits 0..i-1;  i, d (0..9)
 *   Out: n     state for digits 0..i; *s_out = S digit i (0..9)
 *   How: v = Q + 2 d B + d^2 10^i gives S digit i = v mod 10 and the new
 *        Q = floor(v/10).  The sigma2 image is advanced the same way: its
 *        full value B2f += d * (10^i E' mod 10^h), its digit i is read off B2f,
 *        and that digit is appended to B2/Q2/f2/b211 like d to B/Q/f/b11.
 *   Overflow: v < 10^(i+2) and B2f + d*CE[i] < 10^(h+1) for h <= 18 fit in u64.
 *   Errors: none at run time; PARANOID checks that the sigma2 image produces
 *        the same S digit (it must, because B2^2 = B^2 mod 10^h).
 */
static inline void dstep(const dstate_t *a, dstate_t *n, int i, u64 d, u64 *s_out)
{
    const int h = G.h;
    u64 v = a->Q + 2 * d * a->B + d * d * POW10[i];
    u64 s = v % 10;
    n->P = a->P + s * POW10[h - 1 - i];
    n->B = a->B + d * POW10[i];
    n->Q = v / 10;
    n->f = a->f + d * F64[h - i];
    n->b11 = (a->b11 + d * P11[i]) % 11;
    u64 b2f = a->B2f + d * CE[i];
    b2f %= POW10[h];
    n->B2f = b2f;
    u64 d2 = (b2f / POW10[i]) % 10;
    u64 v2 = a->Q2 + 2 * d2 * a->B2 + d2 * d2 * POW10[i];
    n->B2 = a->B2 + d2 * POW10[i];
    n->Q2 = v2 / 10;
    n->f2 = a->f2 + d2 * F64[h - i];
    n->b211 = (a->b211 + d2 * P11[i]) % 11;
#ifdef PARANOID
    if (v2 % 10 != s) { fprintf(stderr, "PARANOID: B2 digit mismatch\n"); abort(); }
#endif
    *s_out = s;
}

/*
 * leaf_dispatch -- hand a complete leaf to the fast or generic leaf routine.
 *   In:  n (complete state, i = h), s2 = test the sigma2 roots too,
 *        bs = window base, st = statistics.   No error conditions.
 */
static inline void leaf_dispatch(const dstate_t *n, int s2, const base_t *bs, stats_t *st)
{
    if (FAST) fast_leaf(n->B, n->P, n->Q, n->f, n->b11, s2 ? n : NULL, bs, st);
    else do_leaf(n->B, n->P, n->Q, n->f, n->b11, s2 ? n : NULL, bs, st);
}

/*
 * dfs -- enumerate all canonical B extending a prefix, depth-first.
 *   In:  i      number of digits already fixed (p <= i <= h-1)
 *        a      state after those digits
 *        unit5  the unit digit is 5
 *        s2     the sigma2 restriction has already been applied at a lower level
 *        bs     window base (NULL until depth IB is reached)
 *        st     statistics
 *   Out: visits every leaf below; leaves are passed to leaf_dispatch().
 *   Digit limits at level i: 0..9, except 0..4 at the top digit (i = h-1),
 *        at the tens digit for unit 5, and at the sigma2 digit (s2_level()).
 *   Recursion depth is at most h - p (<= 17); the base is created on this
 *        function's stack at depth IB and shared by the whole subtree.
 *   Errors: none.
 */
static void dfs(int i, const dstate_t *a, int unit5, int s2, const base_t *bs, stats_t *st)
{
    const int h = G.h;
    base_t local;
    /* at depth IB, P's remaining free digits span < 10^(h-IB): build the window base */
    if (i == IB) { make_base(a->P, &local); bs = &local; }
    int dmax = 9;
    if (i == h - 1) dmax = 4;             /* top digit < 5: partner B + 5*10^(h-1) */
    if (i == 1 && unit5) dmax = 4;        /* unit 5: tens < 5, partner is -B */
    int s2n = s2;
    if (s2_level(i, a->B, unit5)) { dmax = 4; s2n = 1; }
    for (u64 d = 0; d <= (u64)dmax; d++) {
        dstate_t n;
        u64 s;
        dstep(a, &n, i, d, &s);
        if (i == h - 1) {
            /* IB == h (tiny test lengths): an exact base for every leaf */
            if (IB == h) { make_base(n.P, &local); bs = &local; }
            leaf_dispatch(&n, s2n, bs, st);
        } else {
            dfs(i + 1, &n, unit5, s2n, bs, st);
        }
    }
}

/* Task numbering: task k fixes the low p digits of B.  The unit digit is
 * u = 1 + k / 10^(p-1) (1..5) and digits 1..p-1 are the decimal digits of
 * rest = k mod 10^(p-1), lowest first (digit i = (rest / 10^(i-1)) mod 10). */

/*
 * task_valid -- is task k a canonical prefix?
 *   In:  k < G.ntasks.
 *   Out: 0 if the prefix violates a digit restriction (unit 5 with tens > 4,
 *        or a sigma2-restricted digit > 4), else 1.  Invalid tasks are marked
 *        done before the run starts, so they never execute.
 */
static int task_valid(u64 k)
{
    u64 per = POW10[G.p - 1];
    u64 u = 1 + k / per, rest = k % per;
    int unit5 = (u == 5);
    if (unit5 && G.p >= 2 && (rest % 10) > 4) return 0;
    u64 Bi = 0;
    for (int i = 0; i < G.p; i++) {
        u64 d = (i == 0) ? u : (rest / POW10[i - 1]) % 10;
        if (s2_level(i, Bi, unit5) && d > 4) return 0;
        Bi += d * POW10[i];
    }
    return 1;
}

/*
 * run_task -- process task k: replay its p prefix digits, then dfs().
 *   In:  k (task number), st (statistics for this task).
 *   Out: all palindromic squares whose canonical B has this prefix are reported.
 *   Returns immediately for invalid tasks.  No error conditions.
 */
static void run_task(u64 k, stats_t *st)
{
    if (!task_valid(k)) return;
    u64 per = POW10[G.p - 1];
    u64 u = 1 + k / per, rest = k % per;
    int unit5 = (u == 5), s2 = 0;
    dstate_t a = {0}, n;
    for (int i = 0; i < G.p; i++) {
        u64 d = (i == 0) ? u : (rest / POW10[i - 1]) % 10, sd;
        if (s2_level(i, a.B, unit5)) s2 = 1;
        dstep(&a, &n, i, d, &sd);
        a = n;
    }
    dfs(G.p, &a, unit5, s2, NULL, st);
}

/* ---------- checkpointing ---------- */

/* State file format: one text header line
 *   "A034822 L=.. h=.. p=.. ntasks=.. lo=.. hi=.. found=.. nodes=.. cands=.. pass=.. filt=.."
 * followed by ntasks raw bytes (done_map: 1 = task finished or invalid).
 * filt= is optional when reading (files written before it was added). */
static const char *state_file = NULL;

/*
 * save_state -- write the checkpoint (no-op without -S).
 *   Writes "<state>.tmp" and renames it over the state file, so a crash during
 *   the write leaves the previous checkpoint intact.  The header counters and
 *   done_map are copied under commit_lock (see "Shared run state"), so they
 *   describe exactly the committed tasks; found= is found_done.
 *   Errors: any failure (open, write, close/flush, rename) prints a warning on
 *   stderr, removes the temp file and keeps the previous checkpoint; the run
 *   itself continues.  A square found in a task that was in progress when the
 *   process died is printed again when that task reruns after a resume (the
 *   FOUND lines can repeat, the found= count does not).
 */
static void save_state(void)
{
    if (!state_file) return;
    char tmp[1024];
    snprintf(tmp, sizeof tmp, "%s.tmp", state_file);
    FILE *f = fopen(tmp, "wb");
    if (!f) { fprintf(stderr, "\nwarning: cannot write %s: %s\n", tmp, strerror(errno)); return; }
    pthread_mutex_lock(&commit_lock);
    int ok = fprintf(f, "A034822 L=%d h=%d p=%d ntasks=%" PRIu64 " lo=%" PRIu64 " hi=%" PRIu64
                        " found=%llu nodes=%llu cands=%llu pass=%llu filt=%llu\n",
                     G.L, G.h, G.p, G.ntasks, task_lo, task_hi,
                     (unsigned long long)atomic_load(&found_done), (unsigned long long)atomic_load(&nodes_total),
                     (unsigned long long)atomic_load(&cands_total), (unsigned long long)atomic_load(&pass_total),
                     (unsigned long long)atomic_load(&filt_total)) > 0;
    ok = ok && fwrite(done_map, 1, G.ntasks, f) == G.ntasks;
    pthread_mutex_unlock(&commit_lock);
    int err = errno;
    if (fclose(f) != 0 && ok) { ok = 0; err = errno; }
    if (ok && rename(tmp, state_file) != 0) { ok = 0; err = errno; }
    if (!ok) {
        fprintf(stderr, "\nwarning: checkpoint %s not updated: %s\n", state_file, strerror(err));
        remove(tmp);
    }
}

/*
 * load_state -- resume from the checkpoint, if -S was given and the file exists.
 *   Out: 1 if loaded (done_map and counters restored), 0 if there is no state
 *        file to load (fresh run).
 *   Errors (fatal, exit(1)): header unreadable or longer than 511 chars, or
 *        L, h, p, ntasks differ from this run's parameters; file shorter than
 *        ntasks bytes of done_map.
 *        The stored lo/hi range is NOT compared, so a resumed run may use a
 *        different -r range (done_map covers all task numbers).
 */
static int load_state(void)
{
    if (!state_file) return 0;
    FILE *f = fopen(state_file, "rb");
    if (!f) return 0;
    int L, h, p;
    u64 nt, lo, hi;
    unsigned long long fc, nd, cd, ps, fl = 0;
    char line[512];
    /* header line; "filt=" was added later and is optional (older files lack it) */
    if (!fgets(line, sizeof line, f) ||
        sscanf(line, "A034822 L=%d h=%d p=%d ntasks=%" SCNu64 " lo=%" SCNu64 " hi=%" SCNu64
                     " found=%llu nodes=%llu cands=%llu pass=%llu filt=%llu",
               &L, &h, &p, &nt, &lo, &hi, &fc, &nd, &cd, &ps, &fl) < 10 ||
        L != G.L || h != G.h || p != G.p || nt != G.ntasks) {
        fprintf(stderr, "state file %s does not match parameters\n", state_file);
        exit(1);
    }
    if (fread(done_map, 1, G.ntasks, f) != G.ntasks) { fprintf(stderr, "short state file\n"); exit(1); }
    fclose(f);
    atomic_store(&found_count, fc);
    atomic_store(&found_done, fc);
    atomic_store(&nodes_total, nd);
    atomic_store(&cands_total, cd);
    atomic_store(&pass_total, ps);
    atomic_store(&filt_total, fl);
    return 1;
}

/* ---------- threads ---------- */

/*
 * worker -- thread body: take task numbers from next_task until the range is
 * exhausted or stop_flag is set; run each unfinished task and commit it
 * (statistics, found_done, done_map, tasks_done) under commit_lock.  stop_flag
 * is only checked between tasks, so stopping (SIGINT/SIGTERM) lets the tasks
 * in progress finish (seconds each) and never records a half-done task.
 *   In: arg (unused).   Out: NULL.
 */
static void *worker(void *arg)
{
    (void)arg;
    for (;;) {
        if (stop_flag) break;
        u64 k = atomic_fetch_add(&next_task, 1);
        if (k >= task_hi) break;
        if (done_map[k]) continue;
        stats_t st = {0};
        run_task(k, &st);
        pthread_mutex_lock(&commit_lock);
        atomic_fetch_add(&nodes_total, st.nodes);
        atomic_fetch_add(&cands_total, st.cands);
        atomic_fetch_add(&pass_total, st.pass);
        atomic_fetch_add(&filt_total, st.filt);
        atomic_fetch_add(&found_done, st.found);
        done_map[k] = 1;
        atomic_fetch_add(&tasks_done, 1);
        pthread_mutex_unlock(&commit_lock);
    }
    return NULL;
}

/* on_sigint -- SIGINT/SIGTERM handler: request a clean stop (async-signal-safe). */
static void on_sigint(int s) { (void)s; stop_flag = 1; }

/* now_sec -- monotonic wall-clock time in seconds (for rates and save intervals). */
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/*
 * setup -- choose and validate all run parameters and derived constants.
 *   In:  L (digits of S); h (split point, <= 0 = choose default D/2);
 *        p (task depth, <= 0 = choose default h-6 capped at 7).
 *   Out: 1 on success with G, H5, Q25, H5M11, PHM11, ES, CE[], REV2[], F64[],
 *        P11[], FAST, P4, IB set; 0 (after printing a message) if no valid h.
 *   Valid h (each bound keeps an intermediate value inside 64/128 bits):
 *        h >= D - 18      t < 10^(D-h) <= 10^18, so 2t, t^2 and C fit
 *        h >= (L-37)/2    P*10^e < 10^(L-2h) <= 10^38 < 2^127 (isqrt128 input)
 *        h >= 3           the B / H5 / tens-digit restrictions need distinct digits
 *        h <= (L-2)/3     e >= 2, so the x checked digit pairs lie between the
 *                         matched outer digits
 *        h <= 18          B < 10^18 and B^2 < 2^128
 *   Other choices:
 *        x = 2 extra digit pairs if h <= 17 (top h+2 digits < 10^19 fit a u64),
 *            else 1;  e must also be <= 19 so 10^e fits a u64.
 *        p is forced into [1, h-2] so the top digit is always chosen in dfs().
 *        FAST requires even L, h >= 4 and a window bound wmax <= 22.
 *        P4 (stage B) only when x = 2 and e >= 9, so 10^(e-4) >> SLACK.
 *        IB = smallest depth >= p where the Taylor cubic term
 *            Rb*U^3/(16 Pb^3) < 1e-8 (U = 10^(h-IB), Pb >= 10^(h-1),
 *            Rb <= 10^((h+e)/2)) and h - IB <= 7; IB = h (exact base per leaf)
 *            if none qualifies.
 *        E' is found by stepping e = 1, 1 + 5^h, 1 + 2*5^h, ... until
 *            e = -1 (mod 2^h): at most 2^h <= 262144 steps.
 *   Errors: returns 0 for an invalid h; exit(1) if the computed E' fails the
 *        self-check E'^2 = 1 (mod 10^h) (cannot happen for h <= 18).
 */
static int setup(int L, int h, int p)
{
    G.L = L;
    G.D = (L + 1) / 2;
    int hmin = G.D - 18;
    if ((L - 38 + 1) / 2 > hmin) hmin = (L - 38 + 1) / 2;
    if (hmin < 3) hmin = 3;
    int hmax = (L - 2) / 3;
    if (hmax > 18) hmax = 18;
    if (h <= 0) {
        h = G.D / 2;
        if (h < hmin) h = hmin;
        if (h > hmax) h = hmax;
    }
    if (h < hmin || h > hmax) {
        fprintf(stderr, "L=%d: need %d <= h <= %d (got %d)\n", L, hmin, hmax, h);
        return 0;
    }
    G.h = h;
    G.e = L - 3 * h;
    G.x = (h <= 17) ? 2 : 1;
    if (G.e - G.x < 0 || G.e > 19) { fprintf(stderr, "bad e=%d\n", G.e); return 0; }
    G.pe = POW10[G.e];
    G.pex = POW10[G.e - G.x];
    G.mod11 = (L % 2 == 0);
    if (p <= 0) {
        p = h - 6; /* ~1e6 leaves per task */
        if (p > 7) p = 7;
        if (p < 1) p = 1;
    }
    if (p >= h - 1) p = h - 2;
    if (p < 1) p = 1;
    G.p = p;
    G.ntasks = 5 * POW10[p - 1];
    H5 = 5 * POW10[h - 1];
    Q25 = 25 * POW10[h - 2];
    H5M11 = H5 % 11;
    PHM11 = POW10[h] % 11;
    {
        /* E' = -1 (mod 2^h), 1 (mod 5^h) by CRT */
        u64 m2 = 1ull << h, m5 = POW10[h] >> h, M = POW10[h];
        u64 e = 1;
        while ((e % m2) != m2 - 1) e += m5; /* e = 1 (mod 5^h) stepping until = -1 (mod 2^h) */
        ES = e % M;
        for (int k = 0; k < h; k++) CE[k] = (u64)(((u128)POW10[k] * ES) % M);
        if ((u64)(((u128)ES * ES) % M) != 1) { fprintf(stderr, "bad E'\n"); exit(1); }
    }
    for (int k = 0; k < 100; k++) REV2[k] = (unsigned char)(10 * (k % 10) + k / 10);
    for (int k = 0; k < 20; k++) {
        F64[k] = k == 0 ? UINT64_MAX : (u64)(((u128)1 << 64) / POW10[k]);
        P11[k] = POW10[k] % 11;
    }
    /* base depth: cubic Taylor remainder Rb*U^3/(16 Pb^3) < 1e-8 with
     * U = 10^(h-IB), Pb >= 10^(h-1), Rb <= 10^((h+e)/2); keep U <= 1e7 */
    {
        /* widest t-window: sqrt(10^e / 10^(h-1)) / 2 + margins */
        double wmax = sqrt(pow(10.0, G.e - h + 1)) / 2.0 + 3.0;
        FAST = G.mod11 && h >= 4 && wmax <= 22.0;
        P4 = (G.x == 2 && G.e >= 4 + 5) ? POW10[G.e - 4] : 0; /* need 10^(e-4) >> SLACK */
    }
    IB = h;
    for (int ib = p; ib <= h; ib++) {
        double lg = (h + G.e) / 2.0 + 3.0 * (1 - ib) - log10(16.0);
        if (lg < -8 && h - ib <= 7) { IB = ib; break; }
    }
    return 1;
}

/*
 * run_search -- run one complete (or ranged) search for length L.
 *   In:  L, h, p      as for setup() (0 = defaults)
 *        threads      number of worker threads (>= 1)
 *        interval     seconds between checkpoints (<= 0: only at the end)
 *        ra, rb       task range [ra, rb); rb = 0 means "to the end"; rb is
 *                     clamped to ntasks
 *   Out: returns the number of palindromic squares in committed tasks
 *        (including those restored from the state file); prints FOUND lines as they are found
 *        and a final "DONE ..." line, or "PARTIAL ..." if the run was stopped
 *        or the range did not cover all tasks.  The main thread only monitors:
 *        it prints progress every 0.2 s (unless quiet), saves the checkpoint
 *        every `interval` seconds, and returns once all tasks in the range are
 *        done or a stop was requested.
 *   Errors: exit(1) if setup() rejects the parameters, the task range is
 *        empty, the state file does not match, memory allocation or thread
 *        creation fails.  In quiet mode with interval <= 0 (selftest) the
 *        monitor loop is skipped and pthread_join does the waiting.
 */
static u64 run_search(int L, int h, int p, int threads, double interval, u64 ra, u64 rb)
{
    if (!setup(L, h, p)) exit(1);
    task_lo = ra;
    task_hi = rb ? rb : G.ntasks;
    if (task_hi > G.ntasks) task_hi = G.ntasks;
    if (task_lo >= task_hi) {
        fprintf(stderr, "empty task range [%" PRIu64 ",%" PRIu64 ") (L=%d has %" PRIu64 " tasks)\n",
                task_lo, task_hi, L, G.ntasks);
        exit(1);
    }
    done_map = calloc(G.ntasks, 1);
    pthread_t *th = malloc(sizeof(pthread_t) * threads);
    if (!done_map || !th) { fprintf(stderr, "out of memory\n"); exit(1); }
    atomic_store(&found_count, 0);
    atomic_store(&found_done, 0);
    atomic_store(&tasks_done, 0);
    atomic_store(&nodes_total, 0);
    atomic_store(&cands_total, 0);
    atomic_store(&pass_total, 0);
    atomic_store(&filt_total, 0);
    if (load_state() && !quiet) fprintf(stderr, "resumed from %s\n", state_file);
    /* invalid (non-canonical) prefixes count as done from the start, so progress
     * and completion checks only need done_map */
    for (u64 k = 0; k < G.ntasks; k++)
        if (!task_valid(k)) done_map[k] = 1;
    /* tasks of this range still to run (done_map is not touched by anyone else yet) */
    u64 todo = 0;
    for (u64 k = task_lo; k < task_hi; k++) todo += !done_map[k];
    const u64 already = (task_hi - task_lo) - todo;
    atomic_store(&next_task, task_lo);
    if (!quiet)
        fprintf(stderr, "L=%d D=%d h=%d e=%d p=%d tasks=[%" PRIu64 ",%" PRIu64 ") of %" PRIu64
                        " mod11=%d threads=%d\n",
                G.L, G.D, G.h, G.e, G.p, task_lo, task_hi, G.ntasks, G.mod11, threads);
    signal(SIGINT, on_sigint);
    signal(SIGTERM, on_sigint);
    double t0 = now_sec(), last_save = t0;
    for (int i = 0; i < threads; i++)
        if (pthread_create(&th[i], NULL, worker, NULL) != 0) { fprintf(stderr, "pthread_create failed\n"); exit(1); }
    if (!quiet || interval > 0) {
        for (;;) {
            usleep(200000);
            u64 nd = atomic_load(&tasks_done);
            /* done: every remaining task committed, or a stop was requested
             * (pthread_join below then waits for the tasks in progress) */
            int finished = nd >= todo || stop_flag;
            double t = now_sec();
            if (!quiet) {
                double frac = (double)(already + nd) / (double)(task_hi - task_lo);
                double el = t - t0;
                fprintf(stderr, "\r%6.2f%%  nodes %.3e (%.3g/s)  cands %.3e  found %llu  elapsed %.0fs  eta %.0fs   ",
                        100 * frac, (double)atomic_load(&nodes_total), atomic_load(&nodes_total) / el,
                        (double)atomic_load(&cands_total), (unsigned long long)atomic_load(&found_count), el,
                        frac > 0 ? el / frac - el : 0.0);
            }
            if (state_file && interval > 0 && t - last_save > interval) { save_state(); last_save = t; }
            if (finished) break;
        }
    }
    for (int i = 0; i < threads; i++) pthread_join(th[i], NULL);
    double el = now_sec() - t0;
    save_state();
    int complete = 1; /* all threads joined: done_map is stable */
    for (u64 k = task_lo; k < task_hi; k++) if (!done_map[k]) complete = 0;
    /* count = squares in committed tasks (exact across resumes); equals the
     * number of FOUND lines printed by this process when nothing was resumed */
    u64 fc = atomic_load(&found_done);
    if (!quiet) fprintf(stderr, "\n");
    printf("%s L=%d count=%" PRIu64 " tasks=[%" PRIu64 ",%" PRIu64 ") nodes=%llu cands=%llu filt=%llu pass=%llu time=%.2fs\n",
           complete ? "DONE" : "PARTIAL", L, fc, task_lo, task_hi, (unsigned long long)atomic_load(&nodes_total),
           (unsigned long long)atomic_load(&cands_total), (unsigned long long)atomic_load(&filt_total),
           (unsigned long long)atomic_load(&pass_total), el);
    fflush(stdout);
    free(th);
    free(done_map);
    return fc;
}

/* ---------- selftest ---------- */

/* cmp_u128 -- qsort comparator for u128 values (ascending). */
static int cmp_u128(const void *a, const void *b)
{
    u128 x = *(const u128 *)a, y = *(const u128 *)b;
    return x < y ? -1 : x > y;
}

/*
 * selftest -- validate the helpers and the whole search against A263618.
 *   In:  maxL (largest length to test), threads.
 *   Out: 0 if everything matches, 1 otherwise; one line per (L, h) on stderr.
 *   Steps: 2,000,000 random checks of udiv128_64 and isqrt128; then for every
 *        L = 11..maxL and two split points h = D/2 and D/2+1 (when valid),
 *        a full quiet search whose count must equal KNOWN[L] and whose roots
 *        must be distinct.  The second h exercises different window widths.
 *   Errors: maxL is clamped to 69 (KNOWN[] has no entries beyond that), with a
 *        message.  Run time grows ~10x
 *        per 4 lengths (maxL = 40 takes ~10 minutes on an M1 Pro, 10 threads).
 */
static int selftest(int maxL, int threads)
{
    int bad = 0;
    if (maxL > 69) {
        fprintf(stderr, "selftest: A263618 is only known for L <= 69; using maxL = 69\n");
        maxL = 69;
    }
    /* helpers */
    srand(12345);
    for (int i = 0; i < 2000000; i++) {
        u64 d = ((u64)rand() << 33 ^ (u64)rand() << 11 ^ (u64)rand()) % 1000000000000000000ull + 1;
        u64 q = ((u64)rand() << 33 ^ (u64)rand() << 11 ^ (u64)rand());
        u128 x = (u128)q * d + ((u64)rand() * 7919u) % d;
        if (udiv128_64(x, d) != q) { fprintf(stderr, "udiv fail\n"); return 1; }
        u128 y = (u128)q * q + (u64)rand();
        u64 r = isqrt128(y >> 1);
        if (!((u128)r * r <= (y >> 1) && (u128)(r + 1) * (r + 1) > (y >> 1))) { fprintf(stderr, "isqrt fail\n"); return 1; }
    }
    fprintf(stderr, "helpers ok\n");
    quiet = 1;
    for (int L = 11; L <= maxL; L++) {
        /* two different h where possible, to exercise the window logic */
        int D = (L + 1) / 2;
        for (int v = 0; v < 2; v++) {
            int h = D / 2 + v;
            params_t save;
            if (!setup(L, h, 0)) continue;
            save = G;
            (void)save;
            double t0 = now_sec();
            u64 c = run_search(L, h, 0, threads, 0, 0, 0);
            double el = now_sec() - t0;
            int ok = (int)c == KNOWN[L];
            /* all found are distinct genuine palindromic squares */
            u64 m = c < MAXFOUND ? c : MAXFOUND;
            qsort(found_list, m, sizeof(u128), cmp_u128);
            for (u64 i = 1; i < m; i++) if (found_list[i] == found_list[i - 1]) ok = 0;
            fprintf(stderr, "L=%2d h=%2d count=%5" PRIu64 " expected %5d %s  (%.2fs)\n", L, h, c, KNOWN[L],
                    ok ? "ok" : "MISMATCH", el);
            if (!ok) bad++;
        }
    }
    quiet = 0;
    fprintf(stderr, bad ? "SELFTEST FAILED (%d)\n" : "SELFTEST PASSED\n", bad);
    return bad != 0;
}

/*
 * main -- command-line dispatch (see USAGE in the header).
 *   "check n"            prints n^2 and whether it is a palindrome; exit 0
 *                        (exit 2 if n is not a digit string).
 *   "selftest [maxL [-t threads]]"   exit status from selftest().
 *   "search L [options]" runs run_search(); exit 0 even if no square is found
 *                        (read the DONE/PARTIAL line); exit 1 for L outside
 *                        11..76 or rejected parameters; exit 2 for bad options.
 *   Option values are parsed with atoi/atof/sscanf; the thread count must be
 *   >= 1 (exit 1 otherwise), other values are checked by setup()/run_search().
 */
int main(int argc, char **argv)
{
    POW10[0] = 1;
    for (int i = 1; i < 20; i++) POW10[i] = POW10[i - 1] * 10;
    int threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (getenv("A034822_NOS2")) USE_S2 = 0;
    if (argc < 2) goto usage;
    if (!strcmp(argv[1], "check") && argc == 3) {
        u128 n;
        if (!parse_u128(argv[2], &n)) goto usage;
        char ss[128];
        int len = square_digits(n, ss);
        printf("%s^2 = %s (%d digits) %s\n", argv[2], ss, len, is_pal(ss, len) ? "PALINDROME" : "not palindrome");
        return 0;
    }
    if (!strcmp(argv[1], "selftest")) {
        int maxL = argc >= 3 ? atoi(argv[2]) : 40;
        if (argc >= 5 && !strcmp(argv[3], "-t")) threads = atoi(argv[4]);
        if (threads < 1) { fprintf(stderr, "need at least one thread\n"); return 1; }
        return selftest(maxL, threads);
    }
    if (!strcmp(argv[1], "search") && argc >= 3) {
        int L = atoi(argv[2]), h = 0, p = 0;
        double interval = 300;
        u64 ra = 0, rb = 0;
        for (int i = 3; i < argc; i++) {
            if (!strcmp(argv[i], "-t") && i + 1 < argc) threads = atoi(argv[++i]);
            else if (!strcmp(argv[i], "-h") && i + 1 < argc) h = atoi(argv[++i]);
            else if (!strcmp(argv[i], "-p") && i + 1 < argc) p = atoi(argv[++i]);
            else if (!strcmp(argv[i], "-S") && i + 1 < argc) state_file = argv[++i];
            else if (!strcmp(argv[i], "-i") && i + 1 < argc) interval = atof(argv[++i]);
            else if (!strcmp(argv[i], "-r") && i + 1 < argc) {
                if (sscanf(argv[++i], "%" SCNu64 ":%" SCNu64, &ra, &rb) != 2) goto usage;
            } else goto usage;
        }
        if (L < 11 || L > 76) { fprintf(stderr, "L must be in 11..76\n"); return 1; }
        if (threads < 1) { fprintf(stderr, "need at least one thread\n"); return 1; }
        run_search(L, h, p, threads, interval, ra, rb);
        return 0;
    }
usage:
    fprintf(stderr,
            "usage: %s search L [-t threads] [-h H] [-p P] [-S state] [-i secs] [-r a:b]\n"
            "       %s selftest [maxL [-t threads]]\n"
            "       %s check n\n",
            argv[0], argv[0], argv[0]);
    return 2;
}
