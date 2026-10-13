/* hsearch2.c -- coordinate-scan local search for h-bases with large h-range (lower bounds / targets).
 * usage: hsearch2 -h H -k K -max MAXELEM -lim LIM -sec S [-j T] [-seed "a1 ... ak"]... [-kick N]
 * Starts from each seed (and scaled/perturbed copies), repeatedly scans every single coordinate over its
 * whole feasible interval (exact 1-D optimisation for intervals of up to 6000 values; longer intervals
 * get a strided coarse pass plus a local refinement), then applies random kicks and re-descends.
 * Prints every improvement "best <range>: a1 ... ak".
 *
 * Build: clang -O3 -mcpu=apple-m1 -std=gnu11 -pthread -o hsearch2 hsearch2.c      (on Linux add -lm)
 *
 * THEORY OF OPERATIONS
 * ====================
 *
 * 1. What is computed
 *    n_h(A) = the largest N such that 1..N are all sums of at most h elements of the basis A (a_1 = 1);
 *    n(h,k) = n_h(k) = its maximum over all k-element bases (OEIS rows A001212.., columns A001208..).
 *    This program looks for k-element bases with a large h-range by iterated local search (coordinate
 *    descent plus random kicks).  Each printed basis proves n(h,k) >= <range>; nothing is proved about
 *    optimality.  It produced the starting lower bounds and search targets of the exhaustive runs, e.g.
 *    n(9,8) >= 5418 with {1,7,23,88,145,638,1142,1610} and n(27,6) >= 176380 with
 *    {1,19,177,1016,6649,22876} (logs/overnight/lb_h9_k8.log and lb_h27_k6.log).
 *
 * 2. Mathematics
 *    The range comes from the min-stamp DP of tools/hrange.c restricted to [0, LIM]:
 *    r = min(n_h(A), LIM), exact when r < LIM.  The objective is r alone (no coverage term, unlike
 *    hsearch.c), and only strict improvements of r are accepted.
 *
 * 3. Algorithm (0-based arrays: a[0] = 1, a[1..K-1] the other elements, normally increasing)
 *    Each of the NT threads runs "rounds" until its deadline (thread start + SECS):
 *    a) Start basis.  If there are seeds, round 0 (and 2/3 of the later rounds) takes
 *       seeds[(id + round) % nseeds]; in rounds > 0 it is stretched, a[i] *= f^(i/(K-1)) with f uniform
 *       in [0.85, 1.15), so the largest element scales by f and the small ones hardly change.
 *       Otherwise a random geometric basis a[i] = ratio^i * U[0.6, 1.4) with ratio = MAXE^(1/(K-1)).
 *    b) Repair, one pass over i = 1..K-1: a[i] <= a[i-1] becomes a[i-1] + 1, and a[i] > MAXE becomes
 *       MAXE - (K-1-i) (leaving room for the remaining elements).  There is no final validity check.
 *    c) descend(): sweeps over the coordinates 1..K-1 in random order.  Coordinate i is set to the best
 *       value in its open interval (a[i-1], a[i+1]) (for the last one (a[K-2], MAXE]), which keeps the
 *       basis increasing; it is changed only if that strictly increases the range.  An interval of more
 *       than 6000 values gets a coarse pass with stride span/6000 + 1 followed by an exhaustive pass over
 *       +-stride around the coarse optimum, so there the true 1-D optimum can be missed.  Sweeps repeat
 *       until one makes no improvement (a local optimum for single-coordinate changes) or the deadline.
 *    d) Kick loop: copy the local optimum, move 1..KICK randomly chosen elements a[i] (i >= 1) by d
 *       uniform in [-span, span], span = 1 + a[i]/q, q uniform in 4..15 (about 7% to 25% of a[i]); clamp
 *       to [2, MAXE], sort, bump duplicates upward, and skip the kick if the last element then exceeds
 *       MAXE.  Descend from the kicked basis; if the result is strictly better it replaces the current
 *       basis (stale = 0), else stale++.  After 6 consecutive unsuccessful kicks a new round starts.
 *    Every strict improvement made inside descend() goes to report(), which prints it if it beats the
 *    global best.
 *
 * 4. Data structures
 *    seeds[64][64], nseeds: start bases from -seed (read-only once the threads run).  best_global and
 *    best_basis[]: the best range/basis printed so far, guarded by lk (which also serialises stdout).
 *    Per thread: cnt[LIM+2] uint16_t DP table (2*(LIM+2) bytes), a = current basis, b = kicked copy,
 *    and the xorshift64 state.
 *
 * 5. Parallelization and random numbers
 *    The threads are independent searches sharing only best_global/best_basis and stdout (under lk).
 *    Each uses Marsaglia's xorshift64 (shifts 13, 7, 17) seeded with id * 0x9E3779B97F4A7C15 (2^64
 *    divided by the golden ratio) XOR time(), so runs are not reproducible.  The thread id also picks
 *    the seed of each round, (id + round) % nseeds, which spreads the seeds over the threads.
 *
 * 6. Command line (any order, except that -k must come before -seed)
 *    -h H         stamps per sum, >= 1 (<= 65534 for the uint16 counts)                     required
 *    -k K         number of elements, >= 2 (<= 64 assumed, not checked)                     required
 *    -max M       largest element allowed, >= 2                                             required
 *    -lim L       table size: ranges are evaluated only up to L (~1.5x the expected range)  required
 *    -sec S       seconds per thread (default 60)
 *    -j N         threads (default 8; <= 64 assumed, not checked)
 *    -seed "..."  a start basis of exactly K elements (1 first), separated by spaces or commas;
 *                 repeatable, at most 64.  A seed of the wrong length is ignored with a warning.
 *    -kick N      move at most N elements per kick (default 2; N must be >= 1, N = 0 means a modulo by
 *                 zero).  -kick is not listed in the program's own usage message.
 *    Example (illustrative): hsearch2 -h 9 -k 8 -max 2500 -lim 9000 -sec 600 -j 8
 *
 * 7. Output
 *    stdout:  best <R>: a1 ... ak                         new global best; "   [HIT LIM - raise -lim]" is
 *                                                         appended when R >= LIM (true range may be larger)
 *             done h=<H> k=<K> best range <R>: a1 ... ak  the best basis (all zeros if nothing was printed)
 *    stderr:  "seed has N elements, expected K", "bad arg X", the usage line.
 *    Exit status 0; 1 for a bad argument or a missing required option.
 *
 * 8. Why the results are only lower bounds
 *    Local search with restarts and kicks proves nothing about optimality.  Each printed range is exact
 *    for the printed basis (when below LIM), so it is a rigorous lower bound n(h,k) >= R; the project
 *    re-verifies such bases with tools/hrange.
 *
 * 9. Limits and caveats
 *    - Counts are uint16_t with 0xFFFF = unreachable: exact for H <= 65534.  Each evaluation costs
 *      O(K * LIM); one coordinate scan costs up to about 6000 + 2*stride + 1 evaluations.
 *    - The start basis itself is never reported, only strict improvements made by descend(): a seed that
 *      is already a local optimum shows up only if a kick and descent reach it (or beat it) again.
 *    - The repair in step b) can, in rare edge cases (an element landing exactly on MAXE before a later
 *      one is clamped), leave a non-increasing basis or a repeated element; descend() then skips
 *      coordinates whose interval is empty.  Re-verify printed bases with tools/hrange.
 *    - Not checked: K <= 64, at most 64 seeds of at most 64 tokens, NT <= 64 (th[64]), KICK >= 1, seeds
 *      starting with 1 and lying in [1, MAXE], and that a value follows each option (argv[++i]).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

/* Parameters (set by main(), read-only in the threads): H = stamps per sum, K = elements, NT = threads,
 * MAXE = largest allowed element, LIM = evaluation limit / table size, SECS = run time per thread,
 * KICK = most elements moved per kick.  best_global / best_basis[0..K-1] = best range and basis printed
 * so far (guarded by lk, which also serialises stdout).  seeds[s][0..K-1], s < nseeds: start bases. */
static int H, K, NT = 8; static long MAXE = 0, LIM = 0; static double SECS = 60; static int KICK = 2;
static long best_global = 0; static long best_basis[64]; static pthread_mutex_t lk = PTHREAD_MUTEX_INITIALIZER;
static long seeds[64][64]; static int nseeds = 0;
/*
 * rng -- xorshift64 pseudo-random generator (Marsaglia, shifts 13, 7, 17).
 *
 * Purpose: per-thread random source for start bases, sweep orders and kicks.
 * Inputs:  s = the 64-bit state, which must be nonzero (a zero state stays zero forever).
 * Outputs: advances *s and returns the new state; callers reduce it with % (the bias is negligible).
 * Errors:  none.
 */
static inline uint64_t rng(uint64_t *s) { uint64_t x = *s; x ^= x << 13; x ^= x >> 7; x ^= x << 17; return *s = x; }
/*
 * tnow -- monotonic wall-clock time in seconds.
 *
 * Purpose: enforces the per-thread deadline (thread start + SECS).
 * Inputs:  none.
 * Outputs: CLOCK_MONOTONIC time as a double.
 * Errors:  none (a clock_gettime() failure is not checked).
 */
static double tnow(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

/*
 * hrange -- h-range of a basis, computed only up to LIM.
 *
 * Purpose: objective function: the min-stamp DP of tools/hrange.c on the fixed table [0, LIM].
 * Inputs:  a[0..k-1] = basis (any order; a[0] = 1 for a nonzero result; elements > LIM are skipped since
 *          they cannot reach [1, LIM]); k = number of elements; cnt = scratch table of at least LIM+1
 *          uint16_t.  Globals LIM and H.
 * Outputs: returns min(n_h(A), LIM); cnt[] is left holding the minimal stamp counts.
 * Errors:  none; exact for H <= 65534 (uint16 counts, 0xFFFF = unreachable, see tools/hrange.c).
 */
static long hrange(const long *a, int k, uint16_t *cnt) {
    for (long x = 0; x <= LIM; x++) cnt[x] = 0xFFFF;
    cnt[0] = 0;
    /* unbounded-knapsack update per element; the +1 is done in unsigned so 0xFFFF + 1 cannot wrap */
    for (int i = 0; i < k; i++) {
        long ai = a[i]; if (ai > LIM) continue;
        for (long x = ai; x <= LIM; x++) { unsigned v = cnt[x - ai] + 1u; if (v < cnt[x]) cnt[x] = (uint16_t)v; }
    }
    /* r = length of the initial run 1..r of values needing <= H stamps, at most LIM */
    long r = 0; while (r + 1 <= LIM && cnt[r + 1] <= H) r++;
    return r;
}
/*
 * report -- record and print a basis if its range beats the global best.
 *
 * Purpose: the single place where results are published; compare and update are atomic under lk.
 * Inputs:  r = range of the basis from hrange() (so r <= LIM); a[0..k-1] = the basis; globals
 *          best_global and LIM.
 * Outputs: if r > best_global: best_global = r, best_basis = a, and "best R: a1 ... ak" (with
 *          "   [HIT LIM - raise -lim]" appended when r >= LIM) on stdout, flushed.  Else nothing.
 * Errors:  none.
 */
static void report(long r, const long *a, int k) {
    pthread_mutex_lock(&lk);
    if (r > best_global) { best_global = r; memcpy(best_basis, a, sizeof(long) * k);
        printf("best %ld:", r); for (int i = 0; i < k; i++) printf(" %ld", a[i]); printf("%s\n", r >= LIM ? "   [HIT LIM - raise -lim]" : ""); fflush(stdout); }
    pthread_mutex_unlock(&lk);
}
/*
 * descend -- exact coordinate descent; returns the range of the final basis.
 *
 * Purpose: the local optimiser applied to every start basis and after every kick (THEORY 3c).  Each
 *          coordinate is optimised over its whole feasible interval: exactly when that has at most 6000
 *          values, otherwise by a strided coarse pass plus a +-stride refinement.  Only strict range
 *          improvements are taken; the sweeps repeat until one improves nothing.
 * Inputs:  a[0..K-1] = basis (a[0] = 1, normally increasing), modified in place; cnt = this thread's DP
 *          table; seed = this thread's RNG state (sweep order); deadline = tnow() value to stop at.
 *          Globals K, MAXE, LIM, H.
 * Outputs: returns the range of the final a (<= LIM); a holds the improved basis; every strict
 *          improvement is passed to report() (printed only if it beats the global best).
 * Errors:  none.  A coordinate with an empty interval (only possible for a non-increasing input) is
 *          skipped.  When the deadline passes, it returns before reaching a local optimum.
 */
static long descend(long *a, uint16_t *cnt, uint64_t *seed, double deadline) {
    long cur = hrange(a, K, cnt);
    int improved = 1;
    while (improved && tnow() < deadline) {
        improved = 0;
        /* random sweep order of the coordinates 1..K-1: Fisher-Yates shuffle of order[1..K-1]
         * (j uniform in 1..i); order[0] = 0 stays in place because a[0] = 1 is never moved */
        int order[64]; for (int i = 0; i < K; i++) order[i] = i;
        for (int i = K - 1; i > 1; i--) { int j = 1 + rng(seed) % i; int t = order[i]; order[i] = order[j]; order[j] = t; }
        for (int oi = 1; oi < K; oi++) {
            int i = order[oi]; if (i == 0) continue;
            /* feasible interval of a[i]: strictly between its neighbours (the last one up to MAXE), so the
             * basis stays increasing; empty only if the input was not increasing */
            long lo = a[i - 1] + 1, hi = (i < K - 1) ? a[i + 1] - 1 : MAXE;
            if (hi < lo) continue;
            /* at most ~6000 evaluations in the first pass: stride = span/6000 + 1 when the span is larger */
            long span = hi - lo + 1, stride = 1; if (span > 6000) stride = span / 6000 + 1;   /* coarse pass for huge spans */
            long bestx = a[i], bestr = cur, save = a[i];
            for (long x = lo; x <= hi; x += stride) { a[i] = x; long r = hrange(a, K, cnt); if (r > bestr) { bestr = r; bestx = x; } }
            /* refinement: every value within +-stride of the coarse optimum, i.e. all points between its
             * neighbouring coarse samples (other local maxima between samples can still be missed) */
            if (stride > 1) { long c = bestx; for (long x = c - stride; x <= c + stride; x++) { if (x < lo || x > hi) continue; a[i] = x; long r = hrange(a, K, cnt); if (r > bestr) { bestr = r; bestx = x; } } }
            /* keep the best value only if it strictly beats the current range, else restore a[i] */
            a[i] = bestx;
            if (bestr > cur) { cur = bestr; improved = 1; report(cur, a, K); } else a[i] = save;
            if (tnow() > deadline) break;
        }
    }
    return cur;
}
/*
 * worker -- one search thread: rounds of (start basis, descend, kick loop) until the deadline.
 *
 * Purpose: the search itself (THEORY sections 3a, 3b and 3d).
 * Inputs:  arg = thread id 1..NT cast to a pointer (seeds the RNG and picks the seeds of each round);
 *          globals H, K, MAXE, LIM, SECS, KICK, seeds[], nseeds.
 * Outputs: "best" lines, best_global and best_basis through report() (called inside descend()).
 *          Returns NULL.
 * Errors:  malloc() failure of the table is not checked; KICK = 0 is a modulo by zero (undefined
 *          behaviour); K > 64 overflows a[]/b[] (not checked).
 */
static void *worker(void *arg) {
    long id = (long)(uintptr_t)arg;
    /* 0x9E3779B97F4A7C15 = 2^64 / golden ratio spreads the thread ids; time() varies per run */
    uint64_t seed = (uint64_t)id * 0x9E3779B97F4A7C15ULL ^ (uint64_t)time(NULL);
    long a[64], b[64]; uint16_t *cnt = malloc(sizeof(uint16_t) * (LIM + 2));
    double deadline = tnow() + SECS; int round = 0;
    while (tnow() < deadline) {
        /* start basis: seeds in rotation (round 0 always, later rounds with probability 2/3), stretched by
         * a[i] *= f^(i/(K-1)), f in [0.85, 1.15), after round 0; else a random geometric basis whose
         * largest element is about MAXE */
        if (nseeds > 0 && (round == 0 || rng(&seed) % 3)) {
            memcpy(a, seeds[(id + round) % nseeds], sizeof(long) * K);
            if (round > 0) { double f = 0.85 + 0.3 * (rng(&seed) % 1000) / 1000.0; for (int i = 1; i < K; i++) { a[i] = (long)(a[i] * pow(f, (double)i / (K - 1))); } }
        } else {
            a[0] = 1; double ratio = pow((double)MAXE, 1.0 / (K - 1));
            for (int i = 1; i < K; i++) a[i] = (long)(pow(ratio, i) * (0.6 + 0.8 * (rng(&seed) % 1000) / 1000.0));
        }
        /* repair in one pass: bump to make increasing, clamp to MAXE - (K-1-i) to leave room for the
         * elements above.  Not re-checked afterwards (see the caveat in the header). */
        for (int i = 1; i < K; i++) { if (a[i] <= a[i - 1]) a[i] = a[i - 1] + 1; if (a[i] > MAXE) a[i] = MAXE - (K - 1 - i); }
        long cur = descend(a, cnt, &seed, deadline);
        /* kick loop */
        /* perturb 1..KICK random elements by up to 1/q of their value, q in 4..15, restore a sorted
         * strictly increasing basis in [2, MAXE] (skip the kick if that is impossible), re-descend, keep
         * strict improvements; 6 failures in a row end the round */
        int stale = 0;
        while (stale < 6 && tnow() < deadline) {
            memcpy(b, a, sizeof(long) * K);
            int nk = 1 + rng(&seed) % KICK;
            for (int t = 0; t < nk; t++) { int i = 1 + rng(&seed) % (K - 1); long span = 1 + b[i] / (4 + rng(&seed) % 12); b[i] += (long)(rng(&seed) % (2 * span + 1)) - span; }
            for (int i = 1; i < K; i++) { if (b[i] < 2) b[i] = 2; if (b[i] > MAXE) b[i] = MAXE; }
            /* insertion sort of b[0..K-1] (b[0] = 1 stays first since all others are >= 2), then bump */
            for (int i = 1; i < K; i++) for (int j = i; j > 0 && b[j] < b[j - 1]; j--) { long t = b[j]; b[j] = b[j - 1]; b[j - 1] = t; }
            for (int i = 1; i < K; i++) if (b[i] <= b[i - 1]) b[i] = b[i - 1] + 1;
            /* a skipped kick does not count as a failure */
            if (b[K - 1] > MAXE) continue;
            long r = descend(b, cnt, &seed, deadline);
            if (r > cur) { cur = r; memcpy(a, b, sizeof(long) * K); stale = 0; } else stale++;
        }
        round++;
    }
    free(cnt); return NULL;
}
/*
 * main -- parse the options, run the threads, print the best basis.
 *
 * Purpose: entry point (THEORY sections 6 and 7).
 * Inputs:  argc/argv.  Sets H, K, MAXE, LIM, SECS, NT, KICK, seeds[] and nseeds.
 * Outputs: the "done h=H k=K best range R: a1 ... ak" line on stdout after all threads finish; exit
 *          status 0, or 1 for an unknown option or a missing required option (message on stderr).
 * Errors:  value-taking options read argv[++i] without checking that it exists; more than 64 seeds,
 *          seeds of more than 64 tokens and NT > 64 overflow fixed arrays (not checked); a -seed given
 *          before -k always has the "wrong" length; KICK is not validated; strdup() copies are never freed.
 */
int main(int argc, char **argv) {
    /* -seed is tokenised on spaces/commas at once and compared with the K known at that point */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h")) H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-k")) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-max")) MAXE = atol(argv[++i]);
        else if (!strcmp(argv[i], "-lim")) LIM = atol(argv[++i]);
        else if (!strcmp(argv[i], "-sec")) SECS = atof(argv[++i]);
        else if (!strcmp(argv[i], "-j")) NT = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-kick")) KICK = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-seed")) { char *s = strdup(argv[++i]); int n = 0; for (char *t = strtok(s, " ,"); t; t = strtok(NULL, " ,")) seeds[nseeds][n++] = atol(t); if (n == K) nseeds++; else fprintf(stderr, "seed has %d elements, expected %d\n", n, K); }
        else { fprintf(stderr, "bad arg %s\n", argv[i]); return 1; }
    }
    /* only lower limits are checked (the usage text omits -kick) */
    if (H < 1 || K < 2 || MAXE < 2 || LIM < 2) { fprintf(stderr, "usage: hsearch2 -h H -k K -max MAXELEM -lim LIM -sec S [-j T] [-seed \"...\"]...\n"); return 1; }
    /* NT independent workers with ids 1..NT (th[64]: NT <= 64 is assumed, not checked); the final line
     * repeats the best basis (zeros if nothing was ever reported) */
    pthread_t th[64]; for (long i = 0; i < NT; i++) pthread_create(&th[i], NULL, worker, (void *)(i + 1));
    for (int i = 0; i < NT; i++) pthread_join(th[i], NULL);
    printf("done h=%d k=%d best range %ld:", H, K, best_global); for (int i = 0; i < K; i++) printf(" %ld", best_basis[i]); printf("\n"); return 0;
}
