/* hsearch.c -- heuristic search (simulated annealing + seeded local search) for h-bases
 * with large h-range.  Produces LOWER BOUNDS for n_h(k) and a target for the exact search.
 *
 * usage: hsearch -h H -k K -max MAXELEM -lim LIM -sec SECONDS [-j threads] [-seed "a1 a2 ... ak"]...
 *        [-scan "a1 ... a_{k-1}"]   (scan all a_k in (a_{k-1}, MAXELEM] for the given (k-1)-prefix)
 * LIM = upper limit for the range evaluation (table size); choose ~1.5x the expected range.
 * Prints every improvement: "best <range>: a1 ... ak   [ok]" (or "[HIT LIM - raise -lim]", see 7.).
 *
 * Build: clang -O3 -mcpu=apple-m1 -std=gnu11 -pthread -o hsearch hsearch.c      (on Linux add -lm)
 *
 * THEORY OF OPERATIONS
 * ====================
 *
 * 1. What is computed
 *    n_h(A) = the largest N such that 1..N are all sums of at most h elements of the basis A (a_1 = 1);
 *    n(h,k) = n_h(k) = the maximum of n_h(A) over all k-element bases (OEIS rows A001212.., columns
 *    A001208.., see tools/hrange.c).  This program looks for k-element bases with a large h-range by
 *    multi-start simulated annealing.  Every basis it prints is a valid k-element basis whose range was
 *    computed exactly, so it proves n(h,k) >= <range>; it can never prove an upper bound.  Such bounds
 *    serve as starting targets for the exhaustive searches (psph).  hsearch2.c is the companion
 *    coordinate-scan search (it produced the starting bounds n(9,8) >= 5418 and n(27,6) >= 176380).
 *
 * 2. Mathematics
 *    The range is computed with the min-stamp DP of tools/hrange.c, but only on [0, LIM]:
 *      cnt[x] = minimal number of elements summing to x,   r = min(n_h(A), LIM),
 *      cov = #{x in [1, LIM] : cnt[x] <= h}  (every coverable value, also those beyond the first gap).
 *    r < LIM is exact; r = LIM means "at least LIM" (flagged HIT LIM).
 *    Annealing score: s = r + 0.001 * cov.  The coverage term rewards bases with few holes beyond the
 *    first gap, which tend to lie near better bases.  It is a weighted sum, not a strict tie-break:
 *    with LIM = 270000 it can contribute up to 270.
 *
 * 3. Algorithm (arrays are 0-based below: a[0] = 1, a[1..K-1] are the other elements)
 *    a) Optional -scan: for the given (K-1)-prefix evaluate every a[K-1] in (a[K-2], MAXE]; the best
 *       completion is printed, appended to the seeds and becomes the initial global best.
 *    b) NT worker threads, each repeating "restarts" until SECS seconds have passed:
 *       - Start basis.  With seeds: restart 0 takes a random seed unchanged; a later restart takes a
 *         random seed and moves each a[i], i >= 1, by d * (1 + a[i]/200) with d uniform in -3..3 (up to
 *         about +-(3 + 1.5%)); this happens on every even restart and on 2/3 of the odd ones.  Otherwise
 *         (and always without seeds) a random geometric basis a[i] = ratio^i * U[0.6, 1.4) with
 *         ratio = MAXE^(1/(K-1)), clamped to [2, MAXE].
 *       - Repair: sort, bump duplicates upward, clamp the last element to MAXE.  If the result is not a
 *         valid basis (a[0] = 1, strictly increasing, all <= MAXE) the restart is abandoned.
 *       - Annealing for iters = min(400000, 20000 + 2000000 / (LIM/1000 + 1)) steps (fewer steps for
 *         large, expensive tables).  Move: pick i in 1..K-1 uniformly; with probability 1/10 replace a[i]
 *         by a uniform value in [2, MAXE], otherwise add d uniform in [-span, span] (d != 0) with
 *         span = 1 + a[i]/q, q uniform in 8..47 (a relative step of about 2% to 12.5%); clamp to
 *         [2, MAXE], sort, and reject the move if it creates a duplicate.
 *         Temperature: temp = 0.02 * r0 * (1 - it/iters) + 0.5, where r0 is the range of the restart's
 *         start basis, i.e. linear cooling from 2% of r0 (+0.5) down to 0.5.
 *         Metropolis rule: accept if s' >= s or exp((s' - s)/temp) > u, u uniform in [0,1) in 1e-6 steps.
 *       - Every evaluated basis (accepted or not) whose range beats the global best is printed.
 *    c) main() joins the threads and prints the final line.
 *
 * 4. Data structures
 *    seeds[64][64], nseeds: starting bases from -seed and -scan (read-only once the threads run).
 *    best_global: best range printed so far, protected by lk (which also serialises stdout).
 *    Per thread: cnt[LIM+2] uint16_t DP table (2*(LIM+2) bytes), a = current basis, b = proposal,
 *    and the xorshift64 state.
 *
 * 5. Parallelization and random numbers
 *    The threads are independent searches that share only best_global and stdout (under lk).  Each uses
 *    Marsaglia's xorshift64 (shifts 13, 7, 17) seeded with id * 0x9E3779B97F4A7C15 (2^64 divided by the
 *    golden ratio, to spread the ids) XOR time() XOR clock() << 20, so runs are not reproducible.
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
 *    -scan "..."  K-1 elements: scan the last element first (step 3a).
 *    Example (illustrative): hsearch -h 9 -k 8 -max 2500 -lim 9000 -sec 600 -seed "1 7 23 88 145 638 1142 1610"
 *
 * 7. Output
 *    stdout:  best <R>: a1 ... ak   [ok]          new global best; "[HIT LIM - raise -lim]" instead of
 *                                                 "[ok]" when R >= LIM (the true range may be larger)
 *             scan best <R> with a_k=<x>          each improvement during -scan
 *             scan done: best <R>: a1 ... ak
 *             done h=<H> k=<K> best range <R>     the basis is on the last "best" (or "scan done") line
 *    stderr:  "seed has N elements, expected K", "bad arg X", the usage line, "scan prefix must have k-1
 *             elements".  Exit status 0; 1 for a bad argument, a missing required option or a bad -scan.
 *
 * 8. Why the results are only lower bounds
 *    The search is stochastic and local; it samples a tiny part of the space and proves nothing about
 *    optimality.  Each printed range is exact for the printed basis (when below LIM), so it is a
 *    rigorous lower bound n(h,k) >= R; the project re-verifies such bases with tools/hrange.
 *
 * 9. Limits and caveats
 *    - Counts are uint16_t with 0xFFFF = unreachable: exact for H <= 65534.
 *    - Each evaluation costs O(K * LIM); memory is 2*(LIM+2) bytes per thread.
 *    - Not checked: K <= 64 (a[], b[], seeds), at most 64 seeds of at most 64 tokens, NT <= 64 (th[64]),
 *      and that a value follows each option (argv[++i]).
 *    - The temperature scale uses the range of the restart's start basis, not the current range, so a
 *      poor random start (small r0) anneals almost greedily.
 *    - best_global is read without the lock before report() is called (a benign race; report()
 *      re-checks under the lock).
 *    - A seed that does not start with 1 or exceeds MAXE never passes valid().  A -scan with an empty
 *      range (a_{k-1} >= MAXE) leaves a_k = 0 in its seed, which is likewise always rejected.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

/* Parameters (set by main(), read-only in the threads): H = stamps per sum, K = elements, NT = threads,
 * MAXE = largest allowed element, LIM = evaluation limit / table size, SECS = run time per thread.
 * best_global = best range printed so far (guarded by lk, which also serialises stdout).
 * seeds[s][0..K-1], s < nseeds: start bases from -seed and -scan (at most 64 x 64, not checked). */
static int H, K, NT = 8; static long MAXE = 0, LIM = 0; static double SECS = 60;
static long best_global = 0; static pthread_mutex_t lk = PTHREAD_MUTEX_INITIALIZER;
static long seeds[64][64]; static int nseeds = 0;

/*
 * rng -- xorshift64 pseudo-random generator (Marsaglia, shifts 13, 7, 17).
 *
 * Purpose: per-thread random source for start bases, moves and the Metropolis test.
 * Inputs:  s = the 64-bit state, which must be nonzero (a zero state stays zero forever).
 * Outputs: advances *s and returns the new state; callers reduce it with % (the bias is negligible).
 * Errors:  none.
 */
static inline uint64_t rng(uint64_t *s) { uint64_t x = *s; x ^= x << 13; x ^= x >> 7; x ^= x << 17; return *s = x; }
/*
 * tnow -- monotonic wall-clock time in seconds.
 *
 * Purpose: enforces the -sec time budget.
 * Inputs:  none.
 * Outputs: CLOCK_MONOTONIC time as a double.
 * Errors:  none (a clock_gettime() failure is not checked).
 */
static double tnow(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

/*
 * hrange -- h-range of sorted basis a[0..k-1] (a[0]=1) computed up to LIM; also coverage count.
 *
 * Purpose: objective function of the annealing and of -scan: the min-stamp DP of tools/hrange.c on the
 *          fixed table [0, LIM], so that every evaluation costs the same.
 * Inputs:  a[0..k-1] = basis (sorted with a[0] = 1 in all calls, although the DP does not need the order;
 *          elements > LIM are skipped since they cannot reach [1, LIM]); k = number of elements;
 *          cnt = scratch table of at least LIM+1 uint16_t; cov = where to store the coverage, or NULL.
 *          Globals LIM and H.
 * Outputs: returns min(n_h(A), LIM); if cov != NULL, *cov = #{x in [1, LIM] : x is a sum of <= H
 *          elements}; cnt[] is left holding the minimal stamp counts.
 * Errors:  none; exact for H <= 65534 (uint16 counts, 0xFFFF = unreachable, see tools/hrange.c).
 */
static long hrange(const long *a, int k, uint16_t *cnt, long *cov) {
    for (long x = 0; x <= LIM; x++) cnt[x] = 0xFFFF;
    cnt[0] = 0;
    /* unbounded-knapsack update per element; the +1 is done in unsigned so 0xFFFF + 1 cannot wrap */
    for (int i = 0; i < k; i++) {
        long ai = a[i]; if (ai > LIM) continue;
        for (long x = ai; x <= LIM; x++) { unsigned v = cnt[x - ai] + 1u; if (v < cnt[x]) cnt[x] = (uint16_t)v; }
    }
    /* r = length of the initial run 1..r of values needing <= H stamps, at most LIM */
    long r = 0; while (r + 1 <= LIM && cnt[r + 1] <= H) r++;
    long c = 0; if (cov) { for (long x = 1; x <= LIM; x++) c += (cnt[x] <= H); *cov = c; }
    return r;
}
/*
 * sort_basis -- sort a[0..k-1] into increasing order (insertion sort; k is small).
 *
 * Purpose: restores the increasing order after a start perturbation or a move.
 * Inputs:  a = array of k values; k >= 0.
 * Outputs: a, sorted in place.
 * Errors:  none.
 */
static void sort_basis(long *a, int k) { for (int i = 1; i < k; i++) { long v = a[i]; int j = i - 1; while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; j--; } a[j + 1] = v; } }
/*
 * valid -- is a[0..k-1] a usable basis?
 *
 * Purpose: rejects start bases that the seed perturbation or the repair step made unusable.
 * Inputs:  a = candidate (sorted); k = its length; global MAXE.
 * Outputs: 1 if a[0] == 1, the elements are strictly increasing and a[i] <= MAXE for i >= 1; else 0.
 * Errors:  none.
 */
static int valid(const long *a, int k) { if (a[0] != 1) return 0; for (int i = 1; i < k; i++) if (a[i] <= a[i - 1] || a[i] > MAXE) return 0; return 1; }

/*
 * report -- print a basis if its range beats the global best.
 *
 * Purpose: the single place where results are published; compare and update are atomic under lk.
 * Inputs:  r = range of the basis as returned by hrange() (so r <= LIM); a[0..k-1] = the basis (sorted);
 *          globals best_global and LIM.
 * Outputs: if r > best_global: best_global = r and "best R: a1 ... ak   [ok]" (or "[HIT LIM - raise
 *          -lim]" when r >= LIM, i.e. the true range may be larger) on stdout, flushed.  Else nothing.
 * Errors:  none.
 */
static void report(long r, const long *a, int k) {
    pthread_mutex_lock(&lk);
    if (r > best_global) {
        best_global = r;
        printf("best %ld:", r); for (int i = 0; i < k; i++) printf(" %ld", a[i]); printf("   [%s]\n", r >= LIM ? "HIT LIM - raise -lim" : "ok"); fflush(stdout);
    }
    pthread_mutex_unlock(&lk);
}

/*
 * worker -- one annealing thread: restart after restart until SECS seconds have elapsed.
 *
 * Purpose: the search itself (THEORY section 3b): pick a start basis, repair it, anneal it.
 * Inputs:  arg = thread id 1..NT cast to a pointer (used to seed the RNG); globals H, K, MAXE, LIM,
 *          SECS, seeds[], nseeds, best_global.
 * Outputs: "best" lines and best_global through report().  Returns NULL.
 * Errors:  malloc() failure of the table is not checked.  K > 64 overflows a[]/b[] (not checked).
 */
static void *worker(void *arg) {
    long id = (long)(uintptr_t)arg;
    /* 0x9E3779B97F4A7C15 = 2^64 / golden ratio spreads the thread ids; time() and clock() vary per run */
    uint64_t seed = (uint64_t)id * 0x9E3779B97F4A7C15ULL ^ (uint64_t)time(NULL) ^ ((uint64_t)clock() << 20);
    long a[64], b[64]; uint16_t *cnt = malloc(sizeof(uint16_t) * (LIM + 2));
    double start = tnow();
    int restart = 0;
    while (tnow() - start < SECS) {
        /* init: seeds (perturbed) or random geometric-ish */
        /* Seed branch: always on even restarts, on 2/3 of the odd ones (the rng() call is skipped on even
         * restarts).  Restart 0 uses the seed as is; later restarts move each a[i] by d*(1 + a[i]/200),
         * d in -3..3, without clamping (valid() below catches results < 2 or > MAXE). */
        if (nseeds > 0 && (restart % 2 == 0 || (int)(rng(&seed) % 3))) {
            memcpy(a, seeds[rng(&seed) % nseeds], sizeof(long) * K);
            if (restart > 0) for (int i = 1; i < K; i++) { long d = (long)(rng(&seed) % 7) - 3; a[i] += d * (1 + a[i] / 200); }
        } else {
            /* random geometric basis: a[i] ~ ratio^i times a factor in [0.6, 1.4), so a[K-1] ~ MAXE */
            a[0] = 1; double ratio = pow((double)MAXE, 1.0 / (K - 1));
            for (int i = 1; i < K; i++) { double v = pow(ratio, i) * (0.6 + 0.8 * (rng(&seed) % 1000) / 1000.0); a[i] = (long)v; if (a[i] < 2) a[i] = 2; if (a[i] > MAXE) a[i] = MAXE; }
        }
        /* repair: sort, make strictly increasing by bumping, clamp the last element; give up if invalid */
        sort_basis(a, K); for (int i = 1; i < K; i++) if (a[i] <= a[i - 1]) a[i] = a[i - 1] + 1; if (a[K - 1] > MAXE) a[K - 1] = MAXE;
        if (!valid(a, K)) { restart++; continue; }
        /* r = range of the start basis (also the fixed temperature scale r0 of this restart);
         * score = range + 0.001 * coverage of [1, LIM] */
        long cov; long r = hrange(a, K, cnt, &cov); double cur = r + 0.001 * cov;
        report(r, a, K);
        /* steps per restart: 20000 plus a share that shrinks as the table (cost per step) grows;
         * capped at 400000, which applies when LIM < 5000 */
        long iters = 20000 + 2000000 / (LIM / 1000 + 1); if (iters > 400000) iters = 400000;
        for (long it = 0; it < iters && tnow() - start < SECS; it++) {
            /* proposal b = a with one element a[i], i >= 1, moved (a[0] = 1 is never touched):
             * span = 1 + a[i]/q, q in 8..47, i.e. steps of up to about 2%..12.5% of a[i];
             * with probability 1/10 the element is instead redrawn uniformly from [2, MAXE] */
            memcpy(b, a, sizeof(long) * K);
            int i = 1 + rng(&seed) % (K - 1);
            long span = 1 + b[i] / (8 + (rng(&seed) % 40));            /* relative step */
            long d = (long)(rng(&seed) % (2 * span + 1)) - span; if (d == 0) d = 1;
            if (rng(&seed) % 10 == 0) b[i] = 2 + rng(&seed) % (MAXE - 1); else b[i] += d;
            if (b[i] < 2) b[i] = 2; if (b[i] > MAXE) b[i] = MAXE;
            /* keep the basis sorted; a move onto an existing element is discarded (it still counts as a step) */
            sort_basis(b, K);
            int dup = 0; for (int j = 1; j < K; j++) if (b[j] == b[j - 1]) dup = 1; if (dup) continue;
            long cov2; long r2 = hrange(b, K, cnt, &cov2); double s2 = r2 + 0.001 * cov2;
            /* linear cooling from 0.02*r + 0.5 to 0.5, where r is the START range of this restart */
            double temp = (0.02 * r) * (1.0 - (double)it / iters) + 0.5;
            /* Metropolis: always accept a non-worse score, a worse one with probability exp(delta/temp)
             * (the uniform number has a resolution of 1e-6) */
            if (s2 >= cur || exp((s2 - cur) / temp) > (rng(&seed) % 1000000) / 1e6) { memcpy(a, b, sizeof(long) * K); cur = s2; }
            /* publish any proposal that beats the global best, accepted or not (unlocked pre-check,
             * report() re-checks under the lock) */
            if (r2 > best_global) report(r2, b, K);
        }
        restart++;
    }
    free(cnt); return NULL;
}

/*
 * main -- parse the options, run the optional -scan, start the threads, print the final line.
 *
 * Purpose: entry point (THEORY sections 3a, 6 and 7).
 * Inputs:  argc/argv.  Sets H, K, MAXE, LIM, SECS, NT, seeds[], nseeds and (after -scan) best_global.
 * Outputs: the stdout lines of section 7; exit status 0, or 1 for an unknown option, a missing required
 *          option or a -scan prefix of the wrong length (message on stderr).
 * Errors:  value-taking options read argv[++i] without checking that it exists; more than 64 seeds,
 *          seeds/prefixes of more than 64 tokens and NT > 64 overflow fixed arrays (not checked); a
 *          -seed given before -k always has the "wrong" length; the strdup() copies and the scan table
 *          are never freed (harmless).
 */
int main(int argc, char **argv) {
    const char *scan = NULL;
    /* -seed is tokenised on spaces/commas at once and compared with the K known at that point */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h")) H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-k")) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-max")) MAXE = atol(argv[++i]);
        else if (!strcmp(argv[i], "-lim")) LIM = atol(argv[++i]);
        else if (!strcmp(argv[i], "-sec")) SECS = atof(argv[++i]);
        else if (!strcmp(argv[i], "-j")) NT = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-seed")) { char *s = strdup(argv[++i]); int n = 0; for (char *t = strtok(s, " ,"); t; t = strtok(NULL, " ,")) seeds[nseeds][n++] = atol(t); if (n == K) nseeds++; else fprintf(stderr, "seed has %d elements, expected %d\n", n, K); }
        else if (!strcmp(argv[i], "-scan")) scan = argv[++i];
        else { fprintf(stderr, "bad arg %s\n", argv[i]); return 1; }
    }
    /* only lower limits are checked (MAXE >= 2 also keeps rng() % (MAXE - 1) in worker() well defined) */
    if (H < 1 || K < 2 || MAXE < 2 || LIM < 2) { fprintf(stderr, "usage: hsearch -h H -k K -max MAXELEM -lim LIM -sec S [-j T] [-seed \"...\"] [-scan \"...\"]\n"); return 1; }
    /* -scan: single-threaded, before the workers.  Try every last element x in (a_{k-1}, MAXE] after the
     * given prefix (no coverage term), print each strict improvement, then add the best completion to
     * the seeds and make its range the initial global best (so workers only print better bases). */
    if (scan) {
        long a[64]; int n = 0; char *s = strdup(scan); for (char *t = strtok(s, " ,"); t; t = strtok(NULL, " ,")) a[n++] = atol(t);
        if (n != K - 1) { fprintf(stderr, "scan prefix must have k-1 elements\n"); return 1; }
        uint16_t *cnt = malloc(sizeof(uint16_t) * (LIM + 2)); long bestr = 0, besta = 0;
        for (long x = a[K - 2] + 1; x <= MAXE; x++) { a[K - 1] = x; long r = hrange(a, K, cnt, NULL); if (r > bestr) { bestr = r; besta = x; printf("scan best %ld with a_k=%ld\n", r, x); fflush(stdout); } }
        a[K - 1] = besta; memcpy(seeds[nseeds++], a, sizeof(long) * K);
        printf("scan done: best %ld:", bestr); for (int i = 0; i < K; i++) printf(" %ld", a[i]); printf("\n"); fflush(stdout);
        best_global = bestr;
    }
    /* NT independent workers with ids 1..NT (th[64]: NT <= 64 is assumed, not checked) */
    pthread_t th[64]; for (long i = 0; i < NT; i++) pthread_create(&th[i], NULL, worker, (void *)(i + 1));
    for (int i = 0; i < NT; i++) pthread_join(th[i], NULL);
    printf("done h=%d k=%d best range %ld\n", H, K, best_global); return 0;
}
