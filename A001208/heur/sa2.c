/* sa2.c -- simulated annealing for 2-bases: look for k-element sets with 2-range >= target.
 * usage: sa2 k target seconds [maxelem]
 * Objective: range first, then #covered in [1,target], combined as score = range + 0.02 * covered.
 *   (Corrected comment: this is a weighted sum, not strictly lexicographic -- 50 extra covered values
 *   outweigh one unit of range.)  Moves: replace one element.
 *
 * Build: clang -O3 -mcpu=apple-m1 -std=gnu11 -pthread -o sa2 sa2.c      (on Linux add -lm)
 *
 * THEORY OF OPERATIONS
 * ====================
 *
 * 1. What is computed
 *    For h = 2: n_2(A) = the largest N such that 1..N are all sums of at most 2 elements of A (a_1 = 1),
 *    and n(2,k) = its maximum over k-element bases = A001212(k).  The program searches for k-element
 *    sets with 2-range >= target by multi-start simulated annealing and prints every set it meets that
 *    reaches the target.  Such a set proves A001212(k) >= its range; failing to find one proves nothing.
 *    Project results (README section 8, logs/sa2_*.log): it recovers the known optimum for k = 10, 14
 *    and 24 within minutes; for k = 25 (target 229, 400 s and 900 s) it reaches 228 and nothing higher,
 *    for k = 26 (target 245, 600 s) it reaches 244, in both cases the restricted optimum (A006638).
 *
 * 2. Mathematics
 *    The numbers representable with at most 2 elements are {0} u A u (A + A), so eval() simply marks
 *    every a_i and every pairwise sum a_i + a_j (i <= j) in a 0/1 table; O(k^2 + MAXE + target) per
 *    evaluation and exact (no DP needed for h = 2).  Every sum is <= 2*MAXE, so the table [0, 2*MAXE] is complete.
 *    range = the length of the initial run 1..r of marked values; covered = marked values in [1, target].
 *
 * 3. Algorithm (0-based arrays: a[0] = 1 is never moved)
 *    8 threads, each repeating restarts until SECS seconds have passed (checked only between restarts):
 *      - start: a[0] = 1 plus K-1 distinct values drawn uniformly from [2, MAXE] (no admissibility);
 *      - 400000 annealing steps.  Move: pick i in 1..K-1; propose v = a[i] + U{-5..5} (probability 1/2)
 *        or v uniform in [2, MAXE]; retry up to 50 times until v is in [2, MAXE] and differs from the
 *        other elements (v = a[i] itself is allowed, a null move); give up the step otherwise;
 *      - score = range + 0.02 * covered;  Metropolis: accept if s' >= s or exp((s' - s)/temp) > u,
 *        u uniform in [0,1) with resolution 1e-6;
 *      - temperature: 2.0 for the first step, then temp = 2.0 * (1 - it/400000) + 0.05, i.e. linear
 *        cooling from about 2 to 0.05 (updated after each acceptance test);
 *      - every proposal (accepted or not) with range >= target is printed as FOUND, and every new global
 *        best range is printed as "best so far".
 *
 * 4. Data structures
 *    Per thread: m[1024] (static __thread) 0/1 table of representable values; a = current set,
 *    b = proposal.  Shared: best_global (guarded by lk, which also serialises stdout).
 *
 * 5. Parallelization and random numbers
 *    A fixed 8 independent threads.  xorshift64 (shifts 13, 7, 17) seeded with id * 0x9E3779B97F4A7C15
 *    + time(NULL) (ids 1..8), so runs are not reproducible.
 *
 * 6. Command line
 *    sa2 k target seconds [maxelem]
 *      k        number of elements, 2..64 (not checked)
 *      target   2-range to look for; must be <= 1023 (size of m[], not checked)
 *      seconds  time budget per thread (a restart that has begun always runs its 400000 steps)
 *      maxelem  largest element allowed (default = target), silently capped at 500; must be >= k
 *    Example: sa2 25 229 900      (25 elements, look for 2-range >= 229 for 900 s, maxelem = 229)
 *
 * 7. Output (stdout; only the usage line goes to stderr)
 *    FOUND range <R>: a1 ... ak       sorted set with range R >= target; repeated for every such proposal,
 *                                     so the same set can be printed many times
 *    best so far range <R>            new global best (the set is not printed unless R >= target)
 *    done, best range <R> (target <T>)
 *    Exit status 0; 1 if fewer than 3 arguments are given.
 *
 * 8. Why the results are only lower bounds
 *    Annealing samples a tiny part of the search space; it proves nothing about optimality.  eval() is
 *    exact, so every FOUND set really has the printed range (re-verify with tools/hrange 2 ...).
 *
 * 9. Limits and caveats (only the MAXE cap is enforced; nothing else is checked)
 *    - MAXE <= 500 (forced) because m[1024] must hold all sums up to 2*MAXE = 1000; target <= 1023 or the
 *      coverage loop reads past m[]; K <= 64 (a[64], b[64]); K >= 2 (rng() % (K-1)); MAXE >= K, or the
 *      distinct-value start loops forever (MAXE = 1 is a modulo by zero).
 *    - The time limit is only checked between restarts, so a run can overshoot by one restart.
 *    - best_global is read without the lock before the locked re-check (benign race).
 *    - Leftovers without effect: t0 (CPU clock) and bestl (local best) are computed but never used.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

/* Parameters (set by main(), read-only in the threads): K = elements, TGT = target 2-range,
 * MAXE = largest allowed element (<= 500), SECS = time budget per thread.
 * best_global = best range seen by any thread (guarded by lk, which also serialises stdout). */
static int K, TGT, MAXE;
static double SECS;
static int best_global = 0; static pthread_mutex_t lk = PTHREAD_MUTEX_INITIALIZER;

/*
 * rng -- xorshift64 pseudo-random generator (Marsaglia, shifts 13, 7, 17).
 *
 * Purpose: per-thread random source for start sets, moves and the Metropolis test.
 * Inputs:  s = the 64-bit state, which must be nonzero (a zero state stays zero forever).
 * Outputs: advances *s and returns the new state; callers reduce it with % (the bias is negligible).
 * Errors:  none.
 */
static inline uint64_t rng(uint64_t *s) { uint64_t x = *s; x ^= x << 13; x ^= x >> 7; x ^= x << 17; return *s = x; }

/*
 * eval -- exact 2-range and target coverage of a set.
 *
 * Purpose: objective function: marks {0} u A u (A + A) in a per-thread 0/1 table (THEORY section 2).
 * Inputs:  a[0..k-1] = the set (any order, distinct, each in [1, MAXE]; a[0] = 1 for a nonzero range);
 *          k = its size; globals MAXE (<= 500) and TGT (<= 1023).
 * Outputs: *range = largest r with 1..r all marked (r <= 2*MAXE); *cov = number of marked x in [1, TGT].
 * Errors:  none if the limits hold; TGT > 1023 reads past m[] and an element above MAXE could write
 *          past the cleared region (both unchecked; callers keep elements in [2, MAXE]).
 */
static void eval(const int *a, int k, int *range, int *cov) {
    /* one table per thread; only [0, 2*MAXE + 1] is cleared and used for sums, entries above stay 0 */
    static __thread uint8_t m[1024];
    memset(m, 0, 2 * MAXE + 2);
    m[0] = 1;
    /* single elements and all pairs (j >= i includes doubles a_i + a_i) */
    for (int i = 0; i < k; i++) { m[a[i]] = 1; for (int j = i; j < k; j++) m[a[i] + a[j]] = 1; }
    int r = 0; while (r + 1 <= 2 * MAXE && m[r + 1]) r++;
    int c = 0; for (int x = 1; x <= TGT; x++) c += m[x];
    *range = r; *cov = c;
}
/*
 * score -- annealing objective: range plus 0.02 per covered value of [1, target].
 *
 * Purpose: ranks sets mainly by range; the coverage term lets the annealing climb towards sets that
 *          close more holes below the target (a weighted sum: 50 covered values = one unit of range).
 * Inputs:  r = range, c = covered count (from eval()).
 * Outputs: r + 0.02 * c.
 * Errors:  none.
 */
static double score(int r, int c) { return r * 1.0 + c * 0.02; }

/*
 * worker -- one annealing thread: restarts of 400000 steps until SECS seconds have elapsed.
 *
 * Purpose: the search itself (THEORY section 3).
 * Inputs:  arg = thread id 1..8 cast to a pointer (seeds the RNG); globals K, TGT, MAXE, SECS,
 *          best_global.
 * Outputs: "FOUND range R: ..." and "best so far range R" lines on stdout (under lk); best_global.
 *          Returns NULL.
 * Errors:  MAXE < K makes the distinct-value start loop forever and MAXE = 1 is a modulo by zero;
 *          K < 2 is a modulo by zero; K > 64 overflows a[]/b[]/s[] (none checked).
 */
static void *worker(void *arg) {
    /* 0x9E3779B97F4A7C15 = 2^64 / golden ratio spreads the thread ids; time() varies per run */
    uint64_t seed = (uint64_t)(uintptr_t)arg * 0x9E3779B97F4A7C15ULL + time(NULL);
    int a[64], b[64];
    /* t0 (CPU time) is unused; the time budget is measured with the monotonic clock */
    double t0 = (double)clock() / CLOCKS_PER_SEC; (void)t0;
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); double start = ts.tv_sec + ts.tv_nsec * 1e-9;
    for (;;) {
        /* the time budget is checked only here, between restarts */
        clock_gettime(CLOCK_MONOTONIC, &ts); if (ts.tv_sec + ts.tv_nsec * 1e-9 - start > SECS) break;
        /* random admissible-ish init: 1 plus random distinct values */
        /* (values uniform in [2, MAXE], redrawn on collision; admissibility is not enforced) */
        a[0] = 1;
        for (int i = 1; i < K; i++) { int v; again: v = 2 + rng(&seed) % (MAXE - 1); for (int j = 0; j < i; j++) if (a[j] == v) goto again; a[i] = v; }
        /* bestl (best local score) is maintained below but never used */
        int r, c; eval(a, K, &r, &c); double cur = score(r, c), bestl = cur;
        double temp = 2.0;
        for (long it = 0; it < 400000; it++) {
            /* proposal: replace a[i], i >= 1, by a nearby value a[i] + U{-5..5} (probability 1/2) or a
             * uniform value in [2, MAXE]; up to 50 tries to get a value in range that collides with no
             * other element (v == a[i] is allowed); if all fail the step is skipped */
            memcpy(b, a, sizeof(int) * K);
            int i = 1 + rng(&seed) % (K - 1);
            int v; int tries = 0;
            do { if (rng(&seed) & 1) v = b[i] + (int)(rng(&seed) % 11) - 5; else v = 2 + rng(&seed) % (MAXE - 1); tries++; int dup = 0; for (int j = 0; j < K; j++) if (j != i && b[j] == v) dup = 1; if (v >= 2 && v <= MAXE && !dup) break; } while (tries < 50);
            if (tries >= 50) continue;
            b[i] = v;
            int r2, c2; eval(b, K, &r2, &c2); double s2 = score(r2, c2);
            /* Metropolis: always accept a non-worse score, a worse one with probability exp(delta/temp)
             * (uniform number with resolution 1e-6) */
            if (s2 >= cur || exp((s2 - cur) / temp) > (rng(&seed) % 1000000) / 1e6) { memcpy(a, b, sizeof(int) * K); cur = s2; if (cur > bestl) bestl = cur; }
            /* linear cooling 2.05 -> 0.05 over the 400000 steps; set after the test, so step it uses the
             * value computed in step it-1 (2.0 in step 0; a skipped step does not update it) */
            temp = 2.0 * (1.0 - (double)it / 400000) + 0.05;
            /* report every evaluated proposal that reaches the target (accepted or not, possibly the same
             * set many times), printed sorted; s[] is a sorted copy (simple exchange sort) */
            if (r2 >= TGT) {
                pthread_mutex_lock(&lk);
                printf("FOUND range %d:", r2); int s[64]; memcpy(s, b, sizeof(int) * K);
                for (int x = 0; x < K; x++) for (int y = x + 1; y < K; y++) if (s[y] < s[x]) { int t = s[x]; s[x] = s[y]; s[y] = t; }
                for (int x = 0; x < K; x++) printf(" %d", s[x]); printf("\n"); fflush(stdout);
                pthread_mutex_unlock(&lk);
            }
            /* new global best range: unlocked pre-check (benign race), re-checked under the lock */
            if (r2 > best_global) { pthread_mutex_lock(&lk); if (r2 > best_global) { best_global = r2; printf("best so far range %d\n", r2); fflush(stdout); } pthread_mutex_unlock(&lk); }
        }
    }
    return NULL;
}
/*
 * main -- parse "k target seconds [maxelem]", run 8 threads, print the best range.
 *
 * Purpose: entry point (THEORY sections 6 and 7).
 * Inputs:  argv[1] = k, argv[2] = target, argv[3] = seconds, argv[4] = maxelem (default target).
 *          Sets K, TGT, SECS, MAXE.
 * Outputs: "done, best range R (target T)" on stdout after the threads finish; exit status 0, or 1 with
 *          the usage line on stderr if fewer than 3 arguments are given.
 * Errors:  atoi()/atof() accept garbage as 0; k, target and maxelem are not validated (THEORY 9).
 */
int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: sa2 k target seconds [maxelem]\n"); return 1; }
    K = atoi(argv[1]); TGT = atoi(argv[2]); SECS = atof(argv[3]); MAXE = argc > 4 ? atoi(argv[4]) : TGT;
    /* m[1024] in eval() holds sums up to 2*MAXE, hence the silent cap at 500 */
    if (MAXE > 500) MAXE = 500;
    /* fixed thread count: 8 workers with ids 1..8 */
    pthread_t th[8]; for (long i = 0; i < 8; i++) pthread_create(&th[i], NULL, worker, (void *)(i + 1));
    for (int i = 0; i < 8; i++) pthread_join(th[i], NULL);
    printf("done, best range %d (target %d)\n", best_global, TGT);
    return 0;
}
