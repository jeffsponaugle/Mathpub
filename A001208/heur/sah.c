/* sah.c -- simulated annealing for h-bases: look for k-element sets with h-range >= target.
 * usage: sah h k target seconds [maxelem] [threads]   (default maxelem = target, threads = 8)
 *
 * Build: clang -O3 -mcpu=apple-m1 -std=gnu11 -pthread -o sah sah.c      (on Linux add -lm)
 *
 * THEORY OF OPERATIONS
 * ====================
 *
 * 1. What is computed
 *    n_h(A) = the largest N such that 1..N are all sums of at most h elements of the basis A (a_1 = 1);
 *    n(h,k) = its maximum over k-element bases (OEIS rows A001212..A001216, A005342.., columns
 *    A001208..).  sah is the general-h version of sa2.c: multi-start simulated annealing for k-element
 *    sets with h-range >= target.  Each set it prints proves n(h,k) >= the printed range (which is capped
 *    at target+1, see section 7); a failure proves nothing.  It is much weaker than the contest searches
 *    (README section 8: 337 of 385 for (h,k) = (3,15), 1339 of 1545 for (6,9)), so it only gives rough
 *    lower bounds.
 *
 * 2. Mathematics
 *    The min-stamp DP of tools/hrange.c restricted to [0, target+1], with 8-bit counts:
 *      cnt[x] = minimal number of elements summing to x (255 = unreachable),
 *      range r = min(n_h(A), target+1),  covered c = #{x in [1, target] : cnt[x] <= h}.
 *    Annealing score s = r + 0.01 * c (a weighted sum: 100 covered values = one unit of range).
 *
 * 3. Algorithm (0-based arrays: a[0] = 1 is never moved)
 *    NT threads, each repeating restarts until SECS seconds have passed (checked at every step):
 *      - start: a[0] = 1 plus K-1 distinct values drawn uniformly from [2, MAXE];
 *      - up to 300000 steps.  Move: pick i in 1..K-1; propose v = a[i] + U{-10..10} (probability 2/3) or
 *        v uniform in [2, MAXE] (1/3); retry up to 50 times until v is in [2, MAXE] and differs from the
 *        other elements (v = a[i] itself is allowed); otherwise skip the step;
 *      - temperature temp = 3.0 * (1 - it/300000) + 0.05, linear cooling from 3.05 to 0.05, computed
 *        before the acceptance test; Metropolis: accept if s' >= s or exp((s' - s)/temp) > u, u uniform
 *        in [0,1) with resolution 1e-6;
 *      - a proposal (accepted or not) whose range beats the global best is printed with its sorted basis.
 *
 * 4. Data structures
 *    Per thread: cnt[TGT+2] uint8_t DP table, a = current set, b = proposal.  Shared: best_global
 *    (guarded by lk, which also serialises stdout).
 *
 * 5. Parallelization and random numbers
 *    NT independent threads (default 8).  xorshift64 (shifts 13, 7, 17) seeded with
 *    id * 0x9E3779B97F4A7C15 XOR time(NULL) (ids 1..NT), so runs are not reproducible.
 *
 * 6. Command line
 *    sah h k target seconds [maxelem] [threads]
 *      h        stamps per sum, 1..254 (8-bit counts; not checked)
 *      k        number of elements, 2..64 (not checked)
 *      target   h-range to look for (ranges are evaluated only up to target+1)
 *      seconds  time budget per thread
 *      maxelem  largest element allowed (default = target); must be >= k
 *      threads  number of threads (default 8; at most 64, not checked)
 *    Example: sah 3 15 386 600      (look for a 15-element 3-basis with range >= 386 for 600 s)
 *
 * 7. Output (stdout; only the usage line goes to stderr)
 *    best range <R>: a1 ... ak       new global best below the target (sorted basis)
 *    FOUND range <R>: a1 ... ak      new global best with R >= target.  R is capped at target+1, so
 *                                    "FOUND range <target+1>" means "at least target+1", and after
 *                                    that nothing more can be printed.
 *    done h=<H> k=<K> best range <R> (target <T>)
 *    Exit status 0; 1 if fewer than 4 arguments are given.
 *
 * 8. Why the results are only lower bounds
 *    Annealing proves nothing about optimality.  The DP is exact up to target+1 for h <= 254, so each
 *    printed set really has a range >= the printed value (re-verify with tools/hrange).
 *
 * 9. Limits and caveats (nothing is checked)
 *    - 8-bit counts: 255 marks "unreachable", so h must be <= 254.  For h >= 255 every value passes the
 *      cnt <= h test and every set is reported as FOUND (the problem that made tools/hrange switch to
 *      uint16 counts).
 *    - K <= 64 (a[64], b[64], s[64]) and K >= 2 (rng() % (K-1)); NT <= 64 (th[64]); MAXE >= K, or the
 *      distinct-value start loops forever (MAXE = 1 is a modulo by zero).
 *    - best_global is read without the lock before the locked re-check (benign race).
 *    - Each evaluation costs O(K * target); the time check calls clock_gettime() at every step.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
/* Parameters (set by main(), read-only in the threads): H = stamps per sum, K = elements, TGT = target
 * h-range, MAXE = largest allowed element, NT = threads, SECS = time budget per thread.
 * best_global = best range seen by any thread (guarded by lk, which also serialises stdout). */
static int H, K, TGT, MAXE, NT = 8; static double SECS;
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
 * tnow -- monotonic wall-clock time in seconds.
 *
 * Purpose: enforces the time budget (checked at every annealing step).
 * Inputs:  none.
 * Outputs: CLOCK_MONOTONIC time as a double.
 * Errors:  none (a clock_gettime() failure is not checked).
 */
static double tnow(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
/*
 * eval -- h-range via min-stamp DP up to TGT+1, plus the coverage of [1, TGT].
 *
 * Purpose: objective function (THEORY section 2): the DP of tools/hrange.c on the table [0, TGT+1]
 *          with 8-bit counts.
 * Inputs:  a[0..k-1] = the set (any order; a[0] = 1 for a nonzero range; elements > TGT+1 are skipped
 *          since they cannot reach [1, TGT+1]); k = its size; cnt = scratch table of at least TGT+2
 *          bytes; globals TGT and H.
 * Outputs: *range = min(n_h(A), TGT+1); *cov = #{x in [1, TGT] : x is a sum of <= H elements};
 *          cnt[] holds the stamp counts (255 = unreachable, or more than 254 stamps).
 * Errors:  none for H <= 254.  For H >= 255 the 255 marker passes the "<= H" test, so every value looks
 *          covered and the result is wrong (not checked).
 */
static void eval(const int *a, int k, uint8_t *cnt, int *range, int *cov) {
    int lim = TGT + 1;
    /* 255 = unreachable; the +1 below is done in unsigned so 255 + 1 = 256 never wraps to 0 and never
     * beats a stored count (stored counts are <= 254) */
    memset(cnt, 255, lim + 1); cnt[0] = 0;
    for (int i = 0; i < k; i++) { int ai = a[i]; if (ai > lim) continue; for (int x = ai; x <= lim; x++) { unsigned v = cnt[x - ai] + 1u; if (v < cnt[x]) cnt[x] = (uint8_t)v; } }
    /* r stops at lim = TGT+1 at the latest: r == TGT+1 means "range >= TGT+1" */
    int r = 0; while (r + 1 <= lim && cnt[r + 1] <= H) r++;
    int c = 0; for (int x = 1; x <= TGT; x++) c += (cnt[x] <= H);
    *range = r; *cov = c;
}
/*
 * worker -- one annealing thread: restarts of up to 300000 steps until SECS seconds have elapsed.
 *
 * Purpose: the search itself (THEORY section 3).
 * Inputs:  arg = thread id 1..NT cast to a pointer (seeds the RNG); globals H, K, TGT, MAXE, SECS,
 *          best_global.
 * Outputs: "best range R: ..." / "FOUND range R: ..." lines on stdout (under lk); best_global.
 *          Returns NULL.
 * Errors:  malloc() failure is not checked; MAXE < K makes the start loop forever, MAXE = 1 or K < 2 is
 *          a modulo by zero; K > 64 overflows a[]/b[]/s[] (none checked).
 */
static void *worker(void *arg) {
    /* 0x9E3779B97F4A7C15 = 2^64 / golden ratio spreads the thread ids; time() varies per run */
    uint64_t seed = (uint64_t)(uintptr_t)arg * 0x9E3779B97F4A7C15ULL ^ (uint64_t)time(NULL);
    int a[64], b[64]; uint8_t *cnt = malloc(TGT + 2);
    double start = tnow();
    while (tnow() - start < SECS) {
        /* start: 1 plus K-1 distinct values uniform in [2, MAXE] (redrawn on collision).  The start set's
         * own range is never reported, only the proposals below. */
        a[0] = 1;
        for (int i = 1; i < K; i++) { int v; again: v = 2 + rng(&seed) % (MAXE - 1); for (int j = 0; j < i; j++) if (a[j] == v) goto again; a[i] = v; }
        /* score = range + 0.01 * coverage of [1, TGT] */
        int r, c; eval(a, K, cnt, &r, &c); double cur = r + 0.01 * c;
        long iters = 300000; double temp;
        for (long it = 0; it < iters && tnow() - start < SECS; it++) {
            /* proposal: replace a[i], i >= 1, by a[i] + U{-10..10} (probability 2/3) or a uniform value in
             * [2, MAXE]; up to 50 tries to get a value in range that collides with no other element
             * (v == a[i] is allowed); if all fail the step is skipped */
            memcpy(b, a, sizeof(int) * K);
            int i = 1 + rng(&seed) % (K - 1); int v, tries = 0;
            do { if (rng(&seed) % 3) v = b[i] + (int)(rng(&seed) % 21) - 10; else v = 2 + rng(&seed) % (MAXE - 1); tries++; int dup = 0; for (int j = 0; j < K; j++) if (j != i && b[j] == v) dup = 1; if (v >= 2 && v <= MAXE && !dup) break; } while (tries < 50);
            if (tries >= 50) continue;
            b[i] = v;
            int r2, c2; eval(b, K, cnt, &r2, &c2); double s2 = r2 + 0.01 * c2;
            /* linear cooling 3.05 -> 0.05; Metropolis: always accept a non-worse score, a worse one with
             * probability exp(delta/temp) (uniform number with resolution 1e-6) */
            temp = 3.0 * (1.0 - (double)it / iters) + 0.05;
            if (s2 >= cur || exp((s2 - cur) / temp) > (rng(&seed) % 1000000) / 1e6) { memcpy(a, b, sizeof(int) * K); cur = s2; }
            /* new global best (accepted or not): unlocked pre-check (benign race), re-checked under the
             * lock, then printed with a sorted copy s[] as FOUND (r2 >= TGT) or best */
            if (r2 > best_global) {
                pthread_mutex_lock(&lk);
                if (r2 > best_global) { best_global = r2; int s[64]; memcpy(s, b, sizeof(int) * K);
                    for (int x = 0; x < K; x++) for (int y = x + 1; y < K; y++) if (s[y] < s[x]) { int t = s[x]; s[x] = s[y]; s[y] = t; }
                    printf("%s range %d:", r2 >= TGT ? "FOUND" : "best", r2); for (int x = 0; x < K; x++) printf(" %d", s[x]); printf("\n"); fflush(stdout); }
                pthread_mutex_unlock(&lk);
            }
        }
    }
    free(cnt); return NULL;
}
/*
 * main -- parse "h k target seconds [maxelem] [threads]", run the threads, print the best range.
 *
 * Purpose: entry point (THEORY sections 6 and 7).
 * Inputs:  argv[1..4] = h, k, target, seconds; argv[5] = maxelem (default target); argv[6] = threads
 *          (default 8).  Sets H, K, TGT, SECS, MAXE, NT.
 * Outputs: "done h=H k=K best range R (target T)" on stdout after the threads finish; exit status 0, or
 *          1 with the usage line on stderr if fewer than 4 arguments are given.
 * Errors:  atoi()/atof() accept garbage as 0; nothing is validated (h <= 254, 2 <= k <= 64, maxelem >= k,
 *          threads <= 64 are all assumed, see THEORY 9); threads <= 0 starts no thread at all.
 */
int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: sah h k target seconds [maxelem] [threads]\n"); return 1; }
    H = atoi(argv[1]); K = atoi(argv[2]); TGT = atoi(argv[3]); SECS = atof(argv[4]); MAXE = argc > 5 ? atoi(argv[5]) : TGT; if (argc > 6) NT = atoi(argv[6]);
    /* NT workers with ids 1..NT (th[64]: NT <= 64 is assumed, not checked) */
    pthread_t th[64]; for (long i = 0; i < NT; i++) pthread_create(&th[i], NULL, worker, (void *)(i + 1));
    for (int i = 0; i < NT; i++) pthread_join(th[i], NULL);
    printf("done h=%d k=%d best range %d (target %d)\n", H, K, best_global, TGT); return 0;
}
