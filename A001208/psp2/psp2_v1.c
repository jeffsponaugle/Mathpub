/*
 * psp2_v1.c (written as psp2.c; version 1) -- exhaustive search for extremal additive 2-bases
 *           (global postage stamp problem with h = 2 stamps, OEIS A001212).
 *
 * Find every set A = {a_0=0 < a_1=1 < a_2 < ... < a_k} of k positive
 * integers such that every integer in [1,T] is a sum of at most two elements
 * of A (a_i + a_j with 0 <= i <= j <= k).  Running with T = best known + 1
 * and finding nothing proves n_2(k) = best known.
 *
 * Algorithm: Challis-style depth-first search (the "K-program"):
 *   - admissibility: a_{j+1} <= n_2(A_j) + 1   (prefix must cover [1,a_{j+1}-1])
 *   - element-wise upper bound: a_j <= n_2(j-1) + 1  (known A001212 values)
 *   - gaps test: the number of uncovered values in [1,T] after A_j must not
 *     exceed sum_{i=j+1..k} (i+1), the most new sums the remaining elements
 *     can contribute (a_i + a_p for p = 0..i-1, plus 2 a_i).
 *   - optional largest-element split (-M): fix a_k = M first, so that the
 *     sums M + a_p are placed immediately and the gaps test bites earlier.
 *
 * Coverage of [0,T] is kept as a 256-bit mask (T <= 255), elements as a
 * second 256-bit mask; adding x costs one 256-bit shift-or.
 *
 * Build:  clang -O3 -mcpu=apple-m1 -std=c11 -pthread -o psp2 psp2.c
 * Usage:  ./psp2 -k K -t T [-j threads] [-d splitdepth] [-M] [-q]
 *         ./psp2 -selftest
 *         (Written as psp2.c; under its current name build with "-o psp2_v1 psp2_v1.c" and run ./psp2_v1.)
 *
 * THEORY OF OPERATIONS
 * ====================
 * Version 1 of the h = 2 searcher, superseded by psp2.c (version 2) and kept unchanged for the scaling
 * comparison (logs/psp2_v1_scaling.log vs logs/psp2_v2_scaling.log; results in ../README.md, section 5.1).
 *
 * 1. What is computed
 *    For a basis A = {1 = a_1 < ... < a_k}, n_2(A) is the largest N such that every integer in [1,N] is a sum
 *    of at most two elements of A (repetition allowed); n(2,k) = max n_2(A) over k-element bases (written
 *    n_2(k) above) is A001212(k), known for k <= 24 (N2[] below).  For given K and T the program enumerates,
 *    prints and counts every K-basis whose sums cover [1,T]: T = n(2,K) + 1 is proof mode (must find nothing),
 *    T = n(2,K) lists the extremal bases.
 *
 * 2. Mathematics (as in psp2.c, whose header gives the derivations)
 *    S(A) = sums of at most two elements (0 included); n_2(A) + 1 = smallest integer missing from S(A).
 *    Adding x to A_j updates S |= E << x | bit(2x) with the element mask E = {0, a_1..a_j}.  The i-th smallest
 *    element adds at most i+1 sums, hence the gaps-test capacity CAP[j] = sum_{i=j+1..K} (i+1).
 *    -M: n_2(A) <= 2 a_K (the largest sum), so the largest element M = a_K of a basis covering [1,T] is
 *    >= ceil(T/2); it is also <= UB[K] = min(n(2,K-1) + 1, T).  Fixing M first puts M, M+1 and 2M into S at
 *    the root, every later element x also adds x + M (M is in E), and each free element must leave room below
 *    M for the free elements still to come: a_{j+1} <= M - (K-1-j).
 *
 * 3. Algorithm
 *    run() sets K, T, MSPLIT and the tables (setup_tables()), clamps SPLIT, and builds the root A_1 = {1}
 *    (E = {0,1}, S = {0,1,2}); with -M it makes one root per M in [ceil(T/2), UB[K]] (E and S also hold M).
 *    gen_prefixes() expands the root(s) single-threaded to depth SPLIT and stores the surviving nodes in
 *    prefixes[]; NTHREADS worker() threads claim them through an atomic counter and call dfs().
 *    dfs(nd) tries every x in (a_j, min(first zero of S, UB[j+1])] (with -M also x <= M - (K-1-j)): it copies
 *    the node, updates S, counts the node, applies the gaps test and recurses.  At the last free level (K, or
 *    K-1 with -M) it tests every admissible x for full coverage of [0,T] (popcount == T+1).
 *
 * 4. Data structures
 *    bits    256-bit set: integer v <-> bit (v & 63) of w[v >> 6].
 *    node    (240 bytes) j, a[] (a[0] = 0; with -M also a[K] = M), E = {0, a_1..a_j} (plus M), S = sum mask.
 *            S equals the true sum set on [0,T]; above T it lacks the doubles 2x > T (and, with -M, M+1 > T),
 *            so there it is only a subset.  Every child is a full copy of its parent (v2 uses level stacks).
 *    stats   per thread: nodes[j] = candidates tried at level j (gap-pruned ones and every x of the last free
 *            level included), pruned_gap[j] = candidates rejected by the gaps test, found = bases reported.
 *
 * 5. Parallelism
 *    Static split at depth SPLIT (default 9; clamped to [1, K-1], or to [1, K-2] with -M), dynamic scheduling
 *    via atomic_fetch_add on next_prefix, private statistics per thread, FOUND lines serialised by out_lock.
 *
 * 6. Command line
 *    -k K (3 <= K <= MAXK = 40; the upper bound is not checked), -t T (1 <= T <= 255), -j threads (default 10,
 *    clamped to [1,256]), -d splitdepth (default 9), -M (largest-element split), -q (no FOUND lines),
 *    -selftest (k = 3..16, plain and -M).  Values are parsed with atoi; a missing value reads argv[argc] (NULL).
 *
 * 7. Output (stdout; errors on stderr)
 *    k=K T=T msplit=0|1: N prefixes at depth D (Xs)
 *    FOUND k=K range>=T (range in [0,255] window = R): a_1 ... a_K       (with -M the last number is M = a_K)
 *    nodes per level: Lj=n ...   and   gap-pruned per level: Lj=n ...
 *    RESULT k=K T=T found=F total_nodes=N time=Xs rate=YM nodes/s threads=J
 *    -selftest: "selftest k=.. n=.. msplit=..: found F (expect C), found at n+1: G (expect 0)  OK|FAIL" lines,
 *    then SELFTEST PASSED or SELFTEST FAILED.  stderr: "T too large for NW=4", "bad arg ...", the usage line.
 *    Exit status 0 after a search or a passed selftest; 1 on bad arguments, T > 255 or a failed selftest.
 *    R is the first zero of S minus 1.  It is >= T but only a LOWER BOUND on n_2(A), because S lacks the
 *    doubles above T (-k 3 -t 5 prints R = 5 for {1,3,4}, whose range is 8); R = T exactly when n_2(A) = T.
 *    Node counts cover levels SPLIT+1..K only; time and rate include the prefix generation.
 *
 * 8. Exactness and known problems (the code is kept unchanged for the comparison)
 *    Plain mode uses the exact prunes of psp2.c and is exhaustive for T > n(2,K-1); for T <= n(2,K-1) every
 *    a_K <= T above a (K-1)-prefix that already covers [1,T] is reported.
 *    -M is NOT provably exact: setup_tables() keeps the plain capacity over the free levels,
 *    CAP[j] = sum_{i=j+1..K-1} (i+1), but each free a_i can add i+2 values (a_i + a_0..a_{i-1}, 2 a_i and
 *    a_i + M), so the gaps test is stricter than the provable bound by K-1-j.  A check made while documenting
 *    (k = 4..13, every T in (n(2,k-1), n(2,k)]) gave identical plain and -M counts, so no basis has been lost
 *    in practice.  -M also visited 5.5-6x more nodes than plain mode (logs/psp2_v1_scaling.log).
 *    run() clamps the global SPLIT and never restores it, so in -selftest every case after the first runs with
 *    SPLIT = 1: a single work item (one busy thread) in plain mode and one work item per M with -M.
 *    Limits: T <= 255; K <= MAXK = 40 (unchecked); 64-bit counters; at most 256 threads; no checkpoint/resume;
 *    calloc, realloc and pthread_create results are not checked; bits_test() is unused.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>

#define NW 4                    /* 256-bit masks: T <= 255 */
#define MAXK 40                 /* array bound: K <= MAXK (not checked by main) */

typedef struct { uint64_t w[NW]; } bits;  /* integer v <-> bit (v & 63) of w[v >> 6] */

/* Known extremal 2-ranges n_2(k), k = 0..24 (A001212, n_2(0)=0). */
static const int N2[] = {0, 2, 4, 8, 12, 16, 20, 26, 32, 40, 46, 54, 64, 72, 80,
                         92, 104, 116, 128, 140, 152, 164, 180, 196, 212};
#define N2_KNOWN 24

static int K, T;                /* length and target range */
static int UB[MAXK + 2];        /* UB[j]: max allowed a_j */
static int CAP[MAXK + 2];       /* CAP[j]: max new values the elements a_{j+1..k} can add */
static int SPLIT = 9;           /* prefix depth handed to threads */
static int NTHREADS = 10;       /* -j */
static int QUIET = 0;           /* -q */
static int MSPLIT = 0;          /* fix largest element first */
static bits TMASK;              /* bits 0..T set */

/*
 * bits_clear -- clear all 256 bits of a mask.
 *
 * Purpose: initialises TMASK in setup_tables().
 * Inputs:  b: mask.
 * Outputs: *b = 0.
 * Errors:  none.
 */
static inline void bits_clear(bits *b) { memset(b, 0, sizeof *b); }

/*
 * bits_set -- set bit i of a 256-bit mask.
 *
 * Purpose: records the integer i in an element mask E, a sum mask S or TMASK.
 * Inputs:  b: mask to modify; i: bit index, 0 <= i <= 255.
 * Outputs: b->w[i >> 6] gets bit (i & 63) set.
 * Errors:  i outside [0,255] writes outside *b (undefined behaviour); callers pass values <= T or the
 *          root's 0, 1, 2.
 */
static inline void bits_set(bits *b, int i) { b->w[i >> 6] |= 1ULL << (i & 63); }

/*
 * bits_test -- value (0 or 1) of bit i of a mask.
 *
 * Purpose: helper left from development; nothing in this file calls it.
 * Inputs:  b: mask; i: bit index, 0 <= i <= 255.
 * Outputs: 1 if bit i is set, else 0.
 * Errors:  i outside [0,255] reads outside *b (undefined behaviour).
 */
static inline int bits_test(const bits *b, int i) { return (b->w[i >> 6] >> (i & 63)) & 1; }

/*
 * bits_shl_or -- dst |= (src << s), bits shifted beyond 64*NW are dropped.
 *
 * Purpose: the core update.  With src = E and s = x it ORs in x + e for every e in E (the sums of x with
 *          0, a_1..a_j and, with -M, M), i.e. every new sum of the candidate x except 2x.
 * Inputs:  dst: mask to update; src: mask to shift; s: shift, s >= 0 (s >= 256 ORs in nothing).
 * Outputs: dst updated in place.
 * Errors:  none; s < 0 would be undefined (callers pass candidates x >= 2).
 */
static inline void bits_shl_or(bits *dst, const bits *src, int s) {
    int ws = s >> 6, bs = s & 63;  /* whole-word part and bit part of the shift */
    for (int i = NW - 1; i >= ws; i--) {  /* top down; words below ws receive nothing */
        uint64_t v = src->w[i - ws] << bs;
        if (bs && i - ws - 1 >= 0) v |= src->w[i - ws - 1] >> (64 - bs);  /* carry; bs != 0: no shift by 64 */
        dst->w[i] |= v;
    }
}

/*
 * bits_popcount_masked -- number of set bits of b inside [0,T].
 *
 * Purpose: counts the covered integers of [0,T]; the gaps test uses (T+1) minus this count, and the last
 *          free level accepts a basis when it equals T+1.
 * Inputs:  b: sum mask; global TMASK (bits 0..T, built by setup_tables()).
 * Outputs: popcount(b & TMASK), in 0..T+1.
 * Errors:  none.
 */
static inline int bits_popcount_masked(const bits *b) {
    int c = 0;
    for (int i = 0; i < NW; i++) c += __builtin_popcountll(b->w[i] & TMASK.w[i]);
    return c;
}

/*
 * bits_first_zero -- first zero bit (there is always one below 64*NW unless all set).
 *
 * Purpose: on a sum mask S it gives range + 1, the admissibility limit for the next element (exact while the
 *          range is below T, see the header, section 4), and the R printed by report().
 * Inputs:  b: mask.
 * Outputs: index of the lowest clear bit, or 64*NW = 256 if all bits are set.
 * Errors:  none.
 */
static inline int bits_first_zero(const bits *b) {
    for (int i = 0; i < NW; i++) {
        uint64_t v = ~b->w[i];
        if (v) return i * 64 + __builtin_ctzll(v);
    }
    return 64 * NW;
}

/* ---------- search state ---------- */
/* one search node; every child is a full copy of its parent (240 bytes) */
typedef struct {
    int j;                      /* number of elements placed (a_1..a_j) */
    int a[MAXK + 2];            /* a[0] = 0, a[1..j]; with -M also a[K] = M */
    bits E;                     /* element mask (bits at 0 and a_1..a_j, and M if MSPLIT) */
    bits S;                     /* sum mask (sums <= 255) */
} node;

typedef struct {
    uint64_t nodes[MAXK + 2];   /* nodes visited per level */
    uint64_t pruned_gap[MAXK + 2];  /* candidates rejected by the gaps test */
    uint64_t found;             /* bases reported */
} stats;

static node *prefixes;          /* work items: copies of the depth-SPLIT nodes */
static size_t nprefixes, prefix_cap;
static _Atomic size_t next_prefix;  /* index of the next unclaimed work item */
static pthread_mutex_t out_lock = PTHREAD_MUTEX_INITIALIZER;  /* serialises the FOUND lines */

/*
 * report -- count one basis that covers [1,T] and, unless -q, print it.
 *
 * Purpose: called at the last free level for every complete basis.
 * Inputs:  nd: the complete node (nd->j = K, or K-1 with -M, where nd->a[K] = M); st: thread statistics;
 *          globals K, T, QUIET, MSPLIT.
 * Outputs: st->found += 1; unless QUIET prints under out_lock
 *          "FOUND k=K range>=T (range in [0,255] window = R): a_1 ... a_K" on stdout and flushes.
 * Errors:  none.
 */
static void report(const node *nd, stats *st) {
    /* R = first zero of S, minus 1: >= T here, but only a lower bound on n_2(A) (S lacks the doubles > T) */
    int r = bits_first_zero(&nd->S) - 1;
    st->found++;
    if (QUIET) return;  /* -q: count only */
    pthread_mutex_lock(&out_lock);
    printf("FOUND k=%d range>=%d (range in [0,255] window = %d):", K, T, r);
    for (int i = 1; i <= nd->j; i++) printf(" %d", nd->a[i]);
    if (MSPLIT) printf(" %d", nd->a[K]);  /* M = a_K, after a_1..a_{K-1} */
    printf("\n");
    fflush(stdout);
    pthread_mutex_unlock(&out_lock);
}

/*
 * dfs -- depth-first search continuing from node nd; last level handled inline.
 *
 * Purpose: the hot loop of version 1.  Tries every admissible x for a_{j+1}; above the last free level it
 *          copies the node, applies the gaps test and recurses; at the last free level (K, or K-1 with -M) it
 *          only tests full coverage of [0,T].
 * Inputs:  nd: node with j = nd->j elements, SPLIT <= j < last free level; st: thread statistics; globals K,
 *          T, UB, CAP, TMASK, MSPLIT.
 * Outputs: updates st->nodes[] and st->pruned_gap[]; reports complete bases through report().
 * Errors:  none; the recursion depth is at most K - SPLIT, with one 240-byte node per frame.
 */
static void dfs(node *nd, stats *st) {
    int j = nd->j;
    int last = nd->a[j];
    /* with -M, S also holds the sums with M; all are >= M, above every x the room cap below allows */
    int r = bits_first_zero(&nd->S) - 1;          /* range of current set */
    int hi = r + 1;                                /* admissibility */
    if (hi > UB[j + 1]) hi = UB[j + 1];  /* element-wise bound, also caps at T */
    int klast = MSPLIT ? K - 1 : K;                /* last free level */
    if (MSPLIT && hi > nd->a[K] - (K - 1 - j)) hi = nd->a[K] - (K - 1 - j); /* room for the rest below M */
    if (j + 1 == klast) {
        /* final free element: just test coverage */
        for (int x = last + 1; x <= hi; x++) {
            node c = *nd;  /* full node copy per candidate */
            bits_shl_or(&c.S, &nd->E, x);
            if (2 * x <= T) bits_set(&c.S, 2 * x);
            st->nodes[j + 1]++;  /* every admissible x is a node here */
            if (bits_popcount_masked(&c.S) == T + 1) {  /* [0,T] fully covered */
                c.a[j + 1] = x; c.j = j + 1;
                report(&c, st);
            }
        }
        return;
    }
    for (int x = last + 1; x <= hi; x++) {
        node c = *nd;  /* child = copy of the parent */
        bits_shl_or(&c.S, &nd->E, x);
        if (2 * x <= T) bits_set(&c.S, 2 * x);  /* doubles above T are not recorded */
        st->nodes[j + 1]++;  /* every candidate is a node, pruned or not */
        int gaps = (T + 1) - bits_popcount_masked(&c.S);  /* uncovered integers in [1,T] */
        if (gaps > CAP[j + 1]) { st->pruned_gap[j + 1]++; continue; }  /* gaps test (-M: header 8) */
        bits_set(&c.E, x);
        c.a[j + 1] = x; c.j = j + 1;
        dfs(&c, st);
    }
}

/*
 * gen_prefixes -- enumerate admissible prefixes to depth SPLIT (no gaps test needed; apply anyway).
 *
 * Purpose: splits the tree for the worker threads (single-threaded, before they start); same candidate range
 *          (including the -M room limit) and gaps test as dfs(); nothing is counted.
 * Inputs:  nd: node at depth nd->j <= SPLIT; globals SPLIT, K, T, UB, CAP, TMASK, MSPLIT.
 * Outputs: appends copies of the depth-SPLIT nodes to prefixes[] (capacity doubled from 1024 as needed).
 * Errors:  realloc failure is not checked (NULL dereference).
 */
static void gen_prefixes(node *nd) {
    if (nd->j == SPLIT) {  /* depth SPLIT: store a copy as a work item */
        if (nprefixes == prefix_cap) {
            prefix_cap = prefix_cap ? prefix_cap * 2 : 1024;  /* doubling, 1024 first */
            prefixes = realloc(prefixes, prefix_cap * sizeof(node));  /* unchecked */
        }
        prefixes[nprefixes++] = *nd;
        return;
    }
    int j = nd->j, last = nd->a[j];
    int r = bits_first_zero(&nd->S) - 1;
    int hi = r + 1;
    if (hi > UB[j + 1]) hi = UB[j + 1];
    if (MSPLIT && hi > nd->a[K] - (K - 1 - j)) hi = nd->a[K] - (K - 1 - j);  /* room below M */
    for (int x = last + 1; x <= hi; x++) {
        node c = *nd;
        bits_shl_or(&c.S, &nd->E, x);
        if (2 * x <= T) bits_set(&c.S, 2 * x);
        int gaps = (T + 1) - bits_popcount_masked(&c.S);
        if (gaps > CAP[j + 1]) continue;
        bits_set(&c.E, x);
        c.a[j + 1] = x; c.j = j + 1;
        gen_prefixes(&c);
    }
}

static stats *tstats;  /* NTHREADS stats slots, one per thread, allocated by run() */
/*
 * worker -- thread body: claim work items from the shared queue and search below each one.
 *
 * Purpose: dynamic load balancing over the nodes stored by gen_prefixes().
 * Inputs:  arg: this thread's stats slot (&tstats[i]); globals prefixes, nprefixes (read-only here) and
 *          next_prefix (atomic).
 * Outputs: accumulates nodes, pruned_gap and found into *arg; returns NULL.
 * Errors:  none.
 */
static void *worker(void *arg) {
    stats *st = (stats *)arg;
    for (;;) {
        size_t i = atomic_fetch_add(&next_prefix, 1);  /* claim the next work item */
        if (i >= nprefixes) break;  /* queue exhausted */
        node nd = prefixes[i];  /* private copy of the work item */
        dfs(&nd, st);
    }
    return NULL;
}

/*
 * now -- monotonic wall-clock time in seconds.
 *
 * Purpose: timing of the prefix generation and of the search.
 * Inputs:  none.
 * Outputs: CLOCK_MONOTONIC time as a double.
 * Errors:  none (the clock_gettime result is not checked).
 */
static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/*
 * setup_tables -- build TMASK, UB[] and CAP[] for the current K, T and MSPLIT.
 *
 * Purpose: the pruning tables read at every level of the search.
 * Inputs:  globals K (<= MAXK), T (<= 255), MSPLIT, N2[].
 * Outputs: for j = 0..K+1: TMASK = bits 0..T; UB[j] = min(n(2,j-1) + 1, T) (T beyond the table);
 *          CAP[j] = sum_{i=j+1..K} (i+1), or with -M sum_{i=j+1..K-1} (i+1) (too small, header section 8).
 * Errors:  none; K > MAXK overflows UB[] and CAP[].
 */
static void setup_tables(void) {
    bits_clear(&TMASK);
    for (int i = 0; i <= T; i++) bits_set(&TMASK, i);  /* TMASK = bits 0..T */
    for (int j = 0; j <= K + 1; j++) {
        /* a_j <= n_2(j-1) + 1 if known, else <= T */
        UB[j] = (j - 1 <= N2_KNOWN && j >= 1) ? N2[j - 1] + 1 : T;
        if (UB[j] > T) UB[j] = T;
    }
    for (int j = 0; j <= K + 1; j++) {
        int c = 0;
        for (int i = j + 1; i <= (MSPLIT ? K - 1 : K); i++) c += i + 1;  /* a_i adds <= i+1 values */
        /* -M: the original comment on the next line read "M's sums with later elements are counted in their
           (i+1)".  Not so: (i+1) counts a_i + a_0..a_{i-1} and 2 a_i, and a_i + M is one more new value, so with
           -M CAP[j] is K-1-j too small and the gaps test is not provably exact (header, section 8). */
        if (MSPLIT) c += 0; /* no-op; see the note above */
        CAP[j] = c;
    }
}

/*
 * run -- run one search; returns number of bases found with range >= T.
 *
 * Purpose: sets the globals and tables, generates the work items (one root, or one root per M with -M), runs
 *          NTHREADS workers and merges their statistics.
 * Inputs:  k: basis size, 3 <= k <= MAXK (upper bound not checked); t: target, 1 <= t <= 255; msplit: -M;
 *          verbose: print the prefix line, the per-level counts and the RESULT line; globals SPLIT, NTHREADS.
 * Outputs: returns the number of bases reported; sets K, T, MSPLIT, the tables and prefixes[]; leaves SPLIT
 *          clamped (it is not restored, see the header, section 8).
 * Errors:  t > 255: "T too large for NW=4" on stderr, exit(1).  calloc and pthread_create failures are not
 *          checked.
 */
static uint64_t run(int k, int t, int msplit, int verbose) {
    K = k; T = t; MSPLIT = msplit;
    if (T > 64 * NW - 1) { fprintf(stderr, "T too large for NW=%d\n", NW); exit(1); }  /* masks hold 0..255 */
    setup_tables();
    double t0 = now();
    nprefixes = 0;
    /* clamp to [1, K-1] (plain) or [1, K-2] (-M); the clamped value is stored in SPLIT and never restored */
    int split = SPLIT; if (split >= (MSPLIT ? K - 1 : K)) split = (MSPLIT ? K - 1 : K) - 1;
    if (split < 1) split = 1;
    SPLIT = split;
    /* root A_1 = {1}: E = {0,1}, S = {0,1,2} (0+0, 0+1, 1+1) */
    node root; memset(&root, 0, sizeof root);
    root.j = 1; root.a[0] = 0; root.a[1] = 1;
    bits_set(&root.E, 0); bits_set(&root.E, 1);
    bits_set(&root.S, 0); bits_set(&root.S, 1); bits_set(&root.S, 2);
    if (!MSPLIT) {
        gen_prefixes(&root);
    } else {
        /* a_K = M ranges over [ceil(T/2), UB[K]] (n_2(A) <= 2 a_K).  M must also be <= n_2(A_{K-1})+1; the DFS does
           not enforce that via hi, but for T > n(2,K-1) the final full-coverage test implies it (otherwise the
           integer n_2(A_{K-1})+1 < M stays uncovered). */
        for (int M = (T + 1) / 2; M <= UB[K]; M++) {
            node rm = root;
            rm.a[K] = M;
            bits_set(&rm.E, M);  /* M is in E: every later x also adds x + M */
            /* sums of M with a_0 = 0, a_1 = 1 and itself (M+1 and 2M only if <= T) */
            bits_set(&rm.S, M); if (M + 1 <= T) bits_set(&rm.S, M + 1); if (2 * M <= T) bits_set(&rm.S, 2 * M);
            gen_prefixes(&rm);  /* all M share one work queue */
        }
    }
    double t1 = now();
    if (verbose) { printf("k=%d T=%d msplit=%d: %zu prefixes at depth %d (%.2fs)\n", K, T, MSPLIT, nprefixes, SPLIT, t1 - t0); fflush(stdout); }
    atomic_store(&next_prefix, 0);  /* empty the claim counter */
    tstats = calloc(NTHREADS, sizeof(stats));  /* zeroed counters (unchecked) */
    pthread_t th[256];  /* NTHREADS <= 256 (main clamps) */
    for (int i = 0; i < NTHREADS; i++) pthread_create(&th[i], NULL, worker, &tstats[i]);
    for (int i = 0; i < NTHREADS; i++) pthread_join(th[i], NULL);
    double t2 = now();
    /* merge the per-thread statistics; levels <= SPLIT were never counted */
    stats tot; memset(&tot, 0, sizeof tot);
    for (int i = 0; i < NTHREADS; i++) {
        tot.found += tstats[i].found;
        for (int j = 0; j <= K + 1; j++) { tot.nodes[j] += tstats[i].nodes[j]; tot.pruned_gap[j] += tstats[i].pruned_gap[j]; }
    }
    uint64_t total = 0; for (int j = 0; j <= K + 1; j++) total += tot.nodes[j];
    if (verbose) {
        printf("nodes per level:");
        for (int j = 2; j <= K; j++) if (tot.nodes[j]) printf(" L%d=%llu", j, (unsigned long long)tot.nodes[j]);
        printf("\n");
        printf("gap-pruned per level:");
        for (int j = 2; j <= K; j++) if (tot.pruned_gap[j]) printf(" L%d=%llu", j, (unsigned long long)tot.pruned_gap[j]);
        printf("\n");
        printf("RESULT k=%d T=%d found=%llu total_nodes=%llu time=%.2fs rate=%.2fM nodes/s threads=%d\n",
               K, T, (unsigned long long)tot.found, (unsigned long long)total, t2 - t0, total / (t2 - t0) / 1e6, NTHREADS);
        fflush(stdout);
    }
    free(tstats);
    return tot.found;
}

/*
 * selftest -- check both modes against the published n(2,k) and extremal-basis counts for k = 3..16.
 *
 * Purpose: for each case and mode runs listing mode at T = n(2,k) (must find exactly `count` bases) and
 *          proof mode at T = n(2,k) + 1 (must find none).
 * Inputs:  the case table; globals NTHREADS, SPLIT.
 * Outputs: one "selftest k=.. n=.. msplit=..: ... OK|FAIL" line per case and mode on stdout, then SELFTEST
 *          PASSED or SELFTEST FAILED.  QUIET is forced on during the runs and restored.
 * Errors:  any mismatch: SELFTEST FAILED and exit(1).
 */
static void selftest(void) {
    /* expected number of extremal bases, from Challis-Robinson / Kohonen tables */
    struct { int k, n, count; } cases[] = {
        {3, 8, 1}, {4, 12, 1}, {5, 16, 1}, {6, 20, 5}, {7, 26, 3}, {8, 32, 2}, {9, 40, 1},
        {10, 46, 2}, {11, 54, 4}, {12, 64, 1}, {13, 72, 1}, {14, 80, 3}, {15, 92, 1}, {16, 104, 1}
    };
    int ok = 1;
    int saveq = QUIET; QUIET = 1;  /* no FOUND lines during the checks */
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        for (int ms = 0; ms <= 1; ms++) {
            uint64_t f = run(cases[i].k, cases[i].n, ms, 0);  /* listing mode */
            uint64_t g = run(cases[i].k, cases[i].n + 1, ms, 0);  /* proof mode: must find nothing */
            int good = (f == (uint64_t)cases[i].count && g == 0);
            printf("selftest k=%2d n=%3d msplit=%d: found %llu (expect %d), found at n+1: %llu (expect 0)  %s\n",
                   cases[i].k, cases[i].n, ms, (unsigned long long)f, cases[i].count, (unsigned long long)g, good ? "OK" : "FAIL");
            if (!good) ok = 0;
        }
    }
    QUIET = saveq;
    printf(ok ? "SELFTEST PASSED\n" : "SELFTEST FAILED\n");
    if (!ok) exit(1);
}

/*
 * main -- parse the command line and run either the selftest or one verbose search.
 *
 * Purpose: command-line driver; the options are described in the header, section 6.
 * Inputs:  argc, argv.
 * Outputs: sets NTHREADS (clamped to [1,256]), SPLIT, QUIET and the -M flag; then the output of selftest()
 *          or run().
 * Errors:  unknown option: "bad arg ..." on stderr, status 1; missing or invalid -k/-t (k < 3 or t < 1): the
 *          usage line on stderr, status 1; an option without its value reads argv[argc] (NULL, crash).
 *          Returns 0 after a search whether or not anything was found.
 */
int main(int argc, char **argv) {
    int k = 0, t = 0, st = 0, ms = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-k")) k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t")) t = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-j")) NTHREADS = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-d")) SPLIT = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-M")) ms = 1;
        else if (!strcmp(argv[i], "-q")) QUIET = 1;
        else if (!strcmp(argv[i], "-selftest")) st = 1;
        else { fprintf(stderr, "bad arg %s\n", argv[i]); return 1; }
    }
    if (NTHREADS < 1) NTHREADS = 1; if (NTHREADS > 256) NTHREADS = 256;  /* th[] in run() has 256 slots */
    if (st) { selftest(); return 0; }  /* selftest() exits 1 itself on failure */
    /* k <= MAXK = 40 is not checked; run() rejects t > 255 */
    if (k < 3 || t < 1) { fprintf(stderr, "usage: psp2 -k K -t T [-j threads] [-d splitdepth] [-M] [-q] | -selftest\n"); return 1; }
    run(k, t, ms, 1);  /* status 0 whether or not bases were found */
    return 0;
}
