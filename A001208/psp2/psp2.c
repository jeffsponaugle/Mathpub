/*
 * psp2.c -- exhaustive search for extremal additive 2-bases
 *           (global postage stamp problem with h = 2 stamps, OEIS A001212).
 *
 * Find every set A = {a_0=0 < a_1=1 < a_2 < ... < a_k} of k positive
 * integers such that every integer in [1,T] is a sum of at most two elements
 * of A.  Running with T = best known + 1 and finding nothing proves
 * n_2(k) = best known; running with T = best known lists all extremal bases.
 *
 * Challis-style depth-first search ("K-program") with exact prunes only:
 *   admissibility        a_{j+1} <= n_2(A_j) + 1  (prefix must cover [1, a_{j+1}-1])
 *   element-wise bound   a_j <= n_2(j-1) + 1       (known A001212 values)
 *   gaps test            #uncovered values in [1,T] after A_j <= sum_{i=j+1..k} (i+1)
 *                        (a new element a_i creates at most i+1 new sums)
 *   last element         the smallest remaining gap g must be a_k + a_p (or 2 a_k),
 *                        so a_k is one of at most k+1 candidates g - a_p, g/2.
 * Coverage of [0,T] is a 256-bit mask (T <= 255); per-thread level stacks, no copying.
 *
 * Build:  clang -O3 -mcpu=apple-m1 -std=c11 -pthread -o psp2 psp2.c
 * Usage:  ./psp2 -k K -t T [-j threads] [-d splitdepth] [-q]
 *         ./psp2 -selftest
 *
 * THEORY OF OPERATIONS
 * ====================
 * This is version 2 of the h = 2 searcher.  psp2_v1.c is the first version, kept for the scaling comparison
 * (logs/psp2_v1_scaling.log vs logs/psp2_v2_scaling.log); ../psp2gpu/psp2gpu.m is the Metal port of this file.
 * Measured proof-mode node counts per k and the cost estimate for A001212(25) are in ../README.md, section 5.1.
 *
 * 1. What is computed
 *    For a basis A = {1 = a_1 < a_2 < ... < a_k}, n_2(A) is the largest N such that every integer in [1,N] is a
 *    sum of at most two elements of A (repetition allowed); N+1 is the first unrepresentable integer.  With
 *    a_0 = 0 adjoined, "at most two elements of A" is "exactly two of a_0..a_k".  The extremal value
 *    n(2,k) = max n_2(A) over all k-element bases (written n_2(k) above) is A001212(k), offset 1.  It is known
 *    for k <= 24 (Kohonen & Corander 2014 needed 606 CPU-days for k = 24); N2[] holds n(2,0..24).
 *    Given K and T the program enumerates every K-element basis with n_2(A) >= T, i.e. every basis whose sums
 *    cover [1,T], prints each one and counts them:
 *      proof mode    T = b + 1 for the best known value b: found=0 proves n(2,K) <= b, hence n(2,K) = b
 *                    because a basis attaining b is known;
 *      listing mode  T = b = n(2,K): lists every extremal basis (no basis exceeds n(2,K)).
 *    The per-level node counts show how the cost of the search grows with k.
 *
 * 2. Mathematics used by the code
 *    S(A) = {a_p + a_q : 0 <= p <= q <= k} is the set of sums of at most two elements (0 included), and
 *    n_2(A) + 1 is the smallest integer missing from S(A).  The sums that involve a new element x > a_j are
 *    exactly x + a_p (p = 0..j) and 2x, so with the element mask E = {0, a_1, .., a_j} as a bitset
 *    (bit v = integer v)
 *        S(A_j + {x}) = S(A_j) | (E << x) | bit(2x)                    (one 256-bit shift-or)
 *    and the i-th smallest element, placed after a_1..a_{i-1}, adds at most i+1 new values.
 *    Admissibility: if a_{j+1} > n_2(A_j) + 1, the integer v = n_2(A_j) + 1 is never covered (every sum that
 *      uses a later element is >= a_{j+1} > v), so the finished basis has n_2 < v.  The DFS therefore tries
 *      a_{j+1} = a_j + 1 .. n_2(A_j) + 1 only; n_2(A_j) + 1 is the first zero bit of the sum mask.
 *    Element-wise bound: n_2(A_{j-1}) <= n(2,j-1), so a_j <= n(2,j-1) + 1 = UB[j] (from N2[]).
 *    Gaps test: once A_j is fixed, the elements a_{j+1}..a_K add at most CAP[j] = sum_{i=j+1..K} (i+1) values,
 *      so a prefix that leaves more than CAP[j] integers of [1,T] uncovered cannot be completed.
 *    Last element: the smallest uncovered g in [1,T] after A_{K-1} must be a_K + a_p (p = 0..K-1) or 2 a_K,
 *      and admissibility gives a_K <= g.  So a_K is one of g - a_p, g/2 (at most K+1 candidates), each tested
 *      with one shift-or and a subset test instead of a loop over every admissible a_K.
 *
 * 3. Algorithm
 *    run()           sets K, T, TMASK (bits 0..T), UB[j] = min(n(2,j-1) + 1, T) (just T beyond the table) and
 *                    CAP[j]; clamps the split depth SPLIT (-d) to [1, K-2] for this run.
 *    root            A_1 = {1}: E = {0,1}, S = {0,1,2} (0+0, 0+1, 1+1) in L[1]; a[0] = 0, a[1] = 1.
 *    gen_prefixes()  single-threaded DFS from the root to depth SPLIT with the same candidate range and gaps
 *                    test as dfs(); every surviving prefix a_1..a_SPLIT (with its masks) becomes a work item.
 *    worker()        NTHREADS threads take work items from an atomic counter and call dfs(SPLIT, ...).
 *    dfs(j)          for x = a_j + 1 .. min(first zero of S_j, UB[j+1]): S_{j+1} = S_j | E_j << x | bit(2x)
 *                    (2x only when 2x <= T); count the node; prune if more than CAP[j+1] integers of [1,T]
 *                    are uncovered; else E_{j+1} = E_j + {x} and recurse.  With K-1 elements it calls leaf().
 *    leaf()          G = [0,T] minus S_{K-1}, g = min G; for each candidate z in (a_{K-1}, min(g, UB[K])] it
 *                    builds the new sums M = E << z | bit(2z) and reports the basis iff G is a subset of M.
 *    run() finally merges the per-thread statistics and prints the summary lines.
 *
 * 4. Data structures and invariants
 *    bits      256-bit set: integer v <-> bit (v & 63) of w[v >> 6].
 *    a[]       a[0] = 0, a[1] = 1, a[i] = i-th element of the current path; entries above the depth are stale.
 *    L[j]      lvl {E, S} after j elements.  E = {0, a_1, .., a_j} exactly.  S equals S(A_j) on [0,T]; on
 *              T+1..255 it holds the sums a_p + a_q (p < q) but not the doubles 2a_q > T, so there it is only a
 *              subset of S(A_j).  Hence counts are masked with TMASK, and the first zero of S is exactly
 *              n_2(A_j) + 1 whenever n_2(A_j) < T (otherwise it is > T and UB[] <= T is the binding limit).
 *              dfs at depth j reads only L[j] and writes only L[j+1], so a thread's L[] array is its stack:
 *              backtracking needs no undo and no node copies: a candidate costs one 32-byte mask copy, one
 *              shift-or and one masked popcount.
 *    prefix_t  one work item: the whole a[] plus L[SPLIT] (232 bytes); prefixes[] grows by doubling.
 *    stats     per thread (tstats[i]): nodes[j] = level-j candidates generated (gap-pruned ones included; at
 *              level K only the leaf candidates that pass the range filter), pruned[j] = level-j candidates
 *              rejected by the gaps test, found = bases reported.
 *    K, T, UB[], CAP[], TMASK, SPLIT and prefixes[] are written by run() before the threads start and only
 *    read while they run.
 *
 * 5. Parallelism
 *    Static split at depth SPLIT, dynamic scheduling: atomic_fetch_add on next_prefix hands out one work item
 *    at a time, so a thread that draws small subtrees simply takes more of them.  Each thread has a private
 *    a[], L[] and stats slot; FOUND lines are serialised by out_lock, so they never interleave, but their order
 *    is nondeterministic.  Subtree sizes vary widely, so good balance needs many more work items than threads
 *    (the count is printed); raise -d if the end of a run has a single busy thread.
 *
 * 6. Command line
 *    -k K        basis size; main requires K >= 3.  K <= MAXK = 40 is assumed but NOT checked (array overflow).
 *    -t T        target, 1 <= T <= 255 (run() exits on T > 255).
 *    -j threads  worker threads, default 10, clamped to [1,256].
 *    -d depth    split depth, default 8, clamped per run to [1, K-2].
 *    -q          do not print FOUND lines (the counts are still reported).
 *    -selftest   run the built-in checks (k = 3..17) instead of a search; -j and -d still apply.
 *    Values are parsed with atoi (no validation); an option given without its value reads argv[argc] (NULL).
 *
 * 7. Output (stdout, flushed; stderr only for errors)
 *    k=K T=T: N prefixes at depth D (Xs)                      after the prefix generation
 *    FOUND k=K range>=T: a_1 a_2 ... a_K                      one per basis, unless -q
 *    nodes per level: Lj=n ...                                levels with a nonzero count only
 *    gap-pruned per level: Lj=n ...
 *    RESULT k=K T=T found=F total_nodes=N time=Xs rate=YM nodes/s threads=J
 *    -selftest: "selftest k=.. n=..: found F (expect C), at n+1: G (expect 0) Xs OK|FAIL" per case, then
 *    SELFTEST PASSED or SELFTEST FAILED.  stderr: "T too large for NW=4", "bad arg ...", the usage line.
 *    Exit status 0 after a completed search (whatever F is) or a passed selftest; 1 on bad arguments, T > 255
 *    or a failed selftest.  Node counts cover levels SPLIT+1..K only (gen_prefixes counts nothing), so
 *    total_nodes depends slightly on -d; time and rate include the prefix generation.
 *
 * 8. Why the result is exact
 *    Every prune is a necessary condition for a K-basis with n_2(A) >= T, provided T > n(2,K-1) (true for
 *    T >= n(2,K), since n(2,k) is strictly increasing): then every element is <= a_K <= n_2(A_{K-1}) + 1
 *    <= n(2,K-1) + 1 <= T, so capping candidates at T loses nothing, and leaf()'s "already covered" branch
 *    never fires.  UB[] relies on the published n(2,j), j <= 24 (themselves results of exhaustive searches).
 *    Each basis is one increasing path, and the leaf candidates are distinct (g - a_p decreases with p, and
 *    g/2 > a_{K-1} >= a_p implies g - a_p > g/2), so each basis is reported exactly once.  Only integer
 *    arithmetic is used; nothing is heuristic or random.
 *    For T <= n(2,K-1) the count is NOT the number of bases: elements are still capped at T, and a
 *    (K-1)-prefix that already covers [1,T] is reported once, with the placeholder a_K = a_{K-1} + 1.
 *
 * 9. Limits
 *    T <= 255 (NW = 4 words of 64 bits): enough for k = 25 and 26 if n(2,25) and n(2,26) are the expected 228
 *    and 244 (README).  K <= MAXK = 40 (unchecked).  N2[] stops at k = 24; beyond it UB[] is just T.  Counters
 *    are 64-bit per level and thread (1.8e19), far above the 3e17-8e17 nodes estimated for k = 25 (README).
 *    At most 256 threads.  prefixes[] (232 bytes per item) is never freed; calloc, realloc and pthread_create
 *    results are not checked.
 *
 * 10. Build and history
 *    Build with the line above (C11 + pthreads; -mcpu=apple-m1 targets Apple Silicon).  No checkpoint or
 *    resume: a search must finish in one process.  Changes from psp2_v1.c: per-thread level stacks instead of
 *    copying a whole node per child; the first-gap candidate rule at the last level (so the level-K node
 *    counts of v1 and v2 differ); no -M largest-element split (it visited 5.5-6x more nodes at k = 16, 17 in
 *    logs/psp2_v1_scaling.log); SPLIT restored after every run(); default split depth 8 instead of 9;
 *    selftest extended to k = 17.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>

#define NW 4                    /* 4 x 64-bit words: 256-bit masks, so T <= 255 */
#define MAXK 40                 /* array bound: K <= MAXK (not checked by main) */
typedef struct { uint64_t w[NW]; } bits;  /* integer v <-> bit (v & 63) of w[v >> 6] */

/* N2[k] = n(2,k) = A001212(k) for k = 1..24, N2[0] = 0: the proven values behind UB[] */
static const int N2[] = {0, 2, 4, 8, 12, 16, 20, 26, 32, 40, 46, 54, 64, 72, 80,
                         92, 104, 116, 128, 140, 152, 164, 180, 196, 212};
#define N2_KNOWN 24             /* largest k in N2[] */

static int K, T;                /* basis size and target: find K-bases covering [1,T] */
static int UB[MAXK + 2];        /* UB[j]: largest allowed a_j, min(n(2,j-1)+1, T) */
static int CAP[MAXK + 2];       /* CAP[j] = sum_{i=j+1..K} (i+1): most new values a_{j+1..K} can add */
static int SPLIT = 8, NTHREADS = 10, QUIET = 0;  /* -d, -j, -q */
static bits TMASK;              /* bits 0..T set */

/*
 * bits_set -- set bit i of a 256-bit mask.
 *
 * Purpose: records the integer i in an element mask E, a sum mask S or TMASK.
 * Inputs:  b: mask to modify; i: bit index, 0 <= i <= 255.
 * Outputs: b->w[i >> 6] gets bit (i & 63) set.
 * Errors:  i outside [0,255] writes outside *b (undefined behaviour); callers pass elements and doubles
 *          <= T, 0..T, or the root's 0, 1, 2.
 */
static inline void bits_set(bits *b, int i) { b->w[i >> 6] |= 1ULL << (i & 63); }

/*
 * bits_shl_or -- dst |= src << s on 256-bit masks; bits shifted past bit 255 are dropped.
 *
 * Purpose: the core update.  With src = E = {0, a_1..a_j} and s = x it ORs in the sums x + a_p (p = 0..j),
 *          i.e. every new sum of the candidate x except 2x.
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
 * Purpose: counts the covered integers of [0,T]; the gaps test uses (T+1) minus this count.
 * Inputs:  b: sum mask; global TMASK (bits 0..T, built by run()).
 * Outputs: popcount(b & TMASK), in 0..T+1.
 * Errors:  none.
 */
static inline int bits_popcount_masked(const bits *b) {
    int c = 0; for (int i = 0; i < NW; i++) c += __builtin_popcountll(b->w[i] & TMASK.w[i]); return c;
}

/*
 * bits_first_zero -- index of the lowest clear bit of b, or 256 if all 256 bits are set.
 *
 * Purpose: on a sum mask S it yields n_2(A_j) + 1, the admissibility limit for the next element (exact when
 *          that value is <= T, see the header, section 4).
 * Inputs:  b: mask.
 * Outputs: 0..256.
 * Errors:  none.
 */
static inline int bits_first_zero(const bits *b) {
    for (int i = 0; i < NW; i++) { uint64_t v = ~b->w[i]; if (v) return i * 64 + __builtin_ctzll(v); }
    return 64 * NW;
}

typedef struct { bits E, S; } lvl;  /* state after j elements (header, section 4) */
typedef struct { uint64_t nodes[MAXK + 2], pruned[MAXK + 2], found; } stats;  /* per-thread counters */
typedef struct { int a[MAXK + 2]; lvl L; } prefix_t;  /* work item: a[] and L[SPLIT] */

static prefix_t *prefixes; static size_t nprefixes, prefix_cap;  /* work items from gen_prefixes() */
static _Atomic size_t next_prefix;  /* index of the next unclaimed work item */
static pthread_mutex_t out_lock = PTHREAD_MUTEX_INITIALIZER;  /* serialises the FOUND lines */

/*
 * report -- count one basis that covers [1,T] and, unless -q, print it.
 *
 * Purpose: called by leaf() for every complete basis found.
 * Inputs:  a: a[1..j] = the basis (a[0] = 0 is not printed); j: its size (K); st: the calling thread's
 *          statistics; globals K, T, QUIET.
 * Outputs: st->found += 1; unless QUIET prints "FOUND k=K range>=T: a_1 ... a_K" on stdout under out_lock
 *          (lines from different threads never interleave) and flushes.
 * Errors:  none.
 */
static void report(const int *a, int j, stats *st) {
    st->found++;
    if (QUIET) return;  /* -q: count only */
    pthread_mutex_lock(&out_lock);
    printf("FOUND k=%d range>=%d:", K, T);
    for (int i = 1; i <= j; i++) printf(" %d", a[i]);
    printf("\n"); fflush(stdout);
    pthread_mutex_unlock(&out_lock);
}

/*
 * leaf -- last level: choose a_K to cover every remaining gap.
 *
 * Purpose: first-gap candidate rule.  The smallest uncovered g must be a_K + a_p (p = 0..K-1) or 2 a_K, so
 *          a_K is g - a_p or g/2 (at most K+1 candidates); a candidate is accepted iff its new sums cover
 *          every remaining gap.  Replaces psp2_v1.c's loop over every admissible a_K.
 * Inputs:  j: number of placed elements, j = K-1; a: a[0..j]; L: level-j state (E, S); st: thread
 *          statistics; globals T, TMASK, UB.
 * Outputs: a[j+1] receives each accepted value (scratch) and report() is called once per basis;
 *          st->nodes[K] counts the candidates that pass the range filter a_{K-1} < z <= min(g, UB[K]).
 *          If the K-1 elements already cover [1,T] (possible only when T <= n(2,K-1)) it reports once with
 *          the placeholder a_K = a_{K-1} + 1 and counts no node.
 * Errors:  none.
 */
static inline void leaf(int j, int *a, const lvl *L, stats *st) {
    bits G; int any = 0;  /* G = uncovered integers of [0,T]; any = (G nonempty) */
    for (int i = 0; i < NW; i++) { G.w[i] = TMASK.w[i] & ~L->S.w[i]; any |= (G.w[i] != 0); }
    int last = a[j];
    /* only when T <= n(2,K-1): report the covering (K-1)-prefix once, with a placeholder a_K */
    if (!any) { /* already covered with K-1 elements */ a[j + 1] = last + 1; report(a, j + 1, st); return; }
    /* g = smallest gap = n_2(A_{K-1}) + 1 <= T */
    int g = 0; for (int i = 0; i < NW; i++) if (G.w[i]) { g = i * 64 + __builtin_ctzll(G.w[i]); break; }
    int hi = g; if (hi > UB[j + 1]) hi = UB[j + 1];  /* admissibility a_K <= g; element-wise bound */
    /* candidates z = g - a_p (p = 0..j) and z = g/2 */
    /* p = j+1 is the g/2 candidate (even g only); all candidates are distinct (header, section 8) */
    for (int p = 0; p <= j + 1; p++) {
        int z;
        if (p <= j) z = g - a[p]; else { if (g & 1) break; z = g / 2; }
        if (z <= last || z > hi) continue;  /* keep a_{K-1} < z <= min(g, UB[K]) */
        st->nodes[j + 1]++;  /* level-K node = candidate in range */
        /* M = the sums z + a_p (p = 0..K-1) and 2z (if <= T); accept z iff G is a subset of M */
        bits M = {{0}}; bits_shl_or(&M, &L->E, z); if (2 * z <= T) bits_set(&M, 2 * z);
        int ok = 1; for (int i = 0; i < NW; i++) if (G.w[i] & ~M.w[i]) { ok = 0; break; }
        if (ok) { a[j + 1] = z; report(a, j + 1, st); }
    }
}

/*
 * dfs -- depth-first search for a_{j+1}, a_{j+2}, ... below the fixed path a_1..a_j.
 *
 * Purpose: the hot loop.  For each x in (a_j, min(n_2(A_j) + 1, UB[j+1])] it builds the level-(j+1) sum
 *          mask, applies the gaps test and recurses; with K-1 elements placed it hands over to leaf().
 * Inputs:  j: number of placed elements, SPLIT <= j <= K-1; a: a[0..j] valid; L: this thread's level stack,
 *          L[j] valid; st: thread statistics; globals K, T, UB, CAP, TMASK.
 * Outputs: writes a[j+1..K] and L[j+1..K-1] as scratch; updates st->nodes[] and st->pruned[]; bases are
 *          reported through leaf().
 * Errors:  none; the recursion depth is at most K - SPLIT.
 */
static void dfs(int j, int *a, lvl *L, stats *st) {
    if (j + 1 == K) { leaf(j, a, &L[j], st); return; }  /* K-1 elements placed */
    int last = a[j];
    /* admissibility a_{j+1} <= n_2(A_j) + 1 (exact if <= T; otherwise the UB <= T cap below binds) */
    int hi = bits_first_zero(&L[j].S);           /* = range + 1 */
    if (hi > UB[j + 1]) hi = UB[j + 1];  /* element-wise bound, also caps at T */
    for (int x = last + 1; x <= hi; x++) {
        /* S_{j+1} = S_j | E_j << x | bit(2x); doubles above T are not recorded (header, section 4) */
        L[j + 1].S = L[j].S;
        bits_shl_or(&L[j + 1].S, &L[j].E, x);
        if (2 * x <= T) bits_set(&L[j + 1].S, 2 * x);
        st->nodes[j + 1]++;  /* every candidate is a node, pruned or not */
        int gaps = (T + 1) - bits_popcount_masked(&L[j + 1].S);  /* uncovered integers in [1,T] */
        if (gaps > CAP[j + 1]) { st->pruned[j + 1]++; continue; }  /* gaps test */
        L[j + 1].E = L[j].E; bits_set(&L[j + 1].E, x);  /* E is built only for survivors */
        a[j + 1] = x;
        dfs(j + 1, a, L, st);
    }
}

/*
 * gen_prefixes -- enumerate every surviving prefix a_1..a_SPLIT and store it as a work item.
 *
 * Purpose: splits the search tree for the worker threads.  Same candidate range and gaps test as dfs(), run
 *          single-threaded before the workers start; nothing is counted in the statistics.
 * Inputs:  j: current depth (1 at the root); a: a[0..j]; L: level stack with L[j] valid; globals SPLIT, T,
 *          UB, CAP, TMASK.
 * Outputs: appends {a[0..MAXK+1], L[SPLIT]} to prefixes[] (capacity doubled from 4096 as needed) and
 *          increments nprefixes; writes a[j+1..SPLIT] and L[j+1..SPLIT] as scratch.
 * Errors:  realloc failure is not checked (NULL dereference).
 */
static void gen_prefixes(int j, int *a, lvl *L) {
    if (j == SPLIT) {  /* depth SPLIT: store one work item */
        /* grow by doubling (4096 entries first); realloc failure is not checked */
        if (nprefixes == prefix_cap) { prefix_cap = prefix_cap ? 2 * prefix_cap : 4096; prefixes = realloc(prefixes, prefix_cap * sizeof(prefix_t)); }
        memcpy(prefixes[nprefixes].a, a, sizeof(int) * (MAXK + 2));  /* whole a[]; entries > SPLIT are stale */
        prefixes[nprefixes].L = L[j];  /* masks at depth SPLIT */
        nprefixes++; return;
    }
    /* same candidate range and gaps test as dfs(), but nothing is counted */
    int last = a[j];
    int hi = bits_first_zero(&L[j].S); if (hi > UB[j + 1]) hi = UB[j + 1];
    for (int x = last + 1; x <= hi; x++) {
        L[j + 1].S = L[j].S; bits_shl_or(&L[j + 1].S, &L[j].E, x); if (2 * x <= T) bits_set(&L[j + 1].S, 2 * x);
        int gaps = (T + 1) - bits_popcount_masked(&L[j + 1].S);
        if (gaps > CAP[j + 1]) continue;
        L[j + 1].E = L[j].E; bits_set(&L[j + 1].E, x);
        a[j + 1] = x;
        gen_prefixes(j + 1, a, L);
    }
}

static stats *tstats;  /* NTHREADS stats slots, one per thread, allocated by run() */
/*
 * worker -- thread body: claim work items from the shared queue and search below each one.
 *
 * Purpose: dynamic load balancing over the prefixes made by gen_prefixes().
 * Inputs:  arg: this thread's stats slot (&tstats[i]); globals prefixes, nprefixes, SPLIT (read-only here)
 *          and next_prefix (atomic).
 * Outputs: accumulates nodes, pruned and found into *arg; returns NULL.
 * Errors:  none.
 */
static void *worker(void *arg) {
    stats *st = arg;
    int a[MAXK + 2]; lvl L[MAXK + 2];  /* thread-private path and level stack */
    for (;;) {
        size_t i = atomic_fetch_add(&next_prefix, 1);  /* claim the next work item */
        if (i >= nprefixes) break;  /* queue exhausted */
        memcpy(a, prefixes[i].a, sizeof a);
        L[SPLIT] = prefixes[i].L;  /* dfs reads only levels >= SPLIT */
        dfs(SPLIT, a, L, st);
    }
    return NULL;
}

/*
 * now -- monotonic wall-clock time in seconds.
 *
 * Purpose: timing of the prefix generation, the search and the selftest cases.
 * Inputs:  none.
 * Outputs: CLOCK_MONOTONIC time as a double.
 * Errors:  none (the clock_gettime result is not checked).
 */
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

/*
 * run -- one complete search for k-element bases covering [1,t].
 *
 * Purpose: sets the globals and pruning tables, generates the work items, runs NTHREADS workers and merges
 *          their statistics.
 * Inputs:  k: basis size, 3 <= k <= MAXK (upper bound not checked); t: target, 1 <= t <= 255; verbose:
 *          nonzero prints the prefix line, the per-level counts and the RESULT line (header, section 7);
 *          globals SPLIT, NTHREADS, QUIET.
 * Outputs: returns the number of bases reported (exact for t > n(2,k-1), header section 8); sets K, T,
 *          TMASK, UB[], CAP[] and prefixes[]; SPLIT is clamped to [1, k-2] during the run and then restored.
 * Errors:  t > 255: "T too large for NW=4" on stderr, exit(1).  calloc and pthread_create failures are not
 *          checked.
 */
static uint64_t run(int k, int t, int verbose) {
    K = k; T = t;
    if (T > 64 * NW - 1) { fprintf(stderr, "T too large for NW=%d\n", NW); exit(1); }  /* masks hold 0..255 */
    memset(&TMASK, 0, sizeof TMASK); for (int i = 0; i <= T; i++) bits_set(&TMASK, i);  /* TMASK = bits 0..T */
    /* UB[j] = min(n(2,j-1) + 1, T) while n(2,j-1) is tabulated, else T; CAP[j] = sum_{i=j+1..K} (i+1) */
    for (int j = 0; j <= K + 1; j++) {
        UB[j] = (j >= 1 && j - 1 <= N2_KNOWN) ? N2[j - 1] + 1 : T; if (UB[j] > T) UB[j] = T;
        int c = 0; for (int i = j + 1; i <= K; i++) c += i + 1; CAP[j] = c;
    }
    int save = SPLIT; if (SPLIT >= K - 1) SPLIT = K - 2; if (SPLIT < 1) SPLIT = 1;  /* 1 <= SPLIT <= K-2 */
    double t0 = now();
    nprefixes = 0;  /* reuses prefixes[] of an earlier run */
    /* root A_1 = {1}: E = {0,1}, S = {0,1,2} (0+0, 0+1, 1+1); a[0] = 0 throughout */
    int a[MAXK + 2] = {0}; lvl L[MAXK + 2]; memset(L, 0, sizeof L);
    a[1] = 1; bits_set(&L[1].E, 0); bits_set(&L[1].E, 1); bits_set(&L[1].S, 0); bits_set(&L[1].S, 1); bits_set(&L[1].S, 2);
    gen_prefixes(1, a, L);
    double t1 = now();
    if (verbose) { printf("k=%d T=%d: %zu prefixes at depth %d (%.2fs)\n", K, T, nprefixes, SPLIT, t1 - t0); fflush(stdout); }
    atomic_store(&next_prefix, 0);  /* empty the claim counter */
    tstats = calloc(NTHREADS, sizeof(stats));  /* zeroed counters (unchecked) */
    pthread_t th[256];  /* NTHREADS <= 256 (main clamps) */
    /* pthread_create makes the tables written above visible to the threads; join waits for the last item */
    for (int i = 0; i < NTHREADS; i++) pthread_create(&th[i], NULL, worker, &tstats[i]);
    for (int i = 0; i < NTHREADS; i++) pthread_join(th[i], NULL);
    double t2 = now();
    /* merge the per-thread statistics; levels <= SPLIT were never counted */
    stats tot; memset(&tot, 0, sizeof tot);
    for (int i = 0; i < NTHREADS; i++) { tot.found += tstats[i].found; for (int j = 0; j <= K + 1; j++) { tot.nodes[j] += tstats[i].nodes[j]; tot.pruned[j] += tstats[i].pruned[j]; } }
    uint64_t total = 0; for (int j = 0; j <= K + 1; j++) total += tot.nodes[j];
    if (verbose) {
        printf("nodes per level:"); for (int j = 2; j <= K; j++) if (tot.nodes[j]) printf(" L%d=%llu", j, (unsigned long long)tot.nodes[j]); printf("\n");
        printf("gap-pruned per level:"); for (int j = 2; j <= K; j++) if (tot.pruned[j]) printf(" L%d=%llu", j, (unsigned long long)tot.pruned[j]); printf("\n");
        printf("RESULT k=%d T=%d found=%llu total_nodes=%llu time=%.2fs rate=%.2fM nodes/s threads=%d\n",
               K, T, (unsigned long long)tot.found, (unsigned long long)total, t2 - t0, total / (t2 - t0) / 1e6, NTHREADS);
        fflush(stdout);
    }
    free(tstats); SPLIT = save;  /* restore -d for the next run (selftest) */
    return tot.found;
}

/*
 * selftest -- check against the published n(2,k) and extremal-basis counts for k = 3..17.
 *
 * Purpose: for each case runs listing mode at T = n(2,k) (must find exactly `count` bases) and proof mode at
 *          T = n(2,k) + 1 (must find none).
 * Inputs:  the case table; globals NTHREADS and SPLIT (they affect the speed only).
 * Outputs: per case "selftest k=.. n=..: found F (expect C), at n+1: G (expect 0) Xs OK|FAIL" on stdout,
 *          then SELFTEST PASSED or SELFTEST FAILED.  QUIET is forced on during the runs and restored.
 * Errors:  any mismatch: SELFTEST FAILED and exit(1).
 */
static void selftest(void) {
    /* {k, n(2,k), number of extremal bases} (Challis-Robinson / Kohonen tables, as in psp2_v1.c) */
    struct { int k, n, count; } cases[] = {
        {3, 8, 1}, {4, 12, 1}, {5, 16, 1}, {6, 20, 5}, {7, 26, 3}, {8, 32, 2}, {9, 40, 1},
        {10, 46, 2}, {11, 54, 4}, {12, 64, 1}, {13, 72, 1}, {14, 80, 3}, {15, 92, 1}, {16, 104, 1}, {17, 116, 1}
    };
    int ok = 1, sq = QUIET; QUIET = 1;  /* no FOUND lines during the checks */
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        double t0 = now();
        uint64_t f = run(cases[i].k, cases[i].n, 0);  /* listing mode */
        uint64_t g = run(cases[i].k, cases[i].n + 1, 0);  /* proof mode: must find nothing */
        int good = (f == (uint64_t)cases[i].count && g == 0);
        printf("selftest k=%2d n=%3d: found %llu (expect %d), at n+1: %llu (expect 0) %.2fs %s\n", cases[i].k, cases[i].n,
               (unsigned long long)f, cases[i].count, (unsigned long long)g, now() - t0, good ? "OK" : "FAIL");
        fflush(stdout);
        if (!good) ok = 0;
    }
    QUIET = sq;
    printf(ok ? "SELFTEST PASSED\n" : "SELFTEST FAILED\n");
    if (!ok) exit(1);
}

/*
 * main -- parse the command line and run either the selftest or one verbose search.
 *
 * Purpose: command-line driver; the options are described in the header, section 6.
 * Inputs:  argc, argv.
 * Outputs: sets NTHREADS (clamped to [1,256]), SPLIT and QUIET; then the output of selftest() or run().
 * Errors:  unknown option: "bad arg ..." on stderr, status 1; missing or invalid -k/-t (k < 3 or t < 1):
 *          the usage line on stderr, status 1; an option without its value reads argv[argc] (NULL, crash).
 *          Returns 0 after a search whether or not anything was found.
 */
int main(int argc, char **argv) {
    int k = 0, t = 0, st = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-k")) k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t")) t = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-j")) NTHREADS = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-d")) SPLIT = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-q")) QUIET = 1;
        else if (!strcmp(argv[i], "-selftest")) st = 1;
        else { fprintf(stderr, "bad arg %s\n", argv[i]); return 1; }
    }
    if (NTHREADS < 1) NTHREADS = 1; if (NTHREADS > 256) NTHREADS = 256;  /* th[] in run() has 256 slots */
    if (st) { selftest(); return 0; }  /* selftest() exits 1 itself on failure */
    /* k <= MAXK = 40 is not checked; run() rejects t > 255 */
    if (k < 3 || t < 1) { fprintf(stderr, "usage: psp2 -k K -t T [-j threads] [-d splitdepth] [-q] | -selftest\n"); return 1; }
    run(k, t, 1);  /* status 0 whether or not bases were found */
    return 0;
}
