/*
 * psphs.c -- exhaustive search for extremal h-bases with SMALL h (2..8) and
 *            moderate k: the "rows" of the postage stamp table
 *            (A001212 h=2, A001213 h=3, A001214 h=4, A001215 h=5, A001216 h=6,
 *             A005342 h=7, A005343 h=8).
 *
 * Find every A = {a_1=1 < a_2 < ... < a_k} such that every integer in [1,T]
 * is a sum of at most h elements of A (repetition allowed).
 *
 * Bitset DFS (Challis "K-program" style):
 *   S[m] = set of sums of exactly m elements of A u {0}  (= sums of <= m elements of A),
 *   kept as bit masks over [0, 64*NW).  Adding x:  S[m] |= S_old[m-c] << (c*x), c=1..m.
 *   Prunes (all exact, never discard a candidate that could reach T):
 *     admissibility      a_{j+1} <= n_h(A_j) + 1     (= first gap of S[h])
 *     element-wise bound a_j <= n_h(j-1) + 1          (known extremal values)
 *     gaps test          #uncovered in [1,T] <= sum_{i=j+1..k} C(i+h-1, h-1)
 *                        (each new a_i creates at most C(i+h-1,h-1) new sums)
 *
 * Build: clang -O3 -mcpu=apple-m1 -std=c11 -pthread -o psphs psphs.c
 * Usage: ./psphs -h H -k K -t T [-j threads] [-d splitdepth] [-q]
 *        ./psphs -selftest
 *
 * THEORY OF OPERATIONS
 * ====================
 *
 * 1. What is computed
 *    n_h(A) (the "h-range") of a basis A = {a_1 = 1 < ... < a_k} is the largest N such that every integer
 *    1..N is a sum of at most h elements of A; n(h,k) (written n_h(k) above) is the maximum of n_h(A)
 *    over all k-element bases.  For given (h, k, T) the program enumerates EVERY k-element basis with
 *    n_h(A) >= T, prints each one (unless -q) and reports how many there are ("found").  Typical uses:
 *      T = n(h,k)       lists the extremal bases and their number (this is what the selftest checks);
 *      T = n(h,k) + 1   "proof mode": found = 0 proves n(h,k) <= T - 1;
 *      any T            found > 0 proves n(h,k) >= T, and every FOUND line is a witness.
 *    psp2/psp2.c is the specialised h = 2 program; psph/ handles large h (the columns of the table).
 *
 * 2. Mathematics used by the code (S_m = node.S[m] = set of sums of at most m elements, 0 included)
 *    - New element x:  S_m(A u {x}) = union over c = 0..m of ( c*x + S_{m-c}(A) ),  c = copies of x.
 *      The c = 0 term is the old S_m, so add_elem() only ORs in the terms c >= 1.
 *    - Admissibility: if a_{j+1} > n_h(A_j) + 1, the value g = n_h(A_j) + 1 can never be represented
 *      (every later element exceeds g), so n_h(A) = n_h(A_j).  Hence a_{j+1} <= first zero of S_h.
 *    - Element-wise bound: n_h(A_j) <= n(h,j), so a_{j+1} <= n(h,j) + 1 (values from known()).  This is
 *      implied by admissibility; in the code (UB[]) it only caps the candidate loop, and UB[] also caps
 *      every element at T.
 *    - Gaps test: adding the i-th element creates at most C(i+h-1, h-1) new values of S_h, because a new
 *      sum uses the new element at least once and the remaining h-1 summands form a multiset over the
 *      i+1 symbols {0, a_1, ..., a_i}: C((i+1)+(h-1)-1, h-1) = C(i+h-1, h-1) choices.  So once j
 *      elements are placed, more than CAP[j] = sum_{i=j+1..k} C(i+h-1, h-1) uncovered values in [1,T]
 *      can never all be filled, and the node is pruned.
 *    - Completeness: each prune is a necessary condition for n_h(A) >= T provided T > n(h,k-1), i.e. no
 *      proper prefix can already cover [1,T].  That holds for every meaningful target (n(h,k) >
 *      n(h,k-1)); for a smaller T, bases with an element above min(n_h(A_j) + 1, T) would be missed.
 *
 * 3. Algorithm
 *    a) run(): NW = ceil((T+1)/64) words, TMASK = bits 0..T, UB[j] and CAP[j] for j = 0..K+1.
 *    b) Root node: j = 1, a_1 = 1, S_0 = {0}, S_m = {0..m} for m >= 1.
 *    c) gen_prefixes(): sequential DFS down to depth SPLIT (-d, clamped to 1..K-1) with the same candidate
 *       range and gaps test as dfs(); every surviving depth-SPLIT node is appended to prefixes[].
 *    d) NTHREADS workers (-j) claim the prefixes one at a time (atomic counter) and run dfs() below each.
 *    e) dfs(): for x = a_j + 1 .. min(first zero of S_h, UB[j+1]): copy the node and add x; then
 *         at the last level (j+1 == K): report the basis if S_h covers all of [0,T] (exact popcount);
 *         at an inner level: prune by the gaps test, or recurse.
 *    f) The per-thread statistics are summed and printed (RESULT line).
 *
 * 4. Data structures and invariants
 *    - node.S[m], m = 0..H: bitset of NW 64-bit words, bit b of word w <-> value 64*w + b.  For v < 64*NW
 *      the bit of v is set iff v is a sum of at most m of a[1..j].  Bits that would land at or above
 *      64*NW are dropped; shifts only move bits upward, so lower bits are never corrupted and S[m] is
 *      exact on [0, 64*NW), which contains [0,T].  S[0] = {0} always; S[m] is a subset of S[m+1].
 *    - node.a[1..j]: the chosen elements in increasing order (a[0] = 0 is unused); node.j = their number.
 *    - sizeof(node) = 1360 bytes.  A child is a full copy of its parent (node c = *nd), so a thread needs
 *      about K * 1.4 KB of stack, and prefixes[] needs 1360 bytes per prefix.
 *    - UB[j] = min(n(h,j-1) + 1, T), or T if n(h,j-1) is not tabulated: the largest value tried for a_j.
 *    - CAP[j] = sum_{i=j+1..K} C(i+H-1, H-1): the most new sums that elements j+1..K can still add.
 *    - TMASK has bits 0..T set, so (T+1) - covered_count(S_h) = number of uncovered values in [1,T].
 *
 * 5. Parallelization
 *    The tree is cut at depth SPLIT (default 6) into independent subtrees; the workers take them from
 *    prefixes[] with atomic_fetch_add on next_prefix (dynamic load balancing).  Each worker writes only
 *    its own stats slot, FOUND lines are serialised by out_lock, and every other global is read-only
 *    while the workers run.  The prefix phase is single-threaded and its nodes (levels <= SPLIT) are
 *    not counted in the statistics.
 *
 * 6. Command line
 *    -h H        stamps per sum, 2 <= H <= 8 (MAXH)                                          required
 *    -k K        number of elements, K >= 3 (K <= MAXK = 48 is assumed but NOT checked)     required
 *    -t T        target: report every basis with n_h(A) >= T, 1 <= T <= 1023               required
 *    -j N        worker threads (default 10, clamped to 1..256)
 *    -d D        split depth of the prefix phase (default 6; each run clamps it to 1..K-1)
 *    -q          quiet: do not print FOUND lines (they are still counted)
 *    -selftest   run the regression table instead (-h/-k/-t/-q are ignored, -j/-d are used)
 *    Example (proof mode for n(3,10) = 154):  ./psphs -h 3 -k 10 -t 155 -j 10
 *
 * 7. Output (everything on stdout; "bad arg ...", the usage line and "T too large" go to stderr)
 *    h=H k=K T=T: <n> prefixes at depth <D> (<sec>s)
 *    FOUND h=H k=K range>=T: a1 a2 ... ak           one line per basis, in nondeterministic order
 *    nodes per level: L<j>=<count> ...               nonzero levels only, i.e. j > D
 *    gap-pruned per level: L<j>=<count> ...
 *    RESULT h=H k=K T=T found=<n> total_nodes=<n> time=<sec>s rate=<x>M nodes/s threads=<n>
 *    A "node" is one candidate x tried below the split depth (one add_elem() call).  A FOUND line only
 *    says n_h(A) >= T; tools/hrange gives the exact range.  For the selftest output see selftest().
 *    Exit status: 0 after a search or a selftest (even a failing one); 1 for a bad argument, a missing
 *    or invalid -h/-k/-t, or T too large.
 *
 * 8. Why the results are exact
 *    The bitsets are exact on [0,T], the leaf test is an exact count of the covered values in [0,T],
 *    every prune is a necessary condition (section 2), and elements are generated in increasing order,
 *    so for T > n(h,k-1) each basis with n_h(A) >= T is visited and reported exactly once.  Nothing is
 *    sampled or approximated.
 *
 * 9. Limits and caveats
 *    - H <= 8 (MAXH and the known() table); T <= 1023 (MAXNW = 16 words; checked in run()).
 *    - K <= 48 (MAXK sizes UB[], CAP[], node.a[] and the stats); main() does not check it, so a larger
 *      K overflows these arrays.
 *    - Node counters are uint64_t; binom() and CAP[] use long, and the largest values (C(55,7) =
 *      202927725 and CAP[0] < 1.5e9 at K = 48, H = 8) are far from overflow.
 *    - realloc() of prefixes[] and calloc() of the stats are not checked; a deep split (-d) can create
 *      millions of prefixes at 1360 bytes each.  pthread_create() failures are not checked.
 *    - Options that take a value read argv[++i] unchecked: a value-less last option passes NULL to atoi.
 *    - The selftest verdict is printed but not reflected in the exit status.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>

/* Compile-time limits (see "Limits and caveats" above):
 *   MAXNW = 64-bit words per bitset (16 words = 1024 bits, hence T <= 1023);
 *   MAXH  = largest h (sizes node.S[0..MAXH]);
 *   MAXK  = largest k (sizes node.a[], UB[], CAP[] and the per-level stats; NOT checked by main()). */
#define MAXNW 16                /* up to 1024-bit masks: T <= 1023 */
#define MAXH 8
#define MAXK 48

/* Run parameters, set by main()/run() and read-only while the worker threads run:
 *   H, K, T  = h, k and the target of the current run;  NW = words per bitset = ceil((T+1)/64).
 *   UB[j]    = largest value tried for a_j: min(n(h,j-1) + 1, T), for j = 0..K+1.
 *   CAP[j]   = sum_{i=j+1..K} C(i+H-1, H-1): the most new h-sums that elements j+1..K can create.
 *   SPLIT    = prefix depth (-d), NTHREADS = worker threads (-j), QUIET = suppress FOUND lines (-q).
 *   TMASK    = bitset with bits 0..T set (the target interval, 0 included). */
static int H, K, T, NW;
static int UB[MAXK + 2];
static long CAP[MAXK + 2];
static int SPLIT = 6, NTHREADS = 10, QUIET = 0;
static uint64_t TMASK[MAXNW];

/*
 * known -- tabulated extremal value n(h,k) for 2 <= h <= 8.
 *
 * Purpose: known extremal ranges n_h(k) (OEIS, proven values only), looked up as [h][k].  They supply
 *          the element-wise bounds UB[] used by run(): a_{j+1} <= n(h,j) + 1.  Rows: A001212 (h=2),
 *          A001213, A001214, A001215, A001216, A005342, A005343 (h=8).
 * Inputs:  h = stamps per sum (tables exist for 2..8); k = number of elements, k >= 0.
 * Outputs: returns n(h,k); 0 for k == 0 (empty basis); -1 if (h,k) is not tabulated.
 * Errors:  none for k >= 0 (an unknown h or a k beyond the table gives -1, which run() treats as
 *          "no bound").  k < 0 is not checked and would index before the table; run() never passes it.
 */
static int known(int h, int k) {
    /* r2..r8: r<h>[k] = n(h,k) for k = 0..n, where n is the length set in the switch below (r<h>[0] = 0).
     * Values above 1023 (the T limit) still work as bounds because UB[] is capped at T. */
    static const int r2[] = {0,2,4,8,12,16,20,26,32,40,46,54,64,72,80,92,104,116,128,140,152,164,180,196,212};
    static const int r3[] = {0,3,7,15,24,36,52,70,93,121,154,186,225,271,323,385};
    static const int r4[] = {0,4,10,26,44,70,108,162,228,310,422,550,700};
    static const int r5[] = {0,5,14,35,71,126,211,336,524,726,1016};
    static const int r6[] = {0,6,18,52,114,216,388,638,1007,1545};
    static const int r7[] = {0,7,23,69,165,345,664,1137,1911};
    static const int r8[] = {0,8,28,89,234,512,1045,2001,3485};
    const int *r = NULL; int n = 0;
    switch (h) {
        case 2: r = r2; n = 24; break; case 3: r = r3; n = 15; break; case 4: r = r4; n = 12; break;
        case 5: r = r5; n = 10; break; case 6: r = r6; n = 9; break; case 7: r = r7; n = 8; break; case 8: r = r8; n = 8; break;
    }
    if (k == 0) return 0;
    if (r && k <= n) return r[k];
    return -1;
}

/* One search node: a prefix a[1..j] of the basis together with its sum sets.
 *   j        number of elements placed so far (a[1] = 1 always);
 *   a[i]     the i-th element, 1 <= i <= j, strictly increasing (a[0] = 0 is unused);
 *   S[m][w]  bitset of the sums of at most m elements of a[1..j], m = 0..H; value v is bit (v & 63) of
 *            word (v >> 6) and is exact for v < 64*NW (THEORY 4).  sizeof(node) = 1360 bytes. */
typedef struct {
    int j;
    int a[MAXK + 2];
    uint64_t S[MAXH + 1][MAXNW];
} node;

/* Per-thread counters (one slot per worker, summed by run()):
 *   nodes[j]   candidates tried as element j (add_elem calls), counted only below the split depth;
 *   pruned[j]  of those, the ones removed by the gaps test (inner levels j < K only);
 *   found      complete bases with n_h(A) >= T. */
typedef struct { uint64_t nodes[MAXK + 2], pruned[MAXK + 2], found; } stats;

/* Work queue: prefixes[0..nprefixes-1] are the depth-SPLIT nodes from gen_prefixes() (capacity
 * prefix_cap, grown by doubling, reused by later runs); next_prefix is the next unclaimed index.
 * out_lock serialises the FOUND lines printed by different threads. */
static node *prefixes; static size_t nprefixes, prefix_cap;
static _Atomic size_t next_prefix;
static pthread_mutex_t out_lock = PTHREAD_MUTEX_INITIALIZER;

/*
 * shl_or -- dst |= (src << s) on NW-word bitsets, dropping bits shifted to 64*NW or beyond.
 *
 * Purpose: the primitive of add_elem(): ORs the sum set src, translated by s = c*x, into dst.
 * Inputs:  dst, src = bitsets of NW words (global NW), word w holding values 64*w .. 64*w+63;
 *          s = shift >= 0.  In every call dst and src are different rows of the same node.
 * Outputs: dst, updated in place; no return value.
 * Errors:  none.  s >= 64*NW is a no-op because every bit would be shifted out.
 */
static inline void shl_or(uint64_t *dst, const uint64_t *src, int s) {
    if (s >= 64 * NW) return;
    /* ws = whole words of the shift, bs = remaining bits.  Destination word i gets source word i-ws
     * shifted left by bs, plus the top bs bits of source word i-ws-1 (when that word exists).  bs == 0
     * is excluded from the carry term because a 64-bit shift by 64 is undefined in C.  Words below ws
     * receive nothing; the bits leaving word NW-1 are the dropped ones. */
    int ws = s >> 6, bs = s & 63;
    for (int i = NW - 1; i >= ws; i--) {
        uint64_t v = src[i - ws] << bs;
        if (bs && i - ws - 1 >= 0) v |= src[i - ws - 1] >> (64 - bs);
        dst[i] |= v;
    }
}
/*
 * first_zero -- position of the lowest clear bit of an NW-word bitset.
 *
 * Purpose: applied to S_h it gives n_h(A_j) + 1, the first value that is not a sum of at most h
 *          elements, i.e. the admissibility limit for the next element.
 * Inputs:  b = bitset of NW words (global NW).
 * Outputs: the lowest v in [0, 64*NW) whose bit is 0, or 64*NW if every bit is set.
 * Errors:  none.
 */
static inline int first_zero(const uint64_t *b) {
    /* ~b[i] has a 1 for every missing value of word i; count-trailing-zeros finds the lowest one */
    for (int i = 0; i < NW; i++) { uint64_t v = ~b[i]; if (v) return i * 64 + __builtin_ctzll(v); }
    return 64 * NW;
}
/*
 * covered_count -- how many values of [0,T] a bitset contains.
 *
 * Purpose: coverage measure for the gaps test and the leaf test (T+1 means all of [0,T] is covered).
 * Inputs:  b = bitset of NW words; globals NW and TMASK (bits 0..T).
 * Outputs: popcount(b & TMASK), a number in [0, T+1].
 * Errors:  none.
 */
static inline int covered_count(const uint64_t *b) {
    int c = 0; for (int i = 0; i < NW; i++) c += __builtin_popcountll(b[i] & TMASK[i]); return c;
}
/*
 * add_elem -- add element x to a node (in place): update the sum sets S[1..H].
 *
 * Purpose: implements S_m(A u {x}) = union over c = 0..m of (c*x + S_{m-c}(A)) for m = 1..H.
 * Inputs:  nd = node whose S[0..H] are the sum sets of the current prefix; x = the new element
 *          (> nd->a[nd->j]).  Globals H and NW.
 * Outputs: nd->S[1..H] updated; nd->a[] and nd->j are NOT changed (the caller sets them).
 * Errors:  none; sums >= 64*NW are dropped silently, which is harmless because T < 64*NW.
 */
static inline void add_elem(node *nd, int x) {
    /* m runs downward: row m is built from rows m-c (c >= 1), which are lower rows and therefore still
     * hold the OLD sets when they are read (they are updated later in this loop).  c = copies of x.
     * The c = 0 term is the old S[m] itself.  Cost: H(H+1)/2 shift-ORs of NW words. */
    for (int m = H; m >= 1; m--)
        for (int c = 1; c <= m; c++)
            shl_or(nd->S[m], nd->S[m - c], c * x);
}

/*
 * report -- count (and normally print) one basis that reaches the target.
 *
 * Purpose: called by dfs() at the last level for every complete basis with n_h(A) >= T.
 * Inputs:  nd = complete node (nd->j == K, nd->a[1..K] = the basis); st = the calling thread's stats;
 *          globals QUIET, H, K, T.
 * Outputs: st->found incremented; unless QUIET, prints "FOUND h=H k=K range>=T: a1 ... ak" on stdout
 *          under out_lock and flushes it immediately (so partial runs keep their hits).
 * Errors:  none.
 */
static void report(const node *nd, stats *st) {
    st->found++;
    if (QUIET) return;
    pthread_mutex_lock(&out_lock);
    printf("FOUND h=%d k=%d range>=%d:", H, K, T);
    for (int i = 1; i <= nd->j; i++) printf(" %d", nd->a[i]);
    printf("\n"); fflush(stdout);
    pthread_mutex_unlock(&out_lock);
}

/*
 * dfs -- exhaustive depth-first search of all completions of one node.
 *
 * Purpose: tries every admissible next element x of the node; at an inner level prunes with the gaps
 *          test and recurses, at the last level reports each basis whose S_h covers [0,T].
 * Inputs:  nd = node with j = nd->j elements, 1 <= j < K, sum sets consistent with nd->a[1..j];
 *          st = this thread's stats; globals H, K, T, NW, UB, CAP, TMASK, QUIET.
 * Outputs: st->nodes[j+1..K] and st->pruned[j+1..K-1] incremented; report() for every hit.
 * Errors:  none.  Recursion depth is at most K - j, each frame holding one 1360-byte node copy.
 */
static void dfs(const node *nd, stats *st) {
    int j = nd->j, last = nd->a[j];
    int hi = first_zero(nd->S[H]);            /* = range + 1 */
    /* hi = n_h(A_j) + 1 is the admissibility limit; UB adds the element-wise bound and the cap at T */
    if (hi > UB[j + 1]) hi = UB[j + 1];
    /* Last level: x completes the basis.  It is a hit iff all T+1 values 0..T are covered; the bitset
     * is exact there, so this test is exact.  No gaps test is needed at this level. */
    if (j + 1 == K) {
        for (int x = last + 1; x <= hi; x++) {
            node c = *nd; add_elem(&c, x); st->nodes[j + 1]++;
            if (covered_count(c.S[H]) == T + 1) { c.a[j + 1] = x; c.j = j + 1; report(&c, st); }
        }
        return;
    }
    /* Inner level: gaps = uncovered values in [1,T] (0 is always covered).  If they exceed CAP[j+1], the
     * most that elements j+2..K can still add, no completion reaches T: prune.  Otherwise recurse. */
    for (int x = last + 1; x <= hi; x++) {
        node c = *nd; add_elem(&c, x); st->nodes[j + 1]++;
        long gaps = (T + 1) - covered_count(c.S[H]);
        if (gaps > CAP[j + 1]) { st->pruned[j + 1]++; continue; }
        c.a[j + 1] = x; c.j = j + 1;
        dfs(&c, st);
    }
}

/*
 * gen_prefixes -- collect the surviving nodes at depth SPLIT as work units for the threads.
 *
 * Purpose: same expansion and gaps test as dfs(), but a node with nd->j == SPLIT is appended to
 *          prefixes[] instead of being searched.  Runs single-threaded before the workers start.
 * Inputs:  nd = node with 1 <= nd->j <= SPLIT; globals SPLIT (already clamped to 1..K-1 by run(), so
 *          this never reaches the last level), H, T, NW, UB, CAP, TMASK.
 * Outputs: appends to prefixes[] (nprefixes, prefix_cap; the array starts at 4096 nodes and doubles).
 *          No statistics are recorded for these levels.
 * Errors:  realloc() failure is not checked (NULL dereference when out of memory).
 */
static void gen_prefixes(const node *nd) {
    if (nd->j == SPLIT) {
        /* store a copy of this depth-SPLIT node, growing the array by doubling when it is full */
        if (nprefixes == prefix_cap) { prefix_cap = prefix_cap ? 2 * prefix_cap : 4096; prefixes = realloc(prefixes, prefix_cap * sizeof(node)); }
        prefixes[nprefixes++] = *nd; return;
    }
    /* expand exactly as dfs() does: admissible candidates up to min(first gap, UB), gaps test */
    int j = nd->j, last = nd->a[j];
    int hi = first_zero(nd->S[H]); if (hi > UB[j + 1]) hi = UB[j + 1];
    for (int x = last + 1; x <= hi; x++) {
        node c = *nd; add_elem(&c, x);
        long gaps = (T + 1) - covered_count(c.S[H]);
        if (gaps > CAP[j + 1]) continue;
        c.a[j + 1] = x; c.j = j + 1;
        gen_prefixes(&c);
    }
}

/* per-thread statistics of the current run (NTHREADS slots, allocated and freed by run()) */
static stats *tstats;
/*
 * worker -- thread body: run dfs() on prefixes until none are left.
 *
 * Purpose: dynamic load balancing; atomic_fetch_add hands each prefix index to exactly one thread.
 * Inputs:  arg = this thread's stats slot (zeroed by calloc); globals prefixes, nprefixes, next_prefix
 *          and everything dfs() reads.
 * Outputs: the stats slot; FOUND lines through report().  Returns NULL.
 * Errors:  none.
 */
static void *worker(void *arg) {
    stats *st = arg;
    for (;;) { size_t i = atomic_fetch_add(&next_prefix, 1); if (i >= nprefixes) break; dfs(&prefixes[i], st); }
    return NULL;
}
/*
 * now -- monotonic wall-clock time in seconds.
 *
 * Purpose: timing of the prefix phase, of a whole run and of each selftest case.
 * Inputs:  none.
 * Outputs: CLOCK_MONOTONIC time as a double.
 * Errors:  none (a clock_gettime() failure is not checked).
 */
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
/*
 * binom -- binomial coefficient C(n, r).
 *
 * Purpose: computes the per-element limits C(i+H-1, H-1) of the gaps test (CAP[] in run()).
 * Inputs:  n >= 0 and any r.
 * Outputs: C(n,r), or 0 if r < 0 or r > n.
 * Errors:  none in the range used: after step i, v = C(n-r+i, i), so each division is exact, and the
 *          largest product (about C(55,7) * 55 = 1.1e10 for K = 48, H = 8) is far below 2^63.
 */
static long binom(int n, int r) { if (r < 0 || r > n) return 0; long v = 1; for (int i = 1; i <= r; i++) v = v * (n - r + i) / i; return v; }

/*
 * run -- one complete search for (h, k, t); returns the number of bases with n_h(A) >= t.
 *
 * Purpose: sets the globals and the bounds, generates the prefixes, runs the worker threads, sums the
 *          statistics and (if verbose) prints the summary.  Used by main() and by selftest().
 * Inputs:  h = 2..MAXH; k = 3..MAXK; t = 1..1023 (target); verbose = print progress and summary lines.
 *          Globals SPLIT, NTHREADS, QUIET.
 * Outputs: returns the found count.  Writes H, K, T, NW, TMASK, UB[0..K+1], CAP[0..K+1], prefixes[],
 *          nprefixes, next_prefix and tstats (freed again); SPLIT is clamped for the run and restored.
 *          With verbose: the prefix line, "nodes per level", "gap-pruned per level" and RESULT (stdout).
 * Errors:  a T that needs more than MAXNW words prints "T too large" on stderr and calls exit(1).
 *          k > MAXK overflows UB/CAP/node.a (not checked); calloc/pthread_create failures not checked.
 */
static uint64_t run(int h, int k, int t, int verbose) {
    H = h; K = k; T = t;
    /* bits 0..T need ceil((T+1)/64) = (T+64)/64 words */
    NW = (T + 64) / 64; if (NW > MAXNW) { fprintf(stderr, "T too large\n"); exit(1); }
    /* TMASK = {0, 1, ..., T} */
    memset(TMASK, 0, sizeof TMASK);
    for (int i = 0; i <= T; i++) TMASK[i >> 6] |= 1ULL << (i & 63);
    /* UB[j] = known(H, j-1) + 1 = n(H, j-1) + 1 when tabulated, else T; always capped at T.  (UB[0] and
     * UB[1] = 1 are never used: the root already holds a_1 = 1.)  CAP[j] = sum_{i=j+1..K} C(i+H-1, H-1);
     * CAP[K] = CAP[K+1] = 0. */
    for (int j = 0; j <= K + 1; j++) {
        int kn = (j >= 1) ? known(H, j - 1) : -1;
        UB[j] = (kn >= 0) ? kn + 1 : T;
        if (UB[j] > T) UB[j] = T;
        long c = 0; for (int i = j + 1; i <= K; i++) c += binom(i + H - 1, H - 1);
        CAP[j] = c;
    }
    /* split depth: at least 1 (the root itself), at most K-1 so that the last level is always searched
     * inside dfs(); the user's value is restored at the end of the run */
    int split = SPLIT; if (split >= K) split = K - 1; if (split < 1) split = 1;
    int save_split = SPLIT; SPLIT = split;
    double t0 = now();
    nprefixes = 0;
    /* root = the basis {1}: with stamps of value 1 only, the sums of at most m stamps are 0..m
     * (m + 1 <= MAXH + 1 = 9 bits, so the ~0ULL branch below is never taken) */
    node root; memset(&root, 0, sizeof root);
    root.j = 1; root.a[0] = 0; root.a[1] = 1;
    root.S[0][0] = 1;                       /* {0} */
    for (int m = 1; m <= H; m++) root.S[m][0] = (m + 1 < 64) ? ((1ULL << (m + 1)) - 1) : ~0ULL;  /* sums of <= m ones: [0,m] */
    gen_prefixes(&root);
    double t1 = now();
    if (verbose) { printf("h=%d k=%d T=%d: %zu prefixes at depth %d (%.2fs)\n", H, K, T, nprefixes, SPLIT, t1 - t0); fflush(stdout); }
    /* parallel phase: NTHREADS workers (main() clamps NTHREADS to 1..256 = size of th[]) */
    atomic_store(&next_prefix, 0);
    tstats = calloc(NTHREADS, sizeof(stats));
    pthread_t th[256];
    for (int i = 0; i < NTHREADS; i++) pthread_create(&th[i], NULL, worker, &tstats[i]);
    for (int i = 0; i < NTHREADS; i++) pthread_join(th[i], NULL);
    double t2 = now();
    /* sum the per-thread statistics; total = candidates tried at levels SPLIT+1..K */
    stats tot; memset(&tot, 0, sizeof tot);
    for (int i = 0; i < NTHREADS; i++) { tot.found += tstats[i].found; for (int j = 0; j <= K + 1; j++) { tot.nodes[j] += tstats[i].nodes[j]; tot.pruned[j] += tstats[i].pruned[j]; } }
    uint64_t total = 0; for (int j = 0; j <= K + 1; j++) total += tot.nodes[j];
    /* summary; the rate divides by the whole run time, prefix phase included */
    if (verbose) {
        printf("nodes per level:"); for (int j = 2; j <= K; j++) if (tot.nodes[j]) printf(" L%d=%llu", j, (unsigned long long)tot.nodes[j]); printf("\n");
        printf("gap-pruned per level:"); for (int j = 2; j <= K; j++) if (tot.pruned[j]) printf(" L%d=%llu", j, (unsigned long long)tot.pruned[j]); printf("\n");
        printf("RESULT h=%d k=%d T=%d found=%llu total_nodes=%llu time=%.2fs rate=%.2fM nodes/s threads=%d\n",
               H, K, T, (unsigned long long)tot.found, (unsigned long long)total, t2 - t0, total / (t2 - t0) / 1e6, NTHREADS);
        fflush(stdout);
    }
    free(tstats); SPLIT = save_split;
    return tot.found;
}

/*
 * selftest -- regression test against published extremal values and basis counts.
 *
 * Purpose: for each case (h, k, n = n(h,k), cnt), run(h,k,n) must find exactly cnt bases (the extremal
 *          ones) and run(h,k,n+1) must find none, which re-proves n(h,k) = n with the full machinery.
 * Inputs:  the built-in table; globals NTHREADS and SPLIT (from -j/-d) are used as given.
 * Outputs: stdout, one line per case:
 *            "selftest h=H k=K n=N: found F (expect C), at n+1: G (expect 0) <sec>s OK|FAIL"
 *          then "SELFTEST PASSED" or "SELFTEST FAILED".  QUIET is forced to 1 during the runs (no FOUND
 *          lines) and restored afterwards.
 * Errors:  failures are only printed; main() still exits with status 0.
 */
static void selftest(void) {
    /* h, k, n(h,k), expected number of extremal bases per Challis-Robinson tables (where listed) */
    struct { int h, k, n, cnt; } cases[] = {
        {2,6,20,5},{2,7,26,3},{2,10,46,2},{2,11,54,4},{2,12,64,1},
        {3,3,15,1},{3,4,24,1},{3,5,36,1},{3,6,52,2},{3,7,70,1},{3,8,93,1},{3,9,121,1},{3,10,154,1},
        {4,3,26,1},{4,4,44,1},{4,5,70,1},{4,6,108,2},{4,7,162,3},{4,8,228,1},
        {5,3,35,1},{5,4,71,2},{5,5,126,1},{5,6,211,2},{5,7,336,1},
        {6,3,52,1},{6,4,114,1},{6,5,216,1},{6,6,388,1},
        {7,3,69,1},{7,4,165,1},{7,5,345,1},
        {8,3,89,1},{8,4,234,1},{8,5,512,2},
    };
    int ok = 1, sq = QUIET; QUIET = 1;
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        double t0 = now();
        /* f = bases with range >= n = the extremal bases; g = bases with range >= n+1, must be 0 */
        uint64_t f = run(cases[i].h, cases[i].k, cases[i].n, 0);
        uint64_t g = run(cases[i].h, cases[i].k, cases[i].n + 1, 0);
        int good = (f == (uint64_t)cases[i].cnt && g == 0);
        printf("selftest h=%d k=%2d n=%4d: found %llu (expect %d), at n+1: %llu (expect 0) %.2fs %s\n", cases[i].h, cases[i].k, cases[i].n,
               (unsigned long long)f, cases[i].cnt, (unsigned long long)g, now() - t0, good ? "OK" : "FAIL");
        fflush(stdout);
        if (!good) ok = 0;
    }
    QUIET = sq;
    printf(ok ? "SELFTEST PASSED\n" : "SELFTEST FAILED\n");
}

/*
 * main -- parse the command line and run one search or the selftest.
 *
 * Purpose: entry point; the options are described under "Command line" in the header.
 * Inputs:  argc/argv.  Sets NTHREADS (-j, clamped to 1..256), SPLIT (-d) and QUIET (-q).
 * Outputs: exit status 0 after a search or a selftest; 1 for an unknown argument ("bad arg X" on
 *          stderr) or a missing/invalid -h, -k or -t (usage line on stderr).  run() may exit(1) itself.
 * Errors:  a value-taking option given last reads argv[argc] = NULL (atoi(NULL), undefined behaviour);
 *          k > MAXK (48) is not rejected; atoi() turns garbage into 0 without a message.
 */
int main(int argc, char **argv) {
    int h = 0, k = 0, t = 0, st = 0;
    /* every value-taking option consumes the next argument without checking that it exists */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h")) h = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-k")) k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t")) t = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-j")) NTHREADS = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-d")) SPLIT = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-q")) QUIET = 1;
        else if (!strcmp(argv[i], "-selftest")) st = 1;
        else { fprintf(stderr, "bad arg %s\n", argv[i]); return 1; }
    }
    /* thread count must fit th[256] in run(); the selftest ignores -h/-k/-t */
    if (NTHREADS < 1) NTHREADS = 1; if (NTHREADS > 256) NTHREADS = 256;
    if (st) { selftest(); return 0; }
    /* only the lower limits and h <= MAXH are checked; k <= MAXK is assumed, t <= 1023 is checked in run() */
    if (h < 2 || h > MAXH || k < 3 || t < 1) { fprintf(stderr, "usage: psphs -h H -k K -t T [-j threads] [-d depth] [-q] | -selftest\n"); return 1; }
    run(h, k, t, 1);
    return 0;
}
