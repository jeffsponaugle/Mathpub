/*
 * psp2gpu.m -- Metal (Apple GPU) version of the exhaustive 2-basis search
 *              (postage stamp problem, h = 2, OEIS A001212).
 *
 * The CPU enumerates admissible prefixes a_1..a_D (same exact prunes as psp2.c);
 * each GPU thread then runs the depth-first search below one prefix with an
 * explicit stack: admissibility, element-wise upper bounds, Challis gaps test,
 * and the first-gap candidate rule for the last element.  Masks are 8 x 32-bit.
 *
 * Build: clang -O2 -fobjc-arc -framework Metal -framework Foundation -o psp2gpu psp2gpu.m
 * Usage: ./psp2gpu -k K -t T [-d prefixdepth] [-b batchsize] [-q]
 *        ./psp2gpu -selftest
 *
 * THEORY OF OPERATIONS
 * ====================
 * 1. What is computed
 *    The same search as psp2.c, whose header gives the mathematics: every K-element basis A whose sums of at
 *    most two elements cover [1,T] (n_2(A) >= T) is printed and counted.  T = n(2,K) + 1 is proof mode (must
 *    find nothing), T = n(2,K) lists the extremal bases; n(2,k) = A001212(k), known for k <= 24 (N2[]).
 *    It produced proof-mode node counts for the A001212(25) cost estimate (k = 16..18 and 20 in ../README.md
 *    section 5.1; the long k = 21 run in ../OVERNIGHT.md).
 *
 * 2. Mathematics (summary of psp2.c)
 *    n_2(A) + 1 = first integer missing from S(A) = {a_p + a_q}; adding x updates S |= E << x | bit(2x) with
 *    E = {0, a_1..a_j}.  Prunes: admissibility a_{j+1} <= n_2(A_j) + 1; element-wise bound a_j <= n(2,j-1) + 1
 *    (UB[]); gaps test: at most CAP[j] = sum_{i=j+1..K} (i+1) integers of [1,T] may be uncovered after A_j;
 *    last element: a_K is g - a_p or g/2 for the smallest gap g.
 *
 * 3. Algorithm
 *    a. run() fills Params P (TMASK, UB[], CAP[]) and copies it into bufParams.
 *    b. gen() enumerates on the CPU every surviving prefix a_1..a_D from the root A_1 = {1} (same prunes) and
 *       appends a Prefix record (a[0..D], E, S) to batchbuf.
 *    c. When BATCH records are queued, and once at the end, flush_batch() calls run_batch(): it copies them
 *       into bufPre, dispatches one GPU thread per record and waits.  The CPU enumeration is paused meanwhile
 *       (CPU and GPU work do not overlap).
 *    d. Each GPU thread runs kernel search: an explicit-stack DFS over levels D+1..K below its prefix
 *       (described above kernelSrc), reporting bases through an atomic counter into the found buffer.
 *    e. run_batch() sums the per-thread node counts and prints the found bases; run() prints the RESULT line.
 *
 * 4. Data structures
 *    Masks are NW = 8 words of 32 bits (integer v <-> bit (v & 31) of word v >> 5), so T <= 255 as in psp2.c.
 *    Params, Prefix and Found are shared with the kernel (layouts described above kernelSrc).  All Metal
 *    buffers use MTLResourceStorageModeShared (unified memory, no copies): bufPre (BATCH Prefix records),
 *    bufParams, bufNfound (one atomic uint), bufFound (1024 Found records), bufNodes (one uint64 per thread)
 *    and bufN (the record count).  On the host, lvl {E, S} per level serves the prefix enumeration as in
 *    psp2.c, and foundlist collects the stored bases (up to 1024 per dispatch, also with -q; nothing reads it).
 *
 * 5. Parallelism and the macOS GPU watchdog
 *    One GPU thread per prefix, threadgroups of min(pipeline limit, 256) threads, non-uniform grid
 *    (dispatchThreads).  Subtree sizes differ by orders of magnitude, so a dispatch lasts as long as its
 *    slowest threads, and SIMD lanes idle while others work (README 5.1: the Metal port runs at about the
 *    speed of the 10 CPU cores).  macOS kills a command buffer that keeps the GPU busy too long ("Impacting
 *    Interactivity"); this hit long k = 21 runs while the CPU was saturated.  run_batch() therefore splits a
 *    failed batch into halves and re-runs them (the failed attempt's results are discarded); it gives up when
 *    a failing batch has <= 256 records or more than 12 halvings.  Keep -b small (k = 21 used -b 32768 and
 *    65536) and -d large (k = 21 used -d 15) so that each dispatch and each thread's subtree stay short.
 *
 * 6. Command line
 *    -k K       basis size; main requires K >= 3.  K <= MAXK = 32 is NOT checked (P.UB/P.CAP would overflow).
 *    -t T       target, 1 <= T <= 255.
 *    -d D       prefix depth = number of elements fixed on the CPU.  Default K - 7; values below 2 become 2;
 *               run() then clamps D to [1, K-2] and exits unless K - D <= 8 (MAXDEPTH).
 *    -b N       prefixes per GPU dispatch (BATCH), default 1048576; sizes the buffers (about 2 x 200 MiB +
 *               8 MiB at the default); must be >= 1, which is not checked.
 *    -q         suppress the device line and the FOUND lines.
 *    -selftest  run the built-in checks (k = 8..17) instead of a search; -b and -q apply.
 *    Values are parsed with atoi/atol (no validation); an option without its value reads argv[argc] (NULL).
 *
 * 7. Output
 *    stdout:  Metal device: <name>, maxThreadsPerThreadgroup=<n>          (unless -q)
 *             FOUND k=K range>=T: a_1 ... a_K                            (unless -q; at most 1024 per dispatch)
 *             RESULT k=K T=T D=D prefixes=P found=F gpu_nodes=N time=Xs (gpu Gs) rate=YM nodes/s
 *             selftest lines, then SELFTEST PASSED or SELFTEST FAILED
 *    stderr:  [progress] batches=B prefixes=P nodes=N found=F elapsed=Xs rate=YM/s   (every 16 batches)
 *             GPU error on batch of N (depth d): <reason> -- splitting and retrying
 *             fatal errors and the usage line
 *    gpu_nodes counts levels D+1..K only (the CPU prefix levels are not counted); the node definition is
 *    psp2.c's, so the totals agree when D equals psp2's split depth.  time and rate use the wall clock of the
 *    whole run; (gpu G) is the time spent in run_batch().  Exit status 0 after a completed search or any
 *    selftest (even a failed one); 1 on bad arguments or a fatal error.
 *
 * 8. Exactness
 *    The prunes are the necessary conditions of psp2.c, evaluated with the same integer arithmetic, so the
 *    result is exact for T > n(2,K-1) (psp2.c header, section 8).  The found count stays exact when more than
 *    1024 bases occur in one dispatch (only the printing is cut), and a failed command buffer is re-run,
 *    never partially counted.  The order of FOUND lines within a dispatch is nondeterministic.
 *
 * 9. Limits and caveats
 *    T <= 255; K <= 32 (unchecked); K - D <= 8; per-thread node counters are 64-bit; nfound is a 32-bit
 *    counter per dispatch; BATCH must fit in 32 bits and in the Metal buffer size limit, and failed
 *    allocations are not checked.  No checkpoint/resume: an interrupted or abandoned run restarts from
 *    scratch.  The kernel is compiled from source at every start.  The progress counters (batch number,
 *    start time) are static across runs, so in -selftest their elapsed time and rate mix several runs.
 *    The raw string literal holding the kernel needs a GNU C dialect (clang's default); -std=c11 rejects it.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define NW 8                    /* 8 x 32-bit words = 256-bit masks, so T <= 255 */
#define MAXK 32                 /* array bound: K <= MAXK (not checked); equal in kernelSrc */

/*
 * METAL KERNEL SOURCE (kernelSrc)
 * ===============================
 * kernelSrc is Metal Shading Language source held in a raw string literal R"MSL( ... )MSL" (a Clang extension
 * in GNU C / Objective-C modes); setup_metal() compiles it at run time with newLibraryWithSource.  Every byte
 * inside the literal is kernel source, so its documentation lives here, outside the literal.
 *
 * Parts of the source, in order:
 *   #define NW 8, MAXK 32  must equal the host definitions above (256-bit masks of 32-bit words; array bound).
 *   #define MAXDEPTH 8     size of the private DFS stack; run() enforces K - D <= 8 with a literal 8.
 *   struct Params          K, T, D, pad (unused; UB starts at byte 16), UB[MAXK+2], CAP[MAXK+2], TMASK[NW]:
 *                          the tables of psp2.c, filled by run() and read through the constant address space.
 *   struct Prefix          a[MAXK+2] (a[0..D] valid, the rest 0), E[NW], S[NW]: the state after the prefix
 *                          a_1..a_D, i.e. one work item.
 *   struct Found           a[MAXK+2]: one reported basis, a[1..K] (a[K] is the last element chosen).
 *   All fields are 32-bit uints in both languages, so the host typedefs below have identical layouts
 *   (320, 200 and 136 bytes) with no padding.
 *   shl_or, popc_masked, first_zero   32-bit-word versions of psp2.c's bit helpers (blocks below).
 *   kernel search                     one thread = the DFS below one prefix (block below).
 */

/*
 * shl_or (MSL) -- dst |= src << s on 8 x 32-bit masks; bits shifted past bit 255 are dropped.
 *
 * Purpose: with src = E and s = x it ORs the sums x + a_p (p = 0..j) into a sum mask (bits_shl_or in psp2.c).
 * Inputs:  dst, src: thread-private 8-word masks; s: shift (s >= 256 returns at once).
 * Outputs: dst updated in place.
 * Errors:  none; the bs != 0 test avoids an undefined shift by 32.
 */

/*
 * popc_masked (MSL) -- number of set bits of s inside [0,T].
 *
 * Purpose: the covered count for the gaps test, gaps = (T+1) - popc_masked(S, P.TMASK).
 * Inputs:  s: thread-private mask; tm: P.TMASK (bits 0..T) in constant memory.
 * Outputs: popcount(s & tm), 0..T+1.
 * Errors:  none.
 */

/*
 * first_zero (MSL) -- index of the lowest clear bit of s, or 256 if all bits are set.
 *
 * Purpose: the admissibility limit n_2(A_j) + 1 for the next element (exact when it is <= T).
 * Inputs:  s: thread-private mask.
 * Outputs: 0..256.
 * Errors:  none.
 */

/*
 * search (MSL kernel) -- each GPU thread runs the depth-first search below one prefix.
 *
 * Purpose: GPU version of psp2.c's dfs() + leaf(), with an explicit stack instead of recursion, over the
 *          K - D <= 8 levels below the prefix.
 * Inputs:  pre (buffer 0): the dispatch's Prefix records; P (buffer 1): Params; nprefix (buffer 5): number of
 *          valid records; gid: thread index (threads with gid >= nprefix return at once).
 * Outputs: nfound (buffer 2): atomically incremented once per basis (the host resets it per dispatch);
 *          found (buffer 3): the first 1024 bases of the dispatch, a[1..K] (later ones are counted, not stored);
 *          nodecount[gid] (buffer 4): this thread's node count, defined as in psp2.c (every candidate at levels
 *          D+1..K-1, gap-pruned or not, plus the leaf candidates that pass the range filter).
 * Errors:  none detected.  K - D > MAXDEPTH or K > MAXK would overflow the private arrays (the host prevents
 *          the first, not the second).  A thread that runs too long gets the whole command buffer killed by
 *          the macOS watchdog (handled by run_batch() on the host).
 *
 * Thread-private state (j = number of placed elements, D <= j <= K-1; lv = j - D is the stack level):
 *   a[MAXK+2]        the current path: a[0] = 0, a[1..D] from the prefix, a[D+1..j] chosen below.
 *   S[lv][NW]        sum mask after a_1..a_j, one per level (S[0] = the prefix's S), so popping needs no undo.
 *   E[NW]            ONE element mask {0, a_1..a_j}: a push sets bit x, a pop clears bit a[j+1].
 *   cur[lv], hi[lv]  next candidate and last admissible candidate for a_{j+1} at that level.
 * Main loop (one candidate or one leaf per iteration):
 *   j + 1 == K       leaf, exactly psp2.c's leaf(): G = gaps of [0,T]; if there are none (only when
 *                    T <= n(2,K-1)) record the placeholder a_K = a_{K-1} + 1; else test z = g - a_p
 *                    (p = 0..K-1) and z = g/2 in (a_{K-1}, min(g, UB[K])]; then pop.
 *   cur[lv] > hi[lv] level exhausted: pop (j--, clear a[j+1] from E); dropping below D ends the thread.
 *   otherwise        x = cur[lv]++; S[lv+1] = S[lv] | E << x | bit(2x) (2x only if <= T); count the node;
 *                    gaps test against CAP[j+1]; a survivor is pushed (E |= x, a[j+1] = x, j++) and the new
 *                    level gets hi = min(first_zero(S), UB[j+1]) and cur = x + 1.
 * The atomic increment is relaxed; that suffices because the host reads nfound and the found buffer only
 * after the command buffer has completed.
 */
static const char *kernelSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;
#define NW 8
#define MAXK 32
#define MAXDEPTH 8
struct Params { uint K, T, D, pad; uint UB[MAXK+2]; uint CAP[MAXK+2]; uint TMASK[NW]; };
struct Prefix { uint a[MAXK+2]; uint E[NW]; uint S[NW]; };
struct Found  { uint a[MAXK+2]; };

inline void shl_or(thread uint *dst, thread const uint *src, uint s) {
    if (s >= 32*NW) return;
    int ws = s >> 5; uint bs = s & 31;
    for (int i = NW-1; i >= ws; i--) {
        uint v = src[i-ws] << bs;
        if (bs != 0 && i-ws-1 >= 0) v |= src[i-ws-1] >> (32-bs);
        dst[i] |= v;
    }
}
inline uint popc_masked(thread const uint *s, constant uint *tm) {
    uint c = 0; for (int i = 0; i < NW; i++) c += popcount(s[i] & tm[i]); return c;
}
inline uint first_zero(thread const uint *s) {
    for (int i = 0; i < NW; i++) { uint v = ~s[i]; if (v) return i*32 + ctz(v); } return 32*NW;
}

kernel void search(device const Prefix *pre      [[buffer(0)]],
                   constant Params &P            [[buffer(1)]],
                   device atomic_uint *nfound    [[buffer(2)]],
                   device Found *found           [[buffer(3)]],
                   device ulong *nodecount       [[buffer(4)]],
                   constant uint &nprefix        [[buffer(5)]],
                   uint gid [[thread_position_in_grid]])
{
    if (gid >= nprefix) return;
    const uint K = P.K, T = P.T, D = P.D;
    uint a[MAXK+2];
    for (uint i = 0; i < MAXK+2; i++) a[i] = pre[gid].a[i];
    uint S[MAXDEPTH+1][NW], E[NW];                   /* S per level (index j-D); E shared, undone on pop */
    uint cur[MAXDEPTH+1], hi[MAXDEPTH+1];
    for (int i = 0; i < NW; i++) { S[0][i] = pre[gid].S[i]; E[i] = pre[gid].E[i]; }
    ulong nodes = 0;
    int j = D;                                     /* elements a_1..a_j placed */
    { uint fz = first_zero(S[0]); uint h = min(fz, P.UB[D+1]); hi[0] = h; cur[0] = a[D] + 1; }
    while (true) {
        int lv = j - D;
        if ((uint)j + 1 == K) {
            /* leaf: choose a_K covering all gaps */
            uint G[NW]; bool any = false;
            for (int i = 0; i < NW; i++) { G[i] = P.TMASK[i] & ~S[lv][i]; any = any || (G[i] != 0); }
            uint last = a[j];
            if (!any) {
                uint idx = atomic_fetch_add_explicit(nfound, 1u, memory_order_relaxed);
                if (idx < 1024) { for (uint i = 0; i < MAXK+2; i++) found[idx].a[i] = (i == K) ? last+1 : a[i]; }
            } else {
                uint g = 0; for (int i = 0; i < NW; i++) if (G[i]) { g = i*32 + ctz(G[i]); break; }
                uint h = min(g, P.UB[K]);
                for (uint p = 0; p <= (uint)j + 1; p++) {
                    uint z;
                    if (p <= (uint)j) z = g - a[p]; else { if (g & 1) break; z = g / 2; }
                    if (z <= last || z > h) continue;
                    nodes++;
                    uint M[NW]; for (int i = 0; i < NW; i++) M[i] = 0;
                    shl_or(M, E, z);
                    if (2*z <= T) M[(2*z) >> 5] |= 1u << ((2*z) & 31);
                    bool ok = true; for (int i = 0; i < NW; i++) if (G[i] & ~M[i]) { ok = false; break; }
                    if (ok) {
                        uint idx = atomic_fetch_add_explicit(nfound, 1u, memory_order_relaxed);
                        if (idx < 1024) { for (uint i = 0; i < MAXK+2; i++) found[idx].a[i] = (i == K) ? z : a[i]; }
                    }
                }
            }
            j--; if (j < (int)D) break; { uint ax = a[j+1]; E[ax >> 5] &= ~(1u << (ax & 31)); } continue;
        }
        uint x = cur[lv];
        if (x > hi[lv]) { j--; if (j < (int)D) break; { uint ax = a[j+1]; E[ax >> 5] &= ~(1u << (ax & 31)); } continue; }
        cur[lv] = x + 1;
        /* S[lv+1] = S[lv] | E[lv] << x | bit(2x) */
        for (int i = 0; i < NW; i++) S[lv+1][i] = S[lv][i];
        shl_or(S[lv+1], E, x);
        if (2*x <= T) S[lv+1][(2*x) >> 5] |= 1u << ((2*x) & 31);
        nodes++;
        uint gaps = (T + 1) - popc_masked(S[lv+1], P.TMASK);
        if (gaps > P.CAP[j+1]) continue;
        E[x >> 5] |= 1u << (x & 31);
        a[j+1] = x;
        j++;
        { uint fz = first_zero(S[lv+1]); uint h = min(fz, P.UB[j+1]); hi[lv+1] = h; cur[lv+1] = x + 1; }
    }
    nodecount[gid] = nodes;
}
)MSL";

/* Host mirrors of the kernel structs (must stay identical to the MSL definitions); Params is 320 bytes. */
typedef struct { uint32_t K, T, D, pad; uint32_t UB[MAXK+2]; uint32_t CAP[MAXK+2]; uint32_t TMASK[NW]; } Params;
typedef struct { uint32_t a[MAXK+2]; uint32_t E[NW]; uint32_t S[NW]; } Prefix;  /* 200 bytes */
typedef struct { uint32_t a[MAXK+2]; } Found;  /* 136 bytes */

/* N2[k] = n(2,k) = A001212(k) for k = 1..24, N2[0] = 0: the proven values behind UB[] (as in psp2.c) */
static const int N2[] = {0, 2, 4, 8, 12, 16, 20, 26, 32, 40, 46, 54, 64, 72, 80,
                         92, 104, 116, 128, 140, 152, 164, 180, 196, 212};
#define N2_KNOWN 24             /* largest k in N2[] */

static int K, T, D, QUIET = 0;  /* basis size, target, prefix depth (CPU levels), -q */
static size_t BATCH = 1 << 20;  /* prefixes per GPU dispatch (-b) */
static Params P;                /* host copy of the kernel parameters; run() uploads it to bufParams */

/* ---- CPU prefix enumeration (same prunes as psp2.c), streaming in batches ---- */
typedef struct { uint32_t E[NW], S[NW]; } lvl;  /* per-level element and sum masks, as psp2.c's lvl */

/*
 * cset -- set bit i of an 8 x 32-bit host mask.
 *
 * Purpose: host counterpart of psp2.c's bits_set (prefix enumeration, TMASK, the root).
 * Inputs:  m: mask of NW words; i: bit index, 0 <= i <= 255.
 * Outputs: m[i >> 5] gets bit (i & 31) set.
 * Errors:  i outside [0,255] writes outside the mask (undefined); callers pass values <= T or the root's 0, 1, 2.
 */
static inline void cset(uint32_t *m, int i) { m[i >> 5] |= 1u << (i & 31); }

/*
 * cshl_or -- dst |= src << s on 8 x 32-bit host masks; bits shifted past bit 255 are dropped.
 *
 * Purpose: host counterpart of the kernel's shl_or, used by gen() to add the sums x + a_p of a candidate x.
 * Inputs:  dst, src: masks of NW words; s: shift, s >= 0 (s >= 256 ORs in nothing: the loop does not run).
 * Outputs: dst updated in place.
 * Errors:  none; s < 0 would be undefined (callers pass candidates x >= 2).
 */
static inline void cshl_or(uint32_t *dst, const uint32_t *src, int s) {
    int ws = s >> 5, bs = s & 31;
    for (int i = NW - 1; i >= ws; i--) { uint32_t v = src[i - ws] << bs; if (bs && i - ws - 1 >= 0) v |= src[i - ws - 1] >> (32 - bs); dst[i] |= v; }
}

/*
 * cpopc -- number of set bits of s inside [0,T] (host).
 *
 * Purpose: the covered count for gen()'s gaps test.
 * Inputs:  s: mask of NW words; global P.TMASK (bits 0..T, set by run()).
 * Outputs: popcount(s & P.TMASK), 0..T+1.
 * Errors:  none.
 */
static inline int cpopc(const uint32_t *s) { int c = 0; for (int i = 0; i < NW; i++) c += __builtin_popcount(s[i] & P.TMASK[i]); return c; }

/*
 * cfz -- index of the lowest clear bit of s, or 256 if all bits are set (host).
 *
 * Purpose: the admissibility limit n_2(A_j) + 1 for gen() (exact when it is <= T).
 * Inputs:  s: mask of NW words.
 * Outputs: 0..256.
 * Errors:  none.
 */
static inline int cfz(const uint32_t *s) { for (int i = 0; i < NW; i++) { uint32_t v = ~s[i]; if (v) return i * 32 + __builtin_ctz(v); } return 32 * NW; }

static Prefix *batchbuf; static size_t nbatch;  /* host queue of prefixes awaiting dispatch, nbatch <= BATCH */
static uint64_t total_nodes = 0, total_found = 0, total_prefixes = 0;  /* totals of the current run */
static double gpu_time = 0;     /* seconds spent in run_batch() this run */
/* Metal device, compiled pipeline for kernel search and command queue, created once by setup_metal() */
static id<MTLDevice> dev; static id<MTLComputePipelineState> pso; static id<MTLCommandQueue> queue;
static id<MTLBuffer> bufPre, bufParams, bufNfound, bufFound, bufNodes, bufN;  /* kernel buffers 0..5 */
static Found *foundlist; static size_t nfoundlist, foundcap;  /* stored bases (<= 1024 per dispatch); never read */

/*
 * now -- monotonic wall-clock time in seconds.
 *
 * Purpose: timing of the runs, of the GPU batches and of the progress lines.
 * Inputs:  none.
 * Outputs: CLOCK_MONOTONIC time as a double.
 * Errors:  none (the clock_gettime result is not checked).
 */
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

/*
 * run_batch -- run the GPU on batchbuf[off .. off+n); on a command-buffer error, split the batch and retry.
 *
 * Purpose: one synchronous dispatch of n GPU threads (one per prefix), then collection of the results.  The
 *          error case is the macOS watchdog ("Impacting Interactivity") killing a dispatch that runs too long;
 *          the two halves are then re-run recursively.
 * Inputs:  off, n: slice of batchbuf (n <= BATCH); depth: halvings so far (0 for a fresh batch); globals
 *          batchbuf, the Metal objects and buffers, K, T, QUIET.
 * Outputs: adds the node counts to total_nodes, nfound to total_found and n to total_prefixes; appends up to
 *          1024 found bases to foundlist and, unless QUIET, prints each as "FOUND k=K range>=T: a_1 ... a_K"
 *          on stdout.
 * Errors:  a failed command buffer is reported on stderr and its results discarded (the halves re-run from
 *          scratch, so nothing is counted twice).  A failure with n <= 256, or after more than 12 halvings
 *          (depth > 12), prints "GPU error persists at minimal batch size; giving up" and exits with status 1.
 *          More than 1024 bases in one dispatch: all are counted, only the first 1024 are printed.  realloc
 *          is unchecked.
 */
static void run_batch(size_t off, size_t n, int depth) {
    if (n == 0) return;
    memcpy(bufPre.contents, batchbuf + off, n * sizeof(Prefix));  /* slice to the start of bufPre */
    *(uint32_t *)bufNfound.contents = 0;  /* found counter of this dispatch */
    *(uint32_t *)bufN.contents = (uint32_t)n;  /* nprefix: threads with gid >= n return */
    /* one command buffer holding one compute pass */
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    /* indices match the [[buffer(i)]] attributes of kernel search */
    [enc setBuffer:bufPre offset:0 atIndex:0];
    [enc setBuffer:bufParams offset:0 atIndex:1];
    [enc setBuffer:bufNfound offset:0 atIndex:2];
    [enc setBuffer:bufFound offset:0 atIndex:3];
    [enc setBuffer:bufNodes offset:0 atIndex:4];
    [enc setBuffer:bufN offset:0 atIndex:5];
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup; if (tg > 256) tg = 256;  /* threadgroup width */
    /* exactly n threads; the last threadgroup may be partial (non-uniform dispatch) */
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];  /* synchronous: the CPU waits for the GPU */
    if (cb.error) {  /* e.g. killed by the macOS watchdog */
        fprintf(stderr, "GPU error on batch of %zu (depth %d): %s -- splitting and retrying\n", n, depth, cb.error.localizedDescription.UTF8String);
        /* give up when the batch is already small (<= 256) or has been halved more than 12 times */
        if (n <= 256 || depth > 12) { fprintf(stderr, "GPU error persists at minimal batch size; giving up\n"); exit(1); }
        /* re-run both halves from scratch; nothing of the failed attempt was counted */
        run_batch(off, n / 2, depth + 1);
        run_batch(off + n / 2, n - n / 2, depth + 1);
        return;
    }
    uint64_t *nc = (uint64_t *)bufNodes.contents;  /* per-thread node counts */
    for (size_t i = 0; i < n; i++) total_nodes += nc[i];
    uint32_t nf = *(uint32_t *)bufNfound.contents;  /* bases found in this dispatch, may exceed 1024 */
    Found *f = (Found *)bufFound.contents;
    for (uint32_t i = 0; i < nf && i < 1024; i++) {  /* only 1024 records fit in bufFound */
        /* grow foundlist by doubling (64 first; realloc unchecked) */
        if (nfoundlist == foundcap) { foundcap = foundcap ? 2 * foundcap : 64; foundlist = realloc(foundlist, foundcap * sizeof(Found)); }
        foundlist[nfoundlist++] = f[i];
        /* f[i].a[1..K]; a[K] is the element chosen by the leaf */
        if (!QUIET) { printf("FOUND k=%d range>=%d:", K, T); for (int q = 1; q <= K; q++) printf(" %u", f[i].a[q]); printf("\n"); fflush(stdout); }
    }
    total_found += nf;  /* includes bases beyond the 1024 printed */
    total_prefixes += n;
}

/*
 * flush_batch -- send the queued prefixes batchbuf[0 .. nbatch) to the GPU and empty the queue.
 *
 * Purpose: called by gen() when BATCH prefixes are queued and by run() at the end; times the GPU work and
 *          reports progress.
 * Inputs:  globals nbatch, batchbuf and the totals.
 * Outputs: gpu_time += time spent in run_batch(); nbatch = 0; every 16th call prints
 *          "[progress] batches=B prefixes=P nodes=N found=F elapsed=Xs rate=YM/s" on stderr (B and the start
 *          time are static and never reset; P, N and F are the current run's totals).
 * Errors:  those of run_batch().
 */
static void flush_batch(void) {
    if (nbatch == 0) return;
    double t0 = now();
    run_batch(0, nbatch, 0);
    gpu_time += now() - t0;  /* includes retries after GPU errors */
    nbatch = 0;
    static int nb = 0; static double tstart = 0; if (!tstart) tstart = t0;  /* process-wide, never reset */
    /* progress on stderr every 16 batches (current run totals, time since the first batch) */
    if ((++nb % 16) == 0) { fprintf(stderr, "[progress] batches=%d prefixes=%llu nodes=%.4g found=%llu elapsed=%.0fs rate=%.0fM/s\n", nb, (unsigned long long)total_prefixes, (double)total_nodes, (unsigned long long)total_found, now() - tstart, total_nodes / (now() - tstart) / 1e6); fflush(stderr); }
}

/*
 * gen -- CPU enumeration of the prefixes a_1..a_D (same prunes as psp2.c), streamed to the GPU in batches.
 *
 * Purpose: recursive DFS from the root to depth D; each surviving prefix is appended to batchbuf, and a full
 *          batch is run at once (the enumeration waits for the GPU).
 * Inputs:  j: current depth (1 at the root); a: path a[0..j]; L: per-level host masks, L[j] valid;
 *          globals D, T, P (UB, CAP, TMASK), BATCH.
 * Outputs: appends Prefix records (a[0..D], E, S; other fields 0); writes a[j+1..D] and L[j+1..D] as scratch;
 *          may call flush_batch().
 * Errors:  none of its own (see run_batch()); BATCH = 0 would overflow batchbuf.
 */
static void gen(int j, uint32_t *a, lvl *L) {
    if (j == D) {  /* depth D: queue one Prefix */
        Prefix *p = &batchbuf[nbatch++];
        memset(p, 0, sizeof *p);  /* a[D+1..] = 0 */
        for (int i = 0; i <= D; i++) p->a[i] = a[i];
        memcpy(p->E, L[j].E, sizeof p->E); memcpy(p->S, L[j].S, sizeof p->S);
        if (nbatch == BATCH) flush_batch();  /* full batch: run it on the GPU now */
        return;
    }
    int last = a[j];
    int hi = cfz(L[j].S); if (hi > (int)P.UB[j + 1]) hi = P.UB[j + 1];  /* admissibility and UB */
    for (int x = last + 1; x <= hi; x++) {
        /* S_{j+1} = S_j | E_j << x | bit(2x) (2x only if <= T) */
        memcpy(L[j + 1].S, L[j].S, sizeof L[j].S); cshl_or(L[j + 1].S, L[j].E, x); if (2 * x <= T) cset(L[j + 1].S, 2 * x);
        int gaps = (T + 1) - cpopc(L[j + 1].S);
        if (gaps > (int)P.CAP[j + 1]) continue;  /* gaps test */
        memcpy(L[j + 1].E, L[j].E, sizeof L[j].E); cset(L[j + 1].E, x);
        a[j + 1] = x;
        gen(j + 1, a, L);
    }
}

/*
 * setup_metal -- create the device, compile kernelSrc, build the pipeline and allocate all buffers.
 *
 * Purpose: one-time initialisation before any run; BATCH must already be final.
 * Inputs:  globals BATCH, QUIET, kernelSrc.
 * Outputs: dev, pso, queue; shared buffers bufPre (BATCH x 200 bytes), bufParams (320), bufNfound (4),
 *          bufFound (1024 x 136), bufNodes (BATCH x 8), bufN (4); host batchbuf (BATCH x 200 bytes).  Unless
 *          QUIET prints "Metal device: <name>, maxThreadsPerThreadgroup=<n>" on stdout.
 * Errors:  no device, kernel compile error or pipeline error: message on stderr, exit(1).  A nil buffer
 *          (e.g. for a -b too large for the device), a missing kernel function and a malloc failure are not
 *          checked.
 */
static void setup_metal(void) {
    dev = MTLCreateSystemDefaultDevice();
    if (!dev) { fprintf(stderr, "no Metal device\n"); exit(1); }
    NSError *err = nil;
    MTLCompileOptions *opts = [MTLCompileOptions new];  /* default options */
    /* compile the MSL source at run time */
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kernelSrc] options:opts error:&err];
    if (!lib) { fprintf(stderr, "compile error: %s\n", err.localizedDescription.UTF8String); exit(1); }
    id<MTLFunction> fn = [lib newFunctionWithName:@"search"];  /* kernel entry point (nil not checked) */
    pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) { fprintf(stderr, "pso error: %s\n", err.localizedDescription.UTF8String); exit(1); }
    queue = [dev newCommandQueue];
    /* shared storage: CPU and GPU use the same memory, so no explicit transfers are needed */
    bufPre = [dev newBufferWithLength:BATCH * sizeof(Prefix) options:MTLResourceStorageModeShared];
    bufParams = [dev newBufferWithLength:sizeof(Params) options:MTLResourceStorageModeShared];
    bufNfound = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
    bufFound = [dev newBufferWithLength:1024 * sizeof(Found) options:MTLResourceStorageModeShared];
    bufNodes = [dev newBufferWithLength:BATCH * sizeof(uint64_t) options:MTLResourceStorageModeShared];
    bufN = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
    batchbuf = malloc(BATCH * sizeof(Prefix));  /* host queue (unchecked) */
    if (!QUIET) printf("Metal device: %s, maxThreadsPerThreadgroup=%lu\n", dev.name.UTF8String, (unsigned long)pso.maxTotalThreadsPerThreadgroup);
}

/*
 * run -- one complete search for K-bases covering [1,t] with prefix depth d.
 *
 * Purpose: sets the tables, enumerates the prefixes on the CPU, runs them on the GPU and prints the summary.
 * Inputs:  k: basis size, 3 <= k <= MAXK = 32 (upper bound not checked); t: target, 1 <= t <= 255; d: prefix
 *          depth, clamped here to [1, k-2]; verbose: print the RESULT line.  setup_metal() must have run.
 * Outputs: returns total_found (exact for t > n(2,k-1)); sets K, T, D, P and bufParams; resets and fills the
 *          totals.  Verbose: "RESULT k=K T=T D=D prefixes=P found=F gpu_nodes=N time=Xs (gpu Gs)
 *          rate=YM nodes/s" on stdout.
 * Errors:  t > 255 ("T too large") or k - D > 8 ("K-D must be <= 8 (MAXDEPTH)"): stderr, exit(1); and those
 *          of run_batch().
 */
static uint64_t run(int k, int t, int d, int verbose) {
    K = k; T = t; D = d;
    if (T > 32 * NW - 1) { fprintf(stderr, "T too large\n"); exit(1); }  /* masks hold 0..255 */
    if (D >= K - 1) D = K - 2; if (D < 1) D = 1;  /* 1 <= D <= K-2 */
    if (K - D > 8) { fprintf(stderr, "K-D must be <= 8 (MAXDEPTH)\n"); exit(1); }  /* kernel stack depth */
    memset(&P, 0, sizeof P); P.K = K; P.T = T; P.D = D;
    for (int i = 0; i <= T; i++) cset(P.TMASK, i);  /* TMASK = bits 0..T */
    /* UB[j] = min(n(2,j-1) + 1, T) (T beyond the table) and CAP[j] = sum_{i=j+1..K} (i+1), as in psp2.c */
    for (int j = 0; j <= K + 1; j++) {
        int ub = (j >= 1 && j - 1 <= N2_KNOWN) ? N2[j - 1] + 1 : T; if (ub > T) ub = T; P.UB[j] = ub;
        int c = 0; for (int i = j + 1; i <= K; i++) c += i + 1; P.CAP[j] = c;
    }
    memcpy(bufParams.contents, &P, sizeof P);  /* upload once per run */
    total_nodes = total_found = total_prefixes = 0; gpu_time = 0; nfoundlist = 0; nbatch = 0;
    double t0 = now();
    /* root A_1 = {1}: E = {0,1}, S = {0,1,2} (0+0, 0+1, 1+1) */
    uint32_t a[MAXK + 2] = {0}; lvl L[MAXK + 2]; memset(L, 0, sizeof L);
    a[1] = 1; cset(L[1].E, 0); cset(L[1].E, 1); cset(L[1].S, 0); cset(L[1].S, 1); cset(L[1].S, 2);
    gen(1, a, L);
    flush_batch();  /* the last, partial batch */
    double t1 = now();
    if (verbose) {
        printf("RESULT k=%d T=%d D=%d prefixes=%llu found=%llu gpu_nodes=%llu time=%.2fs (gpu %.2fs) rate=%.2fM nodes/s\n",
               K, T, D, (unsigned long long)total_prefixes, (unsigned long long)total_found, (unsigned long long)total_nodes,
               t1 - t0, gpu_time, total_nodes / (t1 - t0) / 1e6);
        fflush(stdout);
    }
    return total_found;
}

/*
 * selftest -- compare the GPU search with the published n(2,k) and extremal-basis counts for k = 8..17.
 *
 * Purpose: per case runs listing mode (T = n(2,k), expect `count` bases) and proof mode (T = n(2,k) + 1,
 *          expect none) with D = max(k - 7, 2).
 * Inputs:  the case table; globals as for run().
 * Outputs: per case "selftest k=.. n=.. D=..: found F (expect C), at n+1: G (expect 0), nodes N1/N2, Xs OK|FAIL"
 *          (N1, N2 = gpu_nodes of the two runs), then SELFTEST PASSED or SELFTEST FAILED, on stdout.  QUIET
 *          is forced on during the runs and restored.
 * Errors:  a failure is only printed: unlike psp2.c, the exit status stays 0.
 */
static void selftest(void) {
    /* {k, n(2,k), number of extremal bases}: the data of psp2.c's selftest from k = 8 on */
    struct { int k, n, count; } cases[] = {
        {8, 32, 2}, {9, 40, 1}, {10, 46, 2}, {11, 54, 4}, {12, 64, 1}, {13, 72, 1}, {14, 80, 3}, {15, 92, 1}, {16, 104, 1}, {17, 116, 1}
    };
    int ok = 1, sq = QUIET; QUIET = 1;
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        int d = cases[i].k - 7; if (d < 2) d = 2;  /* as main's default: 6-7 GPU levels */
        double t0 = now();
        uint64_t f = run(cases[i].k, cases[i].n, d, 0);
        uint64_t n1 = total_nodes;  /* gpu_nodes of the listing-mode run */
        uint64_t g = run(cases[i].k, cases[i].n + 1, d, 0);
        int good = (f == (uint64_t)cases[i].count && g == 0);
        printf("selftest k=%2d n=%3d D=%d: found %llu (expect %d), at n+1: %llu (expect 0), nodes %llu/%llu, %.2fs %s\n", cases[i].k, cases[i].n, d,
               (unsigned long long)f, cases[i].count, (unsigned long long)g, (unsigned long long)n1, (unsigned long long)total_nodes, now() - t0, good ? "OK" : "FAIL");
        fflush(stdout);
        if (!good) ok = 0;
    }
    QUIET = sq;
    printf(ok ? "SELFTEST PASSED\n" : "SELFTEST FAILED\n");
}

/*
 * main -- parse the options, initialise Metal, then run the selftest or one verbose search.
 *
 * Purpose: command-line driver; the options are described in the header, section 6.
 * Inputs:  argc, argv.
 * Outputs: sets BATCH and QUIET; the default prefix depth is d = k - 7, raised to at least 2 (run() then
 *          clamps it to k-2); then the output of selftest() or run().
 * Errors:  unknown option: "bad arg ..." on stderr, status 1; missing or invalid -k/-t (k < 3 or t < 1): the
 *          usage line on stderr, status 1; an option without its value reads argv[argc] (NULL, crash).
 *          setup_metal() runs before the usage check, so even a usage error initialises the GPU.
 */
int main(int argc, char **argv) {
    @autoreleasepool {
        int k = 0, t = 0, d = 0, st = 0;
        for (int i = 1; i < argc; i++) {
            if (!strcmp(argv[i], "-k")) k = atoi(argv[++i]);
            else if (!strcmp(argv[i], "-t")) t = atoi(argv[++i]);
            else if (!strcmp(argv[i], "-d")) d = atoi(argv[++i]);
            else if (!strcmp(argv[i], "-b")) BATCH = (size_t)atol(argv[++i]);
            else if (!strcmp(argv[i], "-q")) QUIET = 1;
            else if (!strcmp(argv[i], "-selftest")) st = 1;
            else { fprintf(stderr, "bad arg %s\n", argv[i]); return 1; }
        }
        setup_metal();  /* before the usage check; needs the final BATCH */
        if (st) { selftest(); return 0; }  /* status 0 even if the selftest failed */
        /* k <= MAXK = 32 is not checked; run() rejects t > 255 and K - D > 8 */
        if (k < 3 || t < 1) { fprintf(stderr, "usage: psp2gpu -k K -t T [-d prefixdepth] [-b batch] [-q] | -selftest\n"); return 1; }
        if (d == 0) d = k - 7; if (d < 2) d = 2;  /* default D = K-7 (7 GPU levels), at least 2 */
        run(k, t, d, 1);
    }
    return 0;
}
