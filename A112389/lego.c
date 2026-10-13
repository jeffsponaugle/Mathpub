/*
 * lego.c — count contiguous buildings of n 2x4 LEGO bricks (OEIS A112389)
 *          and the related counts A123829, A123831 and A112390
 *
 * =========================================================================
 * THEORY OF OPERATION
 * =========================================================================
 *
 * 1. What is counted
 * ------------------
 * The stud lattice is the integer grid; a "cell" is a unit square
 * [x,x+1) x [y,y+1).  A 2x4 brick lies in an integer layer z and covers
 * either a 4x2 block of cells ("H": [x,x+4) x [y,y+2)) or a 2x4 block
 * ("V": [x,x+2) x [y,y+4)).  Such a (z, x, y, o), with o = 0 for H and
 * o = 1 for V, is a "placement".
 *
 *   - Two placements in the same layer CONFLICT if their footprints share
 *     a cell: real bricks cannot interpenetrate.
 *   - Two placements in adjacent layers are ADJACENT (the studs snap
 *     together) if their footprints share a cell.  Every placement has
 *     exactly 46 adjacent placements in the layer above and 46 in the
 *     layer below: 21 with the same orientation and 25 crossed.
 *   - A BUILDING is a finite nonempty set of pairwise non-conflicting
 *     placements that is connected under adjacency.
 *
 * A112389(n) counts the buildings of n bricks up to translation (in x, y
 * and z) and rotation by multiples of 90 degrees about a vertical axis.
 * Reflections are not identified.
 *
 * The LAYER PROFILE (Deleuran's "refinement") of a building is the list
 * <z_0 z_1 ... z_{h-1}> of its brick counts per layer, bottom to top; h is
 * its height.  Everything below is computed per profile and summed.
 * (Turning a building upside down maps a profile to its reverse without
 * changing the counts.  This program does not rely on that; the helper
 * scripts do.)
 *
 * 2. Burnside: three numbers per profile
 * --------------------------------------
 * For a profile R let
 *   F(R)    = number of buildings up to translation only ("fixed"),
 *   S180(R) = number of those translation classes that a rotation by 180
 *             degrees (about some vertical axis) maps to themselves,
 *   S90(R)  = the same for a rotation by 90 degrees.
 * The rotation group C4 acts on translation classes, so Burnside's lemma
 * gives
 *   classes up to rotation:          count(R) = (F + S180 + 2*S90) / 4,
 *   classes fixed by 180 degrees:    sym(R)   = (S180 + S90) / 2.
 * (A class fixed by 90 degrees is alone in its orbit; one fixed by 180 but
 * not 90 degrees shares its orbit with exactly one other class.)  Output
 * is printed as "count (sym)", the convention of Deleuran's table, so
 * values can be compared directly.  Summing over profiles:
 *   A112389(n)   = sum of count(R) over the profiles R of n bricks,
 *   A123829(n)   = sum of sym(R),
 *   A123831(n-1) = sum of S180(R)/2 over the R with z_0 = 1 (a building on
 *                  a fixed base brick; the rotation must fix that brick),
 *   A112390(n,k) = sum of count(R) over the R of height k.
 * S90(R) = 0 unless every layer holds a multiple of 4 bricks: a 90-degree
 * rotation fixes no brick, and an orbit of 2 bricks would overlap itself.
 * So S90 only matters for n = 8, 12, ...
 *
 * 3. Bottlenecks
 * --------------
 * Suppose an interior layer i (0 < i < h-1) holds a single brick b.  Every
 * path from the bricks below b to the bricks above passes through b, so
 * the building is "lower part + b" and "b + upper part", chosen
 * independently.  Fixing b's position and orientation fixes the
 * translation; with b fixed each part is counted by F/2 of its own
 * profile, so
 *     F(<R1 1 R2>)    = F(<R1 1>) * F(<1 R2>) / 2,
 *     S180(<R1 1 R2>) = S180(<R1 1>) * S180(<1 R2>) / 2   (rotation about b),
 *     S90(<R1 1 R2>)  = 0.
 * A profile without such a layer is FAT.  Only fat profiles are enumerated;
 * compose() builds the others.
 *
 * 4. Data representation
 * ----------------------
 * Coordinates are shifted by OFF = 64 into a GW x GW = 128 x 128 grid, and
 * every search roots its buildings at grid (OFF, OFF).  With at most 11
 * bricks all coordinates stay inside the grid: within about OFF +- 35 for
 * the plain and symmetric searches, and within [3, 125] in the worst case
 * of the aggregated method, whose virtual edges jump up to 6 cells.  So
 * no bounds checks are made anywhere.
 *
 * A placement is packed into a 19-bit integer "pid":
 *     pid = z<<15 | y<<8 | x<<1 | o      (x, y grid coordinates, 7 bits).
 * Integer order on pids is lexicographic order on (z, y, x, o); "least
 * brick" always means least pid.  Stepping to another placement adds a
 * precomputed signed delta dz*2^15 + dy*256 + dx*2 + (o2 - o) (tables OVD,
 * CFD, VJD, VJX).  Inside the grid the fields never carry into each
 * other.  Callers check z > 0 before stepping down, because z = -1 would
 * wrap around to a huge pid.
 *
 * Occupancy is one 128-bit mask per (layer, row y).  A brick covers 2 rows
 * (H, 4 bits each) or 4 rows (V, 2 bits each), so a conflict test is a few
 * ORs and one AND (conflict()), and placing or removing a brick is an XOR
 * (toggle()).  A byte per pid ("mark") serves the Redelmeier search.
 *
 * 5. Redelmeier enumeration (frec, srec, arec)
 * --------------------------------------------
 * Redelmeier's algorithm (1981, for polyominoes) lists every connected set
 * of vertices containing a fixed root exactly once, on any graph whose
 * vertices are totally ordered.  It keeps a list of UNTRIED vertices.  A
 * level takes its untried vertices in order: it adds one (c), appends c's
 * neighbours that are larger than the root and not yet MARKED, recurses,
 * and removes c again.  A vertex is marked from the moment it is first
 * appended until the level that appended it returns.  So once a vertex has
 * been tried and dropped it is never offered again in that subtree, and
 * that is what makes every set appear exactly once.
 *
 * The untried list is one array G used as a stack.  A level owns the range
 * [s, e) of G.  When it adds G[i] it appends the new neighbours at
 * G[e..k), and the child level owns [i+1, k): the rest of this level's list
 * followed by the new vertices, with no copying.
 *
 * Translation is fixed by rooting every building at its least brick: the
 * root is placed at (z0, OFF, OFF) in both orientations, and only
 * placements with pid > root are ever added.
 *
 * Same-layer conflicts and per-layer caps are hereditary: once violated,
 * they stay violated in the whole subtree.  Every subset on a building's
 * generation path is itself a valid partial building, so candidates that
 * conflict are simply skipped without affecting uniqueness or completeness.
 *
 * The plain search (frec) adds two refinements:
 *   - fat pruning: an interior layer holding exactly one brick needs
 *     another one, so a branch with more such layers than bricks left
 *     cannot end in a fat profile and is cut;
 *   - leaf counting: at n-1 bricks the last brick is not placed.  Instead
 *     the valid candidates left in the untried range are counted per layer
 *     and credited to the profile each would complete (tally_leaf).
 *
 * Parallelism (all three searches): every thread walks the same top of the
 * search tree.  The nodes at depth SPLIT are numbered in DFS order, and a
 * thread descends into node t only after taking t from a shared atomic
 * counter.  That counter is the only communication, and the load balances
 * dynamically.  Each thread has private search state (Geo) and private
 * tallies, which are summed at the end.
 *
 * 6. Symmetric buildings (srec, count_sym)
 * ----------------------------------------
 * A building fixed by a rotation g about a vertical axis through c is a
 * union of g-orbits of placements.
 *   - 180 degrees: 2c is a lattice point, so there are 4 classes of centres
 *     modulo translation (CX2, CY2 in {0,1}).  Orbits have 2 bricks, or 1
 *     brick centred on c (possible only when c is a lattice point).
 *   - 90 degrees: c lies in Z^2 or (Z+1/2)^2, so 2 classes, and every
 *     valid orbit has 4 bricks.
 * The centre of a symmetric building is unique, because two different
 * centres would make it invariant under a translation.  So fixing the
 * centre class fixes the translation in x and y, and z is fixed by putting
 * the lowest layer at 0.
 *
 * srec runs Redelmeier on orbits.  Each orbit is identified by its least
 * brick, and two orbits are adjacent when some brick of one is adjacent to
 * some brick of the other.  The union of a connected set of orbits can
 * still fall apart into 2 or 4 pieces that the rotation permutes, so only
 * unions that are actually connected are counted.  A branch is cut when
 * its pieces can no longer be joined within the brick budget (gapbound()).
 * The root runs over all valid layer-0 orbits in a box around the centre.
 *
 * 7. Layer aggregation (arec, core_count): the fast method for one profile
 * -----------------------------------------------------------------------
 * Choose a set A of pairwise non-adjacent layers (not all of them); R is
 * the rest.  A brick in an A-layer touches only R-bricks, because its
 * neighbours lie in the adjacent layers.  Now fix the R-bricks (an
 * "R-set"), and let K_1..K_r be its components under real adjacency.  Each
 * A-brick touches some of the K_i (its MASK), and a choice of A-bricks gives
 * a connected building exactly when the masks together join all r
 * components into one block.  Writing that condition with the Moebius
 * function of the lattice of set partitions,
 *     [pi = 1] = sum over sigma >= pi of mu(sigma, 1),
 *     mu(sigma, 1) = (-1)^(b-1) * (b-1)!     (b = number of blocks),
 * and exchanging the order of summation gives
 *     #buildings with this R-set
 *       = sum over partitions sigma of {K_1..K_r} of
 *             mu(sigma, 1) * product over l in A of I_{z_l}(X^l_sigma)
 * where
 *     X^l       = the placements in layer l touching at least one R-brick
 *                 (the CANDIDATES; a brick touching nothing could never be
 *                 connected),
 *     X^l_sigma = the candidates whose mask lies inside one block of sigma,
 *     I_k(X)    = the number of k-subsets of X with no two bricks
 *                 overlapping.
 * The product is legitimate because for a fixed sigma the A-layers do not
 * interact: their bricks lie in different, non-adjacent layers.
 *
 * Enumerating the R-sets.  The R-part of a building need not be connected
 * by itself, but it is connected in an augmented graph.  That graph has a
 * "virtual" edge between two R-bricks whenever a single A-brick could
 * overlap both:
 *     VJD: same layer, 160 offsets per orientation (overlapping ones
 *          excluded);
 *     VJX: two layers apart, 206 offsets.
 * A path in the building from one R-brick to another passes through
 * A-bricks one at a time, and each such passage is a virtual edge.  So
 * Redelmeier on the augmented graph, rooted at the least R-brick (layer
 * ZR), produces the R-set of every building exactly once.  Some R-sets
 * produced this way admit no completion; they simply contribute 0.
 *
 * Symmetry of the plane (canon_weight).  Rotating or reflecting an R-set
 * (dihedral group D4) does not change its count, and the R-sets are closed
 * under these maps.  So only the least image of each R-set is evaluated,
 * weighted by 8 / |stabilizer|.
 *
 * Computing I_k on a union of mask classes.  Candidates are grouped by
 * mask, and X^l_sigma is always a union of such classes.
 *   - I_1 = N,  I_2 = C(N,2) - E,  I_3 = C(N,3) - E*(N-2) + P3 - K3,
 *     by inclusion-exclusion over the overlapping pairs inside a triple.
 *     Here E = overlapping pairs, P3 = paths of two edges = sum of
 *     C(deg,2), and K3 = triangles.  class_tables() tallies these per
 *     class, pair and triple of classes, so ind_class() evaluates any
 *     union in O(q^3) for q classes.
 *   - I_4 (ind4_formula) uses the same inclusion-exclusion over every edge
 *     set spanning at most 4 vertices.  The cliques come from the Helly
 *     property of axis-parallel boxes: pairwise-overlapping bricks share a
 *     common box of cells.  Hence
 *         #k-cliques = sum over grid features f of sign(f) * C(cover(f), k)
 *     where the features are cells (+), horizontally or vertically
 *     adjacent cell pairs (-) and 2x2 blocks (+), because this signed count
 *     of the common box's own features is 1.
 *   - I_k with k >= 5 (ind_big): DFS over the first k-4 bricks, with the I_4
 *     formula below that.  ind_dfs() is the brute-force reference used by
 *     the -x check.
 * Within one R-set many partitions sigma allow the same class set, so
 * memo_ind() caches I_k per (layer, k, class set).
 *
 * Sampling (-p): a task (search node at depth SPLIT) is processed with
 * probability p, decided by a fixed hash of its index.  Every run and every
 * build therefore picks the same tasks, and two builds can be compared
 * exactly.  F/p estimates F and elapsed/p the full run time.
 *
 * 8. Arithmetic
 * -------------
 * The per-profile F of the plain search fits in uint64 for n <= 11.  The
 * aggregated method sums F in unsigned __int128 and the Moebius sum in
 * signed __int128.  Terms carry factorials up to 9! and products of up to
 * NAGG values I_k; with at most 3 aggregated layers these stay far below
 * 2^127.  The I_k values themselves fit in int64 (at most C(512,5)).
 * Totals over profiles use __int128.  The integrality of
 * (F + S180 + 2*S90)/4 and (S180 + S90)/2 is checked and reported as
 * "NON-INTEGRAL" or "PARITY-FAIL".
 *
 * 9. Limits and error conditions
 * ------------------------------
 *   - n <= MAXN-1 = 11 in the modes N, -s, -r and -b, and n <= MAXN = 12
 *     for -S ("N out of range" otherwise).  Grid size, profile tables and
 *     per-brick arrays assume these bounds.
 *   - Threads: 1..MAXTHR = 256, the size of the fixed pthread_t arrays.  An
 *     explicit count outside that range, or not a number, is rejected
 *     (parse_threads); the default (online CPUs) is capped at MAXTHR.
 *   - Aggregated method:
 *       - at most MAXR = 10 R-components ("too many components");
 *       - at most 256 distinct masks per layer ("too many masks");
 *       - at most MAXC = 512 candidates per layer (an A-layer touches at
 *         most 10 R-bricks, so at most 460);
 *       - the I_k cache holds MEMO = 1024 entries per R-set.  If an R-set
 *         needs more (7+ components under some hand-picked plans), the
 *         extra values are computed without being cached: still exact,
 *         only slower.  The default plans stay far below the limit.
 *   - Internal consistency checks exit with "table error" or "partition
 *     error"; the -x debug checks exit with status 2 on a mismatch.
 *   - Profiles are digit strings, so a layer holds at most 9 bricks.
 *
 * =========================================================================
 * USAGE
 * =========================================================================
 *   ./lego N [threads]         a(1..N), a180(1..N) and the refinement table
 *                              of N, by plain enumeration (practical for
 *                              N <= 8; N <= 11)
 *   ./lego -s [threads]        self test against OEIS A112389 / A123829, n<=7
 *   ./lego -S N [threads]      symmetric counts only, all profiles, n<=N:
 *                              A123829(n) and A123831(n-1); -SV also
 *                              prints S180/S90 for every profile
 *   ./lego -r PROFILE [opts] [threads]
 *                              count one fat refinement by aggregation, e.g.
 *                              ./lego -r 353 -a 02
 *        -a LAYERS   aggregated layers (digits, pairwise non-adjacent);
 *                    default: fewest enumerated bricks, using only layers
 *                    of at most 4 bricks (choose_agg)
 *        -p FRAC     process a random fraction of tasks and extrapolate
 *                    (count and running time estimate)
 *        -d DEPTH    task split depth (default 4)
 *        -n          disable the D4 reduction (cross-check)
 *        -q          skip the task-count pre-pass (no % progress)
 *        -c N        use class tables when there are >= N partitions
 *                    (default 5; 1000000 = never, for cross-checks)
 *        -x 1        check every I_4 formula and class-table value
 *                    against brute force; -x 2 uses brute force only
 *   ./lego -b PROFILE [threads]
 *                              same refinement by plain enumeration
 *
 *   threads: 1..256; default = number of online CPUs (at most 256)
 */

#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

typedef unsigned __int128 u128;

/* Size limits (section 9 of the header).                                   */
#define MAXN 12          /* bricks; n <= MAXN-1 except for -S (n <= MAXN)  */
#define NZ 16            /* layers representable                           */
#define GW 128           /* grid width/height (offset coordinates)         */
#define OFF 64           /* raw (0,0) -> grid (OFF,OFF)                     */
#define NPL (1u << 19)   /* placement ids: z(4) y(7) x(7) o(1)             */
#define MAXPROF (1 << (MAXN - 1))   /* profiles of n <= MAXN: 2^(n-1)   */

/* placement id: (z<<15)|(y<<8)|(x<<1)|o, lexicographic in (z,y,x,o);   */
/* x, y are grid coordinates (raw + OFF), o = 0 for H (4x2), 1 for V.  */
#define PZ(p) ((int)((p) >> 15))
#define PY(p) ((int)(((p) >> 8) & 127))
#define PX(p) ((int)(((p) >> 1) & 127))
#define PO(p) ((int)((p) & 1))
#define MKP(z, x, y, o) (((uint32_t)(z) << 15) | ((uint32_t)(y) << 8) | ((uint32_t)(x) << 1) | (uint32_t)(o))

/* Globals shared by all modes.  They are set before any worker thread   */
/* starts and are read-only while the threads run.                       */
#define MAXTHR 256                /* max worker threads (pthread_t arrays) */
static int N;                     /* target number of bricks              */
static int NTHR = 1;              /* worker threads, 1..MAXTHR (default:   */
                                  /* online CPUs, capped at MAXTHR)        */

/* neighbour tables: for orientation o, the 46 (dx,dy,o2) with overlap, */
/* stored as pid deltas dy*256 + dx*2 + (o2 - o) (see init_tables())    */
static int32_t OVD[2][46];        /* pid delta within a layer (incl. self) */
static int NOV[2];
static int32_t CFD[2][45];        /* same-layer conflicts, excluding self  */

/*
 * now()
 *   Function: wall-clock time, for the timings printed with the results.
 *   Inputs:   none.
 *   Outputs:  seconds since the epoch, with microsecond resolution.
 *   Errors:   none.
 */
static double now(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/*
 * init_tables()
 *   Function: builds the neighbour tables used by every search.  For a
 *             brick of orientation o at the origin it lists the 46
 *             placements (dx, dy, o2) whose footprint shares a cell with
 *             it: 21 with o2 == o and 25 crossed.  They are stored as pid
 *             deltas dy*256 + dx*2 + (o2 - o).
 *               OVD[o][0..45]  all 46.  Adding +-(1 << 15) gives the 46
 *                              adjacent placements in the layer above or
 *                              below.  The zero delta (the brick straight
 *                              above or below) is included.
 *               CFD[o][0..44]  the same without the zero delta: the 45
 *                              other placements of the SAME layer that
 *                              conflict with the brick.
 *             Two boxes [x1, x1+w1) and [x2, x2+w2) overlap iff
 *             -w2 < x2 - x1 < w1, hence the loop bounds.
 *   Inputs:   none.
 *   Outputs:  fills OVD, NOV and CFD.
 *   Errors:   exits with "table error" if the counts are not 46 and 45,
 *             which would mean the geometry code is wrong.
 */
static void init_tables(void)
{
    for (int o = 0; o < 2; o++) {
        int w1 = o ? 2 : 4, h1 = o ? 4 : 2, k = 0, kc = 0;   /* this brick */
        for (int o2 = 0; o2 < 2; o2++) {
            int w2 = o2 ? 2 : 4, h2 = o2 ? 4 : 2;
            for (int dy = -h2 + 1; dy < h1; dy++)
                for (int dx = -w2 + 1; dx < w1; dx++) {
                    int32_t d = dy * 256 + dx * 2 + (o2 - o);
                    OVD[o][k++] = d;
                    if (d != 0) CFD[o][kc++] = d;
                }
        }
        NOV[o] = k;
        if (k != 46 || kc != 45) { fprintf(stderr, "table error\n"); exit(1); }
    }
}

/* ------------------------------------------------------------------ */
/* profiles (compositions of n) <-> index                              */
/*                                                                      */
/* A profile <z_0 ... z_{h-1}> of n bricks is a composition of n.  It   */
/* is indexed by the bit mask of its partial sums: bit s-1 is set iff   */
/* z_0 + ... + z_i = s for some i < h-1.  This is a bijection between   */
/* the compositions of n and [0, 2^(n-1)), so per-profile tallies are   */
/* plain arrays of MAXPROF entries.  Examples for n = 5: <2 3> has      */
/* partial sum 2 -> index 2; <1 2 2> has 1 and 3 -> index 5; <5> -> 0.  */

/*
 * prof_index()
 *   Function: index of a profile (see above).
 *   Inputs:   cnt[0..h-1]: layer counts, each >= 1; h >= 1; total n <= MAXN.
 *   Outputs:  the index, in [0, 2^(n-1)).
 *   Errors:   not checked: zero counts or n > MAXN give a wrong or
 *             out-of-range index.
 */
static int prof_index(const int *cnt, int h)
{
    int idx = 0, s = 0;
    for (int i = 0; i < h - 1; i++) { s += cnt[i]; idx |= 1 << (s - 1); }
    return idx;
}

/*
 * prof_decode()
 *   Function: inverse of prof_index(): the profile with index idx.
 *   Inputs:   idx in [0, 2^(n-1)); n >= 1; cnt with room for n ints.
 *   Outputs:  fills cnt[0..h-1] and returns the height h.
 *   Errors:   none; bits of idx at or above position n-1 are ignored.
 */
static int prof_decode(int idx, int n, int *cnt)   /* returns height */
{
    int h = 0, run = 0;
    for (int i = 0; i < n; i++) {
        run++;
        if (i == n - 1 || (idx >> i & 1)) { cnt[h++] = run; run = 0; }
    }
    return h;
}

/*
 * prof_fat()
 *   Function: whether a profile is "fat", i.e. no interior layer
 *             (0 < i < h-1) holds exactly one brick.  Only fat profiles are
 *             enumerated; the others are composed (header, section 3).
 *   Inputs:   cnt[0..h-1], h >= 1.
 *   Outputs:  1 if fat, 0 if the profile has a bottleneck.
 *   Errors:   none.
 */
static int prof_fat(const int *cnt, int h)
{
    for (int i = 1; i < h - 1; i++) if (cnt[i] == 1) return 0;
    return 1;
}

/*
 * prof_str()
 *   Function: a profile as a digit string, e.g. "353" for <3 5 3>.
 *   Inputs:   cnt[0..h-1]; buf with room for h+1 characters.
 *   Outputs:  NUL-terminated string in buf.
 *   Errors:   a layer of 10 or more bricks prints as a non-digit (':' for 10).
 *             The only such profiles for n <= 11 are single-layer ones,
 *             which have count 0.
 */
static void prof_str(const int *cnt, int h, char *buf)
{
    for (int i = 0; i < h; i++) buf[i] = (char)('0' + cnt[i]);
    buf[h] = 0;
}

/* ------------------------------------------------------------------ */
/* per-thread geometry state                                            */
/*                                                                      */
/* The state of one Redelmeier search (header, sections 4-5):           */
/*   occ[z][y]  128-bit mask of the occupied cells in row y of layer z. */
/*   mark[pid]  1 while pid sits in the untried list or was tried at an */
/*              ancestor level (Redelmeier's "seen" flag), else 0.      */
/*   G[]        the untried list, used as a stack.  At most 92 entries  */
/*              are pushed per brick in frec, 4*92 per orbit in srec,   */
/*              and 2*46 + 160 + 2*206 per R-brick in arec.  The largest */
/*              total, about 6000 in arec, stays below 8192.            */
/*   cnt[z]     bricks currently placed in layer z.                     */
/*   np, P[]    number and list of the placed bricks (read by           */
/*              core_count and canon_weight; srec keeps its own list).  */
/* Each Geo is about 600 KB, dominated by mark[].                       */

typedef struct {
    u128 occ[NZ][GW];
    uint8_t mark[NPL];
    uint32_t G[8192];
    int cnt[NZ];
    int np;
    uint32_t P[MAXN + 4];
} Geo;

/*
 * bmask()
 *   Function: the row mask of a brick: its 4 (H) or 2 (V) columns.
 *   Inputs:   pid p.
 *   Outputs:  128-bit mask with bits x .. x+3 (H) or x .. x+1 (V) set.
 *   Errors:   none; x <= 124 is guaranteed by the grid bounds (header, 4).
 */
static inline u128 bmask(uint32_t p) { return (u128)(PO(p) ? 0x3 : 0xF) << PX(p); }

/*
 * conflict()
 *   Function: whether placement p overlaps a brick already placed in its
 *             own layer.
 *   Inputs:   g: search state; p: pid.
 *   Outputs:  nonzero iff some cell of p is occupied.
 *   Errors:   none.  It reads rows y .. y+1 (H) or y .. y+3 (V), which lie
 *             inside the grid (header, section 4).
 */
static inline int conflict(const Geo *g, uint32_t p)
{
    int z = PZ(p), y = PY(p);
    u128 m = bmask(p);
    const u128 *r = g->occ[z];
    if (PO(p)) return ((r[y] | r[y + 1] | r[y + 2] | r[y + 3]) & m) != 0;
    return ((r[y] | r[y + 1]) & m) != 0;
}

/*
 * toggle()
 *   Function: places brick p if its cells are free, or removes it if it is
 *             placed (XOR of its cells into the occupancy rows).
 *   Inputs:   g: search state; p: pid.
 *   Outputs:  updates g->occ.
 *   Errors:   none checked: toggling a brick that overlaps a placed one
 *             corrupts occ.  Callers always test conflict() first and pair
 *             every place with a remove.
 */
static inline void toggle(Geo *g, uint32_t p)
{
    int z = PZ(p), y = PY(p), hgt = PO(p) ? 4 : 2;
    u128 m = bmask(p);
    for (int i = 0; i < hgt; i++) g->occ[z][y + i] ^= m;
}

/*
 * overlap2d()
 *   Function: whether the footprints of two placements share a cell,
 *             ignoring their layers.  It is used for adjacency (layers
 *             differ by 1) and for self-overlap of orbits (same layer).
 *   Inputs:   pids a, b.
 *   Outputs:  1 if the 2-D boxes intersect, else 0.
 *   Errors:   none.
 */
static inline int overlap2d(uint32_t a, uint32_t b)
{
    int wa = PO(a) ? 2 : 4, ha = PO(a) ? 4 : 2, wb = PO(b) ? 2 : 4, hb = PO(b) ? 4 : 2;
    int ax = PX(a), ay = PY(a), bx = PX(b), by = PY(b);
    return bx < ax + wa && ax < bx + wb && by < ay + ha && ay < by + hb;
}

/*
 * adjacent()
 *   Function: whether two placements snap together: adjacent layers and
 *             overlapping footprints.
 *   Inputs:   pids a, b.
 *   Outputs:  1 if adjacent, else 0.
 *   Errors:   none.
 */
static inline int adjacent(uint32_t a, uint32_t b)
{
    int dz = PZ(a) - PZ(b);
    return (dz == 1 || dz == -1) && overlap2d(a, b);
}

/* ------------------------------------------------------------------ */
/* F: Redelmeier enumeration of fat buildings, rooted at least brick    */
/*                                                                      */
/* Search parameters, set before the threads start (read-only during):  */
/*   CAP[z]    most bricks allowed in layer z: n for "any profile",     */
/*             z_l for one profile, 0 above the profile's height.       */
/*   HMAX      number of usable layers (placements need z < HMAX).      */
/*   REFMODE   1 when CAP is one exact profile (count_ref_base): then   */
/*             the fat pruning is off, because CAP already forces the   */
/*             profile.                                                 */
/*   SPLIT     depth (bricks placed) where the tree is cut into tasks;  */
/*             0 means single-threaded.                                 */
/*   next_task shared atomic task counter (header, section 5).          */

static int CAP[NZ];               /* per-layer caps (refinement mode)    */
static int HMAX;                  /* max height                          */
static int REFMODE;               /* 1: exact profile CAP[0..HMAX-1]      */
static int SPLIT;                 /* task split depth                    */
static atomic_int next_task;

/* Per-thread state of the plain search.                                */
typedef struct {
    Geo g;
    uint32_t root;                /* pid of the current root brick       */
    int task, claimed;            /* split nodes seen / task owned now   */
    uint64_t F[MAXPROF];          /* F per profile index (fat only)      */
    uint64_t nodes;               /* search nodes visited (statistics)   */
} FT;

/*
 * tally_leaf()
 *   Function: credits the last bricks found by leaf counting.  Adding one
 *             brick in layer z turns the current profile cnt[] into
 *             cnt + e_z; z == top+1 starts a new top layer.  If that
 *             profile is fat, lc[z] buildings are added to it.
 *   Inputs:   t: thread state (t->g.cnt holds the current profile of n-1
 *             bricks); lc[z]: number of valid last bricks in layer z;
 *             top: highest occupied layer.
 *   Outputs:  adds to t->F[].
 *   Errors:   none.
 */
static inline void tally_leaf(FT *t, const int *lc, int top)
{
    int *cnt = t->g.cnt;
    for (int z = 0; z <= top + 1 && z < NZ; z++) {
        if (!lc[z]) continue;
        cnt[z]++;
        int h = (z > top) ? z + 1 : top + 1;
        if (prof_fat(cnt, h)) t->F[prof_index(cnt, h)] += (uint64_t)lc[z];
        cnt[z]--;
    }
}

/*
 * frec()
 *   Function: one level of the plain Redelmeier search (header, section 5).
 *             It takes each untried placement G[i], i in [s, e), in turn.
 *             A placement whose layer is full or that conflicts is skipped.
 *             Otherwise it is placed, its unseen neighbours with pid > root
 *             are appended (above if z+1 < HMAX, below if z > 0), and then:
 *               - with fewer than n-1 bricks, recurse on [i+1, k);
 *               - at n-1 bricks, count the possible last bricks per layer
 *                 in [i+1, k) instead (leaf counting).
 *             Finally everything is undone.
 *   Inputs:   t: thread state holding the current partial building;
 *             [s, e): this level's untried range of t->g.G;
 *             top: highest occupied layer so far.
 *             Globals N, CAP, HMAX, REFMODE, SPLIT.
 *   Outputs:  adds to t->F[profile] (fat profiles only) and t->nodes.
 *   Errors:   none.  Requires N >= 3 (count_fixed answers N <= 2 directly)
 *             and the grid and stack bounds of header, section 9.
 */
static void frec(FT *t, int s, int e, int top)
{
    Geo *g = &t->g;
    for (int i = s; i < e; i++) {
        uint32_t c = g->G[i];
        int z = PZ(c);
        /* hereditary constraints: a full layer or a conflict stays so */
        if (g->cnt[z] >= CAP[z] || conflict(g, c)) continue;
        int np1 = g->np + 1;
        /* task split: every thread meets the split nodes in the same     */
        /* order; it enters only the one it has claimed, then claims next */
        if (np1 == SPLIT && SPLIT > 0) {
            if (t->task++ != t->claimed) continue;
            t->claimed = atomic_fetch_add(&next_task, 1);
        }
        toggle(g, c);
        g->cnt[z]++;
        g->P[g->np++] = c;
        int ntop = z > top ? z : top;
        /* prune: interior single layers need a brick each */
        int need = 0;
        for (int l = 1; l < ntop; l++) need += (g->cnt[l] == 1);
        if (REFMODE) need = 0;
        if (need <= N - np1) {
            /* append c's unseen neighbours after this level's range */
            int k = e;
            const int32_t *ov = OVD[PO(c)];
            for (int j = 0; j < 46; j++) {
                if (z + 1 < HMAX) {
                    uint32_t q = (uint32_t)((int32_t)c + (1 << 15) + ov[j]);
                    if (q > t->root && !g->mark[q]) { g->mark[q] = 1; g->G[k++] = q; }
                }
                if (z > 0) {
                    uint32_t q = (uint32_t)((int32_t)c - (1 << 15) + ov[j]);
                    if (q > t->root && !g->mark[q]) { g->mark[q] = 1; g->G[k++] = q; }
                }
            }
            t->nodes++;
            if (np1 == N - 1) {
                /* leaf counting: every valid entry of [i+1, k) completes */
                /* exactly one building whose path runs through here      */
                int lc[NZ + 1] = {0};
                for (int j = i + 1; j < k; j++) {
                    uint32_t q = g->G[j];
                    int zq = PZ(q);
                    if (g->cnt[zq] < CAP[zq] && !conflict(g, q)) lc[zq]++;
                }
                tally_leaf(t, lc, ntop);
            } else {
                frec(t, i + 1, k, ntop);
            }
            for (int j = e; j < k; j++) g->mark[g->G[j]] = 0;   /* unsee */
        }
        g->np--;
        g->cnt[z]--;
        toggle(g, c);
    }
}

/*
 * fthread()
 *   Function: worker thread of the plain search.  It runs frec() from the
 *             root brick at (0, OFF, OFF), first horizontal and then
 *             vertical, with the root as a one-element untried list.  The
 *             task numbering continues across the two roots, so all threads
 *             agree on it.
 *   Inputs:   arg: this thread's FT (zeroed by calloc).
 *   Outputs:  t->F and t->nodes.
 *   Errors:   none.
 */
static void *fthread(void *arg)
{
    FT *t = arg;
    t->claimed = atomic_fetch_add(&next_task, 1);
    for (int o = 0; o < 2; o++) {
        Geo *g = &t->g;
        memset(g->mark, 0, sizeof g->mark);
        memset(g->occ, 0, sizeof g->occ);
        memset(g->cnt, 0, sizeof g->cnt);
        uint32_t r = MKP(0, OFF, OFF, o);
        t->root = r;
        g->np = 0;
        /* root as a one-element untried list */
        g->G[0] = r;
        g->mark[r] = 1;
        frec(t, 0, 1, 0);
        g->mark[r] = 0;
    }
    return NULL;
}

/*
 * count_fixed()
 *   Function: F for all fat profiles of N bricks by the plain search, or,
 *             with REFMODE, for the single profile given by CAP.  It uses
 *             NTHR threads when N >= 7 (tasks at depth 5) and one thread
 *             otherwise.  N = 1 and N = 2 are answered directly: <1> has
 *             F = 2 (two orientations) and <11> has F = 2*46 = 92.
 *   Inputs:   globals N, CAP, HMAX, REFMODE, NTHR.
 *   Outputs:  Fout[profile index] for every fat profile, 0 elsewhere.
 *   Errors:   none expected: NTHR is in 1..MAXTHR (checked by main).  A
 *             calloc failure is not checked (each FT is about 600 KB).
 */
static void count_fixed(uint64_t *Fout)
{
    memset(Fout, 0, sizeof(uint64_t) * MAXPROF);
    if (N == 1) { Fout[0] = 2; return; }
    if (N == 2) { /* <11>: 2 orientations * 46 */
        Fout[1] = 92;
        return;
    }
    SPLIT = N - 1 > 5 ? 5 : 0;
    if (SPLIT && N - 1 <= SPLIT) SPLIT = 0;   /* (redundant safety check) */
    int nt = SPLIT ? NTHR : 1;
    atomic_store(&next_task, 0);
    FT *ts = calloc((size_t)nt, sizeof(FT));
    pthread_t th[MAXTHR];
    for (int i = 0; i < nt; i++) pthread_create(&th[i], NULL, fthread, &ts[i]);
    for (int i = 0; i < nt; i++) pthread_join(th[i], NULL);
    for (int i = 0; i < nt; i++)
        for (int p = 0; p < MAXPROF; p++) Fout[p] += ts[i].F[p];
    free(ts);
}

/* ------------------------------------------------------------------ */
/* S180 / S90: Redelmeier over orbits under rotation about a centre     */

/* Rotation group of the current search (header, section 6): order GK,  */
/* centre c = (CX2/2, CY2/2) in raw coordinates.  CX2 and CY2 pick one   */
/* of the 4 (180 degrees) or 2 (90 degrees, CX2 == CY2) centre classes.  */
static int GK;          /* group order: 2 (180) or 4 (90)                */
static int CX2, CY2;    /* twice the centre, raw coords, in {0,1}        */

/*
 * rot()
 *   Function: image of a placement under the group's generator: 180
 *             degrees, or 90 degrees counterclockwise, about the centre c.
 *             Point maps: 180: (u,v) -> 2c - (u,v);
 *                         90:  (u,v) -> (cx + cy - v, cy - cx + u).
 *             Applying them to the box corners gives
 *               180, H at x: H at 2cx - x - 4 (V: - 2), y similarly;
 *               90,  H (x,y) -> V (cx+cy - y - 2, cy-cx + x),
 *                    V (x,y) -> H (cx+cy - y - 4, cy-cx + x).
 *             In grid coordinates every raw term gets an OFF: hence the
 *             2*OFF.  For 90 degrees cx+cy and cy-cx are integers because
 *             CX2 == CY2.
 *   Inputs:   pid p; globals GK, CX2, CY2.
 *   Outputs:  pid of the image (same layer).
 *   Errors:   none; images stay in the grid for the root box of count_sym.
 */
static inline uint32_t rot(uint32_t p)
{
    int z = PZ(p), x = PX(p), y = PY(p), o = PO(p);
    int nx, ny, no;
    if (GK == 2) {
        if (!o) { nx = CX2 + 2 * OFF - x - 4; ny = CY2 + 2 * OFF - y - 2; }
        else    { nx = CX2 + 2 * OFF - x - 2; ny = CY2 + 2 * OFF - y - 4; }
        no = o;
    } else {
        int s = (CX2 + CY2) / 2, d = (CY2 - CX2) / 2;
        if (!o) { nx = s + 2 * OFF - y - 2; ny = d + x; }
        else    { nx = s + 2 * OFF - y - 4; ny = d + x; }
        no = 1 - o;
    }
    return MKP(z, nx, ny, no);
}

/*
 * orbit()
 *   Function: the orbit of p under the group: p, rot(p), rot(rot(p)), ...
 *   Inputs:   pid p; el with room for 4 pids.
 *   Outputs:  fills el[0..m-1] (el[0] = p) and returns m: 1, 2 or 4.
 *             Under 90 degrees m = 2 means rot(rot(p)) = p, an orbit of two
 *             bricks centred on c.  Those two overlap, and orbit_ok()
 *             rejects them.
 *   Errors:   none.
 */
static inline int orbit(uint32_t p, uint32_t *el)
{
    int m = 0;
    uint32_t q = p;
    do { el[m++] = q; q = rot(q); } while (q != p && m < 4);
    return m;
}

/*
 * orep()
 *   Function: the representative of p's orbit, i.e. its least pid.  Orbits
 *             are identified by it, so the mark array is indexed by it.
 *   Inputs:   pid p.
 *   Outputs:  least pid of the orbit.
 *   Errors:   none.
 */
static inline uint32_t orep(uint32_t p)
{
    uint32_t el[4], r = p;
    int m = orbit(p, el);
    for (int i = 1; i < m; i++) if (el[i] < r) r = el[i];
    return r;
}

/*
 * orbit_ok()
 *   Function: whether the bricks of one orbit (all in the same layer)
 *             avoid each other, so that the orbit can be placed at all.
 *   Inputs:   el[0..m-1].
 *   Outputs:  1 if pairwise non-overlapping, else 0.
 *   Errors:   none.
 */
static int orbit_ok(const uint32_t *el, int m)
{
    for (int i = 0; i < m; i++)
        for (int j = i + 1; j < m; j++)
            if (overlap2d(el[i], el[j])) return 0;
    return 1;
}

/*
 * gapbound()
 *   Function: a lower bound on the number of bricks strictly between a and
 *             b on any path of adjacent bricks joining them.  Each step
 *             changes z by +-1 and x, y by at most 3 cells.  So a path of
 *             L steps needs L >= |dz|, L >= |dx|/3, L >= |dy|/3 and
 *             L == |dz| (mod 2).  Two different placements in one layer need
 *             L >= 2.  The bound returned is L - 1.
 *   Inputs:   pids a, b.
 *   Outputs:  the bound; 0 when a == b.
 *   Errors:   none.
 */
static inline int gapbound(uint32_t a, uint32_t b)
{
    int dz = abs(PZ(a) - PZ(b)), dx = abs(PX(a) - PX(b)), dy = abs(PY(a) - PY(b));
    int L = dz;
    int lx = (dx + 2) / 3, ly = (dy + 2) / 3;
    if (lx > L) L = lx;
    if (ly > L) L = ly;
    if ((L - dz) & 1) L++;
    if (L == 0 && a != b) L = 2;
    return L > 0 ? L - 1 : 0;
}

/* Per-thread state of the symmetric search.                           */
typedef struct {
    Geo g;
    uint32_t root;                /* representative of the root orbit    */
    int nb;                       /* bricks placed (all orbit elements)  */
    uint32_t B[MAXN + 4];         /* the placed bricks                   */
    uint64_t S[MAXPROF];          /* invariant buildings per profile     */
    uint64_t nodes;               /* search nodes visited (statistics)   */
} ST;

/* root orbits of the current centre class, shared by the threads, which */
/* claim them through next_root                                          */
static uint32_t *roots;
static int nroots;
static atomic_int next_root;

/*
 * components()
 *   Function: connected components of the placed bricks under adjacency
 *             (depth-first search, O(nb^2) adjacency tests).
 *   Inputs:   t: symmetric-search state (t->B[0..nb-1]); comp with room
 *             for nb ints.
 *   Outputs:  comp[i] = component of brick i, numbered from 0; returns the
 *             number of components.
 *   Errors:   none.
 */
static int components(const ST *t, int *comp)
{
    int nc = 0, st[MAXN + 4];
    for (int i = 0; i < t->nb; i++) comp[i] = -1;
    for (int i = 0; i < t->nb; i++) {
        if (comp[i] >= 0) continue;
        int sp = 0;
        st[sp++] = i;
        comp[i] = nc;
        while (sp) {
            int u = st[--sp];
            for (int v = 0; v < t->nb; v++)
                if (comp[v] < 0 && adjacent(t->B[u], t->B[v])) { comp[v] = nc; st[sp++] = v; }
        }
        nc++;
    }
    return nc;
}

/*
 * srec()
 *   Function: one level of the orbit search (header, section 6).  Each
 *             untried entry G[i] is an orbit representative.  The orbit is
 *             skipped if it would exceed N bricks or a layer cap, overlaps
 *             itself, or conflicts with placed bricks.  Otherwise all its
 *             bricks are placed and the components of the union are
 *             computed.  Then:
 *               - at N bricks, the building is counted if it is connected;
 *               - below N, the branch is cut if the closest pair of bricks
 *                 in different components cannot be joined with the
 *                 remaining bricks (gapbound).  Otherwise the unseen
 *                 neighbouring orbits (by representative, > root) are
 *                 appended and the search recurses on [i+1, k).
 *   Inputs:   t: thread state (placed bricks in t->B[0..nb-1]);
 *             [s, e): untried range.  Globals N, CAP, HMAX, GK, CX2, CY2.
 *   Outputs:  adds to t->S[profile] and t->nodes.
 *   Errors:   none.
 */
static void srec(ST *t, int s, int e)
{
    Geo *g = &t->g;
    for (int i = s; i < e; i++) {
        uint32_t q = g->G[i];
        uint32_t el[4];
        int m = orbit(q, el);
        if (t->nb + m > N) continue;
        int z = PZ(q);
        if (g->cnt[z] + m > CAP[z]) continue;
        if (!orbit_ok(el, m)) continue;
        int bad = 0;
        for (int j = 0; j < m && !bad; j++) bad = conflict(g, el[j]);
        if (bad) continue;
        for (int j = 0; j < m; j++) { toggle(g, el[j]); t->B[t->nb++] = el[j]; }
        g->cnt[z] += m;
        t->nodes++;
        int comp[MAXN + 4];
        int nc = components(t, comp);
        if (t->nb == N) {
            /* complete: count it only if the union is really connected */
            if (nc == 1) {
                int h = 0;
                while (h < NZ && g->cnt[h]) h++;
                t->S[prof_index(g->cnt, h)]++;
            }
        } else {
            int ok = 1;
            if (nc > 1) {
                /* the pieces must still be joinable: at least gapbound()   */
                /* new bricks lie between some pair in different pieces     */
                int lb = 1 << 20;
                for (int a = 0; a < t->nb; a++)
                    for (int b = a + 1; b < t->nb; b++)
                        if (comp[a] != comp[b]) {
                            int gb = gapbound(t->B[a], t->B[b]);
                            if (gb < lb) lb = gb;
                        }
                if (t->nb + lb > N) ok = 0;
            }
            if (ok) {
                /* append the unseen orbits adjacent to any brick of this one */
                int k = e;
                for (int j = 0; j < m; j++) {
                    uint32_t c = el[j];
                    const int32_t *ov = OVD[PO(c)];
                    int zc = PZ(c);
                    for (int d = 0; d < 46; d++) {
                        if (zc + 1 < HMAX) {
                            uint32_t f = (uint32_t)((int32_t)c + (1 << 15) + ov[d]);
                            uint32_t r = orep(f);
                            if (r > t->root && !g->mark[r]) { g->mark[r] = 1; g->G[k++] = r; }
                        }
                        if (zc > 0) {
                            uint32_t f = (uint32_t)((int32_t)c - (1 << 15) + ov[d]);
                            uint32_t r = orep(f);
                            if (r > t->root && !g->mark[r]) { g->mark[r] = 1; g->G[k++] = r; }
                        }
                    }
                }
                srec(t, i + 1, k);
                for (int j = e; j < k; j++) g->mark[g->G[j]] = 0;
            }
        }
        g->cnt[z] -= m;
        for (int j = 0; j < m; j++) { toggle(g, el[j]); t->nb--; }
    }
}

/*
 * sthread()
 *   Function: worker thread of the symmetric search.  It claims root
 *             orbits from next_root until none are left and runs srec()
 *             from each, with the root as a one-element untried list.
 *   Inputs:   arg: this thread's ST.  Globals roots, nroots, next_root.
 *   Outputs:  t->S, t->nodes.
 *   Errors:   none.
 */
static void *sthread(void *arg)
{
    ST *t = arg;
    Geo *g = &t->g;
    memset(g->mark, 0, sizeof g->mark);
    memset(g->occ, 0, sizeof g->occ);
    memset(g->cnt, 0, sizeof g->cnt);
    for (;;) {
        int ri = atomic_fetch_add(&next_root, 1);
        if (ri >= nroots) break;
        uint32_t r = roots[ri];
        t->root = r;
        t->nb = 0;
        g->G[0] = r;
        g->mark[r] = 1;
        srec(t, 0, 1);
        g->mark[r] = 0;
    }
    return NULL;
}

/*
 * count_sym()
 *   Function: S180 (gk = 2) or S90 (gk = 4) for every profile of N bricks
 *             allowed by CAP and HMAX.  For each centre class (4 for 180
 *             degrees, 2 for 90) it lists the possible root orbits.  A root
 *             orbit lies in layer 0 within R = 3(N-1)/2 + 6 cells of the
 *             centre, is its own representative, has the right size and
 *             does not overlap itself, and its bricks could be joined within
 *             N bricks.  NTHR threads then share these roots.
 *             Why R suffices: in an N-brick symmetric building each brick
 *             is joined to its own image by a path of at most N bricks, each
 *             step moving at most 3 cells.  So every brick lies within about
 *             3(N-1)/2 + 4 cells of the centre.
 *   Inputs:   gk in {2, 4}; globals N, CAP, HMAX, NTHR.
 *   Outputs:  Sout[profile index] = number of invariant translation classes.
 *   Errors:   none expected: NTHR is in 1..MAXTHR (checked by main).
 *             malloc and calloc failures are not checked.
 */
static void count_sym(int gk, uint64_t *Sout)
{
    memset(Sout, 0, sizeof(uint64_t) * MAXPROF);
    GK = gk;
    int R = 3 * (N - 1) / 2 + 6;
    ST *ts = calloc((size_t)NTHR, sizeof(ST));
    roots = malloc(sizeof(uint32_t) * (size_t)(2 * (2 * R + 1) * (2 * R + 1)));
    for (int cls = 0; cls < 4; cls++) {
        CX2 = cls & 1;
        CY2 = cls >> 1;
        if (gk == 4 && CX2 != CY2) continue;
        nroots = 0;
        for (int y = -R; y <= R; y++)
            for (int x = -R; x <= R; x++)
                for (int o = 0; o < 2; o++) {
                    uint32_t p = MKP(0, OFF + x, OFF + y, o), el[4];
                    if (orep(p) != p) continue;
                    int m = orbit(p, el);
                    /* sizes: 2 or 1 (centred brick) for 180 deg, 4 for 90 */
                    if (m != gk && !(gk == 2 && m == 1)) continue;
                    if (m > N || !orbit_ok(el, m)) continue;
                    if (m > 1) {  /* all orbit elements must be joinable */
                        int lb = gapbound(el[0], el[1]);
                        if (m + lb > N) continue;
                    }
                    roots[nroots++] = p;
                }
        atomic_store(&next_root, 0);
        pthread_t th[MAXTHR];
        for (int i = 0; i < NTHR; i++) pthread_create(&th[i], NULL, sthread, &ts[i]);
        for (int i = 0; i < NTHR; i++) pthread_join(th[i], NULL);
    }
    for (int i = 0; i < NTHR; i++)
        for (int p = 0; p < MAXPROF; p++) Sout[p] += ts[i].S[p];
    free(roots);
    free(ts);
}

/* ------------------------------------------------------------------ */
/* Aggregated counter for one fat refinement.                           */
/*                                                                      */
/* Choose a set A of pairwise non-adjacent layers ("aggregated"); the   */
/* other layers R are enumerated explicitly.  A brick of an aggregated  */
/* layer touches only R-bricks, so given the R-bricks (possibly         */
/* disconnected, real components K_1..K_r) the aggregated layers are    */
/* independent of each other except through connectivity, and Moebius   */
/* inversion on the partition lattice of {K_i} gives                    */
/*                                                                      */
/*   #buildings with these R-bricks                                     */
/*      = sum_sigma mu(sigma, 1) * prod_{l in A} I_{z_l}(X^l_sigma)     */
/*                                                                      */
/* where X^l = placements in layer l touching an R-brick, X^l_sigma the */
/* ones whose touched components lie inside one block of sigma, and     */
/* I_k(X) = number of k-sets of pairwise non-overlapping bricks of X.   */
/* R is enumerated by Redelmeier on an augmented graph: real overlaps   */
/* between R-layers, plus "virtual" edges between two R-bricks that a   */
/* single brick of an aggregated layer could overlap both (same layer   */
/* or two layers apart).  A building's R-part is connected in this      */
/* graph, so each is generated exactly once, rooted at its least brick. */
/*                                                                      */
/* See header, section 7, for the derivation.  Code map:                */
/*   init_vj        offsets of the virtual edges                        */
/*   collect        candidate set X^l with masks and conflict bitsets   */
/*   class_tables, ind_class   I_1..I_3 of a union of mask classes      */
/*   ind4_formula, k4_features, ind_big, ind_dfs, ind_count   I_k       */
/*   init_parts, xsigma, allowed_classes   the partitions sigma         */
/*   memo_ind       cached I_k per (layer, k, allowed class set)        */
/*   core_count     the Moebius sum for one R-set                       */
/*   canon_weight   D4 reduction                                        */
/*   arec, athread  enumeration of the R-sets                           */
/*   count_ref_agg  driver (threads, pre-pass, progress, sampling)      */

static void u128str(u128 v, char *buf);

/* An overlap offset as a triple: brick (dx, dy, o2) relative to a brick */
/* of orientation o at the origin.  OVT[o] lists the 46 of them (same    */
/* order as OVD); VJD/VJX are two-step compositions, as pid deltas.      */
typedef struct { int8_t dx, dy, o2; } Off;
static Off OVT[2][46];
static int32_t VJD[2][512];      /* same layer, via one brick above/below */
static int NVJ[2];
static int32_t VJX[2][512];      /* two layers apart, via one brick       */
static int NVX[2];

/*
 * init_vj()
 *   Function: builds the virtual-edge offsets of the augmented graph.  A
 *             brick b2 is reachable from b in two overlap steps if a brick
 *             e in the layer between (or beside) overlaps both, i.e. b2 =
 *             b + OVT[o][i] + OVT[o_e][j].  All distinct such offsets go
 *             into
 *               VJX[o]: R-bricks two layers apart (206 per orientation),
 *               VJD[o]: R-bricks in the same layer (160), without those
 *                       that overlap b itself (they could not coexist).
 *   Inputs:   none.
 *   Outputs:  fills OVT, VJD, NVJ, VJX, NVX.
 *   Errors:   none; at most 46*46 offsets fit the 512-entry tables, and
 *             the components stay within [-6, 6], inside seen[][] range.
 */
static void init_vj(void)
{
    for (int o = 0; o < 2; o++) {
        int w1 = o ? 2 : 4, h1 = o ? 4 : 2, k = 0;
        for (int o2 = 0; o2 < 2; o2++) {
            int w2 = o2 ? 2 : 4, h2 = o2 ? 4 : 2;
            for (int dy = -h2 + 1; dy < h1; dy++)
                for (int dx = -w2 + 1; dx < w1; dx++) OVT[o][k++] = (Off){(int8_t)dx, (int8_t)dy, (int8_t)o2};
        }
    }
    for (int o = 0; o < 2; o++) {
        static uint8_t seen[64][64][2];
        memset(seen, 0, sizeof seen);
        int n = 0, nx = 0;
        for (int i = 0; i < 46; i++) {
            Off a = OVT[o][i];
            for (int j = 0; j < 46; j++) {
                Off b = OVT[a.o2][j];
                int dx = a.dx + b.dx, dy = a.dy + b.dy, o2 = b.o2;
                if (seen[dx + 32][dy + 32][o2]) continue;
                seen[dx + 32][dy + 32][o2] = 1;
                int32_t d = dy * 256 + dx * 2 + (o2 - o);
                VJX[o][nx++] = d;
                /* same layer: exclude bricks overlapping the origin brick */
                int ov = 0;
                for (int q = 0; q < 46; q++)
                    if (OVT[o][q].dx == dx && OVT[o][q].dy == dy && OVT[o][q].o2 == o2) ov = 1;
                if (!ov) VJD[o][n++] = d;
            }
        }
        NVJ[o] = n;
        NVX[o] = nx;
    }
}

#define MAXC 512                 /* candidates per aggregated layer       */
#define CW (MAXC / 64)           /* 64-bit words of a candidate bitset    */
#define MAXAGG 6                 /* aggregated layers                     */
#define CTQ 20                   /* max mask classes for class tables     */

/*
 * Candidate set X^l of one aggregated layer for the current R-set
 * (built by collect()):
 *   n, p[i]     number of candidates and their pids.
 *   mask[i]     bit j set iff candidate i touches R-component j.
 *   adj[i]      bitset of the candidates overlapping candidate i (built
 *               only when some I_k with k >= 2 is needed); W = number of
 *               64-bit words in use.
 *   nm, mv[m], mbits[m], cls[i]
 *               the distinct masks ("classes"), the bitset of each class's
 *               candidates, and the class of each candidate.
 *   ct_ok, nci, eij, Q, tri
 *               class tables (class_tables()), valid for I_k with
 *               k <= ct_ok; 0 = not built.  nci[a] = class sizes,
 *               eij[a][b] = overlapping pairs between classes a <= b,
 *               Q[a][b][c] = sum over v in class a of deg_b(v)*deg_c(v),
 *               tri[a][b][c] = triangles by sorted class triple.
 */
typedef struct {
    int n, W;
    uint32_t p[MAXC];
    uint16_t mask[MAXC];
    uint64_t adj[MAXC][CW];
    int nm;                  /* distinct masks present */
    uint16_t mv[256];
    uint64_t mbits[256][CW];
    uint8_t cls[MAXC];       /* mask class of each candidate */
    int ct_ok;               /* class tables below are valid */
    int32_t nci[CTQ];
    int32_t eij[CTQ][CTQ];
    int64_t Q[CTQ][CTQ][CTQ];
    int32_t tri[CTQ][CTQ][CTQ];
} Cand;

/* memo_ind() cache entry: I_k of the class set `key` of layer `tag`;   */
/* valid only if gen equals the current R-set's generation number.      */
#define MEMO 1024
typedef struct { uint64_t key; int64_t val; uint32_t gen; uint16_t tag, k; } MemoE;

/* The plan of the current profile (set by count_ref_agg):               */
/*   AMASK/RMASK  bit masks of the aggregated / enumerated layers;       */
/*   ZR           lowest R-layer (where the root brick goes);            */
/*   MR           number of R-bricks;                                    */
/*   NAGG, AGL[]  number and list of aggregated layers.                  */
static int AMASK, RMASK, ZR, MR, NAGG;
static int AGL[MAXAGG];          /* aggregated layers                    */

/* Per-thread state of the aggregated method.                           */
/*   g, root          the R-set search (Geo.P holds the R-bricks)        */
/*   task, claimed    task split (as in FT)                              */
/*   F, cores         sum of weighted counts; R-sets evaluated           */
/*   cmark[pid]       index+1 of a candidate during collect(), else 0    */
/*   cd[]             candidate set of each aggregated layer             */
/*   memogen, memo[]  I_k cache; a new generation number per R-set       */
/*   fg[4][y][x]      feature counters for k4_features (all zero between */
/*                    calls)                                             */
/* Each AT is a few MB (cd[] is the bulk).                               */
typedef struct {
    Geo g;
    uint32_t root;
    int task, claimed;
    u128 F;
    uint64_t cores;
    uint16_t cmark[NPL];
    Cand cd[MAXAGG];
    uint32_t memogen;
    MemoE memo[MEMO];
    uint8_t fg[4][GW][GW];
} AT;

/* whether layer z exists and is enumerated (inR) / aggregated (inA)   */
static inline int inR(int z) { return z >= 0 && z < NZ && (RMASK >> z & 1); }
static inline int inA(int z) { return z >= 0 && z < NZ && (AMASK >> z & 1); }

/*
 * collect()
 *   Function: builds the candidate set X^l of aggregated layer `layer`.
 *             These are all placements of that layer overlapping at least
 *             one R-brick in layer-1 or layer+1, each with the mask of
 *             R-components it touches.  It deduplicates through t->cmark
 *             (index+1 per pid, cleared again before returning), builds
 *             the conflict bitsets when k >= 2, and groups the candidates
 *             into mask classes.
 *   Inputs:   t: thread state (R-set in t->g.P[0..np-1]); c: output;
 *             layer: the aggregated layer; k: the largest I_k that will be
 *             requested (decides whether adj is needed); comp[i]: the
 *             component of R-brick i.
 *   Outputs:  fills c (n, p, mask, W, adj if k >= 2, nm, mv, mbits, cls)
 *             and sets c->ct_ok = 0.
 *   Errors:   exits with "too many masks" past 256 distinct masks.  More
 *             than MAXC candidates would overflow, but at most 10 R-bricks
 *             touch a layer, so there are at most 460.  comp[] values must
 *             be below 16 (mask is 16 bits; MAXR = 10 guarantees it).
 */
static void collect(AT *t, Cand *c, int layer, int k, const int *comp)
{
    Geo *g = &t->g;
    c->n = 0;
    for (int i = 0; i < g->np; i++) {
        uint32_t b = g->P[i];
        int dz = layer - PZ(b);
        if (dz != 1 && dz != -1) continue;
        const int32_t *ov = OVD[PO(b)];
        int32_t base = (int32_t)b + dz * (1 << 15);
        for (int j = 0; j < 46; j++) {
            uint32_t q = (uint32_t)(base + ov[j]);
            int kk = t->cmark[q];
            if (!kk) {
                kk = ++c->n;
                t->cmark[q] = (uint16_t)kk;
                c->p[kk - 1] = q;
                c->mask[kk - 1] = 0;
            }
            c->mask[kk - 1] |= (uint16_t)(1u << comp[i]);
        }
    }
    c->W = (c->n + 63) / 64;
    if (k >= 2) {
        /* conflict graph: look up the 45 same-layer conflicts of each */
        for (int i = 0; i < c->n; i++) {
            memset(c->adj[i], 0, sizeof(uint64_t) * (size_t)c->W);
            uint32_t q = c->p[i];
            const int32_t *cf = CFD[PO(q)];
            for (int j = 0; j < 45; j++) {
                int kk = t->cmark[(uint32_t)((int32_t)q + cf[j])];
                if (kk) c->adj[i][(kk - 1) >> 6] |= 1ull << ((kk - 1) & 63);
            }
        }
    }
    for (int i = 0; i < c->n; i++) t->cmark[c->p[i]] = 0;
    /* group by mask (linear search; the number of classes is small) */
    c->nm = 0;
    for (int i = 0; i < c->n; i++) {
        int m;
        for (m = 0; m < c->nm; m++) if (c->mv[m] == c->mask[i]) break;
        if (m == c->nm) {
            if (m == 256) { fprintf(stderr, "too many masks\n"); exit(1); }
            c->mv[m] = c->mask[i];
            memset(c->mbits[m], 0, sizeof(uint64_t) * CW);
            c->nm++;
        }
        c->mbits[m][i >> 6] |= 1ull << (i & 63);
        c->cls[i] = (uint8_t)m;
    }
    c->ct_ok = 0;
}

/* Per-class tallies so that I_1..I_3 of any union of classes costs O(q^3): */
/* n_i, edges e_ij, Q_ijl = sum_{v in class i} deg_j(v) deg_l(v), and        */
/* triangles by sorted class triple.  Then for S = union of classes          */
/*   E_S = sum e_ij, Ch_S = (sum_{i,j,l in S} Q_ijl - 2E_S)/2, Tr_S = sum tri */
/*
 * class_tables()
 *   Function: builds the tables above for the candidate set c.
 *             Ch_S = sum over v in S of C(d_S(v), 2) with
 *             d_S(v) = sum over j in S of deg_j(v), and
 *             sum of d_S(v)^2 = sum over i, j, l in S of Q_ijl.
 *             Each edge (u < v) and each triangle (u < v < w) is visited
 *             once through the adjacency bitsets.
 *   Inputs:   c: candidate set from collect() with adj built if k >= 2;
 *             k: largest I_k needed.
 *   Outputs:  fills nci (and eij for k >= 2, plus Q and tri for k == 3) and
 *             sets c->ct_ok = k.  It leaves ct_ok = 0 (tables unusable) when
 *             there are more than CTQ classes or k > 3.
 *   Errors:   none.  Counts fit easily: degrees <= 45 in int16, triangles
 *             per class triple <= C(512,3) in int32.
 */
static void class_tables(Cand *c, int k)
{
    int q = c->nm, n = c->n, W = c->W;
    c->ct_ok = 0;
    if (q > CTQ || k > 3) return;
    memset(c->nci, 0, sizeof c->nci);
    for (int i = 0; i < n; i++) c->nci[c->cls[i]]++;
    if (k >= 2) {
        for (int a = 0; a < q; a++) memset(c->eij[a], 0, sizeof(int32_t) * (size_t)q);
        static __thread int16_t degc[MAXC][CTQ];
        if (k == 3) {
            for (int a = 0; a < q; a++)
                for (int b = 0; b < q; b++) {
                    memset(c->Q[a][b], 0, sizeof(int64_t) * (size_t)q);
                    memset(c->tri[a][b], 0, sizeof(int32_t) * (size_t)q);
                }
            for (int i = 0; i < n; i++) memset(degc[i], 0, sizeof(int16_t) * (size_t)q);
        }
        for (int u = 0; u < n; u++) {
            int cu = c->cls[u];
            for (int w = u >> 6; w < W; w++) {
                uint64_t y = c->adj[u][w];
                if (w == (u >> 6)) y &= ~((2ull << (u & 63)) - 1);
                while (y) {
                    int v = w * 64 + __builtin_ctzll(y);
                    y &= y - 1;
                    int cv = c->cls[v];
                    if (cu <= cv) c->eij[cu][cv]++; else c->eij[cv][cu]++;
                    if (k < 3) continue;
                    degc[u][cv]++;
                    degc[v][cu]++;
                    for (int x = v >> 6; x < W; x++) {
                        uint64_t z = c->adj[u][x] & c->adj[v][x];
                        if (x == (v >> 6)) z &= ~((2ull << (v & 63)) - 1);
                        while (z) {
                            int tw = x * 64 + __builtin_ctzll(z);
                            z &= z - 1;
                            int a = cu, b = cv, d = c->cls[tw], tmp;
                            if (a > b) { tmp = a; a = b; b = tmp; }
                            if (b > d) { tmp = b; b = d; d = tmp; }
                            if (a > b) { tmp = a; a = b; b = tmp; }
                            c->tri[a][b][d]++;
                        }
                    }
                }
            }
        }
        if (k == 3) {
            for (int v = 0; v < n; v++) {
                int nz[CTQ], nn = 0, cv = c->cls[v];
                for (int j = 0; j < q; j++) if (degc[v][j]) nz[nn++] = j;
                for (int a = 0; a < nn; a++)
                    for (int b = 0; b < nn; b++)
                        c->Q[cv][nz[a]][nz[b]] += (int64_t)degc[v][nz[a]] * degc[v][nz[b]];
            }
        }
    }
    c->ct_ok = k;
}

/*
 * ind_class()
 *   Function: I_k (k <= 3) of the union of the classes in `key`, from the
 *             class tables: I_1 = N, I_2 = C(N,2) - E,
 *             I_3 = C(N,3) - E(N-2) + Ch - Tr.
 *   Inputs:   c with ct_ok >= k; key: bit m set for each class m in the
 *             union (nm <= CTQ); k in {1, 2, 3}.
 *   Outputs:  the count.
 *   Errors:   none checked; the caller ensures ct_ok >= k.
 */
static int64_t ind_class(const Cand *c, uint64_t key, int k)
{
    int s[CTQ], ns = 0;
    for (int i = 0; i < c->nm; i++) if (key >> i & 1) s[ns++] = i;
    int64_t N0 = 0;
    for (int a = 0; a < ns; a++) N0 += c->nci[s[a]];
    if (k == 1) return N0;
    int64_t E = 0;
    for (int a = 0; a < ns; a++)
        for (int b = a; b < ns; b++) E += c->eij[s[a]][s[b]];
    if (k == 2) return N0 * (N0 - 1) / 2 - E;
    int64_t SQ = 0, Tr = 0;
    for (int a = 0; a < ns; a++)
        for (int b = 0; b < ns; b++)
            for (int d = 0; d < ns; d++) SQ += c->Q[s[a]][s[b]][s[d]];
    for (int a = 0; a < ns; a++)
        for (int b = a; b < ns; b++)
            for (int d = b; d < ns; d++) Tr += c->tri[s[a]][s[b]][s[d]];
    int64_t Ch = (SQ - 2 * E) / 2;
    return N0 * (N0 - 1) * (N0 - 2) / 6 - E * (N0 - 2) + Ch - Tr;
}

/* I_4 by inclusion-exclusion over edge sets spanning <= 4 vertices:      */
/*   I4 = C(n,4) - E C(n-2,2) + P3 (n-3) + 2K2 - K3 (n-3) - P4 - K13       */
/*        + C4 + Paw - Diamond + K4                                        */
/* (subgraph, not induced, counts).  Bricks are axis-parallel boxes, so    */
/* pairwise-overlapping bricks share a cell (Helly); K4 is counted over    */
/* grid features (cells +, adjacent cell pairs -, 2x2 blocks +).           */
/*                                                                         */
/* Derivation: I_4 = sum over 4-sets T of prod over pairs in T of           */
/* (1 - [overlap]) = sum over edge sets F with |V(F)| <= 4 of (-1)^|F| times  */
/* C(n - |V(F)|, 4 - |V(F)|).  Grouping the edge sets by shape gives the    */
/* terms above:                                                             */
/*   1 edge: E;  2 edges: P3 (sharing a vertex) or 2K2 (disjoint);         */
/*   3 edges: K3, P4 (path), K13 (star);  4 edges: C4, Paw (triangle plus  */
/*   pendant);  5 edges: Diamond;  6 edges: K4.                            */
/*                                                                         */
/* DEBUG4 (-x): 1 = also compute every I_k (k >= 4) and class-table value  */
/* by brute force and exit(2) on a mismatch; 2 = brute force only.         */
static int DEBUG4;

/* binomial coefficients C(x,2), C(x,3), C(x,4); 0 for small x >= 0      */
static inline int64_t C2(int64_t x) { return x * (x - 1) / 2; }
static inline int64_t C3(int64_t x) { return x * (x - 1) * (x - 2) / 6; }
static inline int64_t C4n(int64_t x) { return x * (x - 1) * (x - 2) * (x - 3) / 24; }

/*
 * k4_features()
 *   Function: number of 4-cliques (4 pairwise-overlapping bricks) among
 *             the candidates vs[0..n-1], by the Helly feature count:
 *               K4 = sum over features f of sign(f) * C(cover(f), 4).
 *             The features of a brick are its cells (fg[0], +), pairs of
 *             horizontally (fg[1], -) or vertically (fg[2], -) adjacent
 *             cells, and 2x2 blocks (fg[3], +), each indexed by its lower
 *             left cell.  Pass 0 adds the bricks one at a time and uses
 *             C(m, 4) = sum over j < m of C(j, 3): each increment
 *             contributes C(count before, 3).  Pass 1 resets every touched
 *             counter to 0.
 *   Inputs:   c: candidates (pids); vs: candidate indices; n; fg: feature
 *             grid, all zero on entry.
 *   Outputs:  the clique count; fg is all zero again on return.
 *   Errors:   none.  Counters are uint8; at most 16 placements cover a
 *             given cell (8 H and 8 V).
 */
static int64_t k4_features(const Cand *c, const int *vs, int n, uint8_t (*fg)[GW][GW])
{
    int64_t K4 = 0;
    for (int pass = 0; pass < 2; pass++) {
        for (int i = 0; i < n; i++) {
            uint32_t p = c->p[vs[i]];
            int x = PX(p), y = PY(p), w = PO(p) ? 2 : 4, h = PO(p) ? 4 : 2;
            for (int yy = y; yy < y + h; yy++)
                for (int xx = x; xx < x + w; xx++) {
                    if (pass) { fg[0][yy][xx] = 0; if (xx + 1 < x + w) fg[1][yy][xx] = 0;
                                if (yy + 1 < y + h) fg[2][yy][xx] = 0;
                                if (xx + 1 < x + w && yy + 1 < y + h) fg[3][yy][xx] = 0; continue; }
                    K4 += C3(fg[0][yy][xx]++);
                    if (xx + 1 < x + w) K4 -= C3(fg[1][yy][xx]++);
                    if (yy + 1 < y + h) K4 -= C3(fg[2][yy][xx]++);
                    if (xx + 1 < x + w && yy + 1 < y + h) K4 += C3(fg[3][yy][xx]++);
                }
        }
    }
    return K4;
}

/*
 * ind4_formula()
 *   Function: I_4(X) by the formula above.  It gathers the vertex list and
 *             degrees, then for every edge (u, v), u < v, the number cc of
 *             common neighbours (triangles through the edge).  From these:
 *               E, K3 = sum cc / 3, t_u = triangles at u, Dia = sum C(cc,2),
 *               P4 = sum (d_u-1)(d_v-1) - 3 K3, Paw = sum t_u (d_u - 2).
 *             It also collects, for each u, the set H of vertices at
 *             distance exactly 2 (neighbours of neighbours, not adjacent,
 *             > u), because C4 = (1/2) * sum over all pairs {u, w} of
 *             C(codeg(u,w), 2) needs adjacent and non-adjacent pairs alike.
 *             K4 comes from k4_features().
 *   Inputs:   c with adj built; X: bitset of the vertex subset; fg: zeroed
 *             feature grid.
 *   Outputs:  I_4(X).
 *   Errors:   none.  Cost about E*W + (2-hop pairs)*W + 21n operations.
 *             It uses thread-local scratch arrays.
 */
static int64_t ind4_formula(const Cand *c, const uint64_t *X, uint8_t (*fg)[GW][GW])
{
    int W = c->W, vs[MAXC], n = 0;
    static __thread int d[MAXC];
    static __thread int64_t tv[MAXC];
    for (int w = 0; w < W; w++) {
        uint64_t x = X[w];
        while (x) { vs[n++] = w * 64 + __builtin_ctzll(x); x &= x - 1; }
    }
    if (n < 4) return 0;
    for (int i = 0; i < n; i++) {
        int u = vs[i], dd = 0;
        for (int w = 0; w < W; w++) dd += __builtin_popcountll(c->adj[u][w] & X[w]);
        d[u] = dd;
        tv[u] = 0;
    }
    int64_t E = 0, S3 = 0, P4a = 0, Dia = 0, C4p = 0;
    uint64_t H[CW];
    for (int i = 0; i < n; i++) {
        int u = vs[i];
        for (int w = 0; w < W; w++) H[w] = 0;
        for (int w = 0; w < W; w++) {
            uint64_t y = c->adj[u][w] & X[w];
            while (y) {
                int v = w * 64 + __builtin_ctzll(y);
                y &= y - 1;
                for (int q = 0; q < W; q++) H[q] |= c->adj[v][q];
                if (v < u) continue;
                int cc = 0;
                for (int q = 0; q < W; q++) cc += __builtin_popcountll(c->adj[u][q] & c->adj[v][q] & X[q]);
                E++;
                S3 += cc;
                tv[u] += cc;
                tv[v] += cc;
                Dia += C2(cc);
                C4p += C2(cc);
                P4a += (int64_t)(d[u] - 1) * (d[v] - 1);
            }
        }
        /* non-adjacent pairs u < w at distance 2 */
        for (int w = 0; w < W; w++) {
            uint64_t above = (w < u / 64) ? 0 : (w > u / 64) ? ~0ull : ~((2ull << (u & 63)) - 1);
            H[w] &= X[w] & ~c->adj[u][w] & above;
        }
        for (int w = 0; w < W; w++) {
            uint64_t y = H[w];
            while (y) {
                int v = w * 64 + __builtin_ctzll(y);
                y &= y - 1;
                int cc = 0;
                for (int q = 0; q < W; q++) cc += __builtin_popcountll(c->adj[u][q] & c->adj[v][q] & X[q]);
                C4p += C2(cc);
            }
        }
    }
    int64_t K3 = S3 / 3, P3 = 0, K13 = 0, Paw = 0;
    for (int i = 0; i < n; i++) {
        int u = vs[i];
        P3 += C2(d[u]);
        K13 += C3(d[u]);
        Paw += (tv[u] / 2) * (d[u] - 2);
    }
    int64_t D2 = C2(E) - P3, P4 = P4a - 3 * K3, C4 = C4p / 2;
    int64_t K4 = k4_features(c, vs, n, fg);
    return C4n(n) - E * C2(n - 2) + P3 * (n - 3) + D2 - K3 * (n - 3) - P4 - K13 + C4 + Paw - Dia + K4;
}

static int64_t ind_dfs(const Cand *c, const uint64_t *X, int k, int W);

/* I_k by one step of the recursion I_k = (1/k) sum_u I_{k-1}(X - N[u])   */
/* is not used; k >= 5 falls back to DFS over independent (k-4)-sets with */
/* the I_4 formula at the bottom.                                          */
/*
 * ind_big()
 *   Function: I_k(X) for k >= 4.  For k == 4 it applies ind4_formula().
 *             Otherwise it fixes the least element u of the k-set and
 *             recurses on the elements of X above u that do not overlap u.
 *   Inputs:   c with adj built; X: bitset; k >= 4; fg: zeroed feature grid.
 *   Outputs:  I_k(X).
 *   Errors:   none; the cost grows like |X|^(k-4) times the cost of I_4.
 */
static int64_t ind_big(const Cand *c, const uint64_t *X, int k, uint8_t (*fg)[GW][GW])
{
    if (k == 4) return ind4_formula(c, X, fg);
    int W = c->W;
    int64_t s = 0;
    uint64_t Y[CW];
    for (int w = 0; w < W; w++) {
        uint64_t x = X[w];
        while (x) {
            int b = __builtin_ctzll(x);
            x &= x - 1;
            int u = w * 64 + b;
            for (int v = 0; v < W; v++) Y[v] = (v < w ? 0 : v == w ? x : X[v]) & ~c->adj[u][v];
            s += ind_big(c, Y, k - 1, fg);
        }
    }
    return s;
}

/*
 * ind_dfs()
 *   Function: I_k(X) by plain depth-first search over the independent sets
 *             (least element first), counting the last level with popcount.
 *             It is the reference for the -x checks and the fallback when
 *             no feature grid is available.
 *   Inputs:   c with adj built; X: bitset; k >= 1; W: words in use.
 *   Outputs:  I_k(X).
 *   Errors:   none; the cost is about the number of independent (k-1)-sets.
 */
static int64_t ind_dfs(const Cand *c, const uint64_t *X, int k, int W)
{
    if (k == 1) {
        int64_t s = 0;
        for (int w = 0; w < W; w++) s += __builtin_popcountll(X[w]);
        return s;
    }
    int64_t s = 0;
    uint64_t Y[CW];
    for (int w = 0; w < W; w++) {
        uint64_t x = X[w];
        while (x) {
            int b = __builtin_ctzll(x);
            x &= x - 1;
            int u = w * 64 + b;
            for (int v = 0; v < W; v++) Y[v] = (v < w ? 0 : v == w ? x : X[v]) & ~c->adj[u][v];
            s += ind_dfs(c, Y, k - 1, W);
        }
    }
    return s;
}

/*
 * ind_count()
 *   Function: number of independent k-subsets of X (k bricks, no two
 *             overlapping), dispatching on k:
 *               k = 0: 1;  k = 1: |X|;  k = 2, 3: closed formulas over the
 *               bitsets (triangles listed as u < v < w);
 *               k >= 4: ind_big(), or ind_dfs() when fg is NULL or with
 *               -x 2.  With -x 1 both are computed and compared.
 *   Inputs:   c (adj built if k >= 2); X: bitset over c's candidates; k;
 *             fg: zeroed feature grid or NULL.
 *   Outputs:  the count.
 *   Errors:   with -x 1, exits with status 2 and "I_k mismatch" if the
 *             formula and the brute force disagree.
 */
static int64_t ind_count(const Cand *c, const uint64_t *X, int k, uint8_t (*fg)[GW][GW])
{
    int W = c->W;
    int64_t N0 = 0;
    for (int w = 0; w < W; w++) N0 += __builtin_popcountll(X[w]);
    if (k == 0) return 1;
    if (k == 1 || N0 < 2) return k == 1 ? N0 : 0;
    if (k > 3) {
        if (!fg || DEBUG4 == 2) return ind_dfs(c, X, k, W);
        int64_t v = ind_big(c, X, k, fg);
        if (DEBUG4) {
            int64_t v2 = ind_dfs(c, X, k, W);
            if (v != v2) { fprintf(stderr, "I_%d mismatch: formula %" PRId64 " dfs %" PRId64 "\n", k, v, v2); exit(2); }
        }
        return v;
    }
    int64_t E2 = 0, Ch = 0, Tr = 0;
    for (int w = 0; w < W; w++) {
        uint64_t x = X[w];
        while (x) {
            int b = __builtin_ctzll(x);
            x &= x - 1;
            int u = w * 64 + b;
            int64_t d = 0;
            for (int v = 0; v < W; v++) d += __builtin_popcountll(c->adj[u][v] & X[v]);
            E2 += d;
            Ch += d * (d - 1) / 2;
            if (k == 3) {
                /* triangles u < v < t */
                for (int v = w; v < W; v++) {
                    uint64_t y = c->adj[u][v] & X[v];
                    if (v == w) y &= ~((2ull << b) - 1);
                    while (y) {
                        int bb = __builtin_ctzll(y);
                        y &= y - 1;
                        int vv = v * 64 + bb;
                        for (int q = v; q < W; q++) {
                            uint64_t zz = c->adj[u][q] & c->adj[vv][q] & X[q];
                            if (q == v) zz &= ~((2ull << bb) - 1);
                            Tr += __builtin_popcountll(zz);
                        }
                    }
                }
            }
        }
    }
    int64_t E = E2 / 2;
    if (k == 2) return N0 * (N0 - 1) / 2 - E;
    return N0 * (N0 - 1) * (N0 - 2) / 6 - E * (N0 - 2) + Ch - Tr;
}

/* restricted-growth enumeration of partitions of r elements           */
/* A partition of {0..r-1} is stored as its restricted growth string:  */
/* a[i] = block of element i, with a[0] = 0 and                        */
/* a[i] <= 1 + max(a[0], ..., a[i-1]); each partition has exactly one  */
/* such string.  For every r <= MAXR all Bell(r) partitions are         */
/* precomputed:                                                         */
/*   PARTS[r][s][i]  block of element i in partition s;                 */
/*   PNB[r][s]       number of blocks b (mu(sigma, 1) = (-1)^(b-1)(b-1)!) */
/*   PCM[r][s][i]    bit mask of the elements in i's block.             */
#define MAXR 10
static int NPART[MAXR + 1];
static uint8_t (*PARTS[MAXR + 1])[MAXR];
static uint8_t *PNB[MAXR + 1];   /* number of blocks                    */
static uint16_t (*PCM[MAXR + 1])[MAXR];  /* mask of the block holding i */

/*
 * init_parts()
 *   Function: precomputes PARTS, PNB and PCM for r = 1 .. MAXR, listing
 *             the restricted growth strings in lexicographic order: the
 *             last position that may still grow is incremented and every
 *             later position is reset to 0.
 *   Inputs:   none.
 *   Outputs:  allocates and fills the tables (about 4 MB, mostly r = 10).
 *   Errors:   exits with "partition error" if a count differs from the Bell
 *             number.  malloc failures are not checked.
 */
static void init_parts(void)
{
    static const int bell[] = {1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975};
    for (int r = 1; r <= MAXR; r++) {
        int cap = bell[r];
        PARTS[r] = malloc(sizeof(uint8_t[MAXR]) * (size_t)cap);
        PNB[r] = malloc((size_t)cap);
        PCM[r] = malloc(sizeof(uint16_t[MAXR]) * (size_t)cap);
        NPART[r] = 0;
        uint8_t a[MAXR] = {0};
        for (;;) {
            int k = 0;
            for (int i = 0; i < r; i++) if (a[i] + 1 > k) k = a[i] + 1;
            PNB[r][NPART[r]] = (uint8_t)k;
            for (int i = 0; i < r; i++) {
                uint16_t bm = 0;
                for (int j = 0; j < r; j++) if (a[j] == a[i]) bm |= (uint16_t)(1u << j);
                PCM[r][NPART[r]][i] = bm;
            }
            memcpy(PARTS[r][NPART[r]++], a, MAXR);
            int i = r - 1;
            for (; i > 0; i--) {
                int mx = 0;
                for (int j = 0; j < i; j++) if (a[j] > mx) mx = a[j];
                if (a[i] <= mx) { a[i]++; for (int j = i + 1; j < r; j++) a[j] = 0; break; }
            }
            if (i == 0) break;
        }
        if (NPART[r] != cap) { fprintf(stderr, "partition error\n"); exit(1); }
    }
}

/*
 * xsigma()
 *   Function: X^l_sigma as a bitset: the union of the classes whose mask
 *             lies inside one block of sigma.  Only used when a layer has
 *             more than 64 classes (otherwise allowed_classes() and
 *             memo_ind() are used).
 *   Inputs:   c: candidate set; blk[i]: block of component i in sigma.
 *   Outputs:  X[0..W-1].
 *   Errors:   none.
 */
static void xsigma(const Cand *c, const uint8_t *blk, uint64_t *X)
{
    memset(X, 0, sizeof(uint64_t) * (size_t)c->W);
    for (int m = 0; m < c->nm; m++) {
        unsigned mk = c->mv[m];
        int b0 = -1, ok = 1;
        for (int i = 0; mk; i++, mk >>= 1)
            if (mk & 1) { if (b0 < 0) b0 = blk[i]; else if (blk[i] != b0) { ok = 0; break; } }
        if (ok) for (int w = 0; w < c->W; w++) X[w] |= c->mbits[m][w];
    }
}

/* bitmask over c's mask classes that are allowed by partition s of r */
/*
 * allowed_classes()
 *   Function: the classes allowed by partition s.  A mask is inside one
 *             block iff it is a subset of the block of its lowest set bit
 *             (PCM).
 *   Inputs:   c with nm <= 64; r: number of R-components; s: partition
 *             index.
 *   Outputs:  bit m set iff class m is allowed.
 *   Errors:   none checked; the caller ensures nm <= 64.
 */
static inline uint64_t allowed_classes(const Cand *c, int r, int s)
{
    const uint16_t *cm = PCM[r][s];
    uint64_t key = 0;
    for (int m = 0; m < c->nm; m++) {
        unsigned mk = c->mv[m];
        if ((mk & ~(unsigned)cm[__builtin_ctz(mk)]) == 0) key |= 1ull << m;
    }
    return key;
}

/* SAME01: profile of height 3 with A = {0, 2}.  Both aggregated layers   */
/* then see exactly layer 1, so their candidate sets coincide (up to z)   */
/* and the second one reuses cd[0].                                      */
/* USECT: class tables are built only when there are at least this many  */
/* partitions to evaluate (-c; 0 = always).                               */
static int SAME01;               /* h=3, A={0,2}: both layers see layer 1 */
static int USECT = 5;            /* class tables when >= this many sigmas  */


/*
 * memo_ind()
 *   Function: I_k of the union of the classes in `key`, cached per R-set.
 *             It uses an open-addressing hash table with linear probing.
 *             Entries from earlier R-sets are recognised by their
 *             generation stamp, so the table never needs clearing.  On a
 *             miss the value comes from the class tables (k <= ct_ok) or
 *             from ind_count() over the bitset of the union.
 *   Inputs:   t: thread state (fg for I_4); tab: the cache; gen: current
 *             generation; c: candidate set; key: allowed class set
 *             (nm <= 64); k; tag: which candidate set (layer) c is.
 *   Outputs:  the count.  It is also stored in tab unless all MEMO slots
 *             are already taken by this R-set; then it is just returned
 *             (exact, only uncached).
 *   Errors:   with -x 1, exits with status 2 and "class-table mismatch" if
 *             the class tables disagree with the bitset computation.
 */
static int64_t memo_ind(AT *t, MemoE *tab, uint32_t gen, const Cand *c, uint64_t key, int k, int tag)
{
    uint64_t hk = (key ^ ((uint64_t)tag << 40) ^ ((uint64_t)k << 48)) * 0x9E3779B97F4A7C15ull;
    unsigned h = (unsigned)(hk >> 54) & (MEMO - 1);
    /* probe at most MEMO slots: a full table must not loop forever */
    int free_slot = 0;
    for (int probes = 0; probes < MEMO; probes++) {
        MemoE *e = &tab[h];
        if (e->gen != gen) { free_slot = 1; break; }
        if (e->key == key && e->tag == tag && e->k == k) return e->val;
        h = (h + 1) & (MEMO - 1);
    }
    int64_t v;
    if (c->ct_ok >= k && !DEBUG4) {
        v = ind_class(c, key, k);
    } else {
        uint64_t X[CW] = {0};
        for (int m = 0; m < c->nm; m++)
            if (key >> m & 1) for (int w = 0; w < c->W; w++) X[w] |= c->mbits[m][w];
        v = ind_count(c, X, k, t->fg);
        if (DEBUG4 && c->ct_ok >= k && ind_class(c, key, k) != v) {
            fprintf(stderr, "class-table mismatch k=%d\n", k);
            exit(2);
        }
    }
    if (free_slot) tab[h] = (MemoE){key, v, gen, (uint16_t)tag, (uint16_t)k};
    (void)t;
    return v;
}

/*
 * core_count()
 *   Function: the number of buildings with the current R-set (header,
 *             section 7):
 *               1. components K_1..K_r of the R-set (depth-first search);
 *               2. candidate set of each aggregated layer; it is 0 if a
 *                  layer has fewer candidates than the bricks it needs;
 *               3. r == 1: only sigma = {all} is needed, so the answer is
 *                  the product of I_{z_l} over all candidates;
 *               4. otherwise build class tables where useful and evaluate
 *                    sum over sigma of mu(sigma,1) * prod over l in A of
 *                    I_{z_l}(X^l_sigma),
 *                  with mu = (-1)^(b-1) (b-1)! and memoized I values.
 *   Inputs:   t: thread state with the complete R-set in t->g.P; plan
 *             globals (CAP, NAGG, AGL, SAME01, USECT).
 *   Outputs:  the count, as a signed 128-bit integer.  It is >= 0 when the
 *             method is applied correctly; intermediate partial sums may be
 *             negative.
 *   Errors:   exits with "too many components" if r > MAXR; see also
 *             collect() and memo_ind().
 */
static __int128 core_count(AT *t)
{
    Geo *g = &t->g;
    int m = g->np, comp[MAXN + 4], r = 0, st[MAXN + 4];
    for (int i = 0; i < m; i++) comp[i] = -1;
    for (int i = 0; i < m; i++) {
        if (comp[i] >= 0) continue;
        int sp = 0;
        st[sp++] = i;
        comp[i] = r;
        while (sp) {
            int u = st[--sp];
            for (int v = 0; v < m; v++)
                if (comp[v] < 0 && adjacent(g->P[u], g->P[v])) { comp[v] = r; st[sp++] = v; }
        }
        r++;
    }
    if (r > MAXR) { fprintf(stderr, "too many components\n"); exit(1); }
    /* candidate sets; with SAME01 cd[0] serves both layers, so it is     */
    /* built for the larger of the two k (adjacency needed if either >= 2) */
    for (int a = 0; a < NAGG; a++) {
        if (a == 1 && SAME01) continue;
        collect(t, &t->cd[a], AGL[a], CAP[AGL[a]] > CAP[AGL[SAME01 ? 1 : a]] ? CAP[AGL[a]] : CAP[AGL[SAME01 ? 1 : a]], comp);
        if (t->cd[a].n < CAP[AGL[a]]) return 0;
    }
    if (SAME01 && t->cd[0].n < CAP[AGL[1]]) return 0;
    if (r == 1) {   /* connected R-set: the only partition is {all} */
        __int128 prod = 1;
        for (int a = 0; a < NAGG && prod; a++) {
            const Cand *c = &t->cd[(a == 1 && SAME01) ? 0 : a];
            uint64_t X[CW];
            for (int w = 0; w < c->W; w++) X[w] = ~0ull;
            if (c->n & 63) X[c->W - 1] = (1ull << (c->n & 63)) - 1;
            prod *= ind_count(c, X, CAP[AGL[a]], t->fg);
        }
        return prod;
    }
    for (int a = 0; a < NAGG; a++) {
        if (a == 1 && SAME01) continue;
        int kk = CAP[AGL[a]];
        if (a == 0 && SAME01 && CAP[AGL[1]] > kk) kk = CAP[AGL[1]];
        if (kk <= 3 && (!USECT || NPART[r] >= USECT)) class_tables(&t->cd[a], kk);
    }
    uint32_t gen = ++t->memogen;   /* new cache generation for this R-set */
    if (!gen) { memset(t->memo, 0, sizeof t->memo); gen = t->memogen = 1; }
    /* the Moebius sum over all partitions sigma of the r components */
    __int128 tot = 0;
    for (int s = 0; s < NPART[r]; s++) {
        int k = PNB[r][s];
        __int128 prod = 1;
        for (int a = 0; a < NAGG && prod; a++) {
            int ca = (a == 1 && SAME01) ? 0 : a;
            const Cand *c = &t->cd[ca];
            int kk = CAP[AGL[a]];
            if (c->nm <= 64) {
                uint64_t key = allowed_classes(c, r, s);
                prod *= memo_ind(t, t->memo, gen, c, key, kk, ca);
            } else {
                uint64_t X[CW];
                xsigma(c, PARTS[r][s], X);
                prod *= ind_count(c, X, kk, t->fg);
            }
        }
        if (!prod) continue;
        for (int i = 2; i < k; i++) prod *= i;   /* |mu| = (k-1)!, k blocks */
        tot += (k & 1) ? prod : -prod;         /* sign (-1)^(k-1)         */
    }
    return tot;
}

/* D4 symmetry of the plane: Count(R) is invariant under rotations and   */
/* reflections of the R-bricks, so only the least image is evaluated and */
/* weighted by the orbit size.  Returns 0 if P is not the least image.   */
static const int D4M[8][4] = {
    {1, 0, 0, 1}, {0, -1, 1, 0}, {-1, 0, 0, -1}, {0, 1, -1, 0},
    {-1, 0, 0, 1}, {1, 0, 0, -1}, {0, 1, 1, 0}, {0, -1, -1, 0}};

/*
 * isort()
 *   Function: insertion sort of a few pids, ascending.
 *   Inputs:   a[0..m-1], m small (at most about 11).
 *   Outputs:  a sorted in place.
 *   Errors:   none.
 */
static inline void isort(uint32_t *a, int m)
{
    for (int i = 1; i < m; i++) {
        uint32_t v = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; j--; }
        a[j + 1] = v;
    }
}

/*
 * canon_weight()
 *   Function: the D4 test for an R-set.  Each of the 8 linear maps D4M
 *             (rotations by 0/90/180/270 degrees, reflections in x, y and
 *             both diagonals) is applied to the box corners of every brick.
 *             The image's least brick is translated back to (OFF, OFF),
 *             and the sorted image is compared with the sorted original.
 *             The original was generated with its least brick at (OFF, OFF)
 *             as well, so equal lists mean "same R-set up to translation".
 *   Inputs:   P[0..m-1]: the R-set (any order).
 *   Outputs:  0 if some image is smaller (another member of the orbit is
 *             the one evaluated); otherwise 8 / (number of maps fixing it),
 *             i.e. 1, 2, 4 or 8, the size of its D4 orbit.
 *   Errors:   none (the identity always fixes P, so no division by 0).
 */
static int canon_weight(const uint32_t *P, int m)
{
    uint32_t base[MAXN + 4], img[MAXN + 4];
    int bx[MAXN + 4], by[MAXN + 4], bo[MAXN + 4], bz[MAXN + 4];
    memcpy(base, P, sizeof(uint32_t) * (size_t)m);
    isort(base, m);
    int stab = 0;
    for (int gi = 0; gi < 8; gi++) {
        const int *M = D4M[gi];
        int zmin = 99, ymin = 1 << 20, xmin = 1 << 20, omin = 9;
        for (int i = 0; i < m; i++) {
            uint32_t p = base[i];
            int x = PX(p) - OFF, y = PY(p) - OFF, w = PO(p) ? 2 : 4, h = PO(p) ? 4 : 2;
            /* images of the box corners (x, y) and (x+w, y+h) */
            int ax = M[0] * x + M[1] * y, ay = M[2] * x + M[3] * y;
            int cx = M[0] * (x + w) + M[1] * (y + h), cy = M[2] * (x + w) + M[3] * (y + h);
            int nx = ax < cx ? ax : cx, ny = ay < cy ? ay : cy, nw = ax < cx ? cx - ax : ax - cx;
            bx[i] = nx; by[i] = ny; bo[i] = (nw == 2); bz[i] = PZ(p);
            if (bz[i] < zmin || (bz[i] == zmin && (ny < ymin || (ny == ymin && (nx < xmin || (nx == xmin && bo[i] < omin)))))) {
                zmin = bz[i]; ymin = ny; xmin = nx; omin = bo[i];
            }
        }
        for (int i = 0; i < m; i++) img[i] = MKP(bz[i], bx[i] - xmin + OFF, by[i] - ymin + OFF, bo[i]);
        isort(img, m);
        int c = 0;
        for (int i = 0; i < m && !c; i++) c = (img[i] < base[i]) ? -1 : (img[i] > base[i]) ? 1 : 0;
        if (c < 0) return 0;
        if (c == 0) stab++;
    }
    return 8 / stab;
}

static int USE_D4 = 1;            /* -n turns the D4 reduction off      */

/*
 * push()
 *   Function: Redelmeier append for the R-set search: q is added to the
 *             untried list (and marked) if q > root and it is not marked.
 *   Inputs:   t; q: pid; k: current end of the untried list.
 *   Outputs:  may append q and advance *k.
 *   Errors:   none (stack bound in the Geo comment).
 */
static inline void push(AT *t, uint32_t q, int *k)
{
    Geo *g = &t->g;
    if (q > t->root && !g->mark[q]) { g->mark[q] = 1; g->G[(*k)++] = q; }
}

static int COUNTTASKS;            /* pre-pass: only count split nodes    */
static double SAMPLE = 1.0;       /* fraction of tasks processed          */
static uint64_t SAMPLE_SEED = 12345;

/*
 * sampled()
 *   Function: whether task number `task` belongs to the sample (-p).  The
 *             task index plus a fixed seed goes through a splitmix64-style
 *             mixer and is compared with SAMPLE.  Every run, on any
 *             machine, therefore picks the same tasks.
 *   Inputs:   task index (DFS order of the split nodes).
 *   Outputs:  1 if the task is processed.
 *   Errors:   none.
 */
static inline int sampled(uint64_t task)
{
    if (SAMPLE >= 1.0) return 1;
    uint64_t x = (task + SAMPLE_SEED) * 0x9E3779B97F4A7C15ull;
    x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29;
    return (double)(x >> 11) * (1.0 / 9007199254740992.0) < SAMPLE;
}

/*
 * arec()
 *   Function: one level of the R-set search (header, section 7).  Same
 *             Redelmeier scheme as frec(), on the augmented graph.  The
 *             neighbours of an R-brick c in layer z are:
 *               - real: the 46 overlapping placements in z+1 and z-1 when
 *                 those are R-layers;
 *               - virtual, same layer (VJD): when z+1 or z-1 is aggregated;
 *               - virtual, two layers apart (VJX): in z+2 (z-2) when z+1
 *                 (z-1) is aggregated and z+2 (z-2) is an R-layer.
 *             When the R-set is complete (MR bricks) it is evaluated:
 *             canon_weight() keeps one representative per D4 orbit, and
 *             core_count() * weight is added to t->F.
 *             At the split depth it also handles the pre-pass (count
 *             tasks only) and the sample selection.
 *   Inputs:   t; [s, e): untried range.  Plan globals (CAP, AMASK, RMASK,
 *             MR, SPLIT, COUNTTASKS, SAMPLE, USE_D4).
 *   Outputs:  adds to t->F and t->cores.
 *   Errors:   none besides those of core_count().
 */
static void arec(AT *t, int s, int e)
{
    Geo *g = &t->g;
    for (int i = s; i < e; i++) {
        uint32_t c = g->G[i];
        int z = PZ(c);
        if (g->cnt[z] >= CAP[z] || conflict(g, c)) continue;
        int np1 = g->np + 1;
        if (np1 == SPLIT && SPLIT > 0) {
            if (COUNTTASKS) { t->task++; continue; }   /* pre-pass only */
            int my = t->task++;
            if (my != t->claimed) continue;
            t->claimed = atomic_fetch_add(&next_task, 1);
            if (!sampled((uint64_t)my)) continue;
        }
        toggle(g, c);
        g->cnt[z]++;
        g->P[g->np++] = c;
        if (np1 == MR) {
            int wgt = USE_D4 ? canon_weight(g->P, g->np) : 1;
            if (wgt) {
                t->F += (u128)(core_count(t) * wgt);
                t->cores++;
            }
        } else {
            int k = e, o = PO(c);
            const int32_t *ov = OVD[o];
            if (inR(z + 1)) for (int j = 0; j < 46; j++) push(t, (uint32_t)((int32_t)c + (1 << 15) + ov[j]), &k);
            if (inR(z - 1)) for (int j = 0; j < 46; j++) push(t, (uint32_t)((int32_t)c - (1 << 15) + ov[j]), &k);
            if (inA(z + 1) || inA(z - 1))
                for (int j = 0; j < NVJ[o]; j++) push(t, (uint32_t)((int32_t)c + VJD[o][j]), &k);
            if (inA(z + 1) && inR(z + 2))
                for (int j = 0; j < NVX[o]; j++) push(t, (uint32_t)((int32_t)c + (2 << 15) + VJX[o][j]), &k);
            if (inA(z - 1) && inR(z - 2))
                for (int j = 0; j < NVX[o]; j++) push(t, (uint32_t)((int32_t)c - (2 << 15) + VJX[o][j]), &k);
            arec(t, i + 1, k);
            for (int j = e; j < k; j++) g->mark[g->G[j]] = 0;
        }
        g->np--;
        g->cnt[z]--;
        toggle(g, c);
    }
}

/*
 * athread()
 *   Function: worker thread of the aggregated method.  It runs arec() from
 *             the root at (ZR, OFF, OFF), horizontal then vertical.  The
 *             least R-brick always lies in ZR, because every R-layer is
 *             non-empty in a complete R-set.  With COUNTTASKS it is also
 *             the single-threaded pre-pass that only counts the split
 *             nodes.
 *   Inputs:   arg: this thread's AT (zeroed by calloc).
 *   Outputs:  t->F, t->cores (t->task in the pre-pass).
 *   Errors:   none.
 */
static void *athread(void *arg)
{
    AT *t = arg;
    if (!COUNTTASKS) t->claimed = atomic_fetch_add(&next_task, 1);
    for (int o = 0; o < 2; o++) {
        Geo *g = &t->g;
        memset(g->mark, 0, sizeof g->mark);
        memset(g->occ, 0, sizeof g->occ);
        memset(g->cnt, 0, sizeof g->cnt);
        uint32_t r = MKP(ZR, OFF, OFF, o);
        t->root = r;
        g->np = 0;
        g->G[0] = r;
        g->mark[r] = 1;
        arec(t, 0, 1);
        g->mark[r] = 0;
    }
    return NULL;
}

/* Progress reporting for -r.  TOTTASKS comes from the pre-pass (0 if   */
/* skipped); the claimed-task counter next_task measures progress.  ETAs */
/* are only indicative, because tasks early in DFS order are larger.     */
static int SPLITMAX = 4;          /* -d                                  */
static int NOPREPASS;             /* -q                                  */
static int64_t TOTTASKS;
static volatile int progress_stop;
static double progress_t0;

/*
 * progress_thread()
 *   Function: prints a status line to stderr: every 2 s, rewritten in
 *             place, on a terminal; one line per minute otherwise.
 *   Inputs:   none (globals above).
 *   Outputs:  stderr text; returns when progress_stop is set.
 *   Errors:   none.
 */
static void *progress_thread(void *arg)
{
    (void)arg;
    int tty = isatty(2);
    double last = now();
    while (!progress_stop) {
        usleep(200000);
        double tn = now();
        if (tn - last < (tty ? 2.0 : 60.0)) continue;
        last = tn;
        int64_t done = atomic_load(&next_task);
        if (!TOTTASKS) {
            fprintf(stderr, "%s  tasks claimed %" PRId64 "  elapsed %.0fs   %s", tty ? "\r" : "", done,
                    tn - progress_t0, tty ? "" : "\n");
            fflush(stderr);
            continue;
        }
        if (done > TOTTASKS) done = TOTTASKS;
        double el = tn - progress_t0, fr = TOTTASKS ? (double)done / (double)TOTTASKS : 0;
        double eta = fr > 0 ? el / fr - el : 0;
        fprintf(stderr, "%s  tasks %" PRId64 "/%" PRId64 " (%.2f%%)  elapsed %.0fs  eta %.0fs   %s",
                tty ? "\r" : "", done, TOTTASKS, 100 * fr, el, eta, tty ? "" : "\n");
        fflush(stderr);
    }
    if (tty) fprintf(stderr, "\r%80s\r", "");
    return NULL;
}

/*
 * choose_agg()
 *   Function: the default plan.  Over all non-empty proper sets of pairwise
 *             non-adjacent layers holding at most 4 bricks each (I_4 has a
 *             closed formula; I_5 and up would need the slow DFS), it picks
 *             the one leaving the fewest bricks to enumerate.  Ties go to
 *             the smaller largest aggregated layer.
 *   Inputs:   prof[0..h-1].
 *   Outputs:  bit mask of the aggregated layers, or 0 if no layer qualifies
 *             (then every layer is enumerated, and the method degenerates to
 *             a plain search: Count = 1 for each connected R-set).
 *   Errors:   none.
 */
static int choose_agg(const int *prof, int h)
{
    int bestm = 0, bestr = 1 << 20, bestk = 99;
    for (int m = 1; m < (1 << h) - 1; m++) {
        if (m & (m >> 1)) continue;
        int r = 0, k = 0, ok = 1;
        for (int l = 0; l < h; l++) {
            if (m >> l & 1) { if (prof[l] > 4) ok = 0; if (prof[l] > k) k = prof[l]; }
            else r += prof[l];
        }
        if (ok && (r < bestr || (r == bestr && k < bestk))) { bestr = r; bestk = k; bestm = m; }
    }
    return bestm;
}

/*
 * count_ref_agg()
 *   Function: F for one fat profile by layer aggregation.  It sets the plan
 *             globals and picks the split depth (min(SPLITMAX, MR-1); no
 *             threads below 2).  Unless sampling or -q, it runs the
 *             single-threaded pre-pass that counts the tasks.  Then it
 *             starts the progress thread and NTHR workers and sums their
 *             results.
 *   Inputs:   prof[0..h-1]; amask: aggregated layers (pairwise non-
 *             adjacent, not all); ncores: optional output.  Globals NTHR,
 *             SPLITMAX, SAMPLE, NOPREPASS.
 *   Outputs:  F, which is only the sampled part when SAMPLE < 1; *ncores =
 *             number of R-sets evaluated (D4 representatives).
 *   Errors:   none expected: NTHR is in 1..MAXTHR (checked by main).  A
 *             calloc failure is not checked (each AT is a few MB).  N is not
 *             set here, because only the plain and symmetric searches use
 *             it.
 */
static u128 count_ref_agg(const int *prof, int h, int amask, uint64_t *ncores)
{
    for (int z = 0; z < NZ; z++) CAP[z] = z < h ? prof[z] : 0;
    HMAX = h;
    AMASK = amask;
    RMASK = ((1 << h) - 1) & ~amask;
    SAME01 = (h == 3 && amask == 5);
    NAGG = 0;
    MR = 0;
    ZR = -1;
    for (int l = 0; l < h; l++) {
        if (amask >> l & 1) AGL[NAGG++] = l;
        else { MR += prof[l]; if (ZR < 0) ZR = l; }
    }
    SPLIT = MR - 1 > SPLITMAX ? SPLITMAX : MR - 1;
    if (SPLIT < 2) SPLIT = 0;
    int nt = SPLIT ? NTHR : 1;
    AT *ts = calloc((size_t)nt, sizeof(AT));
    TOTTASKS = 0;
    if (SPLIT && SAMPLE >= 1.0 && !NOPREPASS) {  /* pre-pass: number of tasks, for progress */
        COUNTTASKS = 1;
        athread(&ts[0]);
        TOTTASKS = ts[0].task;
        ts[0].task = 0;
        COUNTTASKS = 0;
    }
    atomic_store(&next_task, 0);
    pthread_t th[MAXTHR], pth;
    progress_stop = 0;
    progress_t0 = now();
    if (SPLIT) pthread_create(&pth, NULL, progress_thread, NULL);
    for (int i = 0; i < nt; i++) pthread_create(&th[i], NULL, athread, &ts[i]);
    for (int i = 0; i < nt; i++) pthread_join(th[i], NULL);
    if (SPLIT) { progress_stop = 1; pthread_join(pth, NULL); }
    u128 F = 0;
    uint64_t nc = 0;
    for (int i = 0; i < nt; i++) { F += ts[i].F; nc += ts[i].cores; }
    free(ts);
    if (ncores) *ncores = nc;
    return F;
}

/*
 * count_ref_base()
 *   Function: F for one profile by the plain search restricted to it
 *             (REFMODE: CAP = the profile), as an independent check of
 *             count_ref_agg().
 *   Inputs:   prof[0..h-1], total n <= MAXN-1.
 *   Outputs:  F of the profile.
 *   Errors:   as count_fixed().  Practical only for small counts (about
 *             1e8 buildings/s/core).
 */
static u128 count_ref_base(const int *prof, int h)
{
    int n = 0;
    for (int z = 0; z < h; z++) n += prof[z];
    N = n;
    for (int z = 0; z < NZ; z++) CAP[z] = z < h ? prof[z] : 0;
    HMAX = h;
    REFMODE = 1;
    static uint64_t F[MAXPROF];
    count_fixed(F);
    REFMODE = 0;
    return F[prof_index(prof, h)];
}

/*
 * parse_prof()
 *   Function: parses a profile given as digits, "353" -> <3 5 3>.
 *   Inputs:   s; prof with room for NZ ints.
 *   Outputs:  the height, or 0 if s has a character outside '1'..'9' or
 *             more than NZ layers.
 *   Errors:   reported through the 0 return.
 */
static int parse_prof(const char *s, int *prof)
{
    int h = 0;
    for (; *s; s++) {
        if (*s < '1' || *s > '9' || h >= NZ) return 0;
        prof[h++] = *s - '0';
    }
    return h;
}

/*
 * ref_mode()
 *   Function: the -r / -b modes.  It checks the profile (height >= 2,
 *             fat), picks the aggregated layers (choose_agg() or -a),
 *             computes F (aggregation or plain search), then S180 and S90
 *             for that profile with the orbit search, and prints
 *               <profile> count (sym)  F= S180= S90= agg={..} Rsets= [times]
 *             With -p it prints the sampled estimate and returns before
 *             the symmetric counts.
 *   Inputs:   ps: profile digits; base: 1 = plain search (-b); aspec:
 *             layer digits from -a, or NULL for the default plan.
 *   Outputs:  one line on stdout.
 *   Errors:   exits with "bad profile", "profile has a bottleneck; use
 *             composition", or "aggregated layers must be non-adjacent
 *             layers of the profile".  "NON-INTEGRAL" in the output means
 *             F + S180 + 2*S90 is not divisible by 4, i.e. something is
 *             wrong.
 */
static void ref_mode(const char *ps, int base, const char *aspec)
{
    int prof[NZ], h = parse_prof(ps, prof), n = 0;
    if (h < 2) { fprintf(stderr, "bad profile\n"); exit(1); }
    for (int i = 0; i < h; i++) n += prof[i];
    if (!prof_fat(prof, h)) { fprintf(stderr, "profile has a bottleneck; use composition\n"); exit(1); }
    int amask = choose_agg(prof, h);
    if (aspec) {
        amask = 0;
        for (const char *p = aspec; *p; p++) if (*p >= '0' && *p <= '9') amask |= 1 << (*p - '0');
        if ((amask & (amask >> 1)) || amask >= (1 << h) || amask == (1 << h) - 1) {
            fprintf(stderr, "aggregated layers must be non-adjacent layers of the profile\n");
            exit(1);
        }
    }
    double t0 = now();
    uint64_t ncores = 0;
    u128 F = base ? count_ref_base(prof, h) : count_ref_agg(prof, h, amask, &ncores);
    double t1 = now();
    if (SAMPLE < 1.0) {
        double Fe = (double)F / SAMPLE;
        char fr[48];
        u128str(F, fr);
        printf("<%s> SAMPLE p=%g: F~%.4e  classes~%.4e  Rsets=%" PRIu64 "  time %.1fs -> full ~%.0fs (%.2f h) on %d threads  [raw %s]\n",
               ps, SAMPLE, Fe, Fe / 4, ncores, t1 - t0, (t1 - t0) / SAMPLE, (t1 - t0) / SAMPLE / 3600, NTHR, fr);
        return;
    }
    N = n;
    for (int z = 0; z < NZ; z++) CAP[z] = z < h ? prof[z] : 0;
    HMAX = h;
    static uint64_t S2[MAXPROF], S4[MAXPROF];
    count_sym(2, S2);
    count_sym(4, S4);
    double t2 = now();
    int idx = prof_index(prof, h);
    u128 s180 = S2[idx], s90 = S4[idx];
    char b1[48], b2[48], b3[48], as[NZ + 1];
    int na = 0;
    for (int l = 0; l < h; l++) if (amask >> l & 1) as[na++] = (char)('0' + l);
    as[na] = 0;
    u128str((F + s180 + 2 * s90) / 4, b1);
    u128str((s180 + s90) / 2, b2);
    u128str(F, b3);
    printf("<%s> %s (%s)   F=%s S180=%" PRIu64 " S90=%" PRIu64 "%s  agg={%s} Rsets=%" PRIu64 "  [F %.1fs, S %.1fs]\n",
           ps, b1, b2, b3, (uint64_t)s180, (uint64_t)s90,
           ((F + s180 + 2 * s90) % 4) ? " NON-INTEGRAL" : "", base ? "-" : as, ncores, t1 - t0, t2 - t1);
    fflush(stdout);
}

/* ------------------------------------------------------------------ */
/* composition over profiles                                            */

/* Burnside ingredients of one profile; TAB[n][index] for all n <= MAXN. */
/* known = 0 if a composed profile depends on missing values.           */
typedef struct { u128 F, S180, S90; int known; } PV;
static PV TAB[MAXN + 1][MAXPROF];

/*
 * u128str()
 *   Function: decimal string of an unsigned 128-bit integer (printf has
 *             no format for it).
 *   Inputs:   v; buf with room for 40 characters.
 *   Outputs:  NUL-terminated digits in buf.
 *   Errors:   none.
 */
static void u128str(u128 v, char *buf)
{
    char tmp[48];
    int k = 0;
    if (!v) { strcpy(buf, "0"); return; }
    while (v) { tmp[k++] = (char)('0' + (int)(v % 10)); v /= 10; }
    for (int i = 0; i < k; i++) buf[i] = tmp[k - 1 - i];
    buf[k] = 0;
}

/*
 * lookup()
 *   Function: the TAB entry of a profile.
 *   Inputs:   cnt[0..h-1], total <= MAXN.
 *   Outputs:  pointer into TAB.
 *   Errors:   none checked.
 */
static PV *lookup(const int *cnt, int h)
{
    int n = 0;
    for (int i = 0; i < h; i++) n += cnt[i];
    return &TAB[n][prof_index(cnt, h)];
}

/*
 * compose()
 *   Function: fills every bottleneck profile of size n from smaller ones
 *             (header, section 3), splitting at the first interior layer of
 *             size 1: lo = cnt[0..b] and hi = cnt[b..h-1] share the brick b.
 *   Inputs:   n; TAB must hold all profiles of sizes < n (fat or composed).
 *   Outputs:  TAB[n][idx] for every non-fat idx; known = 0 if a part is
 *             unknown.
 *   Errors:   none.
 */
static void compose(int n)
{
    int cnt[MAXN + 1];
    for (int idx = 0; idx < (1 << (n - 1)); idx++) {
        int h = prof_decode(idx, n, cnt);
        if (prof_fat(cnt, h)) continue;
        int b = 1;
        while (cnt[b] != 1) b++;
        PV *lo = lookup(cnt, b + 1), *hi = lookup(cnt + b, h - b);
        PV *v = &TAB[n][idx];
        if (!lo->known || !hi->known) { v->known = 0; continue; }
        v->F = lo->F * hi->F / 2;
        v->S180 = lo->S180 * hi->S180 / 2;
        v->S90 = 0;
        v->known = 1;
    }
}

/*
 * run_all()
 *   Function: the plain mode `./lego N` and the self test.  For each
 *             n = 1..nmax it computes:
 *               - F of all fat profiles (count_fixed);
 *               - S180 and S90 of all profiles (count_sym, including the
 *                 bottleneck profiles, as a check);
 *               - the bottleneck profiles by composition, warning if their
 *                 composed S180 differs from the enumerated one;
 *               - a(n) and a180(n).
 *             With verbose it then prints the table of nmax ("L1" marks
 *             composed profiles).
 *   Inputs:   nmax in 1..MAXN-1 (checked by main); verbose.
 *   Outputs:  stdout lines "a(n) = ... a180(n) = ...", warnings on stderr;
 *             TAB filled.
 *   Errors:   "(NON-INTEGRAL!)" or "WARNING: S180 composition mismatch"
 *             signal an internal error.  The run time grows about 100x
 *             per n (n = 8 takes hours).
 */
static void run_all(int nmax, int verbose)
{
    for (int n = 1; n <= nmax; n++) {
        N = n;
        for (int z = 0; z < NZ; z++) CAP[z] = n;
        HMAX = n;
        REFMODE = 0;
        static uint64_t F[MAXPROF], S2[MAXPROF], S4[MAXPROF];
        double t0 = now();
        count_fixed(F);
        double t1 = now();
        count_sym(2, S2);
        count_sym(4, S4);
        double t2 = now();
        int cnt[MAXN + 1];
        for (int idx = 0; idx < (1 << (n - 1)); idx++) {
            int h = prof_decode(idx, n, cnt);
            PV *v = &TAB[n][idx];
            if (prof_fat(cnt, h)) {
                v->F = F[idx]; v->S180 = S2[idx]; v->S90 = S4[idx]; v->known = 1;
            }
        }
        compose(n);
        /* cross-check symmetric counts of bottleneck profiles */
        for (int idx = 0; idx < (1 << (n - 1)); idx++) {
            int h = prof_decode(idx, n, cnt);
            if (prof_fat(cnt, h)) continue;
            if (TAB[n][idx].S180 != S2[idx] || S4[idx]) {
                char ps[16];
                prof_str(cnt, h, ps);
                fprintf(stderr, "WARNING: S180 composition mismatch at <%s>\n", ps);
            }
        }
        u128 tot = 0, tots = 0;
        int ok = 1;
        for (int idx = 0; idx < (1 << (n - 1)); idx++) {
            PV *v = &TAB[n][idx];
            u128 num = v->F + v->S180 + 2 * v->S90;
            if (num % 4 || (v->S180 + v->S90) % 2) ok = 0;
            tot += num / 4;
            tots += (v->S180 + v->S90) / 2;
        }
        char b1[48], b2[48];
        u128str(tot, b1);
        u128str(tots, b2);
        printf("a(%d) = %s   a180(%d) = %s%s   [F %.1fs, S %.1fs]\n", n, b1, n, b2,
               ok ? "" : "  (NON-INTEGRAL!)", t1 - t0, t2 - t1);
        if (verbose && n == nmax) {
            for (int idx = 0; idx < (1 << (n - 1)); idx++) {
                int h = prof_decode(idx, n, cnt);
                PV *v = &TAB[n][idx];
                char ps[16];
                prof_str(cnt, h, ps);
                u128str((v->F + v->S180 + 2 * v->S90) / 4, b1);
                u128str((v->S180 + v->S90) / 2, b2);
                printf("  <%s> %s (%s)%s\n", ps, b1, b2, prof_fat(cnt, h) ? "" : " L1");
            }
        }
        fflush(stdout);
    }
}

/* Symmetric counts only, all profiles of size 1..nmax:                   */
/*   A123829(n) = classes of 180-degree-symmetric buildings                */
/*              = sum over profiles of (S180 + S90) / 2                    */
/*   A123831(n-1) = 180-symmetric buildings on a fixed base brick that is  */
/*              alone in the bottom layer = sum over profiles <1 ...> of   */
/*              S180 / 2 (the rotation must fix the base brick)            */
/*
 * sym_all()
 *   Function: the -S / -SV modes: the counts above for n = 1..nmax, using
 *             only the orbit search (no F is needed, which is why these
 *             reach n = 11 while A112389 does not).
 *   Inputs:   nmax in 1..MAXN (checked by main); verbose (-SV) prints
 *             S180/S90 of every profile with a nonzero count.
 *   Outputs:  stdout lines "n=.. A123829=.. S90total=.. A123831(n-1)=..".
 *   Errors:   "PARITY-FAIL" if a halved quantity is odd, which signals an
 *             internal error.  The run time grows about 13x per n (n = 11
 *             takes about 45 min on 10 threads).
 */
static void sym_all(int nmax, int verbose)
{
    for (int n = 1; n <= nmax; n++) {
        N = n;
        for (int z = 0; z < NZ; z++) CAP[z] = n;
        HMAX = n;
        static uint64_t S2[MAXPROF], S4[MAXPROF];
        double t0 = now();
        count_sym(2, S2);
        count_sym(4, S4);
        double t1 = now();
        u128 a180 = 0, s90 = 0, base = 0;
        int ok = 1, cnt[MAXN + 1];
        for (int idx = 0; idx < (1 << (n - 1)); idx++) {
            if ((S2[idx] + S4[idx]) & 1) ok = 0;
            a180 += (S2[idx] + S4[idx]) / 2;
            s90 += S4[idx];
            prof_decode(idx, n, cnt);
            if (cnt[0] == 1) {
                if (S2[idx] & 1) ok = 0;
                base += S2[idx] / 2;
            }
        }
        char b1[48], b2[48], b3[48];
        u128str(a180, b1);
        u128str(s90, b2);
        u128str(base, b3);
        printf("n=%d  A123829=%s  S90total=%s  A123831(%d)=%s%s  [%.1fs]\n", n, b1, b2, n - 1,
               n >= 2 ? b3 : "-", ok ? "" : "  PARITY-FAIL", t1 - t0);
        if (verbose) {
            for (int idx = 0; idx < (1 << (n - 1)); idx++) {
                if (!S2[idx] && !S4[idx]) continue;
                int h = prof_decode(idx, n, cnt);
                char ps[16];
                prof_str(cnt, h, ps);
                printf("  <%s> S180=%" PRIu64 " S90=%" PRIu64 "\n", ps, S2[idx], S4[idx]);
            }
        }
        fflush(stdout);
    }
}

/*
 * selftest()
 *   Function: runs the plain mode for n <= 7 and compares a(n) with
 *             A112389 and a180(n) with A123829.
 *   Inputs:   none (NTHR from the command line).
 *   Outputs:  one line per n and a final verdict.  Returns 0 if all match.
 *   Errors:   returns 1 on any mismatch.  Takes a few minutes.
 */
static int selftest(void)
{
    static const char *A[] = {"", "1", "24", "1560", "119580", "10166403", "915103765", "85747377755"};
    static const char *S[] = {"", "1", "2", "44", "185", "3276", "15682", "282377"};
    int bad = 0, cnt[MAXN + 1];
    run_all(7, 0);
    for (int n = 1; n <= 7; n++) {
        u128 tot = 0, tots = 0;
        for (int idx = 0; idx < (1 << (n - 1)); idx++) {
            PV *v = &TAB[n][idx];
            tot += (v->F + v->S180 + 2 * v->S90) / 4;
            tots += (v->S180 + v->S90) / 2;
        }
        char b1[48], b2[48];
        u128str(tot, b1);
        u128str(tots, b2);
        int ok = !strcmp(b1, A[n]) && !strcmp(b2, S[n]);
        printf("n=%d  a=%s  a180=%s  %s\n", n, b1, b2, ok ? "OK" : "FAIL");
        bad += !ok;
    }
    (void)cnt;
    printf(bad ? "SELFTEST FAILED\n" : "SELFTEST PASSED\n");
    return bad != 0;
}

/*
 * parse_threads()
 *   Function: reads a thread-count argument.
 *   Inputs:   s: the argument string.
 *   Outputs:  the count, an integer in 1..MAXTHR.
 *   Errors:   exits with status 1 and a message unless s is exactly such an
 *             integer (the pthread_t arrays hold MAXTHR threads).
 */
static int parse_threads(const char *s)
{
    char *end;
    long v = strtol(s, &end, 10);
    if (end == s || *end || v < 1 || v > MAXTHR) {
        fprintf(stderr, "thread count must be an integer from 1 to %d\n", MAXTHR);
        exit(1);
    }
    return (int)v;
}

/*
 * main()
 *   Function: builds the tables, sets the thread count to the number of
 *             online CPUs (at most MAXTHR) and dispatches on the arguments
 *             (see USAGE in the header).
 *   Inputs:   command line.
 *   Outputs:  exit status 0, or 1 for usage errors, a failed self test,
 *             an n out of range or an invalid thread count.
 *   Errors:   n and thread counts are range-checked.  The other numeric
 *             options (-p, -d, -c, -x) are read with atoi/atof without
 *             validation; a non-number gives 0.
 */
int main(int argc, char **argv)
{
    init_tables();
    init_vj();
    init_parts();
    long nc = sysconf(_SC_NPROCESSORS_ONLN);
    NTHR = nc > 0 ? (int)nc : 1;
    if (NTHR > MAXTHR) NTHR = MAXTHR;
    if (argc >= 3 && (!strcmp(argv[1], "-S") || !strcmp(argv[1], "-SV"))) {
        int nmax = atoi(argv[2]);
        if (nmax < 1 || nmax > MAXN) { fprintf(stderr, "N out of range (1..%d)\n", MAXN); return 1; }
        if (argc >= 4) NTHR = parse_threads(argv[3]);
        sym_all(nmax, argv[1][2] == 'V');
        return 0;
    }
    if (argc >= 2 && !strcmp(argv[1], "-s")) {
        if (argc >= 3) NTHR = parse_threads(argv[2]);
        return selftest();
    }
    if (argc >= 3 && (!strcmp(argv[1], "-r") || !strcmp(argv[1], "-b"))) {
        const char *aspec = NULL;
        int ai = 3;
        for (;;) {
            if (argc >= ai + 2 && !strcmp(argv[ai], "-a")) { aspec = argv[ai + 1]; ai += 2; continue; }
            if (argc >= ai + 2 && !strcmp(argv[ai], "-p")) { SAMPLE = atof(argv[ai + 1]); ai += 2; continue; }
            if (argc >= ai + 2 && !strcmp(argv[ai], "-d")) { SPLITMAX = atoi(argv[ai + 1]); ai += 2; continue; }
            if (argc >= ai + 1 && !strcmp(argv[ai], "-n")) { USE_D4 = 0; ai += 1; continue; }
            if (argc >= ai + 1 && !strcmp(argv[ai], "-q")) { NOPREPASS = 1; ai += 1; continue; }
            if (argc >= ai + 2 && !strcmp(argv[ai], "-x")) { DEBUG4 = atoi(argv[ai + 1]); ai += 2; continue; }
            if (argc >= ai + 2 && !strcmp(argv[ai], "-c")) { USECT = atoi(argv[ai + 1]); ai += 2; continue; }
            break;
        }
        if (argc > ai) NTHR = parse_threads(argv[ai]);
        ref_mode(argv[2], argv[1][1] == 'b', aspec);
        return 0;
    }
    if (argc < 2) {
        fprintf(stderr, "usage: %s N [threads] | -s [threads] | -r PROFILE [threads] | -b PROFILE [threads]\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    if (argc >= 3) NTHR = parse_threads(argv[2]);
    if (n < 1 || n > MAXN - 1) { fprintf(stderr, "N out of range\n"); return 1; }
    run_all(n, 1);
    return 0;
}
