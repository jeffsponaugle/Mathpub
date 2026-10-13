/*
 * a171775_metal.m -- Apple GPU (Metal) driver for the swapped 2D A171775 search.
 *
 * Everything except the inner loop comes from ../a171775_2d.c (included below): the problem
 * setup, work units (c, e_0), per-block ranges of check bases B, the per-(c, B) constants and
 * lookup tables, the survivor post-processing (canonical bases, lengths n-2..3, checksum) and
 * the state-file format.  So a run of this tool and of a171775_2d are directly comparable
 * (identical statistics, same checkpoint files).  Read the theory of operations there first;
 * the kernel is a171775_2d.metal, and cuda/a171775_2d_cuda.cu is a port of this tool.
 *
 * ========================================================================
 * THEORY OF OPERATIONS (the GPU part)
 * ========================================================================
 *
 * 1. Work.  A batch = consecutive units of one base c.  For each check base B the blocks of
 *    the batch whose B-range contains B form contiguous runs (batch_blocks(), make_items());
 *    runs are cut into threadgroups of 256 blocks.  One threadgroup = one base pair (c, B),
 *    one thread = one block of the search, walked exactly like block_B() (see the kernel).
 *
 * 2. Tables.  Per base c the host builds the pairs' tables with cctx_set() and converts them
 *    to the kernel's layout (slot_set_c(): PairC with 96-bit values as 32-bit limbs, the
 *    base-B digits of the block weights, the magic multiplier for W div g); the kernel copies
 *    its pair's tables into threadgroup memory (c <= 1024).
 *
 * 3. Pipeline.  Two batch slots alternate: the host prepares one while the GPU runs the
 *    other, and batches finish in issue order.  So the statistics always cover exactly the
 *    units below the oldest batch in flight, which is the frontier saved with -S: every
 *    checkpoint is exact.
 *
 * 4. The display watchdog.  On a Mac the GPU also drives the display, and macOS kills a
 *    command buffer that runs too long.  Batches are therefore sized for about 50 ms of GPU
 *    time (g_target_steps adapts to the measured rate and shrinks quickly).  A failed
 *    command buffer is redone with smaller batches (slot_retry(): nothing of it was
 *    counted).  If macOS reports the process as "Ignored (for causing prior/excessive GPU
 *    errors)", or after 8 failures in a row, the tool stops cleanly and saves its state; a
 *    new process gets a new GPU context, so rerunning the same command resumes.
 *
 * 5. The kernel source is compiled at run time by the OS Metal compiler (no Xcode Metal
 *    toolchain needed), from a171775_2d.metal next to the executable or the file given with
 *    -L (a .metallib is loaded as is).  A171775_ABLATE=n compiles the kernel with ABLATE=n
 *    (profiling switches that skip parts of the work); A171775_TIMING prints pipeline and
 *    timing details.
 *
 * Limits: n = 8..14 (n + 2 <= MAXL_G digits, 1..3 block digits); c <= 1024; P, Ptop and the
 * weights must fit in 96 bits; N <= 2^31 when g > 1; at most -b threadgroups per batch and
 * 2^20 survivors.  Every violation stops the run with a message (exit status 1).
 *
 * Usage
 *   a171775_metal selftest                 n = 8, 9, 10 full searches vs a171775_2d statistics
 *   a171775_metal search n LO HI [-S state] [-i sec] [-L metallib] [-b threadgroups] [-E end_unit]
 * Exit status: 0 = done, 3 = interrupted (also when stopped by the GPU errors above), 1 =
 * usage or a fatal error.
 */
#define A171775_2D_LIB
#include "../a171775_2d.c"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#define TGSIZE 256                     /* threads per threadgroup = blocks of the search per item */
#define MAXL_G 16                      /* digit buffer size in the kernel: n + 2 <= MAXL_G */
#define MAXC_G 1024                    /* largest c: the kernel's threadgroup tables hold 1024 entries */

/* PairC, GItem, Surv, Params -- the kernel's structures (see a171775_2d.metal and the
   identical ones in cuda/a171775_2d_cuda.cu): one pair's constants with 96-bit values as
   32-bit limbs; one threadgroup's work (pair, first block index j0, block count, range-test
   flag); one survivor (q1, rem, B, c, x, j); [lo, hi] as limbs, survivor capacity and n. */
typedef struct {                       /* must match PairC in a171775_2d.metal */
    uint32_t B, k, N, g, Np, u, lim, R;
    uint32_t Wx, sh, nb, toff, boff, Lc, Lk, c;
    uint32_t Wc[4];
    uint32_t P[4], Rc[4], Ptop[4];
    uint32_t wx[4], wy[4], wz[4], NpWz[4];
    uint32_t ox_lo, ox_hi, Bk, bd;
    uint32_t blkw[3][MAXL_G];
    float invPtop, invg;
    uint32_t gm, gs;
} PairC;

typedef struct { uint32_t pair, j0, count, pad; } GItem;
typedef struct { uint32_t q, r0, r1, r2, B, c, x, j; } Surv;
typedef struct { uint32_t lo[4], hi[4]; uint32_t outcap, n, pad0, pad1; } Params;

/* put96 -- v as three 32-bit limbs in d[0..2] (d[3] = 0); exits with status 1 if
   v >= 2^96.  put64 -- v as two limbs (currently unused). */
static void put96(uint32_t *d, u128 v, const char *what)
{
    if (v >> 96) { fprintf(stderr, "%s does not fit in 96 bits\n", what); exit(1); }
    d[0] = (uint32_t)v; d[1] = (uint32_t)(v >> 32); d[2] = (uint32_t)(v >> 64); d[3] = 0;
}

static void put64(uint32_t *d, u64 v) { d[0] = (uint32_t)v; d[1] = (uint32_t)(v >> 32); }

/* Slot -- one of the two batch slots: its Metal buffers (items, the pair constants and
   tables of its base c, survivors and their count, per-threadgroup statistics), the
   command buffer in flight, the host tables (cctx_t) of base c, and the batch: items,
   units [u_begin, u_end), steps, busy.  The buffers are shared CPU/GPU memory. */
typedef struct {
    id<MTLBuffer> items, pairs, tvals, tys, tbst, outs, nout, tgstat, tay_lo, tay_hi, taz_lo, taz_hi;
    id<MTLCommandBuffer> cmd;
    cctx_t cc;                         /* host tables of this slot's base c */
    u64 c;
    size_t nitems, maxitems;
    u64 u_begin, u_end, steps;
    int busy;
} Slot;

static size_t MAXITEMS = 1 << 14;      /* threadgroups per batch (-b) */
#define OUTCAP (1u << 20)              /* survivors per batch */

/*
 * slot_set_c -- make the slot ready for base c (nothing to do if it already is): build the
 * host tables (cctx_set), allocate the slot's table buffers for this c, and convert every
 * pair (c, B) to the kernel layout: PairC (96-bit limbs, the base-B digits of the block
 * weights, the magic multiplier for W div g, spot-checked) and the packed lookup, bucket and
 * digit-k tables.  Call only when the slot is idle.
 * Errors: exits with status 1 if c > 1024 or n is too large for the kernel, a value does not
 * fit in 96 bits, N > 2^31 with g > 1, or the magic division fails its check.
 */
static void slot_set_c(Slot *s, const prob_t *pr, id<MTLDevice> dev, u64 c)
{
    if (s->c == c) return;
    if (c > MAXC_G) { fprintf(stderr, "c = %llu too large for the GPU kernel (c <= %d)\n", (unsigned long long)c, MAXC_G); exit(1); }
    cctx_set(&s->cc, pr, c);
    s->c = c;
    const cctx_t *cc = &s->cc;
    size_t np = cc->Bcount ? cc->Bcount : 1;
    s->pairs = [dev newBufferWithLength:np * sizeof(PairC) options:MTLResourceStorageModeShared];
    s->tvals = [dev newBufferWithLength:np * c * sizeof(uint32_t) + 16 options:MTLResourceStorageModeShared];
    s->tys = [dev newBufferWithLength:np * c * sizeof(uint16_t) + 16 options:MTLResourceStorageModeShared];
    s->tbst = [dev newBufferWithLength:np * (c + 3) * sizeof(uint16_t) + 16 options:MTLResourceStorageModeShared];
    s->tay_lo = [dev newBufferWithLength:np * c * sizeof(uint32_t) + 16 options:MTLResourceStorageModeShared];
    s->tay_hi = [dev newBufferWithLength:np * c * sizeof(uint16_t) + 16 options:MTLResourceStorageModeShared];
    s->taz_lo = [dev newBufferWithLength:np * c * sizeof(uint32_t) + 16 options:MTLResourceStorageModeShared];
    s->taz_hi = [dev newBufferWithLength:np * c * sizeof(uint16_t) + 16 options:MTLResourceStorageModeShared];
    uint32_t *ayl = (uint32_t *)s->tay_lo.contents, *azl = (uint32_t *)s->taz_lo.contents;
    uint16_t *ayh = (uint16_t *)s->tay_hi.contents, *azh = (uint16_t *)s->taz_hi.contents;
    PairC *pc = (PairC *)s->pairs.contents;
    uint32_t *tv = (uint32_t *)s->tvals.contents;
    uint16_t *ty = (uint16_t *)s->tys.contents, *tb = (uint16_t *)s->tbst.contents;
    size_t toff = 0, boff = 0;
    const int n = pr->n;
    if (n + 2 > MAXL_G) { fprintf(stderr, "n too large for the GPU kernel\n"); exit(1); }
    for (u64 i = 0; i < cc->Bcount; i++) {
        const bconst_t *K = &cc->tab[i];
        PairC *p = &pc[i];
        memset(p, 0, sizeof *p);
        p->B = (uint32_t)K->B; p->k = (uint32_t)K->k; p->N = (uint32_t)K->N; p->g = (uint32_t)K->g;
        p->Np = (uint32_t)K->Np; p->u = (uint32_t)K->u; p->lim = (uint32_t)K->lim; p->R = (uint32_t)K->R;
        p->Wx = (uint32_t)K->Wx; p->sh = (uint32_t)K->sh; p->nb = K->nb;
        p->toff = (uint32_t)toff; p->boff = (uint32_t)boff;
        p->Lc = (uint32_t)n; p->Lk = (uint32_t)(n - (int)K->k); p->c = (uint32_t)c;
        for (int j = 0; j < (int)K->k; j++) p->Wc[j] = (uint32_t)K->Wc[j];
        put96(p->P, K->P, "P");
        put96(p->Rc, K->Rc, "Rc");
        put96(p->Ptop, K->Ptop, "Ptop");
        put96(p->wx, cc->wx, "w_x");
        put96(p->wy, cc->wy, "w_y");
        put96(p->wz, cc->wz, "w_z");
        if (K->Np < c) put96(p->NpWz, (u128)K->Np * cc->wz, "N' w_z");
        p->ox_lo = (uint32_t)(K->oxK % K->Bk);
        p->ox_hi = (uint32_t)(K->oxK / K->Bk);
        p->Bk = (uint32_t)K->Bk;
        p->bd = (uint32_t)pr->bd;
        for (int t = 0; t < pr->bd; t++) {
            u128 v = cc->w[t];
            for (int d = 0; d < MAXL_G; d++) { p->blkw[t][d] = (uint32_t)(v % K->B); v /= K->B; }
            if (v) { fprintf(stderr, "block weight has more than %d base-B digits\n", MAXL_G); exit(1); }
        }
        p->invPtop = (float)(1.0 / u128d(K->Ptop));
        p->invg = (float)(1.0 / (double)K->g);
        if (K->g > 1) {                    /* exact for W < 2^31: M = ceil(2^(31+p)/g), p = ceil(log2 g) */
            if (K->N > (1ull << 31)) { fprintf(stderr, "N too large for the magic division\n"); exit(1); }
            int pl = 0;
            while ((1ull << pl) < K->g) pl++;
            u64 M = (u64)((((u128)1 << (31 + pl)) + K->g - 1) / K->g);
            if (M >> 32) { fprintf(stderr, "magic overflow\n"); exit(1); }
            p->gm = (uint32_t)M;
            p->gs = (uint32_t)(pl - 1);
            for (u64 t = 0; t < 4096; t++) {        /* spot check */
                u64 w = (t * 2654435761ull) % K->N;
                if (((w * M) >> 32 >> (pl - 1)) != w / K->g) { fprintf(stderr, "magic division check failed\n"); exit(1); }
            }
        }
        for (u64 t = 0; t < c; t++) { tv[toff + t] = K->val[t]; ty[toff + t] = K->ys[t]; }
        for (u64 t = 0; t < c; t++) {     /* t w_y and t w_z mod B^(k+1), split at B^k */
            u64 vy = (u64)((u128)t * K->oyK % K->BK1), vz = (u64)((u128)t * K->ozK % K->BK1);
            ayl[toff + t] = (uint32_t)(vy % K->Bk); ayh[toff + t] = (uint16_t)(vy / K->Bk);
            azl[toff + t] = (uint32_t)(vz % K->Bk); azh[toff + t] = (uint16_t)(vz / K->Bk);
        }
        for (u32 t = 0; t <= K->nb; t++) tb[boff + t] = (uint16_t)K->bst[t];
        toff += c;
        boff += K->nb + 1;
    }
}

/* BlkR -- one block: linear index j among the blocks of base c, its check-base range
   [Blo, Bhi], and whether it lies wholly inside [lo, hi]. */
typedef struct { u64 j; u64 Blo, Bhi; int inside; } BlkR;

/*
 * batch_blocks -- the blocks of the units [u0, u1) of base c that do_unit()/do_block()
 * would process, with their B ranges.
 * Out:    the number written to out[]; *steps = their x-steps (the CPU tool's "steps").
 * Errors: exits with status 1 if more than cap blocks ("block buffer overflow").
 */
static size_t batch_blocks(const prob_t *pr, const cctx_t *cc, u64 u0, u64 u1, BlkR *out, size_t cap, u64 *steps)
{
    size_t nb_ = 0;
    u64 c = cc->c;
    *steps = 0;
    for (u64 u = u0; u < u1; u++) {
        u64 pref[MAXN];
        unit_decode(pr, u, pref);
        u128 Mu = 0;
        for (int i = 0; i < pr->ud; i++) Mu += (u128)pref[i] * cc->w[i];
        if (Mu + cc->span_unit < pr->lo || Mu > pr->hi) continue;
        int nbd = pr->bd - pr->ud;
        u64 nblk = nbd ? c : 1;
        /* linear block index: j = (e_0 - 1) c^(bd-1) + e_1 c^(bd-2) + ... */
        u64 jbase = 0;
        for (int i = 0; i < pr->ud; i++) jbase = jbase * c + (i == 0 ? pref[0] - 1 : pref[i]);
        for (int i = pr->ud; i < pr->bd; i++) jbase *= c;
        for (u64 d = 0; d < nblk; d++) {
            u128 Mb = Mu + (nbd ? (u128)d * cc->w[pr->ud] : 0);
            if (nbd && Mb > pr->hi) break;
            if (!cc->Bcount) continue;
            u128 Mmin = Mb, Mmax = Mb + cc->span_block;
            if (Mmax < pr->lo || Mmin > pr->hi) continue;
            int inside = Mmin >= pr->lo && Mmax <= pr->hi;
            if (Mmin < pr->lo) Mmin = pr->lo;
            if (Mmax > pr->hi) Mmax = pr->hi;
            u64 Blo = iroot(Mmin, pr->Lc) + 1, Bhi = iroot(Mmax, pr->Lc - 1);
            if (Blo < cc->Bbase) Blo = cc->Bbase;
            if (Bhi > cc->Bbase + cc->Bcount - 1) Bhi = cc->Bbase + cc->Bcount - 1;
            if (Bhi < Blo) continue;
            if (nb_ == cap) { fprintf(stderr, "block buffer overflow\n"); exit(1); }
            out[nb_].j = jbase + d;
            out[nb_].Blo = Blo;
            out[nb_].Bhi = Bhi;
            out[nb_].inside = inside;
            nb_++;
            *steps += (Bhi - Blo + 1) * c;
        }
    }
    return nb_;
}

/* make_items -- for each B, the runs of consecutive blocks whose range contains B, in
   pieces of at most TGSIZE; returns the number of items, or (size_t)-1 if more than cap. */
static size_t make_items(const cctx_t *cc, const BlkR *bl, size_t nbl, GItem *it, size_t cap)
{
    size_t ni = 0;
    if (!nbl) return 0;
    u64 Bmin = bl[0].Blo, Bmax = bl[0].Bhi;
    for (size_t i = 1; i < nbl; i++) {
        if (bl[i].Blo < Bmin) Bmin = bl[i].Blo;
        if (bl[i].Bhi > Bmax) Bmax = bl[i].Bhi;
    }
    for (u64 B = Bmin; B <= Bmax; B++) {
        size_t i = 0;
        while (i < nbl) {
            if (bl[i].Blo > B || bl[i].Bhi < B) { i++; continue; }
            size_t a = i;
            while (i + 1 < nbl && bl[i + 1].j == bl[i].j + 1 && bl[i + 1].Blo <= B && bl[i + 1].Bhi >= B &&
                   i + 1 - a < TGSIZE)
                i++;
            if (ni == cap) return (size_t)-1;
            it[ni].pair = (uint32_t)(B - cc->Bbase);
            it[ni].j0 = (uint32_t)bl[a].j;
            it[ni].count = (uint32_t)(i - a + 1);
            it[ni].pad = 0;
            for (size_t t = a; t <= i; t++)
                if (!bl[t].inside) it[ni].pad = 1;   /* some block reaches outside [lo, hi]: range check on */
            ni++;
            i++;
        }
    }
    return ni;
}

static u64 g_target_steps = 1ull << 24;   /* thread-steps per command buffer, adapted to ~50 ms */
static double g_gpu_rate = 0;             /* measured thread-steps per GPU second */
static double g_gpu_time, g_wait_time, g_prep_time, g_post_time, g_max_cmd;   /* A171775_TIMING */
static u64 g_ncmd, g_nthreads, g_ntg;

static u64 g_retries, g_overflow, g_consec_fail;   /* g_overflow: left from the old queue design, always 0 */
static int g_gpu_banned;                  /* macOS stopped accepting our command buffers */

/* returns 0 if the GPU refused the command buffer (display watchdog); nothing is counted then */
/*
 * slot_finish -- wait for the slot's command buffer and account for its batch: adapt
 * g_target_steps, rebuild each survivor M = q1 P + rem, re-check both palindromes on the
 * host and call survivor(), add the threadgroups' lookups, hits and f1 and the batch's steps,
 * and move done_units to the end of the batch.
 * In:     s (busy); st; done_units; n.
 * Out:    1 if the batch was accounted for; 0 if the command buffer failed: nothing is
 *         counted, and g_gpu_banned is set if retrying cannot help (see the header).
 * Errors: exits with status 1 on a survivor buffer overflow or a GPU survivor that fails
 *         the host check (a kernel bug).
 */
static int slot_finish(Slot *s, wstat_t *st, u64 *done_units, int n)
{
    double tw = now_sec();
    [s->cmd waitUntilCompleted];
    g_wait_time += now_sec() - tw;
    if (s->cmd.status != MTLCommandBufferStatusCompleted) {
        const char *msg = s->cmd.error.localizedDescription.UTF8String;
        g_consec_fail++;
        /* "Ignored (for causing prior/excessive GPU errors)": this process is barred from the GPU;
           retrying cannot help, a new process gets a fresh context */
        if ((msg && strstr(msg, "Ignored")) || g_consec_fail > 8) g_gpu_banned = 1;
        fprintf(stderr, "GPU command buffer failed (%s) for units %llu..%llu%s\n", msg ? msg : "?",
                (unsigned long long)s->u_begin, (unsigned long long)s->u_end,
                g_gpu_banned ? "; stopping (restart the same command to resume)" : "; retrying with smaller batches");
        s->busy = 0;
        return 0;
    }
    g_consec_fail = 0;
    double gt = s->cmd.GPUEndTime - s->cmd.GPUStartTime;
    g_gpu_time += gt;
    if (gt > 0.002 && s->steps > 0) {
        double r = s->steps / gt;
        g_gpu_rate = g_gpu_rate > 0 ? 0.7 * g_gpu_rate + 0.3 * r : r;
        double tgt = (r < g_gpu_rate ? r : g_gpu_rate) * 0.05;   /* ~50 ms per buffer, reacting fast to slowdowns */
        if (tgt < 1e6) tgt = 1e6;
        if (tgt > 4e10) tgt = 4e10;
        g_target_steps = (u64)tgt;
    }
    if (gt > g_max_cmd) g_max_cmd = gt;
    g_ncmd++;
    double tp = now_sec();
    uint32_t no = *(uint32_t *)s->nout.contents;
    if (no > OUTCAP) { fprintf(stderr, "survivor buffer overflow\n"); exit(1); }
    const Surv *sv = (const Surv *)s->outs.contents;
    for (uint32_t i = 0; i < no; i++) {
        const bconst_t *K = &s->cc.tab[sv[i].B - s->cc.Bbase];
        u128 rf = (u128)sv[i].r0 | ((u128)sv[i].r1 << 32) | ((u128)sv[i].r2 << 64);
        u128 M = (u128)sv[i].q * K->P + rf;
        if (!is_pal_base(M, sv[i].B, n) || !is_pal_base(M, sv[i].c, n - 1)) {
            char buf[48];
            fprintf(stderr, "GPU survivor fails the host check: M=%s B=%u c=%u\n", u128s(M, buf), sv[i].B, sv[i].c);
            exit(1);
        }
        survivor(st, M, sv[i].B, sv[i].c);
    }
    const uint64_t *ts = (const uint64_t *)s->tgstat.contents;
    for (size_t i = 0; i < s->nitems; i++) {
        st->lookups += ts[4 * i]; st->hits += ts[4 * i + 1]; st->f1 += ts[4 * i + 2]; g_overflow += ts[4 * i + 3];
    }
    st->steps += s->steps;
    if (s->u_end > *done_units) *done_units = s->u_end;
    s->busy = 0;
    g_post_time += now_sec() - tp;
    return 1;
}

/* the older slot failed: drop the newer slot's batch too, rewind, and shrink the batches */
/* slot_retry -- In: the failed slot, the other slot, &unext.  Out: the other slot's batch is
   waited for and dropped uncounted, unext rewinds to the failed batch's first unit, and the
   batch size target drops by a factor 8 (at least 100000 steps).  No errors. */
static void slot_retry(Slot *failed, Slot *other, u64 *unext)
{
    if (other->busy) {
        [other->cmd waitUntilCompleted];
        other->busy = 0;
    }
    *unext = failed->u_begin;
    g_target_steps = g_target_steps / 8 > 100000 ? g_target_steps / 8 : 100000;
    g_gpu_rate /= 8;
    g_retries++;
}


/*
 * gpu_search -- the "search" command (and the selftest's searches); the counterpart of
 * run_search() in ../a171775_2d.c.
 *
 * In:     n (8..14); lo, hi; state (checkpoint file or NULL); interval (seconds between
 *         progress lines); libpath (kernel source or .metallib); quiet; result (may be NULL).
 * Out:    SOLUTION lines, progress, summary and checkpoints as run_search(); returns 0 when
 *         complete (also without units), 3 if interrupted or stopped by GPU errors, 1 if
 *         there is no Metal device, the kernel cannot be loaded, or n is not supported.
 * How:    load and compile the kernel; then loop: finish the slot's previous batch (on a
 *         failure: slot_retry and try again); build a batch of whole units of one base c
 *         (at most g_target_steps steps and 4e6 threads); encode and commit it; switch
 *         slots.  When nothing is left to issue, finish both slots, oldest first.
 *         Checkpoints use the first unit of the oldest batch in flight, so they are exact.
 * Errors: exits with status 1 on the limits of slot_set_c(), batch_blocks() and
 *         make_items(); load_state() refuses a state file with a frontier beyond -E.
 */
static int gpu_search(int n, u128 lo, u128 hi, const char *state, double interval, const char *libpath, int quiet,
                      wstat_t *result)
{
    @autoreleasepool {
        char a[48], b[48];
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no Metal device\n"); return 1; }
        NSError *err = nil;
        id<MTLLibrary> lib = nil;
        NSString *path = [NSString stringWithUTF8String:libpath];
        if ([path hasSuffix:@".metallib"]) {
            lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
        } else {                          /* compile the kernel source with the OS Metal compiler */
            NSString *srcText = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&err];
            if (srcText) {
                MTLCompileOptions *opt = [MTLCompileOptions new];
                opt.mathMode = MTLMathModeSafe;
                const char *abl = getenv("A171775_ABLATE");
                if (abl) opt.preprocessorMacros = @{@"ABLATE" : [NSNumber numberWithInt:atoi(abl)]};
                lib = [dev newLibraryWithSource:srcText options:opt error:&err];
            }
        }
        if (!lib) { fprintf(stderr, "cannot load %s: %s\n", libpath, err.localizedDescription.UTF8String); return 1; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"scan2"];
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { fprintf(stderr, "pipeline: %s\n", err.localizedDescription.UTF8String); return 1; }
        if (pso.maxTotalThreadsPerThreadgroup < TGSIZE) {
            fprintf(stderr, "kernel allows only %lu threads per threadgroup\n", (unsigned long)pso.maxTotalThreadsPerThreadgroup);
            return 1;
        }
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        if (getenv("A171775_TIMING"))
            fprintf(stderr, "pipeline: maxThreadsPerThreadgroup %lu, threadExecutionWidth %lu, static threadgroup memory %lu bytes\n",
                    (unsigned long)pso.maxTotalThreadsPerThreadgroup, (unsigned long)pso.threadExecutionWidth,
                    (unsigned long)pso.staticThreadgroupMemoryLength);

        prob_t pr;
        prob_init(&pr, n, lo, hi);
        if (g_end_unit < pr.nunits) pr.nunits = g_end_unit;
        if (pr.bd < 1 || pr.bd > 3) { fprintf(stderr, "n = %d not supported by the GPU kernel\n", n); return 1; }
        wstat_t st, base;
        memset(&st, 0, sizeof st);
        memset(&base, 0, sizeof base);
        st.pr = &pr;
        u64 frontier = 0;
        if (state && load_state(state, &pr, &frontier, &base) && !quiet)
            fprintf(stderr, "resuming at unit %llu of %llu\n", (unsigned long long)frontier, (unsigned long long)pr.nunits);
        if (!quiet)
            fprintf(stderr, "metal search n=%d [%s, %s] on %s: bases c in [%llu, %llu], %llu units\n", n, u128s(lo, a),
                    u128s(hi, b), dev.name.UTF8String, (unsigned long long)pr.cmin, (unsigned long long)pr.cmax,
                    (unsigned long long)pr.nunits);

        Params prm;
        memset(&prm, 0, sizeof prm);
        for (int i = 0; i < 4; i++) { prm.lo[i] = (uint32_t)(pr.lo >> (32 * i)); prm.hi[i] = (uint32_t)(pr.hi >> (32 * i)); }
        prm.outcap = OUTCAP;
        prm.n = (uint32_t)n;

        Slot sl[2];
        memset(sl, 0, sizeof sl);
        for (int i = 0; i < 2; i++) {
            sl[i].maxitems = MAXITEMS;
            sl[i].items = [dev newBufferWithLength:MAXITEMS * sizeof(GItem) options:MTLResourceStorageModeShared];
            sl[i].outs = [dev newBufferWithLength:OUTCAP * sizeof(Surv) options:MTLResourceStorageModeShared];
            sl[i].nout = [dev newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
            sl[i].tgstat = [dev newBufferWithLength:4 * MAXITEMS * sizeof(uint64_t) options:MTLResourceStorageModeShared];
        }
        size_t blkcap = 1 << 22;
        BlkR *blk = malloc(blkcap * sizeof(BlkR));
        GItem *itmp = malloc(MAXITEMS * sizeof(GItem));

        signal(SIGINT, on_sig);
        signal(SIGTERM, on_sig);
        double t0 = now_sec(), tlast = t0, tsave = t0;
        u64 unext = frontier, done_units = frontier;
        int cur = 0;
        for (;;) {
            Slot *s = &sl[cur];
            if (s->busy && !slot_finish(s, &st, &done_units, n)) {   /* the older batch failed */
                slot_retry(s, &sl[cur ^ 1], &unext);
                continue;
            }
            int issued = 0;
            if (unext < pr.nunits && !g_stop && !g_gpu_banned) {
                double tprep = now_sec();
                /* a batch: units of one base c until the item buffer is about full */
                u64 pref[MAXN];
                u64 c = unit_decode(&pr, unext, pref);
                slot_set_c(s, &pr, dev, c);
                u64 ub = unext, ue = unext, steps = 0, threads = 0;
                size_t nbl = 0;
                while (ue < pr.nunits) {
                    if (unit_decode(&pr, ue, pref) != c) break;
                    u64 st_, th = 0;
                    size_t add = batch_blocks(&pr, &s->cc, ue, ue + 1, blk + nbl, blkcap - nbl, &st_);
                    for (size_t i = nbl; i < nbl + add; i++) th += blk[i].Bhi - blk[i].Blo + 1;
                    if (ue > ub && (steps + st_ > g_target_steps || threads + th > 4000000)) break;
                    nbl += add;
                    threads += th;
                    steps += st_;
                    ue++;
                }
                size_t ni = make_items(&s->cc, blk, nbl, itmp, MAXITEMS);
                if (ni == (size_t)-1) { fprintf(stderr, "item buffer overflow (raise -b)\n"); exit(1); }
                memcpy(s->items.contents, itmp, ni * sizeof(GItem));
                s->nitems = ni;
                s->u_begin = ub;
                s->u_end = ue;
                s->steps = steps;
                *(uint32_t *)s->nout.contents = 0;
                s->cmd = [queue commandBuffer];
                if (ni) {
                    id<MTLComputeCommandEncoder> enc = [s->cmd computeCommandEncoder];
                    [enc setComputePipelineState:pso];
                    [enc setBuffer:s->items offset:0 atIndex:0];
                    [enc setBuffer:s->pairs offset:0 atIndex:1];
                    [enc setBuffer:s->tvals offset:0 atIndex:2];
                    [enc setBuffer:s->tys offset:0 atIndex:3];
                    [enc setBuffer:s->tbst offset:0 atIndex:4];
                    [enc setBytes:&prm length:sizeof prm atIndex:5];
                    [enc setBuffer:s->outs offset:0 atIndex:6];
                    [enc setBuffer:s->nout offset:0 atIndex:7];
                    [enc setBuffer:s->tgstat offset:0 atIndex:8];
                    [enc setBuffer:s->tay_lo offset:0 atIndex:9];
                    [enc setBuffer:s->tay_hi offset:0 atIndex:10];
                    [enc setBuffer:s->taz_lo offset:0 atIndex:11];
                    [enc setBuffer:s->taz_hi offset:0 atIndex:12];
                    [enc dispatchThreadgroups:MTLSizeMake(ni, 1, 1) threadsPerThreadgroup:MTLSizeMake(TGSIZE, 1, 1)];
                    [enc endEncoding];
                }
                g_prep_time += now_sec() - tprep;
                g_nthreads += threads;
                g_ntg += ni;
                [s->cmd commit];
                s->busy = 1;
                unext = ue;
                issued = 1;
            }
            cur ^= 1;
            int drained = !issued && !sl[0].busy && !sl[1].busy;
            if (!issued) {                 /* nothing left to issue: drain, oldest first */
                int ok = 1;
                if (sl[cur].busy && !slot_finish(&sl[cur], &st, &done_units, n)) {
                    slot_retry(&sl[cur], &sl[cur ^ 1], &unext);
                    ok = 0;
                } else if (sl[cur ^ 1].busy && !slot_finish(&sl[cur ^ 1], &st, &done_units, n)) {
                    unext = sl[cur ^ 1].u_begin;   /* the newer one failed: redo it, smaller */
                    g_target_steps = g_target_steps / 8 > 100000 ? g_target_steps / 8 : 100000;
                    g_gpu_rate /= 8;
                    g_retries++;
                    ok = 0;
                }
                if (!ok && !g_stop && !g_gpu_banned) continue;
                drained = 1;
            }
            double t = now_sec();
            if (!quiet && (t - tlast >= interval || drained)) {
                u64 pr_[MAXN];
                u64 cc_ = unit_decode(&pr, done_units < pr.nunits ? done_units : pr.nunits - 1, pr_);
                fprintf(stderr, "[%7.0fs] frontier %llu/%llu (c=%llu)  %.3e steps/s %.3e lookups/s  st2 %llu canon %llu sol %llu\n",
                        t - t0, (unsigned long long)done_units, (unsigned long long)pr.nunits, (unsigned long long)cc_,
                        st.steps / (t - t0 > 0 ? t - t0 : 1), st.lookups / (t - t0 > 0 ? t - t0 : 1), (unsigned long long)(base.st2 + st.st2),
                        (unsigned long long)(base.canon + st.canon), (unsigned long long)(base.sol + st.sol));
                tlast = t;
            }
            if (state && (t - tsave >= 60 || drained)) {
                wstat_t all = st;              /* the finished batches: exactly the units below fr */
                add_stats(&all, &base);
                u64 fr = done_units;           /* = the first unit of the oldest batch in flight */
                for (int i = 0; i < 2; i++) if (sl[i].busy && sl[i].u_begin < fr) fr = sl[i].u_begin;
                save_state(state, &pr, fr, &all);
                tsave = t;
            }
            if (drained) break;
        }
        wstat_t all = st;
        add_stats(&all, &base);
        int complete = done_units >= pr.nunits;
        if (!quiet) print_summary(stdout, &pr, &all, complete, now_sec() - t0);
        if (getenv("A171775_TIMING"))
            fprintf(stderr, "timing: %llu queue overflows; %llu command buffers (%llu retried), %llu threadgroups, %llu threads; GPU %.3fs (max %.3fs per buffer), "
                            "host wait %.3fs, prep %.3fs, post %.3fs, wall %.3fs\n",
                    (unsigned long long)g_overflow, (unsigned long long)g_ncmd, (unsigned long long)g_retries, (unsigned long long)g_ntg, (unsigned long long)g_nthreads, g_gpu_time,
                    g_max_cmd, g_wait_time, g_prep_time, g_post_time, now_sec() - t0);
        if (result) *result = all;
        free(blk);
        free(itmp);
        cctx_free(&sl[0].cc);
        cctx_free(&sl[1].cc);
        free(pr.uoff);
        return complete ? 0 : 3;
    }
}

/*
 * metal_selftest -- complete searches over [1, a(n)] for n = 8, 9, 10 with the kernel from
 * libpath: each must find exactly one solution, a(n), with all seven statistics of the CPU
 * tool.  Returns 0 if all agree, 1 otherwise.  -E must not be given.
 */
static int metal_selftest(const char *libpath)
{
    /* statistics of a171775_2d (CPU) for the same searches */
    struct { int n; const char *hi; u64 steps, lookups, hits, f1, st2, canon, csum; } ref[] = {
        {8, "2^42", 2795327ULL, 3386590ULL, 61277350ULL, 1488018ULL, 4725, 4724, 0x13747afea2768e7aULL},
        {9, "2^56", 42728678ULL, 50351034ULL, 1504469668ULL, 19481395ULL, 19140, 19140, 0x380d9946bb59f02eULL},
        {10, "2^72", 158029794515ULL, 178478637779ULL, 46118188570ULL, 343754929ULL, 135382, 135382, 0xe9eed6127c51a50fULL},
    };
    int bad = 0;
    for (int i = 0; i < 3; i++) {
        FILE *mem = tmpfile(), *save = sol_out;
        sol_out = mem;
        wstat_t t;
        double t0 = now_sec();
        gpu_search(ref[i].n, 1, parse_or_die(ref[i].hi), NULL, 1e9, libpath, 1, &t);
        sol_out = save;
        rewind(mem);
        char line[1024];
        int nsol = 0;
        u128 best = 0;
        while (fgets(line, sizeof line, mem)) {
            char *p = strstr(line, "M=");
            if (!p) continue;
            char num[64];
            sscanf(p + 2, "%63[0-9]", num);
            u128 M = parse_or_die(num);
            if (!best || M < best) best = M;
            nsol++;
        }
        fclose(mem);
        int ok = nsol == 1 && best == parse_or_die(ref[i].hi) && t.steps == ref[i].steps && t.lookups == ref[i].lookups &&
                 t.hits == ref[i].hits && t.f1 == ref[i].f1 && t.st2 == ref[i].st2 && t.canon == ref[i].canon &&
                 t.csum == ref[i].csum;
        char buf[48];
        double el = now_sec() - t0;
        printf("  n=%d [1, %s]: %.2fs (%.3e lookups/s), steps %llu, lookups %llu, hits %llu, f1 %llu, st2 %llu, canonical %llu, "
               "checksum %016llx, %d solution(s), smallest %s  %s\n",
               ref[i].n, ref[i].hi, el, t.lookups / el, (unsigned long long)t.steps, (unsigned long long)t.lookups,
               (unsigned long long)t.hits, (unsigned long long)t.f1, (unsigned long long)t.st2, (unsigned long long)t.canon,
               (unsigned long long)t.csum, nsol, best ? u128s(best, buf) : "none", ok ? "ok" : "MISMATCH");
        fflush(stdout);
        bad += !ok;
    }
    printf(bad ? "SELFTEST FAILED\n" : "selftest passed\n");
    return bad ? 1 : 0;
}

/* main -- parse -S, -i, -L, -b, -E and the command (search needs n = 8..14).  The kernel
   defaults to a171775_2d.metal in the executable's directory.  Unknown commands print the
   usage and return 1. */
int main(int argc, char **argv)
{
    sol_out = stdout;
    const char *state = NULL, *lib = NULL;
    double interval = 10;
    char *pos[8];
    int npos = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-S") && i + 1 < argc) state = argv[++i];
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) interval = atof(argv[++i]);
        else if (!strcmp(argv[i], "-L") && i + 1 < argc) lib = argv[++i];
        else if (!strcmp(argv[i], "-b") && i + 1 < argc) MAXITEMS = (size_t)atof(argv[++i]);
        else if (!strcmp(argv[i], "-E") && i + 1 < argc) g_end_unit = strtoull(argv[++i], NULL, 10);
        else if (npos < 8) pos[npos++] = argv[i];
    }
    if (MAXITEMS < 1) MAXITEMS = 1;      /* -b 0 would allocate empty buffers */
    char deflib[4096];
    if (!lib) {                          /* default: the kernel source a171775_2d.metal next to the executable */
        snprintf(deflib, sizeof deflib, "%s", argv[0]);
        char *slash = strrchr(deflib, '/');
        if (slash) snprintf(slash + 1, sizeof deflib - (size_t)(slash + 1 - deflib), "a171775_2d.metal");
        else snprintf(deflib, sizeof deflib, "a171775_2d.metal");
        lib = deflib;
    }
    if (npos >= 1 && !strcmp(pos[0], "selftest")) return metal_selftest(lib);
    if (npos == 4 && !strcmp(pos[0], "search")) {
        int n = atoi(pos[1]);
        if (n < 8 || n > 14) { fprintf(stderr, "n must be 8..14\n"); return 1; }
        return gpu_search(n, parse_or_die(pos[2]), parse_or_die(pos[3]), state, interval, lib, 0, NULL);
    }
    fprintf(stderr, "usage: a171775_metal selftest | search n LO HI [-S state] [-i sec] [-L metallib] [-b threadgroups]\n");
    return 1;
}
