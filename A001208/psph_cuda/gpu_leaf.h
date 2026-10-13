/*
 * gpu_leaf.h -- CUDA leaf for psph (postage stamp exhaustive search), C-linkage API.
 *
 * psph.c (built with -DGPU_LEAF) hands every surviving (k-1)-prefix to gpu_submit_prefix() instead of
 * running its own a_k candidate loop.  Prefixes are batched per CPU thread; a full batch runs
 *   kernel 1 (filter: Challis X test, deep-hole targets, reach target, top-block targets, TGT) and
 *   kernel 2 (exact streaming full check of the survivors)
 * on the thread's own CUDA stream, and every basis that passes the full check is handed back through the
 * report callback (psph.c re-verifies it with an independent exact DP in report_solution).
 *
 * Cell type must match psph.c: uint8_t (default) or uint16_t (-DWIDE) -- both sides see the same macro.
 *
 * Threading contract: every psph worker thread tid owns batch tid.  gpu_submit_prefix(tid, ...) and
 * gpu_flush(tid) are called only by worker tid; gpu_init, gpu_finish, gpu_print_stats and gpu_shutdown are
 * called by the main thread while no worker is running; gpu_thread_stats may be called by the progress loop
 * at any time (it reads counters that change once per finished batch).
 *
 * Call sequence per search (psph.c run_search):  gpu_init -> workers call gpu_submit_prefix for every
 * surviving prefix and gpu_flush when they run out of work -> gpu_finish -> gpu_thread_stats / gpu_print_stats.
 * gpu_shutdown at program end.  Errors inside the GPU code terminate the process (exit 7, or 8 for a
 * PSPH_GPU_PARANOID mismatch); only gpu_init reports a failure by its return value.
 */
#ifndef GPU_LEAF_H
#define GPU_LEAF_H
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef WIDE
typedef uint16_t gpu_cell_t;
#else
typedef uint8_t gpu_cell_t;
#endif

/*
 * gpu_report_fn -- callback type for accepted bases.
 *
 * Called (on the submitting CPU thread, from inside gpu_submit_prefix / gpu_flush) for every a_k that
 * passed the exact full check; a[1..k-1] is the prefix, ak the new last element, user the pointer given to
 * gpu_init.  psph.c's callback re-verifies the basis with an independent DP and prints it.
 */
typedef void (*gpu_report_fn)(int thread_id, const int64_t *a, int64_t ak, void *user);

/* Per-thread GPU leaf counters (same meaning as the psph.c statistics, see ctx_t there), plus pipeline data:
   batches run, prefixes and table bytes uploaded, full-check cells, filter launches (> batches when a survivor
   list overflowed and the filter was re-run), and GPU-event / host-wall times. */
typedef struct {
    unsigned long long cand, xpass, rpass, s2pass, fullfail, direct, probes, sols;
    unsigned long long fhist[18];              /* full-check failure position, blocks below the top block */
    unsigned long long batches, prefixes, tab_bytes, fc_cells, filter_runs;
    double t_h2d, t_filter, t_full, t_host;    /* seconds: GPU events (h2d, kernel 1, kernel 2), host wall of the pipeline */
} gpu_stats_t;

/*
 * gpu_init -- prepare the GPU leaf for one search.
 *
 * one call per search (run_search); allocations persist across calls and are released by gpu_shutdown().
 * Inputs:  h, k, tgt  the search (tgt must be < 2^31 - 16 in GPU mode);  nthreads  number of psph workers;
 *          ndeep_cap  maximum number of deep-hole targets accepted per prefix;  report, user  the callback.
 * Outputs: Returns 0 on success; resets all per-thread counters and batches.  The first call also reads the
 *          PSPH_GPU_* environment knobs and prints the device line.
 * Errors:  returns 1 if tgt is too large for the 32-bit candidate indexing; CUDA failures exit(7).
 */
int  gpu_init(int h, int k, int64_t tgt, int nthreads, int ndeep_cap, gpu_report_fn report, void *user);

/*
 * gpu_submit_prefix -- append one (k-1)-prefix and its candidate range to thread tid's batch.
 *
 * append one (k-1)-prefix to thread tid's batch.  T must be valid for indices 0..E (E = min(TGT, h*a_{k-1})),
 * candidates are a_k in [lo, hi] (lo <= hi), a[1..k-1] is the prefix, Z the prefix part of the Challis
 * target, reach[0..h] the reach sequence r_m = n_m(A_{k-1}), deep_y[0..ndeep) the deep-hole remainders
 * (positions y in [1,hi) with large T[y]; any set is exact, only y < a_k are used per candidate).
 * Outputs: the data are copied (the caller may reuse its buffers at once).  If the batch is full it is run first
 *          (so solutions of EARLIER prefixes may be reported from inside this call); a candidate range larger
 *          than the per-batch cap is split into several submissions.
 * Errors:  CUDA or allocation failures exit(7); a PSPH_GPU_PARANOID mismatch exits 8.  lo > hi is a no-op.
 */
void gpu_submit_prefix(int tid, const gpu_cell_t *T, int64_t E, int64_t lo, int64_t hi,
                       const int64_t *a, int64_t Z, const int64_t *reach,
                       const int64_t *deep_y, int ndeep);

/*
 * gpu_flush -- run the pending batch of thread tid (if any) and report its solutions.
 *
 * Inputs: tid.   Outputs: callback calls for accepted bases; counters updated.   Errors: as gpu_submit_prefix.
 */
void gpu_flush(int tid);

/*
 * gpu_thread_stats -- copy the counters of thread tid accumulated since gpu_init.
 *
 * statistics of thread tid accumulated since gpu_init (safe to call from the progress thread: counters are
 * updated once per finished batch, so they may lag by a batch).  An unknown tid yields all zeros.
 */
void gpu_thread_stats(int tid, gpu_stats_t *out);

/*
 * gpu_finish -- run every thread's pending batch, then synchronize the device (call after the workers joined).
 * gpu_print_stats -- print the GPU summary (batches, uploads, GPU times, throughput), summed over threads, to f.
 *
 * Errors: CUDA failures exit(7).
 */
void gpu_finish(void);
void gpu_print_stats(FILE *f);

/*
 * gpu_shutdown -- release all device and pinned host memory, streams and events (end of program).
 *
 * Errors: none reported; if the CUDA runtime is already torn down the buffers are simply abandoned.
 */
void gpu_shutdown(void);

#ifdef __cplusplus
}
#endif
#endif
