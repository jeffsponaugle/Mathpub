/*
 * a053686.c  --  compute and extend OEIS A053686
 *
 *   "Record gaps between consecutive primes that repeat at least once before
 *    a new record occurs."
 *
 * Status (October 2026)
 *   a(1..12) = 2, 4, 6, 14, 34, 36, 52, 86, 132, 154, 250, 336 (in the OEIS
 *              before 2026; all reproduced by 'selftest').
 *   a(13)    = 906, found with this program in Sep 2026: the record gap 906
 *              first occurs after 218209405436543 (A133788(13)) and again
 *              after 543684371469023, before the next record, 916 after
 *              1189459969825483.  In the OEIS since Sep 17 2026, together with
 *              A133788(13) and A085237(80..86).
 *   a(14)   >= 1132: no record between 336 and 1132 other than 906 repeats
 *              (records 35..63 decided).  Whether 1132 repeats needs every gap
 *              below A002386(65) = 43841547845541059, about 6 days on a Mac
 *              Studio.
 *   README.md has the runs, submission.md what is left to submit, and
 *   verify_a053686.py is an independent Python check.
 *
 * ==========================================================================
 * THEORY OF OPERATION
 * ==========================================================================
 *
 * 1. Definitions and the reduction to record intervals
 *
 *    For consecutive primes p < q, q - p is "the gap after p" (A001223).  The
 *    record (maximal) gaps g_1 = 1 < g_2 = 2 < g_3 = 4 < g_4 = 6 < ...
 *    (A005250) are the gaps larger than every earlier gap; g_n first occurs
 *    after the prime P_n (A002386: P_1 = 2, P_2 = 3, P_3 = 7, P_4 = 23, ...).
 *    Call
 *
 *        I_n = [P_n, P_(n+1))
 *
 *    the n-th RECORD INTERVAL.  The intervals partition [2, infinity), and
 *    every gap after a prime p of I_n is at most g_n, because the first gap
 *    larger than g_n is g_(n+1), after P_(n+1).  Since P_(n+1) is prime, the
 *    gap after any p < P_(n+1) ends at P_(n+1) at the latest.  So "g_n
 *    repeats before a new record occurs" means exactly "the gap after some
 *    prime p of I_n other than P_n equals g_n":
 *
 *        g_n is a term of A053686
 *          <=>  nextprime(p) - p = g_n for some prime p, P_n < p < P_(n+1)
 *          <=>  g_n occurs at least twice in A085237 (nondecreasing gaps).
 *
 *    Example: g_34 = 336 first occurs after P_34 = 3842610773 and again after
 *    4275912661 (the next prime is 4275912997), before P_35 = 4302407359, so
 *    336 = a(12).  The terms so far are the records n = 2, 3, 4, 6, 10, 11,
 *    13, 15, 20, 22, 29, 34, 61 (KNOWN_N[]).
 *
 *    Each record is therefore a yes/no question about one finite interval,
 *    and one pass over the primes of [START, END) decides every interval it
 *    covers completely.  ("No" needs the whole interval; a "yes" found in a
 *    part of an interval is final as well.)
 *
 * 2. The table of maximal gaps, and why it need not be trusted
 *
 *    All maximal gaps below 2^64 are known from exhaustive searches.  REC[]
 *    holds those 80 records, copied from the OEIS b-files of A005250 and
 *    A002386 (the last is 1550 after 18361375334787046697).  The program uses
 *    the table to know where each interval starts and what its record is,
 *    and it re-checks both inside the range it scans:
 *      - the gap after each P_n in the range must be exactly g_n (a RECORD
 *        event; a different gap is an ANOMALY, and a P_n that is never seen
 *        is reported as MISSING in the summary);
 *      - no gap after a prime of I_n may exceed g_n (else ANOMALY: a maximal
 *        gap missing from the table, a wrong entry, or a sieve error).
 *    A scan that reports neither has re-proved the part of the table it
 *    covered.  'selftest' also checks that the table is increasing and that
 *    every row is a prime gap of the listed size, with a Miller-Rabin
 *    next-prime search that shares no code with the sieve.
 *
 * 3. Work split: chunks and the ownership rule
 *
 *    [START, END) is cut into CHUNKS of CHUNK numbers (default 10^10; the
 *    last chunk may be shorter).  Chunk c is [grid0 + c*CHUNK, grid0 +
 *    (c+1)*CHUNK) clipped to END, where grid0 is START, or the position a
 *    resumed run continues from (section 7).  Worker threads take chunk
 *    numbers from an atomic counter (S.next), so chunks are started roughly
 *    in increasing order.
 *
 *    OWNERSHIP RULE: the gap (p, q) belongs to the chunk that contains its
 *    smaller prime p.  A worker starts at the first prime >= lo of its chunk
 *    [lo, hi) and reads primes until it has the first prime q >= hi, so it
 *    handles the gap that crosses hi; the gap that crosses lo belongs to the
 *    previous chunk.  Every gap after a prime of [START, END) is therefore
 *    examined exactly once, for any chunk size, thread count or resume
 *    position, and no chunk needs anything from before its lo.  ('selftest'
 *    compares five chunk/thread layouts with a brute-force pass over an
 *    array of primes.)  A chunk without primes (possible only for a short
 *    last chunk) has no gaps of its own.
 *
 *    Each worker owns a primesieve iterator and reads the primes straight
 *    out of the iterator's internal buffer (primesieve_generate_next_primes
 *    refills it with about a thousand primes at a time) instead of making one
 *    call per prime.  The iterator is told to expect primes up to
 *    hi + LOOKAHEAD; the largest gap below 2^64 is 1550, so the crossing gap
 *    never needs more, and the iterator would go past the hint anyway.
 *
 * 4. Which gaps get a closer look
 *
 *    Nearly every gap is tiny next to the record of its interval.  Only these
 *    matter (n = the interval of p):
 *      g = g_n after p = P_n     the record itself            RECORD event
 *      g = g_n after p > P_n     a repeat: g_n is a term      REPEAT event
 *      g > g_n                   impossible if the table      ANOMALY event
 *                                is right
 *      g_n - 198 <= g < g_n      a near miss                  histogram; also
 *                                                             a NEAR event if
 *                                                             g_n - g <= NEAR
 *    A worker computes one threshold per chunk, thr = g_n0 - 198 where n0 is
 *    the interval of the chunk's first number, and calls big_gap() only for
 *    gaps >= thr.  Records grow along the chunk, so for any later interval n
 *    every gap that matters (>= g_n - 198 >= thr) still gets through, and
 *    big_gap() finds the interval of p and ignores gaps too small for it.  In
 *    intervals with g_n <= 198 (below P_24 = 20831323) thr is 1 and every gap
 *    goes through big_gap(); a chunk that starts there keeps thr = 1 to its
 *    end, which costs a little time in the first chunk only.
 *
 *    hist[i][k] counts the gaps of size REC[i].g - 2k (k = 0 .. HK-1) after
 *    the primes of interval i other than P_i itself: hist[i][0] is the number
 *    of repeats of record i, k >= 1 are near misses.  Every gap after a prime
 *    > 3 is even, and so is every record after the first, so the deficit 2k
 *    is exact.  The histogram feeds the summary (runner-up gap, gaps within 20
 *    and 100 of the record), the -H "hist" lines and tail_a053686.py.
 *
 * 5. Per-chunk results and the frontier
 *
 *    A worker gathers its chunk's results privately (cstate_t: a histogram,
 *    the events in increasing order of p, the prime count), compacts them
 *    into the ring slot S.slot[c % RINGC], and then FOLDS completed chunks at
 *    the FRONTIER: strictly in the order 0, 1, 2, ..., so the totals always
 *    describe the contiguous range [START, frontier position) and the event
 *    lines come out in increasing order of p.  A chunk finished early waits
 *    in its slot until every chunk before it is done.  Workers stay less than
 *    RINGC - 1 chunks ahead of the frontier, so a slot is free when reused.
 *
 *    Folding a chunk adds its prime count and histogram to the totals and
 *    replays its events:
 *      REPEAT   remember the position (the first REPMAX per record), print it;
 *      RECORD   mark record i as confirmed, print it, and announce record i-1
 *               as DECIDED (or PARTIAL if START was inside interval i-1).
 *               Every repeat of record i-1 lies before P_i, i.e. in this or
 *               an earlier chunk, and this chunk's earlier events have been
 *               replayed already, so the decision is final at this point;
 *      ANOMALY  count it and print it (the run will exit with status 1);
 *      NEAR     print it.
 *
 * 6. Threads, locking and stopping
 *
 *    One mutex, g_mu, protects every field of S that changes during a scan
 *    (frontier, slots, totals, histogram, repeat list, confirmed records) and
 *    also keeps the event lines on stdout in order.  S.next is atomic.  The
 *    fields that describe the scan (start, end, chunk, nchunks, grid0) and
 *    the option globals (g_near, g_quiet_events, ...) are set before the
 *    workers start and only read while they run.
 *
 *    The main thread does no sieving.  It wakes five times a second, draws
 *    the status line on stderr (on a terminal; otherwise it prints a
 *    "progress:" line every SECS seconds), writes the checkpoint every SECS
 *    seconds, and leaves its loop when the frontier reaches the end or g_stop
 *    is set.  Ctrl-C and SIGTERM only set g_stop: the workers take no new
 *    chunks and finish the ones they have (a few seconds each at the default
 *    chunk size), the completed prefix is folded, the checkpoint is written,
 *    and the summary of what was scanned is printed.
 *
 * 7. Checkpoint and resume (-S FILE)
 *
 *    The checkpoint is a small text file, rewritten every SECS seconds and at
 *    the end of the run.  It is written to FILE.tmp, checked for write errors,
 *    synced to disk and only then renamed over FILE, so FILE always holds a
 *    complete checkpoint (if anything fails, FILE keeps the previous one):
 *
 *        A053686 checkpoint 2     format tag
 *        start S                  START of the run; a resume must use the same
 *        pos P                    [S, P) has been scanned completely
 *        chunk C                  informational: the chunk size may change
 *        end E                    informational: END may change
 *        primes N                 primes in [S, P)
 *        anomalies A              anomalies seen in [S, P)
 *        lastrec R                highest record (1-based) confirmed
 *        rec I                    record I (0-based) confirmed, one line each
 *        rep I P                  a repeat of record I after the prime P
 *                                 (at most REPMAX lines per record)
 *        hist I K C               hist[I][K] = C (non-zero entries only)
 *        end                      completeness marker
 *
 *    (The -H output lines "hist N GAP COUNT" are a different format: 1-based
 *    N and the gap itself.)  When FILE exists, the run must have the same
 *    START; it lays out its chunks from P (grid0 = P) and adds to the stored
 *    totals.  END and CHUNK may differ from the earlier run.  END = P resumes
 *    nothing and just prints the summary of the stored state; END < P is
 *    refused.  When FILE does not exist the run starts at START and creates
 *    it.  Every line must have exactly one of the forms above, so a damaged
 *    line is refused rather than misread.
 *
 * 8. Integrity checks and exit status
 *
 *    The summary ends with the number of record gaps in the range and whether
 *    all were confirmed where A002386 lists them (else MISSING lines), the
 *    number of anomalies, and the prime count, which is compared with
 *    pi(10^k) - pi(START-1) when the scan ends at a power of ten (k = 6..18,
 *    START <= 10^14).  'scan' exits with status 1 if a record is missing, an
 *    anomaly occurred, the prime count is wrong, or the decision for a record
 *    the OEIS has decided (n <= OEIS_DECIDED) disagrees with it; else 0.
 *    'selftest' (make test) checks the number parser, the next-prime search
 *    and Miller-Rabin, the table, five chunk/thread layouts against a
 *    brute-force pass on [0, 2*10^8), and every decision below LIMIT (default
 *    10^11: records 1..39) against the OEIS.  'verify P G' rechecks one gap
 *    without the sieve (and with GMP when compiled in).
 *
 * 9. Speed
 *
 *    The sieve is the whole cost.  With 10 threads on an M1 Pro the scan runs
 *    at about 3e10 numbers/s up to 10^13 and 2.1e10/s near 10^15; a Mac Studio
 *    did 8e10/s.  Deciding records 35..63 (a scan from 4302407359 to
 *    1693182318746372) took about six hours on the Mac Studio.  'bench'
 *    measures the rate and projects the time to each record the OEIS has not
 *    decided.
 *
 * 10. Errors and limits
 *
 *    - Fatal conditions (a bad number or option value, out of memory, a
 *      primesieve error, a checkpoint that is unreadable, damaged, truncated or
 *      for another START) print a message on stderr and exit with status 1
 *      via die(); an unknown command or option prints the usage and exits with
 *      status 2.  die() may run on a worker thread: the process ends at once,
 *      and the checkpoint on disk is the last one completely written, so a
 *      rerun resumes from there.
 *    - Numbers are 64-bit.  END must be at most 2^64 - 1 - 2*LOOKAHEAD
 *      (primesieve's limit minus the look-ahead); the table ends at 2^64.
 *    - At most REPMAX repeat positions are kept per record (the count is
 *      exact).  At most EVMAX events are kept per chunk: beyond that near-miss
 *      events are dropped (a "note" line says how many), but repeats, records
 *      and anomalies never are; more than EVMAX of those in one chunk could
 *      only come from a wrong table or sieve and is fatal.
 *    - Known limitations: a checkpoint line that is missing as a whole cannot
 *      be detected when loading (save_state's error checks are what prevent
 *      writing one).  A second Ctrl-C does not force an exit.  g_stop is never
 *      reset, so after a Ctrl-C in 'selftest' the remaining checks fail.
 *
 * Usage
 * -----
 *   a053686 scan [START] END [-t T] [-c CHUNK] [-s KIB] [-n NEAR] [-S FILE] [-i SECS] [-H] [-q]
 *       Examine every gap (p, q) with START <= p < END (START defaults to 0).
 *       Prints RECORD / REPEAT / DECIDED lines as they become final, then a
 *       summary table.  -n NEAR also prints gaps within NEAR (<= 198) of the
 *       record ("near" lines).  -H appends the histogram as "hist N GAP COUNT"
 *       lines.  -S FILE keeps a checkpoint (every SECS seconds, default 60,
 *       and at the end, also after Ctrl-C); rerunning with the same START
 *       resumes from it, and END may be larger than before.  -t T threads
 *       (default: all cores), -c CHUNK chunk size (default 10^10, at least
 *       10^6), -s KIB primesieve sieve size (16..8192, default: the library's
 *       choice), -q no status line and no start-up notes.
 *   a053686 verify P G
 *       Check with deterministic Miller-Rabin (and GMP when compiled in) that
 *       P and P+G are consecutive primes, and say which record interval the
 *       gap is in and whether it repeats that record.
 *   a053686 bench [N] [-d SPAN] [-t T] [-c CHUNK] [-s KIB]
 *       Time a scan of [N, N+SPAN) (defaults 10^15 and 10^11) and project how
 *       long a scan to each undecided record would take at that rate.
 *   a053686 selftest [LIMIT] [-t T]
 *       The checks of section 8 (about 10 seconds with LIMIT = 10^11).
 *
 * Numbers may be written as decimal, B^E, MeE (M may have a decimal point:
 * 1.7e15), X+Y or X-Y of two such terms, or rN on its own for the prime
 * A002386(N) where record N starts (r35 = 4302407359).
 *
 * Build
 * -----
 *   make        (= cc -O2 -std=gnu11 -Wall -Wextra -pthread -I/opt/homebrew/include
 *                    a053686.c -L/opt/homebrew/lib -lprimesieve -lm [-DHAVE_GMP -lgmp];
 *                the Makefile adds GMP when gmp.h is installed)
 *   make test   (= ./a053686 selftest)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdarg.h>
#include <ctype.h>
#include <math.h>
#include <signal.h>
#include <errno.h>
#include <fcntl.h>                  /* F_FULLFSYNC (macOS) for the checkpoint */
#include <limits.h>                 /* PATH_MAX */
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>
#include <time.h>
#include <inttypes.h>
#include <primesieve.h>             /* primesieve 12.x: iterator, counts, prime arrays */
#ifdef HAVE_GMP
#include <gmp.h>                    /* optional second opinion in 'verify' */
#endif

typedef uint64_t u64;
typedef unsigned __int128 u128;     /* GCC/Clang extension, for exact 64x64-bit products */

/* Sizes and limits (THEORY OF OPERATION, sections 4, 5 and 10). */
#define NREC      80        /* number of maximal gaps below 2^64 (rows of REC[]) */
#define HK        100       /* histogram depth: gaps g_n - 2k for k = 0..HK-1, i.e. down to 198 below the record */
#define REPMAX    16        /* repeat positions kept per record (the count itself, hist[i][0], is exact) */
#define RINGC     4096      /* chunk-completion ring slots; workers stay < RINGC-1 chunks ahead of the frontier */
#define EVMAX     4096      /* events buffered per chunk; only near-miss events are ever dropped */
#define LOOKAHEAD 65536     /* sieve hint past a chunk end, for the gap that crosses it (largest gap < 2^64: 1550) */
#define NKNOWN    13        /* terms of A053686 in the OEIS (KNOWN_N[]) */

/*
 * REC[]: the maximal prime gaps below 2^64, from the OEIS b-files of A005250
 * (the gap) and A002386 (the prime it follows).  Row i (0-based) is the
 * record n = i + 1: the gap between the prime REC[i].p = A002386(n) and the
 * next prime is REC[i].g = A005250(n), and it is larger than every gap between
 * smaller primes.  Both columns increase.  Record interval i is
 * [REC[i].p, REC[i+1].p); the last one runs to 2^64.  Beyond 2^64 the known
 * records continue 1552 (after 18470057946260698231), 1572, 1676, 1724, 1854.
 * 'selftest' checks the table; every scan re-checks the rows in its range.
 */
static const struct { unsigned g; u64 p; } REC[NREC] = {
    {    1,                     2ULL},   /*  1 */
    {    2,                     3ULL},   /*  2 */
    {    4,                     7ULL},   /*  3 */
    {    6,                    23ULL},   /*  4 */
    {    8,                    89ULL},   /*  5 */
    {   14,                   113ULL},   /*  6 */
    {   18,                   523ULL},   /*  7 */
    {   20,                   887ULL},   /*  8 */
    {   22,                  1129ULL},   /*  9 */
    {   34,                  1327ULL},   /* 10 */
    {   36,                  9551ULL},   /* 11 */
    {   44,                 15683ULL},   /* 12 */
    {   52,                 19609ULL},   /* 13 */
    {   72,                 31397ULL},   /* 14 */
    {   86,                155921ULL},   /* 15 */
    {   96,                360653ULL},   /* 16 */
    {  112,                370261ULL},   /* 17 */
    {  114,                492113ULL},   /* 18 */
    {  118,               1349533ULL},   /* 19 */
    {  132,               1357201ULL},   /* 20 */
    {  148,               2010733ULL},   /* 21 */
    {  154,               4652353ULL},   /* 22 */
    {  180,              17051707ULL},   /* 23 */
    {  210,              20831323ULL},   /* 24 */
    {  220,              47326693ULL},   /* 25 */
    {  222,             122164747ULL},   /* 26 */
    {  234,             189695659ULL},   /* 27 */
    {  248,             191912783ULL},   /* 28 */
    {  250,             387096133ULL},   /* 29 */
    {  282,             436273009ULL},   /* 30 */
    {  288,            1294268491ULL},   /* 31 */
    {  292,            1453168141ULL},   /* 32 */
    {  320,            2300942549ULL},   /* 33 */
    {  336,            3842610773ULL},   /* 34 */
    {  354,            4302407359ULL},   /* 35 */
    {  382,           10726904659ULL},   /* 36 */
    {  384,           20678048297ULL},   /* 37 */
    {  394,           22367084959ULL},   /* 38 */
    {  456,           25056082087ULL},   /* 39 */
    {  464,           42652618343ULL},   /* 40 */
    {  468,          127976334671ULL},   /* 41 */
    {  474,          182226896239ULL},   /* 42 */
    {  486,          241160624143ULL},   /* 43 */
    {  490,          297501075799ULL},   /* 44 */
    {  500,          303371455241ULL},   /* 45 */
    {  514,          304599508537ULL},   /* 46 */
    {  516,          416608695821ULL},   /* 47 */
    {  532,          461690510011ULL},   /* 48 */
    {  534,          614487453523ULL},   /* 49 */
    {  540,          738832927927ULL},   /* 50 */
    {  582,         1346294310749ULL},   /* 51 */
    {  588,         1408695493609ULL},   /* 52 */
    {  602,         1968188556461ULL},   /* 53 */
    {  652,         2614941710599ULL},   /* 54 */
    {  674,         7177162611713ULL},   /* 55 */
    {  716,        13829048559701ULL},   /* 56 */
    {  766,        19581334192423ULL},   /* 57 */
    {  778,        42842283925351ULL},   /* 58 */
    {  804,        90874329411493ULL},   /* 59 */
    {  806,       171231342420521ULL},   /* 60 */
    {  906,       218209405436543ULL},   /* 61 */
    {  916,      1189459969825483ULL},   /* 62 */
    {  924,      1686994940955803ULL},   /* 63 */
    { 1132,      1693182318746371ULL},   /* 64 */
    { 1184,     43841547845541059ULL},   /* 65 */
    { 1198,     55350776431903243ULL},   /* 66 */
    { 1220,     80873624627234849ULL},   /* 67 */
    { 1224,    203986478517455989ULL},   /* 68 */
    { 1248,    218034721194214273ULL},   /* 69 */
    { 1272,    305405826521087869ULL},   /* 70 */
    { 1328,    352521223451364323ULL},   /* 71 */
    { 1356,    401429925999153707ULL},   /* 72 */
    { 1370,    418032645936712127ULL},   /* 73 */
    { 1442,    804212830686677669ULL},   /* 74 */
    { 1476,   1425172824437699411ULL},   /* 75 */
    { 1488,   5733241593241196731ULL},   /* 76 */
    { 1510,   6787988999657777797ULL},   /* 77 */
    { 1526,  15570628755536096243ULL},   /* 78 */
    { 1530,  17678654157568189057ULL},   /* 79 */
    { 1550,  18361375334787046697ULL},   /* 80 */
};

/*
 * The terms of A053686 in the OEIS, as record numbers: a(k) = REC[KNOWN_N[k-1] - 1].g,
 * e.g. KNOWN_N[12] = 61 gives a(13) = REC[60].g = 906.  Every record n <=
 * OEIS_DECIDED is decided in the OEIS (a(13) = 906 and "a(14) >= 1132", A053686
 * revision #28, Sep 17 2026); decisions about those records are compared with
 * KNOWN_N, decisions about later ones are reported as new.
 */
static const int KNOWN_N[NKNOWN] = { 2, 3, 4, 6, 10, 11, 13, 15, 20, 22, 29, 34, 61 };
#define OEIS_DECIDED 63     /* records 1..63 are decided in the OEIS (A053686 rev. #28, Sep 17 2026) */

static bool stderr_tty;                     /* stderr is a terminal: draw a one-line status, erase it before other output */
static volatile sig_atomic_t g_stop = 0;    /* set by SIGINT/SIGTERM: stop handing out chunks (never reset) */

/* ------------------------------------------------------------------ */
/* Generic helpers                                                     */
/* ------------------------------------------------------------------ */

/*
 * now: monotonic wall-clock time, for rates, ETAs and checkpoint intervals.
 *   In:     nothing.
 *   Out:    seconds since an arbitrary fixed point (CLOCK_MONOTONIC), as a double.
 *   Errors: none (CLOCK_MONOTONIC cannot fail on the supported systems).
 */
static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

/*
 * die: report a fatal error and end the process.
 *   In:     printf-style format and arguments (without a trailing newline).
 *   Out:    does not return: erases the status line if stderr is a terminal,
 *           prints the message and a newline on stderr and exits with status 1.
 *   Errors: none.  Also called on worker threads (out of memory, a primesieve
 *           error, an event overflow): exit() then ends every thread at once,
 *           and the checkpoint on disk is the last one completely written.
 */
static void die(const char *fmt, ...)
{
    va_list ap;
    if (stderr_tty) fputs("\r\033[K", stderr);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(1);
}

/*
 * fmt_dur: format a duration for the status line and reports.
 *   In:     s = seconds; buf, n = the caller's buffer and its size (32 bytes is
 *           plenty).
 *   Out:    buf, holding "12.3s", "4m05s", "2h03m04s" or "3d04h05m"; returns buf.
 *   Errors: none.  A negative value, NaN or more than 1e9 s gives "?" (used for
 *           an unknown ETA); longer text than n would be cut by snprintf.
 */
static const char *fmt_dur(double s, char *buf, size_t n)
{
    if (s < 0 || s != s || s > 1e9) { snprintf(buf, n, "?"); return buf; }
    long t = (long)s;
    if (t >= 86400) snprintf(buf, n, "%ldd%02ldh%02ldm", t / 86400, (t / 3600) % 24, (t / 60) % 60);
    else if (t >= 3600) snprintf(buf, n, "%ldh%02ldm%02lds", t / 3600, (t / 60) % 60, t % 60);
    else if (t >= 60) snprintf(buf, n, "%ldm%02lds", t / 60, t % 60);
    else snprintf(buf, n, "%.1fs", s);
    return buf;
}

/*
 * mul_ok: overflow-checked 64-bit multiplication.
 *   In:     a, b = factors; r = where to store the product.
 *   Out:    true and *r = a*b if the product fits in 64 bits; false (*r
 *           unchanged) otherwise.
 *   Errors: none.
 */
static bool mul_ok(u64 a, u64 b, u64 *r)
{
    u128 x = (u128)a * b;
    if (x >> 64) return false;
    *r = (u64)x;
    return true;
}

/*
 * all_digits: is s[0..n) a non-empty run of decimal digits?
 *   In:     s = text (need not be NUL-terminated); n = its length.
 *   Out:    true iff n > 0 and every character is '0'..'9' (no sign, no blank).
 *   Errors: none.
 */
static bool all_digits(const char *s, size_t n)
{
    if (n == 0) return false;
    for (size_t i = 0; i < n; i++)
        if (!isdigit((unsigned char)s[i])) return false;
    return true;
}

/*
 * dec_u64: convert a string of decimal digits to a 64-bit number.
 *   In:     s = NUL-terminated text; out = where to store the value.
 *   Out:    true and *out = the value if s is one or more digits (leading zeros
 *           are fine) and the value is at most 2^64 - 1; false (*out unchanged)
 *           for anything else, including an empty string, a sign or a blank.
 *   Errors: none beyond the false return.
 */
static bool dec_u64(const char *s, u64 *out)
{
    if (!all_digits(s, strlen(s))) return false;
    errno = 0;
    unsigned long long v = strtoull(s, NULL, 10);
    if (errno == ERANGE) return false;
    *out = (u64)v;
    return true;
}

/*
 * parse_term: read one number term (no '+' or '-').
 *   In:     s[0..n) = the term (need not be NUL-terminated; at most 63
 *           characters); out = where to store the value.  Accepted forms:
 *             D      decimal digits                         1234567
 *             B^E    a power; B and E decimal               2^40, 10^15
 *             MeE    M * 10^E, E decimal, M decimal with    1e12, 1.7e15
 *                    at most one '.' ('E' works too)
 *   Out:    true and *out = the value; false for anything else.  0^0 = 1.
 *           0^E, 1^E and 0eE are computed without a loop (E may be huge), and
 *           the other loops stop as soon as the value passes 2^64, so no input
 *           makes them run long.  Trailing zeros after the '.' do not count
 *           (1.50e1 = 15).
 *   Errors: false if a part is not plain digits, a value (also B or E itself)
 *           does not fit in 64 bits, or MeE is not an integer (1.5e0).
 */
static bool parse_term(const char *s, size_t n, u64 *out)
{
    char buf[64], *c;
    if (n == 0 || n >= sizeof buf) return false;
    memcpy(buf, s, n);
    buf[n] = 0;
    if ((c = strchr(buf, '^'))) {                   /* B^E */
        *c = 0;
        u64 b, e, r = 1;
        if (!dec_u64(buf, &b) || !dec_u64(c + 1, &e)) return false;
        if (b <= 1) {                               /* 0^0 = 1, 0^E = 0, 1^E = 1 */
            *out = (b == 1 || e == 0) ? 1 : 0;
            return true;
        }
        while (e--)                                 /* b >= 2: overflows within 64 steps */
            if (!mul_ok(r, b, &r)) return false;
        *out = r;
        return true;
    }
    if ((c = strpbrk(buf, "eE"))) {                 /* MeE */
        *c = 0;
        u64 e, m;
        if (!dec_u64(c + 1, &e)) return false;
        char mant[64];                              /* the digits of M without the '.' */
        int md = 0, frac = -1;                      /* md = digits, frac = digits after the '.' (-1: none) */
        for (const char *q = buf; *q; q++) {
            if (*q == '.') { if (frac >= 0) return false; frac = 0; continue; }
            if (!isdigit((unsigned char)*q)) return false;
            mant[md++] = *q;
            if (frac >= 0) frac++;
        }
        while (frac > 0 && md > 1 && mant[md - 1] == '0') { md--; frac--; }   /* 1.50e1 -> 1.5e1 */
        if (md == 0) return false;
        mant[md] = 0;
        if (!dec_u64(mant, &m)) return false;       /* M without its '.', i.e. M * 10^frac */
        if (frac > 0) {
            if (e < (u64)frac) return false;        /* not an integer */
            e -= (u64)frac;
        }
        if (m == 0) { *out = 0; return true; }
        while (e--)                                 /* m >= 1: overflows within 20 steps */
            if (!mul_ok(m, 10, &m)) return false;
        *out = m;
        return true;
    }
    return dec_u64(buf, out);                       /* D */
}

/*
 * parse_num: read a number from the command line.
 *   In:     s = NUL-terminated text: rN (N = 1..NREC, giving the prime
 *           REC[N-1].p = A002386(N) where record N starts: r35 = 4302407359),
 *           or one term (see parse_term), or X+Y or X-Y of two terms (split at
 *           the last '+' or '-', so 2^40+1e6 works but 1+2+3 and r35+1 do not).
 *   Out:    the value.
 *   Errors: die() for anything parse_term rejects, a record number outside
 *           1..NREC, a sum above 2^64 - 1 and a negative difference.
 */
static u64 parse_num(const char *s)
{
    size_t n = strlen(s);
    if ((s[0] == 'r' || s[0] == 'R') && all_digits(s + 1, n - 1)) {   /* rN = start prime of record N */
        u64 k = strtoull(s + 1, 0, 10);
        if (k < 1 || k > NREC) die("record number in '%s' must be 1..%d", s, NREC);
        return REC[k - 1].p;
    }
    for (size_t i = n; i-- > 1; ) {           /* split at the last + or - (not at position 0) */
        if (s[i] != '+' && s[i] != '-') continue;
        u64 a, b;
        if (!parse_term(s, i, &a) || !parse_term(s + i + 1, n - i - 1, &b)) die("bad number '%s'", s);
        if (s[i] == '+') {
            if (a + b < a) die("overflow in '%s'", s);
            return a + b;
        }
        if (b > a) die("negative result in '%s'", s);
        return a - b;
    }
    u64 v;
    if (!parse_term(s, n, &v)) die("bad number '%s'", s);
    return v;
}

/*
 * parse_int: read a small integer option (threads, seconds, KiB, NEAR).
 *   In:     s = text as for parse_num; lo..hi = the allowed range; what = the
 *           option's name for the error message.
 *   Out:    the value as an int.
 *   Errors: die() if s is not a number or the value is outside lo..hi.
 */
static int parse_int(const char *s, int lo, int hi, const char *what)
{
    u64 v = parse_num(s);
    if (v < (u64)lo || v > (u64)hi) die("%s must be between %d and %d", what, lo, hi);
    return (int)v;
}

/* ------------------------------------------------------------------ */
/* Independent primality: deterministic Miller-Rabin below 2^64         */
/* (used by 'verify' and 'selftest' only, never by the scan)            */
/* ------------------------------------------------------------------ */

/*
 * mulmod: a*b mod m without overflow, through a 128-bit product.
 *   In:     a, b, m with m > 0.
 *   Out:    (a*b) mod m.
 *   Errors: m = 0 would divide by zero; callers never pass it.
 */
static u64 mulmod(u64 a, u64 b, u64 m) { return (u64)(((u128)a * b) % m); }

/*
 * powmod: a^e mod m by binary exponentiation.
 *   In:     a, e, m with m > 1.
 *   Out:    a^e mod m.
 *   Errors: none for m > 1 (m = 0 would divide by zero).
 */
static u64 powmod(u64 a, u64 e, u64 m)
{
    u64 r = 1;
    a %= m;
    while (e) {
        if (e & 1) r = mulmod(r, a, m);
        a = mulmod(a, a, m);
        e >>= 1;
    }
    return r;
}

/*
 * is_prime64: deterministic primality test for any 64-bit n.
 *   In:     n.
 *   Out:    true iff n is prime.  Trial division by the primes up to 37, then
 *           strong-probable-prime (Miller-Rabin) tests to the seven bases of
 *           Jim Sinclair's set, which has no strong pseudoprime below 2^64.  A
 *           base that is 0 mod n carries no information and is skipped.
 *   Errors: none.  Shares no code with primesieve, so it is an independent
 *           check of the sieve.
 */
static bool is_prime64(u64 n)
{
    static const u64 small[] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 };
    static const u64 bases[] = { 2, 325, 9375, 28178, 450775, 9780504, 1795265022 };
    if (n < 2) return false;
    for (size_t i = 0; i < sizeof small / sizeof *small; i++)
        if (n % small[i] == 0) return n == small[i];
    u64 d = n - 1;                                  /* n - 1 = d * 2^r with d odd */
    int r = 0;
    while (!(d & 1)) { d >>= 1; r++; }
    for (size_t i = 0; i < sizeof bases / sizeof *bases; i++) {
        u64 a = bases[i] % n;
        if (a == 0) continue;
        u64 x = powmod(a, d, n);
        if (x == 1 || x == n - 1) continue;
        bool composite = true;
        for (int j = 1; j < r && composite; j++) {  /* square up to r-1 times, looking for -1 */
            x = mulmod(x, x, n);
            if (x == n - 1) composite = false;
        }
        if (composite) return false;
    }
    return true;
}

#define LARGEST_PRIME64 18446744073709551557ULL      /* 2^64 - 59 */

/*
 * next_prime64: the smallest prime above n, found by testing odd numbers with
 * is_prime64 (independent of the sieve).
 *   In:     n.
 *   Out:    the next prime after n (2 for n < 2), or 0 if there is none below
 *           2^64, i.e. for n >= LARGEST_PRIME64.
 *   Errors: none.  The loop cannot run past 2^64: LARGEST_PRIME64 stops it.
 *           Cost: about ln(n)/2 primality tests, some microseconds.
 */
static u64 next_prime64(u64 n)
{
    if (n < 2) return 2;
    if (n >= LARGEST_PRIME64) return 0;               /* no larger prime fits in 64 bits */
    u64 m = n + 1;
    if (!(m & 1)) m++;
    while (!is_prime64(m)) m += 2;
    return m;
}

/* ------------------------------------------------------------------ */
/* Record table helpers                                                */
/* ------------------------------------------------------------------ */

/*
 * rec_index: which record interval contains x.
 *   In:     x.
 *   Out:    the largest i with REC[i].p <= x, i.e. the 0-based index of the
 *           interval [REC[i].p, REC[i+1].p) containing x; 0 for x < 3 (also
 *           for x = 0 and 1, which lie before every interval).
 *   Errors: none (binary search over the NREC rows).
 */
static int rec_index(u64 x)
{
    int lo = 0, hi = NREC - 1;
    while (lo < hi) {                               /* invariant: REC[lo].p <= x or lo == 0; the answer is <= hi */
        int mid = (lo + hi + 1) / 2;
        if (REC[mid].p <= x) lo = mid; else hi = mid - 1;
    }
    return lo;
}

/*
 * rec_end: where record interval i ends.
 *   In:     i = 0-based record index, 0 <= i < NREC (not checked).
 *   Out:    REC[i+1].p, the start of the next interval; UINT64_MAX for the last
 *           interval, whose next record lies beyond 2^64.
 *   Errors: none for a valid i.
 */
static u64 rec_end(int i) { return i + 1 < NREC ? REC[i + 1].p : UINT64_MAX; }

/*
 * known_term_index: is record n a known term of A053686?
 *   In:     n = 1-based record number.
 *   Out:    k if g_n = a(k) in the OEIS (KNOWN_N[k-1] == n), else 0.
 *   Errors: none.
 */
static int known_term_index(int n)
{
    for (int k = 0; k < NKNOWN; k++)
        if (KNOWN_N[k] == n) return k + 1;
    return 0;
}

/*
 * oeis_tag: describe how a finding about record n compares with the OEIS, for
 * the bracketed tag on REPEAT / DECIDED lines and the status in the summary.
 *   In:     n = 1-based record number; repeats = whether g_n repeats (for a
 *           partly scanned interval: whether a repeat was seen).
 *   Out:    a static string: "= OEIS a(k)" or "as in OEIS" for agreement and
 *           "DIFFERS FROM OEIS (...)" for disagreement when n <= OEIS_DECIDED;
 *           "NEW" or "new information, undecided in OEIS" for later records.
 *   Errors: none.  The buffer is static, so the result must be used before the
 *           next call; the callers hold g_mu or run on the main thread only.
 */
static const char *oeis_tag(int n, bool repeats)
{
    static char buf[96];
    if (n <= OEIS_DECIDED) {
        int k = known_term_index(n);
        if (repeats && k) snprintf(buf, sizeof buf, "= OEIS a(%d)", k);
        else if (!repeats && !k) snprintf(buf, sizeof buf, "as in OEIS");
        else snprintf(buf, sizeof buf, "DIFFERS FROM OEIS (%s)", k ? "listed as a term" : "not listed");
    } else {
        snprintf(buf, sizeof buf, repeats ? "NEW" : "new information, undecided in OEIS");
    }
    return buf;
}

/* ------------------------------------------------------------------ */
/* Scan engine                                                         */
/* ------------------------------------------------------------------ */

/* Event types (THEORY OF OPERATION, section 4). */
enum { EV_REPEAT, EV_RECORD, EV_ANOMALY, EV_NEAR };

/* One event: its type, the 0-based record interval i, the prime p and the gap g after it. */
typedef struct { unsigned char type, i; u64 p, g; } event_t;
/* One non-zero histogram cell of a chunk: hist[i][k] += c. */
typedef struct { unsigned char i, k; u64 c; } hent_t;

/* A completed chunk's results, waiting in the ring S.slot[] until the frontier folds them. */
typedef struct {
    bool done;                      /* complete and not folded yet */
    u64 primes;                     /* primes in the chunk */
    unsigned nh, nev, dropped;      /* cells in h[], events in ev[], near-miss events dropped */
    hent_t *h;                      /* compacted histogram (malloc'd; freed when folded) */
    event_t *ev;                    /* events in increasing order of p (malloc'd; freed when folded) */
} slot_t;

/*
 * S: the scan in progress (one at a time).  The first group is set by
 * reset_state()/load_state() before the workers start and only read while they
 * run.  The second group changes during the scan and is protected by g_mu; its
 * totals cover [start, frontier position), including the part done by earlier
 * runs that this one resumed.
 */
static struct {
    u64 start, end, chunk, nchunks; /* the scan is [start, end), nchunks chunks of `chunk` numbers from grid0 */
    u64 grid0;                      /* chunk i of this run is [grid0 + i*chunk, ...): start, or the resumed position */
    _Atomic u64 next;               /* next chunk index to hand out */
    /* protected by g_mu: */
    u64 frontier;                   /* chunks [0, frontier) of this run are complete and folded */
    slot_t slot[RINGC];             /* completed chunks waiting to be folded, at chunk % RINGC */
    u64 primes;                     /* primes in [start, frontier position) */
    u64 hist[NREC][HK];             /* hist[i][k] = gaps of size REC[i].g - 2k after primes in interval i (not at REC[i].p) */
    u64 rep[NREC][REPMAX];          /* the first repeat positions, ascending */
    unsigned nrep[NREC];            /* entries in rep[i] (the number of repeats itself is hist[i][0]) */
    bool recseen[NREC];             /* the gap after REC[i].p was found and equals REC[i].g */
    u64 anomalies;                  /* gaps that contradict the table */
    int last_rec;                   /* highest 1-based record confirmed so far, 0 = none */
} S;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;   /* guards S's second group and the order of stdout lines */
static int g_sieve_kib = 0;         /* -s: primesieve sieve size in KiB, 0 = library default */
static unsigned g_near = 0;         /* -n: also report gaps within this of the record (0..198) */
static bool g_hist_lines = false;   /* -H: print the histogram as "hist N GAP COUNT" lines */
static bool g_quiet_events = false;  /* bookkeeping only, no REPEAT/RECORD/DECIDED lines (bench, selftest) */

/* A worker's private results for the chunk it is scanning (reused for every chunk). */
typedef struct {
    int ci;                         /* record interval containing the current p (only moves forward) */
    u64 (*lhist)[HK];               /* NREC x HK histogram of this chunk, same layout as S.hist */
    event_t *ev;                    /* up to EVMAX events of this chunk, in increasing order of p */
    unsigned nev, dropped;          /* events stored; near-miss events dropped */
} cstate_t;

/* One worker thread. */
typedef struct {
    int id;                         /* 0 .. threads-1 (informational) */
    primesieve_iterator it;         /* this thread's own iterator */
    cstate_t c;
} worker_t;

/*
 * add_event: append an event to the chunk's event list.
 *   In:     c = the worker's chunk state; type = EV_*; i = 0-based record
 *           interval; p = the prime; g = the gap after it.  Calls come in
 *           increasing order of p, so the list stays sorted.
 *   Out:    the event is stored at c->ev[c->nev++].  When the list is full
 *           (EVMAX), a NEAR event is dropped instead (c->dropped++); any other
 *           event makes room by deleting the most recent NEAR event and moving
 *           the later events down one place, so repeats, records and anomalies
 *           are never lost and the order is kept.
 *   Errors: die() if the list is full of events other than NEAR (more than
 *           EVMAX records, repeats and anomalies in one chunk), which a correct
 *           table and sieve cannot produce: there are 80 records, and repeats
 *           are rare.
 */
static void add_event(cstate_t *c, int type, int i, u64 p, u64 g)
{
    if (c->nev >= EVMAX) {
        if (type == EV_NEAR) { c->dropped++; return; }
        /* never lose a repeat, record or anomaly: delete the most recent near miss
         * and move the events after it down one place, keeping their order */
        unsigned j = c->nev;
        while (j > 0 && c->ev[j - 1].type != EV_NEAR) j--;
        if (j == 0) die("more than %d record/repeat/anomaly events in one chunk: the record table or the sieve is wrong", EVMAX);
        memmove(&c->ev[j - 1], &c->ev[j], (c->nev - j) * sizeof *c->ev);
        c->nev--;
        c->dropped++;
    }
    event_t *e = &c->ev[c->nev++];
    e->type = (unsigned char)type;
    e->i = (unsigned char)i;
    e->p = p;
    e->g = g;
}

/*
 * big_gap: classify a gap that passed the chunk's threshold (section 4).
 *   In:     c = the worker's chunk state (c->ci = the interval of the previous
 *           big gap or of the chunk start, <= the interval of p); p = a prime;
 *           g = the gap after it.
 *   Out:    c->ci moves forward to the interval of p; then, with G = REC[ci].g:
 *             p = REC[ci].p     RECORD if g = G, else ANOMALY;
 *             g > G             ANOMALY;
 *             otherwise         hist[ci][(G-g)/2] += 1 if (G-g)/2 < HK, and
 *                               REPEAT if g = G, NEAR if G - g <= g_near.
 *   Errors: none here; add_event() may die (see there).  Inline: it sits in
 *           the hot loop, though only a tiny fraction of gaps reach it.
 */
static inline void big_gap(cstate_t *c, u64 p, u64 g)
{
    while (c->ci + 1 < NREC && p >= REC[c->ci + 1].p) c->ci++;
    const int ci = c->ci;
    const unsigned G = REC[ci].g;
    if (p == REC[ci].p) {                       /* the record itself */
        add_event(c, g == G ? EV_RECORD : EV_ANOMALY, ci, p, g);
        return;
    }
    if (g > G) {                                /* larger than the record of its interval */
        add_event(c, EV_ANOMALY, ci, p, g);
        return;
    }
    const u64 k = (G - g) >> 1;                 /* both even: ci >= 1 here (interval 0 is just p = 2) */
    if (k < HK) c->lhist[ci][k]++;
    if (g == G) add_event(c, EV_REPEAT, ci, p, g);
    else if (G - g <= g_near) add_event(c, EV_NEAR, ci, p, g);
}

static void fold_locked(void);

/*
 * scan_chunk: examine every gap after a prime of chunk idx and hand the
 * results to the frontier (sections 3 to 5).
 *   In:     w = the calling worker (its iterator and private buffers); idx =
 *           the chunk index, 0 <= idx < S.nchunks.  The chunk is [lo, hi) =
 *           [S.grid0 + idx*S.chunk, + S.chunk), clipped to S.end.  The caller
 *           (worker) has made sure that ring slot idx % RINGC is free.
 *   Out:    the chunk's prime count, compacted histogram and events are put in
 *           S.slot[idx % RINGC], marked done, and fold_locked() folds every
 *           chunk that is now at the frontier, printing their events.
 *   Errors: die() on a primesieve error or when out of memory.
 */
static void scan_chunk(worker_t *w, u64 idx)
{
    const u64 lo = S.grid0 + idx * S.chunk;         /* no overflow: idx < nchunks (see chunk_count) */
    u64 hi = lo + S.chunk;
    if (hi > S.end || hi < lo) hi = S.end;          /* the last chunk, or lo + chunk beyond 2^64 */
    cstate_t *c = &w->c;

    memset(c->lhist, 0, sizeof(u64) * NREC * HK);
    c->nev = c->dropped = 0;
    c->ci = rec_index(lo);
    /* Gaps below thr cannot matter anywhere in this chunk (section 4): every
     * later interval has a larger record and so a higher threshold. */
    const u64 thr = REC[c->ci].g > 2 * (HK - 1) ? REC[c->ci].g - 2 * (HK - 1) : 1;
    u64 count = 0;                                  /* primes in [lo, hi) */

    primesieve_jump_to(&w->it, lo, hi + LOOKAHEAD);
    /* The primes are read straight out of the iterator's buffer, which
     * primesieve_generate_next_primes() fills with primes[0..size): the primes
     * >= lo after the jump, then the primes after the last one.  (Mixing in
     * primesieve_next_prime() would discard the rest of a buffer.) */
    primesieve_generate_next_primes(&w->it);
    if (w->it.is_error || w->it.size == 0) die("primesieve error in chunk %" PRIu64, idx);
    const u64 *P = w->it.primes;
    size_t np = w->it.size, i = 1;
    u64 prev = P[0];                                /* the first prime >= lo */
    if (prev >= hi) goto finish;                    /* no prime in this chunk: the gap over it belongs to the previous one */
    count = 1;
    for (;;) {
        /* the hot loop: one subtraction and one compare per prime */
        for (; i < np; i++) {
            const u64 q = P[i];
            const u64 g = q - prev;                 /* the gap after prev, a prime of this chunk */
            if (g >= thr) big_gap(c, prev, g);
            if (q >= hi) goto finish;               /* the gap crossing hi is done; q belongs to the next chunk */
            count++;
            prev = q;
        }
        primesieve_generate_next_primes(&w->it);    /* buffer used up: the next ~1000 primes */
        if (w->it.is_error || w->it.size == 0) die("primesieve error in chunk %" PRIu64, idx);
        P = w->it.primes;
        np = w->it.size;
        i = 0;
    }

finish:;
    /* Compact the histogram to its non-zero cells and copy the events, so the
     * worker's own buffers are free for its next chunk at once. */
    unsigned nh = 0;
    for (int a = 0; a < NREC; a++)
        for (int k = 0; k < HK; k++)
            if (c->lhist[a][k]) nh++;
    hent_t *h = nh ? malloc(nh * sizeof *h) : NULL;
    event_t *ev = c->nev ? malloc(c->nev * sizeof *ev) : NULL;
    if ((nh && !h) || (c->nev && !ev)) die("out of memory");
    for (int a = 0, j = 0; a < NREC; a++)
        for (int k = 0; k < HK; k++)
            if (c->lhist[a][k]) { h[j].i = (unsigned char)a; h[j].k = (unsigned char)k; h[j].c = c->lhist[a][k]; j++; }
    if (c->nev) memcpy(ev, c->ev, c->nev * sizeof *ev);

    /* publish the result and fold whatever is now contiguous with the frontier */
    pthread_mutex_lock(&g_mu);
    slot_t *s = &S.slot[idx % RINGC];
    s->done = true;
    s->primes = count;
    s->nh = nh;
    s->h = h;
    s->nev = c->nev;
    s->ev = ev;
    s->dropped = c->dropped;
    fold_locked();
    pthread_mutex_unlock(&g_mu);
}

/*
 * clear_line: erase the status line so the next output starts on a clean line.
 *   In/Out: nothing; writes "\r\033[K" to stderr if it is a terminal.
 *   Errors: none.
 */
static void clear_line(void) { if (stderr_tty) fputs("\r\033[K", stderr); }

/*
 * announce_decision_locked: print the outcome for record interval i once the
 * scan has passed its end (called when the RECORD event of interval i+1 is
 * folded).
 *   In:     i = 0-based record interval; upto = the position the scan has
 *           reached (REC[i+1].p).  The caller holds g_mu.
 *   Out:    a "DECIDED" line if the whole interval [REC[i].p, REC[i+1].p) was
 *           scanned (START <= REC[i].p): TERM with the number of repeats and
 *           the first one, or "not a term"; a "PARTIAL" line if START was
 *           inside the interval.  Both end with oeis_tag().  Nothing at all
 *           when g_quiet_events is set.
 *   Errors: none.
 */
static void announce_decision_locked(int i, u64 upto)
{
    const int n = i + 1;
    const u64 reps = S.hist[i][0];
    const bool full = S.start <= REC[i].p && upto >= rec_end(i);
    if (g_quiet_events) return;
    clear_line();
    if (full) {
        if (reps)
            printf("DECIDED  #%-2d gap %-5u repeats %" PRIu64 " time%s before the next record, first after %" PRIu64
                   "  -> TERM  [%s]\n", n, REC[i].g, reps, reps == 1 ? "" : "s", S.rep[i][0], oeis_tag(n, true));
        else
            printf("DECIDED  #%-2d gap %-5u no second occurrence in (%" PRIu64 ", %" PRIu64 ")  -> not a term  [%s]\n",
                   n, REC[i].g, REC[i].p, rec_end(i), oeis_tag(n, false));
    } else {
        u64 from = S.start > REC[i].p ? S.start : REC[i].p, to = upto < rec_end(i) ? upto : rec_end(i);
        if (reps)
            printf("PARTIAL  #%-2d gap %-5u repeats %" PRIu64 " time%s in the scanned part [%" PRIu64 ", %" PRIu64
                   "), first after %" PRIu64 "  -> TERM  [%s]\n", n, REC[i].g, reps, reps == 1 ? "" : "s", from, to,
                   S.rep[i][0], oeis_tag(n, true));
        else
            printf("PARTIAL  #%-2d gap %-5u no second occurrence in the scanned part [%" PRIu64 ", %" PRIu64 ") of (%" PRIu64
                   ", %" PRIu64 ")\n", n, REC[i].g, from, to, REC[i].p, rec_end(i));
    }
    fflush(stdout);
}

/*
 * handle_event_locked: replay one event of a chunk being folded (section 5).
 *   In:     e = the event.  The caller holds g_mu; events arrive in increasing
 *           order of p.
 *   Out:    REPEAT   the position is added to S.rep (the first REPMAX) and a
 *                    REPEAT line is printed;
 *           RECORD   S.recseen and S.last_rec are updated, a RECORD line is
 *                    printed and the previous interval's decision announced;
 *           ANOMALY  S.anomalies++ and an ANOMALY line, printed even with
 *                    g_quiet_events because it makes the run invalid;
 *           NEAR     a "near" line.
 *           g_quiet_events suppresses all lines but ANOMALY.
 *   Errors: none.
 */
static void handle_event_locked(const event_t *e)
{
    const int i = e->i, n = i + 1;
    switch (e->type) {
    case EV_REPEAT:
        if (S.nrep[i] < REPMAX) S.rep[i][S.nrep[i]++] = e->p;
        if (g_quiet_events) return;
        clear_line();
        printf("REPEAT   #%-2d gap %-5u after %" PRIu64 " (next prime %" PRIu64 "); first occurrence after %" PRIu64
               "  [%s]\n", n, REC[i].g, e->p, e->p + e->g, REC[i].p, oeis_tag(n, true));
        break;
    case EV_RECORD:
        S.recseen[i] = true;
        if (n > S.last_rec) S.last_rec = n;
        if (g_quiet_events) return;
        clear_line();
        printf("RECORD   #%-2d gap %-5u after %" PRIu64 " confirmed where A002386 lists it\n", n, REC[i].g, e->p);
        if (i >= 1 && S.start < REC[i].p) announce_decision_locked(i - 1, REC[i].p);   /* interval i-1 ends here */
        break;
    case EV_ANOMALY:
        S.anomalies++;
        clear_line();
        if (e->p == REC[i].p)
            printf("ANOMALY  gap after %" PRIu64 " is %" PRIu64 " but A005250 says record #%d is %u\n", e->p, e->g, n, REC[i].g);
        else
            printf("ANOMALY  gap %" PRIu64 " after %" PRIu64 " exceeds the record %u (#%d) of its interval: unlisted maximal gap or sieve error\n",
                   e->g, e->p, REC[i].g, n);
        break;
    case EV_NEAR:
        if (g_quiet_events) return;
        clear_line();
        printf("near     #%-2d gap %-5" PRIu64 " after %" PRIu64 " (record %u, short by %" PRIu64 ")\n", n, e->g, e->p, REC[i].g, REC[i].g - e->g);
        break;
    }
    fflush(stdout);
}

/*
 * fold_locked: fold every completed chunk at the frontier into the totals,
 * strictly in chunk order (section 5).
 *   In:     nothing; the caller holds g_mu.
 *   Out:    for each done slot S.slot[frontier % RINGC]: its primes and
 *           histogram are added to S, its events are replayed, a "note" line
 *           reports dropped near misses, its buffers are freed and the frontier
 *           advances.  Stops at the first chunk that is not done yet.
 *   Errors: none.
 */
static void fold_locked(void)
{
    while (S.frontier < S.nchunks && S.slot[S.frontier % RINGC].done) {
        slot_t *s = &S.slot[S.frontier % RINGC];
        s->done = false;
        S.primes += s->primes;
        for (unsigned j = 0; j < s->nh; j++) S.hist[s->h[j].i][s->h[j].k] += s->h[j].c;   /* before the events, which read it */
        for (unsigned j = 0; j < s->nev; j++) handle_event_locked(&s->ev[j]);
        if (s->dropped) {
            clear_line();
            printf("note     %u near-miss lines dropped in chunk %" PRIu64 " (lower -n)\n", s->dropped, S.frontier);
        }
        free(s->h);
        free(s->ev);
        s->h = NULL;
        s->ev = NULL;
        s->nh = s->nev = s->dropped = 0;
        S.frontier++;
    }
}

/*
 * worker: thread body.  Takes chunk numbers from S.next until they run out or
 * g_stop is set, and scans each with scan_chunk().
 *   In:     arg = this thread's worker_t.
 *   Out:    NULL.  Before scanning chunk idx it waits (polling every 2 ms)
 *           until idx < frontier + RINGC - 1, i.e. until chunk idx - RINGC has
 *           been folded and slot idx % RINGC is free.  This cannot deadlock:
 *           the chunk at the frontier is always held by a worker that does not
 *           wait.  A chunk in progress is always completed, even after g_stop.
 *   Errors: die() if its buffers cannot be allocated (and via scan_chunk()).
 */
static void *worker(void *arg)
{
    worker_t *w = arg;
    primesieve_init(&w->it);
    w->c.lhist = malloc(sizeof(u64) * NREC * HK);
    w->c.ev = malloc(sizeof(event_t) * EVMAX);
    if (!w->c.lhist || !w->c.ev) die("out of memory");
    while (!g_stop) {
        u64 idx = atomic_fetch_add(&S.next, 1);
        if (idx >= S.nchunks) break;
        for (;;) {                                   /* stay within RINGC of the frontier */
            pthread_mutex_lock(&g_mu);
            u64 f = S.frontier;
            pthread_mutex_unlock(&g_mu);
            if (idx < f + RINGC - 1 || g_stop) break;
            usleep(2000);
        }
        if (g_stop) break;
        scan_chunk(w, idx);
    }
    free(w->c.lhist);
    free(w->c.ev);
    primesieve_free_iterator(&w->it);
    return NULL;
}

/*
 * chunk_count: the number of chunks of `chunk` numbers that cover [from, to).
 *   In:     from <= to; chunk > 0.
 *   Out:    ceil((to - from) / chunk), computed without overflow (0 for an
 *           empty range).  Then (count - 1) * chunk < to - from, so no chunk
 *           start from + i*chunk with i < count can overflow.
 *   Errors: none.
 */
static u64 chunk_count(u64 from, u64 to, u64 chunk)
{
    const u64 span = to - from;
    return span / chunk + (span % chunk != 0);
}

/*
 * frontier_pos: the position up to which the scan is complete.
 *   In:     nothing (reads S; hold g_mu while workers run).
 *   Out:    S.grid0 + S.frontier * S.chunk, or S.end once every chunk of this
 *           run has been folded (the last chunk may be short).
 *   Errors: none; no overflow (see chunk_count).
 */
static u64 frontier_pos(void)
{
    if (S.frontier >= S.nchunks) return S.end;
    return S.grid0 + S.frontier * S.chunk;
}

/* ---- checkpoint file ---- */

/*
 * sync_file: push a file's data to stable storage.
 *   In:     fd = an open file descriptor.
 *   Out:    0 on success, -1 with errno set on failure.  On macOS F_FULLFSYNC
 *           also flushes the drive's write cache, which plain fsync() there
 *           does not; elsewhere, or if F_FULLFSYNC is refused, fsync().
 *   Errors: those of fcntl() and fsync().
 */
static int sync_file(int fd)
{
#ifdef F_FULLFSYNC
    if (fcntl(fd, F_FULLFSYNC) == 0) return 0;
#endif
    return fsync(fd);
}

/*
 * save_state: write the checkpoint (format in section 7) so that it replaces
 * the old one in a single step.
 *   In:     path = the -S file.  Called by the main thread every SECS seconds
 *           while the workers run, and once at the end; it holds g_mu while it
 *           copies S into the file buffer, so the workers wait a few ms.
 *   Out:    path.tmp is written, flushed, checked for write errors, synced to
 *           disk and renamed over path, so path always holds a complete
 *           checkpoint, the old one or the new one.
 *   Errors: if path.tmp cannot be created, or a write (e.g. on a full disk), the
 *           sync, the close or the rename fails, a warning goes to stderr,
 *           path.tmp is removed and path keeps the previous checkpoint; the scan
 *           goes on.  (A stdio write error can silently drop part of a line, so
 *           the error flag must be checked before the file is trusted.)  A path
 *           longer than PATH_MAX - 5 characters is not supported.
 */
static void save_state(const char *path)
{
    char tmp[PATH_MAX];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "w");
    if (!f) { fprintf(stderr, "\nwarning: cannot write %s: %s\n", tmp, strerror(errno)); return; }
    pthread_mutex_lock(&g_mu);
    u64 pos = frontier_pos();
    fprintf(f, "A053686 checkpoint 2\nstart %" PRIu64 "\npos %" PRIu64 "\nchunk %" PRIu64 "\nend %" PRIu64
               "\nprimes %" PRIu64 "\nanomalies %" PRIu64 "\nlastrec %d\n",
            S.start, pos, S.chunk, S.end, S.primes, S.anomalies, S.last_rec);
    for (int i = 0; i < NREC; i++)
        if (S.recseen[i]) fprintf(f, "rec %d\n", i);
    for (int i = 0; i < NREC; i++)
        for (unsigned j = 0; j < S.nrep[i]; j++) fprintf(f, "rep %d %" PRIu64 "\n", i, S.rep[i][j]);
    for (int i = 0; i < NREC; i++)
        for (int k = 0; k < HK; k++)
            if (S.hist[i][k]) fprintf(f, "hist %d %d %" PRIu64 "\n", i, k, S.hist[i][k]);
    pthread_mutex_unlock(&g_mu);
    fputs("end\n", f);
    const char *why = NULL;                         /* the first thing that went wrong */
    if (fflush(f) != 0) why = strerror(errno);
    else if (ferror(f)) why = "write error";
    else if (sync_file(fileno(f)) != 0) why = strerror(errno);
    if (fclose(f) != 0 && !why) why = strerror(errno);
    if (!why && rename(tmp, path) != 0) why = strerror(errno);
    if (why) {
        fprintf(stderr, "\nwarning: cannot update %s (%s); the previous checkpoint is unchanged\n", path, why);
        remove(tmp);
    }
}

/*
 * load_state: read the checkpoint for the scan that reset_state() just set up.
 *   In:     path = the -S file.  S.start, S.end and S.chunk describe the new run.
 *   Out:    false if path does not exist (a new scan).  Otherwise true, with
 *           the stored totals, histogram, repeats and confirmed records in S
 *           and the chunks laid out from the stored position: grid0 = pos,
 *           nchunks = the chunks that cover [pos, END), frontier = 0.
 *   Errors: die() if the file cannot be read or is not a format-2 checkpoint,
 *           if a line is not exactly one of the known forms (e.g. damaged by a
 *           write error), if the final "end" or the position is missing, if it
 *           was written for another START, or if it already covers more than
 *           END.  A line that is missing as a whole cannot be detected here;
 *           save_state's error checks are what prevent writing such a file.
 */
static bool load_state(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        if (errno == ENOENT) return false;
        die("cannot read %s: %s", path, strerror(errno));
    }
    char line[256];
    if (!fgets(line, sizeof line, f) || strncmp(line, "A053686 checkpoint 2", 20) != 0)
        die("%s is not an A053686 checkpoint (format 2)", path);
    u64 start = 0, pos = 0, primes = 0, anomalies = 0;
    int lastrec = 0, lineno = 1;
    bool complete = false, have_pos = false;
    while (fgets(line, sizeof line, f)) {
        int i, k, used = 0;
        u64 v;
        lineno++;
        /* Each form must match the whole line: the format ends in " %n", which
         * skips the newline and stores how many characters were used, and the
         * line must end there.  When sscanf returns the full count the " %n"
         * has been reached, so `used` is valid. */
        if (sscanf(line, "start %" SCNu64 " %n", &start, &used) == 1 && !line[used]) continue;
        if (sscanf(line, "pos %" SCNu64 " %n", &pos, &used) == 1 && !line[used]) { have_pos = true; continue; }
        if (sscanf(line, "chunk %" SCNu64 " %n", &v, &used) == 1 && !line[used]) continue;    /* informational */
        if (sscanf(line, "end %" SCNu64 " %n", &v, &used) == 1 && !line[used]) continue;      /* informational */
        if (sscanf(line, "primes %" SCNu64 " %n", &primes, &used) == 1 && !line[used]) continue;
        if (sscanf(line, "anomalies %" SCNu64 " %n", &anomalies, &used) == 1 && !line[used]) continue;
        if (sscanf(line, "lastrec %d %n", &lastrec, &used) == 1 && !line[used]) continue;
        if (sscanf(line, "rec %d %n", &i, &used) == 1 && !line[used]) {
            if (i < 0 || i >= NREC) die("%s: bad rec line", path);
            S.recseen[i] = true;
            continue;
        }
        if (sscanf(line, "rep %d %" SCNu64 " %n", &i, &v, &used) == 2 && !line[used]) {
            if (i < 0 || i >= NREC) die("%s: bad rep line", path);
            if (S.nrep[i] < REPMAX) S.rep[i][S.nrep[i]++] = v;
            continue;
        }
        if (sscanf(line, "hist %d %d %" SCNu64 " %n", &i, &k, &v, &used) == 3 && !line[used]) {
            if (i < 0 || i >= NREC || k < 0 || k >= HK) die("%s: bad hist line", path);
            S.hist[i][k] = v;
            continue;
        }
        if (!strcmp(line, "end\n") || !strcmp(line, "end")) { complete = true; break; }   /* the final marker */
        line[strcspn(line, "\n")] = 0;
        die("%s is damaged: line %d is '%.60s'", path, lineno, line);
    }
    fclose(f);
    if (!complete || !have_pos) die("%s is truncated", path);
    if (start != S.start)
        die("%s was written for START=%" PRIu64 ", not %" PRIu64, path, start, S.start);
    if (pos < S.start) die("%s: bad position", path);
    if (pos > S.end)
        die("%s already covers [%" PRIu64 ", %" PRIu64 "), beyond END=%" PRIu64, path, start, pos, S.end);
    /* lay out this run's chunks from the resumed position (END and CHUNK may differ from the earlier run) */
    S.grid0 = pos;
    S.nchunks = chunk_count(pos, S.end, S.chunk);
    S.frontier = 0;
    S.primes = primes;
    S.anomalies = anomalies;
    S.last_rec = lastrec;
    return true;
}

/*
 * on_sigint: the SIGINT/SIGTERM handler.
 *   In:     sig (unused).
 *   Out:    sets g_stop; the workers and the main loop notice it (section 6).
 *   Errors: none; async-signal-safe (it only writes a volatile sig_atomic_t).
 */
static void on_sigint(int sig)
{
    (void)sig;
    g_stop = 1;
}

/* What run_scan() reports about one run (one session of a resumable scan). */
typedef struct {
    double seconds;                 /* wall time of this run */
    u64 primes;                     /* primes in [start, frontier position), earlier runs included */
    u64 numbers;                    /* numbers scanned by this run */
} scan_stats_t;

/*
 * reset_state: set up S for a new scan of [start, end) in chunks of `chunk`.
 *   In:     start < end; chunk > 0.
 *   Out:    S describes the range (grid0 = start, nchunks chunks covering it,
 *           the last one possibly short); totals, histogram, repeats, confirmed
 *           records and ring slots are cleared (buffers left in the slots by an
 *           interrupted earlier scan are freed).
 *   Errors: die() if end <= start, chunk = 0, or end is too close to 2^64
 *           (above primesieve's limit minus 2*LOOKAHEAD).
 */
static void reset_state(u64 start, u64 end, u64 chunk)
{
    if (end <= start) die("END must be larger than START");
    if (chunk == 0) die("CHUNK must be positive");
    S.start = start;
    S.grid0 = start;
    S.chunk = chunk;
    S.nchunks = chunk_count(start, end, chunk);         /* the last chunk may be short */
    S.end = end;
    if (S.end > primesieve_get_max_stop() - 2 * LOOKAHEAD) die("END beyond primesieve's limit");
    S.frontier = 0;
    S.primes = 0;
    S.anomalies = 0;
    S.last_rec = 0;
    for (u64 i = 0; i < RINGC; i++) { free(S.slot[i].h); free(S.slot[i].ev); }
    memset(S.slot, 0, sizeof S.slot);
    memset(S.hist, 0, sizeof S.hist);
    memset(S.rep, 0, sizeof S.rep);
    memset(S.nrep, 0, sizeof S.nrep);
    memset(S.recseen, 0, sizeof S.recseen);
}

/*
 * run_scan: scan [start, end) with `threads` workers (sections 3 to 7) and
 * leave the results in S.
 *   In:     start < end; chunk = chunk size (callers enforce >= 10^6); threads
 *           >= 1; state = checkpoint path or NULL; interval = seconds between
 *           checkpoints and between "progress:" lines when stderr is not a
 *           terminal; quiet = no status line and no start-up notes (the event
 *           lines are controlled by g_quiet_events instead).
 *   Out:    S holds the totals for [start, frontier_pos()), which is end unless
 *           the scan was stopped by Ctrl-C or SIGTERM.  Returns this run's wall
 *           time, the total prime count and the numbers scanned by this run (a
 *           resumed run does not count the earlier part).
 *   Errors: die() via reset_state()/load_state() (a bad range or checkpoint) or
 *           if a thread cannot be created.  The SIGINT/SIGTERM handlers it
 *           installs stay installed afterwards.
 */
static scan_stats_t run_scan(u64 start, u64 end, u64 chunk, int threads, const char *state, int interval, bool quiet)
{
    reset_state(start, end, chunk);
    if (start > 0 && start != REC[rec_index(start)].p && !quiet)
        fprintf(stderr, "note: START is inside record interval #%d (gap %u after %" PRIu64 "), which will only be partially scanned\n",
                rec_index(start) + 1, REC[rec_index(start)].g, REC[rec_index(start)].p);

    if (state && load_state(state))
        fprintf(stderr, "resuming from %s: [%" PRIu64 ", %" PRIu64 ") done, %" PRIu64 " primes examined; %" PRIu64
                        " chunks to go\n", state, S.start, S.grid0, S.primes, S.nchunks);
    atomic_store(&S.next, S.frontier);

    if (!quiet)
        fprintf(stderr, "scanning gaps after the primes in [%" PRIu64 ", %" PRIu64 ") in %" PRIu64 " chunks of %" PRIu64
                        " with %d threads\n", S.start, S.end, S.nchunks, S.chunk, threads);

    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_sigint;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    worker_t *ws = calloc((size_t)threads, sizeof *ws);
    pthread_t *th = calloc((size_t)threads, sizeof *th);
    if (!ws || !th) die("out of memory");
    for (int i = 0; i < threads; i++) {
        ws[i].id = i;
        if (pthread_create(&th[i], NULL, worker, &ws[i]) != 0) die("pthread_create failed");
    }

    /* rate window: one sample of the frontier position per second, ~2 minutes deep */
    enum { NS = 128 };
    double ts[NS];
    u64 ps[NS];
    int ns = 0;
    const double t0 = now();
    const u64 pos0 = frontier_pos();
    double last_status = 0, last_ckpt = t0, last_sample = 0;
    ts[0] = t0; ps[0] = pos0; ns = 1;
    char b1[32], b2[32];

    /* The main thread's loop: status, checkpoints, and the end of the scan (section 6). */
    for (;;) {
        usleep(200000);
        double t = now();
        pthread_mutex_lock(&g_mu);
        u64 f = S.frontier, primes = S.primes, pos = frontier_pos(), reps = 0;
        int lastrec = S.last_rec;
        for (int i = 0; i < NREC; i++) reps += S.hist[i][0];
        pthread_mutex_unlock(&g_mu);
        if (f >= S.nchunks || g_stop) break;

        if (t - last_sample >= 1.0) {
            last_sample = t;
            if (ns < NS) { ts[ns] = t; ps[ns] = pos; ns++; }
            else { memmove(ts, ts + 1, (NS - 1) * sizeof *ts); memmove(ps, ps + 1, (NS - 1) * sizeof *ps); ts[NS - 1] = t; ps[NS - 1] = pos; }
        }
        double rate = (t - ts[0] > 0.5) ? (double)(pos - ps[0]) / (t - ts[0]) : 0;
        double eta = rate > 0 ? (double)(S.end - pos) / rate : -1;
        bool show = stderr_tty ? (t - last_status >= 1.0) : (t - last_status >= interval);
        if (show && !quiet) {
            last_status = t;
            fprintf(stderr, "%s%s  pos %.6g (%.2f%%)  %.3g/s  %.3g primes  ETA %s  records to #%d  repeats %" PRIu64,
                    stderr_tty ? "\r\033[K" : "progress: ",
                    fmt_dur(t - t0, b1, sizeof b1), (double)pos,
                    100.0 * (double)(pos - S.start) / (double)(S.end - S.start),
                    rate, (double)primes, fmt_dur(eta, b2, sizeof b2), lastrec, reps);
            if (!stderr_tty) fputc('\n', stderr);
            fflush(stderr);
        }
        if (state && t - last_ckpt >= interval) {
            last_ckpt = t;
            save_state(state);
        }
    }
    for (int i = 0; i < threads; i++) pthread_join(th[i], NULL);   /* workers finish their chunks; all are folded */
    double t1 = now();
    if (stderr_tty && !quiet) fputs("\r\033[K", stderr);

    if (state) save_state(state);
    if (g_stop && S.frontier < S.nchunks && !quiet)
        fprintf(stderr, "stopped at position %" PRIu64 " (%" PRIu64 " of %" PRIu64 " chunks complete)%s\n",
                frontier_pos(), S.frontier, S.nchunks, state ? ", checkpoint saved" : "");

    scan_stats_t st;
    st.seconds = t1 - t0;
    st.primes = S.primes;
    st.numbers = frontier_pos() - pos0;
    free(ws);
    free(th);
    return st;
}

/* pi(10^k) for k = 6..18 (A006880), to check the prime count of a scan that ends at 10^k */
static const u64 PI10[19] = {
    0, 0, 0, 0, 0, 0, 78498ULL, 664579ULL, 5761455ULL, 50847534ULL, 455052511ULL,
    4118054813ULL, 37607912018ULL, 346065536839ULL, 3204941750802ULL,
    29844570422669ULL, 279238341033925ULL, 2623557157654233ULL, 24739954287740860ULL
};

/*
 * print_summary: the table printed at the end of 'scan'.  It is built from S,
 * so it covers everything scanned since START, earlier resumed runs included.
 *   In:     nothing (S after run_scan()).
 *   Out:    on stdout, one row per record interval that overlaps [START,
 *           frontier position): n, gap, P_n, P_(n+1), repeats, the runner-up
 *           (the largest gap below the record) and how often it occurs, the
 *           gaps within 20 and within 100 of the record, and the status (TERM
 *           or not a term with the OEIS comparison, or "undecided" for a partly
 *           scanned interval), each followed by its repeat positions; then
 *           MISSING lines, the record/anomaly line, the prime count with the
 *           pi(10^k) comparison and, with -H, the "hist N GAP COUNT" lines.
 *           Returns the number of failures: disagreements with the OEIS
 *           (records <= OEIS_DECIDED), missing records, anomalies (counted
 *           once) and a wrong prime count.
 *   Errors: none.  pi(START-1) for the prime-count check is computed with
 *           primesieve only when START <= 10^14.
 */
static int print_summary(void)
{
    const u64 pos = frontier_pos();
    int fails = 0;
    printf("\nRecord intervals touched by the scan [%" PRIu64 ", %" PRIu64 "):\n", S.start, pos);
    printf("  n   gap   first after            next record after      repeats   runner-up gap (count)  gaps within 20 / 100 of record   status\n");
    for (int i = 0; i < NREC; i++) {
        const u64 lo = REC[i].p, hi = rec_end(i);
        if (hi <= S.start || lo >= pos) continue;   /* interval does not overlap the scan */
        const bool full = S.start <= lo && pos >= hi;
        const u64 reps = S.hist[i][0];
        u64 w20 = 0, w100 = 0;
        int krun = 0;                               /* smallest k >= 1 with a count: the runner-up gap is G - 2*krun */
        for (int k = 1; k < HK; k++) {
            if (S.hist[i][k] && !krun) krun = k;
            if (k <= 10) w20 += S.hist[i][k];
            if (k <= 50) w100 += S.hist[i][k];
        }
        char run[40], nxt[24], status[160];
        if (krun) snprintf(run, sizeof run, "%u (%" PRIu64 ")", REC[i].g - 2 * krun, S.hist[i][krun]);
        else if ((int)REC[i].g - 2 * (HK - 1) <= 2) snprintf(run, sizeof run, "none");
        else snprintf(run, sizeof run, "< %d", (int)REC[i].g - 2 * (HK - 1));
        if (hi == UINT64_MAX) snprintf(nxt, sizeof nxt, "(beyond 2^64)");
        else snprintf(nxt, sizeof nxt, "%" PRIu64, hi);
        if (full) {
            snprintf(status, sizeof status, "%s, %s", reps ? "TERM" : "not a term", oeis_tag(i + 1, reps > 0));
            if (i + 1 <= OEIS_DECIDED && (reps > 0) != (known_term_index(i + 1) > 0)) fails++;
        } else {
            u64 from = S.start > lo ? S.start : lo, to = pos < hi ? pos : hi;
            snprintf(status, sizeof status, "%s scanned [%" PRIu64 ", %" PRIu64 ") only", reps ? "TERM (repeat found);" : "undecided:", from, to);
        }
        printf(" %2d  %5u  %-21" PRIu64 "  %-21s  %-8" PRIu64 "  %-21s  %8" PRIu64 " / %-8" PRIu64 "  %s\n",
               i + 1, REC[i].g, lo, nxt, reps, run, w20, w100, status);
        if (reps) {
            printf("             repeats after:");
            for (unsigned j = 0; j < S.nrep[i]; j++) printf(" %" PRIu64, S.rep[i][j]);
            if (reps > S.nrep[i]) printf(" ... (%" PRIu64 " more)", reps - S.nrep[i]);
            putchar('\n');
        }
    }

    /* integrity: every record start in the range must have been seen */
    int nrec = 0, missing = 0;
    for (int i = 0; i < NREC; i++) {
        if (REC[i].p < S.start || REC[i].p >= pos) continue;
        nrec++;
        if (!S.recseen[i]) {
            missing++;
            printf("MISSING  record #%d (gap %u after %" PRIu64 ") was not seen where A002386 lists it\n", i + 1, REC[i].g, REC[i].p);
        }
    }
    printf("\n%d record gap%s in the range%s; %" PRIu64 " anomal%s\n", nrec, nrec == 1 ? "" : "s",
           missing ? "" : (nrec ? ", all confirmed where A002386 lists them" : ""), S.anomalies, S.anomalies == 1 ? "y" : "ies");
    fails += missing + (S.anomalies > 0);

    /* the prime count, checked against pi(10^k) when the scan ends at a power of ten */
    printf("%" PRIu64 " primes in [%" PRIu64 ", %" PRIu64 ")", S.primes, S.start, pos);
    for (int k = 6; k <= 18; k++) {
        if (pos != (u64)pow(10, k)) continue;       /* exact: 10^k (k <= 18) is a double */
        u64 expect = PI10[k];
        if (S.start > 0) {
            if (S.start > 100000000000000ULL) break;          /* pi(START) would cost too much */
            expect -= primesieve_count_primes(0, S.start - 1);
        }
        printf(S.primes == expect ? "  = pi(10^%d)%s" : "  BUT pi(10^%d)%s = %" PRIu64 " -- MISMATCH", k,
               S.start ? " - pi(START-1)" : "", expect);
        if (S.primes != expect) fails++;
        break;
    }
    putchar('\n');

    if (g_hist_lines) {
        printf("\n# hist N GAP COUNT: gaps of size GAP after primes strictly inside record interval N (gap g_N first after A002386(N))\n");
        for (int i = 0; i < NREC; i++)
            for (int k = 0; k < HK; k++)
                if (S.hist[i][k]) printf("hist %d %u %" PRIu64 "\n", i + 1, REC[i].g - 2 * k, S.hist[i][k]);
    }
    fflush(stdout);
    return fails;
}

/* Brute-force reference for 'selftest': the statistics the scan keeps in S,
 * computed by a plain loop over an array of all primes, without chunks,
 * threads, thresholds or events. */
typedef struct {
    u64 primes;
    u64 hist[NREC][HK];
    u64 rep[NREC][REPMAX];
    unsigned nrep[NREC];
    bool recseen[NREC];
    u64 anomalies;
} ref_t;

/*
 * reference_scan: compute ref_t for the gaps after the primes in [0, limit).
 *   In:     limit (selftest uses 2*10^8: 11 million primes, about 90 MB);
 *           r = the result.
 *   Out:    *r filled with the prime count, histogram, repeat positions,
 *           confirmed records and anomalies, by the definitions of section 4
 *           applied to every gap.
 *   Errors: die() if primesieve cannot generate the primes (out of memory).
 */
static void reference_scan(u64 limit, ref_t *r)
{
    size_t n;
    u64 *pr = primesieve_generate_primes(0, limit + LOOKAHEAD, &n, UINT64_PRIMES);
    if (!pr || n < 2) die("primesieve_generate_primes failed");
    memset(r, 0, sizeof *r);
    for (size_t j = 0; j + 1 < n && pr[j] < limit; j++) {
        const u64 p = pr[j], g = pr[j + 1] - p;
        r->primes++;
        int i = 0;
        while (i + 1 < NREC && REC[i + 1].p <= p) i++;  /* the interval of p, by linear search */
        if (p == REC[i].p) {
            if (g == REC[i].g) r->recseen[i] = true; else r->anomalies++;
            continue;
        }
        if (g > REC[i].g) { r->anomalies++; continue; }
        u64 k = (REC[i].g - g) / 2;
        if (k < HK) r->hist[i][k]++;
        if (g == REC[i].g && r->nrep[i] < REPMAX) r->rep[i][r->nrep[i]++] = p;
    }
    primesieve_free(pr);
}

/*
 * same_as_reference: compare the scan's results in S with a reference.
 *   In:     r = reference_scan()'s result for the same range [0, S.end).
 *   Out:    true iff the prime count, the anomalies, the whole histogram, the
 *           confirmed records and the repeat positions all agree.
 *   Errors: none.
 */
static bool same_as_reference(const ref_t *r)
{
    if (r->primes != S.primes || r->anomalies != S.anomalies) return false;
    if (memcmp(r->hist, S.hist, sizeof S.hist) != 0) return false;
    if (memcmp(r->recseen, S.recseen, sizeof S.recseen) != 0) return false;
    if (memcmp(r->nrep, S.nrep, sizeof S.nrep) != 0) return false;
    for (int i = 0; i < NREC; i++)
        if (memcmp(r->rep[i], S.rep[i], S.nrep[i] * sizeof(u64)) != 0) return false;
    return true;
}

/* ------------------------------------------------------------------ */
/* Commands                                                            */
/* ------------------------------------------------------------------ */

/*
 * default_threads: the default for -t, the number of online CPUs.
 *   In:     nothing.
 *   Out:    the CPU count, limited to 1..1024 (1 if it cannot be determined).
 *   Errors: none.
 */
static int default_threads(void)
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n < 1 ? 1 : (n > 1024 ? 1024 : (int)n);
}

/*
 * usage: print the command summary on stderr and exit with status 2.
 *   In/Out: nothing; does not return.
 *   Errors: none.
 */
static void usage(void)
{
    fprintf(stderr,
        "usage: a053686 scan [START] END [-t T] [-c CHUNK] [-s KIB] [-n NEAR] [-S FILE] [-i SECS] [-H] [-q]\n"
        "       a053686 verify P G\n"
        "       a053686 bench [N] [-d SPAN] [-t T] [-c CHUNK] [-s KIB]\n"
        "       a053686 selftest [LIMIT] [-t T]\n"
        "START may be rN, the prime after which record gap N first occurs (e.g. r35 = 4302407359).\n");
    exit(2);
}

/*
 * cmd_scan: the 'scan' command (see Usage).
 *   In:     argc/argv after the word "scan": [START] END and options.
 *   Out:    runs the scan, prints the summary on stdout and a timing line on
 *           stderr; returns 0, or 1 if print_summary() found a failure.
 *   Errors: usage() for an unknown option, a missing END or a third number;
 *           die() for a bad number, CHUNK < 10^6, an option value out of range,
 *           or anything run_scan() rejects.
 */
static int cmd_scan(int argc, char **argv)
{
    u64 pos[2];
    int npos = 0, threads = default_threads(), interval = 60;
    u64 chunk = 10000000000ULL;
    const char *state = NULL;
    bool quiet = false;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "-t") && i + 1 < argc) threads = parse_int(argv[++i], 1, 1024, "threads");
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) chunk = parse_num(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) g_sieve_kib = parse_int(argv[++i], 16, 8192, "sieve KiB");
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) g_near = (unsigned)parse_int(argv[++i], 0, 2 * (HK - 1), "NEAR");
        else if (!strcmp(argv[i], "-S") && i + 1 < argc) state = argv[++i];
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) interval = parse_int(argv[++i], 1, 86400, "SECS");
        else if (!strcmp(argv[i], "-H")) g_hist_lines = true;
        else if (!strcmp(argv[i], "-q")) quiet = true;
        else if (argv[i][0] == '-' && !isdigit((unsigned char)argv[i][1])) usage();
        else if (npos < 2) pos[npos++] = parse_num(argv[i]);
        else usage();
    }
    if (npos == 0) usage();
    u64 start = npos == 2 ? pos[0] : 0, end = pos[npos - 1];
    if (chunk < 1000000) die("CHUNK must be at least 10^6");
    if (g_sieve_kib) primesieve_set_sieve_size(g_sieve_kib);
    scan_stats_t st = run_scan(start, end, chunk, threads, state, interval, quiet);
    int fails = print_summary();
    char b[32];
    fprintf(stderr, "%" PRIu64 " numbers, %" PRIu64 " primes in %s (%.3g numbers/s)\n",
            st.numbers, S.primes, fmt_dur(st.seconds, b, sizeof b),
            st.seconds > 0 ? (double)st.numbers / st.seconds : 0);
    return fails ? 1 : 0;
}

/*
 * cmd_verify: the 'verify P G' command: check one gap without the sieve.
 *   In:     argv[0] = P, argv[1] = G (any number syntax).
 *   Out:    prints whether P is prime, the next prime and the gap (Miller-Rabin
 *           search; with GMP also mpz_nextprime), the record interval of P, and
 *           whether the gap is the record itself, a repeat (a term, with the
 *           OEIS tag), larger than the record (which would contradict the
 *           table) or how far it falls short.  Returns 0 if the gap after P is
 *           G (and GMP agrees), else 1.
 *   Errors: usage() with fewer than two arguments; die() if P is not prime or
 *           no prime follows P below 2^64.  (mpz_*_ui take an unsigned long,
 *           64 bits on macOS and Linux.)
 */
static int cmd_verify(int argc, char **argv)
{
    if (argc < 2) usage();
    u64 p = parse_num(argv[0]), g = parse_num(argv[1]);
    if (!is_prime64(p)) die("%" PRIu64 " is not prime", p);
    u64 q = next_prime64(p);
    if (q == 0) die("there is no prime between %" PRIu64 " and 2^64", p);
    printf("%" PRIu64 " is prime; the next prime is %" PRIu64 ", gap %" PRIu64 "%s\n", p, q, q - p,
           q - p == g ? "" : "  -- NOT the claimed gap");
    bool ok = q - p == g;
#ifdef HAVE_GMP
    {
        mpz_t z;
        mpz_init_set_ui(z, p);
        mpz_nextprime(z, z);
        bool agree = mpz_cmp_ui(z, q) == 0;
        mpz_clear(z);
        printf("GMP mpz_nextprime %s\n", agree ? "agrees" : "DISAGREES");
        ok = ok && agree;
    }
#endif
    int i = rec_index(p);
    printf("record interval #%d: gap %u first after %" PRIu64 ", next record after ", i + 1, REC[i].g, REC[i].p);
    if (rec_end(i) == UINT64_MAX) printf("2^64 or beyond\n"); else printf("%" PRIu64 "\n", rec_end(i));
    if (p == REC[i].p)
        printf("this is the first occurrence of record #%d itself%s\n", i + 1, q - p == REC[i].g ? "" : " -- but the gap does not match the table!");
    else if (q - p == REC[i].g)
        printf("REPEAT: gap %u occurs again before the next record, so %u is a term of A053686  [%s]\n",
               REC[i].g, REC[i].g, oeis_tag(i + 1, true));
    else if (q - p > REC[i].g)
        printf("the gap exceeds the record of its interval: this would be an unlisted maximal gap (check the table)\n");
    else
        printf("not a repeat: the record of this interval is %u, short by %" PRIu64 "\n", REC[i].g, REC[i].g - (q - p));
    return ok ? 0 : 1;
}

/*
 * cmd_bench: the 'bench' command: measure the scan rate and project run times.
 *   In:     argc/argv after "bench": [N] and -d SPAN, -t, -c, -s.
 *   Out:    scans [N, N+SPAN) without event lines or checkpoint, prints the
 *           rate, and for each record the OEIS has not decided the time a scan
 *           from P_64 (OEIS_DECIDED + 1) to the end of its interval would take
 *           at that rate, up to 400 days.  Returns 0.
 *   Errors: as cmd_scan(); N + SPAN beyond 2^64 wraps and gives "END must be
 *           larger than START".  The rate is the one near N, and primesieve
 *           slows down slowly as numbers grow, so projections far above N are
 *           optimistic.
 */
static int cmd_bench(int argc, char **argv)
{
    u64 N = 1000000000000000ULL, span = 100000000000ULL, chunk = 10000000000ULL;
    int threads = default_threads(), npos = 0;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "-t") && i + 1 < argc) threads = parse_int(argv[++i], 1, 1024, "threads");
        else if (!strcmp(argv[i], "-d") && i + 1 < argc) span = parse_num(argv[++i]);
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) chunk = parse_num(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) g_sieve_kib = parse_int(argv[++i], 16, 8192, "sieve KiB");
        else if (argv[i][0] == '-' && !isdigit((unsigned char)argv[i][1])) usage();
        else if (npos++ == 0) N = parse_num(argv[i]);
        else usage();
    }
    if (chunk < 1000000) die("CHUNK must be at least 10^6");
    if (g_sieve_kib) primesieve_set_sieve_size(g_sieve_kib);
    g_quiet_events = true;
    scan_stats_t st = run_scan(N, N + span, chunk, threads, NULL, 60, true);
    g_quiet_events = false;
    double rate = (double)st.numbers / st.seconds;  /* seconds >= 0.2: the main loop polls every 200 ms */
    printf("bench: [%" PRIu64 ", %" PRIu64 ") with %d threads, chunk %" PRIu64 ": %.2f s\n",
           N, N + span, threads, chunk, st.seconds);
    printf("       %.4g numbers/s, %.4g primes/s (%" PRIu64 " primes)\n", rate, (double)st.primes / st.seconds, st.primes);
    const int i0 = OEIS_DECIDED;          /* 0-based index of the first record the OEIS has not decided */
    const u64 s0 = REC[i0].p;
    printf("       time to decide each record the OEIS has not decided, scanning from P_%d = %" PRIu64 " at this rate\n"
           "       (the sieve is faster below %.3g and slower above):\n", i0 + 1, s0, (double)N);
    char b[32];
    for (int i = i0; i + 1 < NREC; i++) {
        double t = (double)(rec_end(i) - s0) / rate;
        if (t > 400 * 86400) break;
        printf("         #%-2d gap %-5u needs the scan to %-21" PRIu64 " %s\n", i + 1, REC[i].g, rec_end(i), fmt_dur(t, b, sizeof b));
    }
    return 0;
}

/*
 * cmd_selftest: the 'selftest' command (make test): the checks of section 8.
 *   In:     argc/argv after "selftest": [LIMIT] (default 10^11) and -t T.
 *   Out:    one "ok" or "FAIL" line per check, then "selftest passed" or
 *           "SELFTEST FAILED (k)"; returns 0 or 1.  The final scan of [0, LIMIT)
 *           prints its RECORD / REPEAT / DECIDED lines as a normal scan does.
 *   Errors: usage()/die() for bad arguments, die() for a failure inside the
 *           scans; after a Ctrl-C the remaining scans stop at once and their
 *           checks fail.
 */
static int cmd_selftest(int argc, char **argv)
{
    u64 limit = 100000000000ULL;
    int threads = default_threads(), npos = 0, fails = 0;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "-t") && i + 1 < argc) threads = parse_int(argv[++i], 1, 1024, "threads");
        else if (argv[i][0] == '-' && !isdigit((unsigned char)argv[i][1])) usage();
        else if (npos++ == 0) limit = parse_num(argv[i]);
        else usage();
    }
    /* CHECK(cond, fmt, ...): print "ok" or "FAIL" with the message, and count the failures */
#define CHECK(cond, ...) do { if (cond) printf("ok    " __VA_ARGS__); else { printf("FAIL  " __VA_ARGS__); fails++; } putchar('\n'); } while (0)

    /* number parsing, including the inputs that used to hang or be clamped */
    CHECK(parse_num("2^54-32") == 18014398509481952ULL && parse_num("1e12") == 1000000000000ULL &&
          parse_num("1.7e15") == 1700000000000000ULL && parse_num("10^15+7") == 1000000000000007ULL &&
          parse_num("r35") == 4302407359ULL && parse_num("1^1000000000000000000") == 1 && parse_num("0^0") == 1 &&
          parse_num("0^5") == 0 && parse_num("0e999999999999999999") == 0 && parse_num("1.50e1") == 15 &&
          parse_num("0.0e3") == 0 && parse_num("18446744073709551615") == UINT64_MAX,
          "number parsing");
    {
        static const char *bad[] = { "99999999999999999999^1", "2^64", "18446744073709551616", "1.5e0", "0.5e0", "1e",
                                     "e5", "1.2.3e4", "2e19", "", "x" };
        bool ok = true;
        u64 v;
        for (size_t j = 0; j < sizeof bad / sizeof *bad; j++)
            if (parse_term(bad[j], strlen(bad[j]), &v)) { ok = false; printf("      '%s' was accepted as %" PRIu64 "\n", bad[j], v); }
        CHECK(ok, "number parsing rejects overflow and non-integers");
    }
    /* the independent primality code */
    CHECK(next_prime64(0) == 2 && next_prime64(2) == 3 && next_prime64(3) == 5 && next_prime64(LARGEST_PRIME64 - 1) == LARGEST_PRIME64 &&
          next_prime64(LARGEST_PRIME64) == 0 && next_prime64(UINT64_MAX) == 0, "next-prime search, including the top of the 64-bit range");
    CHECK(is_prime64(2) && is_prime64(3) && !is_prime64(1) && !is_prime64(9) && is_prime64(1000000007ULL) &&
          !is_prime64(3215031751ULL) && is_prime64(18446744073709551557ULL) && !is_prime64(18446744073709551555ULL) &&
          is_prime64(4302407359ULL) && is_prime64(4302407359ULL + 354) && !is_prime64(4302407359ULL + 2),
          "Miller-Rabin spot checks");

    /* the table: increasing, each entry a genuine prime gap of the listed size */
    {
        bool mono = true, gaps = true;
        for (int i = 0; i < NREC; i++) {
            if (i && (REC[i].g <= REC[i - 1].g || REC[i].p <= REC[i - 1].p)) mono = false;
            if (!is_prime64(REC[i].p) || next_prime64(REC[i].p) != REC[i].p + REC[i].g) {
                gaps = false;
                printf("      table entry #%d (%u after %" PRIu64 ") is not a prime gap of that size\n", i + 1, REC[i].g, REC[i].p);
            }
        }
        CHECK(mono, "A005250/A002386 table is increasing in both columns");
        CHECK(gaps, "all %d table entries are prime gaps of the listed size (Miller-Rabin next-prime search)", NREC);
        CHECK(rec_index(2) == 0 && rec_index(3) == 1 && rec_index(6) == 1 && rec_index(7) == 2 && rec_index(22) == 2 &&
              rec_index(23) == 3 && rec_index(4302407358ULL) == 33 && rec_index(4302407359ULL) == 34 &&
              rec_index(UINT64_MAX) == NREC - 1, "record interval lookup");
    }

    /* chunk boundaries: odd chunk sizes and thread counts against the brute-force reference */
    {
        const u64 lim = 200000000ULL;
        ref_t *ref = malloc(sizeof *ref);
        if (!ref) die("out of memory");
        static const struct { u64 chunk; int threads; } cfg[] = {
            { 1234567ULL, 0 }, { 7000001ULL, 3 }, { 50000000ULL, 1 }, { 200000000ULL, 1 }, { 1000000ULL, 0 }
        };                                          /* threads 0 = all */
        unsigned save_near = g_near;
        g_near = 0;
        for (size_t ci = 0; ci < sizeof cfg / sizeof *cfg; ci++) {
            int t = cfg[ci].threads ? cfg[ci].threads : threads;
            g_quiet_events = true;                  /* no REPEAT/RECORD lines for these small scans */
            run_scan(0, lim, cfg[ci].chunk, t, NULL, 60, true);
            g_quiet_events = false;
            reference_scan(S.end, ref);
            u64 tot = 0;
            for (int i = 0; i < NREC; i++) for (int k = 0; k < HK; k++) tot += S.hist[i][k];
            CHECK(same_as_reference(ref), "chunked scan (chunk %" PRIu64 ", %d threads) matches brute force on [0, %" PRIu64
                  "): %" PRIu64 " primes, %" PRIu64 " histogram entries, %d records", cfg[ci].chunk, t, S.end, S.primes, tot, S.last_rec);
            if (S.end == lim) CHECK(S.primes == 11078937ULL, "prime count is pi(2*10^8) = 11078937");
        }
        g_near = save_near;
        free(ref);
    }

    /* every record decision below LIMIT against the OEIS */
    printf("sieve scan of [0, %" PRIu64 ") with %d threads ...\n", limit, threads);
    fflush(stdout);
    u64 chunk = limit >= 100000000000ULL ? 10000000000ULL : (limit >= 10000000000ULL ? 1000000000ULL : 100000000ULL);
    scan_stats_t st = run_scan(0, limit, chunk, threads, NULL, 60, true);
    char b[32];
    printf("      %" PRIu64 " primes in %s (%.3g numbers/s)\n", st.primes, fmt_dur(st.seconds, b, sizeof b),
           st.seconds > 0 ? (double)st.numbers / st.seconds : 0);
    int ndecided = 0;
    for (int i = 0; i + 1 < NREC && REC[i + 1].p <= S.end; i++) {   /* intervals that end inside the scan */
        const int n = i + 1;
        const bool reps = S.hist[i][0] > 0;
        ndecided++;
        if (n <= OEIS_DECIDED) {
            int k = known_term_index(n);
            CHECK(reps == (k > 0), "record #%d = %u %s before the next record (OEIS: %s)", n, REC[i].g,
                  reps ? "repeats" : "does not repeat", k ? "a term" : "not a term");
        } else {
            printf("      record #%d = %u %s before the next record (not decided in OEIS)\n", n, REC[i].g, reps ? "REPEATS" : "does not repeat");
        }
        if (reps)
            CHECK(next_prime64(S.rep[i][0]) == S.rep[i][0] + REC[i].g, "first repeat of %u after %" PRIu64 " confirmed by next-prime search",
                  REC[i].g, S.rep[i][0]);
    }
    CHECK(ndecided >= 34 || limit < REC[34].p, "%d records decided below %" PRIu64, ndecided, S.end);
    {
        int missing = 0;
        for (int i = 0; i < NREC && REC[i].p < S.end; i++) if (!S.recseen[i]) missing++;
        CHECK(missing == 0 && S.anomalies == 0, "every record below the limit found where A002386 lists it, no anomalies");
    }
    for (int k = 6; k <= 18; k++)
        if (S.end == (u64)pow(10, k)) CHECK(S.primes == PI10[k], "prime count %" PRIu64 " = pi(10^%d)", S.primes, k);
#undef CHECK
    printf(fails ? "SELFTEST FAILED (%d)\n" : "selftest passed\n", fails);
    return fails ? 1 : 0;
}

/*
 * main: run the command named by argv[1].
 *   In:     the command line.
 *   Out:    the command's exit status (0 = success).  stdout is line-buffered
 *           so that results reach a pipe (tee) as soon as they are printed.
 *   Errors: usage() (status 2) for a missing or unknown command.
 */
int main(int argc, char **argv)
{
    stderr_tty = isatty(2);
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 2) usage();
    if (!strcmp(argv[1], "scan")) return cmd_scan(argc - 2, argv + 2);
    if (!strcmp(argv[1], "verify")) return cmd_verify(argc - 2, argv + 2);
    if (!strcmp(argv[1], "bench")) return cmd_bench(argc - 2, argv + 2);
    if (!strcmp(argv[1], "selftest")) return cmd_selftest(argc - 2, argv + 2);
    usage();
    return 2;
}
