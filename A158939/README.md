# A158939 — first prime followed by exactly n increasing prime gaps

Tool to reproduce and extend [OEIS A158939](https://oeis.org/A158939):

> First primes followed by sequences of exactly n monotonic increasing prime gaps.

Known terms (offset 1):

| n | a(n) | gaps after a(n), then the gap that ends the run |
|--:|--:|:--|
| 1 | 3 | 2 \| 2 |
| 2 | 2 | 1 2 \| 2 |
| 3 | 17 | 2 4 6 \| 2 |
| 4 | 347 | 2 4 6 8 \| 6 |
| 5 | 2903 | 6 8 10 12 14 \| 4 |
| 6 | 15373 | 4 6 8 10 12 14 \| 12 |
| 7 | 128981 | 2 4 6 8 10 12 14 \| 12 |
| 8 | 1319407 | 4 8 10 14 16 18 32 34 \| 18 |
| 9 | 17797517 | 2 4 8 10 12 20 28 42 50 \| 30 |
| 10 | 94097537 | 2 4 8 10 12 14 16 18 30 32 \| 18 |
| 11 | 6927837557 | 2 4 8 12 16 18 24 32 40 44 70 \| 50 |
| 12 | 48486712783 | 4 6 8 10 12 14 16 18 20 22 36 38 \| 4 |
| 13 | 968068681511 | 8 10 14 18 22 26 28 30 36 44 46 48 50 \| 6 |
| 14 | 1472840004017 | 2 4 6 8 10 12 14 28 30 38 48 64 66 74 \| 22 |
| 15 | 129001208165717 | 2 4 6 12 18 20 28 32 34 36 44 54 72 84 114 \| 40 |

a(15) is from Giovanni Resta (2016); a(16) and a(17) are from this work and have
been in the entry since September 2026 (see Results and
[submission.md](submission.md)). The same data appears shifted in
[A229832](https://oeis.org/A229832) (runs of weak primes: A229832(n) is the
prime after a(n+1)) and as prime indices in [A133697](https://oeis.org/A133697)
(A133697(n) = pi(a(n+2))). Resta's scan reached 1.3e14 (comment in A158940).

## Definition

For a prime p let p = q_0 < q_1 < q_2 < ... be the consecutive primes from p
on and g_i = q_i - q_(i-1). The run length of p is

    L(p) = largest n with g_1 < g_2 < ... < g_n

(so g_(n+1) <= g_n ends the run), and a(n) is the smallest prime with
L(p) = n. This is the PARI program in the entry; the offset is 1 (nothing has
L(p) = 0).

## Method

Every prime below a(n) has to be looked at, so the cost is one pass of a
prime sieve over the range and nothing cleverer applies. The range is cut
into chunks (default 10^10) handed to pthreads through an atomic counter; each
worker sieves its chunk with a primesieve iterator, reads primes straight from
the iterator's buffer and keeps the current increasing run and a ring of the
last 256 primes. When a gap fails to increase, the run of length L that just
ended assigns run lengths 1..L to the previous L primes; only runs longer
than anything seen so far in the chunk need attention, so the per-prime work
is a compare, an increment and a store. Workers continue past their chunk end
until the run in progress ends, so every prime is settled by the chunk that
owns it and no context from before a chunk is needed.

Results are merged under a mutex. A term found in chunk c is *confirmed*
once every chunk before c is complete (the frontier). The frontier, the
candidates, the exact prime count and the exact run-length histogram are
checkpointed (`-S FILE`), and rerunning the same command resumes.

Independent checks built in:

* `selftest` compares the chunked scan (odd chunk sizes, all threads) with a
  brute-force pass over an array of primes, checks a(1..13) by sieving to
  10^12, checks pi(a(n)) against A133697, and checks every known term with a
  Miller-Rabin next-prime search that shares no code with the sieve.
* The prime count of a scan [0, 10^k) is compared with pi(10^k).
* `verify P` recomputes the gaps after P with deterministic Miller-Rabin and
  with GMP's `mpz_nextprime`; `verify_run.py` does the same in pure Python.

## Build and use

    brew install primesieve gmp      # gmp optional
    make && make test                # selftest, ~1 minute

    ./a158939 bench 1e15             # throughput at 10^15 plus run-length densities
    ./a158939 verify 129001208165717 # gaps after a prime, its run length
    ./verify_run.py 129001208165717 15

The extension run (start from 2 so a(1..15) are re-derived on the way;
stops when a(16) is confirmed; resumable; keeps the Mac awake):

    caffeinate -i ./a158939 scan 1e16 -n 16 -S a158939.state | tee -a a158939.log

Options: `-t T` threads (default all cores), `-c CHUNK` chunk size (default
10^10), `-r NMIN` print new minima for run lengths >= NMIN (default 14),
`-i SECS` checkpoint/progress interval (default 60), `-q` no status line,
`-s KIB` primesieve sieve size. Numbers may be written as `2^54-32`, `1e15`,
`1.29e14`, `10^16`. `scan START END` restricts the scan to [START, END) for
splitting a run across machines (results are then "first prime >= START").

## Speed and expected running time (Apple M1 Pro, 8P+2E cores)

Measured with 10 threads and 10^10 chunks (primesieve's counting-only speed
is about 25% higher; the rest is turning sieve bits into 64-bit primes):

| region | numbers/s | cumulative time from 0 |
|--:|--:|--:|
| 10^11 .. 10^12 | 2.3e10 | 10^12: 45 s |
| 10^14 | 2.0e10 | 10^14: 1.3 h |
| 10^15 | 1.78e10 | 10^15: 14 h |
| 2e15 | 1.75e10 | 2e15: 30 h |
| 5e15 | 1.7e10 | 5e15: 3.3 days |
| 10^16 | 1.65e10 | 10^16: 6.7 days |

Where a(16) is likely: for i.i.d. exponential gaps P(L(p) = n) = n/(n+1)!,
but even gaps tie, so strict increase is rarer. The measured densities in
[10^15, 1.002e15] (5.79e10 primes) relative to that model are

| n | 8 | 9 | 10 | 11 | 12 | 13 |
|--:|--:|--:|--:|--:|--:|--:|
| primes with L(p) = n | 764019 | 74339 | 6523 | 539 | 41 | 4 |
| ratio to model | 0.60 | 0.52 | 0.45 | 0.41 | 0.37 | 0.46 (4 samples) |

The ratio falls by about 0.9 per step, giving ~0.24 at n = 16, i.e. about
one prime with L(p) = 16 per 9e13 primes. The number of such primes below x
is then Poisson with mean about 1.1e-14 pi(x), which gives

| a(16) below | 5e14 | 1e15 | 2e15 | 3e15 | 5e15 | 1e16 | 2e16 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| probability | 14% | 26% | 45% | 59% | 76% | 94% | 99.6% |
| wall time to reach | 7 h | 14 h | 30 h | 46 h | 3.3 d | 6.7 d | 14 d |

So the median expectation is about 1.5 days, the mean about 2.3 days, with a
1-in-4 chance of more than 3 days. (Consistency check: the same model gives
an expected 0.8 primes with L = 15 below the actual a(15), and 0.17 with
L = 14 below a(14).) a(17) is expected near 4e16 and is out of reach here.

## GPU version (DGX Spark, CUDA)

[cuda/a158939_cuda.cu](cuda/a158939_cuda.cu) is the same scan on an NVIDIA GB10
(DGX Spark), with no library dependencies (it sieves its own sieving primes).
Build on the Spark with `make` in `cuda/` (nvcc 13, `-arch=sm_121`).

Design: numbers coprime to 30 are one byte per 30 numbers, bit j for residue
R[j]. A chunk is 96 segments of 2,580,480 numbers (2.48e8). Per chunk the
sieving primes above QSPLIT (default 2e6, up to sqrt(END)) are marked by one
thread each into an 8 MB bitmap that stays inside the 24 MB L2, where random
atomics run at ~2e10/s (fifteen times the DRAM rate); each block then loads its
segment into 88 KB of shared memory, ORs in a periodic pattern for 7..19,
marks the primes 23..QSPLIT (warp per prime below 2048, thread per prime
above; the first multiple in each of the eight residue classes comes from a
32-bit Barrett reduction of chunk_lo mod q and a CRT step with q^-1 mod 30),
and walks the finished bitmap: each thread takes a slice of words, extracts
primes and gaps and runs the same increasing-run bookkeeping as the CPU tool
with register-only state, continuing past its slice until the run in progress
ends (blocks sieve an overlap of 61,440 numbers for that; a run still open at
the end is settled on the CPU, which never happens in practice). The kernels
of a chunk run in order on one stream (overlapping two chunks on two streams
was measured slower, see below). Chunks complete in order, so every term is
confirmed as it is found; `-S FILE` checkpoints position, table and histogram, and rerunning the
same command resumes (the large-prime state is recomputed). A run that ends
inside a chunk records END as its position, and a resume with a larger END
rescans that chunk from there, so END can be extended freely.

Measured: 2.2-2.5e11 numbers/s at 10^15 and 1.6-1.7e11 at 8e15, i.e. 13x the
M1 Pro; 10^16 in about 15 hours. Above 10^16 the large sieving primes dominate
(17 million at 10^17, 51 million at 10^18), and the design for them went
through three versions:

1. one thread per prime with a 64-bit multiplier state, touched every chunk:
   1.1e11 numbers/s at 2e16, 0.5e11 at 10^18;
2. a 4-byte state (chunk mod 64, byte offset, wheel index) so that a prime
   whose next multiple is chunks away costs one read: 1.7e11 at 2e16,
   1.3e11 at 8e16, 0.5-0.7e11 at 10^18. Ablation showed the L2 atomics cost
   only ~0.2 ms per chunk there; the time went into state traffic, made worse
   by the sparse 4-byte write-backs of the ~18% of primes that hit;
3. bucket lists: primes above CHUNK_NUMS/6 (which can skip chunks) live as
   8-byte entries {state, q} in 64 lists, one per upcoming chunk. A chunk
   processes only the entries due in it and re-appends each to the list of
   its next multiple's chunk, staged per block so the appends are coalesced;
   the ~2.5 million smaller "dense" primes keep a flat, always-rewritten
   state array. 1.58e11 at 5e16, 1.51e11 at 10^17, 1.29e11 at 10^18.

The residue-class start offsets of the medium primes are computed as
(c + R[k]*beta) mod q with a per-prime inverse (item "cheaper setup"); it is
correct but does not allow a larger QSPLIT, because each segment block has
to re-read every medium prime's table entries and that bandwidth, not the
arithmetic, is the limit. QSPLIT stays at 10^6. Two-stream overlap of the
kernels and pinning both bitmaps in L2 were both slower than one stream with
one 8 MB bitmap pinned by a persisting access window (`-N` disables it).
Running the CPU tool on the Spark's 20 Arm cores alongside adds 3e10
numbers/s at 10^17 but slows the GPU by 14%, and the CPU tool falls to 1e10
near 1.7e18, so it is a wash and is not used.

A wheel-210 variant (48 bits per 210 numbers, kept as
[cuda/a158939_cuda_wheel210.cu](cuda/a158939_cuda_wheel210.cu)) was built and
validated identically, but it is slower: the large-prime kernel gains the
expected 7% from 14% fewer hits, while the medium primes need 48 residue-class
setups per prime per segment instead of 8, a latency-bound chain that doubles
the segment kernel (0.96 ms per chunk at QSPLIT 5e5 versus 0.43 ms), and
lowering QSPLIT to shed medium primes only reaches parity. Measured on an idle
GPU: 1.10e11 at 10^18 versus 1.27e11 for the wheel-30 bucket version, which
therefore stays in production.

Every version was checked against the CPU tool on windows at 3e16 and 10^18
(identical first occurrences, prime indices, prime counts and histograms; the
bucket version's first build missed 86,000 marks per 2e11 window from
sleepers filed under the starting chunk's predecessor, caught by that check)
and by interrupting and resuming at 10^17 (identical to a straight run).

    ./a158939_cuda selftest
    ./a158939_cuda bench 1e15 -p                # per-kernel profile
    setsid nohup ./a158939_cuda scan 1e16 -Q 2e6 -S gpu_a16.state -i 60 > gpu_a16.txt 2> gpu_a16.log < /dev/null &

## Results

**a(16) = 17293451238695141**, found 2026-09-19 by the GPU scan on a DGX Spark
(13h18m into the extended run, at 1.73e16). The run of 16 increasing gaps is

    2 4 6 8 22 26 30 34 36 44 46 54 56 64 86 108   (the next gap is 46)

over the primes 17293451238695141, ...143, ...147, ...153, ...161, ...183,
...209, ...239, ...273, ...309, ...353, ...399, ...453, ...509, ...573, ...659,
...767, ...813. Its prime index is pi(a(16)) = 475618519121221.

Derived new terms of the sister sequences: **A229832(15) = 17293451238695143**
(the prime after a(16), first of 15 consecutive weak primes) and
**A133697(14) = 475618519121221**. a(16), A229832(15) and A133697(14) were
submitted to the OEIS on 2026-09-19 and approved the same day; a(17),
A229832(16) and A133697(15) followed on 2026-09-22. What remains submittable
is listed in [submission.md](submission.md); the b-file is
[b158939.txt](b158939.txt) (n = 1..17).

Verification:

* The run length at 17293451238695141 was re-derived from the definition by
  three independent implementations: `a158939 verify` (deterministic
  Miller-Rabin next-prime search), GMP `mpz_nextprime`, and `verify_run.py`
  (pure Python).
* Minimality: the GPU scan covered [0, 10^16) in one run whose prime count is
  exactly pi(10^16) = 279238341033925, then [10^16 + 41615360, a(16)] in the
  extended run. The 41.6-million-number sliver between the two was scanned by
  the CPU tool (1129839 primes, longest run 9). The sliver existed because the
  first version of the GPU checkpoint recorded a run that ended inside a chunk
  as if the whole chunk were done; this is fixed (the checkpoint now records
  END and a resume rescans that chunk from there, with a selftest case), and
  primecount confirms the corrected prime index: pi(10^16 + 41615359) minus
  pi(10^16) is exactly the 1129839 primes the GPU's running index had missed.
* Both tools reproduce a(1)..a(15) with pi(a(n)) = A133697(n-2), and agree
  exactly (prime counts, first occurrences, full run-length histograms) on
  [0, 10^13) and on a window around a(15).

**a(17) = 52461866207504471**, found 2026-09-22 by the continued GPU scan
(2d09h after resuming from 2e16, at 5.25e16). The run of 17 increasing gaps is

    2 4 6 8 12 18 22 30 36 38 40 44 46 50 54 58 84   (the next gap is 66)

over the primes 52461866207504471, ...473, ...477, ...483, ...491, ...503,
...521, ...543, ...573, ...609, ...647, ...687, ...731, ...777, ...827, ...881,
...939, ...5023, ...5089. Its prime index is pi(a(17)) = 1400080864310974
(primecount; the GPU's running index plus the 1129839 primes of the sliver
gives the same value). The run length was re-derived by `a158939 verify`
(Miller-Rabin and GMP) and `verify_run.py`. Every prime below a(17) was
examined: [0, 10^16) and [10^16 + 41615360, 2e16) by the first two GPU runs,
the sliver by the CPU tool, and [2e16, a(17)] by the continued run (the
2e16 checkpoint was hand-corrected to END before resuming, and the resume-
from-inside-a-chunk logic was verified by selftest case (d)).

Derived: **A229832(16) = 52461866207504473** (the prime after a(17), first of
16 consecutive weak primes) and **A133697(15) = 1400080864310974**.

**a(18) > 54557804097699840** (5.46×10^16). The a(17) run was stopped at
5.30×10^16 on 2026-09-22 (the bound a(18) > 5.30×10^16 is the one in the entry); the search was resumed the same day with the bucket
version and paused again at 5.46×10^16 that afternoon, at a checkpoint. The
median expected location of a(18) is about 2e18 (90% point near 8e18), about
five months of one Spark at 1.3-1.6e11 numbers/s, so continuing is a matter
of hardware and patience rather than code. To continue on atom2 (resumable,
END can be changed):

    cd /home/jbs/A158939/cuda && setsid nohup ./a158939_cuda scan 2e18 -Q 1e6 -S gpu_a18.state -i 60 > gpu_a18.txt 2> gpu_a18.log < /dev/null &

Exact run-length distribution of the primes below 2×10^16 (GPU scan plus
the CPU-scanned sliver; the total, 547863431950008, equals primecount's pi(2×10^16); the model column is the
i.i.d.-exponential-gap value n/(n+1)! times pi):

| n | primes with L(p) = n | ratio to model | n | primes with L(p) = n | ratio |
|--:|--:|--:|--:|--:|--:|
| 1 | 278097256853262 | 1.015 | 10 | 63702682 | 0.464 |
| 2 | 183128766525454 | 1.003 | 11 | 5031375 | 0.400 |
| 3 | 66157934063747 | 0.966 | 12 | 362043 | 0.343 |
| 4 | 16677780549621 | 0.913 | 13 | 23856 | 0.292 |
| 5 | 3223980817207 | 0.847 | 14 | 1467 | 0.250 |
| 6 | 503664116099 | 0.772 | 15 | 84 | 0.214 |
| 7 | 65857869949 | 0.692 | 16 | 1 | 0.041 |
| 8 | 7394602321 | 0.612 | 17 | 0 | (1.45 expected) |
| 9 | 727430840 | 0.535 | | | |

a(16) landing above 10^16 was a roughly 15% outcome under the estimate in
this README: about 2.7 runs of length 16 were expected below 10^16 and about
4 below where it was actually found. The pre-run estimate (median 1.6e15) was
also biased low because it was calibrated on the first-occurrence data rather
than on measured densities. For a(17) the same densities (ratio ~0.19 at
n = 17) predict a median near 6e16 and a 90% point near 1.5e17, i.e. 4 to 10
days on one Spark at the rate the large-prime kernel sustains out there.
