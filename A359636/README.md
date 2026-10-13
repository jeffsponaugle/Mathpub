# A359636 — prime gaps whose composites all have ≥ n distinct prime factors

Tool for computing and extending [OEIS A359636](https://oeis.org/A359636):

> a(n) is the least odd prime not in A001359 such that all subsequent composites
> in the gap up to the next prime have at least n distinct prime factors.

Take consecutive primes p < q with q − p ≥ 4 (so p is not the lesser member of a
twin prime pair, A001359). The gap *qualifies at level n* when ω(x) ≥ n for every
x in p+1 … q−1, where ω = A001221 counts distinct prime factors. a(n) is the
smallest such p.

The OEIS entry (Sep 2026, keyword `hard,more`) lists a(1..8):

```
7, 19, 643, 51427, 8083633, 1077940147, 75582271489, 34710483181813
```

with the bounds a(9) ≤ 76340177205657727 and a(10) ≤ 225096507194749219819
(David A. Corneth). Every known term and both bounds are prime gaps of length 4,
for example a(8) = 34710483181813:

```
34710483181814 = 2*11*17*29*47*229*409*727
34710483181815 = 3^2*5*7*19*31*101*263*7043
34710483181816 = 2^3*13*41*43*61*83*139*269      34710483181817 prime
```

## Results

Computed with this tool on an Apple M1 Pro (10 cores, Sep 16 2026).

* `a359636 selftest` reproduces a(1..7) in about two seconds and a(8) in
  10 min 56 s (exhaustive scan of all m ≤ 3.47·10¹³ at 53 Gm/s on 10 threads: 142,705,091 survivors, 7 triples, 1 qualifying gap; `repro_a8.txt`).
* `verify_a359636.py` (independent Python: numpy ω-sieve over every integer,
  every prime gap walked directly, own Miller–Rabin and Pollard rho) recomputes
  a(1..6) from the definition and re-verifies each known gap.
* **New term: a(9) = 31610535900218923.** Exhaustive GPU scan of every
  m ≡ 3 (mod 6) up to 3.161·10¹⁶ (DGX Spark, 10 h 42 min at 810 Gm/s;
  `gpu_n9.txt`/`.log`/`.state`): 2,417,423,797 survivors, 98 triples, exactly
  one qualifying gap. Verified independently by `a359636 verify`, by
  `verify_a359636.py` (own Miller–Rabin and Pollard rho), and by re-detecting
  it with the CPU sieve on a window around it. The gap:

  ```
  31610535900218923 prime
  31610535900218924 = 2^2*11*17*19*53*79*149*359*9931
  31610535900218925 = 3*5^2*13*23*31*37*157*1559*5021
  31610535900218926 = 2*7*29*61*103*197*223*311*907
  31610535900218927 prime
  ```

  It is 41% of Corneth's bound 76340177205657727 and does not qualify at
  level 10. Before the scan, the structured hunts had lowered the bound to
  36189117287569243 (`-L 1000`, m with all prime factors ≤ 1000) and then to
  33955545649252303 (`-L 300 -Q 2e5`, one free prime); the true a(9) has
  m = 3·5²·13·23·31·37·157·1559·5021, not squarefree, which no hunt variant
  enumerates. Every one of the 53 hunt triples below a(9) reappears in the GPU
  scan's triple list, and the CPU scans (M1 Pro to 1.45·10¹⁵, 0 triples) agree
  with the GPU's first triple at 5.79·10¹⁵.

  The sequence is now

  ```
  7, 19, 643, 51427, 8083633, 1077940147, 75582271489, 34710483181813, 31610535900218923
  ```

* **Lower bound a(10) > 31610555571634177**, for free from the same scan: a
  level-10 gap must contain a level-9 triple with all three ω ≥ 10, and of the
  98 triples below the covered limit only four have even one member with
  ω = 10 (m = 5791430882243721, 8382386489666475, 21469520444799855,
  24048717372672915), none all three. Corneth's a(10) ≤ 225096507194749219819
  remains the upper bound; the ratio a(9)/a(8) ≈ 911 suggests a(10) around
  10¹⁹, near the 2⁶⁴ limit of the current tools. A direct level-10 GPU scan to
  10¹⁸ (T = 1000 with six forced small factors, ~870 Gm/s per machine) is in
  progress to strengthen this bound; see **Status** below.

## Status (Oct 3 2026)

**Done.** a(9) = 31610535900218923 is established, verified, and **in the
OEIS entry since Sep 18 2026** (revision #31, with the comment
a(10) > 31610555571634177). What can still be submitted is listed in
`submission.md`: chiefly the improved bound a(10) > 408697968217030657 from the
paused level-10 scan, plus an EXAMPLE line for a(9).

**In progress: level-10 lower-bound scan to 10¹⁸**, split over two DGX Sparks
since Sep 22 (each half in its own run with its own checkpoint):

| machine | range of m | covered so far | remaining | state |
|---|---|---|---|---|
| atom1 (10.1.30.36) | 0 … 664309293225476097 | m ≤ 4.087·10¹⁷ (61.5%) | ≈ 3 d 12 h at 884 Gm/s | paused Sep 23 11:40, `gpu_n10.state` |
| atom2 (10.1.30.37) | 664309293225476097 … 10¹⁸+2 | m ≤ 7.410·10¹⁷ (22.8%) | ≈ 3 d 17 h at 847 Gm/s | paused Sep 23 11:40, `gpu_n10b.state` |

Both runs were stopped cleanly (SIGINT, checkpoint written) on Sep 23 to free
the machines for other projects and have not been resumed since (state files
unchanged as of Oct 3; copies in this directory); nothing restarts them
automatically. No triple with
all three ω ≥ 10 has appeared anywhere below 4.087·10¹⁷, nor in atom2's
stretch 6.643·10¹⁷ … 7.410·10¹⁷, so the interim rigorous bound is
**a(10) > 408697968217030657**; the two ranges together will give
a(10) > 10¹⁸ (or a(10) itself, should a qualifying gap turn up) about
3 d 17 h of running after resumption.

To resume, on each machine in `/home/jbs/A359636/cuda` (the same command
lines as before; the state files carry the frontier):

```
# atom1
(nohup setsid ./a359636_cuda.run scan 10 0 664309293225476095 -t 16 -b 262144 -S gpu_n10.state -i 60 >> gpu_n10.txt 2>> gpu_n10.log < /dev/null &)
# atom2
(nohup setsid ./a359636_cuda.run scan 10 664309293225476096 1e18 -t 16 -b 262144 -S gpu_n10b.state -i 60 >> gpu_n10b.txt 2>> gpu_n10b.log < /dev/null &)
```

A run is finished when its `.log` ends with a `scanned m <= ...` line and its
`.txt` with a result line; atom1's result is about a(10) directly, atom2's is
about its range only, and the final statement combines the two (no solution in
either range means a(10) > 10¹⁸ − 3).

**Beyond 10¹⁸.** a(10) plausibly lies near 10¹⁹ and Corneth's bound is
2.25·10²⁰, above 2⁶⁴. Finding it would need 128-bit arithmetic in the sieve
targets and the verifier (the reduction and the kernel structure carry over
unchanged) and roughly 2–3 GPU-weeks per 10¹⁹ of range on one Spark.

For comparison, the numbers of *triples* m ≡ 3 (mod 6) with ω(m−1), ω(m), ω(m+1)
all ≥ n found below a(n) (each is a potential term; it becomes one only when the
surrounding gap is bounded by primes and every composite in it qualifies):

| n | a(n) | triples with m < a(n)+2 | of which qualifying gaps |
|---|------|------|------|
| 5 | 8083633 | 10 | 1 |
| 6 | 1077940147 | 16 | 1 |
| 7 | 75582271489 | 3 | 1 |
| 8 | 34710483181813 | 7 | 1 |
| 9 | 31610535900218923 | 98 | 1 |

## Method

**Reduction.** Every qualifying gap contains an odd multiple of 3, call it m,
such that m−1, m, m+1 are all composites of the gap: in a gap of 4 the prime
p is 1 (mod 3), so m = p+2; in a gap of 6, m = p+2 or p+4; a gap of 8 or more
holds three consecutive odd numbers, one of them 0 (mod 3). So it is enough to
find every m ≡ 3 (mod 6) with

    ω(m−1) ≥ n,  ω(m) ≥ n,  ω(m+1) ≥ n                          (*)

and then look at the prime gap around each of them. a(n) is the least
p = prevprime(m−1) over the m satisfying (*) whose whole gap qualifies.

**Three-target sieve.** Write m = 6k+3. For every prime 5 ≤ p ≤ T and each of
the three targets N = 6k+2, 6k+3, 6k+4, the k with p | N form one residue class
mod p. One byte counter per target, packed three to a 32-bit word per k, is
incremented along these classes on L1-sized segments (16384 k). The primes
5·7·11·13, 17·19·23 and 29·31·37 are applied as precomputed periodic patterns
(periods 5005, 7429, 33263); the remaining primes are marked with a strided
loop that handles the three targets of a prime together. The forced divisors
2 | m±1 and 3 | m are not counted; the thresholds are lowered by one instead.

**Why a small T is enough.** If N ≤ Bmax has n distinct prime factors, at most
rmax of them exceed T, where rmax is the largest r with

    (product of the first n−r primes) · (product of the first r primes above T) ≤ Bmax.

So N has c = n − rmax prime factors ≤ T, and a k whose three counters are all
≥ c−1 is a *survivor*. The tool picks T automatically as the smallest bound
that gives c ≥ 5; for n = 9 up to Corneth's bound this is T = 1000 (c = 5, i.e.
five prime factors below 1000 in each of m−1, m, m+1). Survivors occur at a
rate of about 2·10⁻⁵ per k near 7·10¹⁶, so the sieve — about 1.8–2.1 counter
increments per k — is essentially the whole cost.

**Exact verification.** Each survivor's three numbers are tested with trial
division using the same size bound as an early exit (a cofactor that still
needs r distinct primes but is smaller than the r-th power of the next prime
fails at once; a cofactor needing one more prime succeeds iff it exceeds 1; one
needing two is settled by Miller–Rabin plus a perfect-power test). For a k
satisfying (*), p = prevprime(m−1) and q = nextprime(m+1) are found with
deterministic Miller–Rabin and every composite in (p, q) is checked and printed
with its factorization.

**Hunt (upper bounds only).** `a359636 hunt N END -L L` enumerates every
m = 3^e · p₁⋯p_(N−1) (e = 1, 2; distinct primes 5 ≤ pᵢ ≤ L) with m ≤ END+2 and
tests m−1, m+1 and the gap exactly as above. It only sees the m whose own prime
factors are all small, so it can miss terms (a(8)'s m = 3²·5·7·19·31·101·263·7043
needs L ≥ 7043; a(6)'s needs L ≥ 1021), but for n = 9, L = 1000 it visits about
10¹¹ candidates below Corneth's bound instead of 1.3·10¹⁶ sieve positions, and
any hit is an upper bound for a(9) and a point where the exhaustive scan can stop.
Work is split over threads by the first two primes chosen.

With `-Q Q` the largest of the chosen primes may instead be any prime ≤ Q (this
is the shape of Corneth's solution, whose m = 3·7·13·17·23·137·163·211·151783).
Hunts run so far at level 9, all below the first hunt's bound 36189117287569243 unless noted:

| hunt | candidates | time | triples | qualifying gaps |
|---|---|---|---|---|
| `-L 1000` (below Corneth's 7.63·10¹⁶) | 1.47·10¹¹ | 2 h 12 min, 6 threads | 148 | 6, smallest p = 36189117287569243 |
| `-L 2000` | 2.22·10¹¹ | 3 h 39 min, 5 threads | 50 | none |
| `-L 300 -Q 2e5` | 1.69·10¹¹ | 3 h 08 min, 4 threads | 34 | 1, p = 33955545649252303 (m = 3·5·13·19·61·71·127·251·66383) |

**Parallel scan.** Chunks of 2²⁶ k are handed to worker threads through an
atomic counter; the lowest unfinished chunk (the *frontier*) drives the
progress line, the checkpoint file (`-S`, written every `-i` seconds and on
Ctrl-C) and the early stop: once the frontier has passed the smallest solution
found, that solution is a(n) and the scan ends (unless `-a` asks for all
solutions in the range).

## Performance and outlook

| range of m | k values | 10 threads (idle M1 Pro) |
|---|---|---|
| level 8 to a(8) = 3.47·10¹³ | 5.8·10¹² | 10 min 56 s |
| level 9 to 10¹⁶ | 1.7·10¹⁵ | ≈ 2 days |
| level 9 to Corneth's bound 7.63·10¹⁶ | 1.27·10¹⁶ | ≈ 15 days |

The sieve runs at ≈ 2.2 cycles per k on one performance core (≈ 57 Gm/s, i.e.
9.5·10⁹ k/s, on all ten cores). The ratios a(n+1)/a(n) are 80, 157, 133, 70, 459 and now 911. a(10) is
beyond 64-bit arithmetic (its bound is 2.25·10²⁰) and out of reach for this
method.

## GPU version (cuda/a359636_cuda.cu)

`cuda/a359636_cuda.cu` is a CUDA port of the scan for Jeff's DGX Spark (GB10:
48 SMs, 6144 CUDA cores, 20 Arm cores). The sieve runs on the GPU: one thread
block (256 threads) per segment of 8192 k, the three packed byte counters in
shared memory, the primes 5..23 from two pattern tables and the primes 29..T
by shared-memory `atomicAdd` (one warp per progression up to 127, one thread
per progression above, snake-ordered for balance). Survivors are appended to a
device list and verified on the Arm cores by the same host code as the CPU
tool; output, checkpoint (`-S`) and early stop work the same way. Because the
GPU has cycles to spare, the default T is the smallest bound forcing *six*
small prime factors into each target (T = 2000 for the level-9 range), which
makes survivors about ten times rarer than the CPU tool's default.

Validation: the selftest reproduces a(1..7) in 1.7 s and a(8) in 6 min (same 7
triples as the CPU tool); on identical windows with identical T the GPU and CPU
tools report exactly the same survivor counts (e.g. 23,562,061 for
m in [7·10¹⁶, 7·10¹⁶+6·10¹²] with T = 1000).

| | rate | remaining level-9 range (≈3.4·10¹⁶ m) |
|---|---|---|
| M1 Pro, 10 threads (CPU tool) | 57 Gm/s | 6.6 days |
| DGX Spark, 20 Arm cores (CPU tool) | 82 Gm/s | 4.6 days |
| DGX Spark GPU, first version | 100 Gm/s | 3.8 days |
| DGX Spark GPU, 32-bit modulo + coalesced pattern reads | 620 Gm/s | 15 h |
| DGX Spark GPU, 8192-k segments, 256 threads (used for the run) | **810 Gm/s** | **11.6 h** (actual: 10 h 42 min to a(9)) |

Segment size matters (occupancy): 16384-k segments run at 420 Gm/s, 4096-k at
640; 128-thread blocks are slower. With marking and pattern init compiled out
the pipeline alone runs at 4800 Gm/s, so the shared-memory marking is now the
cost and a persistent-block design with the pattern tables in shared memory
would be the next step.

Build and run on a Spark (`nvcc` lives in `/usr/local/cuda/bin`; no other
dependencies, the CUDA tool has its own small prime sieve):

```
cd cuda && make            # NVCC=/usr/local/cuda/bin/nvcc ARCH=sm_121
./a359636_cuda selftest -8
nohup ./a359636_cuda scan 9 0 33955545649252303 -t 16 -b 262144 -S gpu_n9.state -i 60 > gpu_n9.txt 2> gpu_n9.log &
```

Splitting a range over machines: give each its own START/END and `-S` file;
a resumed run may lower or raise END (the frontier is relative to START), and
a run with START > 0 reports the smallest qualifying gap in its range rather
than claiming a(n).

## Usage

```
make                       # needs primesieve (brew install primesieve)
./a359636 selftest         # a(1..7); add -8 to include a(8) (~10 min)
./a359636 scan 9 0 76340177205657727 -t 10 -S scan_n9.state -i 120 > scan_n9.txt 2> scan_n9.log &
./a359636 scan 9 1e16 2e16 -a        # all solutions with 1e16 <= m <= 2e16+2
./a359636 hunt 9 76340177205657727 -L 1000   # structured upper-bound search (~1e11 candidates)
./a359636 hunt 9 36189117287569242 -L 300 -Q 2e5   # 7 primes <= 300 plus one free prime <= 2e5
./a359636 verify 9 76340177205657727 # check one gap, with factorizations
./a359636 omega 76340177205657729    # factor numbers / print omega
python3 verify_a359636.py selftest   # independent brute force a(1..5) + gap checks
python3 verify_a359636.py brute 6 1077940147
python3 verify_a359636.py check-log scan_n9.txt
```

Numbers may be written as decimal, `2^k`, `10^k`, `1e12`, or sums and
differences of those (`10^12-1`, `7e16+12e9`), evaluated left to right; the CPU
tool rejects values that do not fit in 64 bits and negative results. `scan`
searches m ≡ 3 (mod 6) with START ≤ m ≤ END+2. That covers every gap with
p ≤ END except a prime p = END when END ≡ 5 (mod 6) or p = END − 1 when
END ≡ 0 (mod 6), whose m is p + 4; the result line of the CPU tool states the
proven bound (END − 1 or END − 2 in those two cases). A scan with START > 9
reports only on its own range ("smallest qualifying p with m in [A, B] is P" or
"no qualifying gap with m in [A, B]"). Options: `-t` threads, `-c` chunk size (k per chunk),
`-T` sieve prime bound (default automatic), `-S` state file, `-i` checkpoint
interval, `-a` scan the whole range instead of stopping at the first certain
a(n), `-q` quiet. Rerunning a `scan` with the same arguments and `-S` file
resumes from the checkpoint; the state file also records solutions found so far.

## Source documentation and the Oct 12 2026 review

Both sources carry a THEORY OF OPERATION header (the reduction to triples
around m ≡ 3 (mod 6) with its coverage lemma, the packed three-target sieve,
the threshold trick, the choice of T and c, exact verification, threads,
frontier and checkpoint, output, limits) and an In/Out/Errors block for every
function. `a359636.c` also lists known imperfections (benign data races, the
`-Q` pruning gap of `hunt`), and `cuda/a359636_cuda.cu` lists the issues
below that are not yet fixed there.

The review fixed five defects in the CPU tool `a359636.c`, each reproduced
first and then retested; the CUDA file was changed only in its comments
(code tokens and PTX identical), because the paused level-10 runs depend on it.

* **Partial ranges.** A scan with START > 9 printed claims about a(N) itself,
  for example `a(7) > 90000000000` for the range [8·10¹⁰, 9·10¹⁰] although
  a(7) = 75582271489. It now reports only on its range, as the CUDA tool
  already did.
* **END ≡ 5 or 0 (mod 6).** A complete scan printed `a(N) > END`, which is not
  proven when the prime END (or END − 1) has its m at p + 4, beyond END + 2;
  example: the level-3 gap at p = 3911 ≡ 5 (mod 6) has m = 3915. The bound
  printed is now min(END, covered_m + 1).
* **Number parsing.** `10-3-2` gave 9, values above 2⁶⁴ were silently clamped,
  `2^63+2^63` wrapped to 0, `5-7` wrapped, `1^4000000000` looped for 4.5 s,
  and an empty argument was read past its end. Parsing is now left-to-right
  with range checks, and those cases are rejected with a message.
* **Checkpoint writes.** A failed write (e.g. disk full) replaced the state
  file with an empty one, after which the scan could not resume ("bad state
  file"). Write errors are now detected and the previous checkpoint is kept.
* **Smaller items.** END within 65536 of 2⁶⁴ and START > END are rejected
  up front, and the prime table is released with `primesieve_free()` as the
  primesieve API requires (the code had used `free()`).

Verification: the selftest passes (also under AddressSanitizer and
UBSan); every case above was rerun; the comment pass changed no code (identical
token stream, byte-identical object file). Not yet ported to the CUDA tool:
the parsing, END-residue and checkpoint-write fixes, plus removing an unused
128 MB host buffer there. Also noted: building the CPU tool with `-DPATC=0` sieves
about 10% faster on the M1 Pro (2.10 vs 2.35 cycles per k, same results).

## Files

* `a359636.c`, `Makefile` — the tool (single C file, pthreads, primesieve).
* `verify_a359636.py` — independent Python cross-check (numpy for the brute force).
* `repro_a8.txt`, `repro_a8.log` — exhaustive level-8 scan reproducing a(8).
* `hunt_n9_L1000.txt`, `hunt_n9_L1000.log` — level-9 hunt (L = 1000) below Corneth's bound.
* `hunt_n9_L2000.txt`, `hunt_n9_L2000.log` — level-9 hunt (L = 2000) below 36189117287569243.
* `hunt_n9_L300_Q2e5.txt`, `.log` — level-9 hunt with a free largest prime (L = 300, Q = 2·10⁵).
* `OEIS_draft.md`, `b359636.txt` — the a(9) submission text (submitted Sep 18 2026) and b-file.
* `submission.md` — what remains submittable (improved a(10) bound, EXAMPLE, optional comments).
* `gpu_n10.*`, `gpu_n10b.*` — the paused level-10 runs' output/progress/checkpoints (copies from atom1/atom2).
* `cuda/a359636_cuda.cu`, `cuda/Makefile` — the CUDA version (DGX Spark); `gpu_n9.txt`/`.log`/`.state` are the run files of the scan that found a(9).
* `scan_n9.txt`, `scan_n9.log`, `scan_n9.state` — the M1 Pro CPU scan (m ≤ 1.45·10¹⁵, paused; superseded by the GPU run).
* `scan_n9.txt`, `scan_n9.log`, `scan_n9.state` — the level-9 scan (output, progress, checkpoint).
