# A350826 — prime sextuplets with n-digit initial term

Tools for extending [OEIS A350826](https://oeis.org/A350826), the number of prime
sextuplets (p, p+4, p+6, p+10, p+12, p+16) with 10^(n-1) < p < 10^n, and its
partial sums [A063501](https://oeis.org/A063501) (sextuplets up to 10^n).

## Status (2026-09-30; OEIS update 2026-10-03)

A063501 has been extended with a(18)-a(20) (edit by Jeff Sponaugle, approved as #29 on 2026-10-03);
the A350826 edit is still to be submitted (see `submission.md`).

OEIS (last edit Jan 2022) lists a(1..17):

```
1, 1, 0, 0, 3, 0, 13, 64, 235, 1296, 7013, 41782, 253420, 1607418, 10520883, 70785653, 488096844
```

N. Luhn's table [PI_06](http://www.pzktupel.de/counting/PI_06.php) has since added
pi_6(10^18) = 4,010,758,480 and pi_6(10^19) = 28,722,086,297 (K. Desfontaines, May 2026)
and pi_6(2^64) = 48,629,687,343 (Jul 2026). They imply

| n  | a(n) | source |
|----|------|--------|
| 18 | 3,439,443,854 | Desfontaines' pi_6, not in OEIS; **confirmed here** (GPU run 2026-09-30, `runs/a18.*`) |
| 19 | 24,711,327,817 | Desfontaines' pi_6, not in OEIS; **confirmed here** (GPU run 2026-10-01, `runs/a19.*`) |
| 20 | **180,637,585,474** | **new**, computed here (two Sparks, Oct 1–2 2026); A063501(20) = pi_6(10^20) = 209,359,671,771 |
| 21 | ~1.34e12 (Hardy–Littlewood) | unknown — about 3 weeks on one Spark |

No sextuplet straddles 10^n (n >= 3) or 2^64, so a(n) = A063501(n) - A063501(n-1) there.
The Hardy–Littlewood estimate C_6 · ∫ dt/ln^6 t (C_6 = 17.29862...) is within 0.001%
of every known pi_6 value from 10^17 on.

### a(18) run

`./a350826_cuda count 1e17 1e18` on atom1 (GB10), 30 min 30 s, wheel 31, B = 2^16:

```
RESULT [100000000000000000, 1000000000000000000) count 3439443854 cks fd8d3b40f799d14e  A350826(18)
RESULT survivors 260495147039 candidates 3439443859 spsp 5 bits 219542181371179
```

= pi_6(10^18) − pi_6(10^17) = 4,010,758,480 − 571,314,626 (Desfontaines / Luhn). Of the
3,439,443,859 candidates whose six members are all base-2 strong probable primes, five
contain a base-2 strong pseudoprime (and five primes); all five were excluded by the Lucas
test and checked independently (`verify_a350826.py` primality, Pollard rho):

| p | composite member |
|---|---|
| 156502671571374757 | p+4 = 212630861 · 736029901 |
| 174519900169098217 | p+0 = 79150597 · 2204909461 |
| 273314385115345027 | p+10 = 426860153 · 640290229 |
| 483757682409253657 | p+6 = 169691191 · 2850811993 |
| 532522504106799457 | p+16 = 161832109 · 3290586197 |

GPU chunk 0 and randomly chosen chunks were recomputed by the CPU tool (`recheck`) with
identical log lines.

### a(19) run

`./a350826_cuda count 1e18 1e19` on atom1, wheel 37, 4 h 00 m at 1.27e11 bits/s (stopped once after
577 chunks to change a speed-only parameter and resumed from the checkpoint):

```
RESULT [1000000000000000000, 10000000000000000000) count 24711327817 cks 584f516044b075a6  A350826(19)
RESULT survivors 2613502357571 candidates 24711327820 spsp 3 bits 1839407465541574
```

= pi_6(10^19) − pi_6(10^18) = 28,722,086,297 − 4,010,758,480 (Desfontaines). The chunk log has
all 92,569 chunks once and its counts add up to the total; 10 random chunks recomputed by the
CPU tool (`recheck`) agree line for line. The three excluded pseudo-sextuplets:

| p | composite member |
|---|---|
| 1157159728167006787 | p+6 = 563142103 · 2054827231 |
| 4211906013907595737 | p+4 = 1510446653 · 2788516897 |
| 6526567117896650257 | p+16 = 154618367 · 42210813919 |

### a(20) run

`a350826_cuda count 1e19 1e20 -b 2^64,2e19,3e19,2^65,4e19,5e19,6e19,7e19,2^66,8e19,9e19` (wheel 37, B = 2^16,
92,569 chunks) on both Sparks: atom1 chunks 0–53417 (Oct 1 00:55 – Oct 2 11:12) and 91858–92568 (tail after an
automatic rebalance), atom2 chunks 53416–91861 (Oct 1 10:07 – Oct 2 11:40); about 60 GPU-hours at 8.6e10 bits/s.
Merged with `combine_a20.py` (`runs/a20_combined.txt`): every chunk once, the 10 chunks computed on both GPUs
identical, candidates − count = 15 pseudo-sextuplets.

```
RESULT [1e19, 1e20) count 180637585474 cks 6fde7e53b8dbdda6  A350826(20)
RESULT A063501(20) = pi_6(10^20) = 209359671771
```

Within 2e-6 of the Hardy–Littlewood estimate (1.806376e11). pi_6 at the other boundaries: pi_6(2^64) =
48,629,688,861, pi_6(2·10^19) = 52,135,772,038, pi_6(3·10^19) = 73,944,895,315, pi_6(2^65) = 88,393,273,302,
pi_6(4·10^19) = 94,783,018,566, pi_6(5·10^19) = 114,931,926,413, pi_6(6·10^19) = 134,551,084,595,
pi_6(7·10^19) = 153,741,842,404, pi_6(2^66) = 160,912,739,529, pi_6(8·10^19) = 172,574,793,380,
pi_6(9·10^19) = 191,101,091,054.

**pi_6(2^64) differs from Luhn's table** (48,629,687,343, Desfontaines Jul 2026) by +1518. A full recount of
[1e19, 2^64) on both GPUs with the halves swapped reproduced all 92,569 chunk lines exactly (`runs/v64a.*`,
`runs/v64b.*`, `verify/compare_bin0.py`), and the CPU program recomputed 40 random chunks of that range with no
mismatch (`verify/spot64.*`). Every counted sextuplet passed BPSW in all six members, so the count cannot
contain composites; we believe the table value is 1518 too small.

A table of pi_6(x) at 131 points of [1e19, 2^64] (every 10^17, every multiple of 2^58, and 2^64 − 2^j for
j = 40..56) is in `runs/pi6_table_1e19_2p64.txt` / `.md` (`make_table_2p64.py`; run `count 1e19 2^64 -b
$(cat runs/bounds_2p64.txt)` split over both Sparks, Oct 2): its intervals add up to 19,907,602,564 again, and
every chunk total equals the recount's. Luhn's 48,629,687,343 lies between our pi_6(2^64 − 2^40) =
48,629,686,408 and pi_6(2^64) = 48,629,688,861.

The whole a(20) range (all 12 bins, so also the 96-bit GPU arithmetic above 2^64) was spot-checked by the CPU
program on 100 distinct random chunks, crosswise (atom1's CPU on atom2's GPU chunks and vice versa, plus 20 on
the M1 Pro): 100 identical chunk lines, 0 mismatches (`verify/recheck_a20*.txt`).

The 15 excluded pseudo-sextuplets (base-2 strong pseudoprime member, five primes; all re-verified):

| p | composite member |
|---|---|
| 14436089061180203227 | p+6 = 921510559 · 15665679487 |
| 18195704945653017187 | p+6 = 635884423 · 28614798991 |
| 19023809129048783947 | p+4 = 96536551 · 197063277401 |
| 19241635766281308547 | p+10 = 2471529041 · 7785316477 |
| 20006509539068916037 | p+12 = 632558449 · 31627922401 |
| 20838543370207726057 | p+16 = 1813917397 · 11488143509 |
| 22282765621488210187 | p = 2360231219 · 9440924873 |
| 29516860589310759097 | p+12 = 3436094329 · 8590235821 |
| 31830699322617753907 | p = 2820935099 · 11283740393 |
| 49058042171497345027 | p+10 = 5718860153 · 8578290229 |
| 61665756833788609627 | p+6 = 4533790057 · 13601370169 |
| 62953548084920616577 | p = 1360725889 · 46264680193 |
| 70455417088719278377 | p = 1041119347 · 67672757491 |
| 78948038145292461907 | p+6 = 2809769353 · 28097693521 |
| 80348597812901062357 | p+16 = 637025117 · 126130972969 |

## Method

Every sextuplet except (7, ..., 23) has p = 97 (mod 210), and for every prime q >= 11 the
residue p mod q must avoid the six values -d (mod q). A wheel M = 2·3·5···w leaves
C(w) = ∏ (q − 6) residue classes (1.5e9 for w = 37, a fraction 2.04e-4 of all integers).
Each class is an arithmetic progression p0 + M k, sieved as a bitmap over k:

* each sieving prime w < q <= B kills six residues phi + K_j (mod q); the offsets
  K_j = −d·M⁻¹ (mod q) are the same for every class, only the phase phi depends on it;
* primes q < 256 are applied as precomputed 64-bit word patterns (CPU and GPU);
* larger primes mark one period ("round", six bits) per loop iteration (CPU), or with
  shared-memory atomicOr (GPU: one warp per prime below 4096, one thread per prime above);
* survivors (~1.4e-3 of the candidates for w = 37, B = 2^16) get a base-2 strong
  probable-prime test member by member, breadth first (all survivors for p, the passing
  ones for p+4, ...: four lock-step lanes on the CPU, separate full-occupancy kernels on
  the GPU, 64-bit Montgomery below 2^64 and 96-bit (3×32-bit limbs) above);
* sextuplets that pass all six are confirmed with a strong Lucas test of every member
  (BPSW: no counterexample below 2^64, none known at all). A member that passes base 2
  but fails Lucas is reported as an `SPSP` line and not counted — this happens: the
  a(18) range contains base-2 strong pseudoprimes such as
  156502671571374761 = 212630861 × 736029901 next to five primes (p = 156502671571374757),
  so a base-2-only count would be wrong.

Counts and the order-independent checksum `cks = Σ mix64(p) mod 2^64` are kept per bin
(`-b`), so any two runs over the same range agree regardless of wheel, sieving bound,
segment size, thread count or CPU/GPU. Chunk logs (`-L`, one line per chunk:
survivors, candidates, pseudoprimes, count:cks per bin) agree line by line for equal
w, B and chunk size, and `a350826 recheck` recomputes sampled chunks of a GPU log on
the CPU.

## Files

* `a350826.c` — CPU tool (C, pthreads, no dependencies). `make && ./a350826 selftest`.
* `cuda/a350826_cuda.cu` — CUDA tool (GB10 / sm_121; `cd cuda && make`). Same classes,
  chunks, checksums, chunk logs and checkpoint format as the CPU tool.
* `verify_a350826.py` — independent Python check (numpy sieve of p = 97 + 210k, Python
  big-integer Miller–Rabin with the first 13 prime bases, deterministic below 3.3e24).
* `verify/` — reference windows computed by `verify_a350826.py`.
* Runs on atom1: `~/A350826/runs/` (`a18.*`, ...).

## Validation

* GPU: A350826(1..17) reproduced exactly (a(17) = 488,096,844 in 3.4 min); CPU:
  A350826(1..14) in its selftest, (16) also; identical checksums on both.
* `verify_a350826.py` agrees count for count and checksum for checksum with both tools
  on A350826(1..10) and on 2e12-wide windows at 1e15, 1e17, 1e18, 1e19, 2^64, 1e20 and
  1e22 (built into both selftests).
* `primesieve -c6` agrees up to 1e11.
* Arithmetic: base-2 strong pseudoprimes (A001262) pass SPRP and fail BPSW, strong Lucas
  pseudoprimes (A217255) pass Lucas and fail SPRP, every odd n < 2e5, 200,000 random
  n < 2^80 (CPU) and 65,536 (GPU) agree with deterministic Miller–Rabin.
* Wheel invariance (CPU selftest): three w = 23 classes against their refinements for
  w = 29 ... 43, with different B, segment sizes and pattern bounds, at 1e18 and across 2^64.
* Small-prime patterns (CPU selftest, since 2026-10-12): the small ranges run with pattern
  bounds 0, 64, 256 and 1024 and must agree with direct search; configurations with the same
  w and B must leave identical survivor counts. Mutation tests confirm that an off-by-one in
  the pattern phase and a dropped pattern are both caught (the earlier selftest ran without
  patterns and missed them).
* a(18) GPU chunk 0 recomputed by the CPU tool: identical log line.
* Related OEIS data re-verified with `verify/reverify_oeis.py` (`verify/reverify_oeis.txt`): A022008
  b-file (10000 terms), A271000 b-file (5940 terms), A200503/A200504/A233426 terms 1..56 (all record
  gaps ending below 1e15), A343636(0..23); no sextuplet straddles 10^n (3 <= n <= 24) or 2^k
  (60 <= k <= 79). OEIS edit draft: `submission.md`.

## Performance and time estimates

Candidate bits per second, measured with `bench` (thousands of classes sampled over the
whole decade; projections include the per-class overhead):

| machine | a(18) range | a(19) range | a(20) range |
|---|---|---|---|
| candidate bits (wheel) | 2.2e14 (w = 31) | 1.84e15 (w = 37) | 1.84e16 (w = 37) |
| M1 Pro, 10 threads | 3.6e9 → **17 h** | 3.3–3.6e9 → **6–6.5 days** | 2.4e9 → **88 days** |
| DGX Spark, 20 Arm cores (CPU tool) | 9.0e9 → 6.8 h | 8.0e9 → 2.6 days | — |
| DGX Spark GB10 GPU | 1.2e11 → **30.5 min** (run) | 1.26e11 → **4.1 h** | 8.6e10 → **2.5 days** |
| two Sparks (`-P 0/2`, `-P 1/2`) | 16 min | 2.1 h | **1.25 days** |

Estimates for machines not measured: Mac Studio (24 threads) ≈ 2.3× the M1 Pro
(a(19) ≈ 2.7 days); a 96-core x86 box ≈ 8–10× (a(19) ≈ 15–20 h, a(20) ≈ 9–11 days).

Where the time goes on the GPU (a(19) range, 20,000 classes): patterns 18%, medium
primes 30%, large primes 27% (both bound by shared-memory atomic throughput, about
7–9 marks per SM-cycle with the inherent 2.6–3.5-way bank conflicts), survivor scan 10%,
base-2 tests 15–20%. In the a(20) range the 96-bit tests (7.9e8/s vs 2.7e9/s at 64 bits)
take half the time. Deeper sieving (B = 2^17, 2^18) does not pay off on either range.

## Running

```
cd ~/A350826                       # atom1
./a350826_cuda count 1e17 1e18 -S runs/a18.state -L runs/a18.chunks > runs/a18.out 2> runs/a18.err
./a350826_cuda count 1e18 1e19 -S runs/a19.state -L runs/a19.chunks > runs/a19.out 2> runs/a19.err
# a(20): started automatically by cuda/queue_a20.sh when the a(19) run ends with a RESULT line
./a350826_cuda count 1e19 1e20 -b 2^64,2e19,3e19,2^65,4e19,5e19,6e19,7e19,2^66,8e19,9e19 \
    -S runs/a20.state -L runs/a20.chunks -i 300 >> runs/a20.out 2>> runs/a20.err
# spot check 20 random chunks of the a(20) GPU log on the CPU (same wheel, chunk and bins)
./a350826 recheck 1e19 1e20 runs/a20.chunks -w 37 -c 16384 -b 2^64,2e19,3e19,2^65,4e19,5e19,6e19,7e19,2^66,8e19,9e19 -n 20
```

Rerunning the same command resumes from the checkpoint (Ctrl-C/SIGTERM saves it); when
resuming with `>>`, the stopped run's `PARTIAL` lines stay in the .out file above the final
`RESULT`. `-P I/N` splits the classes into N chunk-aligned parts whose `PART` results add
up (for two Sparks: `-P 0/2` on one, `-P 1/2` on the other).

Two Sparks on one run (as done for a(20)): atom1 keeps its full-range run; atom2 runs the same command with
`-c 16384 -C X*16384:1516640125` (chunks X.. to the end, own state/log); `cuda/stop_a20.sh X PID` stops atom1
once it has logged chunk X+1, and `combine_a20.py runs/a20.chunks runs/a20b.chunks X` merges the two chunk logs
(checking coverage, the doubly computed chunks X, X+1, and pi_6(2^64)).

The a(20) bins give pi_6 at 2^64 (check: [1e19, 2^64) must hold
pi_6(2^64) - pi_6(10^19) = 48,629,687,343 - 28,722,086,297 = 19,907,601,046 sextuplets),
at k·10^19 and at 2^65, 2^66.
