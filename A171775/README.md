# A171775 — numbers that are k-digit palindromes for every k = 2..n

Tools for computing and extending [OEIS A171775](https://oeis.org/A171775):

> a(n) = smallest number M such that there exist bases b_2, b_3, ..., b_n with the
> property that M written in base b_k is a k-digit palindrome for all k = 2..n.

The OEIS entry (Sep 2026, keywords `base,hard,more`) lists a(1..9):

```
1, 3, 5, 52, 130, 1885, 1073741824, 4398046511104, 72057594037927936
```

with a(7..9) = 2^30, 2^42, 2^56. The entry notes the bound
a(n) ≤ 2^((n-1)(n-2)) (James G. Merickel) and Max Alekseyev's conjecture (Jun 2026)
that equality holds for n ≥ 7. So the next term satisfies a(10) ≤ 2^72.

**This work: a(10) = 2^72 and a(11) = 2^90**, both new and both by exhaustive search,
confirming the conjecture for n = 10 and 11 (see Results). Neither is on the OEIS yet;
[submission.md](submission.md) has the ready-to-paste edit and the OEIS status as of
2026-10-03.

Why 2^((n-1)(n-2)) works: in base b = 2^e − 1 the number 2^r (b+1)^(L−1) has the
digits 2^r·C(L−1, j), which form an L-digit palindrome as long as every 2^r·C(L−1, j) < b.
For N = (n−1)(n−2), every length L = 2..n has a split N = e(L−1) + r that works.

## Results

### OEIS status (checked 2026-10-03)

A171775 is unchanged since revision #64 (a(9) by Max Alekseyev, approved Jun 06 2026): data
a(1..9), no pending edit or draft. Section 1 of [submission.md](submission.md) is the edit
to submit: DATA a(10)–a(11), a comment with the survivor counts, EXTENSIONS, a link to this
directory in the public repository, and optional EXAMPLE and PROG lines. Its appendix
records which of the other `src/math` submissions were live on the OEIS that day (16
approved entries) and points to the per-project `submission.md` files for what is still
open.

### a(10)

**a(10) = 2^72 = 4722366482869645213696** (new term; confirms the conjecture for n = 10).

The search was exhaustive over every M ≤ 2^72 (Sep 29 2026). It enumerated all
3.7·10¹² ten-digit palindromes in bases 2..255 and imposed the 9-digit condition in
every admissible base c. That produced 31,468,592 three-digit matches and
**135,382** numbers that are simultaneously 10-digit and 9-digit palindromes. Every
one of them except 2^72 fails at length 8: it is an 8-digit palindrome in no base.
No candidate even reached length 7.

| run | hardware | time | tuples | (n−1)-palindromes | checksum | solutions |
|---|---|---|---|---|---|---|
| `a171775 search 10 1 2^72` | M1 Pro, 10 threads | 1750 s | 5,126,601,984,459 | 135,382 | (v1 binary, no checksum) | 2^72 |
| `a171775_cuda search 10 1 2^72` | DGX Spark GB10, GPU shared with another job | 154 s | 5,126,601,984,459 | 135,382 | e9eed6127c51a50f | 2^72 |
| `a171775 search 10 1 2^72` | DGX Spark, 18 CPU threads | 653 s | 5,126,601,984,459 | 135,382 | e9eed6127c51a50f | 2^72 |
| `a171775_2d search 10 1 2^72` (swapped 2D method) | M1 Pro, 10 threads | 715 s | 1.78·10¹¹ lookups | 135,382 | e9eed6127c51a50f | 2^72 |

(`scan_n10.txt`/`.log` from the M1 Pro run; the Spark logs are `cpu_n10.*` and
`cuda/gpu_n10.*` in `/home/jbs/A171775` on atom2.) The last row uses the second
method below, which enumerates the other length, so it is an algorithmically
independent confirmation of the same survivor set.

The smallest base of each length for 2^72 (`verify_a171775.py check 2^72 10`):

```
L= 2 base 4722366482869645213695  1 1
L= 3 base 18669329   13548844 15833512 13548844
L= 4 base 524287     32768 98304 98304 32768                  = 2^15 * C(3,j)
L= 5 base 32767      4096 16384 24576 16384 4096              = 2^12 * C(4,j)
L= 6 base 8191       128 640 1280 1280 640 128                = 2^7  * C(5,j)
L= 7 base 2047       64 384 960 1280 960 384 64               = 2^6  * C(6,j)
L= 8 base 1023       4 28 84 140 140 84 28 4                  = 4    * C(7,j)
L= 9 base 511        1 8 28 56 70 56 28 8 1                   =        C(8,j)
L=10 base 255        1 9 36 84 126 126 84 36 9 1              =        C(9,j)
```

### a(11)

**a(11) = 2^90 = 1237940039285380274899124224** (new term; confirms the conjecture for n = 11).

The swapped 2D tool (see Methods) searched every M ≤ 2^90 on the M1 Pro (9 threads).
It ran Sep 29–30 2026 in three checkpointed segments totalling 10,023 s (2 h 47 min),
switching to faster binaries along the way without changing any result. It enumerated
all 7.0·10¹⁵ ten-digit palindromes in bases c ≤ 1024, did 5.39·10¹² table lookups and
1.24·10¹² two- or three-digit matches, and found **524,157** numbers that are
simultaneously 11-digit and 10-digit palindromes. Every one of them except 2^90 fails
at length 9. The single solution is 2^90, found with B = 511 and c = 1023.

| run | hardware | time | lookups | n/(n−1)-palindromes | checksum | solutions |
|---|---|---|---|---|---|---|
| `a171775_2d search 11 1 2^90` | M1 Pro, 9 threads | 10,023 s | 5,386,659,681,911 | 524,157 | 0a5d53d568af3f34 | 2^90 |
| `a171775_2d search 11 1 2^90` | DGX Spark atom2, 16 CPU threads (gcc build) | 4,474 s | 5,386,659,681,911 | 524,157 | 0a5d53d568af3f34 | 2^90 |
| `a171775_2d_cuda search 11 1 2^90` | DGX Spark atom2 GPU (CUDA port) | 3,411 s | 5,386,659,681,911 | 524,157 | 0a5d53d568af3f34 | 2^90 |

(`scan_n11.txt`/`.log`/`.state`; the atom2 runs write `cpu2d_n11.*` and `cuda2d_n11.*` in
`/home/jbs/A171775`.)

Cross-method check: the 1D tool enumerates the eleven-digit palindromes instead of the
ten-digit ones. On the window [600^9, 600^9 + 3·600^8] it gives exactly the 2D tool's
result: 211 survivors, checksum 1dc0b33a97158c66 (1.15·10¹³ tuples, 3,820 s,
`cross1d_n11.*`).

The Metal port reached unit 175,527 of 523,775 (c = 594) before macOS started refusing
its command buffers ("Ignored for causing prior/excessive GPU errors", after repeated
display-watchdog kills). It is paused there (`metal/gpu_n11.state`). The tool now stops
cleanly in that situation, and the same command resumes it in a fresh process.

2^90 itself (`verify_a171775.py check 2^90 11`):

```
L= 2 base 1237940039285380274899124223  1 1
L= 3 base 2147483647   268435456 536870912 268435456          = 2^28 * (1,2,1)
L= 4 base 8388607      2097152 6291456 6291456 2097152        = 2^21 * C(3,j)
L= 5 base 524287       16384 65536 98304 65536 16384          = 2^14 * C(4,j)
L= 6 base 65535        1024 5120 10240 10240 5120 1024        = 2^10 * C(5,j)
L= 7 base 16383        64 384 960 1280 960 384 64             = 2^6  * C(6,j)
L= 8 base 4095         64 448 1344 2240 2240 1344 448 64      = 2^6  * C(7,j)
L= 9 base 2047         4 32 112 224 280 224 112 32 4          = 4    * C(8,j)
L=10 base 1023         1 9 36 84 126 126 84 36 9 1            =        C(9,j)
L=11 base 511          1 10 45 120 210 252 210 120 45 10 1    =        C(10,j)
```

### Verification

* `a171775 selftest` checks a(1..6) by naive enumeration and checks that
  2^((n−1)(n−2)) passes lengths 2..n for n = 7..12. It then compares the fast filter
  with a naive scan on 400+ random blocks for n = 7..12 (B up to 511), both for full
  (n−1)-digit palindromes and for every intermediate k-digit match. Finally it
  reproduces a(7) and a(8) by full searches.
* `a171775_paranoid paranoid` is a debug build of the real search code. At every step
  it recomputes the incremental state (remainders, digits of q, w, the carry deltas)
  from scratch, and it brute-forces the innermost digit on about 1.4·10⁸ steps. This
  found and fixed one bug during development: the double-carry delta for k ≥ 3.
* a(9) = 2^56 is reproduced by an exhaustive search of [1, 2^56]: 32 s on the M1 Pro,
  19,140 (n−1)-palindromes, 1 solution.
* `verify_a171775.py brute` (pure Python, no shared code: plain enumeration, every base
  tried) gives identical statistics to the C tool on four ranges: n=7 to 2^30, n=8 to
  2^36, n=9 to 2^36, n=10 to 2^40. That covers pair counts, canonical survivors,
  checksums, failures by length and solutions.
* The CPU and GPU tools agree exactly, statistics and checksum, on n = 7, 8, 9 (full),
  on the n = 10 window [2^71, 2^71 + 2^68] (259,348,259,906 tuples, 3,792 survivors,
  checksum bf99c892387553d8), and on the full n = 10 search.

## Methods

L = 2 is free (M = "11" in base M−1). Every solution is an n-digit palindrome in some
base B and an (n−1)-digit palindrome in some base c > B. Both tools enumerate one of
those two sets and impose the other condition by solving a linear congruence for the
innermost digits, instead of trying their values.

### 1D (`a171775.c`, `cuda/a171775_cuda.cu`), used for a(10)

The tool enumerates the n-digit palindromes in base B (the smaller set for even n).

* Write M = Mblock + x·w_x + y·w_y + z·s, where x, y, z are the three innermost free
  digits of the base-B palindrome.
* Put k = n−2−⌊n/2⌋ and P = c^(⌊n/2⌋+1). As z runs over 0..B−1, M moves by less than
  P, so q = ⌊M/P⌋ (the k leading base-c digits of M) takes at most two values.
* The k trailing digits must mirror them: M ≡ T(q) (mod c^k), with T the reversed
  digits of q. That is z·s ≡ T − M0 (mod c^k).
* With g = gcd(s, c^k) and u the inverse of a unit s'' ≡ s/g (mod c^k/g), put
  w = (T − M0)·u mod c^k. A solution exists iff g | w, and the smallest one is
  z = w/g. So the per-(x, y) test is a single comparison, w < g·B.
* Stepping (x, y) changes w, M0 mod P and the digits of q by additions only. When q
  grows with the carry stopping at digit m, T moves by c^(m+1) + c^m (mod c^k).
* Hits (probability ≈ B/c^k) are checked exactly: all n−1 digits in base c.

For n = 10 this makes 5.1·10¹² (x, y, c) steps instead of about 5·10¹⁴ digit tests.

### Swapped 2D (`a171775_2d.c`, `metal/`), used for a(11)

For odd n the (n−1)-digit palindromes are the smaller set: below 2^90 there are 7.0·10¹⁵
ten-digit ones against 6.1·10¹⁶ eleven-digit ones. So this tool enumerates the
(n−1)-digit palindromes in base c and solves their two innermost digits y, z jointly
against the n-digit condition in every base B < c.

* Per base pair it uses the largest k ≤ 3 with max(span, w_x) < P = B^(n−k), where
  span = (c−1)(w_y + w_z). Then q = ⌊M/P⌋ takes at most two values over the whole
  (y, z) box, and each step of x raises it by at most one.
* The condition y·w_y + z·w_z ≡ T(q) − M1 (mod B^k) becomes, with the same gcd/unit
  transform, V(y) = (W − y·R) mod B^k with g | V and z = V/g + i·N′.
* A per-(c, B) table of y·R mod B^k, stored under the key (v mod g)·N′ + ⌊v/g⌋ and
  bucketed, turns the matching y into one short window of the table for every g. That
  is one lookup per x, or two when the box straddles q+1.
* Matches are filtered by the next digit pair: digit k comes from precomputed tables
  of (t·w_y) and (t·w_z) mod B^(k+1), and digit n−1−k from a floating-point quotient
  with an exact fallback. Only then is the full palindrome test run.

For n = 10 this needs 1.8·10¹¹ lookups, and for n = 11 about 5·10¹² (against 8.6·10¹⁶
1D steps).

Both tools keep a survivor only for the smallest bases B and c, then test each
remaining length n−2..3 by scanning its bases: last digit, a floating-point filter on
the leading digit, then an exact check. Survivor counts, the order-independent
checksum and failures-by-length mean the same thing in both tools, so their runs are
directly comparable. The file headers have the full derivations.

## Files

| file | purpose |
|---|---|
| `a171775.c`, `Makefile` | 1D CPU tool (single file, pthreads, no dependencies) |
| `a171775_2d.c` | swapped 2D CPU tool (`make a171775_2d`, debug build `make a171775_2d_paranoid`) |
| `metal/a171775_metal.m`, `metal/a171775_2d.metal`, `metal/Makefile` | Apple GPU (Metal) port of the 2D tool; the kernel source is compiled at run time |
| `cuda/a171775_cuda.cu`, `cuda/Makefile` | 1D GPU tool (CUDA, sm_121 / DGX Spark) |
| `cuda/a171775_2d_cuda.cu` | CUDA port of the 2D tool (`make a171775_2d_cuda`); host logic included from `a171775_2d.c` |
| `verify_a171775.py` | independent Python checks (`check`, `small`, `brute`) |
| `scan_n10.*` | the M1 Pro a(10) run |
| `scan_n11.*`, `metal/gpu_n11.*` | the a(11) runs (CPU and Metal) |
| `b171775.txt` | b-file n = 1..11 |
| `submission.md` | OEIS edit for a(10)–a(11) (not yet submitted) and OEIS status check |

## Usage

```bash
make && ./a171775 selftest                     # 1D: ~4 min; 'selftest full' adds a(9)
make a171775_2d && ./a171775_2d selftest       # 2D: n = 8, 9 vs 1D stats; 'selftest full' adds n = 10
make a171775_2d_paranoid && ./a171775_2d_paranoid paranoid
./a171775_2d search 11 1 2^90 -t 9 -S scan_n11.state -i 60 > scan_n11.txt 2> scan_n11.log
./a171775 check 2^72 10
python3 verify_a171775.py brute 9 2^36
```

Metal (`make` in `metal/`; no Xcode Metal toolchain needed):

```bash
./metal/a171775_metal selftest                 # n = 8, 9, 10 vs the CPU tool's statistics
./metal/a171775_metal search 11 1 2^90 -S metal/gpu_n11.state -i 120 > metal/gpu_n11.txt 2> metal/gpu_n11.log
```

On the Spark (nvcc is `/usr/local/cuda/bin/nvcc`; `make` in `cuda/`):

```bash
./a171775_cuda selftest                        # a(7), a(8), a(9) + CPU-tool statistics
./a171775_cuda search 10 1 2^72 -S gpu_n10.state -i 30 > gpu_n10.txt 2> gpu_n10.log
./a171775_2d_cuda selftest                     # 2D: n = 8, 9, 10 vs the CPU tool's statistics
./a171775_2d_cuda search 11 1 2^90 -S cuda2d_n11.state -i 60 > cuda2d_n11.txt 2> cuda2d_n11.log
```

`a171775_2d.c` compiles as C or C++. The Metal and CUDA hosts include it with
`A171775_2D_LIB` (no `main`), and the CUDA one also sets `A171775_2D_NO_DRIVER` (no C11
thread driver), so all three tools share the tables, survivor logic and state files.

Searches are checkpointed with `-S`: the state file stores the frontier unit, the
statistics of exactly the units below it and the `fail` counts, and rerunning the same
command resumes. Since 2026-10-12 every checkpoint is exact, including the ones written
every 60 s while threads run, so a run stopped by `kill -9`, a crash or a power cut
resumes to the same statistics and checksum as an uninterrupted one. (Before that, the
CPU tools' periodic snapshots could include unfinished work, and resuming from one after a
hard stop over-counted; a clean SIGINT/SIGTERM stop was always exact. The a(10) and a(11)
results are unaffected: the segmented M1 Pro run matches the two uninterrupted atom2 runs
in every statistic.) The 1D CPU and CUDA tools share one state-file format, and the 2D
CPU, Metal and CUDA tools another, with the same unit numbering. `-E u` stops at unit u,
and `a171775_2d -u n c HI` prints the first unit of base c; together they allow exact
comparisons of slices (a state file whose frontier is beyond `-E` is refused). After
SIGINT the tools finish their units in flight and save the checkpoint. A range without
units prints "nothing to search" and an empty summary. Solutions go to stdout, progress
to stderr.

Each source file starts with a theory of operations, and every function has a comment
with its inputs, outputs and error conditions.

## Performance notes

* M1 Pro CPU (9 threads): about 6.5–7·10⁸ lookups/s with the 2D tool on n = 11.
* M1 Pro GPU (Metal, 16 cores): the tool gives identical results, but on this
  workload it is only about CPU speed. Its state updates run at 10¹⁰ steps/s, but the
  table lookups and candidate checks are data-dependent and irregular, so the SIMD
  lanes diverge. It is fastest at small c, where lookups have few matches, and slow at
  large c, where most base pairs have k = 2 and many matches. Long command buffers
  trigger the display watchdog ("Impacting Interactivity"); the tool keeps batches
  short and retries a killed batch with smaller ones.
* DGX Spark GB10 (CUDA, 1D kernel): 3.3·10¹⁰ steps/s while sharing the GPU with
  another job; a(10) took 154 s.
* DGX Spark (2D tool, n = 11): 16 CPU threads give about 1.35·10⁹ lookups/s and did
  a(11) in 4,474 s. The GB10 with the CUDA port did it in 3,411 s. It is about
  6·10⁹ lookups/s at small c and slower than the CPU threads at large c, where most
  pairs have k = 2 and the lookups diverge; the 20-unit c = 600 test slice took 10.5 s
  against 2.4 s. Kernels there run up to 50 s, which is fine on the Spark.

## Outlook: a(12)

The conjectured value is 2^110 ≈ 1.3·10³³. 2^90 is not a 12-digit palindrome
(`a171775 check 2^90 12`), so a(12) > 2^90, but that is only the trivial bound
a(12) ≥ a(11). For even n the n-digit palindromes are the smaller set: about 7·10¹⁸
twelve-digit palindromes lie below 2^110. So a search would use the 2D lookup in the
original orientation, solving two base-B digits against the 11-digit condition, and
needs about 1.4·10¹⁶ lookups. At the GB10 rate of the a(11) run (1.6·10⁹ lookups/s)
that is about 100 GPU-days, or two months on both Sparks. The 96-bit limbs would have
to grow, because M reaches 2^110. Not attempted.
