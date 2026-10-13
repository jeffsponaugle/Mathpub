# A252768 — primes whose prime-gap power sums are all prime

Tool for computing and extending [OEIS A252768](https://oeis.org/A252768):

> Primes p with property that the sum of the k-th powers of the successive gaps
> between primes <= p are prime numbers for k = 1 to n.

Take the primes 2 = q₁ < q₂ < … < qₘ = p and their gaps gᵢ = qᵢ₊₁ − qᵢ, and let

    S_k(p) = Σ gᵢ^k        (k = 1, 2, …)

The *depth* of p is the largest n such that S₁(p), …, Sₙ(p) are all prime, and
a(n) is the smallest prime of depth ≥ n. Example: the primes up to 13 have gaps
1, 2, 2, 4, 2, so S₁ = 11, S₂ = 29, S₃ = 89 (all prime) and S₄ = 305 = 5·61, so
13 has depth 3 and a(3) = 13.

Since S₁(p) = p − 2, every term is the larger member of a twin prime pair
(A006512), and a(n) is non-decreasing in n.

When this work started (Sep 16 2026) the OEIS entry (keyword `hard,more`)
listed a(1..7): 5, 5, 13, 14593, 372313, 2315773, 541613713. The two terms
found below were submitted the same day and approved on Sep 17 2026
(revision #21); the entry now lists a(1..9). See `submission.md` for what is
still worth submitting.

## Results

Everything below was computed with this tool on an Apple M1 Pro (10 cores, Sep 16 2026).
Both new terms were verified independently: by `a252768 verify` (a separate
single-threaded code path, every verdict cross-checked with GMP) and by
`verify_a252768.py` (numpy sieve, exact Python integers, its own Miller–Rabin),
which reproduce every power sum digit for digit.

**Two new terms:**

| n | a(n) | S₁..Sₙ prime, S_(n+1) composite |
|---|------|------|
| 8 | **7952072743** | S₉ = 38009162738943211881748481 = composite |
| 9 | **21814967833** | S₁₀ = 30245374383129007149396784129 = composite |

The exhaustive scan of all primes below 10¹² (37,607,912,018 primes, 111 s)
found no prime of depth ≥ 10, so **a(10) > 10¹²** (and a(11), a(12) > 10¹²;
sums were tested up to k = 12). The OEIS terms a(1..7) are reproduced by the
same scan. a(8) and a(9) are in the OEIS entry since Sep 17 2026; the bound
a(10) > 10¹² is not yet (see `submission.md`). The sequence is now

```
5, 5, 13, 14593, 372313, 2315773, 541613713, 7952072743, 21814967833
```

The power sums at the new terms (all prime unless marked):

```
p = 7952072743   S_1 = 7952072741            S_2 = 306387204649           S_3 = 17232652308257
                 S_4 = 1275921083018209      S_5 = 117138327618559361     S_6 = 12826308390187298689
                 S_7 = 1629701772261995408897                 S_8 = 235364125518401969499649
                 S_9 = 38009162738943211881748481 (composite)

p = 21814967833  S_1 = 21814967831           S_2 = 883642617829           S_3 = 52322085406601
                 S_4 = 4080850722829393      S_5 = 394810145761684001     S_6 = 45578875566881080129
                 S_7 = 6111338676278533840001                 S_8 = 932919209682371038708993
                 S_9 = 159652836113962880019857921            S_10 = 30245374383129007149396784129 (composite)
```

### Depth statistics below 10¹²

Number of primes p < 10¹² of exact depth n (S₁..Sₙ prime, S_(n+1) not):

| n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| count | 1747205024 | 112645225 | 9907858 | 756501 | 65240 | 4934 | 401 | 32 | 5 |

Each extra k costs a factor of only 12–15 in the count (a per-k pass rate of
6.6–8.7 %; the conditions are positively correlated, see below). The five primes of depth 9 below 10¹² are 21814967833,
60308763733, 73668807349, 331274803549 and 844836146389; all 438 primes of
depth ≥ 7 are listed in `scan_1e12.txt`.

### Search-range estimates for a(10) .. a(13)

`estimate.py` models the density of primes of depth ≥ n and is calibrated on
the scan below 10¹²:

* p is the upper member of a twin pair with probability 2·C₂/ln p (C₂ = 0.66016);
* given S₁..S_(k−1) prime, S_k is prime with probability B_k · 2/ln S_k, where
  B_k = Π_{3 ≤ q ≤ k, q prime} q/(q−1) is the congruence boost from the section
  "Why the depths are correlated" (1, 1.5, 1.5, 1.875, 1.875, 2.19, 2.19, 2.19,
  2.19, 2.41, 2.41, … for k = 2, 3, …);
* S_(k+1)/S_k ≈ 0.8·(k+1)·ln p, fitted to the actual sums at a(9).

The model reproduces the measured per-k pass rates N(≥k)/N(≥k−1) below 10¹²
within 1–3 % (k = 2..7: 0.066, 0.087, 0.077, 0.085, 0.076, 0.082 measured vs
0.066, 0.087, 0.077, 0.086, 0.078, 0.082 predicted) and the counts of primes of
depth ≥ 7, 8, 9 (465, 35, 2.4 predicted vs 438, 37, 5 found). Integrating it
beyond 10¹², with the Poisson probability 1 − e^(−E) that a(n) < X:

| n | 50 % point | 90 % point | P(a(n) < 10¹³) | < 10¹⁴ | < 10¹⁵ | < 10¹⁶ | < 10¹⁷ |
|---|---|---|---|---|---|---|---|
| 10 | 1.1·10¹³ | 5·10¹³ | 46 % | 98 % | ~100 % | | |
| 11 | 4.3·10¹⁴ | 2.2·10¹⁵ | 4 % | 21 % | 73 % | 99.9 % | |
| 12 | 2.3·10¹⁶ | 1.2·10¹⁷ | 0.2 % | 1 % | 7 % | 31 % | 87 % |
| 13 | 1.2·10¹⁸ | 6·10¹⁸ | | | 0.4 % | 2 % | 10 % |

Each further term costs about 1.7 decades: the per-k pass rate settles near
6–7 % while the density of qualifying primes halves per decade, so the count
per decade grows only ~5×. Both a(10) and a(11) are within reach of this
machine (hours and about two days respectively, see the timings); a(12) is
not — its median sits at two weeks of continuous computing for a 1-in-3 chance.
The scatter is Poisson: a(9) at 2.2·10¹⁰ was a mildly lucky draw (2.4 primes of
depth 9 expected below 10¹², 5 found), so expect a factor of 3–5 either side
of the medians. Run `python3 estimate.py 1e13` after the next scan to refresh
the table for the new lower bound.

### Timings (M1 Pro, 10 threads)

| scan | time | rate |
|------|------|------|
| 0 .. 6·10⁸ (selftest) | 0.1 s | |
| 0 .. 10¹⁰ | 1.2 s | 8.0·10⁹ numbers/s |
| 0 .. 10¹¹ | 10.5 s | 9.5·10⁹ numbers/s |
| 0 .. 10¹² | 111 s | 9.0·10⁹ numbers/s |
| 0 .. 10¹³ (estimate) | ~20 min | |
| 0 .. 10¹⁴ (estimate) | ~3 h | |
| 0 .. 10¹⁵ (estimate) | ~1.3 days | |
| 0 .. 10¹⁶ (estimate) | ~2 weeks | |
| totals-only pass (scan with START > 0, `verify -t 10`) | 3.5 s per 10¹¹ | 2.9·10¹⁰ numbers/s |

primesieve alone counts primes 3–8× faster than this (1.4 s per 10¹¹ near 0,
4 s per 10¹¹ near 10¹⁵), so the tool is bound by its per-prime work — mostly
the S₂ tests of the 1.87·10⁹ twin primes below 10¹² — which shrinks relative
to the range as the primes thin out; the rate should stay near 9·10⁹/s up to
10¹⁵. `verify` (single-threaded, from scratch) takes 5 s for 2.2·10¹⁰.

## Building

Needs [primesieve](https://github.com/kimwalisch/primesieve) (`brew install primesieve`).
GMP (`brew install gmp`) is optional; when present the `verify` command and
the selftest cross-check every primality verdict with `mpz_probab_prime_p`.

```
make          # builds ./a252768 (adds -DHAVE_GMP -lgmp when gmp.h is found)
make test     # runs the selftest (about 1 s)
```

or by hand:

```
cc -O2 -std=gnu11 -pthread -I/opt/homebrew/include a252768.c \
   -L/opt/homebrew/lib -lprimesieve -lm -DHAVE_GMP -lgmp -o a252768
```

On a Mac where the Xcode and Command Line Tools versions disagree (the link
fails with a `tapi` / SDK error, or `cc` demands the Xcode license), build with
the Command Line Tools toolchain instead:

```
DEVELOPER_DIR=/Library/Developer/CommandLineTools make
```

The source carries a "Theory of operation" section in its header comment and a
description (inputs, effect, errors) above every function.

## Usage

```
a252768 scan [START] END [-t T] [-c CHUNK] [-k KMAX] [-r DEPTH] [-S STATE] [-i SECS] [-q]
a252768 verify P [-k KMAX] [-t T]
a252768 selftest [-t T]
```

Numbers may be written as `123`, `2^40`, `10^12`, `1e12` or `2^40-1`.

**scan** searches every prime in [START, END] (START defaults to 0). It prints
each prime of depth ≥ DEPTH (default 7) as it is found, with all its power
sums and the first composite one, then a table of a(n) for n ≤ KMAX (default
12, at most 16). With START > 0 the power sums up to START are first computed
in a fast totals-only pass, so any sub-range can be scanned on its own.

`-S FILE` writes a checkpoint every SECS seconds (default 60) and on Ctrl-C;
running the *same* command again resumes where it left off, and END may be
raised on resume. `-t` sets the thread count (default: all cores), `-c` the
chunk size (default: range/(32·threads), clamped to 10⁶..10⁹), `-q` silences
the status line on stderr.

```
./a252768 scan 1e12 -S a252768.state          # 2 minutes on an M1 Pro
./a252768 scan 1e13 -S a252768.state          # continues from the checkpoint
./a252768 scan 5e11 6e11 -r 8                 # only this sub-range, print depth >= 8
```

**verify P** recomputes S₁..S_KMAX at the prime P from scratch with a plain
single-threaded accumulation (no chunks, no offsets) and tests each sum, with
GMP as an independent second opinion. `-t T` with T > 1 uses the parallel
totals pipeline instead. With GMP, KMAX may exceed 16 here.

```
./a252768 verify 21814967833 -k 12
```

**selftest** checks the Miller–Rabin code against known (pseudo)primes and
20000 + 3000 random numbers versus GMP, recomputes the OEIS example and a(7),
reproduces a(1..7) with a chunked scan, checks that different chunk sizes and a
mid-range start give identical hits and sums, and round-trips a checkpoint.

`verify_a252768.py P [KMAX]` is a fully independent Python check (numpy
segmented sieve, exact Python integers, its own Miller–Rabin); it shares no
code with the C tool and needs about 30 s per 10¹⁰.

## Tips for the next run

**Resume, don't restart.** `a252768.state` holds the exact power sums at 10¹²
(and the a(n) found so far). Any `scan … -S a252768.state` with a larger END
continues from there; scanning from 0 again would waste 2 minutes, a scan
starting mid-range without the state file costs a totals-only pass over
[0, START] (about a third of a full scan: ~6 min to 10¹³, ~1 h to 10¹⁴).

**The command.** Log to a file, and keep the Mac awake — a scan that sleeps
halfway through is the most likely way to lose a day:

```
caffeinate -i ./a252768 scan 1e14 -S a252768.state -r 8 -i 120 2> scan_1e14.log | tee -a scan_1e14.txt
```

* `-r 8` prints only primes of depth ≥ 8. Depth-7 hits come at 4–5× the
  previous decade's count (83 in [10¹⁰, 10¹¹], 336 in [10¹¹, 10¹²], so
  ~1500 in [10¹², 10¹³]) and are only interesting as statistics; the depth
  histogram in the final summary counts them anyway.
* `-i 120` checkpoints every 2 minutes (default 60 s); the file is rewritten
  atomically, so a power cut costs at most that much work.
* Ctrl-C (or SIGTERM) finishes the chunks in flight, writes the checkpoint and
  prints the summary for the completed prefix; rerun the same command to go on.
* The status line on stderr shows the completion frontier, rate, ETA, and the
  best depth so far. Without a terminal (e.g. under `nohup`) it is printed
  once a minute instead.
* Default threads = all cores, default chunk = range/(32·threads) clamped to
  10⁶..10⁹ (10⁹ from 3·10¹¹ on); memory is ~40 MB of gap bytes per thread at
  chunk 10⁹, so nothing to tune. `-t 8` leaves two cores free if the machine
  is in use.

**How long, and what to expect.** At the measured 9·10⁹ numbers/s:
10¹³ ≈ 20 min (a(10) with 46 % probability), 10¹⁴ ≈ 3 h (a(10) with 98 %,
a(11) with 21 %), 10¹⁵ ≈ 31 h (a(11) with 73 %), 2·10¹⁵ ≈ 2.6 days (a(11)
with 90 %). See the estimate table above; `python3 estimate.py X` refreshes it
once X has been scanned without success.

**Splitting across machines.** A checkpoint can only continue from its own
position (with `-S`, a START argument is ignored in favour of the checkpoint).
To split [M, Y] between two machines, let machine A continue from the
checkpoint with `scan M -S a252768.state` and give machine B a plain
`scan M Y` (no state file): B first pays the totals-only pass over [0, M],
about a third of a full scan (~1 h for M = 10¹⁴, ~3 h for 3·10¹⁴), and its
a(n) table then says "no prime of depth ≥ n in [M, Y]" instead of
`a(n) > Y`; combine the two ranges by hand. The checkpoint file is plain text
(power sums at its position in decimal), so it can be copied or inspected.

**When a hit appears** (`p = … n = 10 …` on stdout, or `a(10) = … NEW` in the
summary):

1. `./a252768 verify P -k 16` — recomputes from scratch and shows GMP's
   verdict beside the tool's for every k, including k > 12 that the scan could
   not test. Single-threaded it takes 5 s per 2·10¹⁰ (40 min at 10¹³, 6 h at
   10¹⁴); `-t 10` uses the parallel totals pipeline instead (6 min at 10¹³,
   1 h at 10¹⁴) at the price of sharing the chunk code with the scan.
2. `python3 verify_a252768.py P 16` — the independent numpy/Python check,
   about 30 s per 10¹⁰ single-threaded: 9 h at 10¹³, days at 10¹⁴. For a
   term that far out, either let it run over a weekend or parallelise it
   (the per-segment histograms are independent; a `multiprocessing.Pool`
   over segments with the boundary primes merged afterwards would bring 10¹⁴
   to about 10 h on 10 cores).
3. For the OEIS: the sums beyond 3.3·10²⁴ are probable primes (24 Miller–Rabin
   bases + GMP's BPSW). A PARI/GP `isprime(S)` certificate (APR-CL, instant
   for ≤ 130-bit numbers) makes them rigorous; PARI is not installed here
   (`brew install pari`). Then extend `b252768.txt` and `DATA.txt`, and note
   the scan bound for the next term ("a(11) > X") in the entry. Ready-to-paste
   OEIS lines, and the state of the entry as of Oct 3 2026, are in
   `submission.md`.

**Above 1.5·10¹⁴** the tool stops testing k = 12 (S₁₂ needs more than 128
bits) and prints a note; k = 11 lasts to about 10¹⁶. A hit's depth beyond that
is settled in seconds by `verify`, which uses GMP for any k. An a(12) hunt
beyond 10¹⁶ would need wider accumulators for the top k: the running offsets
`O[k]` and chunk totals `T[k]`, plus the lazy Σ count(g)·g^k loop in
`check_chunk` — all cheap places, since S₅ and up are only evaluated for the
rare survivors of S₂..S₄.

## How it works

### The sequential problem, made parallel

S_k(p) is a running sum over *all* gaps below p, so a naive scan is inherently
sequential. The tool cuts the range into chunks and uses the fact that the sum
over a chunk depends only on the chunk's own gaps:

1. A worker takes the next chunk index from an atomic counter, sieves the
   chunk with primesieve (starting a little below the chunk to find the prime
   before it, so every prime's incoming gap is attributed to the chunk that
   contains the prime), and stores the gaps — one byte per prime (g/2, with
   an escape for the gap 1 and gaps ≥ 510) — together with a gap histogram.
2. From the histogram it forms the chunk totals T_k = Σ count(g)·g^k for
   k ≤ KMAX in 128-bit arithmetic.
3. **Ordered hand-off**: the worker waits until the running sums O_k at the
   start of its chunk have been published (chunk i publishes O_k(i+1) =
   O_k(i) + T_k(i) as soon as its *sieve* is done), then proceeds.
4. It walks its stored gaps a second time with the correct starting sums and
   tests candidates. Meanwhile the other workers do the same with their
   chunks, so the checking runs fully in parallel, the range is sieved exactly
   once, no barrier is needed and memory stays bounded (one chunk of gaps per
   thread; 10⁹ numbers ≈ 40 MB).

The bookkeeping ring is 4096 chunks deep; a completion frontier tracks the
lowest unfinished chunk, which is where checkpoints are written and where
`a(n) > X` statements are certified.

### Checking a chunk

S₁(p) = p − 2 is prime exactly when the gap into p is 2, so only twin primes
(about 1/ln p of all primes) are examined at all. S₂, S₃, S₄ are maintained
incrementally and tested in turn; the ~1 in 10⁵ primes surviving S₂..S₄ get
S₅, S₆, … evaluated from the running gap histogram (Σ count(g)·g^k over the
few hundred distinct gap sizes), so the cost per prime is a handful of adds.
Every sum is 128-bit with overflow detection; if some S_k would exceed 2¹²⁸,
depths ≥ k stop being tested and the tool says so (S₁₃ crosses 128 bits near
10¹³, S₁₂ near 1.5·10¹⁴, S₁₁ near 10¹⁶; with the default `-k 12` nothing is
lost below 1.5·10¹⁴).

### Primality

* n < 2⁶⁴: Miller–Rabin to the bases 2, 325, 9375, 28178, 450775, 9780504,
  1795265022, which is deterministic for all 64-bit n. Trial division by the
  primes up to 47 first (two 32-bit remainders) removes two thirds of the
  candidates before the expensive part. Everything runs in Montgomery
  arithmetic; the base-2 exponentiation multiplies by the base with a modular
  doubling.
* n ≥ 2⁶⁴: 128-bit Montgomery Miller–Rabin to the 24 prime bases 2..89. The
  first 13 bases are deterministic below 3.3·10²⁴ (Sorenson–Webster); above
  that the verdict is a strong probable-prime test, and `verify` adds GMP's
  independent BPSW + Miller–Rabin verdict. The sums that matter here have
  at most ~110 bits.

### Why the depths are correlated

By Fermat, g^k mod q depends only on k mod (q − 1), so S_k ≡ S_k′ (mod q)
whenever k ≡ k′ (mod q − 1). Once S₁ and S₂ are prime, no later S_k is
divisible by 3 (odd k ≡ S₁, even k ≡ S₂ mod 3), S₅ ≡ S₁ and S₆ ≡ S₂ (mod 5),
S₇ ≡ S₁ (mod 7), and so on. The boost for S_k is
B_k = Π q/(q−1) over the primes 3 ≤ q ≤ k, and the per-k pass rate
B_k·2/ln S_k(p) with S_k(p) ≈ 0.8^(k−1)·k!·p·(ln p)^(k−1) matches the measured
rates within a few percent (see the estimate section). This is why a(n) grows
more slowly than independent conditions would suggest: each extra k costs a
factor of only 12–16 in the count of qualifying primes, and the pass rate
shrinks only slowly with p.

## Verification

* `a252768 selftest` reproduces a(1..7) and cross-checks 23000 random
  Miller–Rabin verdicts against GMP.
* `a252768 verify P` recomputes the sums single-threaded from scratch (a code
  path without chunks or offsets) and shows the GMP verdict next to the
  tool's own for every k.
* `verify_a252768.py P` repeats the whole computation without primesieve or
  any of the C code.
* Inside every chunk the tool asserts that the published running sum S₁ equals
  (previous prime − 2), which pins the chunk hand-off to the actual primes.

Sums below 3.3·10²⁴ are proven prime by the deterministic bases; that covers
every sum of a(8) and S₁..S₈ of a(9). Sums above it — so far only S₉ of a(9),
and S₉ and up for future terms — have probable-prime status from 24
Miller–Rabin bases plus GMP's BPSW test; a PARI/GP `isprime` certificate
(APR-CL) makes them rigorous. All the sums involved have at most ~110 bits.

## Files

* `a252768.c` — the tool (single file, C11, pthreads, primesieve, optional GMP)
* `Makefile` — `make`, `make test`
* `verify_a252768.py` — independent Python/numpy verification of one prime; `verify_py_a8.txt`, `verify_py_a9.txt` are its transcripts for the new terms
* `estimate.py` — the calibrated density model: expected search ranges for a(10)..a(13) and the S_k bit sizes
* `scan_1e12.txt`, `scan_1e12.log` — output of the scan to 10¹² (all primes of depth ≥ 7)
* `a252768.state` — checkpoint of the scan (resume with the same `scan … -S a252768.state`)
* `b252768.txt`, `DATA.txt` — b-file and OEIS DATA line with the new terms
* `submission.md` — state of the OEIS entry (a(8), a(9) approved Sep 17 2026) and the items still worth submitting, as ready-to-paste OEIS lines
