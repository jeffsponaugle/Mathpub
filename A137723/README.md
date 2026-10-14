# A137723 — runs of numbers with a prime gap in their factorization

Tool for computing and extending [OEIS A137723](https://oeis.org/A137723):

> First occurrence of a set of n consecutive numbers having at least one prime
> gap in their factorization: a(n) = smallest number of this set.

A number has a *prime gap in its factorization* (A073490 > 0) when its distinct
prime factors are not consecutive primes: 10 = 2·5 skips 3, 84 = 2²·3·7 skips 5.
Numbers **without** a gap (A073491, "gap-free" below) are 1, the primes, prime
powers, and products of consecutive primes with any exponents: 12 = 2²·3,
35 = 5·7, 2940 = 2²·3·5·7², 65536 = 2¹⁶.

If g < g' are consecutive gap-free numbers, then g+1 .. g'−1 is a maximal run of
n = g'−g−1 numbers that all have a gap, and a(n) is the smallest g+1 over all
runs of length exactly n. Example: 83 and 89 are consecutive gap-free numbers
and 84..88 all have gaps, so a(5) = 84.

Until Sep 16 2026 the OEIS entry listed a(1..31) and noted a(32) > 10¹¹. The terms
a(32)..a(193) found here were submitted that day and approved on Sep 17 2026
(b-file n = 1..193). `submission.md` holds the follow-up edit package.

## Results

Everything below was computed with this tool on an Apple M1 Pro (10 cores) and
the new terms were verified independently with `verify_run.py` (pure Python:
its own Miller–Rabin and Pollard rho) and, for the prime endpoints, `openssl prime`.

**The sequence is now known for n = 1..193** (the OEIS entry stops at a(31)).
`b137723.txt` is the b-file and `DATA.txt` the OEIS DATA line; both are produced by `make_bfile.py`.
The first 61 terms:

```
10, 33, 20, 55, 84, 114, 390, 513, 182, 200, 468, 2941, 774, 65522, 1832, 1261, 1130, 1332, 1638, 524289, 1952, 4298, 4524, 69960, 5120, 16385, 2972, 4832, 5352, 10801, 5592, 18014398509481952, 8468, 1119745, 9552, 69264, 39462, 33554394, 20810, 74960, 38502, 107521, 15684, 274877906900, 81464, 214994, 28230, 124368, 31908, 2417851639229258349412302, 19610, 748388, 35618, 306109962, 82074, 268435400, 44294, 360092, 43332, 2799300, 34062
```

followed by a(62) = 23384026197294446691258957323460528314494920687554 = 2¹⁶⁴ − 62 (50 digits),
a(63) = 87494, and so on; 38 of the 193 terms exceed 10¹¹, the largest being
a(176) = 2³⁷² − 176 with 112 digits.

The terms the exhaustive scan could not reach were found by the structural
search, every one of them adjacent to a power of two:

| n   | a(n)                                     | form         | search                 |
|-----|------------------------------------------|--------------|------------------------|
| 32  | 18014398509481952                        | 2⁵⁴ − 32     | 128-bit, exhaustive to 2¹²⁷ |
| 44  | 274877906900                             | 2³⁸ − 44     | 128-bit, to 2¹²⁷       |
| 50  | 2417851639229258349412302                | 2⁸¹ − 50     | 128-bit, to 2¹²⁷       |
| 62  | 23384026197294446691258957323460528314494920687554 | 2¹⁶⁴ − 62 | GMP, exhaustive to 2¹⁰⁰⁰ |
| 68  | 562949953421313                          | 2⁴⁹ + 1      | 128-bit, to 2⁶⁴        |
| 74  | 9671406556917033397649409                | 2⁸³ + 1      | 128-bit, to 2¹²⁷       |
| 80  | 562949953421232                          | 2⁴⁹ − 80     | 128-bit, to 2⁶⁴        |
| 86  | 1099511627690                            | 2⁴⁰ − 86     | 128-bit, to 2⁶⁴        |
| 92  | 1152921504606846884                      | 2⁶⁰ − 92     | 128-bit, to 2⁶⁴        |
| 94  | 31701690482689                           | 2²⁹·3¹⁰ + 1  | 128-bit, to 2⁶⁴        |
| 98  | 81129638414606681695789005144065         | 2¹⁰⁶ + 1     | 128-bit, to 2¹²⁷       |
| 104 | 9903520314283042199192993793             | 2⁹³ + 1      | 128-bit, to 2¹²⁷       |
| 110 | 9007199254740882                         | 2⁵³ − 110    | 128-bit, to 2⁶⁴        |
| 116 | 81129638414606681695789005143948         | 2¹⁰⁶ − 116   | 128-bit, to 2¹²⁷       |
| 122 | 254629497041810760783555711051172270131433549208242031329517556169297662470417088272924550 | 2²⁹⁷ − 122 | GMP, to 2¹⁰⁰⁰ |
| 128 | 2251799813685120                         | 2⁵¹ − 128    | 128-bit, to 2⁶⁴        |
| 134 | 4611686018427387905                      | 2⁶² + 1      | 128-bit, to 2⁶⁴        |
| 140 | 158456325028528675187087900532           | 2⁹⁷ − 140    | 128-bit, to 2¹²⁷       |
| 146 | 649037107316853453566312041152513        | 2¹⁰⁹ + 1     | 128-bit, to 2¹²⁷       |
| 152 | 89202980794122492566142873090593446023921512 | 2¹⁴⁶ − 152 | GMP, to 2¹⁰⁰⁰         |
| 154 | 383808888404050968577                    | 2²⁴·3²⁸ + 1  | GMP, to 2¹⁰⁰⁰          |
| 158 | 18014398509481985                        | 2⁵⁴ + 1      | 128-bit, to 2⁶⁴        |
| 164 | 10633823966279326983230456482242756609   | 2¹²³ + 1     | 128-bit, to 2¹²⁷       |
| 170 | 38685626227668133590597633               | 2⁸⁵ + 1      | 128-bit, to 2¹²⁷       |
| 176 | 9619630419041620901435312524449124464130795720328478190417063819395928166869436184427311097384012607618805661520 | 2³⁷² − 176 | GMP, to 2¹⁰⁰⁰ |
| 182 | 6427752177035961102167848369364650410088811975131171341205322 | 2²⁰² − 182 | GMP, to 2¹⁰⁰⁰ |
| 188 | 13479973333575319897333507543509815336818572211270286240551805124609 | 2²²³ + 1 | GMP, to 2¹⁰⁰⁰ |

(a(94), a(114) = 2³⁶·3⁵ + 1, a(124), a(132), a(154) and a few others in
`next_1e11.txt` are adjacent to other even gap-free numbers such as 2²⁹·3¹⁰ or
2²⁴·3²⁸; they are the cases where d has no factor 3 — for a(154), d = 155 = 5·31
lets E = 2^a·3^b have a prime partner.) Every a(n) with n ≡ 2 (mod 6) found so
far is adjacent to a power of two, as step 3 of "How it works" predicts.

**Open cases.** a(194) is the first unknown term; the odd n are known up to
385 and the first unknown odd term is a(387). See "Extending the sequence"
below for the current lower bounds, the cost of each open case and the
workflow for adding a term. The complete scan table for n ≤ 463 is in
`next_1e11.txt`; `search_2e127.txt` holds the 2¹²⁷ searches.

### The a(32) run in detail

`./a137723 run 2^54-32` prints the whole run. Consecutive primes are joined
with `*`; a ` | ` marks a prime gap, i.e. every prime strictly between the two
neighbours is missing from the factorization (`2^5 | 127` skips 3, 5, 7, …, 113).

```
18014398509481951 = prime (2^54 - 33)           <- gap-free lower bound
18014398509481952 = 2^5 | 127 | 4432676798593
18014398509481953 = 3 | 347 | 17304897703633
18014398509481954 = 2 | 17 | 1051 | 311533 | 1618207
18014398509481955 = 5*7^2 | 1892477 | 38852867
18014398509481956 = 2^2*3^3 | 709 | 235260911423
18014398509481957 = 11 | 7537 | 23831 | 9117721
18014398509481958 = 2 | 149 | 60451001709671
18014398509481959 = 3 | 13 | 41 | 83 | 167 | 211 | 293 | 13147
18014398509481960 = 2^3 | 5 | 71 | 6343098066719
18014398509481961 = 5029469 | 3581769469
18014398509481962 = 2*3 | 7 | 29 | 137 | 320083 | 337279
18014398509481963 = 47 | 277 | 521 | 2655855337
18014398509481964 = 2^2 | 19 | 659 | 359683701571
18014398509481965 = 3^2*5 | 179 | 17863 | 125198701
18014398509481966 = 2 | 439 | 29023 | 706940639
18014398509481967 = 20297 | 232381 | 3819331
18014398509481968 = 2^4*3 | 11 | 31 | 251 | 601 | 1801 | 4051
18014398509481969 = 7 | 67 | 1277 | 5573 | 5397181
18014398509481970 = 2 | 5 | 839 | 910229 | 2358887
18014398509481971 = 3 | 17^2 | 7757 | 31729 | 84421
18014398509481972 = 2^2 | 13 | 23 | 2192917 | 6868571
18014398509481973 = 43 | 418939500220511
18014398509481974 = 2*3^2 | 4357 | 229699315399
18014398509481975 = 5^2 | 173 | 4057 | 31033 | 33083
18014398509481976 = 2^3 | 7 | 103 | 2143 | 11119 | 131071
18014398509481977 = 3 | 127241 | 149489 | 315691
18014398509481978 = 2 | 2203 | 5741 | 712176643
18014398509481979 = 11 | 1637672591771089
18014398509481980 = 2^2*3*5 | 53 | 157 | 1613 | 2731 | 8191
18014398509481981 = 36217 | 497401731493
18014398509481982 = 2 | 6361 | 69431 | 20394401
18014398509481983 = 3^4 | 7 | 19 | 73 | 87211 | 262657
18014398509481984 = 2^54                          <- gap-free upper bound
```

Observations:

* Every one of the 32 numbers has at least one gap (A073490 ≥ 1). The smallest
  count is 1 — for instance 18014398509481961 = 5029469 · 3581769469 and
  18014398509481973 = 43 · 418939500220511 — and the largest is 7, for
  18014398509481959 = 3 · 13 · 41 · 83 · 167 · 211 · 293 · 13147.
* Only three numbers contain a block of consecutive primes at all: 5·7² in
  …955, 2·3 in …962 and 2²·3·5 in …980 (plus the pure powers 2⁵, 3⁴ and 17²).
  Everything else jumps straight from its smallest prime to a much larger one.
* The bounds are the interesting part. The run ends at 2⁵⁴, a pure power of two
  and therefore gap-free, and it starts right after the prime 2⁵⁴ − 33. The
  sixteen odd numbers in between are all composite, so the run is really a
  prime gap of 33 sitting directly below 2⁵⁴ with no prime power or
  consecutive-prime product inside it. As explained below, that is essentially
  the only way a run of length 32 can arise, which is why a(32) ≈ 1.8·10¹⁶
  while a(31) = 5592 and a(33) = 8468.

### Timings (M1 Pro, 10 threads)

| command                          | time    |
|----------------------------------|---------|
| `scan 10^9`                      | 0.2 s   |
| `scan 10^11`                     | 2.7 s   |
| `search 32 -l 2^127` (k ≤ 4)     | 0.25 s  |
| `search 44 -l 2^100` (any k)     | 10 s    |
| `search 44 -l 2^127` (any k)     | ~6 min  |
| `search 122 -l 2^127` (k ≤ 12)   | 109 s   |
| `search 140 -l 2^127` (k ≤ 14)   | 188 s   |
| `search 62 -l 2^1000` (GMP)      | 0.5 s   |
| `search 122 152 -l 2^1000` (GMP) | 0.9 s   |
| `next -N 10^11 -l 2^64`          | ~4 min  |

## Building

Requires [primesieve](https://github.com/kimwalisch/primesieve)
(`brew install primesieve`) and a C11 compiler with `unsigned __int128`.
[GMP](https://gmplib.org) (`brew install gmp`) is optional; the Makefile
detects it and enables `search -l` from 2¹²⁷ upward.

```bash
make            # builds ./a137723
make test       # runs the self-test (about half a second)
```

## Usage

```
a137723 scan N [-t T] [-m MAXN] [-o FILE]     exhaustive: every a(n) <= N
a137723 search n [n ...] [-l LIMIT] [-t T]    structural search for even n (default LIMIT 2^64);
                                              LIMIT >= 2^127 (e.g. 2^1000) uses GMP, --gmp forces it,
                                              --max-nodes X caps the estimated enumeration (default 5e9)
a137723 next [-N SCAN] [-l LIMIT] [-t T] [-m MAXN] [-o FILE]
                                              scan to SCAN (default 10^10), then search the missing even n
a137723 check m                               factor m and count its prime gaps (A073490)
a137723 run m                                 show the maximal run of gapful numbers around m
                                              (both accept numbers above 2^128 when built with GMP)
a137723 selftest [-t T]
```

Numbers may be written as decimal, `2^k`, `10^k`, `1e15`, or sums/differences
such as `2^54-32`. `-t` defaults to the number of CPUs.

Examples:

```bash
./a137723 scan 10^11                 # reproduces a(1..31), confirms a(32) > 10^11 (about 3 s on 10 cores)
./a137723 search 32 -l 2^127         # a(32) in a quarter of a second
./a137723 search 74 104 116 -l 2^127   # hard cases in 128-bit mode, seconds to minutes each
./a137723 search 62 -l 2^1000          # GMP mode: a(62) = 2^164-62 in half a second
./a137723 search 194 200 204 -l 2^1000 # next open terms
./a137723 run 2^54-32                # print the 32 factorizations of the a(32) run
./a137723 run 2^223+1                # the a(188) run, 68-digit numbers (GMP)
./a137723 check 2940                 # 2^2*3*5*7^2, no gap
python3 verify_run.py 18014398509481951 18014398509481984   # independent check of a run
```

## How it works: the search technique

Two very different engines are combined. The **exhaustive scan** is the
ground truth: it finds every a(n) below its limit N and proves that anything
it did not find is larger than N. The **structural search** covers the even n
whose first run lies far beyond any feasible scan, by exploiting a parity
argument that pins one end of the run to a very restricted set of numbers.

### 1. Reformulating the problem

The definition talks about runs of consecutive integers with a prime gap in
their factorization. It is easier to look at their complement. Call a number
*gap-free* if its distinct prime factors are consecutive primes (A073491):
1, every prime, every prime power, and products such as 2²·3, 5·7, 2·3·5·7,
2⁴·3³·5². Then

    g < g' consecutive gap-free numbers  <=>  g+1 .. g'-1 is a maximal run of n = g'-g-1 gapful numbers

and a(n) is simply g+1 for the smallest such pair with g'−g = n+1. So the
whole problem is about *differences between consecutive gap-free numbers*,
exactly like prime gaps are differences between consecutive primes — and
since almost all gap-free numbers are primes, the two problems are close
cousins: a(n) for odd n is usually just the start of the first prime gap of
size n+1 (a(31) = 5592 comes from the prime gap 5591 → 5623), unless a prime
power or a consecutive-prime product happens to fall inside that gap (the
first prime gap of 34, 1327 → 1361, is spoiled by 1331 = 11³ and
1350 = 2·3³·5², which is why a(33) = 8468 instead of 1328).

### 2. `scan`: exhaustive enumeration of gap-free numbers

Gap-free numbers are sparse: about N/ln N primes plus O(√N) composites (a
composite gap-free number has smallest prime p ≤ √N, since it is at least p²).
Rather than factoring every integer up to N, the scan generates the gap-free
numbers directly and reads off the differences:

* primes come from a `primesieve_iterator` — one iterator per thread;
* composite gap-free numbers are enumerated recursively: for each prime
  p ≤ √N take p^e (e ≥ 2), then optionally multiply by powers of the next
  prime, then the next, and so on while the product stays below N. There are
  only 17131 of them below 10⁹, 91043 below 10¹¹ and about half a million
  below 10¹³; they are sorted once and shared read-only by all threads.

Worker threads take fixed-size chunks of [1, N] from an atomic counter, merge
the prime stream with the composite list, and record the first occurrence of
every difference n = g'−g−1. The pair (g, g') belongs to the thread whose chunk
contains g, and each thread simply keeps iterating past its chunk end until it
has seen the first gap-free number beyond it, so chunk boundaries need no
special handling. primesieve is fast enough that a scan to 10¹¹ takes about
3 s on ten cores; the bound "a(32) > 10¹¹" from the OEIS entry is reproduced
in that time, and the scan is what produced every odd-n term.

### 3. `search`: structural search for even n

The scan cannot reach a(32) ≈ 1.8·10¹⁶ (that would be about 10⁵ times more
work than 10¹¹). The structural search gets there in a quarter of a second by
throwing away almost all candidates before looking at them.

**Step 1 — parity.** For even n, d = n+1 is odd, so the two bounding gap-free
numbers L and R = L + d have different parity: exactly one of them is even.

**Step 2 — the shape of even gap-free numbers.** An even gap-free number
contains the prime 2, and its primes must be consecutive starting from 2, so
it is necessarily

    E = 2^e1 · 3^e2 · 5^e3 · … · p_k^ek     (the first k primes, every exponent ≥ 1).

These numbers are extremely sparse: 2,537,666 below 2⁶⁴ and 1.74·10¹⁰
below 2¹²⁷ (compare with 1.7·10³⁸ integers). Enumerating all of them is cheap,
and for each E only two partners have to be examined, O = E + d and O = E − d.
If O is gap-free and the numbers strictly between E and O all have a gap, then
(min, max) is a valid pair. Since every possible even endpoint is visited, the
search is **exhaustive** for L ≤ limit − d: the smallest hit is a(n), and if
there is no hit then a(n) > limit − d + 1. This is a proof, not a heuristic.

**Step 3 — propagating the constraints of d.** The odd partner O = E ± d is
tightly constrained by the primes that E and d share:

* if p | E and p | d then p | O;
* if p | E and p ∤ d then p ∤ O.

Because O's prime factors must themselves be consecutive, the odd primes ≤ p_k
that divide d must form a contiguous block of consecutive primes — otherwise no
E with k primes can work at all, which caps k (`kmax` in the program). And O can
be *prime* only if no odd prime ≤ p_k divides d (`kprime`). Worked example for
n = 32, d = 33 = 3·11:

* k = 1, E = 2^a: no constraint; O = 2^a ± 33 may be prime, a prime power, or
  a product of consecutive primes ≥ 5.
* k = 2, E = 2^a·3^b: 3 | O, so O = 3^x·5^y·7^z·… (a product starting at 3).
* k = 3, 4 (E also contains 5, or 5 and 7): 3 | O but 5 ∤ O, so O must be a pure
  power of 3.
* k ≥ 5 (E contains 11): 3 | O and 11 | O but 5 ∤ O — impossible for a gap-free
  number. So `kmax` = 4.

Hence a run of exactly 32 with a *prime* endpoint must sit right next to a
power of two, and the only alternatives are the astronomically rare
coincidences 2^a·3^b·… ± 33 = 3^x·5^y·… . With only 36 powers of two below
10¹¹, and each 2^a having perhaps a 1–2 % chance of having 2^a ± 33 prime with
no other prime in between, it is no surprise that nothing exists below 10¹¹; the
first success is 2⁵⁴ − 33. The same argument applies whenever 3 | n+1, i.e. for
every n ≡ 2 (mod 6): 38, 44, 50, 56, 62, 68, 74, …, and indeed every such term
found is adjacent to a power of two. For other even n more shapes of E are
allowed — for d = 35 = 5·7 both 2^a and 2^a·3^b may have prime partners — and
those terms are correspondingly smaller (a(34) = 1119745 = 2⁹·3⁷ + 1).

For odd n no such parity trick exists: both endpoints are odd and usually
both prime, so the even-endpoint enumeration cannot be exhaustive and the tool
refuses odd n in `search`. Those terms are small anyway and come from `scan`.

**Step 4 — the gap-free test.** Deciding whether an arbitrary 128-bit O is
gap-free would normally mean factoring it, but the structure of gap-free numbers
makes it cheap:

* trial division by the primes below 2¹⁶; as soon as the smallest prime p is
  found, divide out p, then the *next* prime, then the next … the number is
  gap-free iff this chain reaches 1. Almost all candidates die within a few
  divisions (e.g. 3 | O but 5 ∤ O with a cofactor left);
* if there is no factor below 2¹⁶: a Miller–Rabin test (deterministic bases
  below 2⁶⁴, 24 bases above) — a prime is gap-free;
* otherwise O is composite with all prime factors above 2¹⁶. If it were
  gap-free with t prime factors (counted with multiplicity), its primes would be
  consecutive and nearly equal, so its smallest prime p₁ satisfies
  p₁ < O^(1/t) < p_s: p₁ is one of the t−1 largest primes below the integer
  t-th root. For each t with (2¹⁶)^t < O (t ≤ 7 for 128-bit numbers) those few
  candidates are generated with prev-prime steps and checked with the chain
  division above. No general factoring is ever needed.

The interior check (are the n numbers between E and O all gapful?) uses the
same routine and is only run when O passed, which is rare.

**Step 5 — parallelism and arithmetic.** The enumeration is a depth-first
walk over exponent vectors. The top three levels (2^a, 2^a·3^b, 2^a·3^b·5^c)
are materialised as a task list — about 12 000 tasks up to 2⁶⁴ and 93 000 up to
2¹²⁷ — and worker threads pull tasks from an atomic counter, recursing into
7, 11, 13, … themselves; the biggest subtrees come first, so the load balances
well. All values are `unsigned __int128`; modular multiplication above 2⁶⁴ is
a plain shift-and-add, which is slow (~0.5 µs) but only reached for the handful
of candidates without small factors. The enumeration runs at roughly 50 million
E per second on ten cores, so a fully unconstrained case (`kmax` = ∞, 1.7·10¹⁰
even gap-free numbers below 2¹²⁷) takes about six minutes and a constrained one
like d = 33 (k ≤ 4, about a million E) a fraction of a second.

**Step 6 — beyond 128 bits (GMP mode).** Above 2¹²⁷ the same theorem is
applied more selectively, because enumerating *every* even gap-free number is
no longer an option (there are well over 10⁴⁰ of them below 2¹⁰⁰⁰). Let p_j be the
smallest odd prime dividing d and {p_j, …, p_m} the maximal block of
consecutive primes starting at p_j that all divide d. For an even endpoint
with k primes the odd partner O = E ± d is

* coprime to the first k primes when k < j — it may be prime, so E has to be
  enumerated;
* a product over {p_j, …, p_k} times an optional consecutive continuation when
  j ≤ k ≤ m — open-ended, so again E is enumerated;
* **exactly** a product over the fixed set {p_j, …, p_m} when k > m — so for
  those k the program enumerates O instead, which is a tiny set (the powers of
  3 when the block is just {3}), and checks whether O ± d is an even gap-free
  number with more than m primes.

For the blockers d = 63, 123, 153 the block is {3}, so the whole search is the
line 2^a ± d plus 2^a·3^b ± d plus the powers of 3: about 316 000 candidates
below 2¹⁰⁰⁰, half a second on ten cores, most of it Miller–Rabin tests on the
2^a ± d values. The gap-free test is the same trial division / probable-prime /
t-th-root routine written with `mpz_t`; below 2¹²⁷ it falls back to the 128-bit
code. The program prints the estimated enumeration size before starting and
refuses if it exceeds `--max-nodes`. What remains expensive is a d whose block
of small prime factors is long: for d = 105 = 3·5·7 the E side runs to k = 3
(45 million candidates below 2¹⁰⁰⁰, minutes) and for d = 245 = 5·7² to k = 4
(4·10⁹, so choose a lower limit such as 2⁴⁰⁰).

**Limits.** Everything is exhaustive only up to the chosen limit, and the
enumeration size grows like (ln limit)^m, so a large block of small prime
factors in d is the one thing that keeps a term out of reach. Primality above
2⁶⁴ is probabilistic; see "Verification".

## Extending the sequence

**Where things stand.** a(1..193) are known (b-file `b137723.txt`). Every odd
n ≤ 385 is known from the scan to 10¹¹; the first unknown odd term is a(387),
and 24 odd n ≤ 463 are unknown in all. 70 even n ≤ 463 are unknown; the
structural pass to 2⁶⁴ (`next -N 10^11 -l 2^64`) proved a(n) > 2⁶⁴ − n for each
of them, and the 2¹²⁷ / 2¹⁰⁰⁰ searches have so far only been run for the even
n ≤ 188.

**Even n.** The table lists the open even n ≤ 300 with the structure that
decides the cost of the GMP search (see step 6 of "How it works"): the block of
small primes dividing d, how many primes the even endpoint E may have on the
E side, the estimated number of candidates to 2¹⁰⁰⁰, and a limit that keeps the
enumeration around 10⁸ or less. "2^a only" means a prime odd partner forces
E to be a power of two (3 | d); "k ≤ 2" means E = 2^a·3^b may also work. Runs
of ~300 k candidates take half a second; ~45 M take a few minutes, longer when
the block does not start at 3 because the k ≤ 2 partners then need Miller–Rabin
tests; the 4 G case (d = 245 = 5·7²) should be run to 2⁴⁰⁰ first.

| n | d = n+1 | block of d's small primes | E side k ≤ | candidates to 2¹⁰⁰⁰ | suggested limit | a(n) > |
|---|---------|---------------------------|------------|--------------------|-----------------|--------|
| 194 | 3·5·13 | {3, 5} | 3 (2^a only for a prime partner) | 45 M | 2^1000 | 2⁶⁴ − 194 |
| 200 | 3·67 | {3} | 2 (2^a only for a prime partner) | 315 k | 2^1000 | 2⁶⁴ − 200 |
| 204 | 5·41 | {5} | 3 (k ≤ 2 for a prime partner) | 45 M | 2^1000 | 2⁶⁴ − 204 |
| 206 | 3^2·23 | {3} | 2 (2^a only for a prime partner) | 315 k | 2^1000 | 2⁶⁴ − 206 |
| 212 | 3·71 | {3} | 2 (2^a only for a prime partner) | 315 k | 2^1000 | 2⁶⁴ − 212 |
| 214 | 5·43 | {5} | 3 (k ≤ 2 for a prime partner) | 45 M | 2^1000 | 2⁶⁴ − 214 |
| 218 | 3·73 | {3} | 2 (2^a only for a prime partner) | 315 k | 2^1000 | 2⁶⁴ − 218 |
| 224 | 3^2·5^2 | {3, 5} | 3 (2^a only for a prime partner) | 45 M | 2^1000 | 2⁶⁴ − 224 |
| 230 | 3·7·11 | {3} | 2 (2^a only for a prime partner) | 315 k | 2^1000 | 2⁶⁴ − 230 |
| 234 | 5·47 | {5} | 3 (k ≤ 2 for a prime partner) | 45 M | 2^1000 | 2⁶⁴ − 234 |
| 236 | 3·79 | {3} | 2 (2^a only for a prime partner) | 315 k | 2^1000 | 2⁶⁴ − 236 |
| 242 | 3^5 | {3} | 2 (2^a only for a prime partner) | 315 k | 2^1000 | 2⁶⁴ − 242 |
| 244 | 5·7^2 | {5, 7} | 4 (k ≤ 2 for a prime partner) | 4.0 G | 2^400 | 2⁶⁴ − 244 |
| 248 | 3·83 | {3} | 2 (2^a only for a prime partner) | 315 k | 2^1000 | 2⁶⁴ − 248 |
| 254 | 3·5·17 | {3, 5} | 3 (2^a only for a prime partner) | 45 M | 2^1000 | 2⁶⁴ − 254 |
| 260 | 3^2·29 | {3} | 2 (2^a only for a prime partner) | 315 k | 2^1000 | 2⁶⁴ − 260 |
| 264 | 5·53 | {5} | 3 (k ≤ 2 for a prime partner) | 45 M | 2^1000 | 2⁶⁴ − 264 |
| 266 | 3·89 | {3} | 2 (2^a only for a prime partner) | 315 k | 2^1000 | 2⁶⁴ − 266 |
| 272 | 3·7·13 | {3} | 2 (2^a only for a prime partner) | 315 k | 2^1000 | 2⁶⁴ − 272 |
| 274 | 5^2·11 | {5} | 3 (k ≤ 2 for a prime partner) | 45 M | 2^1000 | 2⁶⁴ − 274 |
| 278 | 3^2·31 | {3} | 2 (2^a only for a prime partner) | 315 k | 2^1000 | 2⁶⁴ − 278 |
| 284 | 3·5·19 | {3, 5} | 3 (2^a only for a prime partner) | 45 M | 2^1000 | 2⁶⁴ − 284 |
| 290 | 3·97 | {3} | 2 (2^a only for a prime partner) | 315 k | 2^1000 | 2⁶⁴ − 290 |
| 294 | 5·59 | {5} | 3 (k ≤ 2 for a prime partner) | 45 M | 2^1000 | 2⁶⁴ − 294 |
| 296 | 3^3·11 | {3} | 2 (2^a only for a prime partner) | 315 k | 2^1000 | 2⁶⁴ − 296 |

The open even n between 300 and 463 are listed in `next_1e11.txt`. The
search is exhaustive up to its limit, so a result is final; a "no run found"
only raises the lower bound to limit − n + 1.

```bash
./a137723 search 194 -l 2^1000            # first open term, a few minutes
./a137723 search 200 206 212 218 -l 2^1000   # block {3} cases, half a second each
./a137723 search 244 -l 2^400             # long block: use a lower limit first
```

**Odd n.** For odd n both bounding gap-free numbers are odd and in practice
both prime, so a(n) is the start of the first prime gap of n+1 that contains no
prime power or consecutive-prime product. There is no structural shortcut; the
exhaustive scan is the tool, and it is fast: the first prime gap of 388 lies
beyond 10¹¹, and `scan 10^12` (about 30 s) or `scan 10^13` (a few minutes, half
a million composite gap-free numbers) will fill in a(387) and several later odd
terms. `next -N 10^13 -l 2^64` does the scan and the 2⁶⁴ structural pass in
one go and rewrites the whole table.

**Adding a term.** After `search` reports a(n):

1. run the independent check, `python3 verify_run.py L R` (L and R are printed
   by `search`), and for a prime endpoint above 2⁶⁴ optionally `openssl prime L`;
2. append `n a(n) form limit` to `structural_terms.txt` (use `-` for the form
   if the endpoint is not a simple power product);
3. run `python3 make_bfile.py`, which re-evaluates every form, checks the
   scan and search agree, and rewrites `b137723.txt` and `DATA.txt`;
4. add the row to the results table above.

**Submitting to the OEIS.** The b-file is the deliverable (the DATA field only
holds about 260 characters, roughly a(1..45) here). Terms found by the scan are
unconditional. Terms found by the structural search are exhaustive up to the
stated limit, but the primality of an endpoint above 2⁶⁴ was established with a
strong probable-prime test only, so certify those primes (2⁵⁴ − 33 and
2³⁸ − 45 are below 2⁶⁴ and already deterministic; the others are 2⁸¹ − 51,
2¹⁶⁴ − 63, 2⁸³ + 75, … up to 2³⁷² − 177) with a proving algorithm such as
PARI/GP `isprime` (APR-CL) or Primo (ECPP) before submission; all are far below
the sizes where that takes more than seconds. Mention in the entry that the
even-n terms come from the parity argument of step 3, since that is what makes
the huge values credible.

## Verification

* `selftest` compares the gap-free test against brute-force factorization for
  every m ≤ 2·10⁶, compares the multithreaded scan against a brute-force scan and
  against all 31 OEIS terms, checks the primality code against known primes and
  strong pseudoprimes (including 128-bit ones), exercises the root trick on
  products of consecutive primes near 2³², 2⁴¹ and 2⁶², reproduces every even
  OEIS term with the structural search, and compares the structural search with
  the exhaustive scan for all 76 even run lengths occurring below 10⁷.
* `search` re-checks the reported minimum with a completely separate full
  factorization (trial division + Pollard–Brent rho + Miller–Rabin). For runs
  above 2¹⁰⁰ or so this can report INCOMPLETE when rho cannot split some
  interior number in its budget (it happened for a(146) and a(164)); that is a
  limitation of the display factorizer, not of the search.
* `verify_run.py L R` is an independent pure-Python check of any run. Its
  verdict never needs a complete factorization: a number with no prime factor
  below 10⁵ is gap-free only if it is prime or a product of a few nearly equal
  consecutive primes, which the t-th-root test settles with a handful of
  primality tests. It certified every term listed above, including a(146) and
  a(164), in about 10 s each.

Primality above 2⁶⁴ is a strong probable-prime test (24 bases in the 128-bit
code, 30 rounds of `mpz_probab_prime_p` in GMP mode); before submitting a term
whose endpoints exceed 2⁶⁴ to the OEIS, certify the prime endpoint with PARI/GP
`isprime` or similar. The a(32) and a(44) endpoints are below 2⁶⁴, where the
test is deterministic. The prime endpoints of a(62), a(122) and a(152)
(2¹⁶⁴ − 63, 2²⁹⁷ − 123, 2¹⁴⁶ − 153), a(154), a(176), a(182) and a(188) were
re-checked with `verify_run.py` (and, for the first three, `openssl prime`).

## Files

* `a137723.c` — the tool (single file). Its header comment is a full theory of operation
  (definitions, the three methods with their correctness lemmas, the gapless test, rigor, threading,
  limits, self-test, source map), and every function has a block describing its inputs, outputs
  and error conditions
* `Makefile`
* `verify_run.py` — independent verifier for one run
* `next_1e11.txt` — table produced by `a137723 next -N 10^11 -l 2^64` (plus its log)
* `search_2e127.txt` — output of `a137723 search … -l 2^127` for n = 74, 104, 116, 122, 140, 146, 152, 164, 170
* `b137723.txt` — OEIS b-file of the consecutive terms (`n a(n)` per line, `#` header comments)
* `DATA.txt` — the same terms as an OEIS DATA line
* `structural_terms.txt` — the terms found by `search` (n, a(n), form, search limit); edit this when a new one is found
* `submission.md` — OEIS status check, follow-up edit in field format, run plan for more terms, certification list, optional derived sequences
* `primes_to_certify.txt` — the 16 probable-prime endpoints above 2⁶⁴ (for PARI/GP `isprime`)
* `make_bfile.py` — merges `next_1e11.txt` with `structural_terms.txt`, re-evaluates the forms, and rewrites `b137723.txt` and `DATA.txt`
