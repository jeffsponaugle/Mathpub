# A252768 — OEIS submission notes

Checked against the live entry on Oct 3 2026.

## Where the entry stands

Revision #21 of [A252768](https://oeis.org/A252768), approved Sep 17 2026
(edited by Jeff Sponaugle Sep 16, reviewed by Hugo Pfoertner, approved by
Alois P. Heinz):

```
%S 5,5,13,14593,372313,2315773,541613713,7952072743,21814967833
%C Below 10^12 there are 401 primes of depth 7, 32 of depth 8 and 5 of depth 9, so each extra k
   costs a factor of roughly 8 to 13. That heuristic puts a(10) below 10^13 with about 75%
   probability and very high probability below 10^14. - Jeff Sponaugle, Sep 16 2026
%K nonn,hard,more
%E a(8)-a(9) from Jeff Sponaugle, Sep 16 2026
```

So **a(8) and a(9) are in**; nothing else has changed since, and nobody has
added a(10). The related entries A247177 (#25, 2020), A247178 (#24, 2021),
A251623 (#37, 2021) and A252655 (#39, Nov 2025) are untouched and nothing from
this work applies to them (A252655 already has a 1000-term b-file).

## NEW (Oct 14 2026): a(11) = 1009809788912233 — submit together with a(10)

Found by jeff-model2's continuation above 10^15 (37 minutes in, every chunk below it
complete); the scans of [10^12, 10^14] (jeff-model1) and [10^14, 10^15] (jeff-model2)
contain no prime of depth 11, so it is a(11). PARI/GP `isprime` proves S_1..S_11
prime (`certify_a11.txt`). The sums were recomputed from scratch on jeff-model3
(`verify 1009809788912233 -k 14 -t 32`, 6 h: identical S_1..S_11, exact prime
count, GMP agreeing) and S_12 = 8066007422440485148327075748850168631297 is
composite (7043 divides it), so the depth is exactly 11; since a(n) is
non-decreasing, the bound to submit alongside is `a(12) > 1009809788912233`.
Submit-ready.

```
%S A252768 5,5,13,14593,372313,2315773,541613713,7952072743,21814967833,3770303151073,1009809788912233
%e A252768 a(11) = 1009809788912233: the sums of the k-th powers of the gaps between the primes <= 1009809788912233 are prime for k = 1..11 (1009809788912231, 62234731803809413, 5657529453243494201, 680258345614629409297, 101712366014944608886241, 18180346330505067326217793, 3780026913945005398382926721, 896081555456003024041519984897, 238503222782742830604007067123201, 70414217271582880302629394661942273, 22833087308636059205086630623560611841).
%E A252768 a(10)-a(11) from _Jeff Sponaugle_, Oct 14 2026
```

(Item 1's bound comment is then `a(12) > 1009809788912233`; item 2's last sentence becomes "a(10) = 3770303151073 and a(11) =
1009809788912233 lie at the heuristic's 21 % and 73 % points; it puts a(12) near
3*10^16.")

## NEW (Oct 13 2026): a(10) = 3770303151073 — ready to submit once the 10^14 run ends

Found by resuming the 10^12 checkpoint on jeff-model1 (`scan 1e14 -S a252768.state`):
the first prime of depth 10, with every chunk below it complete. Verified on the
M1 Pro by the parallel totals pipeline with different chunking and GMP agreeing on
all twelve verdicts (`verify_a10_pipeline.txt`), by the single-threaded direct
recomputation (`verify_a10_direct.txt`) and by the Python/numpy verifier
(`verify_py_a10.txt`). All ten sums are
**proven prime**: S_1..S_7 by the deterministic bases, S_8..S_10 (90, 98, 106 bits)
by PARI/GP 2.17.2 `isprime` (APR-CL) on jeff-model1 (`certify_a10.txt`); S_11 is
composite. The edit summary can say "all sums certified prime with PARI/GP isprime".

```
%S A252768 5,5,13,14593,372313,2315773,541613713,7952072743,21814967833,3770303151073
%e A252768 a(10) = 3770303151073: the sums of the k-th powers of the gaps between the primes <= 3770303151073 are 3770303151071, 190851794793757, 14196773503347641, 1394426672423205361, 170141494293736680161, 24799307171701546531777, 4202380711978049572942721, 811587406870679014576316161, 175934585824654713314446896641 and 42297466414050242039599107357697 for k = 1..10, all prime, while the sum of the 11th powers, 11167738558029370106410742901913601, is composite.
%E A252768 a(10) from _Jeff Sponaugle_, Oct 13 2026
```

Together with it, replace the "a(10) > 10^12" bound of item 1 below by
`a(11) > 10^14` (established Oct 13 2026 by the jeff-model1 run over [10^12, 10^14],
all sums tested for k <= 12; raise it to the bound jeff-model2's [10^14, 10^15] run
leaves, or to a(11) itself — 65 % by the model), and update item 2's comment: the sentence "This
heuristic puts a(10) near 10^13 ..." becomes "a(10) = 3770303151073 lies at the
heuristic's 21 % point; it puts a(11) near 4*10^14." Also extend `b252768.txt`
(done, n = 1..10) and the DATA line (done).

## What can still be submitted

With a(10) in hand (above), items 1, 2 and 4 go into the same edit; 3, 5, 6 are
optional; 7 is now about a(11).

### 1. Lower bound for a(10)  (recommended)

The exhaustive scan below 10^12 (`scan_1e12.txt`, every sum tested for
k ≤ 12) found no prime of depth 10. OEIS convention is a one-line comment for
the next term:

```
%C A252768 a(12) > 1009809788912233. - _Jeff Sponaugle_, Oct 14 2026
```

(a(n) is non-decreasing, so a(12) >= a(11); S_12(a(11)) is composite, hence the
strict inequality. The earlier bounds for a(10) and a(11) are superseded by the
terms themselves.)

### 2. Repair the Sep 16 comment  (recommended)

Two problems with the comment that is in the entry: "depth" is never defined
there, and the probability came from the rough early heuristic — the model
calibrated on the full 10^12 data (`estimate.py`, matches the measured per-k
rates within 1–3 %) gives **46 %** for a(10) < 10^13, not 75 %, and 98 % for
a(10) < 10^14. The "factor of roughly 8 to 13" mixes exact-depth counts; the
pass rates are 6.6–8.7 %, i.e. a factor 12–15. Suggested replacement
(edit the existing comment rather than adding a second one):

```
%C A252768 Let S_k(p) be the sum of the k-th powers of the gaps between the primes <= p, and call the largest n with S_1(p), ..., S_n(p) all prime the depth of p, so that a(n) is the least prime of depth >= n. Below 10^12 there are 401 primes of depth 7, 32 of depth 8 and 5 of depth 9 (21814967833, 60308763733, 73668807349, 331274803549, 844836146389). For k = 2..8, between 6.6% and 8.7% of the primes of depth >= k-1 also have S_k prime, in agreement with the heuristic probability B_k*2/log(S_k) where B_k = Product_{3 <= q <= k, q prime} q/(q-1), the factor coming from S_k being coprime to every prime q <= k once S_1, ..., S_(k-1) are prime (g^k mod q depends only on k mod (q-1)). This heuristic puts a(10) near 10^13 and a(11) near 4*10^14. - _Jeff Sponaugle_, Oct 03 2026
```

A shorter alternative that only fixes the errors: define depth in the first
sentence, replace "roughly 8 to 13" by "12 to 15", and replace "with about 75%
probability" by "with about even odds".

### 3. b-file  (optional)

`b252768.txt` (n = 1..9) is ready. With nine terms the editors may not want
one; it does no harm. Link line, if used:

```
%H A252768 Jeff Sponaugle, <a href="/A252768/b252768.txt">Table of n, a(n) for n = 1..9</a>
```

### 4. Program link  (recommended)

The C program, the independent Python verifier, the scan output and these
notes are public in the repository:

```
%H A252768 Jeff Sponaugle, <a href="https://github.com/jeffsponaugle/Mathpub-/tree/main/A252768">C program, Python verifier and notes</a>
```

Alternatively upload `a252768.c` as an a-file ("C program"); the editors
assign the file name. The existing PARI program (Michel Marcus) can stay — it
is correct, just limited to p < 2.75*10^6 by `primes(200000)`.

### 5. Example for a(8)  (optional)

```
%e A252768 a(8) = 7952072743: the sums of the k-th powers of the gaps between the primes <= 7952072743 are 7952072741, 306387204649, 17232652308257, 1275921083018209, 117138327618559361, 12826308390187298689, 1629701772261995408897 and 235364125518401969499649 for k = 1..8, all prime, while the sum of the 9th powers, 38009162738943211881748481, is composite.
```

### 6. New "intersection" sequences  (optional, modest interest)

A252768 lists the first prime of each depth; the full sets of primes of a
given depth are not in the OEIS. They are plain intersections of existing
sequences, which the OEIS does accept but does not prize. First terms from the
tool (`a252768 scan 1e6 -r D -q`):

* Primes of depth ≥ 2, i.e. A006512 ∩ A247177 — "Primes p such that p-2 and
  the sum of the squares of the gaps between the primes <= p are both prime":
  5, 13, 139, 643, 661, 823, 1483, 1489, 1699, 1873, 1933, 2083, 2131, 2593,
  3001, 3541, 3559, 3769, 4639, 4723, 4801, 5641, 5653, 6781, 7213, 7351, 8539,
  9013, 9343, 10141, 10273, 10333, 10501, 11059, 11701, 12043, 12241, 13009,
  13759, 14593, … (1099 terms below 10^6; 29 below 10^4).
* Depth ≥ 3 (also in A247178): 13, 643, 1489, 1873, 1933, 11701, 14593, 15139,
  15289, 18043, 21013, 22573, 24109, 31153, 35449, 35839, 49531, 49939, 53269,
  54403, 58111, 58231, 60091, 63313, 66949, 74611, 74731, 76873, 80449, 80671, …
* Depth ≥ 4 (also in A251623): 14593, 74611, 131779, 194269, 218629, 299359,
  299419, 308311, 313993, 361093, 372313, 419563, 433471, 435109, 449989,
  498523, 693661, 798199, … (18 terms below 10^6).

If submitted, each should cross-reference A252768 (whose a(2), a(3), a(4) are
their first terms) and A006512, A247177, A247178, A251623.

### 7. a(10)  (the valuable one; 3 hours of compute)

`caffeinate -i ./a252768 scan 1e14 -S a252768.state -r 8 -i 120 2> scan_1e14.log | tee -a scan_1e14.txt`
finds a(10) with 98 % probability (46 % already by 10^13, 20 minutes in) and
has a 21 % chance of a(11). Before submitting a new term:

1. `./a252768 verify P -k 16 -t 10` and `python3 verify_a252768.py P 16`
   (see README, "When a hit appears").
2. The sums beyond 3.3*10^24 (S_9 and up at 10^13..10^14) are probable primes from 24
   Miller–Rabin bases plus GMP's BPSW. A PARI/GP `isprime(S)` (APR-CL, instant
   at these sizes) makes each one a proven prime; `brew install pari` gives
   `gp`. Mention "all sums certified with PARI/GP isprime" in the edit summary.
3. Submit the term with `%E a(10) from _Jeff Sponaugle_, <date>`, update the
   bound comment to the new scan limit, and extend `b252768.txt` / `DATA.txt`.

## Verification record for the terms already in the entry

* a(8) = 7952072743 and a(9) = 21814967833 found by the exhaustive scan of
  [0, 10^12] (`scan_1e12.txt`), which also reproduces a(1..7).
* Recomputed from scratch by `a252768 verify` (single-threaded path, GMP
  agreeing on every S_k) and by `verify_a252768.py` (numpy sieve, Python
  integers, independent Miller–Rabin): `verify_py_a8.txt`, `verify_py_a9.txt`.
* Sums: see README "Results". All eight sums of a(8) and S_1..S_8 of a(9) are
  below 3.3*10^24 and therefore proven prime by the deterministic Miller–Rabin
  bases; S_9 of a(9) (88 bits) was proven prime with PARI/GP `isprime` on
  jeff-model1 on Oct 13 2026 (`certify.sh 21814967833 scan_1e12.txt`).
