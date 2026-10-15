# OEIS submission — A171775 (smallest number that is a k-digit palindrome for every k = 2..n)

**Done: a(10) = 2^72 and a(11) = 2^90 are in the OEIS** (submitted Oct 08 2026, approved Oct 09 2026,
revision #70). This file records what went in and keeps the items that were not submitted, ready for
an optional follow-up edit. Everything was computed or checked with the tools in this directory
(`a171775.c`, `a171775_2d.c`, `cuda/a171775_cuda.cu`, `cuda/a171775_2d_cuda.cu`, `metal/`,
`verify_a171775.py`); see README.md for the methods and the run logs.

## Status on the OEIS (checked 2026-10-14)

* **A171775 — approved, live as revision #70 (Oct 09 2026).** History of the edit:
  * #65 (Oct 08, Jeff Sponaugle): DATA a(10)–a(11), the survivor-count comment, the a(11) example
    and the EXTENSIONS line, with the short note to the editors from section 3; #66: proposed.
  * #67 (Oct 08, Michel Marcus): put the existing a(6) example back in front of the new a(11)
    example; #68: proposed; #69 (Oct 09): reviewed.
  * #70 (Oct 09, Michael De Vlieger): approved.

  No edit is pending. Keywords are now `nonn,base,hard,more,changed` (`changed` marks the recent
  edit; `more` stays because a(12) is open).
* The cross-referenced entries (A007632, A029961–A029970, A171701–A171706, A171740–A171742,
  A253294, ...) are not affected by this work.
* The other `src/math` entries were checked the same day; see the appendix.

## Summary

| item | status |
|---|---|
| a(10) = 2^72 = 4722366482869645213696 | **in the OEIS** (#70) |
| a(11) = 2^90 = 1237940039285380274899124224 | **in the OEIS** (#70) |
| comment with the survivor counts below 2^72 and 2^90 | **in the OEIS** (dated Oct 08 2026) |
| EXTENSIONS "a(10)-a(11) from _Jeff Sponaugle_, Oct 08 2026" | **in the OEIS** |
| EXAMPLE for a(11) | **in the OEIS** (after the a(6) example) |
| EXAMPLE for a(10) | not submitted; optional (section 2) |
| LINKS: program link | not submitted; recommended (section 2) |
| PROG (Python, reproduces a(1..6)) | not submitted; optional (section 2) |
| Merickel's "equals it for n = 7 and 8 at least" | unchanged; optional, best left to the editors (section 2) |
| a(12) | not proposed: only the trivial a(12) > 2^90 is known (section 5) |

Next step (optional): one small follow-up edit with the program link, plus the a(10) example if
wanted (section 2).

---

## 1. What is in the entry now (revision #70)

The lines added by this work, as they appear in the entry:

```
%S A171775 1,3,5,52,130,1885,1073741824,4398046511104,72057594037927936,
%T A171775 4722366482869645213696,1237940039285380274899124224
%C A171775 There are 135382 numbers M <= 2^72 that are both a 10-digit and a 9-digit palindrome (in some bases); 2^72 is the only one of them that is also an 8-digit palindrome in some base. There are 524157 numbers M <= 2^90 that are both an 11-digit and a 10-digit palindrome; 2^90 is the only one of them that is also a 9-digit palindrome. Hence a(10) = 2^72 and a(11) = 2^90, as conjectured in the Formula section. - _Jeff Sponaugle_, Oct 08 2026
%e A171775 a(11) = 2^90 = 1237940039285380274899124224: it is 11 in base 2^90-1, and in bases 2^31-1, 2^23-1, 2^19-1, 2^16-1, 2^14-1, 2^12-1, 2^11-1, 2^10-1, 2^9-1 its digits are 2^28*C(2,j), 2^21*C(3,j), 2^14*C(4,j), 2^10*C(5,j), 2^6*C(6,j), 2^6*C(7,j), 2^2*C(8,j), C(9,j), C(10,j), palindromes of 3 to 11 digits.
%E A171775 a(10)-a(11) from _Jeff Sponaugle_, Oct 08 2026
```

Everything else is unchanged: Merickel's comments on the bound, the FORMULA conjecture
"a(n) = 2^((n-1)*(n-2)) for n >= 7" (Max Alekseyev), the a(6) example, the cross-references and
the older EXTENSIONS lines. The entry has no LINKS (%H) and no PROG lines.

## 2. Optional follow-up edit

The entry has no pending edit, so a follow-up can be proposed at any time. Everything here is
optional; the program link is the most useful.

### LINKS (recommended)

```
Jeff Sponaugle, <a href="https://github.com/jeffsponaugle/Mathpub/tree/main/A171775">C, CUDA and Metal programs, run logs and notes</a>
```

The repository was renamed from `Mathpub-` to `Mathpub` (GitHub redirects the old name, but the
link should use the new one). Its A171775 directory is current: it matches the local commit
bd00f8c2 (the commented sources with the 2026-10-12 fixes, the run logs, the b-file and this file).
Optional cleanup before linking: the leftover files `a171775.v1`, `a171775_2d.c.bak`,
`a171775_2d.run`, `metal/a171775_2d_queue.metal.bak`, `metal/a171775_2d.run.metal` and
`metal/a171775_metal.run`.

### EXAMPLE for a(10) (optional; it would go between the a(6) and a(11) examples)

```
a(10) = 2^72 = 4722366482869645213696: it is 11 in base 2^72-1; in base 18669329 it is the 3-digit palindrome 13548844 15833512 13548844; in bases 2^19-1, 2^15-1, 2^13-1, 2^11-1, 2^10-1, 2^9-1, 2^8-1 its digits are 2^15*C(3,j), 2^12*C(4,j), 2^7*C(5,j), 2^6*C(6,j), 2^2*C(7,j), C(8,j), C(9,j), palindromes of 4 to 10 digits.
```

Written in the same style as the accepted a(11) line; checked digit by digit with
`verify_a171775.py check 2^72 10`.

### PROG (optional)

Plain brute force; it reproduces a(1..6) in a fraction of a second (tested with Python 3.14):

```
(Python)
def is_kpal(M, k): # is M a k-digit palindrome in some base?
    b = max(2, round(M**(1/k)) - 1)
    while b**(k-1) <= M:
        if M < b**k:
            d, x = [], M
            while x:
                d.append(x % b)
                x //= b
            if d == d[::-1]:
                return True
        b += 1
    return False
def a(n):
    M = 1
    while not all(is_kpal(M, k) for k in range(2, n+1)):
        M += 1
    return M
print([a(n) for n in range(1, 7)]) # _Jeff Sponaugle_, Oct 14 2026
```

### Merickel's comment (optional; leave it to the editors)

Merickel's first comment still says the bound is attained "for n = 7 and 8 at least". With a(9)–a(11)
known it could read "(and equals it for 7 <= n <= 11)", or the FORMULA conjecture could get
"(true for n <= 11)". Since that changes other contributors' text, mention it in the discussion box
rather than editing it.

### Note for the discussion box

```
Follow-up to the a(10)-a(11) edit (#65-#70): adds a link to the programs and run logs that produced these terms (and an example for a(10)). Merickel's comment says the bound is attained "for n = 7 and 8 at least"; it now holds for 7 <= n <= 11, if the editors want to update it.
```

## 3. Notes to the editors

The short note was posted in the discussion of #65 on Oct 08 2026:

```
a(10) = 2^72 and a(11) = 2^90 are new and agree with the conjecture in the Formula section. Both come from exhaustive searches below these bounds: every number that is both an n-digit and an (n-1)-digit palindrome was found (135382 for n = 10, 524157 for n = 11) and then tested for the shorter lengths; only 2^72 and 2^90 pass. Each search was run several times, with different programs, methods and hardware, and gave identical counts and checksums.
```

The longer version was not needed; it is kept here in case anyone asks how the terms were computed:

```
Every candidate is an n-digit palindrome in some base B and an (n-1)-digit palindrome in some base c > B. The programs enumerate the palindromes of one of the two lengths and impose the other length by solving a linear congruence for the innermost free digits (the k lowest base-b digits of M must mirror its k highest), with all state updated by additions; matches are then checked digit by digit, and survivors are tested for every shorter length by scanning all admissible bases.
a(10): all 3.7*10^12 ten-digit palindromes <= 2^72 (bases 2..255) were enumerated, with the 9-digit condition solved for one digit: 5.1*10^12 steps; a C program on an Apple M1 Pro (29 min), a CUDA program on an NVIDIA GB10 (154 s) and the C program on a DGX Spark CPU gave identical statistics, and a second method that enumerates the 9-digit palindromes instead (and solves two digits with a lookup table) gave the same 135382 survivors and checksum (12 min).
a(11): the second method enumerated all 7.0*10^15 ten-digit palindromes <= 2^90 (bases 3..1024), with 5.4*10^12 table lookups; three complete runs (M1 Pro CPU 2.8 h, DGX Spark CPU 1.2 h, GB10 GPU with a separately written CUDA kernel 57 min) gave identical statistics (524157 survivors, 1.2*10^12 congruence matches, checksum 0a5d53d568af3f34). The first method gives the same survivors on the window [600^9, 600^9 + 3*600^8].
Checks: the programs reproduce a(1)..a(9); the congruence filters agree with naive enumeration on hundreds of random blocks for n = 7..12 and with a debug build that recomputes the state at every step; an independent Python brute force agrees on small ranges.
```

---

## 4. Verification chain (for reference)

| check | result |
|---|---|
| a(10), 1D method, M1 Pro CPU (`scan_n10.*`) | 1750 s, 5126601984459 steps, 135382 survivors, 1 solution (2^72) |
| a(10), 1D method, GB10 GPU (CUDA) | 154 s, identical, checksum e9eed6127c51a50f |
| a(10), 1D method, DGX Spark CPU | 653 s, identical, checksum e9eed6127c51a50f |
| a(10), 2D method, M1 Pro CPU | 715 s, 135382 survivors, checksum e9eed6127c51a50f |
| a(10), 2D method, M1 Pro GPU (Metal selftest, 2026-10-13) | 652 s, all seven statistics identical |
| a(11), 2D method, M1 Pro CPU (`scan_n11.*`) | 10023 s, 5386659681911 lookups, 524157 survivors, checksum 0a5d53d568af3f34, 1 solution (2^90) |
| a(11), 2D method, DGX Spark CPU (gcc) | 4474 s, identical in every statistic |
| a(11), 2D method, GB10 GPU (CUDA port) | 3411 s, identical in every statistic |
| n = 11 window [600^9, 600^9 + 3·600^8], 1D method | 211 survivors, checksum 1dc0b33a97158c66 = 2D method |
| a(1..9) | reproduced by both methods (a(9): exhaustive in 32 s / 1.5 s) |
| 2^72, 2^90 | digits checked from the definition in Python for every length |

In every run, all survivors except the solution fail at the next length (8 for n = 10, 9 for n = 11).
The M1 Pro a(11) run was resumed between checkpointed segments; it matches the two uninterrupted
atom2 runs in every statistic, so the checkpoint bug fixed on 2026-10-12 (README) did not affect it.

## 5. Not proposed

* **a(12).** 2^90 is not a 12-digit palindrome in any base (`a171775 check 2^90 12`), so
  a(12) > 2^90, but that follows trivially from a(12) >= a(11). The conjectured value is 2^110. A search
  below 2^110 with the 2D method would need about 1.4·10¹⁶ lookups: roughly 100 GPU-days on one GB10
  at the rate of the a(11) run. It has not been attempted.
* **A b-file.** Not needed; the 11 terms fit in DATA.

---

## Appendix — OEIS status of the other `src/math` submissions (checked 2026-10-14)

Approved and live (credited to Jeff Sponaugle):

| entry | revision | content |
|---|---|---|
| A137723 | #17–#18, Sep 17 2026 | a(32) onward, b-file n = 1..193, comment a(194) > 2^64-194 |
| A252768 | #21, Sep 17 2026 | a(8)-a(9) and the depth-count comment |
| A053686 | #28, Sep 17 2026 | a(13) = 906 and the repeat comment |
| A133788 | #13, Sep 17 2026 | a(13) |
| A085237 | #44, Sep 17 2026 | a(80)-a(86), b-file n = 1..86 |
| A089180 | #28, Sep 17 2026 | example corrected ("corrected by Jeff Sponaugle, Sep 17 2026") |
| A248701 | #47, Sep 18 2026 | a(8)-a(10) |
| A248704 | #19, Sep 17 2026 | a(7)-a(9) |
| A248702 | #25, Sep 20 2026 | a(7)-a(11) |
| A248703 | #25, Sep 20 2026 | a(7)-a(9) |
| A359636 | #31, Sep 18 2026 | a(9) and a(10) > 31610555571634177 |
| A158939 | #38, Sep 22 2026 | a(16)-a(17) and a(18) > 5.30*10^16 (the entry is now at #45, Oct 13: a Python program by Ya-Ping Lu and edits by others; your lines are unchanged) |
| A229832 | #35, Sep 22 2026 | a(15)-a(16) |
| A133697 | #27, Sep 22 2026 | a(14)-a(15) |
| A306256 | #18, Oct 02 2026 | a(4) and the search-bound comment |
| A063501 | #29, Oct 03 2026 | a(18)-a(20) |
| A350826 | #36, Oct 08 2026 | **new:** a(18)-a(20), the comment on sextuplets with members of different lengths, the fixed link to Luhn's `PI_06.php` |
| A053348 | #33, Oct 08 2026 | **new:** a(9) = 6082 |
| A005344 | #39, Oct 08 2026 | **new:** a(8) = 6082 |
| A157250 | #37, Oct 08 2026 | **new:** a(11), a(13), a(14) inserted, a(17)-a(30) added |
| A171775 | #70, Oct 09 2026 | **new:** a(10)-a(11) (this file) |

Proposed by you on Oct 14 2026, waiting for review (A034822 project):

| entry | revision | content |
|---|---|---|
| A263618 | #30–#31 | a(68)-a(69) from Patrick De Geest |
| A002778 | #101–#102 | b-file extended from 8729 to 10173 terms (De Geest, Alekseyev) |
| A016113 | #35–#36 | a(23) from Patrick De Geest; comment that a(24) has at least 70 digits |

Still open. Each project's own `submission.md` is authoritative; this column is only a pointer,
with the state of the OEIS entries on Oct 14:

| project file | open items |
|---|---|
| `../A248701/submission.md` | A248704(10) = 1958854030679863 still not submitted (A248704 still #19); obsolete comment in A248702, `%E` indices in A248703, bounds |
| `../A001208/submission.md` | A053348/A005344 done; A001211(27) = 186942 plus a correction still open (A001211 still #42); literature terms (A001210, A053346), b-files (A001209, A084192, A084193, A196416), caveat comments (A001213–A001216): entries unchanged |
| `../A034822/SUBMISSION.md` | three edits proposed Oct 14 (above); A002779, A027829 and the A034822 comment not yet |
| `../A158939/submission.md` | program link, EXAMPLE for a(17), optional items (none submitted yet) |
| `../A359636/submission.md` | a(10) lower bound; optional EXAMPLE, comments, program link (entry still #31) |
| `../A252768/submission.md` | a(10) lower bound, repair of the Sep 16 comment, program link (entry still #21) |
| `../A053686/submission.md` | comment listing the second occurrences, program link (entry still #28) |
| `../A137723/submission.md` | follow-up edit (no computation needed); new terms need runs (entry still #18) |
| `../A001220/submission.md` | bound updates when the search resumes; R. Fischer's A039951 edit was approved on Oct 08 (#137), so that entry is no longer blocked |
| `../A350826/submission.md` | done (approved Oct 08; that file still says "to do"); only the optional A063501 follow-up is left. The live EXTENSIONS line reads "a(18)-a(19) from Karl Desfontaines confirmed by, a(20) added by _Jeff Sponaugle_, Oct 08 2026", which looks garbled and could be fixed in a follow-up |
| `../A112389/` | A123829 a(10)-a(11), A123831 a(9)-a(10), A112390 corrections: ready, not submitted (entries unchanged) |
| (no draft yet) | A055206(13) = 6470105925 (balanced primes below 10^13), from the A335406 scans (`../A335406/README.md`); A055206 still #13 |

No submission: A335406 (a(6) is out of reach), A002966, A005234 and A007540 (feasibility
studies, verdict: not feasible; the Wilson-number terms of A157250 came out of the A007540 work
and are in, see above), A112389's a(11) (paused); A005115, A007508, A011541 and A101232 have no
recorded results.
