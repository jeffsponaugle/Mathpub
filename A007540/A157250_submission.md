# Draft OEIS submission for A157250 (Wilson numbers) — NOT YET SUBMITTED

Status: search complete to 2³² − 1 = 4294967295 (2026-10-04); 17 new terms, all verified by the
direct definition. Ready for Jeff's review; not submitted.

## Data

Current OEIS data:
1, 5, 13, 563, 5971, 558771, 1964215, 8121909, 12326713, 23025711, 26921605, 341569806, 399292158

Proposed data (30 terms, complete to 2^32 - 1):
1, 5, 13, 563, 5971, 558771, 1964215, 8121909, 12326713, 23025711, 24365727, 26921605, 86286991, 139459265, 341569806, 399292158, 683139612, 798584316, 1024709418, 1197876474, 1366279224, 1597168632, 2049418836, 2340914226, 2395752948, 2732558448, 3074128254, 3194337264, 3623403714, 4098837672

b-file: `b157250.txt` (n a(n), n = 1..30).

## Comments to add / change

* Replace "The sequence contains all Wilson numbers <= 5 x 10^8." by:
  "The terms 24365727 = 3*8121909, 86286991 = 7*12326713 and 139459265 = 71*1964215 are below
  5*10^8 but missing from Table 1 of Agoh, Dilcher and Skula (1998). Their Proposition 3.1 shows
  that if p^2*m is a Wilson number then so is p*m, and on that basis the 1998 search tested only
  squarefree n (plus the special forms 4p and 9m). The converse fails, so the multiples p*m of the
  squarefree Wilson numbers m found still had to be tested; these three terms are such multiples.
  The sequence is complete up to 2^32 - 1 = 4294967295."
* "If W is a Wilson number and p^a*W is one for some prime p, then p^(a-1)*W is one (Agoh, Dilcher,
  Skula, Props. 3.1-3.2). Conversely 2^a*3^b*W for W = 341569806 and 399292158 give the terms
  683139612 = 2*341569806, 1024709418 = 3*341569806, ..., 4098837672 = 12*341569806 (see A157250.md
  for why these lifts hold identically). Terms not of this kind: 2340914226 = 42*55736053 and
  3623403714 = 42*86271517."
* Possibly a new sequence: squarefree Wilson numbers (1, 5, 13, 563, 5971, 558771, 1964215, 8121909,
  12326713, 23025711, 26921605, 341569806, 399292158, 2340914226, 3623403714) — ask Jeff.
* Replace "Every known composite Wilson number n has at least two odd prime factors, so e(n) = -1."
  — still true (the new terms also have e(n) = -1); keep but note that three terms are not squarefree.
* Optional: "If p^2*m is a Wilson number then p*m is one (Agoh-Dilcher-Skula, Prop. 3.1); the
  known examples are 24365727/8121909, 86286991/12326713 and 139459265/1964215."

## Extensions field (OEIS convention: nothing was wrong, terms were missing, so "inserted", not "corrected")

a(11), a(13), a(14) inserted and a(17)-a(30) added by _Jeff Sponaugle_, Oct 08 2026

(Index bookkeeping: old a(11) = 26921605 becomes a(12); old a(12), a(13) = 341569806, 399292158 become
a(15), a(16). The Data field is replaced wholesale by the 30-term line above; the b-file link reads
"Jeff Sponaugle, <a href="/A157250/b157250.txt">Table of n, a(n) for n = 1..30</a>".)

## Fields touched

* Data: replace with the 30-term line.
* Comments: replace the "contains all Wilson numbers <= 5 x 10^8" sentence (see above); add the lift remark.
* Links: add the b-file.
* Extensions: the line above.
* Keywords: keep nonn,more (still "more": the heuristic and the lifts suggest further terms beyond 2^32).

## Method (for the comment / link)

n is a Wilson number iff for every prime power p^k || n, with m = n/p^k, G_k = Product_{j<p^k, p∤j} j
= eps + p^k*w, q(x) = (x^phi(p^k) - 1)/p^k mod p^k:
  eps^phi(m) * (1 + p^k*(eps*phi(m)*w - Sum_{l|m} (phi(m)/(l-1))*q(l))) == -e(n) (mod p^(2k))
(for 2^1: P(2m) == A(m)*(-1)^(phi(m)/2) (mod 4)). Needs the Wilson quotients w_p mod p for all
primes p <= N/2, computed with the Costa-Gerbicz-Harvey algorithm (Gerbicz's pw13), and w(p^k) for
k >= 2 from Gauss factorials. Every n <= N is tested once at its largest prime power; survivors are
fully tested; hits are re-verified by direct computation of Product_{gcd(a,n)=1} a mod n^2 in two
independent programs (C and Python).

Program and data: ~/src/math/A007540 (tools/wilsonnum.c, tools/wilsonnum_brute.c,
tools/wilson_verify.py; logs in runs/). Consider a GitHub link in the submission.

## b-file

b157250.txt: n a(n) for all terms (16+ lines).

## References/links already present

Agoh, Dilcher, Skula, Math. Comp. 67 (1998) 843-861 — keep; cite Table 1 and Prop. 3.1 in the comment.

## Verification record (for the editors, if asked)

| n | factorisation | direct C (`wilsonnum_brute`) | Python (`wilson_verify.py`) | formula components (`wilsonnum -n`) |
|---|---|---|---|---|
| 24365727 | 3²·139·19477 | P ≡ 1, e = −1 | same | 3², 139, 19477 all pass |
| 86286991 | 7²·1760959 | P ≡ 1, e = −1 | same | pass |
| 139459265 | 5·11·71²·503 | P ≡ 1, e = −1 | same | pass |

Self-test: the same program run to 5·10⁸ returns exactly the 13 published numbers plus these three;
3000 random n ≤ 3·10⁷ agree with the direct definition; all n ≤ 200000 agree with a direct scan.
