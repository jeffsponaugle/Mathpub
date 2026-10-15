# A003630 — the "sum of three earlier terms" conjecture

[A003630](https://oeis.org/A003630) lists the inert rational primes in Q(sqrt(3)),
i.e. the primes p ≡ 5 or 7 (mod 12):
5, 7, 17, 19, 29, 31, 41, 43, 53, 67, 79, 89, 101, ...

A pending draft edit by Yves Debeuret ([draft](https://oeis.org/draft/A003630), edits #77–#82,
Oct 2026) adds to the FORMULA section:

> Conjecture: For n > 3, a(n) = 2 + a(n-1) (when a(n) is the larger member of a twin prime pair),
> or a(n) = a(i) + a(j) + a(k) for some distinct indices i, j, k < n, verified up to 9941.

9941 = a(619). `a003630.c` checks the conjecture for every term up to a bound, and
`verify_a003630.py` cross-checks it independently below 10^8.

## Result (Oct 14 2026)

**Verified for every term a(n) ≤ 10^12, i.e. n ≤ 18803973706.** That run took 67 s on 4
threads of the M1 Pro (`check_1e12.txt`, state `a003630.state`).

* Only a(1..4) = 5, 7, 17, 19 are not sums of three distinct smaller terms. The conjecture starts
  at n = 4, and 19 = 17 + 2 is covered by the twin-prime clause, so the conjecture holds.
  In fact the twin clause is needed only for n = 4: **every a(n) with 5 ≤ n ≤ 18803973706 is
  the sum of three distinct earlier terms.**
* **The largest of the three can always be taken to be a(n-1), a(n-2) or a(n-3).** Writing k for
  that offset: k = 1 for 15679201587 terms (83.4%), k = 2 for 3040437202 (16.2%), k = 3 for
  84334913 (0.45%), and never k ≥ 4.
* The run saw 37607912018 primes in [0, 10^12], which is exactly pi(10^12), so no prime was
  skipped.
* The largest gap between consecutive terms below 10^12 is 1106 (a(n) = 137257084039). The mean
  of s(N) is 62.6.

Records of s(N) = N − (largest summand), where the representation is the one with the largest
possible top summand and the smallest possible middle one:

```
           N     s(N)   representation
          29       12   5 + 7 + 17
          67       24   7 + 17 + 43
         199       26   7 + 19 + 173
         271       38   7 + 31 + 233
         449       48   19 + 29 + 401
        1181       58   17 + 41 + 1123
        2083       72   31 + 41 + 2011
        7349       96   43 + 53 + 7253
        7351       98   31 + 67 + 7253
       23537      106   17 + 89 + 23431
       23539      108   41 + 67 + 23431
       88169      132   53 + 79 + 88037
      119813      142   53 + 89 + 119671
      119827      156   67 + 89 + 119671
      150797      180   79 + 101 + 150617
      515143      194   67 + 127 + 514949
      515153      204   101 + 103 + 514949
      743549      216   103 + 113 + 743333
      743551      218   79 + 139 + 743333
     1349669      262   113 + 149 + 1349407
     1349671      264   127 + 137 + 1349407
     2927537      274   101 + 173 + 2927263
     6248551      290   139 + 151 + 6248261
     7064179      326   127 + 199 + 7063853
    24796147      360   163 + 197 + 24795787
    29085857      418   149 + 269 + 29085439
    29085871      432   199 + 233 + 29085439
    71896819      446   163 + 283 + 71896373
    98559017      504   233 + 271 + 98558513
    98559031      518   211 + 307 + 98558513
   505506233      586   269 + 317 + 505505647
  1691604269      610   293 + 317 + 1691603659
  1691604283      624   307 + 317 + 1691603659
  2602947341      682   293 + 389 + 2602946659
  4707625469      780   379 + 401 + 4707624689
 39265870183      840   401 + 439 + 39265869343
 50531790077      948   461 + 487 + 50531789129
137257084039     1106   499 + 607 + 137257082933
```

## Why the check is cheap, and what the conjecture really says

* **Residues force the pattern.** Every term is 5 or 7 (mod 12), so a term ≡ 5 can only be
  (5)+(5)+(7) and a term ≡ 7 only (5)+(7)+(7). No class conditions need checking: any three
  distinct smaller terms that add up to N will do.
* **Pairs.** Let P2 be the set of sums x + y of two distinct terms. Every even s ≤ 65536 that can
  occur as a difference of two terms (s ≡ 0, 2, 10 mod 12) is in P2, except 2, 10 = 5+5 and
  14 = 7+7. The program checks this at startup.
* **So it comes down to gaps.** For N = a(n) ≥ 2^17, take the largest summand z to be the nearest
  earlier term with N − z ∉ {2, 10, 14}. That gives z = a(n-1) or a(n-2) when N ≡ 5 (mod 12),
  and a(n-1), a(n-2) or a(n-3) when N ≡ 7, and N − z is then automatically a sum of two smaller
  distinct terms. For large N the conjecture therefore fails only if three consecutive gaps add up
  to more than 65536. The real cost is enumerating the primes. Below 2^17 the program runs an
  exact search instead.
* For all sufficiently large n the three-sum form also follows from Vinogradov's three-primes
  theorem restricted to residue classes. This is classical; see e.g.
  [R. Ayoub, Trans. AMS 74 (1953) 482–491](https://www.ams.org/journals/tran/1953-074-03/S0002-9947-1953-0053960-0)
  and Liu & Zhan, Acta Arith. (1997). But the threshold is far beyond computation, so the
  conjecture is really about the finite range.

## Program

```
make                                  # needs primesieve >= 11
./a003630 selftest                    # ~10 s
./a003630 check 1e12 -t 4             # all terms <= 10^12
./a003630 check 1e14 -S a003630.state # resumable; reuses the 10^12 chunks in the state file
./a003630 rep 9941                    # 9941 = 5 + 7 + 9929
python3 verify_a003630.py 1e8 --compare ./a003630
```

`check` cuts the range into chunks, and threads take chunks from an atomic counter. For each
chunk, primesieve enumerates the primes from 65536 below the chunk start. Each batch is compacted
to the terms, and every term is checked against its last few predecessors with a table lookup.
Terms below 2^17, or with no witness among the previous 8 terms, go to an exact search
(Miller–Rabin, every candidate summand down to N/3). That exact search was used only for the 6148
terms below 2^17. The source header has the details.

Checks:

* `selftest` compares:
  * the DATA terms and a(100), a(500), a(1000) with the OEIS b-file;
  * the exceptional set {2, 10, 14};
  * a brute-force triple search for all terms below 20000;
  * the fast path against the exact search term by term near 2^17, 10^12, 10^18 and 2^64,
    with frequent chunk boundaries;
  * results across chunk sizes and thread counts;
  * a state-file resume;
  * pi(10^9).
* A `check` from 0 to 10^k compares its prime count with pi(10^k).
* `verify_a003630.py` uses numpy, a full pair table and a vectorised pass, plus itertools
  below 1000. It agrees with the C program on all 9 statistics and all 30 records below 10^8.

Throughput on 4 M1 Pro threads, in integers per second: 1.5e10 near 10^12, 1.3e10 near 10^13,
1.05e10 near 10^14, 7.4e9 near 10^15. On jeff-model3 (32 vCPU), [10^12, 10^14] should take about
20 minutes and [10^14, 10^15] about 4 hours.

## Files

* `a003630.c`, `Makefile`: the checker
* `verify_a003630.py`: the independent cross-check
* `check_1e12.txt`, `check_1e12.log`, `a003630.state`: the 10^12 run (state lines are per 10^10 chunk)
* `submission.md`: suggested text for the draft discussion
