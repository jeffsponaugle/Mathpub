# A003630 — notes for the pending draft

Checked against the live draft on Oct 14 2026: edits #77–#85 are pending (status "proposed").

* #82, Yves Debeuret, FORMULA: "Conjecture: For n > 3, a(n) = 2 + a(n-1) (when a(n) is the
  larger member of a twin prime pair), or a(n) = a(i) + a(j) + a(k) for some distinct indices
  i, j, k < n, verified up to 9941. - Yves Debeuret, Oct 10 2026"
* #84, Jianing Song: NAME typo fix ("Q[(sqrt(3)]" → "Q(sqrt(3))").
* Discussion so far: Michel Marcus suggested moving it to FORMULA, using a(n) and fixing the
  signature; Sean A. Irvine suggested starting with "Conjecture:". Both are done in #82.

Our result (README.md): the conjecture holds for every a(n) <= 10^12 (n <= 18803973706). The twin
clause is needed only for a(4) = 19, and for n >= 5 the largest summand can always be a(n-1),
a(n-2) or a(n-3).

## Suggested comment for the draft discussion (post as Jeff)

> I checked this with a C program for every a(n) <= 10^12 (n <= 18803973706), and it holds.
> In fact the twin-prime alternative is needed only for a(4) = 19: every a(n) with
> 5 <= n <= 18803973706 is the sum of three distinct earlier terms, and the largest of the three
> can always be taken to be a(n-1), a(n-2) or a(n-3) (e.g. 9941 = 5 + 7 + 9929,
> 137257084039 = 499 + 607 + 137257082933). So perhaps: "Conjecture: For n >= 5,
> a(n) = a(i) + a(j) + a(k) for some distinct indices i, j, k < n; verified for a(n) <= 10^12."
> Program: https://github.com/jeffsponaugle/Mathpub-/tree/main/A003630

The link only works once the A003630 directory is pushed to the public repo; otherwise drop the
last sentence. The minimal change, if the editors prefer to keep Yves's wording, is
"verified up to 9941" → "verified for a(n) <= 10^12".

## Optional: a separate comment after the draft is approved

```
%C A003630 The conjecture in the Formula section holds for a(n) <= 10^12 (n <= 18803973706). In that range the twin-prime alternative is needed only for n = 4, and for n >= 5 a(n) - a(n-k) is the sum of two distinct earlier terms for some k in {1, 2, 3}. - _Jeff Sponaugle_, Oct 14 2026
```

If the run is extended (README.md, "Throughput"), replace 10^12 and 18803973706 with the new
bound and the term count that `check` prints (the "terms" line, which is the largest n checked).
