# A034822 — lengths with no palindromic squares

OEIS [A034822](https://oeis.org/A034822): numbers k such that no palindromic square has exactly k digits.
Known: 2, 4, 8, 10, 14, 18, 20, 24, 30, 38, 40, 46, 54, 56, 62, 64 (all even: (10^k+1)^2 covers odd k).

**State of the art (Sep 2026).** P. De Geest's table (worldofnumbers, Nov 2025; CUDA search by R. Xiao) is
complete through 69 digits. Length 68 has exactly one palindromic square,
7256171055736382499839982033184475^2 (found 2025-04-14, not yet in the OEIS b-files), so 68 is not a term.
**The next candidate is L = 70**; De Geest/Xiao estimated ~400 days on one of their GPUs.

## Method (both tools)

n = t·10^h + B with B < 10^h and h = 17 for L = 70. B fixes the low h digits of S = n^2; a palindrome's top h
digits P are their reverse, so t lies in the window isqrt(P·10^e) ≤ t ≤ isqrt((P+1)·10^e), e = L − 3h
(width ~5–16). B is enumerated by a digit DFS (tracking floor(B_i^2/10^i) so S digit i costs a few adds).

Savings:
* **8 roots per window.** B, B + 5·10^(h−1), −B, 5·10^(h−1) − B share B^2 mod 10^h, and so do their images under
  σ2: B → B·E′ mod 10^h with E′ ≡ −1 (mod 2^h), E′ ≡ 1 (mod 5^h). σ2 fixes digits 0..v2(B) and adds 5 to digit
  v2(B)+1, so restricting that digit to 0–4 (plus unit 1–5, top digit 0–4) picks one representative per orbit.
* **11 | n** (every even-length palindrome is divisible by 11): one t class per root, i.e. 1–2 candidates.
* Candidate filter in 64-bit: D = t^2 + floor(2tB/10^h) − P·10^e (mod 2^64) against 3 mirrored digit pairs
  (shared-memory table), then 6 pairs, then an exact 128-bit check; the host does the final full check.

## Files

| file | purpose |
|---|---|
| `a034822.c` | CPU tool (pthreads). `selftest [maxL]` vs A263618; `-DPARANOID` checks every window/filter/σ2 root exactly |
| `cuda/a034822_cuda.cu` | GPU tool for GB10 (sm_121), even L 40..70. Builds: `-DMINB=2` (production), `-DWINDOW_CHECK -DFILTER_CHECK` (exact re-checks) |
| `base_of.py` | maps a known root n to the GPU base index that must find it |
| `xcheck_subtree.py` | independent Python enumeration of one DFS subtree (all 8 roots, exact windows) for comparison with a GPU hit dump |
| `SUBMISSION.md`, `oeis/` | proposed OEIS b-files and edits, current OEIS b-files, De Geest's table, `verify_bfiles.py` |

GPU usage: `a034822_cuda search L [-r a:b] [-b bases_per_launch] [-S state]` (bases 0..5·10^9 for L=70;
resume with the same command line and state file), `a034822_cuda selftest [maxL]`.

Both sources have a full theory-of-operation header and per-function documentation (inputs, outputs, errors).
2026-10-13 hardening: CPU tool commits each task atomically with respect to checkpoints (exact counts across
any number of interruptions, verified with SIGINT and SIGKILL), checks checkpoint write errors, rejects
`-t 0`, empty `-r` ranges and over-long `check` arguments, caps `selftest` at L = 69. GPU tool validates
`-b`, `-B` (multiple of 32, <= 256) and `-r`, caps `selftest` at 68, checks state-file write errors; dead
experiment hooks removed; `make` now builds the production configuration (`-DMINB=2`).

## Validation

* CPU: counts match A263618 for every L = 11..48; paranoid build L = 11..36.
* GPU: selftest L = 40..56 (even) matches A263618; the known 66-digit (both) and 68-digit squares are found in
  the bases predicted by `base_of.py`; WINDOW_CHECK/FILTER_CHECK clean in all five unit-digit regions at L = 70;
  every filter hit on a unit-1 base verified genuine; `xcheck_subtree.py` agrees exactly on subtrees in units 1, 2, 5
  (including the overflow-prone region below).
* **Bug found and fixed (2026-09-30):** GPU versions before v9 cast the 2^-36 fixed-point window offset to i64;
  in the unit-1 region (P ≈ 1.0–2.0·10^16) it exceeds 2^63, so about half of those candidates were never tested
  (xcheck: 269 of 547 missed on prefix 491234567901). Small-L selftests could not trigger it.
  Earlier bug: `__noinline__` device calls received corrupted arguments under register pressure (now all inlined).

## Performance (L = 70, GB10, 24 576 bases per unit-digit region)

| version | unit 1 | 2 | 3 | 4 | 5 | full L=70, one Spark |
|---|---|---|---|---|---|---|
| v2 (first CUDA, buggy in unit 1) | | | 19.4 s | | | ~41 days |
| v10 (current) | 12.1 s | 8.9 s | 8.5 s | 8.9 s | 5.2 s | **~20.6 days (~10.3 on two Sparks)** |

GB10 integer rates (microbenchmark): 32-bit add 4.5e12/s, IMAD.WIDE 2.4e12/s, umul64hi 0.7e12/s.
The root check is instruction-bound (~70 SASS per root in isolation); profiler counters are admin-only here.

## Status

No L = 70 run started yet (Jeff decided 2026-09-30 not to tie up the Sparks for ~10 days; the run can be done in
chunks later via `-r a:b` + `-S state`). Nothing submitted to OEIS.

## OEIS submissions — see [SUBMISSION.md](SUBMISSION.md)

Ready-to-submit b-files and entry edits for A263618, A002778, A002779, A016113, A027829 and a comment on
A034822 are in **[SUBMISSION.md](SUBMISSION.md)**, with the files in `oeis/` and checked by `oeis/verify_bfiles.py`.
OEIS checked 2026-10-03: no newer revisions and no pending drafts; the 68-digit root is not in the OEIS.

### Background: the 68-digit square (independent of the L = 70 run)

The only 68-digit palindromic square is

    7256171055736382499839982033184475^2 =
    52652018390106447787037098707897055079870789073078774460109381025625

found 2025-04-14 by Patrick De Geest with Robert Xiao's CUDA program (worldofnumbers, "Plain Text Squares.txt",
update of Nov 16 2025). As of 2026-09-30 an OEIS search for the root returns **no results**. Check with
`./a034822 check 7256171055736382499839982033184475` (prints PALINDROME, 68 digits).

| sequence | b-file now ends at | add |
|---|---|---|
| [A263618](https://oeis.org/A263618) (number of palindromic squares with n digits) | n = 67 (701) | a(68) = 1, and a(69) = 1443 from the same table |
| [A016113](https://oeis.org/A016113) (roots of even-length palindromic squares) | n = 22 (982503990036767976718486749830913) | a(23) = 7256171055736382499839982033184475 |
| [A002778](https://oeis.org/A002778) (roots of palindromic squares) | n = 8729 (3109885844380152888763910885950363) | n = 8730 = 7256171055736382499839982033184475, then the 1443 roots of 69-digit squares (the table lists them in order) |

A034822 itself does not change: 68 is not a term (it has a palindromic square), and the next term is still unknown
(first open length is 70). Credit the new data to De Geest / Xiao (worldofnumbers table), not to this project.
Worth a comment on A034822: "All lengths <= 69 were searched exhaustively (De Geest, Xiao); 68 has one
palindromic square, so a(17) >= 70."
