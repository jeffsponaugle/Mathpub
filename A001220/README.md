# A001220 and relatives — Wieferich primes to many bases

Tools to search for primes p with

    b^(p-1) ≡ 1 (mod p²)

for many bases b at once, on CPU (`wieferich.c`) and on the DGX Spark GPU
(`cuda/wieferich_cuda.cu`). Base 2 is [OEIS A001220](https://oeis.org/A001220);
the other bases with their own OEIS entries are below, plus
[A039951](https://oeis.org/A039951) (smallest such p for each base n).

| base | OEIS | known terms | searched to before this project | searched to now (Oct 03 2026) |
|--:|:--|:--|:--|:--|
| 2 | A001220 | 1093, 3511 | 2^64 (PrimeGrid, Dec 2022) | unchanged |
| 3 | A014127 | 11, 1006003 | 1.2e15 (Fischer) | unchanged (Phase 2) |
| 5 | A123692 | 2, 20771, 40487, 53471161, 1645333507, 6692367337, 188748146801 | 1.2e15 | unchanged (Phase 2) |
| 6 | A212583 | 66161, 534851, 3152573 | 2e14 | 3.08e14 |
| 7 | A123693 | 5, 491531 | 1.2e15 | unchanged (Phase 2) |
| 10 | A045616 | 3, 487, 56598313 | 2e14 | 3.08e14 |
| 12 | A111027 | 2693, 123653 | 2e14 | 3.08e14 |
| 13 | A128667 | 2, 863, 1747591 | 2e14 | 3.08e14 |
| 14 | A234810 | 29, 353, 7596952219 | 2e14 | 3.08e14 |
| 15 | A242741 | 29131, 119327070011 | 2e14 | 3.08e14 |
| 17 | A128668 | 2, 3, 46021, 48947, 478225523351 | 2e14 | 3.08e14 |
| 18 | A244260 | 5, 7, 37, 331, 33923, 1284043 | 2e14 | 3.08e14 |
| 19 | A090968 | 3, 7, 13, 43, 137, 63061489 | 2e14 | 3.08e14 |
| 20 | A242982 | 281, 46457, 9377747, 122959073 | 2e14 | 3.08e14 |
| 22 | A298951 | 13, 673, 1595813, 492366587, 9809862296159 | 2e14 | 3.08e14 |
| 23 | A128669 | 13, 2481757, 13703077, 15546404183, 2549536629329 | 2e14 | 3.08e14 |
| 26 | A306255 | 3, 5, 71, 486999673, 6695256707 | 2e14 | 3.08e14 |
| 30 | A306256 | 7, 160541, 94727075783, **303632117562967** (new, Oct 2026) | 2e14 | 3.08e14 (own search from 0) |
| 47, 72 | A039951 gaps | none known | 2e14 | 3.08e14 |

Search limits: R. Fischer, [fermatquotient.com](http://www.fermatquotient.com/FermatQuotienten/)
(`FermQ_Sort.txt` 2025-09, `Statistik.txt` 2026-07: bases 2..150 to 2.0e14, bases 3, 5, 7 to
1.2e15). The OEIS comments of several entries still quote older, smaller bounds.

## Why base 2 is out of reach but the other bases are not

For p ∤ b write b^(p-1) = 1 + q·p (mod p²); the Fermat quotient q is in [0, p) and p is a
solution iff q = 0. If q behaves like a random residue, a prime qualifies with probability
1/p, so a search of [x, y] finds ln(ln y / ln x) solutions per base on average (Fischer's
per-decade counts for bases up to 1052 agree with this within about 2σ).

* Base 2 from 2^64: a 1.5% chance up to 2^65, which alone is 4.1e17 primes (about PrimeGrid's
  whole two-year effort again); the median of the next term is about 2^128.
* The 16 bases at 2e14 up to 1.2e15: 0.053 per base, 0.85 in total, for 4.7e14 tests
  (about 3.5 days on the two Sparks). Every test at 2e14 is 90,000 times more likely to hit
  than a test at 2^64.

## Tools

    make                                 # CPU tool (needs primesieve)
    ./wieferich selftest                 # ~25 s on the M1 Pro
    ./wieferich scan 2e14 3e14 -b oeis -S run.state -L run.chunks
    ./wieferich check 1006003 -b 3
    cd cuda && make && ./wieferich_cuda selftest     # on a Spark

* **`wieferich.c`** — pthreads + primesieve. For each prime, N = p² < 2^126 (p < 2^63) is set
  up once and shared by all bases. Montgomery arithmetic with two 64-bit limbs, a dedicated
  squaring, and a fixed exponent window whose multiplications by the small constant b^d need
  no Montgomery step: floor(x·c/N) is estimated with one double multiplication and fixed up by
  one add/subtract. Eight exponentiations run in lock step per thread to hide multiply
  latency (1.3 → 3.1e7 tests/s on the M1 Pro).
* **`cuda/wieferich_cuda.cu`** — one GPU thread per (p, b), five 26-bit limbs with 64-bit column
  accumulators (each limb product is one IMAD.WIDE), lazy reduction to [0, 3N), a
  single-precision quotient estimate for the small multiplications, no FP64. Host threads
  sieve the next chunk with primesieve while the GPU tests the current one.
* **`verify_wieferich.py`** — independent Python (big integers, its own Miller–Rabin prime list):
  checksums of a range, quotients of one prime, brute-force solutions below a limit, and
  `cksheader` which generates `wieferich_cks.h` for the selftests.
* **`dcheck.sh`** — recomputes every STEP-th chunk of a GPU chunk log with the CPU tool and
  records MATCH/MISMATCH.
* **`queue.sh`** — runs the scans listed in a queue file one after another on a Spark, each
  with a light double-checker; stops if a scan ends early.
* **`watch.sh`** — polls the Sparks from the Mac and exits on a solution, a mismatch, an FLT
  error, or a Spark with no scan running.

Both sources start with a THEORY OF OPERATION and document every function (inputs, outputs,
error conditions). The CUDA file's header also lists its known issues. On 2026-10-13 the CPU
tool got these fixes, with search results unchanged (byte-identical chunk logs):
- number parsing no longer hangs on `1^E` or `0eE` with a huge E, and rejects `e5` and an
  overflowing `B^1`;
- a failed checkpoint write keeps the previous checkpoint;
- a resume refuses a checkpoint that lacks a base's checksum;
- `check` rejects even P and compares only residues for composite P;
- `-n` values above 2^63 work, `bench` never prints NaN, duplicate bases no longer count
  against the 64-base limit, and resumed runs report this session's rate.

The CUDA tool was left code-identical (PTX and SASS byte-identical) because the paused queue
resumes with its binary.

Both tools compute, for every tested pair, q = (b^(p-1) mod p² − 1)·p⁻¹ mod 2^64 (exact
division). A wrong residue would give a random 64-bit q, so **q < p** re-checks Fermat's little
theorem on every test (FLT errors are reported). Every q goes into a per-base checksum
Σ mix64(q ⊕ p·0x9E3779B97F4A7C15) mod 2^64, which is independent of chunking, thread count
and device. The chunk logs (`-L`) of the CPU tool, the GPU tool and the Python script agree
line by line. Checkpoints (`-S`) have the same format for both tools.

## Validation

* Selftests (both tools): arithmetic against a shift-and-add reference (20,000 GPU quotients,
  200,000 CPU reductions); a scan of [0, 1e9) for all 23 non-power bases 2..30 gives π(1e9)
  and exactly the 65 known solutions; windows around the known solutions above 1e9 re-find
  each one (12 in the CPU selftest, which includes A306256(4); 11 in the GPU selftest); three checksum windows at 1e15, 1e18 and 2^63 match `verify_wieferich.py`.
* `verify_wieferich.py known 3e6` reproduces every known solution below 3e6.
* GPU = CPU chunk logs at 2e14 (20 chunks × 17 bases) and 1.2e15 (10 chunks × bases 3, 5, 7);
  the first CPU chunk line at 1e15 = Python.
* Interrupted and resumed runs = uninterrupted runs with other chunk sizes and thread counts
  (both tools).
* Radix-26 GPU kernel = radix-32 kernel on 8 × 10^8 values (8 heights × 6 bases).
* In production, `dcheck.sh` recomputes 1% of the GPU chunks on the Grace CPU.

## Speed

| machine | 16–17 bases at 2e14 | base 3 at 1.2e15 |
|:--|--:|--:|
| M1 Pro, 10 threads | 3.2e7 tests/s | 3.1e7 |
| Grace (Spark CPU), 20 threads | 7.6e7 | |
| GB10 GPU (Spark), radix 26 | 8.3e8 | 7.2e8 |

GB10 notes: IMAD.WIDE.U32 issues at half the IMAD rate (measured), which is why radix 2^26 with
40 wide products per squaring beats radix 2^32 with 26 products plus carry handling. The CPU and GPU share a power
budget: 12 busy CPU threads cut the GPU clock from 2366 to 2060 MHz, so the production
double-checker runs 4 threads pinned to efficiency cores (CPUs 10–13), which costs nothing.

## Runs (in `/home/jbs/A001220/runs` on the Sparks)

Phase 1 started on both Sparks on 2026-09-30 (bases 6,10,12,13,14,15,17,18,19,20,22,23,26,30,47,72;
atom1 [2e14, 7e14) as `p1a.*`, atom2 [7e14, 1.2e15) as `p1b.*`). Since 2026-09-30 18:45 only atom2
runs: atom1 was freed, its checkpoint and logs moved to atom2, and atom2 works through
`runs/queue.txt` with `queue.sh`:

    0                303640000000000  v30  30           complete (independent check of base-30 a(4))
    200000000000000  700000000000000  p1a  (16 bases)   checkpoint 3.0848e14, ~2.9 days left
    700000000000000  1200000000000000 p1b  (16 bases)   checkpoint 7.380e14, ~3.3 days left
    1200000000000000 2200000000000000 p2a  oeis,47,72   ~8.7 days
    2200000000000000 3200000000000000 p2b  oeis,47,72   ~8.5 days

`queue.sh` re-reads the file after every scan, runs the first item whose `TAG.txt` lacks the
SUMMARY of its full range (resuming from `TAG.state`), gives each scan a light double-checker,
and stops if a scan ends early (manual stop or crash). FOUND / CONFIRMED lines go to `TAG.txt`,
progress to `TAG.log`, double-checks to `TAG.dcheck`.

**Paused 2026-10-01 10:03 PDT** (atom2 needed for another sequence): `v30` complete; `p1a` checkpoint at
3.0848e14 (chunk 10848 of 50000), `p1b` at 7.380e14, Phase 2 not started. Resume on atom2 with the queue
command below; it skips `v30` and continues `p1a` from its checkpoint.

To give atom1 work again: copy the next item's `TAG.*` files from atom2 to atom1 (if it has
started), delete that line from atom2's `queue.txt`, and start a queue on atom1 with it:

    cd /home/jbs/A001220/runs && (nohup setsid ../queue.sh /home/jbs/A001220/runs/queue.txt >> queue.log 2>&1 < /dev/null &)

To pause: `kill -INT` the `wieferich_cuda scan` process (checkpoint is saved, the queue stops);
rerun the queue command to continue.

To verify a solution p for base b:

    ./wieferich check P -b B
    python3 verify_wieferich.py check P B

## Results

* **A306256 (base 30): a(4) = 303632117562967 — in the OEIS since Oct 02 2026** (Jeff Sponaugle's edit,
  approved as revision #18). Found 2026-10-01 in Phase 1; confirmed by an independent base-30 scan
  of every prime below 3.0364e14 (9,396,616,950,003 primes = π(303640000000000) by primecount;
  solutions exactly 7, 160541, 94727075783, 303632117562967; 316 chunks recomputed on the CPU, all
  matching) and by the CPU tool, Python and OpenSSL. Evidence: [OEIS_draft_A306256.md](OEIS_draft_A306256.md).
* **No other solutions** for bases 6, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 26, 30, 47, 72 among all
  primes in [2e14, 3.0848e14): 3,271,298,909,197 primes = π(3.0848e14) − π(2e14) by primecount,
  10,848 contiguous chunks, 0 FLT errors, 119 CPU re-checks all matching. With Fischer's search to
  2e14 this means no further terms below 3.08e14 for these entries, and a(47), a(72) > 3.08e14 in
  A039951. Base 30: a(5) > 3.0848e14.
* What can be submitted (bound updates, the stale A306256 comment, a note to R. Fischer) and the
  state of the OEIS entries as of Oct 03 2026: [submission.md](submission.md).
