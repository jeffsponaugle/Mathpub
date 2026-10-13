#!/usr/bin/env python3
"""Coverage audit for the A001211(27) = n(27,6) search (psph, split depth 4, chunks of 100000 work items,
chunks 0..29, run on mathd, mathb and mathg at three different targets).

THEORY OF OPERATIONS
--------------------
A search with target T reports every basis with 27-range >= T.  If every work item was searched by a run whose
target is <= the true optimum, the largest reported range is n(27,6); this script checks that coverage and
reports the maximum.  Result of the final run (Oct 3 2026): COMPLETE, n(27,6) = 186942 (logs/n27k6_final_audit.txt).

The depth-4 work-item enumeration depends on the target through the element-wise lower bounds: 2964437 items
at TGT 176381, 2964430 at TGT 177915 (the 7 prefixes missing at the higher target are proven unable to reach
177915, so they cannot be extremal once n(27,6) >= 177915 is known), and 2964418 at TGT 186943 (used for the
chunks started after the 186942 basis appeared).  The index spaces therefore differ by a shift of at most 19,
and items can fall through at a chunk boundary B = 100000*i whose lower chunk ran at a lower target than its
upper chunk.  Boundary windows re-run at TGT 176381 (the largest enumeration) cover such items: [B-8, B+8)
(tag h27k6_bnd) and [B-24, B+24) (tag h27k6_bnd2), both on mathg, for all 29 boundaries.  For each boundary
that needs a window the script requires one covering [B, B + N_176381 - N_upper).
A chunk counts as complete only if its .done marker exists AND its .err file contains the final statistics
line ("worker CPU split"); markers touched by hand to skip a chunk (when it was moved to another box) do not count.

Tags: mathd h27k6d (chunks 6-13), h27k6i (14), h27k6e (27-28), h27k6g (skipped, chunk 25 moved);
      mathb h27k6b (20-21), h27k6h (22-24), h27k6gb (25), h27k6j (29);
      mathg h27k6 (0-5, 15-19), h27k6f (26), h27k6_bnd / h27k6_bnd2 (boundary windows).
Output (stdout): the chunk table, missing / duplicated chunks, item counts seen per target, window coverage,
unexpected log lines, the number of reported bases, the best basis and every basis attaining it, and the status
line "COMPLETE: n(27,6) = A001211(27) = N" or "INCOMPLETE: lower bound ...".
Usage: python3 tools/collect_h27k6.py (needs passwordless ssh to the three boxes).
Errors: an unreachable box contributes nothing (its chunks show as MISSING); unexpected chunk ranges, targets or
item counts are listed as UNEXPECTED LINES and block the COMPLETE verdict.
"""
import subprocess, re, sys
CHUNK = 100000; NCHUNK = 30
N_BY_TGT = {176381: 2964437, 177915: 2964430, 186943: 2964418}
W_TGT = 176381   # target of the boundary windows (largest enumeration)
boxes = [("mathd", "jbs@10.1.30.23", "~/A0001208/psph", ["h27k6d", "h27k6i", "h27k6e", "h27k6g"]),
         ("mathb", "jbs@10.1.30.21", "~/A001208/psph", ["h27k6b", "h27k6h", "h27k6gb", "h27k6j"]),
         ("mathg", "jbs@10.1.30.26", "~/A001208/psph", ["h27k6", "h27k6f", "h27k6_bnd", "h27k6_bnd2"])]
# remote shell snippet: one "DONE tag file TGT=... processing work items [lo, hi) of N" line per finished chunk or
# window (statistics line present AND .done marker), then "@@SOL" and every SOLUTION line of the given tags
remote = ("cd {d}; for t in {tags}; do for f in logs/$t/*.err; do [ -f \"$f\" ] || continue; "
          "grep -q 'worker CPU split' \"$f\" || continue; b=${{f%.err}}; [ -f \"$b.done\" ] || continue; "
          "echo \"DONE $t $(basename $b) $(sed -n 1p \"$f\" | grep -o 'TGT=[0-9]*') $(grep -m1 'processing work items' \"$f\")\"; done; done; "
          "echo @@SOL; for t in {tags}; do grep -h 'SOLUTION h=27 k=6' logs/$t/*.out 2>/dev/null; done")
chunks = {}      # chunk index -> list of (box, tag, tgt)
windows = {}     # boundary index -> list of (box, tag, lo, hi) windows run at W_TGT
bad = []; solutions = set(); enum_seen = {}
for name, host, d, tags in boxes:
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, remote.format(d=d, tags=' '.join(tags))],
                         capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout
    part, _, sol = out.partition('@@SOL')
    for line in part.strip().split('\n'):
        m = re.match(r'DONE (\S+) (\S+) TGT=(\d+) .*processing work items \[(\d+), (\d+)\) of (\d+)', line)
        if not m:
            if line.strip(): bad.append((name, line.strip()))
            continue
        tag, fn, tgt, lo, hi, n = m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6))
        enum_seen.setdefault(tgt, set()).add(n)
        if N_BY_TGT.get(tgt) != n: bad.append((name, f"{tag}/{fn}: TGT {tgt} has {n} items, expected {N_BY_TGT.get(tgt)}")); continue
        if '_bnd' in tag:
            i = (lo + hi) // 2 // CHUNK
            if tgt != W_TGT or lo > i * CHUNK or hi <= i * CHUNK: bad.append((name, f"{tag}/{fn}: unexpected window [{lo},{hi}) TGT {tgt}")); continue
            windows.setdefault(i, []).append((name, tag, lo, hi))
        else:
            if lo % CHUNK != 0 or hi != min(lo + CHUNK, n): bad.append((name, f"{tag}/{fn}: unexpected chunk [{lo},{hi}) of {n}")); continue
            chunks.setdefault(lo // CHUNK, []).append((name, tag, tgt))
    for line in sol.strip().split('\n'):
        if 'SOLUTION h=27 k=6' in line: solutions.add(line.strip())
print("chunk  machine/tag/TGT")
for i in range(NCHUNK): print(f"  {i:2d}   {chunks.get(i, 'MISSING')}")
missing = [i for i in range(NCHUNK) if i not in chunks]; dup = [i for i in range(NCHUNK) if len(chunks.get(i, [])) > 1]
print(f"chunks complete: {NCHUNK - len(missing)} of {NCHUNK}; missing: {missing}; duplicated: {dup}")
print(f"enumeration sizes seen per target: { {t: sorted(s) for t, s in enum_seen.items()} }")
# A boundary B needs a window when the chunk below ran at a lower target than the chunk above (its enumeration is a
# superset, so the upper chunk's index space is shifted down by up to N_W - N_upper items): the window, run at W_TGT,
# must then cover [B, B + N_W - N_upper) i.e. lo <= B and hi >= B + N_W - N_upper.
need = {}
for i in range(1, NCHUNK):
    if i - 1 in chunks and i in chunks and chunks[i-1][0][2] < chunks[i][0][2]:
        need[i] = i * CHUNK + N_BY_TGT[W_TGT] - N_BY_TGT[chunks[i][0][2]]
wok = {i: any(lo <= i * CHUNK and hi >= req for (_, _, lo, hi) in windows.get(i, [])) for i, req in need.items()}
wmissing = [i for i in range(1, NCHUNK) if i not in windows]
print(f"boundaries with a window: {len(windows)} of {NCHUNK - 1}; without: {wmissing}")
print(f"boundaries needing a window (lower target below, higher above) -> covered: {wok}")
if bad: print("UNEXPECTED LINES:"); [print("  ", b) for b in bad]
sols = sorted(solutions, key=lambda s: int(re.search(r'n_h=(\d+)', s).group(1)))
print(f"solutions collected: {len(sols)}")
if sols:
    best = int(re.search(r'n_h=(\d+)', sols[-1]).group(1))
    print("best:", sols[-1]); print("all bases attaining the best:"); [print("  ", s) for s in sols if f"n_h={best} " in s]
    complete = not missing and not bad and all(wok.values())
    print("status:", f"COMPLETE: n(27,6) = A001211(27) = {best}" if complete else f"INCOMPLETE: lower bound n(27,6) >= {best}")
