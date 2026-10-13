#!/usr/bin/env python3
"""Collect n(9,8) search results from all machines: completed item ranges (from run_range progress logs
and .done markers) and all SOLUTION lines. Reports coverage of items [0, 11003) and the maximum range.

THEORY OF OPERATIONS
--------------------
The n(9,8) = A053348(9) = A005344(8) search (psph -h 9 -k 8, split depth 4: 11003 work items, the same count at
every target used) was split into item ranges run by psph/run_range.sh on several machines and at several
targets.  A search with target T reports every basis with 9-range >= T, so once every item has been searched
by SOME run whose target is <= the true optimum, the largest reported range is n(9,8).  This script proves the
"every item" part and finds the largest range:
  1. ssh to each box and, for every logs/<tag>/chunk_<i>.done marker, print the matching "=== chunk i items
     [lo,hi) h=9 k=8 ..." line of logs/<tag>/progress.log (run_range.sh writes it when the chunk starts); also
     collect every SOLUTION line of the chunk outputs.
  2. Add the Mac's runs: the first overnight runner (logs/overnight, chunks of 400 items numbered from 0) and the
     later psph/logs/mac_* run_range directories.
  3. Count how often each item is covered; report missing and double-covered runs of items, the best basis,
     every basis attaining it, and the status line "COMPLETE: n(9,8) = N" (no item missing) or "INCOMPLETE".
A chunk counts as complete when its .done marker exists; a few markers were created by hand after the chunk's
statistics were checked (an early run_range.sh bug did not create them).  tools/collect_h27k6.py is the stricter
successor: it also requires the statistics line in the chunk's .err file.  Targets are not checked here; they
were all <= 6083 = n(9,8) + 1 (see bfiles/A053348_A005344_submission.md).

Usage: python3 tools/collect_h9k8.py   (needs passwordless ssh to the boxes; unreachable boxes contribute nothing,
which shows up as missing items).  Output: text report on stdout.  The final audit is logs/n9k8_final_audit.txt.
Errors: ssh failures are silent (empty output); malformed log lines are ignored.
"""
import subprocess, re, sys
N_ITEMS = 11003
boxes = [("mathd", "jbs@10.1.30.23", "~/A0001208/psph"), ("mathb", "jbs@10.1.30.21", "~/A001208/psph"),
         ("mathg", "jbs@10.1.30.26", "~/A001208/psph"), ("orin1", "jbs@10.1.30.40", "~/A001208/psph")]
# remote shell snippet: per .done marker, the "=== chunk i items [lo,hi) ..." start line prefixed by the log directory;
# then "@@SOL" and every SOLUTION line ({d} is filled in, {{ }} are literal braces for the shell)
cmd_remote = ("cd {d}; for t in logs/*/; do for f in $t/chunk_*.done; do [ -f \"$f\" ] || continue; i=${{f##*chunk_}}; i=${{i%.done}}; "
              "grep -h \"^=== chunk $i items\" $t/progress.log | head -1 | sed \"s|^|$t |\"; done; done; "
              "echo @@SOL; grep -h SOLUTION logs/*/chunk_*.out 2>/dev/null")
done_ranges = {}   # (lo,hi) -> list of (box, tag)
solutions = set()
def parse(box, text):
    """Parse one box's remote output: add each completed (lo, hi) range to done_ranges and each SOLUTION line to
    solutions.  Inputs: box name, text = ranges part + '@@SOL' + solutions part.  No return value; no errors."""
    part, sol = text.split('@@SOL') if '@@SOL' in text else (text, '')
    for line in part.strip().split('\n'):
        m = re.search(r'(\S+)/? === chunk (\d+) items \[(\d+),(\d+)\) h=9 k=8', line)
        if m:
            tag = m.group(1).rstrip('/'); lo, hi = int(m.group(3)), int(m.group(4))
            done_ranges.setdefault((lo, hi), []).append((box, tag))
    for line in sol.strip().split('\n'):
        if 'SOLUTION h=9 k=8' in line: solutions.add(line.strip())
for name, host, d in boxes:
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, cmd_remote.format(d=d)],
                         capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout
    parse(name, out)
# Mac: overnight chunks (logs/overnight) and psph/logs/mac_*
import glob, os
base = '/Users/Jeff.Sponaugle/src/math/A001208'
for f in glob.glob(f'{base}/logs/overnight/h9k8_chunk_*.done'):
    i = int(re.search(r'chunk_(\d+)\.done', f).group(1))
    lo, hi = i * 400, min((i + 1) * 400, N_ITEMS)   # overnight runner: chunks of 400 from item 0
    done_ranges.setdefault((lo, hi), []).append(("mac-overnight", f"chunk{i}"))
for f in glob.glob(f'{base}/logs/overnight/h9k8_chunk_*.out'):
    for line in open(f):
        if 'SOLUTION h=9 k=8' in line: solutions.add(line.strip())
for t in glob.glob(f'{base}/psph/logs/mac_*/'):
    for f in glob.glob(t + 'chunk_*.done'):
        i = int(re.search(r'chunk_(\d+)\.done', f).group(1))
        for line in open(t + 'progress.log'):
            m = re.search(r'=== chunk %d items \[(\d+),(\d+)\)' % i, line)
            if m: done_ranges.setdefault((int(m.group(1)), int(m.group(2))), []).append(("mac", os.path.basename(t.rstrip('/')))); break
    for f in glob.glob(t + 'chunk_*.out'):
        for line in open(f):
            if 'SOLUTION h=9 k=8' in line: solutions.add(line.strip())
# coverage
covered = [0] * N_ITEMS
for (lo, hi), who in sorted(done_ranges.items()):
    for i in range(lo, hi): covered[i] += 1
missing = [i for i in range(N_ITEMS) if covered[i] == 0]
dup = [i for i in range(N_ITEMS) if covered[i] > 1]
def runs(lst):
    """Compress a sorted list of item numbers into [start, end) runs, e.g. [1,2,3,7] -> [[1,4],[7,8]]."""
    out = []; 
    for i in lst:
        if out and out[-1][1] == i: out[-1][1] = i + 1
        else: out.append([i, i + 1])
    return out
print("completed ranges (lo,hi) -> machine/tag:")
for (lo, hi), who in sorted(done_ranges.items()): print(f"  [{lo},{hi}) {who}")
print(f"items covered: {N_ITEMS - len(missing)} of {N_ITEMS}; missing runs: {runs(missing)}; double-covered runs: {runs(dup)}")
sols = sorted(solutions, key=lambda s: int(re.search(r'n_h=(\d+)', s).group(1)))
print(f"solutions collected: {len(sols)}")
if sols:
    best = int(re.search(r'n_h=(\d+)', sols[-1]).group(1))
    print("best:", sols[-1])
    print("all bases attaining the best:"); [print("  ", s) for s in sols if f"n_h={best} " in s]
    print("status:", "COMPLETE: n(9,8) = %d" % best if not missing else "INCOMPLETE: lower bound n(9,8) >= %d" % best)
