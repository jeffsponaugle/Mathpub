#!/bin/sh
# usage: sh restart_tgt.sh NEWTGT   -- restart the job chain with a new target for job 1 (run on the box, in its psph dir)
#
# Used during the n(9,8) search (Oct 1-2 2026) to raise the target of the h9k8 range on mathd / mathb / mathg after a
# better basis had been found (a higher target prunes more; results stay exact as long as the target is <= the optimum).
# Steps: stop jobs.sh, the h9k8 run_range.sh and the workers; rewrite the target in jobs.sh with GNU sed -i (Linux);
# delete the partial .out/.err of the interrupted chunk; restart jobs.sh with nohup and show the new processes.
# Inputs: NEWTGT.  Outputs: edited jobs.sh, a "=== RESTART" line in jobs.log, status on stdout.
# Caveat: completed chunks keep the target they ran with; the audit only needs every target <= the final optimum.
NEW=$1
cd "$(dirname "$0")"
# order matters: stop the chain driver first so it cannot advance to the next job, then the range driver, then the workers
pkill -f "jobs.sh" 2>/dev/null; sleep 1
pkill -f "run_range.sh h9k8" 2>/dev/null; sleep 1
pkill -x psph 2>/dev/null; pkill -x psph16 2>/dev/null; sleep 2
sed -i "s|./run_range.sh h9k8 9 8 [0-9]* 4|./run_range.sh h9k8 9 8 $NEW 4|" jobs.sh
grep "run_range.sh h9k8" jobs.sh
# remove partial output of the chunk that was running (the first chunk without a .done marker)
for f in logs/h9k8/chunk_*.err; do [ -f "$f" ] || continue; b=${f%.err}; [ -f "$b.done" ] || rm -f "$b.out" "$b.err"; done
echo "=== RESTART with TGT=$NEW $(date)" >> jobs.log
nohup ./jobs.sh > jobs.nohup 2>&1 < /dev/null &
sleep 5
echo "--- running ---"; pgrep -fa "psph -h" | cut -c1-100; tail -1 logs/h9k8/progress.log
