#!/bin/sh
# Stop every postage-stamp job on this box and report what is complete (resumable state = .done markers).
#
# Usage: sh ~/<dir>/psph/stop_all.sh   (it changes to its own directory first).  Used for the wind-down of mathd and
# mathg on Oct 3 2026.  Order matters: the job-chain wrappers (jobs*.sh) are killed first so that they cannot start
# their next stage, then the run_range.sh drivers, then the psph / psph16 / psph_gpu workers.
# Output: "=== STOPPED ALL ..." appended to jobs.log; on stdout the number of remaining processes and, for every log
# directory, the number of completed chunks (.done markers, including hand-made skip markers) and of distinct
# SOLUTION lines.  Restart a stopped search by re-running its run_range.sh command (see DEPLOY.md).
# Caveat: the patterns match any process whose command line contains them (for example an editor open on jobs3.sh).
cd "$(dirname "$0")"
for p in $(pgrep -f "jobs[0-9]*\.sh"); do [ "$p" != "$$" ] && kill $p 2>/dev/null; done
pkill -f "run_range.sh" 2>/dev/null; sleep 1
pkill -x psph 2>/dev/null; pkill -x psph16 2>/dev/null; pkill -x psph_gpu 2>/dev/null; sleep 2
echo "=== STOPPED ALL by stop_all.sh $(date)" >> jobs.log
echo "remaining processes: $(pgrep -fa 'psph|run_range' | grep -v stop_all | wc -l)"
for t in logs/*/; do n=$(ls $t/chunk_*.done 2>/dev/null | wc -l); echo "$t: $n chunks done; solutions: $(sort -u $t/solutions.log 2>/dev/null | wc -l)"; done
