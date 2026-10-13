#!/bin/sh
# Mac: tail of the A001211(27) item list (mathg's chunks 20-29), target 176381.
#
# Job-chain wrapper: each stage is one run_range.sh call (resumable through .done markers); "=== ... start/done" lines
# in mac_jobs.log mark the stages, and a wrapper only starts a stage after the previous run_range.sh has returned.  Stop a
# wrapper BEFORE its run_range.sh driver, otherwise it advances to its next stage.
# Ran Oct 2 11:04-14:05 PDT; stopped at 82 % of chunk 20 when the Mac was needed (chunks moved to mathb).
cd /Users/Jeff.Sponaugle/src/math/A001208/psph
echo "=== MAC JOB3 n(27,6) items [2000000,2964437) start $(date)" >> mac_jobs.log
./run_range.sh mac_h27k6 27 6 176381 4 2000000 2964437 100000 10 ./psph >> mac_jobs.log 2>&1
echo "=== MAC JOB3 done $(date)" >> mac_jobs.log
