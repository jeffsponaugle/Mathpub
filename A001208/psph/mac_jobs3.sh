#!/bin/sh
# Mac: after the running [10000,10500) range, run [4200,4700) at target 6083 (the final best 6082 + 1).
#
# Job-chain wrapper: each stage is one run_range.sh call (resumable through .done markers); "=== ... start/done" lines
# in mac_jobs.log mark the stages, and a wrapper only starts a stage after the previous run_range.sh has returned.  Stop a
# wrapper BEFORE its run_range.sh driver, otherwise it advances to its next stage.
# Replaced by mac_jobs4.sh when [4450,4700) moved to mathg.
cd /Users/Jeff.Sponaugle/src/math/A001208/psph
while pgrep -f "run_range.sh mac_h9k8b" > /dev/null; do sleep 60; done
echo "=== MAC JOB1b done $(date)" >> mac_jobs.log
echo "=== MAC JOB1d items [4200,4700) start $(date)" >> mac_jobs.log
./run_range.sh mac_h9k8d 9 8 6083 4 4200 4700 250 10 ./psph >> mac_jobs.log 2>&1
echo "=== MAC JOB1 (all Mac ranges) done $(date)" >> mac_jobs.log
