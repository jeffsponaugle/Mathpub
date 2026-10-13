#!/bin/sh
# Mac: after the current range [10700,11003), run [10000,10500) and the reduced [4200,4700); target 5946.
# (Comment corrected: the code has always used 5946; "5845" was a typo.)
#
# Job-chain wrapper: each stage is one run_range.sh call (resumable through .done markers); "=== ... start/done" lines
# in mac_jobs.log mark the stages, and a wrapper only starts a stage after the previous run_range.sh has returned.  Stop a
# wrapper BEFORE its run_range.sh driver, otherwise it advances to its next stage.
# Waits for the mac_h9k8c driver first; replaced by mac_jobs3.sh / mac_jobs4.sh.
cd /Users/Jeff.Sponaugle/src/math/A001208/psph
while pgrep -f "run_range.sh mac_h9k8c" > /dev/null; do sleep 60; done
echo "=== MAC JOB1c done $(date)" >> mac_jobs.log
echo "=== MAC JOB1b items [10000,10500) start $(date)" >> mac_jobs.log
./run_range.sh mac_h9k8b 9 8 5946 4 10000 10500 250 10 ./psph >> mac_jobs.log 2>&1
echo "=== MAC JOB1d items [4200,4700) start $(date)" >> mac_jobs.log
./run_range.sh mac_h9k8d 9 8 5946 4 4200 4700 250 10 ./psph >> mac_jobs.log 2>&1
echo "=== MAC JOB1 (all Mac ranges) done $(date)" >> mac_jobs.log
