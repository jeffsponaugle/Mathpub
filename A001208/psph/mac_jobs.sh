#!/bin/sh
# Mac share of the n(9,8) search (ranges released by mathb and mathd), target 5823. Resumable: re-run to continue.
#
# Job-chain wrapper: each stage is one run_range.sh call (resumable through .done markers); "=== ... start/done" lines
# in mac_jobs.log mark the stages, and a wrapper only starts a stage after the previous run_range.sh has returned.  Stop a
# wrapper BEFORE its run_range.sh driver, otherwise it advances to its next stage.
# Ran on the M1 Pro from Oct 1 16:36 PDT; replaced by mac_jobs2.sh.
cd /Users/Jeff.Sponaugle/src/math/A001208/psph
echo "=== MAC JOB1c items [10700,11003) start $(date)" >> mac_jobs.log
./run_range.sh mac_h9k8c 9 8 5823 4 10700 11003 303 10 ./psph >> mac_jobs.log 2>&1
echo "=== MAC JOB1b items [10000,10500) start $(date)" >> mac_jobs.log
./run_range.sh mac_h9k8b 9 8 5823 4 10000 10500 250 10 ./psph >> mac_jobs.log 2>&1
echo "=== MAC JOB1d items [4200,5200) start $(date)" >> mac_jobs.log
./run_range.sh mac_h9k8d 9 8 5823 4 4200 5200 500 10 ./psph >> mac_jobs.log 2>&1
echo "=== MAC JOB1 (all Mac ranges) done $(date)" >> mac_jobs.log
