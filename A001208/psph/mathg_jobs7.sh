#!/bin/sh
# mathg overnight v2 (Oct 3 01:10 PDT): finish chunk 19, then chunk 26 (from mathb) at TGT 186943, then the tail of
# A001210(91) in 1000-item chunks.
#
# Job-chain wrapper: each stage is one run_range.sh call (resumable through .done markers); "=== ... start/done" lines
# in jobs.log mark the stages, and a wrapper only starts a stage after the previous run_range.sh has returned.  Stop a
# wrapper BEFORE its run_range.sh driver, otherwise it advances to its next stage.
# Deployed as ~/A001208/psph/jobs7.sh on mathg (switch_g.sh); stopped by stop_all.sh on Oct 3 07:58 PDT during the
# first A001210(91) chunk (h91k5c chunk 0, not completed).
cd ~/A001208/psph
while pgrep -f "run_range.sh h27k6 " > /dev/null; do sleep 60; done
echo "=== JOB3 done $(date)" >> jobs.log
echo "=== JOB3f n(27,6) items [2600000,2700000) (chunk 26, from mathb), TGT 186943 start $(date)" >> jobs.log
./run_range.sh h27k6f 27 6 186943 4 2600000 2700000 100000 32 ./psph >> jobs.log 2>&1
echo "=== JOB3f done $(date)" >> jobs.log
echo "=== JOB4c n(91,5) items [100000,133732) in 1000-item chunks start $(date)" >> jobs.log
./run_range.sh h91k5c 91 5 8897043 3 100000 133732 1000 32 ./psph >> jobs.log 2>&1
echo "=== JOB4c done $(date)" >> jobs.log
echo "=== ALL JOBS DONE $(date)" >> jobs.log
