#!/bin/sh
# mathb overnight v2 (Oct 3 01:10 PDT): finish chunk 21 under the old driver, then chunks 22-24 at TGT 186943
# (25 -> mathd, 26 -> mathg), then A001210(91) items [10000,100000) in 1000-item chunks.
#
# Job-chain wrapper: each stage is one run_range.sh call (resumable through .done markers); "=== ... start/done" lines
# in jobs.log mark the stages, and a wrapper only starts a stage after the previous run_range.sh has returned.  Stop a
# wrapper BEFORE its run_range.sh driver, otherwise it advances to its next stage.
# Deployed as ~/A001208/psph/jobs7.sh on mathb (started by switch_b.sh); replaced by jobs8.sh (switch_b2.sh).
cd ~/A001208/psph
while pgrep -f "run_range.sh h27k6b" > /dev/null; do sleep 60; done
echo "=== JOB3b done (old driver exited after chunk 21) $(date)" >> jobs.log
echo "=== JOB3h n(27,6) items [2200000,2500000) (chunks 22-24), TGT 186943 start $(date)" >> jobs.log
./run_range.sh h27k6h 27 6 186943 4 2200000 2500000 100000 96 ./psph >> jobs.log 2>&1
echo "=== JOB3h done $(date)" >> jobs.log
echo "=== JOB4b n(91,5) items [10000,100000) in 1000-item chunks start $(date)" >> jobs.log
./run_range.sh h91k5b 91 5 8897043 3 10000 100000 1000 96 ./psph >> jobs.log 2>&1
echo "=== JOB4b done $(date)" >> jobs.log
echo "=== ALL JOBS DONE $(date)" >> jobs.log
