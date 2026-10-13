#!/bin/sh
# mathb overnight v3 (Oct 3 05:40 PDT): finish chunk 24 under the h27k6h driver, then chunks 25 and 29 (taken from mathd's
# queue so that the search ends ~09:00 instead of ~11:30), then A001210(91) items [10000,100000) in 1000-item chunks.
#
# Job-chain wrapper: each stage is one run_range.sh call (resumable through .done markers); "=== ... start/done" lines
# in jobs.log mark the stages, and a wrapper only starts a stage after the previous run_range.sh has returned.  Stop a
# wrapper BEFORE its run_range.sh driver, otherwise it advances to its next stage.
# Deployed as ~/A001208/psph/jobs8.sh on mathb (switch_b2.sh).  Killed by winddown_b.sh after chunk 29 started,
# so the A001210(91) stage never ran.
cd ~/A001208/psph
while pgrep -f "run_range.sh h27k6h" > /dev/null; do sleep 60; done
echo "=== JOB3h done $(date)" >> jobs.log
echo "=== JOB3gb n(27,6) items [2500000,2600000) (chunk 25, from mathd), TGT 186943 start $(date)" >> jobs.log
./run_range.sh h27k6gb 27 6 186943 4 2500000 2600000 100000 96 ./psph >> jobs.log 2>&1
echo "=== JOB3gb done $(date)" >> jobs.log
echo "=== JOB3j n(27,6) items [2900000,2964437) (chunk 29, from mathd), TGT 186943 start $(date)" >> jobs.log
./run_range.sh h27k6j 27 6 186943 4 2900000 2964437 100000 96 ./psph >> jobs.log 2>&1
echo "=== JOB3j done $(date)" >> jobs.log
echo "=== JOB4b n(91,5) items [10000,100000) in 1000-item chunks start $(date)" >> jobs.log
./run_range.sh h91k5b 91 5 8897043 3 10000 100000 1000 96 ./psph >> jobs.log 2>&1
echo "=== JOB4b done $(date)" >> jobs.log
echo "=== ALL JOBS DONE $(date)" >> jobs.log
