#!/bin/sh
# Stop MY validation/throughput runs on orin1 by exact command match (never touches ./psph, run_range.sh, jobs.sh).
#
# Matches on whole argv words (via ps -eo pid,args and awk), not on substrings, so the production search running on the
# same box at the time was safe: the test drivers "sh ./validate.sh" / "sh ./throughput.sh", the "sh -c while ..." and
# "sh -c SKIP_SELFTEST=..." launchers, and the ./psph_cpu[16] / ./psph_gpu[16] test binaries.  Prints what is left.
for p in $(ps -eo pid,args | awk '$2=="sh" && ($3=="./validate.sh" || $3=="./throughput.sh") {print $1}'); do kill $p 2>/dev/null; done
for p in $(ps -eo pid,args | awk '$2=="sh" && $3=="-c" && $4=="while" {print $1}'); do kill $p 2>/dev/null; done
for p in $(ps -eo pid,args | awk '$2=="sh" && $3=="-c" && $4 ~ /^SKIP_SELFTEST/ {print $1}'); do kill $p 2>/dev/null; done
for p in $(ps -eo pid,args | awk '$2 ~ /^\.\/psph_(cpu|gpu)(16)?$/ {print $1}'); do kill $p 2>/dev/null; done
sleep 2
ps -eo pid,etime,args | grep -E "psph|validate|throughput|while" | grep -v grep
