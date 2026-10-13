#!/bin/sh
# quick A/B: wall time for the same 100-item window in the a_2~25 region at TGT 177915 vs 186943 (3 threads each, concurrent)
#
# Ran on mathg, Oct 3 2026 ~01:00 PDT, after the 186942 basis appeared: 385 s at 177915 vs 253 s at 186943 (full checks
# 28.8M vs 12.2M), which is why the remaining A001211(27) chunks were switched to TGT 186943.  Output: logs/bench_tgt/
# and one timing line per target on stdout.
cd ~/A001208/psph
L=logs/bench_tgt; mkdir -p $L
for t in 177915 186943; do
  ( s=$(date +%s); ./psph -h 27 -k 6 -t $t -j 3 -p 60 -d 4 -i 2600000:2600100 > $L/t$t.out 2> $L/t$t.err; e=$(date +%s); echo "TGT $t: $((e-s)) s wall, $(grep -o 'leaf cand [0-9.e+]*, full checks [0-9]*' $L/t$t.err | tail -1), sols $(grep -c SOLUTION $L/t$t.out)" ) &
done
wait
for t in 177915 186943; do grep -m1 "processing work items" $L/t$t.err; grep "worker CPU split" $L/t$t.err | cut -c1-120; done
