#!/bin/sh
# A001211(27) boundary windows, version 2: half-width 24 (three enumerations now: 2964437 items at TGT 176381,
# 2964430 at 177915, 2964418 at 186943 -> index shift <= 19 between any two). Windows [B-24,B+24) at TGT 176381
# (the largest enumeration) for every chunk boundary B = 100000*i cover every item that could fall through a
# boundary whose lower chunk ran at a lower target than its upper chunk. See tools/collect_h27k6.py.
#
# Ran on mathg on Oct 3 2026 (started by switch_g.sh after bnd_h27k6.sh), 4 threads, 29/29 windows completed.
# Resumable via w_i.done markers.  Output: logs/h27k6_bnd2/{progress.log, solutions.log, w_i.out, w_i.err, w_i.done}.
cd ~/A001208/psph
L=logs/h27k6_bnd2; mkdir -p $L
for i in $(seq 1 29); do
  B=$((i*100000)); lo=$((B-24)); hi=$((B+24))
  [ -f $L/w_$i.done ] && continue
  echo "=== window $i items [$lo,$hi) h=27 k=6 TGT=176381 start $(date)" >> $L/progress.log
  ./psph -h 27 -k 6 -t 176381 -j 4 -p 300 -d 4 -i $lo:$hi > $L/w_$i.out 2> $L/w_$i.err
  grep SOLUTION $L/w_$i.out >> $L/solutions.log
  grep -q "worker CPU split" $L/w_$i.err && touch $L/w_$i.done
  echo "=== window $i done $(date): $(grep -c SOLUTION $L/w_$i.out) solutions; $(grep -m1 'processing work items' $L/w_$i.err)" >> $L/progress.log
done
echo "=== BOUNDARY WINDOWS COMPLETE $(date)" >> $L/progress.log
