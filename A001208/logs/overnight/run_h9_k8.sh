#!/bin/sh
# Resumable exact search for n(9,8) = A053348(9) = A005344(8).  Work items 0..11002 at split depth 4,
# processed in chunks of 400; a chunk is marked done by an empty .done file.  Re-run this script to resume.
#
# The first phase of the n(9,8) search, on the M1 Pro (Sep 30 - Oct 1 2026, 10 threads).  TGT can be overridden from
# the environment (default 5522 = the best basis known at the time + 1); S2 is only written to the log (psph uses its
# automatic deep-hole count).  Chunk i = items [400 i, 400 (i+1)); tools/collect_h9k8.py relies on this layout.
# Output in logs/overnight: h9k8_progress.log, h9k8_solutions.log, h9k8_chunk_i.{out,err,done}.  The later phases ran
# with psph/run_range.sh on the x86 boxes (see DEPLOY.md).
P=/Users/Jeff.Sponaugle/src/math/A001208/psph/psph
L=/Users/Jeff.Sponaugle/src/math/A001208/logs/overnight
TGT=${TGT:-5522}
S2=auto
N=11003; C=400
i=0
while [ $((i*C)) -lt $N ]; do
  lo=$((i*C)); hi=$(((i+1)*C)); [ $hi -gt $N ] && hi=$N
  if [ -f $L/h9k8_chunk_$i.done ]; then i=$((i+1)); continue; fi
  echo "=== chunk $i items [$lo,$hi) TGT=$TGT S2=$S2 start $(date)" >> $L/h9k8_progress.log
  $P -h 9 -k 8 -t $TGT -j 10 -p 300 -d 4 -i $lo:$hi > $L/h9k8_chunk_$i.out 2> $L/h9k8_chunk_$i.err
  rc=$?
  grep SOLUTION $L/h9k8_chunk_$i.out >> $L/h9k8_solutions.log
  grep -q "worker CPU split" $L/h9k8_chunk_$i.err && touch $L/h9k8_chunk_$i.done
  echo "=== chunk $i done rc=$rc $(date): $(grep -c SOLUTION $L/h9k8_chunk_$i.out) solutions; $(grep 'leaf candidates' $L/h9k8_chunk_$i.err | cut -c1-120)" >> $L/h9k8_progress.log
  i=$((i+1))
done
echo "=== ALL CHUNKS DONE $(date)" >> $L/h9k8_progress.log
