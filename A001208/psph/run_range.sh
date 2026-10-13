#!/bin/sh
# Resumable, chunked driver for psph over a range of work items (for splitting a search across machines).
#
#   run_range.sh TAG H K TGT D LO HI CHUNK THREADS [BINARY]
#
#   TAG      name for the log directory (e.g. h9k8_box1)
#   H K TGT  the search (TGT = best known lower bound + 1)
#   D        split depth (must be the same on every machine so that item indices agree)
#            NOTE: the item enumeration also depends on TGT (higher targets prune a few more prefixes through the
#            element-wise lower bounds), so machines running DIFFERENT targets have slightly different index spaces
#            (e.g. n(27,6) d=4: 2964437 items at TGT 176381, 2964430 at 177915). Either use one TGT everywhere or
#            re-run small windows around every chunk boundary at the lowest TGT afterwards (see bnd_h27k6.sh,
#            tools/collect_h27k6.py). Check with: psph -h H -k K -t TGT -d D -i 0:0 (prints the item count).
#   LO HI    this machine's item range [LO,HI)  (item counts: run  psph -h H -k K -t TGT -d D -i 0:0)
#   CHUNK    items per psph invocation (a finished chunk is marked by a .done file -> re-run to resume)
#   THREADS  worker threads (number of cores)
#   BINARY   psph (h <= 254) or psph16 (h >= 255); default ./psph
#
# Output: logs/TAG/progress.log, logs/TAG/solutions.log (SOLUTION lines), per-chunk .out/.err/.done.
# Result of the whole search = max n_h over ALL machines' solutions.log once every chunk everywhere is done.
#
# How it works: chunk i covers items [LO + i*CHUNK, min(LO + (i+1)*CHUNK, HI)).  Chunks that have a .done marker
# are skipped, so the driver can be stopped (kill it BEFORE its psph process) and simply re-run; an interrupted chunk
# is run again from its start (its partial .out/.err are overwritten).  A chunk is marked done only when psph printed
# its final statistics line ("worker CPU split"), i.e. it ran to the end; psph's exit code is only logged.  A .done
# marker can also be touched by hand to SKIP a chunk (used when a chunk was moved to another machine), which is why
# the audit scripts (tools/collect_*.py) also require the statistics line.
# Run it from the psph directory: the logs go to ./logs/TAG and BINARY is resolved relative to it.
# progress.log lines: "=== chunk i items [lo,hi) h=H k=K TGT=T start DATE", "=== chunk i done rc=RC DATE: N solutions;
# <leaf statistics>", and at the end "=== RANGE [LO,HI) COMPLETE DATE".  Standard output at the end: the sorted,
# de-duplicated SOLUTION lines of this TAG.
# Exit status: 1 for a usage error (fewer than 9 arguments), otherwise 0; failed chunks show up as chunks without .done.
TAG=$1; H=$2; K=$3; TGT=$4; D=$5; LO=$6; HI=$7; C=$8; J=$9; P=${10:-./psph}
[ -z "$J" ] && { echo "usage: run_range.sh TAG H K TGT D LO HI CHUNK THREADS [BINARY]"; exit 1; }
L=logs/$TAG; mkdir -p $L
i=0
while :; do
  lo=$((LO + i*C)); [ $lo -ge $HI ] && break
  hi=$((lo + C)); [ $hi -gt $HI ] && hi=$HI
  if [ -f $L/chunk_$i.done ]; then i=$((i+1)); continue; fi
  echo "=== chunk $i items [$lo,$hi) h=$H k=$K TGT=$TGT start $(date)" >> $L/progress.log
  $P -h $H -k $K -t $TGT -j $J -p 300 -d $D -i $lo:$hi > $L/chunk_$i.out 2> $L/chunk_$i.err
  rc=$?
  grep SOLUTION $L/chunk_$i.out >> $L/solutions.log
  grep -q "worker CPU split" $L/chunk_$i.err && touch $L/chunk_$i.done
  echo "=== chunk $i done rc=$rc $(date): $(grep -c SOLUTION $L/chunk_$i.out) solutions; $(grep -m1 'leaf candidates' $L/chunk_$i.err | cut -c1-110)" >> $L/progress.log
  i=$((i+1))
done
echo "=== RANGE [$LO,$HI) COMPLETE $(date)" >> $L/progress.log
sort -u $L/solutions.log 2>/dev/null
