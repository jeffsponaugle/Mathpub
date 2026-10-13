#!/bin/sh
# One-off reconfiguration script (rebalance #6, Oct 3 2026 ~01:10 PDT, run on mathb in its psph directory): after
# chunk 21 run chunks 22-24 at TGT 186943 (jobs7.sh); chunks 25 and 26 go to mathd and mathg.
# Pattern: kill the old wrapper only (the running run_range.sh driver and psph keep going), touch .done SKIP markers
# for chunks that move to another box (unless they had already started there), start the new wrapper with nohup,
# log the switch in jobs.log and show the running wrappers / drivers.  Kept for the record of how the chunks moved.
cd ~/A001208/psph
for p in $(pgrep -f "sh [j]obs6.sh"); do kill $p; done
sleep 1
for j in 2 3 4 5 6; do
  if [ -f logs/h27k6b/chunk_$j.err ]; then echo "WARNING chunk $((j+20)) already started under h27k6b"; else touch logs/h27k6b/chunk_$j.done; fi
done
nohup sh jobs7.sh > /dev/null 2>&1 < /dev/null &
sleep 2
echo "wrappers: $(pgrep -af '[j]obs[0-9]*.sh' | cut -c1-60 | tr '\n' ' ')"; echo "drivers: $(pgrep -af '[r]un_range' | cut -c1-90)"
echo "=== SWITCH2 $(date): after chunk 21, chunks 22-24 at TGT 186943 (h27k6h); 25 -> mathd, 26 -> mathg" >> jobs.log
tail -1 logs/h27k6b/chunk_1.err | cut -c1-120
