#!/bin/sh
# restart the boundary-window run at normal priority (resumable via .done markers)
#
# One-off on mathg, Oct 3 2026: kills bnd_h27k6.sh and its 4-thread psph (the pattern "-j [4]" leaves the main
# 32-thread chunk run alone), removes "nice -n 19" from bnd_h27k6.sh with GNU sed -i, restarts it with nohup, and
# prints how many window and main processes are running.  Windows already marked done are not redone.
cd ~/A001208/psph
for p in $(pgrep -f "sh [b]nd_h27k6.sh"); do kill $p; done
sleep 1
for p in $(pgrep -f "psph -h 27 -k 6 -t 176381 -j [4] -p"); do kill $p; done
sleep 1
sed -i 's/^  nice -n 19 \.\/psph/  .\/psph/' bnd_h27k6.sh
grep -n "psph -h 27" bnd_h27k6.sh
nohup sh bnd_h27k6.sh > bnd.log 2>&1 < /dev/null &
sleep 3
echo "running: $(pgrep -af '[b]nd_h27k6' | wc -l) script, $(pgrep -af 'psph -h 27 -k 6 -t 176381 -j [4] -p' | wc -l) psph"
echo "main chunk run still up: $(pgrep -af 'psph -h 27 -k 6 -t 176381 -j 3[2] ' | wc -l)"
