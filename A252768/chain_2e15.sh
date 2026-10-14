#!/bin/sh
# chain_2e15.sh -- on jeff-model2: wait for the running [1e14, 1e15] scan to exit, then
# continue from its checkpoint to 2e15.  Resuming from the checkpoint also covers the
# case where the first run died early (it then restarts where the checkpoint left off).
cd "$HOME/src/math/A252768" || exit 1
while pgrep -x a252768 >/dev/null; do sleep 10; done
lo=$(grep '^lo ' range_1e14_1e15.state | cut -d' ' -f2)
echo "$(date -u +%FT%TZ) first run ended; checkpoint lo=$lo" >> chain.log
cp range_1e14_1e15.state "state_at_${lo}.bak"
nohup ./a252768 scan 2e15 -S range_1e14_1e15.state -r 8 -i 120 > scan_1e15_2e15.txt 2> scan_1e15_2e15.log < /dev/null &
echo "$(date -u +%FT%TZ) continuation to 2e15 started, pid $!" >> chain.log
