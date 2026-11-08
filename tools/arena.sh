#!/usr/bin/env bash
# Play agent.py against every baseline bot (full matches) and print a summary.
#
#   ./tools/arena.sh                 # 1000 deals, seed 1
#   DEALS=300 SEED=7 ./tools/arena.sh
#   OPPONENTS="baseline-bots/baseline_random.py other_bot.py" ./tools/arena.sh
#
# Each opponent gets fresh processes for both bots. Full judge reports are
# saved under /tmp/arena/.
set -u
cd "$(dirname "$0")/.."
DEALS=${DEALS:-1000}
SEED=${SEED:-1}
PY=${PYTHON:-python}
OPPONENTS=${OPPONENTS:-$(ls baseline-bots/*.py)}
OUT=/tmp/arena
mkdir -p "$OUT"

wait_for() {  # wait until a bot answers on its port
  for _ in $(seq 1 100); do
    curl -s "http://127.0.0.1:$1/" >/dev/null 2>&1 && return 0
    sleep 0.1
  done
  echo "bot on port $1 did not start" >&2; return 1
}

for opp in $OPPONENTS; do
  name=$(basename "$opp" .py)
  PORT=5108 $PY agent.py >"$OUT/agent_$name.log" 2>&1 & a=$!
  PORT=5109 $PY "$opp" >"$OUT/opp_$name.log" 2>&1 & b=$!
  wait_for 5108 && wait_for 5109 && \
    $PY judge_engine.py --agent1 5108 --agent2 5109 --deals "$DEALS" --seed "$SEED" >"$OUT/$name.txt" 2>&1
  kill $a $b 2>/dev/null; wait $a $b 2>/dev/null
  echo "=== vs $name ==="
  grep -E "bb/deal|interval|RESULT|errors|time|FORFEIT|EMPTY" "$OUT/$name.txt" | head -12
done
