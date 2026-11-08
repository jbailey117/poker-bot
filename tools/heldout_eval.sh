#!/usr/bin/env bash
# Evaluate one agent.py configuration against every held-out style bot.
#   ./tools/heldout_eval.sh "MODEL_PATH=models/v4/q_trees.npz" [seeds] [port_base]
# Prints one POOLED line per style (positive = agent.py wins).
set -u
cd "$(dirname "$0")/.."
ENVSTR=${1:-}
SEEDS=${2:-1-4}
BASE=${3:-5700}
STYLES=${STYLES:-"nit potodds balanced trapper guide"}
OUT=$(mktemp -d)
i=0
for st in $STYLES; do
  p1=$((BASE + 2*i)); p2=$((p1 + 1)); i=$((i+1))
  python tools/selfplay.py --a-file opponents/style_bot.py --a "STYLE=$st" --b "$ENVSTR" \
      --seeds "$SEEDS" --ports "$p1,$p2" > "$OUT/$st.txt" 2>&1 &
done
wait
for st in $STYLES; do printf "%-9s " "$st"; grep POOLED "$OUT/$st.txt" | sed 's/POOLED over //'; done
rm -rf "$OUT"
