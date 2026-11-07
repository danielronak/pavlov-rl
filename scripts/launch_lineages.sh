#!/usr/bin/env bash
# Common-ancestor pilots (DECISIONS 2026-11-07). Idempotent: resumes unfinished lineages exactly,
# skips finished or already-running ones. Safe to re-run after a worker restart.
cd "$(dirname "$0")/.." || exit 1
OUT=results/pilot_lineages
mkdir -p "$OUT"
launch() {  # seed ancestor config name generations
  local seed=$1 anc=$2 cfg=$3 name=$4 gens=$5 dir="$OUT/$4"
  if [ -f "$dir/checkpoints/params_$(printf %05d "$gens").pkl" ]; then echo "$name: finished ($gens)"; return; fi
  if pgrep -f -- "--out $dir" >/dev/null; then echo "$name: running"; return; fi
  echo "$name: launching to $gens (resume if state exists)"
  nohup .venv/bin/python scripts/train_es.py --config "$cfg" --seed "$seed" --generations "$gens" \
    --init-from "results/pilot_r4/${anc}_es/checkpoints/params_04000.pkl" --out "$dir" --resume \
    >> "$OUT/train_$name.log" 2>&1 &
}
launch 116 109 configs/exp/pilot_a3.yaml      anc109_stable    2000
launch 118 110 configs/exp/pilot_a3.yaml      anc110_stable    2000
launch 117 109 configs/exp/pilot_a4_v05.yaml  anc109_variable  4000   # option 2: extension
launch 119 110 configs/exp/pilot_a4_v05.yaml  anc110_variable  4000   # option 2: extension
launch 120 109 configs/exp/pilot_po_v05.yaml  anc109_partialom 2000   # option 1: partial omission
launch 121 110 configs/exp/pilot_po_v05.yaml  anc110_partialom 2000   # option 1: partial omission
