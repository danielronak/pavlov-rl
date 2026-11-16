#!/usr/bin/env bash
# Search-limit check on CPU (DECISIONS 2026-11-16): same seeds/settings as gpu_search_check.sh, both
# lineages in parallel in the background. Idempotent: resumes exactly, skips finished/running ones.
cd "$(dirname "$0")/.." || exit 1
OUT=results/search_check
mkdir -p "$OUT"
run() {  # seed ancestor name
  local seed=$1 anc=$2 name=$3 dir="$OUT/$3"
  if [ -f "$dir/checkpoints/params_02000.pkl" ]; then echo "$name: finished"; return; fi
  if pgrep -f -- "--out $dir" >/dev/null; then echo "$name: running"; return; fi
  echo "$name: launching (resume if state exists)"
  nohup .venv/bin/python scripts/train_es.py --config configs/exp/pilot_po_v05.yaml --seed "$seed" \
    --generations 2000 --pop 256 --lifetimes 32 \
    --init-from "results/pilot_r4/${anc}_es/checkpoints/params_04000.pkl" --out "$dir" --resume \
    >> "$OUT/train_$name.log" 2>&1 &
}
run 122 109 anc109_po_big
run 123 110 anc110_po_big
