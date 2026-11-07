#!/usr/bin/env bash
# Search-limit check (DECISIONS 2026-11-07): partial-omission split from ancestors 109 and 110 with a
# 4x larger ES search (pop 256, 32 lifetimes/member), 2000 generations each, run one after another.
# Safe to stop (Ctrl-C, closing the laptop) and re-run: each lineage resumes exactly from its last
# 100-generation checkpoint, and finished lineages are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY -c "import jax; assert jax.devices()[0].platform == 'gpu', 'JAX does not see the GPU (see docs/GPU.md)'"
OUT=results/search_check
mkdir -p "$OUT"
run() {  # seed ancestor name
  local seed=$1 anc=$2 name=$3
  if [ -f "$OUT/$name/checkpoints/params_02000.pkl" ]; then echo "$name: finished"; return; fi
  echo "== $name (seed $seed, ancestor $anc)"
  $PY scripts/train_es.py --config configs/exp/pilot_po_v05.yaml --seed "$seed" --generations 2000 \
    --pop 256 --lifetimes 32 --init-from "results/pilot_r4/${anc}_es/checkpoints/params_04000.pkl" \
    --out "$OUT/$name" --resume 2>&1 | tee -a "$OUT/train_$name.log"
}
run 122 109 anc109_po_big
run 123 110 anc110_po_big
echo "ALL DONE"
