#!/usr/bin/env bash
# GPU smoke test: checks that JAX sees the GPU, that the tests pass, and measures ES throughput at
# the search-check size (pop 256, 32 lifetimes). Writes only to results/gpu_smoke (git-ignored).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY -c "import jax; d = jax.devices(); print('devices:', d); assert d[0].platform == 'gpu', 'JAX does not see the GPU (see docs/GPU.md)'"
$PY -m pytest -q
rm -rf results/gpu_smoke
$PY scripts/train_es.py --config configs/exp/pilot_po_v05.yaml --seed 199 --generations 6 --pop 256 --lifetimes 32 \
  --init-from results/pilot_r4/109_es/checkpoints/params_04000.pkl --out results/gpu_smoke | tee results/gpu_smoke.log
$PY - <<'EOF'
import json
recs = [json.loads(l) for l in open("results/gpu_smoke/train_log.jsonl")]
r = recs[-1]
sec_per_gen = r["wall_s"] / (r["generation"] + 1)
print(f"\n~{sec_per_gen:.2f} s/generation (includes compile) -> one 2000-generation lineage ~ {sec_per_gen*2000/3600:.1f} h; both ~ {2*sec_per_gen*2000/3600:.1f} h")
EOF
