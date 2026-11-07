"""Oracle gates G9' and G10' for the partial-omission R-O arm (DECISIONS 2026-11-07), closed loop."""
import argparse, json
import jax, numpy as np
from pavlov.config import load_config
from pavlov.ecology import generate_batch
from pavlov.oracles import ORACLES, cue_oracle, gate_aware, gate_probe_then_aware
from pavlov.rollout import fixed_action_returns

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True); ap.add_argument("--n", type=int, default=512)
ap.add_argument("--seed", type=int, default=100); ap.add_argument("--out", default=None)
a = ap.parse_args()
cfg = load_config(a.config); pc = cfg.physiology
assert cfg.ecology.ro_family == "partial_omission"
s = generate_batch(jax.random.PRNGKey(a.seed), cfg.ecology, a.n)
R = lambda act: np.asarray(fixed_action_returns(s, act, cfg))
res = {n: R(ORACLES[n](s, pc)) for n in ("never", "reflex", "any_cue", "pred_cue_on")}
sb = {k: R(cue_oracle(s, pc, k)) for k in range(1, 9)}; kb = max(sb, key=lambda k: sb[k].mean())
res["cue_oracle"] = sb[kb]
sa = {k: R(gate_aware(cue_oracle(s, pc, k), s)) for k in range(1, 9)}; ka = max(sa, key=lambda k: sa[k].mean())
res["cue_oracle_aware"] = sa[ka]
res["reflex_aware"] = R(gate_aware(ORACLES["reflex"](s, pc), s))
res["probe_then_aware"] = R(gate_probe_then_aware(s, pc, ka))
for n, v in sorted(res.items(), key=lambda kv: -kv[1].mean()): print(f"  {n:<18}{v.mean():8.2f}")
def paired(x, y):
    d = res[x] - res[y]; se = d.std(ddof=1) / np.sqrt(len(d)); return float(d.mean()), float(d.mean() - 1.96*se), float(d.mean() + 1.96*se)
gap = res["cue_oracle_aware"].mean() - res["reflex_aware"].mean()
m9 = paired("cue_oracle_aware", "cue_oracle"); m10 = paired("probe_then_aware", "never")
g9 = m9[1] > 0 and m9[0] >= 0.10 * gap; g10 = m10[1] > 0 and m10[0] >= 0.25 * gap
print(f"  aware gap {gap:.2f}; gated lifetimes {float(np.asarray(s.params.gated).mean()):.2f}")
print(f"  G9'  aware - blind      {m9[0]:+.2f} [{m9[1]:+.2f}, {m9[2]:+.2f}]  {'PASS' if g9 else 'FAIL'}")
print(f"  G10' probe - never      {m10[0]:+.2f} [{m10[1]:+.2f}, {m10[2]:+.2f}]  {'PASS' if g10 else 'FAIL'}")
print(f"GATE: {'PASS' if g9 and g10 else 'FAIL'}")
if a.out:
    json.dump({"means": {k: float(v.mean()) for k, v in res.items()}, "G9p": m9, "G10p": m10, "pass": bool(g9 and g10)}, open(a.out, "w"), indent=2)
