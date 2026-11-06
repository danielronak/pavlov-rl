"""Oracle-only landscape over physiology parameters (no agent involved; I9-safe)."""
import itertools
import jax
from pavlov.config import Config, PhysiologyConfig
from pavlov.ecology import generate_batch
from pavlov.oracles import ORACLES, cue_oracle
from scripts_util import lifetime_returns  # noqa

if __name__ == "__main__":
    cfg = Config()
    sched = generate_batch(jax.random.PRNGKey(100), cfg.ecology, 256)
    print(f"{'k_up':>5}{'k_dn':>6}{'cost':>6} | {'reflex':>7}{'predOn':>7}{'cueOr':>7}{'k*':>4} | {'gain%':>6}{'predOn-reflex':>14}")
    for k_up, k_down, cost in itertools.product([0.15, 0.25, 0.35], [0.05, 0.15], [0.005, 0.01, 0.02]):
        pc = PhysiologyConfig(k_up=k_up, k_down=k_down, cost=cost)
        rf = lifetime_returns(sched, ORACLES["reflex"](sched, pc), pc).mean()
        po = lifetime_returns(sched, ORACLES["pred_cue_on"](sched, pc), pc).mean()
        sc = {k: lifetime_returns(sched, cue_oracle(sched, pc, k), pc).mean() for k in range(1, 11)}
        kb = max(sc, key=sc.get)
        print(f"{k_up:>5}{k_down:>6}{cost:>6} | {rf:7.2f}{po:7.2f}{sc[kb]:7.2f}{kb:>4} | {100*(sc[kb]-rf)/rf:6.1f}{po-rf:14.2f}")
