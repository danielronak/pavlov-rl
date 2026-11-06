"""DIAGNOSTIC ONLY (training ecology, not the lab; never used to select hyperparameters — I9).
Where does the agent salivate? Split ticks by context, and early vs late lifetime."""
import json, os, pickle, sys
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import jax, numpy as np
from pavlov.config import Config, EcologyConfig, PhysiologyConfig
from pavlov.ecology import generate_batch
from pavlov.oracles import _lead_window, food_present
sys.path.insert(0, os.path.dirname(__file__))
from eval_training import run_agent, EVAL_KEY

for run in sys.argv[1:]:
    meta = json.load(open(os.path.join(run, "config.json")))
    cfg = Config(physiology=PhysiologyConfig(**meta["config"]["physiology"]), ecology=EcologyConfig(**meta["config"]["ecology"]))
    params = pickle.load(open(os.path.join(run, "checkpoints", "params_latest.pkl"), "rb"))
    s = generate_batch(jax.random.PRNGKey(EVAL_KEY), cfg.ecology, 128)
    _, p = run_agent(params, s, cfg, meta["agent"]["hidden"], jax.random.PRNGKey(1))
    pred_on = np.asarray(s.pred_on); any_on = np.asarray(s.cues).any(-1)
    food = np.asarray(food_present(s.food_onset, cfg.physiology.food_ticks))
    pre = np.asarray(_lead_window(s.expected_us, 2))
    dist_only = any_on & ~pred_on & ~food
    quiet = ~any_on & ~food
    T = p.shape[1]; early = np.zeros_like(pre); early[:, : T // 4] = True
    print(f"== {run}")
    for name, m in [("food in mouth", food), ("pred cue on (last 2 before US)", pre & ~food),
                    ("pred cue on (other)", pred_on & ~pre & ~food), ("distractor only", dist_only), ("no cue", quiet)]:
        print(f"  {name:<32} all={p[m].mean():.3f}  first-quarter={p[m & early].mean():.3f}  last-quarter={p[m & ~early & (np.arange(T) >= 3*T//4)].mean():.3f}")
