"""DIAGNOSTIC: discrimination BEFORE the first food of each lifetime (zero pairings).
Any difference here cannot come from learning the cue-food relation -> leak or artifact."""
import json, os, pickle, sys
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import jax, numpy as np
from pavlov.config import Config, EcologyConfig, PhysiologyConfig
from pavlov.ecology import generate_batch
sys.path.insert(0, os.path.dirname(__file__))
from eval_training import run_agent, EVAL_KEY
for run in sys.argv[1:]:
    meta = json.load(open(os.path.join(run, "config.json")))
    cfg = Config(physiology=PhysiologyConfig(**meta["config"]["physiology"]), ecology=EcologyConfig(**meta["config"]["ecology"]))
    params = pickle.load(open(os.path.join(run, "checkpoints", "params_latest.pkl"), "rb"))
    s = generate_batch(jax.random.PRNGKey(EVAL_KEY), cfg.ecology, 1024)
    _, p = run_agent(params, s, cfg, meta["agent"]["hidden"], jax.random.PRNGKey(1))
    food = np.asarray(s.food_onset); pred = np.asarray(s.pred_on); anyc = np.asarray(s.cues).any(-1)
    first = np.where(food.any(1), food.argmax(1), food.shape[1])
    pre = np.arange(food.shape[1])[None] < first[:, None]
    pm, dm = pred & pre, anyc & ~pred & pre
    # also: ticks where exactly one cue is on (removes compound-stimulus differences)
    one = np.asarray(s.cues).sum(-1) == 1
    print(f"== {run}: ticks before first food: pred={pm.sum()} dist={dm.sum()}")
    print(f"  p|pred={p[pm].mean():.3f}  p|dist={p[dm].mean():.3f}  diff={p[pm].mean()-p[dm].mean():+.3f}")
    print(f"  single-cue only: p|pred={p[pm&one].mean():.3f}  p|dist={p[dm&one].mean():.3f}")
    print(f"  any-cue-on fraction during pred vs dist (compound load): {np.asarray(s.cues).sum(-1)[pm].mean():.2f} vs {np.asarray(s.cues).sum(-1)[dm].mean():.2f}")
