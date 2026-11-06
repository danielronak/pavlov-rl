"""DIAGNOSTIC ONLY (training ecology; I9: never used for selection). Discrimination
p(saliva | predictive cue on) - p(saliva | distractor only) as a function of time in lifetime."""
import json, os, pickle, sys
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import jax, numpy as np
from pavlov.config import Config, EcologyConfig, PhysiologyConfig
from pavlov.ecology import generate_batch
from pavlov.oracles import food_present
sys.path.insert(0, os.path.dirname(__file__))
from eval_training import run_agent, EVAL_KEY

edges = [0, 50, 100, 200, 400, 800, 1600, 3000]
for run in sys.argv[1:]:
    meta = json.load(open(os.path.join(run, "config.json")))
    cfg = Config(physiology=PhysiologyConfig(**meta["config"]["physiology"]), ecology=EcologyConfig(**meta["config"]["ecology"]))
    params = pickle.load(open(os.path.join(run, "checkpoints", "params_latest.pkl"), "rb"))
    s = generate_batch(jax.random.PRNGKey(EVAL_KEY), cfg.ecology, 256)
    _, p = run_agent(params, s, cfg, meta["agent"]["hidden"], jax.random.PRNGKey(1))
    pred = np.asarray(s.pred_on); anyc = np.asarray(s.cues).any(-1)
    food = np.asarray(food_present(s.food_onset, cfg.physiology.food_ticks))
    one = np.asarray(s.cues).sum(-1) == 1  # single-cue ticks only: removes compound-stimulus confound
    dist = anyc & ~pred & ~food & one; predm = pred & ~food & one
    t = np.arange(p.shape[1])[None].repeat(p.shape[0], 0)
    print(f"== {run}\n  {'ticks':>12} {'p|pred':>7} {'p|dist':>7} {'diff':>7}")
    for a, b in zip(edges[:-1], edges[1:]):
        w = (t >= a) & (t < b)
        pp, pd = p[predm & w].mean(), p[dist & w].mean()
        print(f"  {a:>5}-{b:<6} {pp:7.3f} {pd:7.3f} {pp-pd:+7.3f}")
