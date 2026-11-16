"""Confirmatory pipeline driver (docs/PREREGISTRATION.md, frozen at git tag prereg-v1).

For every ancestor seed a (0-19, replacements 20-24):
  1. evolve the ancestor   results/confirm/anc_<a>      configs/confirm/ancestor.yaml   4000 gens
  2. evaluate it (Phase-2 gate + exclusion rule). If excluded, the next replacement seed is added.
  3. split into three sister lineages from params_04000.pkl, 2000 gens each, fresh Adam:
       stable_<a>            seed 25 + a   configs/confirm/stable.yaml
       digestion_<a>         seed 50 + a   configs/confirm/digestion.yaml
       partial_omission_<a>  seed 75 + a   configs/confirm/partial_omission.yaml
  4. for the ancestor and every sister: Phase-2 gate, all lab protocols, R-O learning check.

Idempotent and restart-safe: re-run after a worker restart and it resumes every run exactly from its
last 100-generation checkpoint, skipping finished steps and runs that are already in progress.
Usage:  nohup .venv/bin/python scripts/run_confirmatory.py --jobs 4 >> results/confirm/driver.log 2>&1 &
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

ROOT = "results/confirm"
PY = ".venv/bin/python"
ANC_GENS, SIS_GENS = 4000, 2000
N_SEEDS, MAX_REPLACEMENT = 20, 24
ARMS = {"stable": 25, "digestion": 50, "partial_omission": 75}


def ckpt(gens):
    return f"params_{gens:05d}.pkl"


def run_dir(arm, a):
    return os.path.join(ROOT, f"{'anc' if arm == 'ancestor' else arm}_{a:02d}")


def trained(d, gens):
    return os.path.exists(os.path.join(d, "checkpoints", ckpt(gens)))


def excluded(a):
    """Pre-registered exclusion: meta-training diverged (NaN) or final training return below reflex."""
    e = json.load(open(os.path.join(run_dir("ancestor", a), f"eval_training_params_{ANC_GENS:05d}.json")))
    return (not math.isfinite(e["agent"])) or e["agent"] < e["reflex"]


def running(d):
    return subprocess.run(["pgrep", "-f", "--", f"--out {d}( |$)"], capture_output=True).returncode == 0


def train_cmd(arm, a, d, gens):
    seed = a if arm == "ancestor" else ARMS[arm] + a
    cmd = [PY, "scripts/train_es.py", "--config", f"configs/confirm/{arm}.yaml", "--seed", str(seed),
           "--generations", str(gens), "--out", d, "--resume"]
    if arm != "ancestor":
        cmd += ["--init-from", os.path.join(run_dir("ancestor", a), "checkpoints", ckpt(ANC_GENS))]
    return cmd


def post_cmds(d, gens):
    """Post-training steps, each with the file that marks it done."""
    c = ckpt(gens)
    tag = c.replace(".pkl", "")
    return [([PY, "scripts/eval_training.py", d, "--ckpt", c], os.path.join(d, f"eval_training_{tag}.json")),
            ([PY, "scripts/ro_learning_check.py", d, "--ckpt", c], os.path.join(d, f"ro_learning_{tag}.json")),
            ([PY, "scripts/run_lab.py", "--run", d, "--ckpt", c, "--protocol", "all"],
             os.path.join(d, "lab", "positive.json"))]


def pending(active):
    """Next jobs in priority order: ancestors' training, ancestor post-processing, sisters."""
    jobs, all_done = [], True
    for a in active:
        d = run_dir("ancestor", a)
        if not trained(d, ANC_GENS):
            all_done = False
            jobs.append((f"train anc_{a:02d}", train_cmd("ancestor", a, d, ANC_GENS), d, None))
            continue
        for cmd, marker in post_cmds(d, ANC_GENS):
            if not os.path.exists(marker):
                all_done = False
                jobs.append((f"post anc_{a:02d} {os.path.basename(cmd[1])}", cmd, None, marker))
                break
        if not os.path.exists(os.path.join(d, f"eval_training_params_{ANC_GENS:05d}.json")) or excluded(a):
            continue
        for arm in ARMS:
            sd = run_dir(arm, a)
            if not trained(sd, SIS_GENS):
                all_done = False
                jobs.append((f"train {arm}_{a:02d}", train_cmd(arm, a, sd, SIS_GENS), sd, None))
                continue
            for cmd, marker in post_cmds(sd, SIS_GENS):
                if not os.path.exists(marker):
                    all_done = False
                    jobs.append((f"post {arm}_{a:02d} {os.path.basename(cmd[1])}", cmd, None, marker))
                    break
    return jobs, all_done


def active_seeds():
    """Seeds 0-19, plus one replacement (20, 21, ...) for every excluded ancestor."""
    active, nxt = list(range(N_SEEDS)), N_SEEDS
    i = 0
    while i < len(active):
        a = active[i]
        ev = os.path.join(run_dir("ancestor", a), f"eval_training_params_{ANC_GENS:05d}.json")
        if os.path.exists(ev) and excluded(a):
            if nxt > MAX_REPLACEMENT:
                sys.exit(f"STOP: more than {MAX_REPLACEMENT - N_SEEDS + 1} exclusions — return to the PI")
            active.append(nxt)
            nxt += 1
        i += 1
    return active


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(ROOT, exist_ok=True)
    procs, fails = {}, {}  # label -> (Popen, logfile); label -> failure count
    while True:
        for label in [k for k, (p, _) in procs.items() if p.poll() is not None]:
            p, log = procs.pop(label)
            print(time.strftime("%H:%M:%S"), f"finished ({p.returncode}): {label}", flush=True)
            log.close()
            if p.returncode != 0:
                fails[label] = fails.get(label, 0) + 1
                if fails[label] >= 3:
                    for q, _ in procs.values():
                        q.wait()
                    sys.exit(f"STOP: '{label}' failed 3 times (see its log) — needs a human decision")
        jobs, all_done = pending(active_seeds())
        if all_done and not procs:
            print(time.strftime("%H:%M:%S"), "ALL DONE", flush=True)
            return
        for label, cmd, d, _ in jobs:
            if len(procs) >= args.jobs:
                break
            if label in procs or (d is not None and running(d)):
                continue
            name = label.replace(" ", "_").replace(".py", "")
            log = open(os.path.join(ROOT, f"{name}.log"), "a")
            procs[label] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log)
            print(time.strftime("%H:%M:%S"), f"started: {label}", flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
