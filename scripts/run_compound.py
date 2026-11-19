"""Compound-cue follow-up (exploratory; docs/DECISIONS.md 2026-11-19).

Sisters of the confirmatory ancestors anc_00..09 evolve 2000 generations in the ancestor ecology plus
compound events (configs/exp/compound.yaml), then get the Phase-2 gate, every lab protocol and the
phenomena battery. Comparison: the existing stable sisters of the same ancestors.

Restart-safe like run_confirmatory.py (reuses its helpers):
  nohup .venv/bin/python scripts/run_compound.py --jobs 4 >> results/compound/driver.log 2>&1 &
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from run_confirmatory import ANC_GENS, PY, SIS_GENS, ckpt, post_cmds, run_dir, running, trained  # noqa: E402

ROOT = "results/compound"
RUNS = [(f"compound_{a:02d}", "configs/exp/compound.yaml", 140 + a, a) for a in range(10)]


def pending():
    jobs = []
    for name, cfg, seed, a in RUNS:
        d = os.path.join(ROOT, name)
        if not trained(d, SIS_GENS):
            anc = os.path.join(run_dir("ancestor", a), "checkpoints", ckpt(ANC_GENS))
            jobs.append((f"train {name}", [PY, "scripts/train_es.py", "--config", cfg, "--seed", str(seed),
                                           "--generations", str(SIS_GENS), "--out", d, "--resume",
                                           "--init-from", anc], d))
            continue
        steps = [(c, m) for c, m in post_cmds(d, SIS_GENS) if "ro_learning_check" not in c[1]]
        steps.append(([PY, "scripts/run_phenomena.py", d], os.path.join(d, "phenomena.json")))
        for cmd, marker in steps:
            if not os.path.exists(marker):
                jobs.append((f"post {name} {os.path.basename(cmd[1])}", cmd, None))
                break
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(ROOT, exist_ok=True)
    procs, fails = {}, {}
    while True:
        for label in [k for k, (p, _) in procs.items() if p.poll() is not None]:
            p, log = procs.pop(label)
            log.close()
            print(time.strftime("%H:%M:%S"), f"finished ({p.returncode}): {label}", flush=True)
            if p.returncode != 0:
                fails[label] = fails.get(label, 0) + 1
                if fails[label] >= 3:
                    sys.exit(f"STOP: '{label}' failed 3 times")
        jobs = pending()
        if not jobs and not procs:
            print(time.strftime("%H:%M:%S"), "ALL DONE", flush=True)
            return
        for label, cmd, d in jobs:
            if len(procs) >= args.jobs:
                break
            if label in procs or (d is not None and running(d)):
                continue
            if d is not None:
                os.makedirs(d, exist_ok=True)
                if not os.path.exists(os.path.join(d, "git_hash")):
                    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
                    open(os.path.join(d, "git_hash"), "w").write(head)
            log = open(os.path.join(ROOT, label.replace(" ", "_").replace(".py", "") + ".log"), "a")
            procs[label] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log)
            print(time.strftime("%H:%M:%S"), f"started: {label}", flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
