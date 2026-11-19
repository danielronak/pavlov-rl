"""Robustness check (exploratory; docs/DECISIONS.md 2026-11-19).

Do the core results survive changes to settings that were fixed, not chosen for the result? New
ancestors are evolved exactly like the confirmatory ancestors (4000 generations, A3 ecology, q = 1,
v = 0) but with a smaller or larger GRU (32, 128 units; default 64) or with a weaker or stronger
physiology (saliva build-up k_up and saliva benefit 1 - b0 both x0.75 or x1.25). 3 seeds per variant.
Then the Phase-2 gate and every lab protocol. Reference: the 20 confirmatory ancestors.

Restart-safe like run_confirmatory.py (reuses its helpers):
  nohup .venv/bin/python scripts/run_robust.py --jobs 4 >> results/robust/driver.log 2>&1 &
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from run_confirmatory import ANC_GENS, PY, post_cmds, running, trained  # noqa: E402

ROOT = "results/robust"
BASE = "configs/confirm/ancestor.yaml"
# (variant, config, hidden, first seed); the slowest variant first so it does not finish last alone
VARIANTS = [("h128", BASE, 128, 153), ("h32", BASE, 32, 150),
            ("phys_weak", "configs/exp/robust_phys_weak.yaml", 64, 156),
            ("phys_strong", "configs/exp/robust_phys_strong.yaml", 64, 159)]
RUNS = [(f"{v}_{s}", cfg, h, s) for v, cfg, h, s0 in VARIANTS for s in range(s0, s0 + 3)]


def pending():
    jobs = []
    for name, cfg, hidden, seed in RUNS:
        d = os.path.join(ROOT, name)
        if not trained(d, ANC_GENS):
            jobs.append((f"train {name}", [PY, "scripts/train_es.py", "--config", cfg, "--seed", str(seed),
                                           "--generations", str(ANC_GENS), "--hidden", str(hidden),
                                           "--out", d, "--resume"], d))
            continue
        for cmd, marker in post_cmds(d, ANC_GENS):
            if "ro_learning_check" in cmd[1]:
                continue  # v = 0 runs: the R-O check is not part of this question
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
