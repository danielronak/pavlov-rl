"""Seed discipline (invariant I10). Confirmatory seeds 0-19 (and replacements 20+ < 100) are
refused unless the repository has the git tag `prereg-v1`."""
from __future__ import annotations

import subprocess

PILOT_RANGE = range(100, 200)
PREREG_TAG = "prereg-v1"


def prereg_frozen(repo_dir: str = ".") -> bool:
    try:
        out = subprocess.run(["git", "tag", "--list", PREREG_TAG], cwd=repo_dir,
                             capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return PREREG_TAG in out.split()


def check_seed(seed: int, repo_dir: str = ".") -> None:
    if seed in PILOT_RANGE:
        return
    if 0 <= seed < 100:
        if not prereg_frozen(repo_dir):
            raise PermissionError(
                f"Seed {seed} is confirmatory. Freeze docs/PREREGISTRATION.md and "
                f"`git tag {PREREG_TAG}` first (CLAUDE.md invariant I10)."
            )
        return
    raise ValueError(f"Seed {seed} is outside the allowed ranges (0-99 confirmatory, 100-199 pilot).")
