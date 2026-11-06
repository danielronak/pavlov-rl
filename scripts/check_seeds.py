"""Usage: python scripts/check_seeds.py 0 5 100  -> exits non-zero if any seed is not allowed now."""
import sys

from pavlov.seeds import check_seed, prereg_frozen

if __name__ == "__main__":
    print(f"prereg frozen: {prereg_frozen()}")
    bad = 0
    for s in map(int, sys.argv[1:] or ["0", "100"]):
        try:
            check_seed(s)
            print(f"seed {s}: allowed")
        except (PermissionError, ValueError) as e:
            print(f"seed {s}: REFUSED — {e}")
            bad += 1
    sys.exit(1 if bad else 0)
