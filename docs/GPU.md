# Running on the local GPU (RTX 3050, 4 GB)

JAX's CUDA build runs on Linux. On Windows, use **WSL2 (Ubuntu)**. The NVIDIA Windows driver
already exposes the GPU to WSL2, so do not install a Linux NVIDIA driver inside WSL.

## One-time setup (inside Linux / WSL2)

```bash
nvidia-smi
```

```bash
git clone https://github.com/danielronak/pavlov-rl.git ~/pavlov-rl
```

```bash
cd ~/pavlov-rl && python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -U "jax[cuda12]" optax numpy scipy pandas matplotlib pyyaml pytest && .venv/bin/pip install -e .
```

```bash
.venv/bin/python -c "import jax; print(jax.devices())"
```

The last command must print a `CudaDevice`. If it prints `CpuDevice`, stop and send me the output.

## Each run

Claude will give you the exact run command each time. The pattern is always:

```bash
cd ~/pavlov-rl && git pull
```

```bash
.venv/bin/pytest -q
```

then the training command(s), then:

```bash
git add results/ && git commit -m "GPU run: <what>" && git push
```

Notes:
- Checkpoints are git-ignored except `params_latest.pkl` copies that Claude asks you to commit
  explicitly. Logs and metrics JSON are always committed.
- Do not edit code on the laptop. If something fails, push the log and tell Claude.
- Seeds 0–19 are refused until the preregistration is frozen. That is intended.
