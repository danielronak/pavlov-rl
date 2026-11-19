"""Figures for the confirmatory results, regenerated only from results/confirm/ (CLAUDE.md).

Reads results/confirm/analysis.json (written by analyze_confirmatory.py) and the lab JSONs.
Writes figures/fig{1..4}_*.png. Error bars / bands = 95% percentile bootstrap CIs over seeds.
Usage: python scripts/make_figures.py [--root results/confirm] [--out figures]
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pavlov.analysis.stats import bootstrap_ci  # noqa: E402

ARMS = ["ancestor", "stable", "digestion", "partial_omission"]
LABEL = {"ancestor": "Ancestor", "stable": "Stable\nsister",
         "digestion": "Digestion-\nvariable sister", "partial_omission": "Partial-\nomission sister"}
COLOR = {"ancestor": "#2a78d6", "stable": "#eb6834", "digestion": "#1baf7a", "partial_omission": "#eda100"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


PAPER = False  # --paper: no figure-level titles (the paper's captions carry them)


def _suptitle(fig, text, **kw):
    if not PAPER:
        fig.suptitle(text, **kw)


def run_name(arm, a):
    return f"anc_{a:02d}" if arm == "ancestor" else f"{arm}_{a:02d}"


def mean_ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    lo, hi = bootstrap_ci(x)
    return x.mean(), lo, hi


def curve_ci(rows):
    """rows [seeds, trials] -> mean, lo, hi per trial."""
    rows = np.asarray(rows, float)
    out = np.array([mean_ci(rows[:, t]) for t in range(rows.shape[1])])
    return out[:, 0], out[:, 1], out[:, 2]


def strip(ax, xs, values, color, jitter_seed=0):
    """Per-seed dots plus mean and 95% CI."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(jitter_seed)
    ax.scatter(xs + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=color, alpha=0.45, linewidths=0, zorder=2)
    m, lo, hi = mean_ci(v)
    ax.errorbar([xs + 0.28], [m], yerr=[[m - lo], [hi - m]], fmt="o", color=color, ms=6, lw=2,
                capsize=0, zorder=3, markeredgecolor="#fcfcfb", markeredgewidth=1.5)
    return m


def fig1_curves(root, seeds, out):
    """Acquisition (CS+ vs CS-), extinction and recovery in the ancestors and stable sisters."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, arm in zip(axes, ["ancestor", "stable"]):
        m = {k: [] for k in ("acq_curve_cs_plus", "acq_curve_cs_minus", "ext_curve_cs_plus", "recovery_curve")}
        for a in seeds:
            j = json.load(open(os.path.join(root, run_name(arm, a), "lab", "pavlov_standard.json")))
            for k in m:
                m[k].append(j["metrics"][k])
        segs = [("acq_curve_cs_plus", 1, "Bell (CS+)", COLOR["ancestor"]),
                ("acq_curve_cs_minus", 1, "Metronome (CS−)", COLOR["stable"]),
                ("ext_curve_cs_plus", 28, None, COLOR["ancestor"]),
                ("recovery_curve", 50, None, COLOR["ancestor"])]
        for k, x0, lab, c in segs:
            mu, lo, hi = curve_ci(m[k])
            x = np.arange(x0, x0 + len(mu))
            ax.fill_between(x, lo, hi, color=c, alpha=0.15, linewidth=0)
            ax.plot(x, mu, color=c, lw=2, label=lab, marker="o" if k == "recovery_curve" else None, ms=4)
        for xb in (26.5, 48.5):
            ax.axvline(xb, color=AXIS, lw=0.8, ls=(0, (3, 3)))
        for x, t in ((13, "Acquisition\n(bell → food)"), (37.5, "Extinction\n(bell, no food)"), (52, "After\nrest")):
            ax.text(x, 1.0, t, ha="center", va="top", color=INK2, fontsize=8)
        ax.set_xticks([1, 25, 28, 47, 50, 54], ["1", "25", "1", "20", "1", "5"])
        ax.set_xlabel("Trial within phase")
        ax.set_ylim(-0.1, 1.05)
        ax.set_title(f"{LABEL[arm].replace(chr(10), ' ')}s (N = {len(seeds)})")
    axes[0].set_ylabel("Conditioned response\n(Δ p(salivate) before food)")
    axes[0].legend(loc="center right", bbox_to_anchor=(1.0, 0.55), fontsize=8)
    _suptitle(fig, "Fig. 1  A frozen agent learns to salivate to a new bell, then extinguishes",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_acquisition_extinction.png"), dpi=200)
    plt.close(fig)


def fig2_controls(per, seeds, out):
    """Ancestors: discrimination before vs after pairing; test CR after pairing vs the two controls."""
    A = lambda k: [per[str(a)]["ancestor"][k] for a in seeds]  # noqa: E731
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    c = COLOR["ancestor"]
    for i, (k, lab) in enumerate([("D_naive", "Before any\npairing"), ("D_acq", "After 25\npairings")]):
        strip(axes[0], i, A(k), c if k == "D_acq" else MUTED, i)
    axes[0].set_xticks([0.1, 1.1], ["Before any\npairing", "After 25\npairings"])
    axes[0].set_ylabel("Response to bell minus\nresponse to metronome")
    axes[0].set_title("Bell vs metronome (discrimination)")
    for i, (k, lab) in enumerate([("CR_test_paired", "Bell paired\nwith food"),
                                  ("CR_test_truly_random", "Bell and food\nat random"),
                                  ("CR_test_us_alone", "Food alone\n(no bell)")]):
        strip(axes[1], i, A(k), c if i == 0 else MUTED, 10 + i)
    axes[1].set_xticks([0.1, 1.1, 2.1], ["Bell paired\nwith food", "Bell and food\nat random", "Food alone\n(no bell)"])
    axes[1].set_title("Response to the bell at test, by training history")
    for ax in axes:
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_ylim(-0.15, 1.0)
    _suptitle(fig, "Fig. 2  The response needs the bell to predict food (ancestors, N = 20; dots = seeds)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_controls.png"), dpi=200)
    plt.close(fig)


def fig3_omission(per, seeds, out):
    """Omission vs yoked, per arm: one line per seed."""
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.6), sharey=True)
    for ax, arm in zip(axes, ARMS):
        om = np.array([per[str(a)][arm]["CR_omission"] for a in seeds])
        yk = np.array([per[str(a)][arm]["CR_yoked"] for a in seeds])
        for y, o in zip(yk, om):
            ax.plot([0, 1], [y, o], color=COLOR[arm], alpha=0.35, lw=1)
        for x, v in ((0, yk), (1, om)):
            m, lo, hi = mean_ci(v)
            ax.errorbar([x], [m], yerr=[[m - lo], [hi - m]], fmt="o", color=COLOR[arm], ms=7, lw=2.5,
                        markeredgecolor="#fcfcfb", markeredgewidth=1.5, zorder=3)
        us = np.mean([per[str(a)][arm]["US_delivered_omission"] for a in seeds])
        ax.text(0.5, 0.98, f"food actually received\nunder omission: {us:.0%}", transform=ax.transAxes,
                ha="center", va="top", fontsize=8, color=INK2)
        ax.set_xticks([0, 1], ["Yoked\n(food regardless)", "Omission\n(salivating\ncancels food)"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_title(LABEL[arm].replace("-\n", "-").replace("\n", " "), fontsize=9)
    axes[0].set_ylabel("Response to bell\n(last 15 training trials)")
    axes[0].set_ylim(-0.05, 0.95)
    _suptitle(fig, "Fig. 3  Salivation does not stop when it costs the food (Sheffield-style omission test)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_omission.png"), dpi=200)
    plt.close(fig)


def fig4_arms(per, seeds, out):
    """What each evolutionary history produced: CR strength, R-O learning, omission sensitivity."""
    panels = [("D_acq", "Conditioning to the bell\n(bell − metronome)", "Does a conditioned response exist?"),
              ("M_ro", "Learning about digestion\n(within-lifetime R-O score)", "Did R-O learning evolve?"),
              ("OSI", "Omission sensitivity index\n(0 = ignores, 1 = stops)", "Is the response instrumental?")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for ax, (k, ylab, title) in zip(axes, panels):
        for i, arm in enumerate(ARMS):
            v = [per[str(a)][arm][k] for a in seeds]
            if np.isfinite(v).sum():
                strip(ax, i, v, COLOR[arm], i)
            else:
                ax.text(i, 0.02, "undefined\n(no response)", ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.set_xticks(np.arange(4) + 0.1, [LABEL[a] for a in ARMS], fontsize=8)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_ylabel(ylab)
        ax.set_title(title)
    axes[2].axhline(0.2, color=INK2, lw=0.8, ls=(0, (3, 3)))
    axes[2].text(3.45, 0.21, "pre-registered\nmargin 0.2", ha="right", va="bottom", fontsize=7.5, color=INK2)
    axes[2].set_ylim(-0.05, 1.0)
    _suptitle(fig, "Fig. 4  The evolutionary environment decides whether conditioning exists, not whether it is Pavlovian",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_ecology_arms.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/confirm")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--paper", action="store_true", help="omit figure titles (for the IEEE paper)")
    args = ap.parse_args()
    global PAPER
    PAPER = args.paper
    os.makedirs(args.out, exist_ok=True)
    an = json.load(open(os.path.join(args.root, "analysis.json")))
    seeds, per = an["seeds_analysed"], an["per_seed"]
    fig1_curves(args.root, seeds, args.out)
    fig2_controls(per, seeds, args.out)
    fig3_omission(per, seeds, args.out)
    fig4_arms(per, seeds, args.out)
    split = "results/svar/channel_split.json"
    if os.path.exists(split):
        fig5_svar(json.load(open(split)), args.out)
    if all(os.path.exists(os.path.join(args.root, f"anc_{a:02d}", "phenomena.json")) for a in seeds):
        fig6_phenomena(args.root, seeds, args.out)
    print("wrote", sorted(os.listdir(args.out)))


def fig6_phenomena(root, seeds, out):
    """Phenomena battery (exploratory): per-seed effect for each phenomenon, ancestors and stable
    sisters, each panel on its own scale. Shaded side = the direction animals show."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from analyze_phenomena import effects
    E = {arm: [effects(json.load(open(os.path.join(root, run_name(arm, a), "phenomena.json")))) for a in seeds]
         for arm in ("ancestor", "stable")}
    keys = list(E["ancestor"][0])
    short = ["Blocking\nCR(X): blocked − control", "Latent inhibition\nacq. CR: pre-exposed − control",
             "Partial reinforcement\nrel. extinction: 50% − 100%", "Inhibition of delay\nslope of p over 24-tick CS",
             "Second-order\nCR(B): paired − unpaired"]
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.6))
    for ax, k, title in zip(axes, keys, short):
        sign = E["ancestor"][0][k][1]
        for i, arm in enumerate(("ancestor", "stable")):
            strip(ax, i, [e[k][0] for e in E[arm]], COLOR[arm], i)
        lim = max(abs(v) for arm in E for e in E[arm] for v in [e[k][0]]) * 1.25
        ax.set_ylim(-lim, lim)
        ax.axhspan(0, -lim if sign == "less" else lim, color=GRID, alpha=0.45, lw=0, zorder=0)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_xticks([0.1, 1.1], ["Ancestors", "Stable\nsisters"], fontsize=8)
        ax.set_title(title, fontsize=8.5)
    axes[0].set_ylabel("Effect per seed\n(shaded = animal direction)")
    _suptitle(fig, "Fig. 6  Beyond basic conditioning, the classic phenomena are absent or tiny (exploratory)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig6_phenomena.png"), dpi=200)
    plt.close(fig)


def fig5_svar(split, out):
    """S-O variability control (exploratory; channel split post hoc): innate vs learned responding
    per bell channel, as a function of how often cue meanings were reshuffled during evolution."""
    groups = [("ch0", "Light 0 (always predicts food when q = 0)", "#2a78d6"),
              ("ch1", "Light 1 (predicts food in some lives)", "#eb6834"),
              ("ch2-3", "Lights 2–3 (never predict food when q = 0)", "#1baf7a")]
    qs = ["0", "0.5", "1"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    panels = [("naive_cs_plus_cr", "Inborn: response before any pairing"),
              ("D_acq", "Learned: bell − metronome after 25 pairings")]
    for ax, (key, title) in zip(axes, panels):
        for j, (g, lab, c) in enumerate(groups):
            x = np.arange(3) + (j - 1) * 0.12
            d = [split[q][f"{key}|{g}"] for q in qs]
            m = np.array([v["mean"] for v in d])
            lo = np.array([v["ci95"][0] for v in d])
            hi = np.array([v["ci95"][1] for v in d])
            ax.plot(x, m, color=c, lw=2, alpha=0.5)
            ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="o", color=c, ms=7, lw=2, label=lab,
                        markeredgecolor="#fcfcfb", markeredgewidth=1.5, zorder=3)
        ax.set_xticks(range(3), ["q = 0\nmeanings fixed\n(N = 5)", "q = 0.5\nhalf reshuffled\n(N = 5)",
                                 "q = 1\nall reshuffled\n(N = 20)"])
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_title(title)
        ax.set_ylim(-0.12, 1.05)
    axes[0].set_ylabel("Response to the bell")
    axes[0].legend(loc="upper right", fontsize=7.5)
    _suptitle(fig, "Fig. 5  Inborn responses to signals with a fixed meaning, learning for signals whose meaning varies (exploratory)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_so_variability.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
