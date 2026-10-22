"""Draw every figure in the manuscript from saved study outputs (paper/figures/*.pdf)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, SymLogNorm

PAPER = Path(__file__).resolve().parent
PROJECT = PAPER.parent
S3 = PROJECT / "studies/03-grouping-and-circuit"
S4 = PROJECT / "studies/04-reference-and-sweeps"
OUT = PAPER / "figures"

# Validated palette (dataviz reference instance): categorical slots, ordinal blue ramp, blue-red diverging.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ORDINAL = ["#86b6ef", "#5598e7", "#256abf", "#104281"]
INK, MUTED, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list("bluered", ["#c62f2f", "#e34948", NEUTRAL, "#2a78d6", "#184f95"])

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.05})


def read(p):
    return json.loads(Path(p).read_text())


# Release: the manuscript's LaTeX build is not included, so the bibliography numbers drawn in figure
# legends are taken from the manuscript when no build is present.
MANUSCRIPT_CITATION_NUMBERS = {"zhang2026errors": "50"}


def citation_number(key):
    """Number of a bibliography entry in the last LaTeX build (numbered natbib style), for figure legends.

    review/check_paper.py checks that the number drawn in the figure still matches the build."""
    aux = PAPER / "build/main.aux"
    m = re.search(r"\\bibcite\{%s\}\{\{(\d+)\}" % re.escape(key), aux.read_text()) if aux.exists() else None
    if not m and key in MANUSCRIPT_CITATION_NUMBERS:
        return MANUSCRIPT_CITATION_NUMBERS[key]
    if not m:
        raise SystemExit(f"no citation number for {key} in {aux}; run make paper, then make figures")
    return m.group(1)


def partitions():
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 1.85))
    cmap = ListedColormap(CATEGORICAL)
    for ax, k in zip(axes, [1, 2, 4, 8]):
        grid = np.array([[(r + (s * k) // 8) % 8 for s in range(8)] for r in range(8)])
        ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=7.5)
        for r in range(8):
            for s in range(8):
                ax.text(s, r, str(grid[r, s] + 1), ha="center", va="center", fontsize=5.2, color="white")
        ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="both", length=0)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"K = {k}", color=INK)
        if k == 1:
            ax.set_ylabel("bank (rotation)")
        ax.set_xlabel("context (adjective)")
    fig.savefig(OUT / "fig-partitions.pdf")
    plt.close(fig)


def spread():
    """Bank-level gap against spread, each centred on its recipient x site x direction cell (the fitted model)."""
    r = read(S4 / "results/retrospective-study3.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.55), sharex=True, sharey=True)
    for ax, (model, title) in zip(axes, [("gpt2-small", "GPT-2 Small"), ("qwen3-0.6b", "Qwen3-0.6B")]):
        z = np.load(S3 / f"token/results/analysis-test/{model}-components.npz")
        lg = np.log10(np.abs(z["gap"][:, [1, 3, 5], :, :, :, 0]) + 1e-12)
        ls = np.log10(z["dispersion"][:, [1, 3, 5], :, :, :, 0])
        lg = lg - lg.mean(axis=(3, 4), keepdims=True)
        ls = ls - ls.mean(axis=(3, 4), keepdims=True)
        for ki, k in enumerate([1, 2, 4, 8]):
            x, y = ls[:, :, :, ki].ravel(), lg[:, :, :, ki].ravel()
            ax.scatter(x, y, s=2.0, color=ORDINAL[ki], alpha=0.2, linewidths=0, rasterized=True)
            ax.scatter([np.median(x)], [np.median(y)], s=36, color=ORDINAL[ki], edgecolor="white", linewidth=1.2,
                       zorder=4, label=f"K = {k}")
        slope = r["grouping"][model]["loglog_slope_gap_on_spread"]
        xs = np.linspace(-2.2, 0.8, 20)
        ax.plot(xs, slope * xs, color=INK, linewidth=1.0, linestyle="--", zorder=3, label="fitted slope")
        ax.text(0.04, 0.95, f"fitted slope {slope:.2f}", transform=ax.transAxes, va="top", color=INK)
        ax.plot(xs, xs, color=MUTED, linewidth=0.8, linestyle=":", zorder=3, label="slope 1")
        ax.set_title(title, color=INK)
        ax.set_xlabel("log$_{10}$ spread, relative to cell mean")
        ax.grid(True, color=GRID, linewidth=0.5)
        ax.set_xlim(-2.3, 0.9)
        ax.set_ylim(-3.2, 1.6)
    axes[0].set_ylabel("log$_{10}$ |gap|, relative to cell mean")
    # Upper left, below the slope label: the lower right holds part of the K = 8 cloud.
    axes[1].legend(loc="upper left", bbox_to_anchor=(0.0, 0.9), frameon=False, handletextpad=0.3, borderaxespad=0.2, ncol=2,
                   columnspacing=0.8)
    fig.savefig(OUT / "fig-spread.pdf", dpi=300)
    plt.close(fig)


def _heat(ax, m, vmax, positions, layers, title, marks=None, log=False):
    if log:
        im = ax.imshow(m, cmap=DIVERGING, norm=SymLogNorm(linthresh=0.01, vmin=-vmax, vmax=vmax, base=10),
                       aspect="auto", origin="lower", interpolation="nearest")
    else:
        im = ax.imshow(m, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels(positions, rotation=60, ha="right")
    step = 1 if layers <= 14 else 3
    ax.set_yticks(range(0, layers, step))
    ax.set_yticklabels([str(i) for i in range(0, layers, step)])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, color=INK)
    if marks is not None:
        fp, fn = marks
        for (l, p) in fp:
            ax.scatter(p, l, s=16, facecolor="none", edgecolor=INK, linewidth=0.9)
        for (l, p) in fn:
            ax.scatter(p, l, s=14, marker="x", color=INK, linewidth=0.9)
    return im


def sweeps(stage="test"):
    a = read(S4 / f"results/sweep/analysis-{stage}.json")
    pretty = {"adj": "adjective", "anchor+0": "head", "anchor+1": "+1", "anchor+2": "+2", "anchor+3": "+3", "END": "final"}
    rows = [(k, n) for k, n in [("sva-gpt2-small", "GPT-2 Small"), ("sva-qwen3-0.6b", "Qwen3-0.6B")] if k in a["panels"]]
    if rows:
        bias_max = 0.0
        for panel, _ in rows:
            p = a["panels"][panel]
            b = np.abs(np.array(p["maps"]["mean64"]) - np.array(p["maps"]["individual"]))
            bias_max = max(bias_max, float(np.percentile(b, 99)))
        vb = max(0.05, np.ceil(bias_max * 20) / 20)
        fig, axes = plt.subplots(len(rows), 4, figsize=(6.8, 2.1 * len(rows) + 0.5), squeeze=False,
                                 gridspec_kw={"wspace": 0.12, "hspace": 0.5})
        titles = ["restore: effect", "destroy: effect", "restore: bias", "destroy: bias"]
        for ri, (panel, name) in enumerate(rows):
            p = a["panels"][panel]
            ind, m64 = np.array(p["maps"]["individual"]), np.array(p["maps"]["mean64"])
            flags = p["flags"]["mean64/tau0.05"]
            positions = [pretty.get(x, x) for x in p["positions"]]
            for di in range(2):
                fp = [(l, q) for d, l, q in flags["false_positive_list"] if d == di]
                fn = [(l, q) for d, l, q in flags["false_negative_list"] if d == di]
                im1 = _heat(axes[ri, di], ind[di], 1.0, positions, ind.shape[1], titles[di] if ri == 0 else "")
                im2 = _heat(axes[ri, 2 + di], m64[di] - ind[di], 1.0, positions, ind.shape[1], titles[2 + di] if ri == 0 else "",
                            marks=(fp, fn), log=True)
            for ci in range(1, 4):
                axes[ri, ci].set_yticklabels([])
            axes[ri, 0].set_ylabel(f"{name}\nlayer")
        fig.canvas.draw()
        for cols, im, label in [((0, 1), im1, "normalised effect of individual patches"),
                                ((2, 3), im2, "bias: mean ablation minus individual average")]:
            left = axes[-1, cols[0]].get_position().x0
            right = axes[-1, cols[1]].get_position().x1
            inset = 0.03 * (right - left)
            cax = fig.add_axes([left + inset, 0.02, right - left - 2 * inset, 0.015])
            cb = fig.colorbar(im, cax=cax, orientation="horizontal")
            cb.set_label(label)
            cb.outline.set_linewidth(0.4)
        fig.subplots_adjust(bottom=0.2)
        fig.savefig(OUT / "fig-sweeps-sva.pdf")
        plt.close(fig)
    if "ioi-gpt2-small" in a["panels"]:
        p = a["panels"]["ioi-gpt2-small"]
        ind, m64 = np.array(p["maps"]["individual"])[0], np.array(p["maps"]["mean64"])[0]
        flags = p["flags"]["mean64/tau0.05"]
        fp = [(l, q) for d, l, q in flags["false_positive_list"]]
        fn = [(l, q) for d, l, q in flags["false_negative_list"]]
        vb = max(0.05, np.ceil(float(np.percentile(np.abs(m64 - ind), 99)) * 20) / 20)
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), gridspec_kw={"wspace": 0.3})
        im1 = _heat(axes[0], ind, 1.5, p["positions"], ind.shape[0], "effect of individual patches")
        im2 = _heat(axes[1], m64 - ind, 1.0, p["positions"], ind.shape[0], "bias: mean ablation $-$ individual", marks=(fp, fn), log=True)
        axes[0].set_ylabel("layer")
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.03).set_label("normalised effect")
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.03).set_label("difference")
        fig.savefig(OUT / "fig-sweeps-ioi.pdf")
        plt.close(fig)


def weak_sites(stage="test"):
    a = read(S4 / f"results/sweep/analysis-{stage}.json")
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    names = {"sva-gpt2-small": "agreement, GPT-2 Small", "sva-qwen3-0.6b": "agreement, Qwen3-0.6B", "ioi-gpt2-small": "IOI, GPT-2 Small"}
    for ci, (panel, label) in enumerate(names.items()):
        if panel not in a["panels"]:
            continue
        p = a["panels"][panel]
        import sys
        sys.path.insert(0, str(S4))
        from analyze_sweep import load_panel
        live = ~load_panel(panel, stage)["structural"]
        ind = np.abs(np.array(p["maps"]["individual"]))[live]
        bias = np.abs(np.array(p["maps"]["mean64"]) - np.array(p["maps"]["individual"]))[live]
        keep = (ind > 1e-4) & (bias > 1e-5)                      # log axes; tiny values sit off the plot
        ax.scatter(ind[keep], bias[keep], s=7, color=[BLUE, ORANGE, AQUA][ci], alpha=0.75, linewidths=0, label=label)
    xs = np.geomspace(1e-4, 3, 10)
    ax.plot(xs, xs, color=MUTED, linestyle=":", linewidth=0.8)
    ax.axvline(0.05, color=MUTED, linewidth=0.6, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-4, 3)
    ax.set_ylim(1e-5, 1)
    ax.set_xlabel("|effect of individual patches| (normalised)")
    ax.set_ylabel("|mean ablation minus individual|")
    ax.grid(True, color=GRID, linewidth=0.5)
    # Above the axes, clear of the points and of the dotted line (bias = effect, named in the caption).
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, frameon=False, handletextpad=0.2, columnspacing=0.8,
              borderaxespad=0.0)
    fig.savefig(OUT / "fig-weak-sites.pdf")
    plt.close(fig)


def agreement(stage="test"):
    """Agreement with the model under four references: circuit, random circuits (post hoc) and no heads kept."""
    a = read(S4 / f"results/reference/analysis-{stage}.json")
    rc_path = S4 / "results/posthoc/random-circuits.json"
    rc = read(rc_path) if rc_path.exists() else None
    cells = ["N8", "NPO8", "N128", "NPO128"]
    labels = ["names only\n8 donors", "names, place, object\n8 donors", "names only\n128 donors", "names, place, object\n128 donors"]
    rows = [("error", "agreement with model errors (%)")] + ([("correct", "agreement on correct cases (%)")] if rc else [])
    zhang_label = f"reported by Zhang et al. [{citation_number('zhang2026errors')}]"
    fig, axes = plt.subplots(len(rows), 2, figsize=(6.8, 2.6 * len(rows)), sharey="row", squeeze=False,
                             gridspec_kw={"wspace": 0.08, "hspace": 0.55})
    x = np.arange(4)
    width = 0.25 if rc else 0.36
    for ri, (stratum, ylabel) in enumerate(rows):
        t = a["tables"][f"pooled/{stratum}"]
        for ax, pool, short, title, zhang in [(axes[ri, 0], "mean_agree", "mean", "Mean ablation", (15.2, 99.5)),
                                              (axes[ri, 1], "resample_agree", "resample", "Resample ablation", (40.6, 91.2))]:
            series = [(-width if rc else -0.19, "reference", BLUE, "26-head circuit"),
                      (width if rc else 0.19, "empty", "#a9a8a4", "no heads kept")]
            for off, mode, color, name in series:
                est = np.array([100 * t[f"{c}/{mode}/{pool}"]["estimate"] for c in cells])
                lo = np.array([100 * t[f"{c}/{mode}/{pool}"]["ci"][0] for c in cells])
                hi = np.array([100 * t[f"{c}/{mode}/{pool}"]["ci"][1] for c in cells])
                ax.bar(x + off, est, width=width, color=color, label=name, zorder=2)
                ax.errorbar(x + off, est, yerr=[est - lo, hi - est], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2, zorder=3)
            if rc:
                rt = rc["tables"][f"pair_{stratum}"]
                est = np.array([100 * rt[f"{c}/{short}"]["mean_over_circuits"] for c in cells])
                lo = np.array([100 * rt[f"{c}/{short}"]["min"] for c in cells])
                hi = np.array([100 * rt[f"{c}/{short}"]["max"] for c in cells])
                ax.bar(x, est, width=width, color=ORANGE, label="random 26-head circuits (mean, range)", zorder=2)
                ax.errorbar(x, est, yerr=[est - lo, hi - est], fmt="none", ecolor=INK, elinewidth=0.8, capsize=0, zorder=3)
            ax.scatter([3.0 + (-width if rc else -0.19)], [zhang[ri]], marker="D", s=24, color=INK, edgecolor="white",
                       linewidth=0.8, zorder=4, label=zhang_label)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=6.6)
            ax.set_title(title, color=INK)
            ax.set_ylim(0, 100)
            ax.grid(True, axis="y", color=GRID, linewidth=0.5, zorder=0)
            ax.tick_params(axis="x", length=0)
        axes[ri, 0].set_ylabel(ylabel)
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    wanted = ["26-head circuit", "random 26-head circuits (mean, range)", "no heads kept", zhang_label]
    wanted = [w for w in wanted if w in by_label]
    fig.legend([by_label[w] for w in wanted], wanted, loc="upper center", ncol=len(wanted) if not rc else 4,
               frameon=False, fontsize=7.2, bbox_to_anchor=(0.5, 1.03 if rc else 1.06), columnspacing=1.2, handlelength=1.5)
    fig.savefig(OUT / "fig-agreement.pdf")
    plt.close(fig)


def threshold():
    """Misclassified sites (mean ablation vs individual average) across thresholds, post hoc."""
    d = read(S4 / "results/posthoc/sweep-posthoc.json")
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    names = {"sva-gpt2-small": "agreement, GPT-2 Small", "sva-qwen3-0.6b": "agreement, Qwen3-0.6B", "ioi-gpt2-small": "IOI, GPT-2 Small"}
    for ci, (panel, label) in enumerate(names.items()):
        bt = d["panels"][panel]["misclassified_by_tau"]
        taus = sorted(float(t) for t in bt)
        ys = [bt[str(t) if str(t) in bt else f"{t:g}"] for t in taus]
        ax.plot(taus, ys, color=[BLUE, ORANGE, AQUA][ci], linewidth=2, marker="o", markersize=3.5, label=label)
    ax.axvline(0.05, color=MUTED, linestyle="--", linewidth=0.7)
    ax.set_xlabel("threshold $\\tau$ on |normalised effect|")
    ax.set_ylabel("misclassified sites")
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(OUT / "fig-threshold.pdf")
    plt.close(fig)


S5 = PROJECT / "studies/05-head-confirmation"


def heads(stage="test"):
    """Study 3: every head-site's individual-average effect and the mean-ablation bias, with flag changes marked."""
    a = read(S5 / f"results/analysis-{stage}.json")["settings"]
    panels = [("sva-end-gpt2-small", 0, "agreement, restore"), ("sva-end-gpt2-small", 1, "agreement, destroy"),
              ("ioi-end-gpt2-small", 0, "IOI, final token"), ("ioi-s2-gpt2-small", 0, "IOI, repeated subject")]
    vmax = max(float(np.abs(np.array(a[s]["maps"]["IND"])).max()) for s, _, _ in panels)
    vmax = float(np.ceil(vmax * 10) / 10)
    fig, axes = plt.subplots(2, 4, figsize=(6.8, 3.9), gridspec_kw={"wspace": 0.12, "hspace": 0.32})
    for ci, (setting, d, title) in enumerate(panels):
        ind = np.array(a[setting]["maps"]["IND"])[d]
        mean = np.array(a[setting]["maps"]["MEAN"])[d]
        fi, fm = np.abs(ind) >= 0.05, np.abs(mean) >= 0.05
        for ri, (m, log) in enumerate([(ind, False), (mean - ind, True)]):
            ax = axes[ri, ci]
            if log:
                im = ax.imshow(m, cmap=DIVERGING, norm=SymLogNorm(linthresh=0.001, vmin=-0.1, vmax=0.1, base=10),
                               aspect="auto", origin="lower", interpolation="nearest")
            else:
                im = ax.imshow(m, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto", origin="lower", interpolation="nearest")
            ax.set_xticks(range(0, 12, 2))
            ax.set_yticks(range(0, 12, 2))
            ax.tick_params(length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if ri == 0:
                ax.set_title(title, color=INK)
                for l, h in np.argwhere(fi):
                    ax.scatter(h, l, s=5, color=INK, linewidth=0)
                im_top = im
            else:
                for l, h in np.argwhere(fm & ~fi):
                    ax.scatter(h, l, s=16, facecolor="none", edgecolor=INK, linewidth=0.9)
                for l, h in np.argwhere(fi & ~fm):
                    ax.scatter(h, l, s=14, marker="x", color=INK, linewidth=0.9)
                ax.set_xlabel("head")
                im_bot = im
            if ci:
                ax.set_yticklabels([])
        axes[0, 0].set_ylabel("layer")
        axes[1, 0].set_ylabel("layer")
    fig.canvas.draw()
    for row, im, label in [(0, im_top, "effect, average of individual patches"), (1, im_bot, "bias: mean ablation minus individual average")]:
        pos = axes[row, 3].get_position()
        cax = fig.add_axes([pos.x1 + 0.012, pos.y0, 0.012, pos.y1 - pos.y0])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(label, fontsize=6.5)
        cb.ax.tick_params(labelsize=6)
        cb.outline.set_linewidth(0.4)
    fig.savefig(OUT / "fig-heads.pdf")
    plt.close(fig)


def discrimination():
    """Post hoc: odds ratio and AUROC for separating model errors from correct prompts, per cell."""
    r = read(S4 / "results/posthoc/error-discrimination.json")["results"]
    cells = [(p, c) for p in ["mean", "resample"] for c in ["N8", "N128", "NPO8", "NPO128"]]
    labels = [f"{c}" for _, c in cells]
    x = np.arange(len(cells))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), gridspec_kw={"wspace": 0.28})
    for ax, metric, ylab, chance, log in [(axes[0], "odds_ratio", "odds ratio (log scale)", 1.0, True),
                                          (axes[1], "auroc", "AUROC", 0.5, False)]:
        for off, unit, color, name in [(-0.18, "circuit", BLUE, "IOI circuit"), (0.18, "empty", "#8f8e8a", "no heads kept")]:
            est = np.array([r[f"{unit}/{p}/{c}"][metric] for p, c in cells])
            lo = np.array([r[f"{unit}/{p}/{c}"][f"{metric}_ci95"][0] for p, c in cells])
            hi = np.array([r[f"{unit}/{p}/{c}"][f"{metric}_ci95"][1] for p, c in cells])
            ax.errorbar(x + off, est, yerr=[est - lo, hi - est], fmt="o", color=color, markersize=4, elinewidth=0.9,
                        capsize=2, label=name, zorder=3)
        for k in range(5):
            vals = [r[f"random{k}/{p}/{c}"][metric] for p, c in cells]
            ax.scatter(x, vals, s=10, color=ORANGE, alpha=0.8, linewidths=0, zorder=2,
                       label="random circuits" if k == 0 else None)
        ax.axhline(chance, color=MUTED, linestyle=":", linewidth=0.8)
        if log:
            ax.set_yscale("log")
        ax.axvline(3.5, color=GRID, linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.4, rotation=35, ha="right")
        ax.set_ylabel(ylab)
        ax.grid(True, axis="y", color=GRID, linewidth=0.5)
        ax.text(1.5, 1.02, "mean ablation", transform=ax.get_xaxis_transform(), ha="center", fontsize=7, color=MUTED)
        ax.text(5.5, 1.02, "resampling", transform=ax.get_xaxis_transform(), ha="center", fontsize=7, color=MUTED)
    axes[1].set_ylim(0.45, 1.0)
    handles, labels_ = axes[0].get_legend_handles_labels()
    by = dict(zip(labels_, handles))
    order = ["IOI circuit", "random circuits", "no heads kept"]
    fig.legend([by[k] for k in order], order, loc="upper center", ncol=3, frameon=False, fontsize=7.2, bbox_to_anchor=(0.5, 1.1))
    fig.savefig(OUT / "fig-discrimination.pdf")
    plt.close(fig)


def head_spread():
    """Post hoc: Study 3 head-level pooling gaps on the gap-against-spread axes, and head against residual spread."""
    z = np.load(S5 / "results/posthoc/head-spread.npz")
    s1 = np.load(S3 / "token/results/analysis-test/gpt2-small-components.npz")
    # [G, site, dir, K, rot]; np.take keeps the axis order (a[:, [1, 3, 5], ..., 0] would move the site axis first)
    gap1 = np.abs(np.take(s1["gap"], [1, 3, 5], axis=1)[..., 0])
    sp1 = np.take(s1["dispersion"], [1, 3, 5], axis=1)[..., 0]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), gridspec_kw={"wspace": 0.3, "width_ratios": [1.15, 1]})
    ax = axes[0]
    names = {"sva-end": ("agreement, final token", BLUE), "ioi-end": ("IOI, final token", ORANGE),
             "ioi-s2": ("IOI, repeated subject", AQUA)}
    for key, (label, color) in names.items():
        g = np.median(np.abs(z[f"{key}__gap"]), 0).ravel()
        h = np.median(z[f"{key}__head_spread"], 0).ravel()
        ok = g > 0
        ax.scatter(h[ok], g[ok], s=4, color=color, alpha=0.55, linewidths=0, label=f"heads: {label}", rasterized=True)
    gm = np.median(gap1, axis=(0, 4)).ravel()
    sm = np.median(sp1, axis=(0, 4)).ravel()
    ax.scatter(sm, gm, s=14, marker="s", facecolor="none", edgecolor=INK, linewidth=0.7, zorder=4,
               label="residual banks, Study 1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("donor spread (mean squared radius)")
    ax.set_ylabel("|pooling gap| (logits)")
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.legend(frameon=False, fontsize=6.4, loc="upper left", handletextpad=0.2, markerscale=1.8)
    ax = axes[1]
    for i, (key, (label, color)) in enumerate(names.items()):
        ratio = np.median(z[f"{key}__head_spread"] / z[f"{key}__resid_spread"][..., None], 0)    # [D, L, H]
        for layer in range(12):
            vals = np.log10(ratio[:, layer, :].ravel())
            ax.scatter(np.full(vals.shape, layer + (i - 1) * 0.25), vals, s=3, color=color, alpha=0.5, linewidths=0)
            ax.scatter([layer + (i - 1) * 0.25], [np.median(vals)], s=14, color=color, edgecolor="white", linewidth=0.5,
                       zorder=3)
    ax.set_xticks(range(12))
    ax.set_xlabel("layer")
    ax.set_ylabel("log$_{10}$(head spread / residual spread)")
    ax.grid(True, axis="y", color=GRID, linewidth=0.5)
    fig.savefig(OUT / "fig-head-spread.pdf", dpi=300)
    plt.close(fig)


S6 = PROJECT / "studies/06-component-confirmation"


def components(stage="test"):
    """Study 4: centroid bias against donor spread for every site, and bias by layer for each kind of component."""
    a = read(S6 / f"results/analysis-{stage}.json")["kinds"]
    kinds = [("resid", "residual stream", BLUE), ("mlp", "MLP output", ORANGE), ("attn", "attention layer", AQUA),
             ("head", "single head", "#e87ba4")]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), gridspec_kw={"wspace": 0.3, "width_ratios": [1.1, 1]})
    for kind, label, color in kinds:
        ind, mean = np.array(a[kind]["maps"]["IND"]), np.array(a[kind]["maps"]["MEAN"])
        bias, spread = np.abs(mean - ind).ravel(), np.array(a[kind]["spread_map"]).ravel()
        dis = ((np.abs(mean) >= 0.05) != (np.abs(ind) >= 0.05)).ravel()
        ok = bias > 0
        axes[0].scatter(spread[ok], bias[ok], s=6 if kind == "head" else 14, color=color, alpha=0.6 if kind == "head" else 0.85,
                        linewidths=0, label=label, rasterized=True)
        if dis.any():
            axes[0].scatter(spread[dis], bias[dis], s=30, facecolor="none", edgecolor=INK, linewidth=0.8, zorder=4)
        b = np.abs(mean - ind)
        per_layer = b.reshape(2, 12, -1).mean(axis=(0, 2)) if kind == "head" else b.mean(0)
        axes[1].plot(range(12), per_layer, color=color, linewidth=2, marker="o", markersize=3.5, label=label)
    axes[0].axhline(0.05, color=MUTED, linestyle="--", linewidth=0.7)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("donor spread (mean squared radius)")
    axes[0].set_ylabel("|mean ablation $-$ individual average|")
    axes[0].grid(True, color=GRID, linewidth=0.5)
    axes[0].legend(frameon=False, fontsize=6.6, loc="lower right", handletextpad=0.2, markerscale=1.4)  # clear of the tau line
    axes[1].axhline(0.05, color=MUTED, linestyle="--", linewidth=0.7)
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(12))
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("mean |bias| per layer")
    axes[1].grid(True, color=GRID, linewidth=0.5)
    fig.savefig(OUT / "fig-components.pdf", dpi=300)
    plt.close(fig)


def main(stage="test"):
    OUT.mkdir(exist_ok=True)
    partitions()
    spread()
    if (S4 / f"results/sweep/analysis-{stage}.json").exists():
        sweeps(stage)
        weak_sites(stage)
    if (S4 / f"results/reference/analysis-{stage}.json").exists():
        agreement(stage)
    if (S4 / "results/posthoc/sweep-posthoc.json").exists():
        threshold()
    if (S5 / f"results/analysis-{stage}.json").exists():
        heads(stage)
    if (S4 / "results/posthoc/error-discrimination.json").exists():
        discrimination()
    if (S5 / "results/posthoc/head-spread.npz").exists():
        head_spread()
    if (S6 / f"results/analysis-{stage}.json").exists():
        components(stage)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
