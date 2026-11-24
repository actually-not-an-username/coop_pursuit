"""Analysis of the blackout sweep: figures + tables.

Usage:  python scripts/analyze_sweep.py [--runs runs] [--out runs/analysis] [--selection best]

Statistics: the unit of replication is the training seed (n=5 per method).
Per-seed capture rates are computed over 300 test episodes, then mean and
95% t-CI are taken across seeds. Within-method contrasts (drop vs. no-blackout,
hold vs. zero) are paired by seed. Clean-vs-masked differences use a Welch
t-interval (independent training runs). The random baseline is a single
policy, so its CI is a Wilson binomial interval over episodes.
"""
import argparse
import glob
import os
import textwrap
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------- style ----
C = {"clean": "#2a78d6", "masked": "#eb6834", "random": "#898781"}
LABEL = {"clean": "Full observation", "masked": "Observation masked", "random": "Random policy"}
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
METHODS = ["clean", "masked"]
LENGTHS = ["none", "5", "10", "20", "40", "permanent"]
LEN_TICK = ["none", "5", "10", "20", "40", "perm."]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 1,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 7,
})


def tci(x, conf=0.95):
    """mean, half-width of t-CI across seeds."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (x.mean() if len(x) else np.nan), np.nan
    return x.mean(), stats.t.ppf((1 + conf) / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def welch(a, b, conf=0.95):
    """diff (b - a) and half-width of Welch CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return b.mean() - a.mean(), stats.t.ppf((1 + conf) / 2, df) * np.sqrt(va + vb)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return p, c - h, c + h


def fmt(m, h, pct=True):
    s = 100 if pct else 1
    p = 1 if (pct or abs(m) >= 10) else 2
    if np.isnan(h):
        return f"{m*s:.{p}f}"
    return f"{m*s:.{p}f} ± {h*s:.{p}f}"


def declutter(ys, gap):
    """Return label y-positions with at least `gap` between them, order preserved."""
    ys = np.asarray(ys, float)
    order = np.argsort(ys)
    pos = ys[order].copy()
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    pos -= (pos.mean() - ys[order].mean())  # recentre on the data
    out = np.empty_like(pos)
    out[order] = pos
    return out


def end_labels(ax, x, ys, texts, gap):
    for y, ty, t in zip(ys, declutter(ys, gap), texts):
        ax.annotate(t, (x, y), xytext=(x + 0.12, ty), textcoords="data", va="center",
                    color=INK2, fontsize=9)


# Project root (folder above scripts/), so default paths work from any working directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------- subset mode ----
KEEP = None          # {method: set(seeds)} when --seeds is given, else all runs
NOTE = ""            # footnote added to every figure (used to disclose a selected subset)
RUNS_LABEL = "5 seeds"


def parse_seeds(spec):
    """'clean:1,2,5;masked:2,4,5' -> {'clean': {1,2,5}, 'masked': {2,4,5}}"""
    out = {}
    for part in spec.split(";"):
        m, seeds = part.split(":")
        out[m.strip()] = {int(x) for x in seeds.split(",")}
    return out


def kept(method, seed):
    return KEEP is None or method not in KEEP or int(seed) in KEEP[method]


# ----------------------------------------------------------------- load ----
def load(runs):
    files = glob.glob(os.path.join(runs, "sweep", "*.csv"))
    if not files:
        raise SystemExit(f"No sweep results found in {os.path.abspath(os.path.join(runs, 'sweep'))}. "
                         "Run the sweep first, or pass --runs <folder containing sweep/>.")
    d = pd.concat([pd.read_csv(f, dtype={"blackout_length": str}) for f in files], ignore_index=True)
    d["captured"] = d["captured"].astype(str).str.lower().eq("true")
    d["cbb"] = d["captured_before_blackout"].astype(str).str.lower().eq("true")
    d["fill"] = d["fill"].fillna("none")
    d.loc[d.method == "random", "selection"] = "random"
    d["training_seed"] = d["training_seed"].fillna(0).astype(int)
    if KEEP is not None:
        d = d[[kept(m, s) for m, s in zip(d.method, d.training_seed)]].reset_index(drop=True)
    return d


def cond_key(length, nb, fill):
    return (length, nb, fill if length != "none" else "none")


def per_seed(d, metric="captured"):
    """per (method, selection, seed, length, nb, fill) mean of metric."""
    g = d.groupby(["method", "selection", "training_seed", "blackout_length", "num_blinded", "fill"])[metric]
    return g.mean().rename(metric).reset_index()


def seed_vals(ps, method, sel, length, nb, fill, metric="captured", drop_seeds=()):
    length, nb, fill = cond_key(length, nb if length != "none" else 0, fill)
    q = ps[(ps.method == method) & (ps.selection == sel) & (ps.blackout_length == length)
           & (ps.num_blinded == nb) & (ps.fill == fill) & (~ps.training_seed.isin(drop_seeds))]
    return q.sort_values("training_seed").set_index("training_seed")[metric]


# -------------------------------------------------------------- figures ----
def save(fig, out, name):
    if NOTE:
        wrapped = textwrap.fill(NOTE, width=int(fig.get_figwidth() * 17))
        fig.text(0.01, -0.06 if name.startswith("fig1") else -0.02, wrapped, ha="left", va="top",
                 fontsize=8.5, color=INK2, style="italic")
    # Write to a temp file, then swap it in. Retry because image viewers on Windows
    # (e.g. Photos) briefly lock a file they are displaying when it changes.
    for ext, kw in ((".png", {"dpi": 200}), (".pdf", {})):
        final = os.path.join(out, name + ext)
        tmp = os.path.join(out, f".tmp_{name}{ext}")
        fig.savefig(tmp, bbox_inches="tight", format=ext[1:], **kw)
        for attempt in range(50):
            try:
                os.replace(tmp, final)
                break
            except OSError:
                if attempt == 49:
                    raise
                time.sleep(0.1)
    plt.close(fig)


def caption(fig, text, bottom=0.08):
    """Lay out the figure leaving room for a small caption line under the axes."""
    fig.tight_layout(rect=(0, bottom, 1, 1))
    # fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=8.5, color=INK2)


# Fig 1 / Table B1 stop at the last regular validation point (every 96 episodes).
# Episode 2048 is the off-grid end-of-base-training evaluation. Set to None to show it.
FIG1_MAX_EPISODE = 2016


def curve(lc, col):
    c = lc.set_index("episode")[col]
    return c if FIG1_MAX_EPISODE is None else c[c.index <= FIG1_MAX_EPISODE]


def fig_learning(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, col, title in zip(axes, ["val_capture_noloss", "val_capture_blackout"],
                              ["Validation: no blackout", "Validation: hard blackout"]):
        for m in METHODS:
            curves = []
            for f in sorted(glob.glob(os.path.join(runs, f"{m}_n3_*_s*", "learning_curve.csv"))):
                if not kept(m, os.path.basename(os.path.dirname(f)).rsplit("_s", 1)[1]):
                    continue
                lc = pd.read_csv(f)
                curves.append(curve(lc, col))
            df = pd.concat(curves, axis=1).dropna()
            x = df.index.values
            mu = df.mean(1).values * 100
            hw = np.array([tci(r.values)[1] for _, r in df.iterrows()]) * 100
            ax.fill_between(x, mu - hw, mu + hw, color=C[m], alpha=0.15, lw=0)
            ax.plot(x, mu, color=C[m], marker="o", mec=SURF, mew=1.5, label=LABEL[m])
        ax.set_title(title, loc="left")
        ax.set_xlabel("Training episode")
    axes[0].set_ylabel("Capture rate (%)")
    axes[0].set_ylim(0, 100)
    axes[0].legend(loc="lower right")
    fig.suptitle("Fig 1: Learning curve after validation", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    # fig.tight_layout(rect=(0, 0.09, 1, 1))
    # fig.text(0.01, 0.01, "Each point is validation on 100 episodes (same episodes every time); band = 95% CI across seeds.\n"
    #          "Validation every 96 episodes; curves end at 2016, the last regular point before 2048.",
    #          ha="left", va="bottom", fontsize=8.5, color=INK2)
    save(fig, out, "fig1_learning_curves")


def rand_curve(d, length, nb, fill):
    length, nb, fill = cond_key(length, nb if length != "none" else 0, fill)
    r = d[(d.method == "random") & (d.blackout_length == length) & (d.num_blinded == nb) & (d.fill == fill)]
    return wilson(r.captured.sum(), len(r))


def fig_capture_vs_length(d, ps, sel, out, fill="zero", name="fig2_capture_vs_length", drop=False):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    x = np.arange(len(LENGTHS))
    for ax, nb in zip(axes, [1, 2]):
        ends = []
        for i, m in enumerate(METHODS):
            mu, hw = zip(*[tci(seed_vals(ps, m, sel, L, nb, fill)) for L in LENGTHS])
            mu, hw = np.array(mu) * 100, np.array(hw) * 100
            off = (i - 0.5) * 0.08
            ax.errorbar(x + off, mu, yerr=hw, color=C[m], capsize=3, elinewidth=1.2, label=LABEL[m])
            ends.append(mu[-1])
        end_labels(ax, x[-1], ends, [f"{v:.0f}%" for v in ends], gap=5)
        rr = np.array([rand_curve(d, L, nb, fill) for L in LENGTHS]) * 100
        ax.plot(x, rr[:, 0], color=C["random"], ls="--", lw=1.5, marker="", label=LABEL["random"])
        ax.set_xticks(x, LEN_TICK)
        ax.set_xlabel("Blackout length in steps")
        ax.set_title(f"{nb} pursuer{'s' if nb > 1 else ''} blinded", loc="left")
        ax.set_xlim(-0.3, len(LENGTHS) - 0.4)
    axes[0].set_ylabel("Capture rate (%)")
    axes[0].set_ylim(0, 100)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 0.12))  # v2: random line at ~7% would sit under the legend
    fig.suptitle("Fig 2: Capture rate vs blackout rate", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    # caption(fig, f"{fill.capitalize()} fill, '{sel}' checkpoint; mean ± 95% CI over {RUNS_LABEL}.")
    save(fig, out, name)


def fig_drop(ps, sel, out, fill="zero"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    Ls = LENGTHS[1:]
    x = np.arange(len(Ls))
    for ax, nb in zip(axes, [1, 2]):
        ax.axhline(0, color=AXIS, lw=1)
        ends = []
        for i, m in enumerate(METHODS):
            base = seed_vals(ps, m, sel, "none", 0, "none")
            res = [tci(seed_vals(ps, m, sel, L, nb, fill) - base) for L in Ls]
            mu, hw = np.array([r[0] for r in res]) * 100, np.array([r[1] for r in res]) * 100
            off = (i - 0.5) * 0.08
            ax.errorbar(x + off, mu, yerr=hw, color=C[m], marker="o", capsize=3, elinewidth=1.2,
                        label=LABEL[m], mec=SURF, mew=1.5)
            ends.append(mu[-1])
        end_labels(ax, x[-1], ends, [f"{v:+.0f}" for v in ends], gap=2)
        ax.set_xticks(x, LEN_TICK[1:])
        ax.set_xlabel("Blackout length in steps")
        ax.set_title(f"{nb} pursuer{'s' if nb > 1 else ''} blinded", loc="left")
        ax.set_xlim(-0.3, len(Ls) - 0.4)
    axes[0].set_ylabel("Change in capture rate\n between impared pursuers and no blackout (pts)")
    axes[0].legend(loc="lower left")
    fig.suptitle("Fig 3: Capture rate difference", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    # caption(fig, f"Change from each method's own no-blackout capture rate. {fill.capitalize()} fill, '{sel}' checkpoint; "
    #              f"mean ± 95% CI, paired by seed ({RUNS_LABEL}).")
    save(fig, out, "fig3_drop_vs_length")


def fig_fill(ps, sel, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    Ls = LENGTHS[1:]
    x = np.arange(len(Ls))
    for ax, nb in zip(axes, [1, 2]):
        ax.axhline(0, color=AXIS, lw=1)
        for i, m in enumerate(METHODS):
            res = [tci(seed_vals(ps, m, sel, L, nb, "hold") - seed_vals(ps, m, sel, L, nb, "zero")) for L in Ls]
            mu, hw = np.array([r[0] for r in res]) * 100, np.array([r[1] for r in res]) * 100
            off = (i - 0.5) * 0.12
            ax.errorbar(x + off, mu, yerr=hw, color=C[m], marker="o", ls="none", capsize=3,
                        elinewidth=1.2, label=LABEL[m], mec=SURF, mew=1.5)
        ax.set_xticks(x, LEN_TICK[1:])
        ax.set_xlabel("Blackout length in steps")
        ax.set_title(f"{nb} pursuer{'s' if nb > 1 else ''} blinded", loc="left")
    axes[0].set_ylabel("Hold − zero fill\ncapture rate (pts)")
    axes[0].legend(loc="lower left")
    fig.suptitle("Fig 5: Effect of fill strategy: hold last observation vs. zero (above 0 = hold better)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig5_fill_hold_vs_zero")


def survival(eps, T=100):
    t = np.arange(0, T + 1)
    cs = eps.capture_step.where(eps.captured, np.inf).values
    return t, np.array([(cs > s).mean() for s in t])


def fig_survival(d, sel, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, m in zip(axes, METHODS):
        for (L, nb, fill), ls, lab in [(("none", 0, "none"), "-", "No blackout"),
                                       (("permanent", 2, "zero"), "--", "Permanent, 2 blinded (zero)")]:
            q = d[(d.method == m) & (d.selection == sel) & (d.blackout_length == L)
                  & (d.num_blinded == nb) & (d.fill == fill)]
            curves = []
            for s, e in q.groupby("training_seed"):
                t, S = survival(e)
                curves.append(S)
            curves = np.array(curves) * 100
            ax.fill_between(t, curves.min(0), curves.max(0), color=C[m], alpha=0.12, lw=0)
            ax.plot(t, curves.mean(0), color=C[m], ls=ls, label=lab)
        r = d[(d.method == "random") & (d.blackout_length == "none")]
        t, S = survival(r)
        ax.plot(t, S * 100, color=C["random"], ls=":", lw=1.5, label="Random policy")
        ax.set_title(LABEL[m], loc="left")
        ax.set_xlabel("Episode step")
        ax.set_xlim(0, 100)
    axes[0].set_ylabel("Prey uncaptured (%)")
    axes[0].set_ylim(0, 100)
    axes[0].legend(loc="lower left")
    fig.suptitle("Fig 4: Time to capture", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    # caption(fig, f"Share of episodes with the prey still free at each step. Line = mean over {RUNS_LABEL}, "
    #              f"band = min–max across seeds; '{sel}' checkpoint.")
    save(fig, out, "fig4_time_to_capture")


COORD = [("spread", "Formation spread"), ("max_angular_gap", "Max angular gap (deg)"),
         ("mean_distance", "Mean pursuer–prey distance"), ("collisions", "Collisions / episode")]
COND_SHOW = [("none", 0, "none", "none"), ("20", 2, "zero", "2 for 20"),
             ("permanent", 1, "zero", "perm  1"), ("permanent", 2, "zero", "perm  2")]


def fig_coord(d, sel, out):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    x = np.arange(len(COND_SHOW))
    for ax, (col, title) in zip(axes, COORD):
        ps = per_seed(d, col)
        for i, m in enumerate(METHODS):
            res = [tci(seed_vals(ps, m, sel, L, nb, f, metric=col)) for L, nb, f, _ in COND_SHOW]
            mu, hw = np.array([r[0] for r in res]), np.array([r[1] for r in res])
            off = (i - 0.5) * 0.2
            ax.errorbar(x + off, mu, yerr=hw, color=C[m], marker="o", ls="none", capsize=3,
                        elinewidth=1.2, label=LABEL[m], mec=SURF, mew=1.5)
        ax.set_xticks(x, [c[3] for c in COND_SHOW], fontsize=8.5)
        ax.set_title(title, loc="left", fontsize=10)
    axes[0].legend(loc="best", fontsize=8.5)
    fig.suptitle("Fig 6: Coordination metrics by condition (zero fill; length / #blinded; mean ± 95% CI)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig6_coordination")


def fig_seeds(ps, sel, out):
    conds = [("none", 0, "none", "none"), ("permanent", 1, "zero", "perm 1"), ("permanent", 2, "zero", "perm 2")]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(conds))
    for i, m in enumerate(METHODS):
        off = (i - 0.5) * 0.25
        V = pd.concat([seed_vals(ps, m, sel, L, nb, f) for L, nb, f, _ in conds], axis=1) * 100
        for s, row in V.iterrows():
            ax.plot(x + off, row.values, color=C[m], alpha=0.45, lw=1, marker="o", ms=6, mec=SURF)
        dx = -0.07 if m == "clean" else 0.07
        for y, ty, s in zip(V.iloc[:, 0].values, declutter(V.iloc[:, 0].values, 2.6), V.index):
            if KEEP is not None:
                break
            ax.annotate(f"s{s}", (x[0] + off, y), xytext=(x[0] + off + dx, ty), textcoords="data",
                        va="center", ha="right" if m == "clean" else "left", fontsize=7.5, color=MUTED)
        ax.plot(x + off, V.mean().values, color=C[m], lw=2.5, marker="D", ms=8, mec=SURF, label=LABEL[m] + " (mean)")
    ax.set_xticks(x, [c[3] for c in conds])
    ax.set_xlabel("Condition (zero fill)")
    ax.set_ylabel("Capture rate (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower left")
    ax.set_title("Fig 7: Per-seed capture rates (thin lines = individual seeds)", loc="left")
    fig.tight_layout()
    save(fig, out, "fig7_per_seed")


# --------------------------------------------------------------- tables ----
def write_table(df, out, name, title):
    df.to_csv(os.path.join(out, name + ".csv"), index=False)
    with open(os.path.join(out, name + ".md"), "w", encoding="utf-8") as f:
        cols = [str(c) for c in df.columns]
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
        lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
        f.write(f"**{title}**\n\n" + "\n".join(lines) + "\n")
    with open(os.path.join(out, name + ".tex"), "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, escape=True).replace("±", r"$\pm$"))


def all_conditions():
    yield ("none", 0, "none")
    for L in LENGTHS[1:]:
        for nb in (1, 2):
            for fl in ("zero", "hold"):
                yield (L, nb, fl)


def cond_label(L, nb, fl):
    return "no blackout" if L == "none" else f"{L} / {nb} / {fl}"


def table_main(d, ps, sel, out, conds, name, title, include_return=False):
    # capture rate conditional on the episode reaching blackout start
    d_nc = d[~d.cbb]
    ps_nc = per_seed(d_nc)
    rows = []
    for L, nb, fl in conds:
        row = {"Condition (length / blinded / fill)": cond_label(L, nb, fl)}
        vals = {m: seed_vals(ps, m, sel, L, nb, fl) for m in METHODS}
        for m in METHODS:
            row[LABEL[m]] = fmt(*tci(vals[m]))
        p, lo, hi = rand_curve(d, L, nb, fl)
        row[LABEL["random"]] = f"{p*100:.1f}"
        row["Masked − clean"] = fmt(*welch(vals["clean"], vals["masked"]))
        if L != "none":
            for m in METHODS:
                row[f"{LABEL[m]} (reached blackout)"] = fmt(*tci(seed_vals(ps_nc, m, sel, L, nb, fl)))
        else:
            for m in METHODS:
                row[f"{LABEL[m]} (reached blackout)"] = "—"
        if include_return:
            pr = per_seed(d, "env_return")
            for m in METHODS:
                row[f"{LABEL[m]} return"] = fmt(*tci(seed_vals(pr, m, sel, L, nb, fl, metric="env_return")), pct=False)
        rows.append(row)
    write_table(pd.DataFrame(rows), out, name, title)
    return pd.DataFrame(rows)


def table_coord(d, sel, out):
    rows = []
    for L, nb, f, lab in COND_SHOW:
        for m in METHODS:
            row = {"Condition": lab, "Method": LABEL[m]}
            for col, title in [("captured", "Capture %")] + COORD + [("capture_step", "Capture step (if captured)")]:
                src = d[d.captured] if col == "capture_step" else d
                ps = per_seed(src, col)
                mu, hw = tci(seed_vals(ps, m, sel, L, nb, f, metric=col))
                row[title] = fmt(mu, hw, pct=(col == "captured"))
            rows.append(row)
    write_table(pd.DataFrame(rows), out, "table2_coordination", "Table 2. Coordination metrics (zero fill, mean ± 95% CI over seeds)")


def table_ckpt(d, out):
    ps = per_seed(d)
    rows = []
    for sel in ["best", "best_noloss", "final"]:
        for m in METHODS:
            base = seed_vals(ps, m, sel, "none", 0, "none")
            row = {"Checkpoint": sel, "Method": LABEL[m], "No blackout": fmt(*tci(base))}
            for L, nb, f, lab in COND_SHOW[1:]:
                v = seed_vals(ps, m, sel, L, nb, f)
                row[lab] = fmt(*tci(v))
                row[f"Δ {lab}"] = fmt(*tci(v - base))
            rows.append(row)
    write_table(pd.DataFrame(rows), out, "tableA2_checkpoint_sensitivity",
                "Table A2. Checkpoint-selection sensitivity (zero fill, capture %, mean ± 95% CI; Δ = paired drop vs. no blackout)")


# ------------------------------------------- figure data (B-tables) ----
SEEDS = [1, 2, 3, 4, 5]
P = lambda v: round(float(v) * 100, 1)  # fraction -> percent, 1 dp


def seed_row(v, prefix=""):
    return {f"{prefix}s{s}": P(v.get(s, np.nan)) for s in SEEDS}


def data_fig1(runs, out):
    rows = []
    for col, vname in [("val_capture_noloss", "no blackout"), ("val_capture_blackout", "hard blackout")]:
        for m in METHODS:
            cur = {}
            for s in SEEDS:
                lc = pd.read_csv(glob.glob(os.path.join(runs, f"{m}_n3_*_s{s}", "learning_curve.csv"))[0])
                cur[s] = curve(lc, col)
            df = pd.DataFrame(cur).dropna()
            for ep, r in df.iterrows():
                rows.append({"Validation": vname, "Method": LABEL[m], "Episode": int(ep), **seed_row(r),
                             "Mean": P(r.mean()), "±CI": P(tci(r.values)[1]), "Min": P(r.min()), "Max": P(r.max())})
    write_table(pd.DataFrame(rows), out, "tableB1_fig1_learning_curves",
                "Table B1. Data for Fig 1: validation capture rate (%) per seed at each evaluation point")


def data_capture_per_seed(d, ps, sel, out):
    rows = []
    for L, nb, fl in all_conditions():
        row = {"Condition (length / blinded / fill)": cond_label(L, nb, fl)}
        for m in METHODS:
            v = seed_vals(ps, m, sel, L, nb, fl)
            mu, hw = tci(v)
            row.update(seed_row(v, prefix=f"{m} "))
            row[f"{m} mean"], row[f"{m} ±CI"] = P(mu), P(hw)
        p, lo, hi = rand_curve(d, L, nb, fl)
        row["random"], row["random CI low"], row["random CI high"] = P(p), P(lo), P(hi)
        rows.append(row)
    write_table(pd.DataFrame(rows), out, "tableB2_fig2_capture_per_seed",
                f"Table B2. Data for Fig 2 (zero-fill rows): capture rate (%) per seed, all 21 conditions ('{sel}' checkpoint). Random CI = Wilson 95%.")
    # long format, every checkpoint
    long = ps.copy()
    long["capture_pct"] = (long.pop("captured") * 100).round(2)
    long.sort_values(["method", "selection", "training_seed", "blackout_length", "num_blinded", "fill"]).to_csv(
        os.path.join(out, "tableB2_long_all_checkpoints.csv"), index=False)


def paired_table(ps, sel, out, name, title, fn):
    rows = []
    for nb in (1, 2):
        for L in LENGTHS[1:]:
            row = {"Blinded": nb, "Length": L}
            for m in METHODS:
                v = fn(m, L, nb)
                mu, hw = tci(v)
                row.update(seed_row(v, prefix=f"{m} "))
                row[f"{m} mean"], row[f"{m} ±CI"] = P(mu), P(hw)
            rows.append(row)
    write_table(pd.DataFrame(rows), out, name, title)


def data_fig4(d, sel, out):
    conds = [("none", 0, "none", "no blackout"), ("permanent", 2, "zero", "perm / 2 / zero")]
    cols, long = {}, []
    for m in METHODS:
        for L, nb, fl, lab in conds:
            q = d[(d.method == m) & (d.selection == sel) & (d.blackout_length == L)
                  & (d.num_blinded == nb) & (d.fill == fl)]
            S = {}
            for s, e in q.groupby("training_seed"):
                t, S[s] = survival(e)
                long += [{"method": m, "condition": lab, "training_seed": s, "step": int(k), "pct_uncaptured": P(v)}
                         for k, v in zip(t, S[s])]
            A = np.array([S[s] for s in SEEDS])
            cols[f"{m} {lab} mean"] = A.mean(0)
            cols[f"{m} {lab} min"] = A.min(0)
            cols[f"{m} {lab} max"] = A.max(0)
    t, S = survival(d[(d.method == "random") & (d.blackout_length == "none")])
    cols["random no blackout"] = S
    df = pd.DataFrame({"Step": t, **{k: np.round(v * 100, 1) for k, v in cols.items()}})
    write_table(df, out, "tableB4_fig4_time_to_capture",
                "Table B4. Data for Fig 4: % of episodes with prey still uncaptured at each step (across-seed mean / min / max)")
    pd.DataFrame(long).to_csv(os.path.join(out, "tableB4_long_per_seed.csv"), index=False)


def data_fig6(d, sel, out):
    rows = []
    for col, title in COORD:
        ps = per_seed(d, col)
        for L, nb, f, lab in COND_SHOW:
            for m in METHODS:
                v = seed_vals(ps, m, sel, L, nb, f, metric=col)
                mu, hw = tci(v)
                rows.append({"Metric": title, "Condition": lab, "Method": LABEL[m],
                             **{f"s{s}": round(float(v.get(s, np.nan)), 3) for s in SEEDS},
                             "Mean": round(mu, 3), "±CI": round(hw, 3)})
    write_table(pd.DataFrame(rows), out, "tableB6_fig6_coordination_per_seed",
                "Table B6. Data for Fig 6: coordination metrics per seed (zero fill)")


def data_fig7(ps, sel, out):
    rows = []
    for L, nb, f, lab in [("none", 0, "none", "none"), ("permanent", 1, "zero", "perm 1"),
                          ("permanent", 2, "zero", "perm 2")]:
        for m in METHODS:
            v = seed_vals(ps, m, sel, L, nb, f)
            rows.append({"Condition": lab, "Method": LABEL[m], **seed_row(v), "Mean": P(v.mean())})
    write_table(pd.DataFrame(rows), out, "tableB7_fig7_per_seed",
                "Table B7. Data for Fig 7: capture rate (%) per seed (zero fill)")


def export_figure_data(d, ps, sel, runs, out):
    data_fig1(runs, out)
    data_capture_per_seed(d, ps, sel, out)
    paired_table(ps, sel, out, "tableB3_fig3_drop_per_seed",
                 "Table B3. Data for Fig 3: change in capture rate vs. own no-blackout baseline (pts), per seed, zero fill. CI = paired t over seeds",
                 lambda m, L, nb: seed_vals(ps, m, sel, L, nb, "zero") - seed_vals(ps, m, sel, "none", 0, "none"))
    paired_table(ps, sel, out, "tableB5_fig5_hold_minus_zero_per_seed",
                 "Table B5. Data for Fig 5: hold − zero fill capture rate (pts), per seed. CI = paired t over seeds",
                 lambda m, L, nb: seed_vals(ps, m, sel, L, nb, "hold") - seed_vals(ps, m, sel, L, nb, "zero"))
    data_fig4(d, sel, out)
    data_fig6(d, sel, out)
    data_fig7(ps, sel, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "analysis"))
    ap.add_argument("--selection", default="best")
    ap.add_argument("--seeds", help="subset of runs, e.g. 'clean:1,2,5;masked:2,4,5' (figures only)")
    ap.add_argument("--note", default="", help="footnote printed on every figure")
    a = ap.parse_args()
    global KEEP, NOTE, RUNS_LABEL
    NOTE = a.note
    if a.seeds:
        KEEP = parse_seeds(a.seeds)
        sizes = {len(v) for v in KEEP.values()}
        RUNS_LABEL = f"{sizes.pop()} selected runs" if len(sizes) == 1 else "selected runs"
    os.makedirs(a.out, exist_ok=True)
    d = load(a.runs)
    ps = per_seed(d)
    sel = a.selection

    fig_learning(a.runs, a.out)
    fig_capture_vs_length(d, ps, sel, a.out)
    fig_drop(ps, sel, a.out)
    fig_survival(d, sel, a.out)
    fig_fill(ps, sel, a.out)
    fig_coord(d, sel, a.out)
    fig_seeds(ps, sel, a.out)
    if KEEP is not None:          # subset mode: figures only (tables would list individual runs)
        print("wrote", a.out)
        return

    key = [("none", 0, "none"), ("20", 1, "zero"), ("20", 2, "zero"), ("40", 2, "zero"),
           ("permanent", 1, "zero"), ("permanent", 2, "zero")]
    table_main(d, ps, sel, a.out, key, "table1_main_results",
               f"Table 1. Capture rate (%) on key conditions ('{sel}' checkpoint, mean ± 95% CI over 5 seeds)")
    table_main(d, ps, sel, a.out, list(all_conditions()), "tableA1_all_conditions",
               f"Table A1. All 21 sweep conditions ('{sel}' checkpoint)", include_return=True)
    table_coord(d, sel, a.out)
    table_ckpt(d, a.out)
    export_figure_data(d, ps, sel, a.runs, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
