"""Figures for the paper, computed from the `cdm` package and scripts/experiments.py.

Figures are drawn at their printed size (full width 6.5 in, half width 3.1 in) so text renders at
8-9 pt next to the body type, and they carry no titles: the LaTeX captions do that job.

    uv run python scripts/experiments.py   # once, writes paper_figs/experiments.json
    uv run python scripts/figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, Patch
from scipy.stats import beta

from cdm import Lottery, Policy, expected_improvement
from cdm.belief import temper
from cdm.decide import branches
from cdm.lottery import Competitor, p_best

OUT = Path("paper_figs")
FULL, HALF, H = 6.5, 3.1, 2.25

# Categorical slots in fixed order (validated palette, light surface); grey is the reference.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
    }
)


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  wrote {name}")


def _half() -> tuple[plt.Figure, plt.Axes]:
    return plt.subplots(figsize=(HALF, H), layout="constrained")


# ---------------------------------------------------------------------------------------------
def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(FULL, 1.25))
    ax.set_xlim(0, 100)
    ax.set_ylim(0.5, 17.5)
    ax.axis("off")
    boxes = [
        ("diverge", "options", "people or a\ngenerative model", BLUE),
        (
            "belief",
            r"$\Delta^{\!\circ}(\mathrm{levels})^D$",
            "debias, calibrate,\nlog-pool judges",
            AQUA,
        ),
        (
            "value",
            r"lotteries on $\mathbb{R}$",
            r"$\mathbb{E}$, CVaR, $P(\mathrm{best})$," + "\nEI, EVPI (exact)",
            YELLOW,
        ),
        ("policy", "action", "commit, innovate,\nportfolio, escalate", ORANGE),
    ]
    w, gap = 21.5, 4.67
    for k, (layer, obj, body, col) in enumerate(boxes):
        x = k * (w + gap)
        box = FancyBboxPatch(
            (x, 1), w, 16, boxstyle="round,pad=0,rounding_size=1.6", facecolor="#f7f7f5", lw=0
        )
        ax.add_patch(box)
        head = ax.add_patch(
            FancyBboxPatch((x, 13.6), w, 3.4, boxstyle="square,pad=0", facecolor=col, lw=0)
        )
        head.set_clip_path(box)
        ax.add_patch(
            FancyBboxPatch(
                (x, 1),
                w,
                16,
                boxstyle="round,pad=0,rounding_size=1.6",
                facecolor="none",
                edgecolor=col,
                lw=1.2,
            )
        )
        ax.text(x + w / 2, 15.3, layer, ha="center", va="center", color="white", fontsize=8.5)
        ax.text(x + w / 2, 10.6, obj, ha="center", va="center", color=INK, fontsize=8.5)
        ax.text(
            x + w / 2,
            5.2,
            body,
            ha="center",
            va="center",
            color=INK2,
            fontsize=7.5,
            linespacing=1.3,
        )
        if k < 3:
            ax.annotate(
                "",
                xy=(x + w + gap - 0.6, 9),
                xytext=(x + w + 0.6, 9),
                arrowprops={"arrowstyle": "-|>", "color": INK2, "lw": 1.0, "mutation_scale": 9},
            )
    _save(fig, "fig_pipeline")


def fig_simplex() -> None:
    fig, ax = plt.subplots(figsize=(HALF, H), layout="constrained")
    h = np.sqrt(3) / 2

    def cart(b: np.ndarray) -> np.ndarray:
        b = np.atleast_2d(b)
        return np.stack([b[:, 1] + 0.5 * b[:, 2], h * b[:, 2]], axis=1)

    tri = np.array([[0, 0], [1, 0], [0.5, h], [0, 0]])
    ax.fill(tri[:, 0], tri[:, 1], color="#f3f3f1", zorder=0)
    ax.plot(tri[:, 0], tri[:, 1], color=INK2, lw=0.8)
    for label, xy, ha in (
        ("$e_1$", (-0.03, -0.04), "right"),
        ("$e_2$", (1.03, -0.04), "left"),
        ("$e_3$", (0.5, h + 0.03), "center"),
    ):
        ax.text(*xy, label, ha=ha, va="center", color=INK2)

    u = np.ones(3) / 3
    p = np.array([0.12, 0.30, 0.58])
    q = np.array([0.62, 0.30, 0.08])
    alphas = np.linspace(0, 2.4, 120)
    for v, col in ((p, BLUE), (q, ORANGE)):
        path = cart(np.array([temper(v, 1 / a) if a > 0 else u for a in alphas]))
        ax.plot(path[:, 0], path[:, 1], color=col, lw=1.4, zorder=2)
    # a Bayes update ℓ ⊕ p is a straight translation in clr coordinates
    ell = np.array([0.4, 2.5, 1.0])
    upd = ell * p / (ell * p).sum()
    a, b = cart(p)[0], cart(upd)[0]
    ax.annotate(
        "",
        xy=b,
        xytext=a,
        arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 0.9, "mutation_scale": 8},
    )
    ax.text(*(b + np.array([0.03, -0.02])), r"$\ell\oplus p$", fontsize=8.5, color=INK)
    for v, name, off in (
        (u, "$u$", (-0.07, -0.02)),
        (p, "$p$", (0.03, 0.0)),
        (q, "$q$", (-0.02, 0.04)),
        (upd, "", (0, 0)),
    ):
        xy = cart(v)[0]
        ax.scatter(
            *xy,
            s=22,
            color=INK if name in ("$u$", "") else (BLUE if name == "$p$" else ORANGE),
            edgecolor="white",
            lw=1.0,
            zorder=4,
        )
        if name:
            ax.text(xy[0] + off[0], xy[1] + off[1], name, fontsize=9, color=INK)
    ax.text(
        0.0,
        1.0,
        r"paths $\alpha\odot p$, $\alpha\odot q$:" + "\nlines through $u$,\nstraight in clr",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color=INK2,
    )
    ax.set_aspect("equal")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.08, h + 0.08)
    ax.axis("off")
    _save(fig, "fig_simplex")


def fig_temper() -> None:
    fig, ax = _half()
    p = np.array([0.04, 0.08, 0.18, 0.30, 0.40])
    lv = np.arange(1, 6)
    ax.axhline(0.2, color=MUTED, ls=(0, (2, 2)), lw=0.9)
    ax.text(1, 0.212, "uniform", color=INK2, fontsize=7.5)
    for t, col in ((1, BLUE), (2, ORANGE), (4, AQUA), (8, YELLOW)):
        y = temper(p, t)
        ax.plot(lv, y, "-o", color=col, mec="white", mew=0.8, label=f"$T={t}$")
        ax.text(5.15, y[-1], f"$T={t}$", va="center", fontsize=7.5, color=INK)
    ax.set_xticks(lv)
    ax.set_xlim(0.8, 5.75)
    ax.set_ylim(0, 0.44)
    ax.set_xlabel(r"level (worst $\rightarrow$ best)")
    ax.set_ylabel("probability")
    ax.spines["bottom"].set_bounds(1, 5)
    _save(fig, "fig_temper")


def fig_cvar() -> None:
    fig, ax = _half()
    bold = Lottery.of([0.0, 0.25, 0.5, 0.75, 1.0], [0.08, 0.12, 0.25, 0.30, 0.25])
    base = Lottery.of([0.25, 0.5, 0.75], [0.2, 0.6, 0.2])
    al = np.linspace(0.01, 1, 300)
    tau = 0.05
    cb = np.array([base.cvar(a) for a in al])
    ax.fill_between(al, cb - tau, cb, color=MUTED, alpha=0.18, lw=0, label=r"tolerance $\tau$")
    ax.plot(al, cb, color=MUTED, label="status quo $B$")
    ax.plot(al, [bold.cvar(a) for a in al], color=BLUE, label="bold option $U$")
    a0 = 0.2
    ax.axvline(a0, color=INK2, lw=0.7, ls=(0, (1, 2)))
    ax.scatter(
        [a0, a0],
        [bold.cvar(a0), base.cvar(a0)],
        s=16,
        color=[BLUE, MUTED],
        zorder=4,
        edgecolor="white",
    )
    ax.text(a0 + 0.02, 0.03, r"$\alpha=0.2$: $U$ unreliable", fontsize=7.5, color=INK)
    ax.set_xlabel(r"tail mass $\alpha$")
    ax.set_ylabel(r"$\mathrm{CVaR}_\alpha$ (lower-tail mean)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.68)
    ax.legend(loc="upper left")
    _save(fig, "fig_cvar")


def fig_ties() -> None:
    fig, ax = _half()
    ns = np.arange(1, 9)
    one = Lottery.point(1.0)
    fair = np.array([p_best([Competitor(one) for _ in range(int(n))]).sum() for n in ns])
    ax.plot(ns, ns, "-o", color=ORANGE, mec="white", mew=0.8, label=r"count $\geq$ as a win")
    ax.plot(ns, fair, "-o", color=BLUE, mec="white", mew=0.8, label="fair split (L4)")
    ax.plot(ns, 0 * ns, "-o", color=AQUA, mec="white", mew=0.8, label=r"count only $>$")
    ax.set_xlabel("$n$ identical competitors")
    ax.set_ylabel(r"total $P(\mathrm{best})$ mass")
    ax.set_xticks(ns)
    ax.set_ylim(-0.4, 8.6)
    ax.legend(loc="upper left")
    _save(fig, "fig_ties")


def fig_ei() -> None:
    fig, ax = _half()
    b = Lottery.point(0.5)
    ws = np.linspace(0, 1, 101)
    for f, col in ((0.0, BLUE), (0.3, ORANGE), (0.6, AQUA)):
        ei = [
            expected_improvement(Lottery.of([0.5 - w / 2, 0.5 + w / 2], [0.5, 0.5]), b, f)
            for w in ws
        ]
        ax.plot(ws, ei, color=col, label=f"$f={f:g}$")
    ax.axhline(0, color=MUTED, lw=0.9, ls=(0, (2, 2)))
    ax.text(0.5, 0.006, r"mean gain $\mathbb{E}[V]-B=0$", fontsize=7.5, color=INK2)
    ax.set_xlabel(r"spread $w$ of $V=0.5\pm w/2$ (mean fixed)")
    ax.set_ylabel(r"$\mathrm{EI}(V;B)\cdot(1-f)$")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.01, 0.27)
    ax.legend(loc="upper left", title="hard-fail mass", title_fontsize=7.5)
    _save(fig, "fig_ei")


def fig_verify() -> None:
    fig, ax = _half()
    v = Lottery.of([0.25, 0.5, 0.75, 1.0], [0.1, 0.2, 0.3, 0.4])
    b = Lottery.point(0.5)
    x = np.linspace(-0.02, 1.05, 800)
    for s, col in ((1.0, BLUE), (0.9, ORANGE), (0.7, AQUA)):
        u, _, _, res = branches(v, b, 0.5, Policy(check_sensitivity=s), verify=True)
        ax.step(x, u.cdf(x), where="post", color=col, label=f"$s={s:g}$: $P(U=H)={res:.2f}$")
    ax.axvline(0.5, color=MUTED, lw=0.9, ls=(0, (2, 2)))
    ax.text(0.515, 0.04, "$B$", color=INK2)
    ax.text(0.015, 0.2, "$H$", color=INK2)
    ax.set_xlabel("utility")
    ax.set_ylabel(r"$P(U\leq x)$")
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(0, 1.04)
    ax.legend(loc="upper left", handlelength=1.4)
    _save(fig, "fig_verify")


def fig_unknown() -> None:
    fig, ax = _half()
    x = np.linspace(0.0005, 0.9995, 600)
    pol = Policy()
    ax.axvspan(0, pol.clear_below, color=AQUA, alpha=0.10, lw=0)
    ax.axvspan(pol.reject_above, 1, color=ORANGE, alpha=0.10, lw=0)
    for xpos, lab in (
        (pol.clear_below / 2, "clear"),
        ((pol.clear_below + pol.reject_above) / 2, "verify"),
        ((1 + pol.reject_above) / 2, "reject"),
    ):
        ax.text(xpos, 11.6, lab, ha="center", fontsize=7.5, color=INK2)
    for (a, bb), col, lab in (
        ((1, 1), BLUE, "unknown: Beta(1,1)"),
        ((50, 50), ORANGE, "known coin flip: Beta(50,50)"),
        ((18, 2), AQUA, "evidence of harm: Beta(18,2)"),
    ):
        ax.plot(x, beta.pdf(x, a, bb), color=col, label=lab)
    ax.set_xlabel("violation probability $q$")
    ax.set_ylabel("belief density")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 12.5)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93))
    _save(fig, "fig_unknown")


def fig_dependence() -> None:
    fig, ax = _half()
    q = np.linspace(0, 0.5, 200)
    m = 3
    ax.fill_between(q, q, np.minimum(1, m * q), color=MUTED, alpha=0.16, lw=0)
    ax.plot(q, np.minimum(1, m * q), color=ORANGE)
    ax.plot(q, 1 - (1 - q) ** m, color=BLUE)
    ax.plot(q, q, color=AQUA)
    ax.scatter([0.2], [1 - 0.8**3], s=16, color=BLUE, edgecolor="white", zorder=4)
    ax.text(0.19, 1 - 0.8**3 + 0.03, "0.488", fontsize=7.5, color=INK, ha="right")
    kw = {"fontsize": 7.5, "color": INK, "ha": "right"}
    ax.text(0.495, 0.89, r"worst case $\min(1,\sum q)$", **kw)
    ax.text(0.495, 0.80, "independent", **kw)
    ax.text(0.495, 0.53, r"comonotone $\max q$", **kw)
    ax.text(0.25, 0.37, "any\ndependence", fontsize=7.5, color=INK2, ha="center")
    ax.set_xlabel("$q$ for each of 3 hard checks")
    ax.set_ylabel(r"$f=P(\mathrm{any\ violation})$")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1.03)
    _save(fig, "fig_dependence")


def fig_innovation(exp: dict) -> None:
    inn = exp["innovation"]
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.2), layout="constrained", sharex=True)
    cmap = ListedColormap(["#ecebe7", BLUE, ORANGE, AQUA])
    s = np.array(inn["spreads"])
    for ax, key, ys, lab in (
        (axes[0], "by_tau", np.array(inn["taus"]), r"tolerance $\tau$  ($q=0$)"),
        (axes[1], "by_q", np.array(inn["qs"]), r"$q$ of wide, perfect check"),
        (axes[2], "by_q_check", np.array(inn["qs"]), r"$q$ of wide, $s=0.9$, $H=0$"),
    ):
        ax.pcolormesh(
            s,
            ys,
            np.array(inn[key]),
            cmap=cmap,
            vmin=0,
            vmax=3,
            shading="nearest",
            edgecolors="white",
            lw=0.3,
        )
        ax.set_ylabel(lab)
        ax.set_xlabel("spread of wide")
        ax.grid(visible=False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    seen = {c for k in ("by_tau", "by_q", "by_q_check") for row in inn[k] for c in row}
    names = ("no slot", "narrow", "wide", "wide commits")
    axes[2].legend(
        handles=[Patch(color=cmap(k), label=names[k]) for k in sorted(seen)],
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        title="innovation slot",
        title_fontsize=7.5,
    )
    _save(fig, "fig_innovation")


STYLES = {
    "raw judge": (YELLOW, "s"),
    "accept uncertain": (MAGENTA, "D"),
    "veto uncertain": (AQUA, "^"),
    "cdm, perfect-check model": (BLUE, "o"),
    "cdm, check model, tuned tau": (ORANGE, "o"),
}


def _symlog(ax: plt.Axes, axis: str, top: float) -> None:
    ticks, labels = [0, 0.005, 0.01, 0.1, 0.3], ["0", "0.005", "0.01", "0.1", "0.3"]
    if axis == "x":
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xlim(-0.0005, top)
        ax.set_xticks(ticks, labels)
    else:
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_ylim(-0.0005, top)
        ax.set_yticks(ticks, labels)


def fig_benchmark(exp: dict) -> None:
    b = exp["benchmark"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.45), layout="constrained")
    ax = axes[0]
    for name, (col, mk) in STYLES.items():
        lab = "cdm, check model, tuned $\\tau$" if "tuned" in name else name
        ax.plot(
            b["sens"], b["harm_by_sens"][name], "-" + mk, color=col, mec="white", mew=0.7, label=lab
        )
    ax.set_xlabel("check sensitivity $s$")
    ax.set_ylabel("P(undetected harm)")
    _symlog(ax, "y", 0.45)

    ax = axes[1]
    fr = b["frontier"]
    taus = list(b["taus"])
    hs = [fr[t]["harm"] for t in taus]
    rs = [fr[t]["regret"] for t in taus]
    ax.plot(hs, rs, "-o", color=ORANGE, mec="white", mew=0.7, ms=4, zorder=3)
    same = next(
        t for t in taus if fr[t]["regret"] == fr["inf"]["regret"]
    )  # where the gate stops binding
    labelled = {"0.0": r"$\tau=0$", "0.05": r"$\tau=0.05$", "0.1": r"$\tau=0.1$"}
    labelled[same] = rf"$\tau\geq{float(same):g}$ = no gate"
    for t, h, r in zip(taus, hs, rs, strict=True):
        if t in labelled:
            ax.annotate(
                labelled[t],
                (h, r),
                textcoords="offset points",
                xytext=(6, 2),
                fontsize=7,
                color=INK2,
            )
    t = b["table"]
    for name, (col, mk) in STYLES.items():
        if "tuned" in name:
            continue
        d = t[name]
        ax.errorbar(
            d["harm"],
            d["regret"],
            yerr=1.96 * d["regret_se"],
            fmt=mk,
            color=col,
            mec="white",
            mew=0.7,
            ms=5,
            capsize=0,
            lw=0.9,
        )
    _symlog(ax, "x", 0.4)
    ax.set_xlabel("P(undetected harm)  at $s=0.9$")
    ax.set_ylabel("regret vs. oracle")
    ax.set_ylim(0, 0.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=5,
        fontsize=7,
        handlelength=1.8,
        columnspacing=1.0,
    )
    _save(fig, "fig_benchmark")


def fig_jev(jev: dict) -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(FULL, 2.4), layout="constrained", width_ratios=[1.15, 1]
    )
    ax = axes[0]
    pol = Policy()
    ax.axvspan(0, pol.clear_below, color=AQUA, alpha=0.10, lw=0)
    ax.axvspan(pol.reject_above, 1, color=ORANGE, alpha=0.10, lw=0)
    rng = np.random.default_rng(0)
    rows = (
        ("harm", "explicitly harmful"),
        ("ambiguous", "ambiguous (true rate 0.4)"),
        ("safe", "explicitly safe"),
    )
    for k, (cls, _lab) in enumerate(rows):
        q = np.array(jev["q_by_class"][cls])
        ax.scatter(
            q,
            k + rng.uniform(-0.18, 0.18, q.size),
            s=9,
            color=(ORANGE, BLUE, AQUA)[k],
            alpha=0.75,
            lw=0,
        )
        ax.plot([q.mean()] * 2, [k - 0.3, k + 0.3], color=INK, lw=1.2)
    ax.set_yticks(range(3), [r[1] for r in rows])
    ax.set_xlabel("Jev's violation probability $q$ (raw Noul)")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 2.6)
    ax.grid(axis="y", visible=False)
    for x, lab in (
        (pol.clear_below / 2, "clear"),
        ((pol.clear_below + pol.reject_above) / 2, "verify"),
        ((1 + pol.reject_above) / 2, "reject"),
    ):
        ax.text(x, 2.45, lab, ha="center", fontsize=7, color=INK2)

    ax = axes[1]
    ps = jev["policies"]
    show = {
        "raw judge": (YELLOW, "s", "raw judge"),
        "accept uncertain": (MAGENTA, "D", "accept uncertain"),
        "veto uncertain": (AQUA, "^", "veto uncertain"),
        "cdm, check model, tuned tau": (BLUE, "o", "cdm (all variants)"),
    }
    for name, (col, mk, lab) in show.items():
        d = ps[name]
        ax.errorbar(
            100 * d["harm"],
            d["regret"],
            yerr=1.96 * d["regret_se"],
            fmt=mk,
            color=col,
            mec="white",
            mew=0.7,
            ms=6,
            capsize=0,
            lw=0.9,
            label=lab,
        )
    ax.set_xlabel("undetected harm (%)")
    ax.set_ylabel("regret vs. oracle")
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 0.28)
    ax.legend(loc="upper center", ncol=2, fontsize=7, handlelength=1.2, columnspacing=1.0)
    _save(fig, "fig_jev")


if __name__ == "__main__":
    exp = json.loads((OUT / "experiments.json").read_text())
    fig_pipeline()
    fig_simplex()
    fig_temper()
    fig_cvar()
    fig_ties()
    fig_ei()
    fig_verify()
    fig_unknown()
    fig_dependence()
    fig_innovation(exp)
    fig_benchmark(exp)
    fig_jev(json.loads((OUT / "jev_benchmark.json").read_text()))
