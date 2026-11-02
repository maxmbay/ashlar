"""The survival figure: five SUPPORT patients' survival curves from two models, and whether each model's curves hold.

``compute`` runs in the cloud container in a few minutes; SUPPORT comes from the DeepSurv repository on GitHub, as
pycox ships it. On the five splits of ``benchmark_observed_real.py survival`` it fits the ``mixture`` head on
``observed.survival(np.log(days), died)`` and XGBoost's AFT model tuned on its 42-fit grid, as the benchmark
tunes it, scores both as the benchmark does, and keeps each split's D-calibration weights. On split 0 it keeps the
survival curves over five years of five test patients, chosen at evenly spaced quantiles of the mixture's chance of living a year,
with their outcomes, and the share of pairs of test patients whose curves cross. It writes
``examples/readme/artifacts/survival/survival.json``.

``draw`` writes ``docs/images/readme/survival.png`` and the gallery thumbnail ``thumbs/survival.png``.

    JAX_PLATFORMS=cpu python examples/readme/survival.py compute
    python examples/readme/survival.py draw
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from matplotlib.ticker import NullLocator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import style as k  # noqa: E402

FIGURE = "survival"
SPLITS = (0, 1, 2, 3, 4)
FIGURE_SPLIT = 0
PICKS = (0.1, 0.3, 0.5, 0.7, 0.9)  # quantiles of the mixture's one-year survival
YEAR = 365.25
DAYS = np.geomspace(1.0, 5 * YEAR, 220)  # SUPPORT's follow-up runs to 2,029 days
CROSS_ROWS = 400  # test patients whose curves are compared pair by pair
CROSS_GAP = 0.02  # two curves cross when each is above the other by this much somewhere


def _crossing_share(curves: np.ndarray) -> float:
    """The share of pairs of rows of ``curves`` (rows, days) that each lie above the other by ``CROSS_GAP`` somewhere."""
    rows = len(curves)
    first, second = np.triu_indices(rows, k=1)
    crossed = 0
    for start in range(0, len(first), 20_000):
        a, b = first[start : start + 20_000], second[start : start + 20_000]
        difference = curves[a] - curves[b]
        crossed += int(np.sum((difference.max(axis=1) > CROSS_GAP) & (difference.min(axis=1) < -CROSS_GAP)))
    return crossed / len(first)


def compute() -> None:
    import jax.numpy as jnp
    import xgboost
    from sklearn.model_selection import train_test_split

    import benchmark_observed_real as real
    from ashlar import ASRRegressor, observed

    x, t, e = real.survival_set("SUPPORT")
    log_t = np.log(t)
    print(f"SUPPORT: {len(t)} patients, {x.shape[1]} features, {e.mean():.0%} died during follow-up", flush=True)
    splits, figure = {}, None
    for seed in SPLITS:
        train, test = train_test_split(np.arange(len(t)), test_size=0.2, stratify=e, random_state=seed)
        fit, held = train_test_split(train, test_size=0.15, stratify=e[train], random_state=seed)
        grid = np.unique(np.quantile(t[test], np.linspace(0.1, 0.9, real.GRID_POINTS)))
        grid = grid[grid < t[test].max()]
        with real._Clock() as clock:
            model = ASRRegressor(task="mixture", random_state=seed).fit(x[train], observed.survival(log_t[train], e[train]))
        seconds = {"mixture": clock.cpu}
        with real._Clock() as clock:
            booster, family, scale, single = real._aft(x[fit], t[fit], e[fit], x[held], t[held], e[held], seed)
        seconds["xgboost aft"] = clock.cpu
        scores = {
            "mixture": real._ashlar_scores(model, x[test], log_t[test], e[test], np.log(grid), calibrated=False),
            "xgboost aft": real._aft_scores(booster, family, scale, x[test], log_t[test], e[test], np.log(grid)),
        }
        record = {}
        for name, score in scores.items():
            chi2, p = real.d_calibration(score["own"], e[test])
            record[name] = {
                "nll": score["nll"],
                "c_index": real.concordance(t[test], e[test], score["risk"]),
                "ibs": real.integrated_brier(t[test], e[test], score["curves"], grid),
                "d_chi2": chi2,
                "d_p": p,
                "d_counts": real.d_calibration_counts(score["own"], e[test]),
                "cpu": seconds[name],
            }
        record["xgboost aft"]["chosen"] = f"{family}, scale {scale}"
        record["xgboost aft"]["chosen_cpu"] = single["cpu"]
        record["rows"] = len(test)
        splits[str(seed)] = record
        print(
            f"  split {seed}: "
            + "; ".join(f"{name} NLL {r['nll']:.3f}, C {r['c_index']:.3f}, D-cal chi2 {r['d_chi2']:.1f} (p {r['d_p']:.3f})"
                        for name, r in record.items() if isinstance(r, dict))
            + f"; AFT chose {family}, scale {scale}",
            flush=True,
        )
        if seed != FIGURE_SPLIT:
            continue
        dist = model.predict_dist(x[test])
        rows = len(test)
        mixture = np.stack(
            [np.exp(np.asarray(dist.log_survival(jnp.full((rows,), day, jnp.float32)), np.float64)) for day in np.log(DAYS)],
            axis=1,
        )
        loc = booster.predict(xgboost.DMatrix(x[test]), output_margin=True, iteration_range=(0, booster.best_iteration + 1))
        loc = np.asarray(loc, np.float64)
        aft = real.AFT_FAMILIES[family].sf((np.log(DAYS)[None, :] - loc[:, None]) / scale)
        year = np.exp(np.asarray(dist.log_survival(jnp.full((rows,), np.log(YEAR), jnp.float32)), np.float64))
        order = np.argsort(year, kind="stable")
        picks = order[np.round(np.asarray(PICKS) * (rows - 1)).astype(int)]
        sample = np.random.default_rng(FIGURE_SPLIT).choice(rows, size=min(CROSS_ROWS, rows), replace=False)
        figure = {
            "split": FIGURE_SPLIT,
            "days": DAYS,
            "picks_rule": "test patients at the 10th, 30th, 50th, 70th, and 90th percentiles of the mixture's one-year survival",
            "patients": [
                {
                    "age": float(x[test][row, 0]),
                    "time": float(t[test][row]),
                    "died": bool(e[test][row]),
                    "one_year": {"mixture": float(year[row]), "xgboost aft": float(aft[row, np.searchsorted(DAYS, YEAR)])},
                    "mixture": mixture[row],
                    "xgboost aft": aft[row],
                }
                for row in picks
            ],
            "crossing": {
                "rows": len(sample),
                "gap": CROSS_GAP,
                "mixture": _crossing_share(mixture[sample]),
                "xgboost aft": _crossing_share(aft[sample]),
            },
        }
        print(f"  curves cross for {figure['crossing']['mixture']:.1%} of pairs (mixture), {figure['crossing']['xgboost aft']:.1%} (AFT)", flush=True)
    summary = {}
    for name in ("mixture", "xgboost aft"):
        rows = [split[name] for split in splits.values()]
        summary[name] = {
            "nll_mean": float(np.mean([row["nll"] for row in rows])),
            "nll_sd": float(np.std([row["nll"] for row in rows], ddof=1)),
            "c_index_mean": float(np.mean([row["c_index"] for row in rows])),
            "c_index_sd": float(np.std([row["c_index"] for row in rows], ddof=1)),
            "ibs_mean": float(np.mean([row["ibs"] for row in rows])),
            "ibs_sd": float(np.std([row["ibs"] for row in rows], ddof=1)),
            "d_chi2_mean": float(np.mean([row["d_chi2"] for row in rows])),
            "passing": int(sum(row["d_p"] > 0.05 for row in rows)),
            "splits": len(rows),
        }
        print(f"{name}: {summary[name]}", flush=True)
    data = {
        "table": "SUPPORT",
        "patients": len(t),
        "features": int(x.shape[1]),
        "died": float(e.mean()),
        "summary": summary,
        "splits": splits,
        "figure": figure,
    }
    k.save_json(FIGURE, "survival", data, digits=5)


TICKS = ((7, "1 week"), (30.4, "1 month"), (YEAR, "1 year"), (5 * YEAR, "5 years"))
LETTERS = "ABCDE"


def _ramp(color: str, n: int) -> list[str]:
    """``n`` shades of one colour, light to dark, for patients in order of risk (the riskiest darkest)."""
    light, dark = k.mix(color, k.CARD, 0.55), k.shade(color, -0.3)
    ramp = [k.mix(light, color, 2 * s) if s <= 0.5 else k.mix(color, dark, 2 * s - 1) for s in np.linspace(0, 1, n)]
    return ramp[::-1]


def _spread(values, gap: float) -> np.ndarray:
    """Label heights at least ``gap`` apart, each as near its own value as the gaps allow."""
    order = np.argsort(values)
    placed = np.asarray(values, float)[order].copy()
    for _ in range(50):
        for i in range(1, len(placed)):
            if placed[i] - placed[i - 1] < gap:
                middle = 0.5 * (placed[i] + placed[i - 1])
                placed[i - 1], placed[i] = middle - gap / 2, middle + gap / 2
    out = np.empty_like(placed)
    out[order] = placed
    return out


def _outcome(ax, patient, at: float, size: float, lw: float = 2.1):
    """A death as ×, a patient alive when last seen as ○, on the curve at the time of the event or last contact."""
    if patient["died"]:
        ax.scatter([patient["time"]], [at], marker="x", s=size, color=k.INK, lw=lw + 0.1, zorder=6, clip_on=False)
    else:
        ax.scatter([patient["time"]], [at], s=size, facecolor=k.CARD, edgecolor=k.INK, lw=lw, zorder=7, clip_on=False)


def _curves(ax, days, patients, name: str, colors, label_color: str, title: str):
    log_days = np.log(days)
    ends = []
    for patient, color in zip(patients, colors, strict=True):
        curve = np.asarray(patient[name], float)
        ax.plot(days, curve, color=color, lw=2.4, zorder=3)
        ends.append(curve[-1])
        at = float(np.interp(np.log(patient["time"]), log_days, curve))
        _outcome(ax, patient, at, size=70)
    for y, color, letter in zip(_spread(ends, 0.07), colors, LETTERS, strict=True):
        ax.text(days[-1] * 1.25, y, letter, fontproperties=k.F.value, color=color, ha="left", va="center")
    ax.set_xscale("log")
    ax.set_xlim(3.0, days[-1])
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([day for day, _ in TICKS], [label for _, label in TICKS])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"])
    k.style(ax, frame="lb", grid="y")
    k.axis_names(ax, y="chance of being alive")
    k.panel_title(ax, title, color=label_color)


def draw() -> None:
    data = k.load_json(FIGURE, "survival")
    figure, summary = data["figure"], data["summary"]
    mixture, aft = summary["mixture"], summary["xgboost aft"]
    split = data["splits"][str(figure["split"])]
    days = np.asarray(figure["days"], float)
    patients = figure["patients"]
    fig = k.card(
        "Predicted survival curves for five test patients",
        f"SUPPORT: {k.num(data['patients'])} seriously ill patients, {data['features']} features, "
        f"{k.pct(data['died'])} died during follow-up. Five test patients from one 80/20 split, labelled A to E by "
        "the mixture's predicted chance of surviving a year: × marks a death, ○ a patient alive at the last "
        "follow-up. Ashlar's mixture head fitted to censored log times, and XGBoost's AFT model tuned over 42 settings.",
        height=8.58,
        footer=1,
    )
    axes = k.panels(fig, 1, 3, widths=[1.1, 1.1, 0.9], wspace=0.5, head=0.75, tail=1.1, rpad=0.42)
    _curves(axes[0, 0], days, patients, "mixture", _ramp(k.TEAL, len(patients)), k.TEAL, "Ashlar, mixture head")
    _curves(axes[0, 1], days, patients, "xgboost aft", _ramp(k.CLAY, len(patients)), k.CLAY, "XGBoost AFT, tuned")
    ax = axes[0, 2]
    centres = np.arange(10) / 10 + 0.05
    for offset, name, color in ((-0.021, "mixture", k.TEAL), (0.021, "xgboost aft", k.CLAY)):
        counts = np.asarray(split[name]["d_counts"], float)
        ax.bar(centres + offset, 100 * counts / counts.sum(), width=0.04, color=color, lw=0, zorder=3)
    ax.axhline(10.0, color=k.INK, lw=1.3, linestyle=(0, (4, 3)), zorder=4)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 15.5)
    ax.set_xticks([0.0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.set_yticks([0, 5, 10, 15], ["0", "5%", "10%", "15%"])
    k.style(ax, frame="lb", grid="y")
    k.axis_names(ax, x="predicted survival at the time\nof death or last contact", y="share of test patients")
    k.panel_title(ax, "D-calibration, this split")
    p_mix, p_aft = split["mixture"]["d_p"], split["xgboost aft"]["d_p"]
    k.rich(ax, 0.02, 0.965, [(f"χ² {split['mixture']['d_chi2']:.1f}", k.TEAL, k.F.value),
                             (f", p {p_mix:.2f}" if p_mix >= 0.01 else ", p < 0.01", k.MUTED, k.F.label)], va="top")
    k.rich(ax, 0.02, 0.885, [(f"χ² {split['xgboost aft']['d_chi2']:.1f}", k.CLAY, k.F.value),
                             (f", p {p_aft:.2f}" if p_aft >= 0.01 else ", p < 0.001" if p_aft < 0.001 else ", p < 0.01",
                              k.MUTED, k.F.label)], va="top")
    ax.plot([0.02, 0.085], [0.79, 0.79], transform=ax.transAxes, color=k.INK, lw=1.3, linestyle=(0, (4, 3)))
    ax.text(0.105, 0.79, "10%, if calibrated", transform=ax.transAxes, fontproperties=k.F.small, color=k.INK,
            ha="left", va="center")
    # The five splits' scores are the README's prose, quoted once there from docs/benchmarks.md.
    k.footnote(fig, f"Ashlar's curves cross for {k.pct(figure['crossing']['mixture'])} of pairs of "
                    f"{figure['crossing']['rows']} test patients. AFT's never cross: every patient gets one error "
                    "family with one spread, shifted along log time.")
    k.save(fig, "survival")
    _thumb(days, patients, mixture, aft)


def _thumb(days, patients, mixture: dict, aft: dict) -> None:
    def mark(ax):
        log_days = np.log(days)
        for patient, color in zip(patients, _ramp(k.TEAL, len(patients)), strict=True):
            curve = np.asarray(patient["mixture"], float)
            ax.plot(days, curve, color=color, lw=3.4, zorder=3, solid_capstyle="round")
            _outcome(ax, patient, float(np.interp(np.log(patient["time"]), log_days, curve)), size=120, lw=2.6)
        ax.plot([3.0, days[-1]], [0.0, 0.0], color=k.INK, lw=1.2, solid_capstyle="butt")
        ax.set_xscale("log")
        ax.set_xlim(3.0, days[-1])
        ax.set_ylim(-0.02, 1.04)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticks([])

    k.thumb("survival", "Survival curves",
            [(f"{mixture['passing']} of {mixture['splits']}", k.TEAL), (" splits calibrated, against ", k.MUTED),
             (f"{aft['passing']} of {aft['splits']}", k.CLAY)], mark)


if __name__ == "__main__":
    k.main(compute=compute, draw=draw)
