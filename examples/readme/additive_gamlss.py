"""A model you can read: Ashlar's additive negative binomial beside R's gamlss, term by term, on bike rentals.

The README's additive figure. Two stages, run from the repo root:

    JAX_PLATFORMS=cpu python examples/readme/additive_gamlss.py compute   # needs R with gamlss: the Mac
    python examples/readme/additive_gamlss.py draw                        # needs Inter: the cloud

``compute`` is the ``terms`` stage of ``examples/benchmark_additive.py``: it
refits both models on one of the benchmark's splits and writes every term of
the log mean and the dispersion, on a grid of its columns and centred on the
training rows, with the benchmark's saved scores and seconds for every split,
to ``examples/readme/artifacts/additive-gamlss/terms.json``. Option ``seed=N``
picks the split; by default it is the one whose saved gamlss pair fit was fastest.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

FIGURE = "additive-gamlss"
ARTIFACTS = HERE / "artifacts" / FIGURE


def compute(seed=None):
    import benchmark_additive

    benchmark_additive.run_terms(None if seed is None else int(seed), ARTIFACTS)


# ---------------------------------------------------------------- draw

TEMP_C, HUM_PCT = 41.0, 100.0  # the UCI table's temp is °C / 41 and hum is % / 100
DAYS = ((1, "working days"), (0, "other days"))
# The saved results keep each fit's seconds but not whether it converged. docs/benchmarks.md ("They take a fraction
# of the time") records that the slowest gamlss fit, split 1's with the pair (2,691 seconds), stopped at its limit of
# 200 cycles without converging. The figure draws it as a ring and leaves it out of the title's minutes.
UNCONVERGED = {"gamlss_nbi_pair": ("1",)}


def softplus(z):
    return np.logaddexp(0.0, z)


def to_log_dispersion(raw, base):
    """Ashlar's raw dispersion term on gamlss's scale: the change in log dispersion it makes at the average row,
    the dispersion being softplus(raw) + 1e-4."""
    return np.log(softplus(base + np.asarray(raw, dtype=np.float64)) + 1e-4) - np.log(softplus(base) + 1e-4)


def centred(values, weights=None):
    values = np.asarray(values, dtype=np.float64)
    return values - np.average(values, weights=weights)


def hour_profiles(model: dict, output: str, transform=None) -> dict:
    """For working days and for other days: the hour's shape, every term the hour or the working day enters
    summed (hr, workingday, hr:workingday), on 0 to 23 and centred on its 24 hours. Each model splits the level
    between those terms in its own way; the sum is what both say."""
    terms = model["terms"]

    def table(name):
        term = terms.get(name)
        if term is None:
            return {}
        return {tuple(x) if isinstance(x, list) else x: value for x, value in zip(term["x"], term[output])}

    hour, day, pair = table("hr"), table("workingday"), table("hr:workingday")
    out = {}
    for flag, _label in DAYS:
        raw = np.asarray([hour[h] + day.get(flag, 0.0) + pair.get((h, flag), 0.0) for h in range(24)])
        out[flag] = centred(transform(raw) if transform else raw)
    return out


def column_curve(model: dict, name: str, output: str, transform=None):
    """One term against its column, centred on the training rows (re-centred after ``transform``)."""
    term = model["terms"][name]
    x = np.asarray(term["x"], dtype=np.float64)
    rows = np.asarray(term["rows"], dtype=np.float64)
    y = np.asarray(term[output], dtype=np.float64)
    return x, centred(transform(y) if transform else y, rows), rows


def benchmark_stats(benchmark: dict, model: str) -> dict:
    runs = benchmark.get(model, {})
    if not runs:
        return {}
    pick = lambda key: np.asarray([run[key] for run in runs.values()], dtype=np.float64)  # noqa: E731
    seconds = pick("seconds")
    converged = np.asarray([split not in UNCONVERGED.get(model, ()) for split in runs])
    if not converged.all():  # the record names the slowest fit; check that it still is
        assert seconds[~converged].min() > seconds[converged].max(), (model, seconds)
    return {"crps": float(pick("crps").mean()), "nll": float(pick("nll").mean()), "seconds": seconds,
            "converged": converged, "splits": len(runs)}


def draw():
    import matplotlib.patheffects as path_effects
    from matplotlib.ticker import FixedLocator, NullLocator

    import style as k

    data = k.load_json(FIGURE, "terms")
    paired = "gamlss_nbi_pair" in data["gamlss"]
    ours = data["ashlar"]["ashlar_additive_pair" if paired else "ashlar_additive"]
    theirs = data["gamlss"]["gamlss_nbi_pair" if paired else "gamlss_nbi"]
    bench = {name: benchmark_stats(data["benchmark"], name) for name in
             ("ashlar_additive", "gamlss_nbi", "ashlar_additive_pair", "gamlss_nbi_pair", "ashlar_blackbox")}
    fast = bench["ashlar_additive_pair" if paired else "ashlar_additive"]
    slow = bench["gamlss_nbi_pair" if paired else "gamlss_nbi"]
    base = float(ours["intercept"][1])
    ours_dispersion = lambda raw: to_log_dispersion(raw, base)  # noqa: E731
    halo = [path_effects.withStroke(linewidth=4.0, foreground=k.CARD)]
    rows_total = data["train_rows"] + next(iter(data["benchmark"]["ashlar_additive"].values()))["rows"]
    splits = bench["ashlar_additive"]["splits"]

    k.setup()
    done = slow["seconds"][slow["converged"]]  # the title's range: the gamlss fits that converged
    low, high = (float(value) for value in np.round(np.asarray([done.min(), done.max()]) / 60))
    span = f"{low:.0f} to {high:.0f} minutes" if high >= 2 else f"{done.min():.0f} to {done.max():.0f} seconds"
    fig = k.card(
        f"The same shapes as R's gamlss, in {np.mean(fast['seconds']):.0f} seconds instead of {span}.",
        f"Hourly bike rentals, {rows_total:,} hours: a negative binomial whose log mean and log dispersion are each "
        "a sum of one shape per column" + (", plus the hour × working day pair" if paired else "") +
        f". Ashlar's additive backbone against gamlss's NBI with penalized splines, both fitted to the same "
        f"{data['train_rows']:,} hours, every curve centred on its mean. Right: {WORDS.get(splits, splits)} splits, "
        "Ashlar on every core of a laptop and gamlss on one thread.",
        height=9.0, footer=1,
    )
    cols = (0.0, 0.245, 0.49, 0.735)
    grid = [[k.panel(fig, rect=(cols[c], 0.5 * (1 - r), cols[c + 1] - 0.012, 0.5 * (2 - r)),
                     head=0.72 if r == 0 else 0.42, tail=0.18 if r == 0 else 0.78, lpad=0.62, rpad=0.1, grid="y")
             for c in range(3)] for r in range(2)]

    # The hour, by working day: both tools, both day types.
    light = {"TEAL": k.mix(k.TEAL, k.CARD, 0.55), "CLAY": k.mix(k.CLAY, k.CARD, 0.5)}
    for r, (output_ours, output_theirs, transform) in enumerate((("log_mean", "log_mean", None),
                                                                 ("dispersion", "log_dispersion", ours_dispersion))):
        ax = grid[r][0]
        mine = hour_profiles(ours, output_ours, transform)
        other = hour_profiles(theirs, output_theirs)
        hours = np.arange(24)
        for flag, _label in DAYS:
            strong = flag == 1
            ax.plot(hours, mine[flag], color=k.TEAL if strong else light["TEAL"], lw=2.6, zorder=3 if strong else 2)
            ax.plot(hours, other[flag], color=k.CLAY_DARK if strong else k.CLAY, lw=1.7, ls=(0, (3.2, 2.2)),
                    zorder=4 if strong else 3.5, alpha=1.0 if strong else 0.8)
        ax.set_xlim(-0.5, 23.5)
        ax.xaxis.set_major_locator(FixedLocator([0, 6, 12, 18, 23]))
        if r == 0:
            bottom = min(mine[1].min(), mine[0].min(), other[1].min(), other[0].min())
            reach = max(mine[1].max(), mine[0].max(), other[1].max(), other[0].max()) - bottom
            k.value(ax, 23, bottom + 0.2 * reach, "working days", k.TEAL, ha="right", va="center", size=12)
            k.value(ax, 23, bottom + 0.06 * reach, "other days", light["TEAL"], ha="right", va="center", size=12)

    # Temperature (with the felt temperature, which moves with it) and humidity.
    for c, (name, scale, label) in enumerate((("temp+atemp", TEMP_C, "temperature, °C"), ("hum", HUM_PCT, "humidity, %")),
                                             start=1):
        for r, (output_ours, output_theirs, transform) in enumerate((("log_mean", "log_mean", None),
                                                                     ("dispersion", "log_dispersion", ours_dispersion))):
            ax = grid[r][c]
            x, mine, rows = column_curve(ours, name, output_ours, transform)
            x2, other, _ = column_curve(theirs, name, output_theirs)
            ax.plot(x * scale, mine, color=k.TEAL, lw=2.6, zorder=3)
            ax.plot(x2 * scale, other, color=k.CLAY_DARK, lw=1.7, ls=(0, (3.2, 2.2)), zorder=4)
            low, high = min(mine.min(), other.min()), max(mine.max(), other.max())
            pad = 0.12 * (high - low)
            floor = low - pad - 0.22 * (high - low)
            ax.bar(x * scale, rows / rows.max() * 0.18 * (high - low), bottom=floor, width=np.median(np.diff(x)) * scale,
                   color=k.DATA, lw=0, zorder=1, align="center")
            ax.set_ylim(floor, high + pad)
            ax.set_xlim(0, scale)
            ax.xaxis.set_major_locator(FixedLocator(np.linspace(0, 40, 5) if scale == TEMP_C else np.linspace(0, 100, 5)))
            if r == 1:
                k.axis_names(ax, x=label)

    for r in range(2):
        for c in range(3):
            ax = grid[r][c]
            k.style(ax, frame="lb", grid="y")
            if r == 0:
                ax.tick_params(labelbottom=False)
            ax.yaxis.set_minor_locator(NullLocator())
    k.axis_names(grid[1][0], x="hour of the day")
    k.axis_names(grid[0][0], y="effect on the log mean")
    k.axis_names(grid[1][0], y="effect on the log dispersion")
    for c, text in enumerate(("Hour, by day type", "Temperature", "Humidity")):
        k.panel_title(grid[0][c], text, raise_by=0.36)

    # Right: seconds per fit on a log axis, and the test CRPS, three splits.
    speed = k.panel(fig, rect=(0.765, 0.5, 1.0, 1.0), head=0.72, tail=0.62, lpad=0.05, rpad=0.05, grid="x")
    score = k.panel(fig, rect=(0.765, 0.0, 1.0, 0.5), head=0.42, tail=0.78, lpad=0.05, rpad=0.05, grid="x")
    variants = [("additive", "ashlar_additive", "gamlss_nbi"), ("with the pair", "ashlar_additive_pair", "gamlss_nbi_pair")]
    for row, (label, mine, other) in enumerate(variants):
        y = 1 - row
        quick, slow_ = bench[mine], bench[other]
        if quick and slow_:
            speed.plot([quick["seconds"].max(), slow_["seconds"].min()], [y, y], color=k.HAIR, lw=2.0, zorder=1,
                       solid_capstyle="butt")
        for cell, colour in ((quick, k.TEAL), (slow_, k.CLAY)):
            if not cell:
                continue
            ok = cell["converged"]
            seconds, stopped = cell["seconds"][ok], cell["seconds"][~ok]
            speed.scatter(seconds, np.full(len(seconds), y), s=46, color=colour, edgecolor=k.CARD, lw=1.0, zorder=3)
            if colour == k.TEAL:
                k.value(speed, seconds.min() / 1.45, y, k.num(np.mean(seconds)), colour, ha="right", va="center",
                        size=12.5).set_path_effects(halo)
            else:
                text_ = k.num(seconds.min()) + "–" + k.num(seconds.max())
                k.value(speed, np.sqrt(seconds.min() * seconds.max()), y + 0.2, text_, colour, ha="center",
                        va="bottom", size=12.5).set_path_effects(halo)
            if len(stopped):  # a fit that hit its cycle limit: a ring, named above it
                speed.scatter(stopped, np.full(len(stopped), y), s=40, facecolor=k.CARD, edgecolor=colour, lw=1.6,
                              zorder=3)
                speed.text(stopped.max(), y + 0.2, "did not\nconverge", fontproperties=k.F.small, color=k.MUTED,
                           ha="center", va="bottom", linespacing=1.0)
        for name, colour, dy in ((mine, k.TEAL, 0.12), (other, k.CLAY, -0.12)):
            cell = bench[name]
            if not cell:
                continue
            score.scatter([cell["crps"]], [y + dy], s=80, color=colour, edgecolor=k.CARD, lw=1.2, zorder=3)
            k.value(score, cell["crps"] + (1.6 if colour == k.TEAL else -1.6), y + dy, f"{cell['crps']:.2f}", colour,
                    ha="left" if colour == k.TEAL else "right", va="center", size=12.5).set_path_effects(halo)
        for ax in (speed, score):
            ax.text(0.0, y + 0.5, label, transform=ax.get_yaxis_transform(), fontproperties=k.F.small, color=k.MUTED,
                    ha="left", va="center")
    net = bench["ashlar_blackbox"]
    if net:
        score.scatter([net["crps"]], [-1], s=80, facecolor=k.CARD, edgecolor=k.TEAL, lw=1.8, zorder=3)
        k.value(score, net["crps"] + 1.6, -1, f"{net['crps']:.2f}", k.TEAL, ha="left", va="center",
                size=12.5).set_path_effects(halo)
        score.text(0.0, -0.6, "residual network, not additive", transform=score.get_yaxis_transform(),
                   fontproperties=k.F.small, color=k.MUTED, ha="left", va="center")
    speed.set_xscale("log")
    speed.set_xlim(2.5, 6000)
    speed.xaxis.set_major_locator(FixedLocator([10, 100, 1000]))
    speed.set_xticklabels(["10", "100", "1,000"])
    speed.xaxis.set_minor_locator(NullLocator())
    speed.set_ylim(-0.45, 1.75)
    score.set_xlim(0, 50)
    score.set_ylim(-1.45, 1.6)
    for ax in (speed, score):
        k.style(ax, frame="b", grid="x")
        ax.set_yticks([])
    k.axis_names(speed, x="seconds per fit, each split")
    k.axis_names(score, x="test CRPS, lower is better")
    k.panel_title(speed, "Speed and score", raise_by=0.36)

    k.legend_row(fig, [
        ("line", k.TEAL, "Ashlar, additive"),
        ("dashed", k.CLAY_DARK, "R's gamlss, penalized splines"),
        ("patch", k.DATA, "training hours"),
    ], line=0)
    k.save(fig, FIGURE)

    hours = np.arange(24)
    mine, other = hour_profiles(ours, "log_mean"), hour_profiles(theirs, "log_mean")

    def mark(ax):
        for flag, _label in DAYS:
            strong = flag == 1
            ax.plot(hours, mine[flag], color=k.TEAL if strong else light["TEAL"], lw=3.4, zorder=3 if strong else 2)
            ax.plot(hours, other[flag], color=k.CLAY_DARK if strong else k.CLAY, lw=2.0, ls=(0, (3.2, 2.2)),
                    zorder=4 if strong else 3.5, alpha=1.0 if strong else 0.8)
        ax.set_xlim(-0.5, 23.5)

    k.thumb("additive", "Readable shapes", [
        (f"{np.mean(fast['seconds']):.0f} s", k.TEAL), (" a fit, against ", k.MUTED),
        (span.replace(" minutes", " min").replace(" seconds", " s"), k.CLAY), (" for gamlss", k.MUTED),
    ], mark)


WORDS = {2: "two", 3: "three", 4: "four", 5: "five"}


def main(argv):
    stages = [arg for arg in argv if "=" not in arg] or ["draw"]
    options = dict(arg.split("=", 1) for arg in argv if "=" in arg)
    for stage in stages:
        if stage == "compute":
            compute(options.get("seed"))
        elif stage == "draw":
            draw()
        else:
            raise SystemExit(f"unknown stage {stage!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
