"""A shape for every target: held-out slices with Ashlar's head and the strongest tree distribution.

Two stages, run from the repository root:

    JAX_PLATFORMS=cpu python examples/readme/shapes.py compute   # public data, a few minutes on a laptop CPU
    python examples/readme/shapes.py draw                        # seconds

``compute`` refits split 0 of three runs in ``examples/benchmark_shapes.py`` (the
same rows, models, and settings) and writes, for each tile, the held-out values
of one named slice and the share of them each model expects in every bin, to
``examples/readme/artifacts/shapes/<tile>.json``. A model's share of a bin is
its CDF across the bin averaged over the slice's rows, so a count, a spike, and
the atom at a cap are drawn on the same footing as the data. It also copies the
headline numbers, means over every split, from the benchmark's results
(``/tmp/ashlar-bench/shapes-*.json``, written by ``python
examples/benchmark_shapes.py``) into ``numbers.json``. ``draw`` reads only those
files. ``compute count`` (or ``spike``, ``ceiling``, ``numbers``) redoes one.

The tiles, each against the best tree distribution on its table:

* count: bike rentals per hour, the held-out hours LightGBM's Poisson puts at
  350 to 450 rentals. ``negative_binomial`` against LightGBM's Poisson mean with
  one negative binomial dispersion fitted on its validation slice; the Poisson
  alone is the second line.
* spike: freMTPL2 cost per claim, every held-out claim. ``mixture`` on the log
  cost against CatBoost ``RMSEWithUncertainty`` on the log cost.
* ceiling: California median home values, capped at $500,001, in the held-out
  block groups with a median income of $70,000 or more. ``mixture`` against
  LightGBM's mean with a constant normal, which beats CatBoost
  ``RMSEWithUncertainty`` on CRPS on this table.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402

import style as k  # noqa: E402

FIGURE = "shapes"
SEED = 0  # every tile is split 0 of its benchmark
COUNT_WINDOW = (350.0, 450.0)  # the held-out hours LightGBM's Poisson puts in this range


# ---------------------------------------------------------------- compute helpers


def _ashlar_cdf(dist):
    """``cdf(edges)`` for a fitted head: (edges, rows), in chunks of edges."""
    import jax.numpy as jnp

    def cdf(edges):
        out = []
        for start in range(0, len(edges), 64):
            block = jnp.asarray(np.asarray(edges[start : start + 64], np.float32)[:, None])
            out.append(np.asarray(dist.cdf(block), np.float64))
        return np.concatenate(out)

    return cdf


def _shares(cdf, edges) -> np.ndarray:
    """The share of the slice a model expects in each bin: its CDF across the bin, averaged over rows."""
    return np.diff(cdf(np.asarray(edges, np.float64)), axis=0).mean(axis=1)


def _data_shares(values, edges) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    return counts / max(len(values), 1)


def _tile(name, *, slice_, values, edges, ashlar, baseline, **extra):
    payload = {
        "slice": slice_,
        "rows": int(len(values)),
        "edges": np.asarray(edges, np.float64),
        "data": _data_shares(values, edges),
        "ashlar": ashlar,
        "baseline": baseline,
        **extra,
    }
    k.save_json(FIGURE, name, payload)


# ---------------------------------------------------------------- tiles


def tile_count():
    """Bike rentals per hour: the held-out hours LightGBM's Poisson puts at 350 to 450 rentals."""
    from scipy import stats
    from sklearn.model_selection import train_test_split

    import benchmark_readme as bench
    from benchmark_shapes import _nb_dispersion
    from benchmark_uncertainty import bike

    from ashlar import ASRRegressor

    x, y = bike("cnt")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED)
    model = ASRRegressor(task="negative_binomial", random_state=SEED).fit(x_train, y_train)
    gbm, frame = bench._fit_gbt("lightgbm", "regression", True, SEED, x_train, y_train, objective="poisson")
    rate = np.maximum(gbm.predict(bench._align_categories(frame, x_test)), 1e-9)
    # The tree user's fix for a Poisson that is too narrow: one dispersion, fitted on the validation slice.
    _fit_x, val_x, _fit_y, val_y = bench._gbt_split(frame, y_train, SEED)
    dispersion = _nb_dispersion(val_y, gbm.predict(val_x))
    size = 1.0 / dispersion
    keep = (rate >= COUNT_WINDOW[0]) & (rate <= COUNT_WINDOW[1])
    edges = np.arange(99.5, 820.0, 20.0)  # bins of twenty rentals; a count's CDF steps at whole numbers
    dist = model.predict_dist(x_test[keep])
    mean = rate[keep][None, :]
    _tile(
        "count",
        slice_=f"held-out hours LightGBM's Poisson puts at {COUNT_WINDOW[0]:.0f} to {COUNT_WINDOW[1]:.0f} rentals",
        values=y_test[keep],
        edges=edges,
        ashlar=_shares(_ashlar_cdf(dist), edges),
        baseline=_shares(lambda e: stats.nbinom.cdf(np.floor(e)[:, None], size, size / (size + mean)), edges),
        second=_shares(lambda e: stats.poisson.cdf(np.floor(e)[:, None], mean), edges),
        dispersion=dispersion,
    )


def tile_spike():
    """freMTPL2 claims: the log cost per claim of every held-out claim."""
    from scipy import stats

    import benchmark_readme as bench
    from benchmark_shapes import catboost_normal, claims, tail_split

    from ashlar import ASRRegressor

    frame, columns = claims()
    train, test = tail_split(frame, SEED)
    x_train = bench._strings(train[columns].reset_index(drop=True))
    x_test = bench._strings(test[columns].reset_index(drop=True))
    cost_train, cost_test = train["S"].to_numpy(), test["S"].to_numpy()
    model = ASRRegressor(task="mixture", random_state=SEED).fit(x_train, np.log(cost_train))
    dist = model.predict_dist(x_test)
    loc, scale = catboost_normal(x_train, np.log(cost_train), x_test, SEED)
    # Bins of equal width on the log axis, one of them exactly the band of the four most common amounts.
    values, counts = np.unique(np.round(frame.loc[frame["N"] > 0, "S"].to_numpy(), 2), return_counts=True)
    common = np.sort(values[np.argsort(-counts)[:4]])
    low, high = np.log(common[0] * 0.99), np.log(common[-1] * 1.01)
    width = high - low
    first = np.floor((np.log(100.0) - low) / width)
    last = np.ceil((np.log(20_000.0) - low) / width)
    edges = low + width * np.arange(first, last + 1)
    _tile(
        "spike",
        slice_="every held-out claim",
        values=np.log(cost_test),
        edges=edges,
        ashlar=_shares(_ashlar_cdf(dist), edges),
        baseline=_shares(lambda e: stats.norm.cdf(e[:, None], loc[None, :], scale[None, :]), edges),
        common=common,
        band=[low, high],
    )


def tile_ceiling():
    """California block groups with a median income of $70k or more, where the $500k cap binds."""
    from scipy import stats
    from sklearn.model_selection import train_test_split

    import benchmark_readme as bench
    from benchmark_shapes import RICH_INCOME, lightgbm_normal

    from ashlar import ASRRegressor

    x, y = bench.california()
    index_train, index_test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=SEED)
    x_train, x_test = x.iloc[index_train], x.iloc[index_test]
    y_train, y_test = y[index_train], y[index_test]
    keep = (x_test["median_income"] >= RICH_INCOME).to_numpy()
    model = ASRRegressor(task="mixture", random_state=SEED).fit(x_train, y_train)
    dist = model.predict_dist(x_test[keep])
    loc, scale = lightgbm_normal(x_train, y_train, x_test, SEED)
    edges = 0.9 + 0.2 * np.arange(29)  # $20k bins; the cap, 5.00001, sits in the middle of [4.9, 5.1]
    _tile(
        "ceiling",
        slice_="held-out block groups with a median income of $70,000 or more",
        values=y_test[keep],
        edges=edges,
        ashlar=_shares(_ashlar_cdf(dist), edges),
        baseline=_shares(lambda e: stats.norm.cdf(e[:, None], loc[keep][None, :], scale[keep][None, :]), edges),
        cap=float(np.max(y)),
        at_cap=float(np.mean(y_test[keep] >= np.max(y))),
    )


def numbers():
    """The headline numbers, means over every split of ``benchmark_shapes.py``, into ``numbers.json``."""
    from benchmark_shapes import load

    results = load()

    def means(runs, models, keys):
        return {
            model: {key: float(np.mean([run[model][key] for run in runs.values()])) for key in keys}
            for model in models
        }

    count = {}
    for name in ("bike_count", "taxi_pickups"):
        runs = results["counts"][name]["runs"]
        count[name] = {
            "splits": len(runs),
            "dispersion": float(np.mean([run["lightgbm_poisson_nb"]["dispersion"] for run in runs.values()])),
            **means(runs, ("negative_binomial", "lightgbm_poisson_nb", "lightgbm_poisson"),
                    ("crps", "nll", "coverage80", "width80")),
        }
    tail, ceiling = results["tail"], results["ceiling"]
    fare, tips, diamonds = results["spike"], results["zero"], results["skew"]
    k.save_json(FIGURE, "numbers", {
        "count": count,
        "spike": {
            "splits": len(tail["runs"]),
            **tail["table"],
            **means(tail["runs"], ("mixture", "gaussian", "catboost_uncertainty", "lightgbm_lognormal", "lightgbm_gamma"),
                    ("nll", "crps", "coverage80", "width80", "band_mass")),
        },
        "ceiling": {
            "splits": len(ceiling["runs"]),
            **ceiling["table"],
            **means(ceiling["runs"], ("mixture", "flow", "histogram", "lightgbm_normal", "catboost_uncertainty"),
                    ("crps", "nll", "rich_crps", "rich_brier_cap", "rich_past_cap", "past_cap")),
        },
        # The runs behind the two tiles that were dropped, and the table only the docs keep, for the copy.
        "fare": {
            "splits": len(fare["runs"]),
            **fare["table"],
            **means(fare["runs"], ("histogram", "mixture", "flow", "lightgbm_normal", "catboost_uncertainty"),
                    ("crps", "within_050", "flat_crps")),
        },
        "tips": {
            "splits": len(tips["runs"]),
            **tips["table"],
            **means(tips["runs"], ("hurdle_gamma", "lightgbm_hurdle_gamma", "lightgbm_normal"), ("crps", "brier_zero")),
        },
        "diamonds": {
            "splits": len(diamonds["runs"]),
            **means(diamonds["runs"], ("mixture", "catboost_uncertainty", "lightgbm_normal"), ("crps", "below_zero")),
        },
    }, digits=7)


TILES = {"count": tile_count, "spike": tile_spike, "ceiling": tile_ceiling, "numbers": numbers}


def compute():
    for name in sys.argv[2:] or list(TILES):
        TILES[name]()


# ---------------------------------------------------------------- draw


def _grid(tile, log=False):
    """Left edges, right edges, and centres of the tile's bins (in cost, not log cost, when ``log``)."""
    edges = np.asarray(tile["edges"], float)
    if log:
        edges = np.exp(edges)
        return edges[:-1], edges[1:], np.sqrt(edges[:-1] * edges[1:])
    return edges[:-1], edges[1:], 0.5 * (edges[:-1] + edges[1:])


def _bars(ax, left, right, data, gap=0.1):
    ax.bar(left + (right - left) * gap / 2, data, width=(right - left) * (1 - gap), align="edge", color=k.DATA,
           lw=0, zorder=1)


def _clay(ax, x, y, lw=2.3):
    ax.plot(x, y, color=k.CLAY, lw=lw, linestyle=(0, (4.2, 2.4)), zorder=4)


def _teal(ax, x, y, lw=2.8):
    ax.plot(x, y, color=k.TEAL, lw=lw, zorder=5)


def _label(ax, x, y, text, color, *, ha="left", va="center"):
    """A direct label in data coordinates, in place of a legend."""
    return ax.text(x, y, text, fontproperties=k.font("Medium", 13), color=color, ha=ha, va=va, zorder=9,
                   clip_on=False, linespacing=1.25)


def _header(ax, name, head, about, pair, metric):
    """Three lines over a tile: its name and Ashlar's head, the data, and one number against the baseline."""
    ax.text(0.0, 1.0, name, transform=k._above(ax, 0.98), fontproperties=k.font("Bold", 20), color=k.INK,
            ha="left", va="bottom", clip_on=False)
    ax.text(1.0, 1.0, head, transform=k._above(ax, 0.99), fontproperties=k.font("Medium", 14, mono=True),
            color=k.TEAL, ha="right", va="bottom", clip_on=False)
    ax.text(0.0, 1.0, about, transform=k._above(ax, 0.66), fontproperties=k.font("Regular", 13), color=k.MUTED,
            ha="left", va="bottom", clip_on=False)
    big = k.font("SemiBold", 20)
    k.rich(ax, 0.0, 1.0, [(pair[0], k.TEAL, big), (" against ", k.MUTED, k.font("Regular", 14)),
                          (pair[1], k.CLAY, big), (f" {metric}", k.MUTED, k.font("Regular", 13))],
           transform=k._above(ax, 0.2), ha="left", va="baseline")


def _finish(ax, xlim, ticks, labels, top, note):
    ax.set_xlim(*xlim)
    ax.set_ylim(0, top)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.minorticks_off()
    k.style(ax, frame="b")
    ax.set_yticks([])
    ax.set_xlabel(note, fontproperties=k.F.label, color=k.MUTED, labelpad=8)


def draw_count(ax, tile, n):
    left, right, x = _grid(tile)
    data, ashlar, baseline, second = (np.asarray(tile[key], float) for key in ("data", "ashlar", "baseline", "second"))
    _bars(ax, left, right, data)
    ax.plot(x, second, color=k.SLATE, lw=2.2, linestyle=(0, (1, 2.2)), zorder=3)
    _clay(ax, x, baseline)
    _teal(ax, x, ashlar)
    top = max(data.max(), ashlar.max(), baseline.max(), second.max()) * 1.08
    _finish(ax, (150, 760), [200, 400, 600], ["200", "400", "600 rentals"], top,
            f"{tile['rows']} hours LightGBM puts at {COUNT_WINDOW[0]:.0f}–{COUNT_WINDOW[1]:.0f}")
    _label(ax, x[np.argmax(second)] + 24, second.max() * 0.96, "LightGBM Poisson", k.SLATE)
    far = int(np.searchsorted(x, 540))
    _label(ax, x[far], baseline[far] + top * 0.12, "+ one dispersion", k.CLAY)
    bike = n["count"]["bike_count"]
    _header(ax, "A count", "negative_binomial", "Bike rentals per hour",
            (k.num(bike["negative_binomial"]["width80"]), k.num(bike["lightgbm_poisson_nb"]["width80"])),
            "rentals wide, 80% interval")


def draw_spike(ax, tile, n):
    left, right, x = _grid(tile, log=True)
    data, ashlar, baseline = (np.asarray(tile[key], float) for key in ("data", "ashlar", "baseline"))
    ax.set_xscale("log")
    _bars(ax, left, right, data, gap=0.0)
    _clay(ax, x, baseline)
    _teal(ax, x, ashlar)
    top = data.max() * 1.1
    _finish(ax, (100, 20_000), [100, 1_000, 10_000], ["€100", "€1,000", "€10,000"], top,
            f"{k.num(tile['rows'])} held-out claims, log scale")
    band = int(np.argmax(data))
    _label(ax, left[band] * 0.9, data[band] * 0.96, f"{k.pct(data[band])} of claims", k.DATA_DARK, ha="right")
    _label(ax, right[band] * 1.6, top * 0.17, "CatBoost", k.CLAY)
    s = n["spike"]
    _header(ax, "A spike", "mixture", "Cost per motor claim, freMTPL2",
            (k.pct(s["mixture"]["band_mass"]), k.pct(s["catboost_uncertainty"]["band_mass"])),
            "at four fixed amounts")


def draw_ceiling(ax, tile, n):
    left, right, x = _grid(tile)
    data, ashlar, baseline = (np.asarray(tile[key], float) for key in ("data", "ashlar", "baseline"))
    at = int(np.searchsorted(right, tile["cap"]))  # the bin that holds the cap
    others = np.concatenate([np.delete(series, at) for series in (data, ashlar, baseline)])
    top = others.max() * 1.6  # the cap's bar runs off the top; its share is printed beside it
    _bars(ax, left, right, data)
    past = x > right[at]
    fill_x = np.concatenate([[right[at]], x[past]])
    fill_y = np.concatenate([[np.interp(right[at], x, baseline)], baseline[past]])
    ax.fill_between(fill_x, 0, fill_y, color=k.CLAY_WASH, lw=0, zorder=2)
    _clay(ax, x, baseline)
    _teal(ax, x, ashlar)
    _finish(ax, (1.5, 6.3), [2, 3, 4, 5, 6], ["$200k", "$300k", "$400k", "$500k", "$600k"], top,
            f"{tile['rows']} block groups earning $70k or more")
    _label(ax, left[at] - 0.22, top * 0.93, f"{k.pct(tile['at_cap'])} at the cap", k.DATA_DARK, ha="right")
    _label(ax, right[at] + 0.1, top * 0.66, "LightGBM\n+ normal", k.CLAY)
    c = n["ceiling"]
    _header(ax, "A ceiling", "mixture", "California home values, capped",
            (k.pct(c["mixture"]["rich_past_cap"], 1), k.pct(c["lightgbm_normal"]["rich_past_cap"], 1)),
            "past the cap")


ORDER = (("count", draw_count), ("spike", draw_spike), ("ceiling", draw_ceiling))


def draw():
    n = k.load_json(FIGURE, "numbers")
    fig = k.card(
        "Heads for a count, for point masses, and for a capped target",
        "Held-out data in grey, Ashlar's head in teal, and the best tree-based distribution on each table in dashed "
        "clay. Each line is the share of the data shown that a model expects in each bin, from split 0; the numbers "
        "above each panel are means over three splits.",
        height=8.0,
        footer=3,
    )
    axes = k.panels(fig, 1, 3, wspace=0.62, head=1.32, tail=0.8, lpad=0.04, rpad=0.04, frame="b")
    for ax, (name, fn) in zip(axes.ravel(), ORDER):
        fn(ax, k.load_json(FIGURE, name), n)
    bike, s, c = n["count"]["bike_count"], n["spike"], n["ceiling"]
    k.footnote(
        fig,
        f"Every held-out row, three splits. Counts: the 80% interval holds "
        f"{k.pct(bike['negative_binomial']['coverage80'])} against {k.pct(bike['lightgbm_poisson_nb']['coverage80'])} "
        f"for LightGBM's Poisson mean with one dispersion fitted on validation "
        f"({k.pct(bike['lightgbm_poisson']['coverage80'])} for the Poisson alone); CRPS "
        f"{bike['negative_binomial']['crps']:.1f} against {bike['lightgbm_poisson_nb']['crps']:.1f}. Claims: CRPS "
        f"of the log cost {s['mixture']['crps']:.3f} against {s['catboost_uncertainty']['crps']:.3f} for CatBoost "
        f"RMSEWithUncertainty. Homes: CRPS {c['mixture']['crps']:.3f} against {c['lightgbm_normal']['crps']:.3f} "
        f"for LightGBM's mean with a constant normal, the better of the two trees here.",
    )
    k.save(fig, FIGURE)
    draw_thumb(n)


def draw_thumb(n):
    tile = k.load_json(FIGURE, "spike")
    s = n["spike"]

    def mark(ax):
        left, right, x = _grid(tile, log=True)
        data, ashlar, baseline = (np.asarray(tile[key], float) for key in ("data", "ashlar", "baseline"))
        ax.set_xscale("log")
        _bars(ax, left, right, data, gap=0.0)
        _clay(ax, x, baseline, lw=3.0)
        _teal(ax, x, ashlar, lw=3.4)
        ax.set_xlim(150, 15_000)
        ax.set_ylim(0, data.max() * 1.04)
        ax.minorticks_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(k.INK)

    k.thumb(
        "shapes",
        "Every target's shape",
        [(k.pct(s["mixture"]["band_mass"]), k.TEAL), (" at fixed amounts, against ", k.MUTED),
         (k.pct(s["catboost_uncertainty"]["band_mass"]), k.CLAY)],
        mark,
    )


if __name__ == "__main__":
    k.main(compute=compute, draw=draw)
