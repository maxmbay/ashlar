"""The sold-out figure: one FreshRetailNet-50K series whose shelf ran out, and what each fit learned of its demand.

``compute`` runs on Max's Mac, where FreshRetailNet-50K is cached in ``/tmp/ashlar-bench`` (about 15 minutes). It
follows seed 0 of ``benchmark_observed_real.py stockouts``: the days that recorded their demand, with a simulated
stock, so a day sold out when its demand reached the stock; days 0 to 59 train, 60 to 69 calibrate, 70 to 89 test.
It fits the ``mixture`` head three ways, on ``observed.censored(sales, sold_out)``, on sales as if they were
demand, and on the true demand (the oracle); LightGBM's quantile regression on sales at each level; and XGBoost's
AFT model on the censored sales, chosen as the benchmark chooses it. It writes to
``examples/readme/artifacts/sold-out/``:

- ``series.json``: candidate series chosen by a fixed rule (``candidates``), each with every one of its days (the
  split, sales, stock, whether it sold out, and the true demand) and each fit's quantiles of demand that day;
- ``check.json``: this seed's test scores of each fit, beside the benchmark's own scores for the same seed;
- ``stockouts.json``: a copy of the benchmark's results over its seeds, which the order-cost bars read.

``draw`` runs in the cloud container, which has the fonts, and writes ``docs/images/readme/sold-out.png``.

    JAX_PLATFORMS=cpu python examples/readme/sold_out.py compute
    python examples/readme/sold_out.py draw
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

ARTIFACTS = HERE / "artifacts" / "sold-out"
SEED = 0
LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)
CANDIDATES = 8
TYPICAL_SOLD = (4, 8)  # sold-out test days of 20 around the benchmark's 31%
FITS = ("censored", "sales", "oracle", "lightgbm", "xgboost aft")


def _number(value) -> float | None:
    """Four significant digits keep the artifacts small; ``None`` for a value that is not finite."""
    value = float(value)
    return float(f"{value:.4g}") if np.isfinite(value) else None


def _numbers(values) -> list:
    return [_number(value) for value in np.ravel(values)]


def _write(name: str, payload) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / name
    path.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)", flush=True)


def candidates(series, day, demand, sold_out, test_from: int) -> list[str]:
    """Series typical of the benchmark that read well in one panel, by a rule that looks at the data and not at any fit.

    A candidate recorded its demand on every test day, sold out on 4 to 8 of them, about the benchmark's 31%, and
    has a mean demand in the upper half of the sampled series, so its bars are not a few small values. Candidates
    are ordered by how far their sold-out test days are from 6, then by mean demand, most first, then by the
    series key. If fewer than eight qualify, the range of sold-out days widens by one day each side at a time, and
    then a test day may go unrecorded, then two, and so on.
    """
    keys, index = np.unique(series, return_inverse=True)
    recorded = np.bincount(index, minlength=len(keys))
    test = day >= test_from
    test_days = np.bincount(index, weights=test, minlength=len(keys))
    sold_test = np.bincount(index, weights=test & sold_out, minlength=len(keys))
    level = np.bincount(index, weights=demand, minlength=len(keys)) / recorded
    full = day.max() - test_from + 1
    for missing in range(full):
        readable = (test_days >= full - missing) & (level >= np.median(level))
        low, high = TYPICAL_SOLD
        keep = readable & (sold_test >= low) & (sold_test <= high)
        while keep.sum() < CANDIDATES and (low > 1 or high < full):
            low, high = max(low - 1, 1), high + 1
            keep = readable & (sold_test >= low) & (sold_test <= high)
        if keep.sum() >= CANDIDATES:
            break
    order = np.lexsort((keys, -level, np.abs(sold_test - sum(TYPICAL_SOLD) / 2)))
    return [str(keys[position]) for position in order if keep[position]][:CANDIDATES]


def compute() -> None:
    import lightgbm
    import pandas as pd

    import benchmark_observed_real as real
    from ashlar import ASRRegressor, observed

    cache = real.DATA_DIR / "freshretailnet-50k-train.parquet"
    if not cache.exists():
        raise SystemExit(f"{cache} is missing: FreshRetailNet-50K is cached on the Mac only, and this stage does not download it.")
    frame = real.fresh_frame()
    start = pd.to_datetime(frame["dt"]).min()
    features, day, demand, sales, sold_out, stock = real.fresh_days(frame, SEED, stock=True)
    series = (features["store_id"].astype(str) + "/" + features["product_id"].astype(str)).to_numpy()
    test_from = real.TRAIN_DAYS + real.CALIBRATION_DAYS
    train = np.flatnonzero(day < real.TRAIN_DAYS)
    calibration = np.flatnonzero((day >= real.TRAIN_DAYS) & (day < test_from))
    test = np.flatnonzero(day >= test_from)
    x_test, y_test = features.iloc[test], demand[test]
    print(
        f"seed {SEED}: {len(train)} training, {len(calibration)} calibration and {len(test)} test days, "
        f"{sold_out.mean():.1%} sold out",
        flush=True,
    )
    chosen = candidates(series, day, demand, sold_out, test_from)
    if not chosen:
        raise SystemExit("no series meets the candidate rule")
    rows = {key: np.flatnonzero(series == key) for key in chosen}
    rows = {key: index[np.argsort(day[index])] for key, index in rows.items()}
    quantiles: dict[str, dict[str, np.ndarray]] = {fit: {} for fit in FITS}
    scores = {}
    seconds = {}

    for fit, label in (
        ("censored", observed.censored(sales[train], sold_out[train])),
        ("sales", sales[train]),
        ("oracle", demand[train]),
    ):
        with real._Clock() as clock:
            model = ASRRegressor(task="mixture", random_state=SEED).fit(features.iloc[train], label)
        seconds[fit] = {"cpu": clock.cpu, "wall": clock.wall}
        for key, index in rows.items():
            quantiles[fit][key] = model.predict_quantiles(features.iloc[index], list(LEVELS))
        q90, q75 = model.predict_quantiles(x_test, [0.9, real.RATIO]).T
        half_bin = 0.005 * demand.mean()
        scores[fit] = {
            "nll": float(-np.mean(model.log_prob(x_test, observed.interval(y_test - half_bin, y_test + half_bin)))),
            "q90": float(np.mean(y_test <= q90)),
            "cost_value": real._newsvendor(q75, y_test),
        }
        print(f"  mixture, {fit}: {scores[fit]}, {seconds[fit]['wall']:.0f} s", flush=True)

    with real._Clock() as clock:
        boosters = {}
        for level in LEVELS:
            boosters[level] = lightgbm.LGBMRegressor(
                objective="quantile", alpha=level, n_estimators=400, learning_rate=0.05, verbose=-1, random_state=SEED
            ).fit(features.iloc[train], sales[train])
    seconds["lightgbm"] = {"cpu": clock.cpu, "wall": clock.wall}
    for key, index in rows.items():
        quantiles["lightgbm"][key] = np.stack([boosters[level].predict(features.iloc[index]) for level in LEVELS], axis=1)
    scores["lightgbm"] = {
        "q90": float(np.mean(y_test <= boosters[0.9].predict(x_test))),
        "cost_value": real._newsvendor(boosters[real.RATIO].predict(x_test), y_test),
    }
    print(f"  lightgbm quantile, sales: {scores['lightgbm']}", flush=True)

    fitted = real.stockouts_aft(features, sales, sold_out, train, calibration, SEED)
    seconds["xgboost aft"] = {"cpu": fitted["cpu"], "wall": fitted["wall"]}
    for key, index in rows.items():
        loc = real.stockouts_aft_predict(fitted, features.iloc[index])
        quantiles["xgboost aft"][key] = real.stockouts_aft_quantiles(fitted, loc, LEVELS)
    record = real._stockouts_aft_record(fitted, x_test, y_test, 0.005 * demand.mean(), scores["oracle"]["cost_value"], SEED)
    scores["xgboost aft"] = {name: record[name] for name in ("nll", "q90", "cost_value", "chosen", "validation_nll")}
    print(f"  xgboost aft, censored: {scores['xgboost aft']}", flush=True)

    for fit in scores:
        scores[fit]["cost"] = float(100.0 * (scores[fit]["cost_value"] / scores["oracle"]["cost_value"] - 1.0))

    benchmark = real._load("stockouts")
    names = {
        "censored": "mixture/censored",
        "sales": "mixture/sales as demand",
        "oracle": f"mixture/{real.STOCK_FITS[0]}",
        "lightgbm": "lightgbm quantile/sales as demand",
        "xgboost aft": "xgboost aft/censored",
    }
    check = {
        fit: {"this run": scores[fit], "benchmark": benchmark.get(f"{SEED}/{name}"), "seconds": seconds[fit]}
        for fit, name in names.items()
    }
    _write("check.json", check)

    payload = {
        "seed": SEED,
        "start_date": str(start.date()),
        "train_days": real.TRAIN_DAYS,
        "calibration_days": real.CALIBRATION_DAYS,
        "levels": list(LEVELS),
        "rule": candidates.__doc__.split("\n\n", 1)[1].strip(),
        "fits": {
            "censored": "mixture on observed.censored(sales, sold_out)",
            "sales": "mixture on sales as if they were demand",
            "oracle": "mixture on the true demand",
            "lightgbm": "LightGBM quantile regression on sales, 400 trees, learning rate 0.05, one model per level",
            "xgboost aft": f"XGBoost AFT on the censored sales; grid chose {scores['xgboost aft']['chosen']}",
        },
        "series": [],
    }
    for key, index in rows.items():
        first = features.iloc[index[0]]
        payload["series"].append(
            {
                "key": key,
                "store_id": str(first["store_id"]),
                "product_id": str(first["product_id"]),
                "city_id": str(first["city_id"]),
                "categories": [str(first[name]) for name in ("first_category_id", "second_category_id", "third_category_id")],
                "day": [int(value) for value in day[index]],
                "sales": _numbers(sales[index]),
                "stock": _numbers(stock[index]),
                "sold_out": [bool(value) for value in sold_out[index]],
                "demand": _numbers(demand[index]),
                "discount": _numbers(features["discount"].to_numpy()[index]),
                "quantiles": {fit: [_numbers(row) for row in quantiles[fit][key]] for fit in FITS},
            }
        )
    _write("series.json", payload)
    results = Path(str(real.RESULTS).format(stage="stockouts"))
    if results.exists():
        shutil.copy(results, ARTIFACTS / "stockouts.json")
        print(f"copied {results} to {ARTIFACTS / 'stockouts.json'}", flush=True)
    else:
        print(f"{results} is missing; run benchmark_observed_real.py stockouts for the order-cost bars", flush=True)


SERIES = 0  # the candidate the figure draws, in the order of ``candidates``
BAR = 0.56  # a day's bar width, in days
COSTS = (  # the order-cost bars: the benchmark's key, the label, the bar's colour, the value's colour
    ("xgboost aft/censored", "XGBoost AFT", "CLAY", "CLAY"),
    ("mixture/censored", "Ashlar, mixture head", "TEAL", "TEAL"),
    ("lightgbm quantile/sales as demand", "LightGBM quantile regression", "SLATE", "SLATE"),
    ("mixture/sales as demand", "Ashlar, mixture head", "TEAL_PALE", "MUTED"),
    ("mixture/days not sold out", "Ashlar, sold-out days dropped", "TEAL_PALE", "MUTED"),
)
READ_AS = ((0, "Read as censored"), (2, "Read as demand, or dropped"))  # group headers over the bars


def _steps(day, values):
    """Hold each day's value from half a day before it to half a day after: x of shape (2 days,), values (..., 2 days)."""
    day = np.asarray(day, float)
    return np.column_stack([day - 0.5, day + 0.5]).ravel(), np.repeat(np.asarray(values, float), 2, axis=-1)


def _test_days(data: dict, item: dict) -> dict:
    """The series' test days: day, sales, stock, sold out, demand, and each fit's quantiles as (levels, days)."""
    day = np.asarray(item["day"])
    keep = day >= data["train_days"] + data["calibration_days"]
    out = {name: np.asarray(item[name], float)[keep] for name in ("sales", "stock", "demand")}
    out["day"] = day[keep]
    out["sold_out"] = np.asarray(item["sold_out"], bool)[keep]
    out["quantiles"] = {fit: np.asarray(item["quantiles"][fit], float)[keep].T for fit in item["quantiles"]}
    return out


def _order_costs(bench: dict, k) -> tuple[list[int], list]:
    """Each fit's order cost above the oracle's on every seed, in percent."""
    seeds = sorted({int(key.split("/")[0]) for key in bench})
    rows = []
    for name, label, bar, text in COSTS:
        values = np.asarray([bench[f"{seed}/{name}"]["cost"] for seed in seeds if f"{seed}/{name}" in bench], float)
        rows.append((label, getattr(k, bar), getattr(k, text), values))
    return seeds, rows


def _shelf(ax, k, days: dict) -> None:
    """Sales bars, a lid where the shelf ran out, the demand it hid, and each fit's quantiles of demand."""
    day, sold = days["day"], days["sold_out"]
    q = days["quantiles"]
    x, teal = _steps(day, q["censored"])
    _, clay = _steps(day, q["xgboost aft"])
    _, slate = _steps(day, q["lightgbm"])
    ax.bar(day, days["sales"], width=BAR, color=k.DATA, lw=0, zorder=2)
    ax.fill_between(x, teal[0], teal[-1], color=k.TEAL, alpha=0.17, lw=0, zorder=3)  # a veil: the bars show through
    for level in (0, -1):
        ax.plot(x, teal[level], color=k.TEAL, lw=1.8, zorder=3.5)
        ax.plot(x, clay[level], color=k.CLAY, lw=1.8, zorder=4)
    ax.plot(x, slate[-1], color=k.SLATE, lw=1.8, linestyle=(0, (4, 2.5)), zorder=4.2)
    ax.hlines(days["stock"][sold], day[sold] - BAR / 2, day[sold] + BAR / 2, color=k.INK, lw=3.2, zorder=5)
    hid = sold & (days["demand"] > days["stock"])
    ax.vlines(day[hid], days["stock"][hid], days["demand"][hid], color=k.INK, lw=1.3, linestyles=(0, (1.2, 1.8)),
              zorder=5)
    ax.scatter(day[sold], days["demand"][sold], s=46, color=k.INK, edgecolor=k.CARD, lw=1.2, zorder=6)
    top = max(days["demand"].max(), teal.max(), clay.max(), slate.max())
    ax.set_xlim(day.min() - 0.7, day.max() + 0.7)
    ax.set_ylim(0.0, 1.08 * top)


def _costs(ax, k, costs: list, seeds: list) -> None:
    """The order cost of each fit above the oracle's: a bar at the mean over seeds, a dot for each seed."""
    rows = len(costs) + len(READ_AS)
    low = min(0.0, min(float(values.min()) for *_, values in costs))
    high = max(float(values.max()) for *_, values in costs)
    left, right = 1.2 * low - 1.0, 1.3 * high
    pad = 0.02 * (right - left)  # labels start just right of zero, over the bars that start there
    y = rows - 0.6
    headers = dict(READ_AS)
    for row, (label, bar, text, values) in enumerate(costs):
        if row in headers:
            ax.text(pad, y + 0.05, headers[row], fontproperties=k.F.label, color=k.INK, ha="left", va="center")
            y -= 0.9
        mean = float(values.mean())
        ax.barh(y, mean, height=0.36, color=bar, lw=0, zorder=3)
        ax.scatter(values, np.full(len(values), y), s=16, color=k.INK, edgecolor=k.CARD, lw=0.8, zorder=4)
        ax.text(pad, y + 0.25, label, fontproperties=k.F.small, color=k.MUTED, ha="left", va="bottom")
        k.value(ax, max(0.0, float(values.max())) + 0.03 * (right - left), y, f"{k.signed(mean, 1)}", color=text)
        y -= 1.0
    ax.axvline(0.0, color=k.INK, lw=1.1, zorder=2)
    ax.set_xlim(left, right)
    ax.set_ylim(y + 0.45, rows - 0.1)
    step = 10.0
    ticks = np.arange(np.ceil(left / step) * step, right, step)
    ax.set_xticks(ticks, [k.signed(tick, 0) if tick else "0" for tick in ticks])
    ax.set_yticks([])
    k.style(ax, frame="b", grid="x")
    k.axis_names(ax, y=f"mean of {len(seeds)} seeds, a dot for each")
    k.panel_title(ax, "Order cost above the oracle's")


def draw() -> None:
    from datetime import date, timedelta

    import style as k

    data = json.loads((ARTIFACTS / "series.json").read_text())
    bench = json.loads((ARTIFACTS / "stockouts.json").read_text())
    days = _test_days(data, data["series"][SERIES])
    seeds, costs = _order_costs(bench, k)
    start = date.fromisoformat(data["start_date"])

    def when(day: int) -> str:
        moment = start + timedelta(days=int(day))
        return f"{moment:%b} {moment.day}"

    sold = int(days["sold_out"].sum())
    fig = k.card(
        "Demand estimated from sales, with sold-out days treated as censored",
        f"FreshRetailNet-50K: one store's fresh product over its {len(days['day'])} test days, {when(days['day'][0])} "
        f"to {when(days['day'][-1])}. The stock is simulated on days that recorded their demand, so the true "
        "demand on sold-out days is known. Bands run from 10% to 90% of demand, fitted to the censored sales: Ashlar's "
        "mixture head in teal, XGBoost's AFT model in clay. LightGBM's 90% quantile, fitted to sales, in slate.",
        height=9.18,
        footer=3,
    )
    left, right = k.panels(fig, 1, 2, widths=[2.1, 1.0], wspace=0.55, head=0.75, tail=0.62, rpad=0.25)[0]

    _shelf(left, k, days)
    weekly = np.arange(days["day"][0], days["day"][-1] + 1, 7)
    left.set_xticks(weekly, [when(day) for day in weekly])
    k.style(left, frame="lb", grid="y")
    k.axis_names(left, y="sales and demand, a day")
    k.panel_title(left, f"Sold out on {sold} of {len(days['day'])} days, when sales equal the stock and demand is higher")
    _costs(right, k, costs, seeds)

    # The three seeds' costs are printed on the inset's bars and quoted once in the README's prose.
    k.legend_row(
        fig,
        [("bar", k.DATA, "sales"), ("line", k.INK, "sold out"), ("dot", k.INK, "demand"),
         ("pale", k.TEAL_PALE, "Ashlar, censored"), ("line", k.CLAY, "XGBoost AFT, censored"),
         ("dashed", k.SLATE, "LightGBM, sales")],
        line=2,
    )
    k.footnote(fig, "Order costs are over 2,000 series. The oracle is the mixture head fitted to the true demand. An "
                    "order stocks the day's 75% quantile, for a lost sale that costs three times a leftover unit. "
                    "31% of the days sell out.")
    k.save(fig, "sold-out")


STAGES = {"compute": compute, "draw": draw}

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "draw"
    if stage not in STAGES:
        raise SystemExit(f"unknown stage {stage!r}; choose from {', '.join(STAGES)}")
    STAGES[stage]()
