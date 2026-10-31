"""Training on what was observed: sold-out sales, and weekly totals of daily values.

Two simulations, so the true values are known and every fit is scored
against them. Seeds 0 to 4, mean ± standard deviation.

**Demand from sales when shelves sell out.** A sold-out day records sales
equal to the stock, so the demand was at least the stock.

- 50 stores with their own level, a day-of-week cycle, promotions, a price
  discount and a temperature effect; demand is negative binomial with a mean
  mostly between 3 and 50 units and dispersion 5.
- The stock is a planner's order: the true mean times a lognormal forecast
  error (sd 0.25) times a safety factor drawn from 1.0 to 1.5. Given the
  features it is independent of the demand, which is what the censored
  likelihood assumes. About a third of days sell out.
- 20,000 store-days per seed: 14,000 to train, 2,000 to calibrate on, and
  4,000 to test. Seeds 0 to 4.

Each head is fitted four ways, all with default settings: on the true demand
(the oracle no shop has), on sales as if they were demand, on the days that
did not sell out, and on ``observed.censored(sales, sold_out)``. A LightGBM
quantile regression on sales is the usual tree baseline. Scored on the test
days against the true demand:

- NLL of the true demand: the log mass of a count head, and for the mixture
  the log probability of the count's rounding bin, ``[d - 0.5, d + 0.5]``,
  so the two are comparable. CRPS from 1,000 draws.
- The share of true demand at or below the predicted 90% quantile.
- Newsvendor cost: order the 75% quantile (lost margin 3, leftover cost 1)
  and pay for the true demand, in percent above ordering the true quantile.
- The 80% interval after ``calibrate`` on the calibration days: on their true
  demand for the oracle, their censored labels for the censored fit, and
  their sales for the others.

**Daily values from weekly totals.** 1,500 weeks of 7 days whose daily
values were never recorded, only each week's sum. A daily value has mean
``10 + 3 x1 + 2 sin(3 x2)`` and standard deviation 0.5, or 1.5 on days with
``x2 > 0``; a daily count is Poisson with mean ``exp(0.5 + 0.8 x1)``. The
features change from day to day. Each head is fitted on the days themselves
(the oracle), on ``observed.total(weekly, week)``, and on each week's total
spread evenly over its days (rounded for the count). Scored on 3,000 new
days: NLL of the true daily value, the error of the predicted mean against
the true mean, the predicted standard deviation on calm and on volatile
days, and the 80% interval.

    JAX_PLATFORMS=cpu python examples/benchmark_observed.py           # both
    JAX_PLATFORMS=cpu python examples/benchmark_observed.py totals    # one
    JAX_PLATFORMS=cpu python examples/benchmark_observed.py figure    # docs/images/sold-out-demand.png
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_readme import _crps

from ashlar import ASRRegressor, observed

SEEDS = (0, 1, 2, 3, 4)
ROWS, TRAIN, CALIBRATE = 20_000, 14_000, 2_000
STORES = 50
DISPERSION = 5.0
DRAWS = 1000
UNDER, OVER = 3.0, 1.0
RATIO = UNDER / (UNDER + OVER)
HEADS = ("negative_binomial", "mixture")
COUNT_TASKS = ("poisson", "negative_binomial")
FITS = ("oracle: true demand", "sales as demand", "days not sold out", "censored")


def store_days(seed: int):
    rng = np.random.default_rng(seed)
    store = rng.integers(0, STORES, size=ROWS)
    level = np.random.default_rng(1000 + seed).normal(2.3, 0.45, size=STORES)
    weekday = rng.integers(0, 7, size=ROWS)
    promo = rng.random(ROWS) < 0.2
    discount = rng.uniform(0.0, 0.3, size=ROWS)
    temperature = rng.normal(15.0, 8.0, size=ROWS)
    log_mean = (
        level[store]
        + 0.25 * np.sin(2 * np.pi * weekday / 7)
        + 0.5 * promo
        + 1.5 * discount
        + 0.02 * (temperature - 15.0)
        - 0.0015 * (temperature - 15.0) ** 2
    )
    mean = np.exp(log_mean)
    demand = rng.negative_binomial(DISPERSION, DISPERSION / (DISPERSION + mean))
    stock = np.round(mean * np.exp(rng.normal(0.0, 0.25, size=ROWS)) * rng.uniform(1.0, 1.5, size=ROWS))
    stock = np.maximum(stock, 1).astype(np.int64)
    sales = np.minimum(demand, stock)
    frame = pd.DataFrame(
        {
            "store": pd.Categorical([f"s{index}" for index in store]),
            "weekday": weekday.astype(np.float32),
            "promo": promo.astype(np.float32),
            "discount": discount.astype(np.float32),
            "temperature": temperature.astype(np.float32),
        }
    )
    return frame, mean, demand, sales, demand >= stock


def _negative_binomial_quantile(level, mean):
    return stats.nbinom.ppf(level, DISPERSION, DISPERSION / (DISPERSION + mean))


def _newsvendor(order, demand):
    return np.mean(UNDER * np.maximum(demand - order, 0) + OVER * np.maximum(order - demand, 0))


def _labels(fit, demand, sales, sold_out):
    if fit == FITS[0]:
        return demand, np.ones(len(demand), dtype=bool)
    if fit == FITS[1]:
        return sales, np.ones(len(demand), dtype=bool)
    if fit == FITS[2]:
        return sales, ~sold_out
    return observed.censored(sales, sold_out), np.ones(len(demand), dtype=bool)


def _score(model, x_test, demand, best_cost, seed):
    dist = model.predict_dist(x_test)
    draws = np.asarray(dist.sample(jax.random.PRNGKey(seed), (DRAWS,)), dtype=np.float64).T
    q90, q75 = model.predict_quantiles(x_test, [0.9, RATIO]).T
    # A count head gives each count a mass; a continuous head gives the count's rounding bin one.
    if model.task in COUNT_TASKS:
        target = demand.astype(np.float64)
    else:
        target = observed.interval(demand - 0.5, demand + 0.5)
    return {
        "nll": float(-np.mean(model.log_prob(x_test, target))),
        "crps": _crps(draws, demand),
        "q90": float(np.mean(demand <= q90)),
        "cost": float(100.0 * (_newsvendor(q75, demand) / best_cost - 1.0)),
    }


def demand() -> None:
    import lightgbm

    scores = {(head, fit): [] for head in HEADS for fit in FITS}
    scores[("lightgbm quantile", "sales as demand")] = []
    for seed in SEEDS:
        frame, mean, demand, sales, sold_out = store_days(seed)
        train, cal, test = (
            np.arange(TRAIN),
            np.arange(TRAIN, TRAIN + CALIBRATE),
            np.arange(TRAIN + CALIBRATE, ROWS),
        )
        best_cost = _newsvendor(_negative_binomial_quantile(RATIO, mean[test]), demand[test])
        print(f"seed {seed}: {sold_out.mean():.1%} of days sold out, mean demand {demand.mean():.1f}", flush=True)
        x_test = frame.iloc[test]
        for head in HEADS:
            for fit in FITS:
                start = time.time()
                target, keep = _labels(fit, demand[train], sales[train], sold_out[train])
                rows = train[keep]
                model = ASRRegressor(task=head, random_state=seed).fit(frame.iloc[rows], target[keep])
                record = _score(model, x_test, demand[test], best_cost, seed)
                if fit == FITS[0]:
                    cal_label = demand[cal]
                elif fit == "censored":
                    cal_label = observed.censored(sales[cal], sold_out[cal])
                else:
                    cal_label = sales[cal]
                model.calibrate(frame.iloc[cal], cal_label)
                low, high = model.predict_interval(x_test, coverage=0.8).T
                record["calibrated_80"] = float(np.mean((demand[test] >= low) & (demand[test] <= high)))
                record["seconds"] = time.time() - start
                scores[(head, fit)].append(record)
                print(f"  {head:18s} {fit:20s}", {key: round(value, 4) for key, value in record.items()}, flush=True)
        orders = {}
        for level in (0.9, RATIO):
            booster = lightgbm.LGBMRegressor(
                objective="quantile", alpha=level, n_estimators=400, learning_rate=0.05, verbose=-1, random_state=seed
            )
            booster.fit(frame.iloc[train], sales[train])
            orders[level] = booster.predict(x_test)
        record = {
            "q90": float(np.mean(demand[test] <= orders[0.9])),
            "cost": float(100.0 * (_newsvendor(orders[RATIO], demand[test]) / best_cost - 1.0)),
        }
        scores[("lightgbm quantile", "sales as demand")].append(record)
        print(f"  {'lightgbm quantile':18s} {'sales as demand':20s}", {key: round(value, 4) for key, value in record.items()}, flush=True)

    print(f"\n{len(SEEDS)} seeds, mean ± standard deviation, scored against the true demand")
    columns = (
        ("nll", "NLL ↓", "{:.3f}"),
        ("crps", "CRPS ↓", "{:.3f}"),
        ("q90", "Below the 90% quantile", "{:.1%}"),
        ("cost", "Newsvendor cost above best ↓", "{:.1f}%"),
        ("calibrated_80", "Calibrated 80% interval holds", "{:.1%}"),
        ("seconds", "Seconds", "{:.0f}"),
    )
    print("| Head | Trained on | " + " | ".join(title for _key, title, _form in columns) + " |")
    print("|---|---|" + "---:|" * len(columns))
    for (head, fit), rows in scores.items():
        cells = []
        for key, _title, form in columns:
            if key not in rows[0]:
                cells.append("")
                continue
            values = np.asarray([row[key] for row in rows])
            spread = values.std(ddof=1)
            if "%" in form and key != "cost":
                cells.append(f"{values.mean():.1%} ± {spread:.1%}")
            elif key == "seconds":
                cells.append(f"{values.mean():.0f}")
            else:
                number = form.replace("%", "")
                cells.append(f"{number.format(values.mean())} ± {number.format(spread)}" + ("%" if key == "cost" else ""))
        print(f"| `{head}` | {fit} | " + " | ".join(cells) + " |")


WEEKS, DAYS, TEST_DAYS = 1500, 7, 3000
TOTAL_FITS = ("oracle: the days", "weekly totals", "total spread evenly")


def daily(seed: int, n: int):
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.normal(size=n), rng.uniform(-1.0, 1.0, size=n)]).astype(np.float32)
    mean = 10.0 + 3.0 * x[:, 0] + 2.0 * np.sin(3.0 * x[:, 1])
    spread = 0.5 + 1.0 * (x[:, 1] > 0)
    rate = np.exp(0.5 + 0.8 * np.clip(x[:, 0], -2.0, 2.0))
    return x, mean, spread, mean + spread * rng.normal(size=n), rate, rng.poisson(rate).astype(np.float64)


def totals() -> None:
    scores = {(head, fit): [] for head in ("gaussian", "poisson") for fit in TOTAL_FITS}
    week = np.repeat(np.arange(WEEKS), DAYS)
    for seed in SEEDS:
        x, _mean, _spread, value, _rate, count = daily(seed, WEEKS * DAYS)
        x_test, mean, spread, value_test, rate, count_test = daily(100 + seed, TEST_DAYS)
        for head, y, y_test, truth in (("gaussian", value, value_test, mean), ("poisson", count, count_test, rate)):
            weekly = np.bincount(week, weights=y)[week]
            # A count head takes whole numbers, so the even spread of a count is rounded.
            even = weekly / DAYS if head == "gaussian" else np.round(weekly / DAYS)
            for fit, target in zip(TOTAL_FITS, (y, observed.total(weekly, week), even)):
                start = time.time()
                model = ASRRegressor(task=head, random_state=seed).fit(x, target)
                dist = model.predict_dist(x_test)
                low, high = model.predict_interval(x_test, coverage=0.8).T
                sd = np.asarray(dist.stddev(), dtype=np.float64)
                volatile = x_test[:, 1] > 0
                record = {
                    "nll": float(-np.mean(model.log_prob(x_test, y_test))),
                    "mean_error": float(np.sqrt(np.mean((np.asarray(dist.mean(), dtype=np.float64) - truth) ** 2))),
                    "sd_calm": float(sd[~volatile].mean()),
                    "sd_volatile": float(sd[volatile].mean()),
                    "coverage_80": float(np.mean((y_test >= low) & (y_test <= high))),
                    "seconds": time.time() - start,
                }
                scores[(head, fit)].append(record)
                print(seed, head, fit, {key: round(value, 4) for key, value in record.items()}, flush=True)
    print(f"\n{len(SEEDS)} seeds, mean ± standard deviation, scored against the true daily values")
    print("| Head | Trained on | NLL ↓ | Mean error ↓ | SD, calm days | SD, volatile days | 80% interval holds | Seconds |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for (head, fit), rows in scores.items():
        cells = []
        for key in ("nll", "mean_error", "sd_calm", "sd_volatile", "coverage_80", "seconds"):
            values = np.asarray([row[key] for row in rows])
            if key == "coverage_80":
                cells.append(f"{values.mean():.1%} ± {values.std(ddof=1):.1%}")
            elif key == "seconds":
                cells.append(f"{values.mean():.0f}")
            else:
                cells.append(f"{values.mean():.3f} ± {values.std(ddof=1):.3f}")
        print(f"| `{head}` | {fit} | " + " | ".join(cells) + " |")


FIGURE_FITS = (
    ("sales", "fit to sales"),
    ("censored", "fit to censored sales"),
    ("oracle", "fit to the true demand"),
)


def figure() -> None:
    """``docs/images/sold-out-demand.png``: what the shop recorded, then three fits, under three conditions.

    Seed 0, the negative binomial head, the same split as ``demand``. Each
    panel averages the predicted mass over the held-out days of one
    condition, over the true demand of those days.
    """
    import matplotlib.pyplot as plt

    import figures as style

    style._load_fonts()
    seed = 0
    frame, _mean, demand, sales, sold_out = store_days(seed)
    train, test = np.arange(TRAIN), np.arange(TRAIN + CALIBRATE, ROWS)
    labels = {
        "sales": sales[train],
        "censored": observed.censored(sales[train], sold_out[train]),
        "oracle": demand[train],
    }
    models = {
        key: ASRRegressor(task="negative_binomial", random_state=seed).fit(frame.iloc[train], labels[key])
        for key, _title in FIGURE_FITS
    }

    # Thirds of the stores by their average demand in the training days.
    store = frame["store"].astype(str).to_numpy()
    average = pd.Series(demand[train]).groupby(store[train]).mean()
    third = np.digitize(average.reindex(store).to_numpy(), average.quantile([1 / 3, 2 / 3]).to_numpy())
    conditions = (("quiet stores", third == 0), ("average stores", third == 1), ("busy stores", third == 2))
    share = float(np.mean(sold_out[test]))
    print(f"held-out days sold out {share:.4f}", flush=True)

    width, height = 16.4, 12.2
    fig = plt.figure(figsize=(width, height), facecolor=style.WHITE)
    fig.text(0.055, 0.982, "A sold-out day records the stock, not the demand.", fontproperties=style.TITLE,
             color=style.INK, ha="left", va="top")
    for offset, line in (
        (0.947, f"Simulated daily demand at 50 stores, given the store, weekday, promotion, discount, and temperature. "
                f"{share:.0%} of the days sold out."),
        (0.918, "Each panel is the average predicted demand over the held-out days of that condition, with their demand "
                "behind it. Only the last fit saw demand."),
    ):
        fig.text(0.055, offset, line, fontproperties=style.CLAIM, color=style.MUTED, ha="left", va="top")
    fig.add_artist(plt.Line2D([0.055, 0.945], [0.893, 0.893], transform=fig.transFigure, color=style.HAIR,
                              linewidth=1.05))

    colors = {"sales": style.CLAY, "censored": style.TEAL, "oracle": style.MUTED}
    rows = len(conditions)
    for row, (label, mask) in enumerate(conditions):
        rows_test = test[mask[test]]
        x_group, truth, recorded = frame.iloc[rows_test], demand[rows_test], sales[rows_test]
        top = int(np.ceil((np.quantile(truth, 0.995) + 2) / 10.0) * 10)
        step = 1 if top <= 40 else 2
        bins = np.arange(-0.5, top + step, step)
        grid = np.arange(0, top + 1)
        mass, cover = {}, {}
        for key, _title in FIGURE_FITS:
            dist = models[key].predict_dist(x_group)
            mass[key] = np.asarray(
                [np.mean(np.exp(np.asarray(dist.log_prob(np.full(len(truth), float(k)))))) for k in grid]
            )
            q90 = np.asarray(models[key].predict_quantiles(x_group, [0.9]))[:, 0]
            cover[key] = float(np.mean(truth <= q90))
        heights = np.histogram(truth, bins=bins, density=True)[0]
        ceiling = 1.18 * max(heights.max(), max(values.max() for values in mass.values()))
        print(f"{label} days {len(truth)} sold_out {np.mean(sold_out[rows_test]):.4f} "
              + " ".join(f"{key}_q90 {value:.4f}" for key, value in cover.items()), flush=True)

        panels = [("recorded", "sales (line), true demand (bars)")] + list(FIGURE_FITS)
        for column, (key, title) in enumerate(panels):
            ax = fig.add_axes([0.055 + column * 0.228, 0.075 + (rows - 1 - row) * 0.268, 0.195, 0.195])
            style._style_plot(ax)
            ax.hist(truth, bins=bins, density=True, color=style.HAIR, edgecolor=style.WHITE, linewidth=0.6, zorder=1)
            if key == "recorded":
                ax.hist(recorded, bins=bins, density=True, histtype="step", color=style.INK, linewidth=1.6, zorder=3)
                note, tone = f"sold out on {np.mean(sold_out[rows_test]):.0%} of days", style.MUTED
            else:
                lit = key == "censored"
                ax.plot(grid, mass[key], color=colors[key], linewidth=2.3 if lit else 1.8, zorder=3)
                note = f"90% quantile covers {cover[key]:.0%}"
                tone = style.TEAL if lit else (style.CLAY if key == "sales" else style.MUTED)
            ax.set_xlim(-0.5, top)
            ax.set_ylim(0, ceiling)
            ax.set_yticks([])
            ax.set_xticks(np.linspace(0, top, 5 if top % 4 == 0 else 3).round().astype(int))
            style._ticks(ax)
            if row == 0:
                ax.text(0.0, 1.1, title, transform=ax.transAxes, fontproperties=style.AXIS, color=style.INK,
                        ha="left", va="bottom")
            if column == 0:
                ax.text(-0.085, 0.5, label, transform=ax.transAxes, fontproperties=style.AXIS, color=style.INK,
                        ha="center", va="center", rotation=90)
            if row == rows - 1:
                ax.set_xlabel("units a day", fontproperties=style.AXIS, color=style.INK, labelpad=9)
            ax.text(0.98, 0.93, note, transform=ax.transAxes, fontproperties=style.TICK, color=tone, ha="right",
                    va="top")
    style._save(fig, "sold-out-demand.png")


def main() -> None:
    chosen = sys.argv[1:] or ["demand", "totals"]
    for name in chosen:
        {"demand": demand, "totals": totals, "figure": figure}[name]()


if __name__ == "__main__":
    main()
