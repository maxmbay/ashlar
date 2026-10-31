"""Training on what was observed: sold-out sales, weekly totals of daily values, and products that sell out together.

Three simulations, so the true values are known and every fit is scored
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

**Several products that sell out together.** Four products at 20 stores,
on the log scale. Each day's demand shares the store's level, the weekday,
a store-wide promotion, and the temperature (each product reacts to it
differently), plus a shock common to all four and one of each product's
own. On a rush day, 10% of days and 35% of promotion days, every product's
demand jumps by 80% at once. The planner stocks each product at its usual
demand times a safety factor from 1.2 to 1.6, not knowing whether the day
will rush, so shelves empty together: about a quarter of product-days sell
out, most of them on rush days. 12,000 days per seed, 8,000 to train and
4,000 to test.

Each joint head (``multivariate_gaussian``, ``multivariate_mixture``,
``multivariate_flow``) is fitted on the log of the true demand (the oracle),
on the log of sales as if they were demand, and on
``observed.censored(log_sales, sold_out)``, a table with a cell per product.
The baseline is the usual way to handle censoring without a joint model: a
censored ``mixture`` head per product, its products then independent.
Scored on the test days against the true demand:

- NLL of the true log demand of all four products together.
- The share of days whose total demand over the four products is at or
  below the predicted 90% quantile of the total, from 1,000 joint draws.
- The Brier score of the predicted probability that two or more of the
  day's shelves empty, at that day's stock.

    pip install -e ".[dev]" lightgbm xgboost matplotlib    # the demand stage reads benchmark_readme.py
    JAX_PLATFORMS=cpu python examples/benchmark_observed.py                 # demand and totals
    JAX_PLATFORMS=cpu python examples/benchmark_observed.py totals          # one
    JAX_PLATFORMS=cpu python examples/benchmark_observed.py joint           # several products
    JAX_PLATFORMS=cpu python examples/benchmark_observed.py figure          # docs/images/sold-out-demand.png
    JAX_PLATFORMS=cpu python examples/benchmark_observed.py joint_figure    # docs/images/joint-sold-out.png
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
    from benchmark_readme import _crps

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
    import figures as style
    import matplotlib.pyplot as plt

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


PRODUCTS = 4
SHELF_ROWS, SHELF_TRAIN = 12_000, 8_000
SHELF_STORES = 20
RUSH_LIFT = float(np.log(1.8))
JOINT_HEADS = ("multivariate_gaussian", "multivariate_mixture", "multivariate_flow")
JOINT_FITS = ("oracle: true demand", "sales as demand", "censored")
PER_PRODUCT = ("mixture per product", "censored")


def shelf_days(seed: int, rows: int = SHELF_ROWS):
    """Four products' log demand, log stock, and log sales per store-day, and the features the planner sees."""
    rng = np.random.default_rng(seed)
    store = rng.integers(0, SHELF_STORES, size=rows)
    level = np.random.default_rng(2000 + seed).normal(2.0, 0.35, size=SHELF_STORES)
    product_level = np.array([0.0, 0.4, -0.3, 0.2])
    heat = np.array([0.15, -0.10, 0.05, 0.25])
    weekday = rng.integers(0, 7, size=rows)
    promo = rng.random(rows) < 0.25
    temperature = rng.normal(15.0, 8.0, size=rows)
    usual = (
        level[store][:, None]
        + product_level
        + 0.25 * np.sin(2 * np.pi * weekday / 7)[:, None]
        + 0.3 * promo[:, None]
        + heat * ((temperature - 15.0) / 10.0)[:, None]
    )
    rush = rng.random(rows) < np.where(promo, 0.35, 0.10)
    shock = 0.25 * rng.normal(size=(rows, 1)) + 0.2 * rng.normal(size=(rows, PRODUCTS))
    log_demand = usual + RUSH_LIFT * rush[:, None] + shock
    log_stock = usual + np.log(rng.uniform(1.2, 1.6, size=(rows, PRODUCTS))) + 0.05 * rng.normal(size=(rows, PRODUCTS))
    sold_out = log_demand >= log_stock
    frame = pd.DataFrame(
        {
            "store": pd.Categorical([f"s{index}" for index in store]),
            "weekday": weekday.astype(np.float32),
            "promo": promo.astype(np.float32),
            "temperature": temperature.astype(np.float32),
        }
    )
    return frame, log_demand, log_stock, np.minimum(log_demand, log_stock), sold_out, rush


def _joint_targets(fit, log_demand, log_sales, sold_out):
    if fit == JOINT_FITS[0]:
        return log_demand
    if fit == JOINT_FITS[1]:
        return log_sales
    return observed.censored(log_sales, sold_out)


def _joint_record(log_density, draws, log_demand, log_stock):
    """Scores from the joint log density of the true demand and joint draws ``(draws, rows, products)`` of log demand."""
    total = np.exp(log_demand).sum(axis=1)
    predicted_total = np.exp(draws).sum(axis=2)
    q90 = np.quantile(predicted_total, 0.9, axis=0)
    empty = np.mean((draws >= log_stock[None]).sum(axis=2) >= 2, axis=0)
    happened = (log_demand >= log_stock).sum(axis=1) >= 2
    return {
        "nll": float(-np.mean(log_density)),
        "total_q90": float(np.mean(total <= q90)),
        "two_empty_brier": float(np.mean(np.square(empty - happened))),
        "two_empty_predicted": float(np.mean(empty)),
    }


def joint() -> None:
    scores = {(head, fit): [] for head in JOINT_HEADS for fit in JOINT_FITS}
    scores[PER_PRODUCT] = []
    for seed in SEEDS:
        frame, log_demand, log_stock, log_sales, sold_out, rush = shelf_days(seed)
        train, test = np.arange(SHELF_TRAIN), np.arange(SHELF_TRAIN, SHELF_ROWS)
        x_train, x_test = frame.iloc[train], frame.iloc[test]
        happened = float(np.mean((sold_out[test]).sum(axis=1) >= 2))
        print(
            f"seed {seed}: {sold_out.mean():.1%} of product-days sold out, {np.mean(sold_out.any(axis=1)):.1%} of days "
            f"had one, two or more on {happened:.1%} of test days; rush on {rush.mean():.1%}",
            flush=True,
        )
        for head in JOINT_HEADS:
            for fit in JOINT_FITS:
                start = time.time()
                model = ASRRegressor(task=head, random_state=seed).fit(
                    x_train, _joint_targets(fit, log_demand[train], log_sales[train], sold_out[train])
                )
                draws = model.sample(x_test, n=DRAWS, seed=seed)
                record = _joint_record(model.log_prob(x_test, log_demand[test]), draws, log_demand[test], log_stock[test])
                record["seconds"] = time.time() - start
                scores[(head, fit)].append(record)
                print(f"  {head:22s} {fit:20s}", {key: round(value, 4) for key, value in record.items()}, flush=True)
        start = time.time()
        density = np.zeros(len(test))
        draws = np.zeros((DRAWS, len(test), PRODUCTS))
        for product in range(PRODUCTS):
            label = observed.censored(log_sales[train, product], sold_out[train, product])
            model = ASRRegressor(task="mixture", random_state=seed).fit(x_train, label)
            density += model.log_prob(x_test, log_demand[test, product])
            draws[..., product] = model.sample(x_test, n=DRAWS, seed=seed + 17 * product)
        record = _joint_record(density, draws, log_demand[test], log_stock[test])
        record["seconds"] = time.time() - start
        scores[PER_PRODUCT].append(record)
        print(f"  {PER_PRODUCT[0]:22s} {PER_PRODUCT[1]:20s}", {key: round(value, 4) for key, value in record.items()}, flush=True)

    print(f"\n{len(SEEDS)} seeds, mean ± standard deviation, scored against the true demand")
    print("| Head | Trained on | Joint NLL ↓ | Total below its 90% quantile | Two or more empty: Brier ↓ | predicted share | Seconds |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for (head, fit), rows in scores.items():
        cells = []
        for key in ("nll", "total_q90", "two_empty_brier", "two_empty_predicted", "seconds"):
            values = np.asarray([row[key] for row in rows])
            if key in ("total_q90", "two_empty_predicted"):
                cells.append(f"{values.mean():.1%} ± {values.std(ddof=1):.1%}")
            elif key == "seconds":
                cells.append(f"{values.mean():.0f}")
            else:
                cells.append(f"{values.mean():.3f} ± {values.std(ddof=1):.3f}")
        name = f"`{head}`" if head in JOINT_HEADS else head
        print(f"| {name} | {fit} | " + " | ".join(cells) + " |")


JOINT_FIGURE_HEAD = "multivariate_mixture"


def joint_figure_draws(seed: int = 0):
    """The held-out days of ``seed`` in ``joint``'s split, and each figure fit's joint draws of their log demand."""
    frame, log_demand, _log_stock, log_sales, sold_out, _rush = shelf_days(seed)
    train, test = np.arange(SHELF_TRAIN), np.arange(SHELF_TRAIN, SHELF_ROWS)
    x_train, x_test = frame.iloc[train], frame.iloc[test]
    draws = {}
    for key, target in (
        ("sales", log_sales[train]),
        ("joint", observed.censored(log_sales[train], sold_out[train])),
        ("oracle", log_demand[train]),
    ):
        model = ASRRegressor(task=JOINT_FIGURE_HEAD, random_state=seed).fit(x_train, target)
        draws[key] = model.sample(x_test, n=DRAWS, seed=seed)
        print(f"fitted {key}", flush=True)
    independent = np.zeros((DRAWS, len(test), PRODUCTS))
    for product in range(PRODUCTS):
        label = observed.censored(log_sales[train, product], sold_out[train, product])
        model = ASRRegressor(task="mixture", random_state=seed).fit(x_train, label)
        independent[..., product] = model.sample(x_test, n=DRAWS, seed=seed + 17 * product)
    draws["per_product"] = independent
    return x_test, log_demand[test], log_sales[test], sold_out[test], draws


def _round_axis(top: float):
    """A round upper limit at or above ``top`` and its ticks from 0, the limit itself left unlabelled."""
    step = next(step for step in (10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000) if top / step <= 4)
    end = step * float(np.ceil(top / step))
    return end, np.arange(0, int(end), int(step))


def joint_figure() -> None:
    """``docs/images/joint-sold-out.png``: the total demand of four products, recorded and as four fits see it.

    Seed 0 and the split of ``joint``. Each panel pools the held-out days of
    one condition: the true total demand behind it, and each fit's
    predicted total from its joint draws, or from independent draws of the
    per-product fits.
    """
    import figures as style
    import matplotlib.pyplot as plt

    style._load_fonts()
    x_test, log_demand, log_sales, sold_out, draws = joint_figure_draws(0)
    fits = (
        ("sales", "fit to sales"),
        ("per_product", "censored, product by product"),
        ("joint", "censored, all four together"),
        ("oracle", "fit to the true demand"),
    )
    truth = np.exp(log_demand).sum(axis=1)
    recorded = np.exp(log_sales).sum(axis=1)
    totals = {key: np.exp(value).sum(axis=2) for key, value in draws.items()}
    promo = x_test["promo"].to_numpy() > 0
    conditions = (("ordinary days", ~promo), ("promotion days", promo))
    share = float(np.mean(sold_out))
    together = float(np.mean(sold_out.sum(axis=1) >= 2))
    print(f"held-out product-days sold out {share:.4f}, days with two or more {together:.4f}", flush=True)

    width, height = 18.2, 8.9
    fig = plt.figure(figsize=(width, height), facecolor=style.WHITE)
    fig.text(0.045, 0.975, "Shelves that empty together hide the busiest days.", fontproperties=style.TITLE,
             color=style.INK, ha="left", va="top")
    for offset, line in (
        (0.925, (f"Simulated demand for four products at 20 stores, which jumps for all four on a rush day. "
                 f"{share:.0%} of product-days sold out, two or more products on {together:.0%} of days.")),
        (0.885, ("Bars: the true total demand of the four over the held-out days. Lines: the recorded sales, then "
                 "each fit's predicted total. Only the last fit saw demand.")),
    ):
        fig.text(0.045, offset, line, fontproperties=style.CLAIM, color=style.MUTED, ha="left", va="top")
    fig.add_artist(plt.Line2D([0.045, 0.955], [0.852, 0.852], transform=fig.transFigure, color=style.HAIR,
                              linewidth=1.05))

    colors = {"sales": style.CLAY, "per_product": style.CLAY, "joint": style.TEAL, "oracle": style.MUTED}
    panels = [("recorded", "recorded sales")] + list(fits)
    rows = len(conditions)
    for row, (label, mask) in enumerate(conditions):
        top, ticks = _round_axis(float(np.quantile(truth[mask], 0.995)) * 1.05)
        bins = np.linspace(0.0, top, 46)
        centres = 0.5 * (bins[1:] + bins[:-1])
        heights = np.histogram(truth[mask], bins=bins, density=True)[0]
        sales = np.histogram(recorded[mask], bins=bins, density=True)[0]
        curves, cover = {}, {}
        for key, _title in fits:
            pooled = totals[key][:, mask]
            curves[key] = np.histogram(pooled.reshape(-1), bins=bins, density=True)[0]
            cover[key] = float(np.mean(truth[mask] <= np.quantile(pooled, 0.9, axis=0)))
        ceiling = 1.32 * max(heights.max(), sales.max(), max(values.max() for values in curves.values()))
        print(f"{label} days {int(mask.sum())} " + " ".join(f"{key}_q90 {value:.4f}" for key, value in cover.items()),
              flush=True)
        for column, (key, title) in enumerate(panels):
            ax = fig.add_axes([0.045 + column * 0.185, 0.095 + (rows - 1 - row) * 0.37, 0.16, 0.26])
            style._style_plot(ax)
            ax.hist(truth[mask], bins=bins, density=True, color=style.HAIR, edgecolor=style.WHITE, linewidth=0.6,
                    zorder=1)
            if key == "recorded":
                ax.hist(recorded[mask], bins=bins, density=True, histtype="step", color=style.INK, linewidth=1.6,
                        zorder=3)
                note, tone = f"a shelf empty on {np.mean(sold_out[mask].any(axis=1)):.0%} of days", style.MUTED
            else:
                lit = key == "joint"
                ax.plot(centres, curves[key], color=colors[key], linewidth=2.3 if lit else 1.8, zorder=3)
                note = f"90% quantile covers {cover[key]:.0%}"
                tone = style.TEAL if lit else (style.MUTED if key == "oracle" else style.CLAY)
            ax.set_xlim(0.0, top)
            ax.set_ylim(0, ceiling)
            ax.set_yticks([])
            ax.set_xticks(ticks)
            style._ticks(ax)
            if row == 0:
                ax.text(0.0, 1.08, title, transform=ax.transAxes, fontproperties=style.AXIS, color=style.INK,
                        ha="left", va="bottom")
            if column == 0:
                ax.text(-0.09, 0.5, label, transform=ax.transAxes, fontproperties=style.AXIS, color=style.INK,
                        ha="center", va="center", rotation=90)
            if row == rows - 1:
                ax.set_xlabel("units of all four a day", fontproperties=style.AXIS, color=style.INK, labelpad=9)
            ax.text(0.98, 0.93, note, transform=ax.transAxes, fontproperties=style.TICK, color=tone, ha="right",
                    va="top")
    style._save(fig, "joint-sold-out.png")


def main() -> None:
    chosen = sys.argv[1:] or ["demand", "totals"]
    for name in chosen:
        {"demand": demand, "totals": totals, "joint": joint, "figure": figure, "joint_figure": joint_figure}[name]()


if __name__ == "__main__":
    main()
