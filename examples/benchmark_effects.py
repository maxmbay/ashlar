"""Entity effects on two retail tables: pooled offsets and embeddings, against the alternatives.

The protocol of issue #41. On each table every store is an entity:

- 20% of the stores are held out whole (*unseen*). Each gives 5 rows of
  context and 30 rows to score.
- Every other store gets a random number of training rows, log-uniform from
  1 to 300, and 30 other rows to score, so the test rows fall in buckets by
  how many rows their store trained on: 1 to 5, 6 to 20, 21 to 100, and more.

Methods, all on ``log`` sales with a normal predictive distribution:

- ``embedding``: the store as an ordinary learned embedding (Ashlar's default).
- ``dropped``: no store column.
- ``pooled``: ``random_effects={store: "intercept"}``.
- ``pooled_embedding``: ``random_effects={store: ["intercept", "embedding"]}``.
- ``nested``: the same with the store nested in a broader class (store type
  on Rossmann, city on FreshRetailNet), which has its own offset and embedding.
- ``*_context``: a pooled model after ``with_context`` on each unseen store's
  5 context rows; it changes only the unseen stores.
- ``in_context``: the store's effect read in context, the neural-process view
  of a mixed model. No store column; instead each row carries up to 32 other
  rows of its store as two set features through the existing set encoder:
  their binned outcomes, and their binned outcomes by a key covariate (promo
  on Rossmann, activity on FreshRetailNet), with the set's size as a number.
  A training row's set comes from its store's other training rows, leaving
  the row out, with a random size so that small sets are common. A seen
  store's test row reads its store's training rows, an unseen store's reads
  its 5 context rows, and the plain unseen score reads an empty set. The
  outcome is binned at 24 quantiles of the training rows, since a set
  feature holds tokens.
- ``gpboost``: GPBoost, tree boosting with a grouped random intercept;
  ``gpboost_context`` adds the conjugate update from the 5 context rows with
  its fitted variances.
- ``mixedlm``: a statsmodels linear mixed model with a random intercept, and
  ``mixedlm_context`` the same conjugate update.

Scores: CRPS (log units), 80% coverage and width, and RMSE, per bucket. Three
splits by default. Data: Rossmann daily store sales from the fev benchmark on
Hugging Face (``autogluon/fev_datasets``, ``rossmann/1D``), saved as
``/tmp/ashlar-bench/rossmann-1D.parquet``, and FreshRetailNet-50K
(``freshretailnet-50k-train.parquet``), daily sales of 50,000 store and
product pairs.

    JAX_PLATFORMS=cpu python examples/benchmark_effects.py --tables rossmann
    python examples/benchmark_effects.py --report
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_readme import DATA_DIR, _crps

from ashlar import ASRRegressor

SEEDS = (0, 1, 2)
UNSEEN_SHARE = 0.2
CONTEXT_ROWS = 5
TEST_ROWS = 30
MAX_TRAIN_ROWS = 300
BUCKETS = ((1, 5), (6, 20), (21, 100), (101, MAX_TRAIN_ROWS))
N_DRAWS = 400
ASHLAR_METHODS = ("embedding", "dropped", "pooled", "pooled_embedding", "nested")
CONTEXT_SET = 32
Y_BINS = 24
CONTEXT_KEY = {"rossmann": "promo", "freshretail": "activity"}


# ---------------------------------------------------------------- data


def rossmann():
    """Open days with sales of 1,115 Rossmann stores, 2013 to mid 2015. ``Customers`` is known only after the day."""
    stores = pd.read_parquet(DATA_DIR / "rossmann-1D.parquet")
    daily = ["timestamp", "DayOfWeek", "Sales", "Open", "Promo", "StateHoliday", "SchoolHoliday"]
    frame = stores.explode(daily, ignore_index=True)
    frame = frame[(frame["Open"].astype(float) == 1.0) & (frame["Sales"].astype(float) > 0)].reset_index(drop=True)
    stamp = pd.to_datetime(frame["timestamp"])
    x = pd.DataFrame({
        "store": frame["id"].astype(str).to_numpy(dtype=object),
        "store_type": frame["StoreType"].astype(str).to_numpy(dtype=object),
        "assortment": frame["Assortment"].astype(str).to_numpy(dtype=object),
        "weekday": frame["DayOfWeek"].astype(float).astype(int).astype(str).to_numpy(dtype=object),
        "state_holiday": frame["StateHoliday"].astype(str).to_numpy(dtype=object),
        "promo": frame["Promo"].astype(float).to_numpy(),
        "school_holiday": frame["SchoolHoliday"].astype(float).to_numpy(),
        "promo2": frame["Promo2"].astype(float).to_numpy(),
        "competition_distance": frame["CompetitionDistance"].astype(float).to_numpy(),
        "month": stamp.dt.month.astype(float).to_numpy(),
        "day": stamp.dt.day.astype(float).to_numpy(),
        "year": stamp.dt.year.astype(float).to_numpy(),
    })
    return x, np.log(frame["Sales"].astype(float).to_numpy()), "store_type"


def freshretail():
    """Daily sales of 50,000 store and product pairs over 90 days, ``log1p``; the store is the entity."""
    frame = pd.read_parquet(DATA_DIR / "freshretailnet-50k-train.parquet")
    stamp = pd.to_datetime(frame["dt"])
    x = pd.DataFrame({
        "store": frame["store_id"].astype(str).to_numpy(dtype=object),
        "city": frame["city_id"].astype(str).to_numpy(dtype=object),
        "product": frame["product_id"].astype(str).to_numpy(dtype=object),
        "category": frame["third_category_id"].astype(str).to_numpy(dtype=object),
        "group": frame["first_category_id"].astype(str).to_numpy(dtype=object),
        "weekday": stamp.dt.dayofweek.astype(str).to_numpy(dtype=object),
        "discount": frame["discount"].astype(float).to_numpy(),
        "holiday": frame["holiday_flag"].astype(float).to_numpy(),
        "activity": frame["activity_flag"].astype(float).to_numpy(),
        "rain": frame["precpt"].astype(float).to_numpy(),
        "temperature": frame["avg_temperature"].astype(float).to_numpy(),
        "humidity": frame["avg_humidity"].astype(float).to_numpy(),
        "wind": frame["avg_wind_level"].astype(float).to_numpy(),
    })
    return x, np.log1p(frame["sale_amount"].astype(float).to_numpy()), "city"


TABLES = {"rossmann": rossmann, "freshretail": freshretail}


# ---------------------------------------------------------------- split


def split(entity, seed):
    """Row indices: training rows, each training row's store's training-row count, test rows of seen stores with
    their store's count, and for each unseen store its context and test rows."""
    rng = np.random.default_rng(seed)
    stores = np.unique(entity)
    rng.shuffle(stores)
    unseen = set(stores[: int(round(UNSEEN_SHARE * len(stores)))].tolist())
    order = rng.permutation(len(entity))
    by_store = pd.Series(order).groupby(entity[order], sort=False).apply(list)
    train, train_count, test, test_count, context, unseen_test = [], [], [], [], [], []
    for store, rows in by_store.items():
        rows = np.asarray(rows)
        if store in unseen:
            if len(rows) < CONTEXT_ROWS + 1:
                continue
            context.extend(rows[:CONTEXT_ROWS])
            unseen_test.extend(rows[CONTEXT_ROWS : CONTEXT_ROWS + TEST_ROWS])
            continue
        room = len(rows) - min(TEST_ROWS, max(1, len(rows) // 2))
        if room < 1:
            continue
        count = int(min(room, round(math.exp(rng.uniform(0.0, math.log(MAX_TRAIN_ROWS))))))
        train.extend(rows[:count])
        train_count.extend([count] * count)
        held = rows[count : count + TEST_ROWS]
        test.extend(held)
        test_count.extend([count] * len(held))
    return (np.asarray(train), np.asarray(train_count), np.asarray(test), np.asarray(test_count),
            np.asarray(context), np.asarray(unseen_test))


def _scores(mean, sd, y, draws=None):
    """CRPS, 80% coverage and width, and RMSE of a normal (``mean``, ``sd``) or of ``draws`` ``(rows, n)``."""
    if draws is None:
        z = (y - mean) / sd
        crps = sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / math.sqrt(math.pi))
        lo, hi = mean - 1.2815515655446004 * sd, mean + 1.2815515655446004 * sd
        return {"crps": float(np.mean(crps)), "coverage": float(np.mean((y >= lo) & (y <= hi))),
                "width": float(np.mean(hi - lo)), "rmse": float(np.sqrt(np.mean((mean - y) ** 2))), "rows": int(len(y))}
    lo, hi = np.quantile(draws, [0.1, 0.9], axis=1)
    return {"crps": _crps(draws, y), "coverage": float(np.mean((y >= lo) & (y <= hi))), "width": float(np.mean(hi - lo)),
            "rmse": float(np.sqrt(np.mean((mean - y) ** 2))), "rows": int(len(y))}


def _bucketed(score_fn, test_count):
    """Scores per bucket of the store's training rows, ``score_fn(mask)``."""
    out = {}
    for low, high in BUCKETS:
        mask = (test_count >= low) & (test_count <= high)
        if mask.any():
            out[f"{low}-{high}"] = score_fn(mask)
    out["all_seen"] = score_fn(np.ones(len(test_count), dtype=bool))
    return out


# ---------------------------------------------------------------- methods


def _ashlar(method, x, y, rows, seed, broader):
    frame = x.iloc[rows].reset_index(drop=True)
    params = {"task": "gaussian", "random_state": seed}
    if method == "dropped":
        frame = frame.drop(columns=["store"])
    elif method == "pooled":
        params["random_effects"] = {"store": "intercept"}
    elif method == "pooled_embedding":
        params["random_effects"] = {"store": ["intercept", "embedding"]}
    elif method == "nested":
        params["random_effects"] = {
            broader: ["intercept", "embedding"],
            "store": {"terms": ["intercept", "embedding"], "within": broader},
        }
    return ASRRegressor(**params).fit(frame, y[rows])


def _ashlar_scores(model, x, y, rows, seed, drop):
    frame = x.iloc[rows].reset_index(drop=True)
    if drop:
        frame = frame.drop(columns=["store"])
    draws = np.asarray(model.sample(frame, n=N_DRAWS, seed=seed)).T
    mean = np.asarray(model.predict(frame))
    return lambda mask: _scores(mean[mask], None, y[rows][mask], draws[mask])


def _context_frame(x, y, rows, pool, key, edges, rng, random_size, sets=None):
    """``x`` at ``rows`` without the store, plus each row's context: up to ``CONTEXT_SET`` other rows of its store from
    ``pool`` (never the row itself), or ``sets[store]`` when given, as two token sets and the set's size."""
    stores = x["store"].to_numpy()
    bins = np.clip(np.searchsorted(edges, y, side="right") - 1, 0, Y_BINS - 1)
    keys = x[key].to_numpy()
    if sets is None:
        sets = pd.Series(pool).groupby(stores[pool]).apply(np.asarray).to_dict()
    outcome, by_key, size = [], [], []
    for row in rows:
        candidates = sets.get(stores[row], np.empty(0, dtype=np.int64))
        candidates = candidates[candidates != row]
        limit = min(CONTEXT_SET, len(candidates))
        count = int(rng.integers(0, limit + 1)) if random_size else limit
        chosen = rng.choice(candidates, size=count, replace=False) if count else candidates[:0]
        outcome.append([f"y{bins[c]}" for c in chosen])
        by_key.append([f"{keys[c]:g}:y{bins[c]}" for c in chosen])
        size.append(float(count))
    frame = x.iloc[rows].drop(columns=["store"]).reset_index(drop=True)
    return frame.assign(context_y=outcome, context_key_y=by_key, context_rows=size)


def _in_context(table, x, y, train, test, context, unseen, seed):
    """Fit on each training row in its leave-one-out context and score the seen, unseen, and unseen-with-context rows."""
    rng = np.random.default_rng(seed + 1000)
    key = CONTEXT_KEY[table]
    edges = np.quantile(y[train], np.linspace(0.0, 1.0, Y_BINS + 1))
    fit_frame = _context_frame(x, y, train, train, key, edges, rng, random_size=True)
    model = ASRRegressor(task="gaussian", set_features=["context_y", "context_key_y"], set_max_length=CONTEXT_SET,
                         random_state=seed).fit(fit_frame, y[train])
    frames = {
        "seen": (_context_frame(x, y, test, train, key, edges, rng, random_size=False), test),
        "unseen": (_context_frame(x, y, unseen, context[:0], key, edges, rng, random_size=False), unseen),
        "unseen_context": (_context_frame(x, y, unseen, context, key, edges, rng, random_size=False), unseen),
    }
    scored = {}
    for name, (frame, rows) in frames.items():
        draws = np.asarray(model.sample(frame, n=N_DRAWS, seed=seed)).T
        mean = np.asarray(model.predict(frame))
        scored[name] = (lambda mask, mean=mean, draws=draws, rows=rows: _scores(mean[mask], None, y[rows][mask], draws[mask]))
    return model, scored


def _encoded(x, columns):
    """Categorical columns as integer codes over the whole table, for GPBoost."""
    out = x[columns].copy()
    categorical = [column for column in columns if not pd.api.types.is_numeric_dtype(out[column])]
    for column in categorical:
        out[column] = pd.Categorical(out[column]).codes.astype(np.int32)
    return out, categorical


def _gpboost(x, y, train, seed):
    import gpboost as gpb

    features = [column for column in x.columns if column != "store"]
    table, categorical = _encoded(x, features)
    rng = np.random.default_rng(seed)
    fit = rng.uniform(size=len(train)) >= 0.2
    fit_rows, valid_rows = train[fit], train[~fit]
    groups = x["store"].to_numpy()
    model = gpb.GPModel(group_data=groups[fit_rows], likelihood="gaussian")
    model.set_prediction_data(group_data_pred=groups[valid_rows])
    params = {"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 20, "verbose": -1, "seed": seed}
    train_set = gpb.Dataset(table.iloc[fit_rows], y[fit_rows], categorical_feature=categorical)
    valid_set = gpb.Dataset(table.iloc[valid_rows], y[valid_rows], categorical_feature=categorical, reference=train_set)
    booster = gpb.train(params=params, train_set=train_set, gp_model=model, num_boost_round=2000, valid_sets=[valid_set],
                        early_stopping_rounds=50, use_gp_model_for_validation=True, verbose_eval=False)
    variances = np.asarray(model.get_cov_pars()).reshape(-1)
    noise, tau2 = float(variances[0]), float(variances[1])

    def predict(rows):
        out = booster.predict(data=table.iloc[rows], group_data_pred=groups[rows], predict_var=True, pred_latent=False)
        return np.asarray(out["response_mean"]), np.sqrt(np.asarray(out["response_var"]))

    def fixed(rows):
        out = booster.predict(data=table.iloc[rows], group_data_pred=np.full(len(rows), "__unseen__", dtype=object),
                              predict_var=True, pred_latent=True)
        return np.asarray(out["fixed_effect"])

    return predict, fixed, noise, tau2


def _filled(x, train):
    """Numeric gaps as the training median, plus an indicator where training rows have gaps: patsy refuses NaN."""
    out = x.copy()
    for column in x.columns:
        if not pd.api.types.is_numeric_dtype(out[column]) or not out[column].isna().any():
            continue
        missing = out[column].isna()
        if missing.to_numpy()[train].any():
            out[f"{column}_missing"] = missing.astype(float)
        out[column] = out[column].fillna(float(np.nanmedian(out[column].to_numpy()[train])))
    return out


def _independent(matrix):
    """Columns of a largest linearly independent set, by QR with column pivoting."""
    from scipy.linalg import qr

    _, r, pivot = qr(matrix, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(r))
    return np.sort(pivot[: int(np.sum(diagonal > 1e-9 * diagonal[0]))])


def _mixedlm(x, y, train, other):
    """A linear mixed model with a random intercept per store, on dummies of every categorical but the store and
    product. The design is built once over the training rows and ``other``, the rows it will score, and its columns
    are cut to an independent set on the training rows: a category nested in a broader one (FreshRetailNet's
    third-level category in its first) makes the full design singular."""
    import patsy
    import statsmodels.api as sm

    needed = np.unique(np.concatenate([train, other]))
    local = pd.Series(np.arange(len(needed)), index=needed)
    fit_rows = local[train].to_numpy()
    frame = _filled(x.iloc[needed].reset_index(drop=True), fit_rows)
    features = [column for column in frame.columns if column not in {"store", "product"}]
    terms = [column if pd.api.types.is_numeric_dtype(frame[column]) else f"C({column})" for column in features]
    design = np.asarray(patsy.dmatrix(" + ".join(terms), frame, return_type="matrix"))
    design = design[:, _independent(design[fit_rows])]
    stores = frame["store"].to_numpy()
    result = sm.MixedLM(y[train], design[fit_rows], groups=stores[fit_rows]).fit(method=["lbfgs"])
    noise, tau2 = float(result.scale), float(np.asarray(result.cov_re).reshape(-1)[0])
    try:
        effects = {store: float(np.asarray(value).reshape(-1)[0]) for store, value in result.random_effects.items()}
    except (ValueError, np.linalg.LinAlgError):
        # The stores' variance was fitted at zero, so every store sits at the population mean.
        effects, tau2 = {}, 0.0
    counts = pd.Series(stores[fit_rows]).value_counts().to_dict()
    weights = np.asarray(result.fe_params)

    def fixed(rows):
        return design[local[rows].to_numpy()] @ weights

    def predict(rows):
        names = x["store"].to_numpy()[rows]
        offset = np.array([effects.get(store, 0.0) for store in names])
        n = np.array([counts.get(store, 0) for store in names], dtype=np.float64)
        posterior = tau2 * noise / (noise + n * tau2)
        return fixed(rows) + offset, np.sqrt(noise + posterior)

    return predict, fixed, noise, tau2


def _conjugate(fixed, noise, tau2, x, y, context, rows):
    """A random intercept's posterior from each unseen store's context residuals, and the scores' mean and sd."""
    stores = x["store"].to_numpy()
    residual = pd.Series(y[context] - fixed(context)).groupby(stores[context]).agg(["sum", "count"])
    total = residual["sum"].reindex(stores[rows]).fillna(0.0).to_numpy()
    n = residual["count"].reindex(stores[rows]).fillna(0.0).to_numpy()
    mean_b = tau2 * total / (noise + n * tau2)
    var_b = tau2 * noise / (noise + n * tau2)
    return fixed(rows) + mean_b, np.sqrt(noise + var_b)


# ---------------------------------------------------------------- run


def run(table, results, seeds, methods, redo=False):
    x, y, broader = TABLES[table]()
    entity = x["store"].to_numpy()
    out = results.setdefault(table, {"rows": int(len(y)), "stores": int(len(np.unique(entity))), "runs": {}})
    for seed in seeds:
        train, _train_count, test, test_count, context, unseen = split(entity, seed)
        run_out = out["runs"].setdefault(str(seed), {"train_rows": int(len(train)), "test_rows": int(len(test)),
                                                    "unseen_rows": int(len(unseen))})
        for method in methods:
            if method in run_out and not redo:
                continue
            started = time.perf_counter()
            if method == "in_context":
                model, scored = _in_context(table, x, y, train, test, context, unseen, seed)
                scores = _bucketed(scored["seen"], test_count)
                everything = np.ones(len(unseen), dtype=bool)
                scores["unseen"] = scored["unseen"](everything)
                scores["unseen_context"] = scored["unseen_context"](everything)
                scores["epochs"] = len(model.history_["train_loss"])
            elif method in ASHLAR_METHODS:
                model = _ashlar(method, x, y, train, seed, broader)
                seen = _ashlar_scores(model, x, y, test, seed, method == "dropped")
                cold = _ashlar_scores(model, x, y, unseen, seed, method == "dropped")
                scores = _bucketed(seen, test_count)
                scores["unseen"] = cold(np.ones(len(unseen), dtype=bool))
                if method in {"pooled", "pooled_embedding", "nested"}:
                    adapted = model.with_context(x.iloc[context].reset_index(drop=True), y[context])
                    scores["unseen_context"] = _ashlar_scores(adapted, x, y, unseen, seed, False)(np.ones(len(unseen), dtype=bool))
                scores["epochs"] = len(model.history_["train_loss"])
                if getattr(model.core_, "random_effects", None) is not None:
                    scores["tau"] = float(model.random_effects_["store"]["tau"][0])
            else:
                if method == "gpboost":
                    predict, fixed, noise, tau2 = _gpboost(x, y, train, seed)
                else:
                    predict, fixed, noise, tau2 = _mixedlm(x, y, train, np.concatenate([test, context, unseen]))
                mean, sd = predict(test)
                scores = _bucketed(lambda mask: _scores(mean[mask], sd[mask], y[test][mask]), test_count)
                mean, sd = predict(unseen)
                scores["unseen"] = _scores(mean, sd, y[unseen])
                mean, sd = _conjugate(fixed, noise, tau2, x, y, context, unseen)
                scores["unseen_context"] = _scores(mean, sd, y[unseen])
                scores["tau"] = math.sqrt(tau2)
            scores["seconds"] = time.perf_counter() - started
            run_out[method] = scores
            print(table, seed, method, round(scores["seconds"]), json.dumps({
                key: round(value["crps"], 4) for key, value in scores.items() if isinstance(value, dict)
            }), flush=True)
            _save(table, results)


def _path(table):
    return DATA_DIR / f"effects-{table}.json"


def _save(table, results):
    _path(table).write_text(json.dumps({table: results[table]}, indent=1, sort_keys=True))


def report():
    for path in sorted(DATA_DIR.glob("effects-*.json")):
        for table, data in json.loads(path.read_text()).items():
            runs = list(data["runs"].values())
            print(f"### {table}: {data['rows']:,} rows, {data['stores']:,} stores, {len(runs)} splits\n")
            columns = [f"{low}-{high}" for low, high in BUCKETS] + ["unseen", "unseen_context"]
            print("| Method | " + " | ".join(f"CRPS {c}" for c in columns) + " | Coverage unseen | Coverage 1-5 | tau |")
            print("|---|" + "---:|" * (len(columns) + 3))
            methods = [m for m in (*ASHLAR_METHODS, "in_context", "gpboost", "mixedlm") if m in runs[0]]
            for method in methods:
                cells = []
                for column in columns:
                    values = [r[method][column]["crps"] for r in runs if column in r.get(method, {})]
                    cells.append(f"{np.mean(values):.4f}" if values else "—")
                coverage = [r[method]["unseen"]["coverage"] for r in runs]
                small = [r[method]["1-5"]["coverage"] for r in runs if "1-5" in r[method]]
                tau = [r[method]["tau"] for r in runs if "tau" in r[method]]
                print(f"| {method} | " + " | ".join(cells) + f" | {np.mean(coverage):.3f} | "
                      f"{np.mean(small):.3f} | " + (f"{np.mean(tau):.3f}" if tau else "—") + " |")
            print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", nargs="*", default=list(TABLES))
    parser.add_argument("--methods", nargs="*", default=[*ASHLAR_METHODS, "in_context", "gpboost", "mixedlm"])
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--redo", action="store_true", help="refit the methods named even where results are saved")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.report:
        report()
        return
    for table in args.tables:
        path = _path(table)
        results = json.loads(path.read_text()) if path.exists() else {}
        run(table, results, args.seeds, args.methods, args.redo)


if __name__ == "__main__":
    main()
