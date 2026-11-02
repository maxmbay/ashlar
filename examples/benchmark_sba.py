"""SBA 7(a) loans: when a loan ends, by default or by payoff, from the books as they stood in 2015.

The SBA publishes every 7(a) loan it has guaranteed since 1991, with its
status as of the file's date: paid in full (with the date), charged off
(with the date and the gross amount), or still running. This benchmark
rebuilds the table as it looked on 30 September 2015. A loan that ended by
then has a known cause and time; a loan still running then is censored, its
cause and loss unknown. Every model trains on that view only. The test is
what happened next: the loans running in 2015, scored on how they ended by
the file's date in 2026.

Ashlar fits one chain, trained end to end:

    p(cause, log months, loss share | x)
        = p(cause | x) · p(log months | cause, x) · p(loss share | cause, log months, x)

with the cause two classes (payoff 0, default 1), the time a ``mixture``,
and the loss a ``hurdle`` (0 on a payoff). A running loan's label is an
unknown cause, a censored time, and an unknown loss. Lenders and six-digit
NAICS codes (within two-digit sectors) are pooled with ``random_effects``.

Baselines:

- ``finished``: a LightGBM classifier of default against payoff on the loans
  that had ended by 2015, the common analysis of this table.
- ``running_as_paid``: the same classifier with running loans counted as not defaulted.
- ``hazard``: a LightGBM discrete-time hazard on loan-quarters, three
  classes (none, default, payoff), the duration a feature: the standard
  correct treatment of censoring and competing risks with trees.
- ``aft``: XGBoost AFT on the time to default, payoff and running treated as
  censored, which ignores that payoff competes with default.

Scores, on loans running in 2015 that were not in the training sample:
Brier score and AUC of "defaulted within h years of 2015" and "paid off
within h years" for h = 1, 3, 5, the gap between the mean predicted and
the observed share, and, for Ashlar and the hazard model, the log-likelihood
of what was seen by 2026.

Run (on a machine with the files):

    python examples/benchmark_sba.py --data ~/projects/ashlar-data/sba --rows 100000 --out sba-pilot.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

SNAPSHOT = pd.Timestamp("2015-09-30")
AS_OF = pd.Timestamp("2026-09-30")
HORIZONS = (1, 3, 5)
MONTH = 30.4375
FILES = ("FOIA_7a_FY2000_FY2009_asof_260930.csv", "FOIA_7a_FY2010_FY2019_asof_260930.csv")
CATEGORICAL = [
    "fixed_or_variable", "business_type", "business_age", "sector", "state", "revolver", "collateral",
    "sold", "franchise", "processing", "same_state", "approval_fy",
]


def months_between(later, earlier):
    return (later - earlier).dt.days / MONTH


def load(data: Path, first_fy: int, last_fy: int) -> pd.DataFrame:
    frames = []
    for name in FILES:
        frame = pd.read_csv(data / name, low_memory=False)
        frames.append(frame[(frame["ApprovalFY"] >= first_fy) & (frame["ApprovalFY"] <= last_fy)])
    frame = pd.concat(frames, ignore_index=True).drop_duplicates()
    for column in ("GrossApproval", "SBAGuaranteedApproval", "GrossChargeOffAmount", "TermInMonths",
                   "InitialInterestRate", "JobsSupported", "NaicsCode", "ApprovalFY"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[frame["LoanStatus"].isin(["P I F", "CHGOFF", "EXEMPT"])]
    for column in ("ApprovalDate", "FirstDisbursementDate", "PaidInFullDate", "ChargeOffDate"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame = frame[frame["FirstDisbursementDate"].notna() & (frame["FirstDisbursementDate"] <= SNAPSHOT)]
    frame = frame[~((frame["LoanStatus"] == "CHGOFF") & frame["ChargeOffDate"].isna())]
    end = frame["PaidInFullDate"].fillna(frame["ChargeOffDate"])
    frame = frame[end.isna() | (end > frame["FirstDisbursementDate"])].copy()
    frame["end"] = frame["PaidInFullDate"].fillna(frame["ChargeOffDate"])
    frame["final_cause"] = np.where(frame["LoanStatus"] == "CHGOFF", 1.0, np.where(frame["LoanStatus"] == "P I F", 0.0, np.nan))
    frame["final_months"] = months_between(frame["end"].fillna(AS_OF), frame["FirstDisbursementDate"])
    frame["loss_share"] = np.where(
        frame["LoanStatus"] == "CHGOFF", np.clip(frame["GrossChargeOffAmount"].astype(float) / frame["GrossApproval"].astype(float), 1e-4, 1.5), 0.0
    )
    frame["ended_by_snapshot"] = frame["end"].notna() & (frame["end"] <= SNAPSHOT)
    frame["snapshot_months"] = months_between(pd.Series(SNAPSHOT, index=frame.index), frame["FirstDisbursementDate"])
    return frame.reset_index(drop=True)


def features(frame: pd.DataFrame) -> pd.DataFrame:
    naics = frame["NaicsCode"].fillna(0).astype(np.int64).astype(str)
    out = pd.DataFrame(
        {
            "log_amount": np.log(frame["GrossApproval"].astype(float).clip(lower=1000)),
            "guarantee_share": frame["SBAGuaranteedApproval"].astype(float) / frame["GrossApproval"].astype(float),
            "term": frame["TermInMonths"].astype(float),
            "rate": frame["InitialInterestRate"].astype(float),
            "log_jobs": np.log1p(frame["JobsSupported"].fillna(0).clip(lower=0)),
            "fixed_or_variable": frame["FixedorVariableInterestInd"].fillna("?").astype(str),
            "business_type": frame["BusinessType"].fillna("?").astype(str),
            "business_age": frame["BusinessAge"].fillna("?").astype(str),
            "sector": naics.str[:2],
            "state": frame["ProjectState"].fillna("?").astype(str),
            "revolver": frame["RevolverStatus"].fillna(-1).astype(str),
            "collateral": frame["CollateralInd"].fillna("?").astype(str),
            "sold": frame["SoldSecMrktInd"].fillna("?").astype(str),
            "franchise": frame["FranchiseCode"].notna().astype(str),
            "processing": frame["ProcessingMethod"].fillna("?").astype(str),
            "same_state": (frame["BankState"] == frame["ProjectState"]).astype(str),
            "approval_fy": frame["ApprovalFY"].astype(str),
            "lender": frame["BankName"].fillna("?").astype(str),
            "naics": naics,
        }
    )
    return out


def snapshot_label(frame: pd.DataFrame):
    """The 2015 view: (cause, log months, loss share) with a running loan's cells coarse."""
    from ashlar import observed

    ended = frame["ended_by_snapshot"].to_numpy()
    cause = frame["final_cause"].to_numpy()
    months = np.where(ended, frame["final_months"].to_numpy(), frame["snapshot_months"].to_numpy())
    log_months = np.log(months + 1.0)
    loss = frame["loss_share"].to_numpy()
    return observed.stack(
        [
            observed.interval(np.where(ended, cause, -np.inf), np.where(ended, cause, np.inf)),
            observed.censored(log_months, ~ended),
            observed.interval(np.where(ended, loss, -np.inf), np.where(ended, loss, np.inf)),
        ]
    )


def truth(frame: pd.DataFrame):
    """For each running loan: did it default, or pay off, within h years of the snapshot."""
    t0 = frame["snapshot_months"].to_numpy()
    t = frame["final_months"].to_numpy()
    cause = frame["final_cause"].to_numpy()
    out = {}
    for h in HORIZONS:
        within = np.isfinite(cause) & (t <= t0 + 12 * h)
        out[h] = {"default": (within & (cause == 1)).astype(float), "payoff": (within & (cause == 0)).astype(float)}
    return out


def ashlar_cif(model, X, t0, chunk: int = 4000):
    """P(cause c within h years | running at t0) from the chain's box likelihood, exact."""
    out = {h: {"default": [], "payoff": []} for h in HORIZONS}
    for start in range(0, len(t0), chunk):
        rows = slice(start, start + chunk)
        dist = model.predict_dist(X.iloc[rows])
        a = np.log(t0[rows] + 1.0)
        n = len(a)
        inf = np.full(n, np.inf, dtype=np.float32)

        def box(lower, upper, dist=dist):
            return np.asarray(dist.log_box(np.stack(lower, 1).astype(np.float32), np.stack(upper, 1).astype(np.float32)))

        alive = box([-inf, a, -inf], [inf, inf, inf])
        for h in HORIZONS:
            b = np.log(t0[rows] + 12 * h + 1.0)
            for name, code in (("payoff", 0.0), ("default", 1.0)):
                c = np.full(n, code)
                out[h][name].append(np.exp(box([c, a, -inf], [c, b, inf]) - alive))
    return {h: {k: np.concatenate(v) for k, v in parts.items()} for h, parts in out.items()}


def quarter_rows(X: pd.DataFrame, label_frame: pd.DataFrame, ended: np.ndarray):
    """Loan-quarters for the discrete-time hazard, as of the snapshot."""
    months = np.where(ended, label_frame["final_months"], label_frame["snapshot_months"]).astype(float)
    quarters = np.floor(months / 3.0).astype(int)
    index = np.repeat(np.arange(len(X)), quarters + 1)
    age = np.concatenate([np.arange(q + 1) for q in quarters])
    event = np.zeros(len(index), dtype=int)
    last = np.cumsum(quarters + 1) - 1
    cause = label_frame["final_cause"].to_numpy()
    event[last[ended]] = np.where(cause[ended] == 1, 1, 2)
    rows = X.iloc[index].reset_index(drop=True).copy()
    rows["age"] = age
    return rows, event


def hazard_cif(model, X, t0):
    start = np.floor(t0 / 3.0).astype(int)
    cumulative = {h: {"default": np.zeros(len(X)), "payoff": np.zeros(len(X))} for h in HORIZONS}
    alive = np.ones(len(X))
    for step in range(4 * max(HORIZONS)):
        rows = X.copy()
        rows["age"] = start + step
        p = model.predict_proba(rows)
        default, payoff = alive * p[:, 1], alive * p[:, 2]
        alive = alive * p[:, 0]
        for h in HORIZONS:
            if step < 4 * h:
                cumulative[h]["default"] += default
                cumulative[h]["payoff"] += payoff
    return cumulative


def ashlar_loss(model, X, t0, draws: int = 400, chunk: int = 2000, seed: int = 0):
    """Expected loss share within the last horizon, given the loan was running at t0, from joint draws."""
    out = []
    horizon = 12 * max(HORIZONS)
    for start in range(0, len(t0), chunk):
        rows = slice(start, start + chunk)
        sample = np.asarray(model.sample(X.iloc[rows], n=draws, seed=seed))  # (draws, rows, 3)
        months = np.exp(sample[..., 1]) - 1.0
        alive = months > t0[rows][None, :]
        within = alive & (sample[..., 0] == 1) & (months <= t0[rows][None, :] + horizon)
        loss = np.where(within, np.clip(sample[..., 2], 0.0, 1.5), 0.0).sum(axis=0)
        out.append(loss / np.maximum(alive.sum(axis=0), 1))
    return np.concatenate(out)


def loss_scores(pred, actual):
    order = np.argsort(pred)
    bins = np.array_split(order, 10)
    gap = sum(len(b) * abs(pred[b].mean() - actual[b].mean()) for b in bins) / len(pred)
    return {
        "mse": float(np.mean((pred - actual) ** 2)),
        "mean_pred": float(pred.mean()),
        "observed": float(actual.mean()),
        "decile_gap": float(gap),
        "top_decile_pred": float(pred[bins[-1]].mean()),
        "top_decile_observed": float(actual[bins[-1]].mean()),
    }


def scores(pred, true):
    from sklearn.metrics import roc_auc_score

    out = {}
    for h in HORIZONS:
        for name in ("default", "payoff"):
            if pred[h].get(name) is None:
                continue
            p, y = np.clip(pred[h][name], 0, 1), true[h][name]
            out[f"{name}_{h}y"] = {
                "brier": float(np.mean((p - y) ** 2)),
                "auc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else None,
                "mean_pred": float(p.mean()),
                "observed": float(y.mean()),
            }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=0, help="training loans; 0 uses every loan not in the test set")
    parser.add_argument("--test", type=int, default=50_000)
    parser.add_argument("--first-fy", type=int, default=2010)
    parser.add_argument("--last-fy", type=int, default=2015)
    parser.add_argument("--test-first-fy", type=int, default=2010)
    parser.add_argument("--snapshot", default="2015-09-30", help="date the books are frozen; pair it with --last-fy")
    parser.add_argument("--folds", type=int, default=0, help="split the running loans into this many test folds (overrides --test)")
    parser.add_argument("--fold", type=int, default=0, help="which fold is the test set; --seed fixes the split")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", default="finished,running_as_paid,hazard,aft,ashlar,ashlar_pooled")
    parser.add_argument("--out", type=Path, default=Path("sba-pilot.json"))
    parser.add_argument("--preds", type=Path, default=None, help="also write per-loan test predictions to this CSV")
    args = parser.parse_args()
    wanted = set(args.models.split(","))
    global SNAPSHOT
    SNAPSHOT = pd.Timestamp(args.snapshot)
    if SNAPSHOT + pd.DateOffset(years=max(HORIZONS)) > AS_OF:
        raise SystemExit(f"--snapshot {args.snapshot} leaves under {max(HORIZONS)} years before the data ends ({AS_OF.date()})")

    frame = load(args.data, args.first_fy, args.last_fy)
    rng = np.random.default_rng(args.seed)
    # Test loans first: running at the snapshot and approved from --test-first-fy on,
    # so the test set stays the same as the training years widen. Training takes
    # every other loan (or --rows of them when --rows > 0).
    pool = np.flatnonzero(~frame["ended_by_snapshot"].to_numpy() & (frame["ApprovalFY"].to_numpy() >= args.test_first_fy))
    if args.folds > 0:
        # K-fold over the running loans: fold --fold is the test set, every other loan trains.
        test_index = np.sort(np.array_split(rng.permutation(pool), args.folds)[args.fold])
    else:
        test_index = rng.permutation(pool)[: args.test]
    rest = np.setdiff1d(np.arange(len(frame)), test_index)
    rest = rng.permutation(rest)
    if args.rows > 0:
        rest = rest[: args.rows]
    train = frame.iloc[rest].reset_index(drop=True)
    test = frame.iloc[test_index].reset_index(drop=True)
    X_train, X_test = features(train), features(test)
    t0 = test["snapshot_months"].to_numpy()
    true = truth(test)
    last = max(HORIZONS)
    actual_loss = np.where(true[last]["default"] == 1, test["loss_share"].to_numpy(), 0.0)
    ended = train["ended_by_snapshot"].to_numpy()
    summary = {
        "train_loans": len(train), "train_ended": int(ended.sum()), "train_defaults": int((ended & (train["final_cause"] == 1)).sum()),
        "test_running_loans": len(test), "fold": args.fold if args.folds > 0 else None, "folds": args.folds, "observed": {f"{h}y": {k: float(v.mean()) for k, v in true[h].items()} for h in HORIZONS},
    }
    results = {"summary": summary, "models": {}}
    print(json.dumps(summary, indent=1))
    plain = [c for c in X_train.columns if c not in ("lender", "naics")]

    # Per-loan test predictions, for calibration checks by vintage, lender or industry.
    preds = None
    if args.preds:
        preds = X_test[["approval_fy", "sector", "naics", "lender"]].copy()
        preds["snapshot_months"] = t0
        for h in HORIZONS:
            preds[f"default_{h}y"] = true[h]["default"]
        preds["loss_5y"] = actual_loss

    def record(name, pred, seconds, extra=None):
        results["models"][name] = {"seconds": seconds, "scores": scores(pred, true), **(extra or {})}
        if preds is not None:
            for h in HORIZONS:
                preds[f"{name}_default_{h}y"] = pred[h]["default"]
            preds.to_csv(args.preds, index=False)
        print(name, json.dumps(results["models"][name]["scores"].get("default_5y")), f"{seconds:.0f}s", flush=True)
        args.out.write_text(json.dumps(results, indent=1))

    # One categorical coding for the trees, shared by the training and test rows.
    both = pd.concat([X_train[plain], X_test[plain]], ignore_index=True)
    for c in CATEGORICAL:
        both[c] = both[c].astype(str).astype("category")
    tree_train, tree_test = both.iloc[: len(X_train)].reset_index(drop=True), both.iloc[len(X_train) :].reset_index(drop=True)

    import lightgbm as lgb

    gbm = {"n_estimators": 2000, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50, "verbose": -1}
    if wanted & {"finished", "running_as_paid"}:
        Xt, Xs = tree_train, tree_test
        for name in ("finished", "running_as_paid"):
            if name not in wanted:
                continue
            start = time.time()
            rows = ended if name == "finished" else np.ones(len(train), bool)
            y = (train["final_cause"].to_numpy() == 1) & ended
            fit_rows = np.flatnonzero(rows)
            valid = rng.random(len(fit_rows)) < 0.1
            model = lgb.LGBMClassifier(**gbm).fit(
                Xt.iloc[fit_rows[~valid]], y[fit_rows[~valid]], eval_set=[(Xt.iloc[fit_rows[valid]], y[fit_rows[valid]])],
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            p = model.predict_proba(Xs)[:, 1]
            # A classifier of "ever defaults" has no horizon: the same number at every h, and no payoff.
            pred = {h: {"default": p, "payoff": None} for h in HORIZONS}
            record(name, pred, time.time() - start)

    if "hazard" in wanted:
        start = time.time()
        rows, event = quarter_rows(tree_train, train, ended)
        valid = rng.random(len(rows)) < 0.1
        model = lgb.LGBMClassifier(objective="multiclass", **gbm).fit(
            rows[~valid], event[~valid], eval_set=[(rows[valid], event[valid])], callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        cif = hazard_cif(model, tree_test, t0)
        # Expected loss the usual way: probability of default times a LightGBM loss share fitted on defaults.
        defaults = np.flatnonzero(ended & (train["final_cause"].to_numpy() == 1))
        severity = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20, verbose=-1)
        severity.fit(tree_train.iloc[defaults], train["loss_share"].to_numpy()[defaults])
        expected = cif[last]["default"] * np.clip(severity.predict(tree_test), 0.0, 1.5)
        record("hazard", cif, time.time() - start, {"quarter_rows": len(rows), "loss_5y": loss_scores(expected, actual_loss)})

    if "aft" in wanted:
        import xgboost as xgb

        start = time.time()
        dummies = pd.get_dummies(both, dummy_na=False).astype(float)
        Xt, Xs = dummies.iloc[: len(X_train)], dummies.iloc[len(X_train) :]
        months = np.where(ended, train["final_months"], train["snapshot_months"]) + 1.0
        default = ended & (train["final_cause"].to_numpy() == 1)
        dtrain = xgb.DMatrix(Xt, label_lower_bound=months, label_upper_bound=np.where(default, months, np.inf))
        params = {"objective": "survival:aft", "aft_loss_distribution": "normal", "aft_loss_distribution_scale": 1.0,
                  "learning_rate": 0.05, "max_depth": 6, "tree_method": "hist", "eval_metric": "aft-nloglik"}
        booster = xgb.train(params, dtrain, num_boost_round=600)
        mu = booster.predict(xgb.DMatrix(Xs), output_margin=True)
        from scipy.stats import norm

        def surv(m):
            return norm.sf((np.log(m) - mu) / 1.0)

        pred = {h: {"default": 1 - surv(t0 + 12 * h + 1) / surv(t0 + 1), "payoff": None} for h in HORIZONS}
        record("aft", pred, time.time() - start)

    from ashlar import ASRRegressor

    label = snapshot_label(train)
    columns = [{"kind": "categorical", "classes": 2}, "mixture", "hurdle"]
    for name, effects in (("ashlar", None), ("ashlar_pooled", {"sector": ["intercept", "embedding"],
                                                               "naics": {"terms": ["intercept", "embedding"], "within": "sector"},
                                                               "lender": ["intercept", "embedding"]})):
        if name not in wanted:
            continue
        start = time.time()
        cats = [c for c in CATEGORICAL] + ["lender", "naics"]
        model = ASRRegressor(task="chain", head_kwargs={"columns": columns}, categorical=cats, random_effects=effects,
                             max_epochs=args.epochs, random_state=args.seed).fit(X_train, label)
        history = getattr(model, "history_", {}) or {}
        epochs = max([len(v) for v in history.values() if isinstance(v, list)] or [0])
        cif = ashlar_cif(model, X_test, t0)
        loss = loss_scores(ashlar_loss(model, X_test, t0, seed=args.seed), actual_loss)
        record(name, cif, time.time() - start, {"epochs": epochs, "loss_5y": loss})


if __name__ == "__main__":
    main()
