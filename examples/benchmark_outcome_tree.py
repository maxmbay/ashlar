"""Nested ball outcomes in T20 cricket: one outcome tree beside flat and per-node models.

Data: Cricsheet IPL ball-by-ball JSON (https://cricsheet.org, ODbL), downloaded
once into the bench cache. Each row is one delivery. The features are known
before the ball is bowled: innings, over, legal balls so far, score and
wickets, chase target and required rate, the batter's and bowler's figures so
far in this match, and the batter, bowler, venue and teams as categories.

The outcome tree (12 leaves)::

    wicket:     caught, bowled, lbw, other_out
    no_wicket:  dot
                extra:  wide, noball, bye
                runs:   boundary: four, six
                        ran:      single, multiple

Split by whole seasons, so no match is on both sides: train through 2023,
validation (early stopping) 2024, test 2025 onward.

Models:

* ``lgbm_flat``: one LightGBM multiclass over the 12 leaves.
* ``lgbm_nodes``: one LightGBM per internal node, on the rows that reach it;
  a leaf is the product of its branch probabilities.
* ``ashlar_flat``: ``classifier()`` over the 12 leaves.
* ``ashlar_tree``: ``outcome_tree(TREE)``: one backbone, one branch softmax per node.
* ``prior``: training frequencies of each branch, no features.

``--coarse YEAR`` keeps only depth-one labels (``wicket``, ``extra``,
``runs``, ``dot``) for training seasons up to YEAR, the shape of older data
that did not record the detail. The flat models must drop those seasons; the
tree and the per-node models use them for the splits they reach.

``--season`` adds season context; every option is known before a match:

* ``none`` (default): no season information.
* ``embed``: the season as a learned category. A season the model never
  trained on (validation and test) reuses the latest training season.
* ``era``: an impact-player-rule flag (2023 on) and the previous season's
  league-wide wicket, dot, boundary and extra rates.
* ``both``: ``embed`` and ``era`` together.

Reported on test: leaf log loss, the conditional log loss at each internal
node (on the test rows that reach it), and expected calibration error of
P(wicket), P(dot), and P(boundary) with 20 equal-mass bins.

    python examples/benchmark_outcome_tree.py --models prior lgbm_flat lgbm_nodes ashlar_flat ashlar_tree
    python examples/benchmark_outcome_tree.py --coarse 2015
    python examples/benchmark_outcome_tree.py --season both
    python examples/benchmark_outcome_tree.py --report
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
import urllib.request
import warnings
import zipfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

CACHE = Path(os.environ.get("ASHLAR_BENCH_CACHE", Path.home() / ".cache" / "ashlar-bench"))
OUT = CACHE / "outcome_tree.jsonl"
URL = "https://cricsheet.org/downloads/ipl_json.zip"
GBM_THREADS = int(os.environ.get("GBM_THREADS", "4"))

TREE = {
    "wicket": ["caught", "bowled", "lbw", "other_out"],
    "no_wicket": {
        "dot": None,
        "extra": ["wide", "noball", "bye"],
        "runs": {"boundary": ["four", "six"], "ran": ["single", "multiple"]},
    },
}
CATEGORICAL = ["batter", "bowler", "venue", "batting_team", "bowling_team"]
TRAIN_LAST, VAL_SEASON = 2023, 2024
# Depth-one label for each leaf, used by --coarse.
COARSE = {
    "caught": "wicket", "bowled": "wicket", "lbw": "wicket", "other_out": "wicket",
    "wide": "extra", "noball": "extra", "bye": "extra",
    "four": "runs", "six": "runs", "single": "runs", "multiple": "runs",
    "dot": "dot",
}


# ---------------------------------------------------------------- data


def _download() -> Path:
    folder = CACHE / "cricsheet" / "ipl"
    if not any(folder.glob("*.json")):
        folder.mkdir(parents=True, exist_ok=True)
        archive = folder.parent / "ipl_json.zip"
        urllib.request.urlretrieve(URL, archive)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(folder)
    return folder


def _label(ball) -> str:
    wickets = ball.get("wickets", [])
    if wickets:
        kind = wickets[0]["kind"]
        if kind in {"caught", "caught and bowled"}:
            return "caught"
        if kind in {"bowled", "lbw"}:
            return kind
        return "other_out"
    extras = ball.get("extras", {})
    if "wides" in extras:
        return "wide"
    if "noballs" in extras:
        return "noball"
    if "byes" in extras or "legbyes" in extras or "penalty" in extras:
        return "bye"
    runs = ball["runs"]["batter"]
    if runs == 0:
        return "dot"
    if runs == 4:
        return "four"
    if runs == 6:
        return "six"
    return "single" if runs == 1 else "multiple"


def load_balls() -> pd.DataFrame:
    cached = CACHE / "cricsheet" / "ipl_balls.pkl"
    if cached.exists():
        return pd.read_pickle(cached)
    rows = []
    for path in sorted(glob.glob(str(_download() / "*.json"))):
        match = json.loads(Path(path).read_text())
        info = match["info"]
        if info.get("match_type") != "T20" or info.get("balls_per_over", 6) != 6:
            continue
        season = int(info["dates"][0][:4])
        teams = info["teams"]
        first_total = None
        for number, innings in enumerate(match["innings"][:2], start=1):
            if innings.get("super_over"):
                continue
            batting = innings["team"]
            bowling = teams[1] if teams[0] == batting else teams[0]
            target = innings.get("target", {}).get("runs") if number == 2 else None
            if number == 2 and target is None and first_total is not None:
                target = first_total + 1
            overs_limit = innings.get("target", {}).get("overs", 20) if number == 2 else 20
            runs = wickets = legal = 0
            batter_runs, batter_balls, bowler_balls, bowler_runs = {}, {}, {}, {}
            last = "start"
            for over in innings["overs"]:
                legal_in_over = 0
                for ball in over["deliveries"]:
                    batter, bowler = ball["batter"], ball["bowler"]
                    remaining = max(int(round(overs_limit * 6)) - legal, 1)
                    rows.append(
                        {
                            "match": path.rsplit("/", 1)[-1],
                            "season": season,
                            "innings": number,
                            "over": over["over"],
                            "ball_in_over": legal_in_over,
                            "legal_balls": legal,
                            "runs": runs,
                            "wickets": wickets,
                            "run_rate": 6 * runs / max(legal, 1),
                            "target": np.nan if target is None else float(target),
                            "required_rate": np.nan if target is None else 6 * (target - runs) / remaining,
                            "batter_runs": batter_runs.get(batter, 0),
                            "batter_balls": batter_balls.get(batter, 0),
                            "bowler_balls": bowler_balls.get(bowler, 0),
                            "bowler_runs": bowler_runs.get(bowler, 0),
                            "previous": last,
                            "batter": batter,
                            "bowler": bowler,
                            "venue": info.get("venue", "unknown").split(",")[0],
                            "batting_team": batting,
                            "bowling_team": bowling,
                            "y": _label(ball),
                        }
                    )
                    extras = ball.get("extras", {})
                    is_legal = "wides" not in extras and "noballs" not in extras
                    runs += ball["runs"]["total"]
                    wickets += len(ball.get("wickets", []))
                    batter_runs[batter] = batter_runs.get(batter, 0) + ball["runs"]["batter"]
                    if "wides" not in extras:
                        batter_balls[batter] = batter_balls.get(batter, 0) + 1
                    bowler_runs[bowler] = bowler_runs.get(bowler, 0) + ball["runs"]["total"]
                    if is_legal:
                        legal += 1
                        legal_in_over += 1
                        bowler_balls[bowler] = bowler_balls.get(bowler, 0) + 1
                    last = rows[-1]["y"]
            if number == 1:
                first_total = runs
    frame = pd.DataFrame(rows)
    for column in [*CATEGORICAL, "previous"]:
        frame[column] = frame[column].astype("category")
    frame.to_pickle(cached)
    return frame


def split(frame, coarse: int | None):
    train = frame[frame.season <= TRAIN_LAST].copy()
    val = frame[frame.season == VAL_SEASON].copy()
    test = frame[frame.season > VAL_SEASON].copy()
    train["y"] = train["y"].astype(object)
    train["coarse_row"] = False
    if coarse is not None:
        old = train.season <= coarse
        train.loc[old, "y"] = train.loc[old, "y"].map(COARSE)
        train.loc[old, "coarse_row"] = True
    return train, val, test


def add_season(frame, train, val, test, mode):
    """Season columns for ``mode`` and the names to add to ``FEATURES``."""
    extra = []
    if mode in {"embed", "both"}:
        last = str(int(train.season.max()))
        seen = sorted({str(int(v)) for v in train.season.unique()})
        for part in (train, val, test):
            name = part.season.astype(int).astype(str)
            part["season_cat"] = pd.Categorical(name.where(name.isin(seen), last), categories=seen)
        extra.append("season_cat")
    if mode in {"era", "both"}:
        paths = frame.y.astype(str).map(lambda v: _tree().path(v))
        flags = pd.DataFrame(
            {event: paths.map(lambda p, e=event: e in p) for event in ("wicket", "dot", "boundary", "extra")}
        )
        rates = flags.groupby(frame.season.values).mean()
        previous = rates.shift(1).add_prefix("last_season_")
        for part in (train, val, test):
            part["impact_player"] = (part.season >= 2023).astype(float)
            for column in previous.columns:
                part[column] = part.season.map(previous[column]).astype(float)
        extra += ["impact_player", *previous.columns]
    return extra


FEATURES = [
    "innings", "over", "ball_in_over", "legal_balls", "runs", "wickets", "run_rate", "target",
    "required_rate", "batter_runs", "batter_balls", "bowler_balls", "bowler_runs", "previous", *CATEGORICAL,
]


# ---------------------------------------------------------------- models


def _tree():
    from ashlar.tree import OutcomeTree

    return OutcomeTree(TREE)


def _from_conditionals(tree, conditionals, n):
    """Node marginals ``(n, node)`` from ``{internal node: (n, children)}`` conditionals."""
    nodes = np.ones((n, tree.n_nodes))
    for node in tree.internal:  # parents come first in depth-first order
        for slot, child in enumerate(tree.children[node]):
            nodes[:, child] = nodes[:, node] * conditionals[node][:, slot]
    return nodes


def _leaf_rows(train, tree):
    """Rows a flat model can use: whole seasons with full labels.

    A coarse season still has ``dot`` as a leaf label; keeping only those rows
    would bias a flat model toward dots, so the whole season is dropped.
    """
    return train[~train.coarse_row & train.y.isin(tree.leaf_names)]


def fit_prior(train, val, test, tree):
    from ashlar.tree import empirical_edge_prior

    prior = np.asarray(empirical_edge_prior(tree, tree.encode(train.y.tolist()), np.ones(len(train))))
    conditionals, start = {}, 0
    for node in tree.internal:
        width = len(tree.children[node])
        conditionals[node] = np.repeat(prior[None, start : start + width], len(test), axis=0)
        start += width
    return _from_conditionals(tree, conditionals, len(test))


def _lgbm(n_classes):
    import lightgbm as lgb

    params = dict(
        n_estimators=4000, learning_rate=0.03, num_leaves=31, min_child_samples=100,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, cat_smooth=50, min_data_per_group=200,
        reg_lambda=1.0, n_jobs=GBM_THREADS, verbose=-1,
    )
    if n_classes > 2:
        params["objective"] = "multiclass"
    return lgb.LGBMClassifier(**params)


def _fit_lgbm(X, y, Xv, yv, classes):
    import lightgbm as lgb

    model = _lgbm(len(classes))
    y_index = pd.Categorical(y, categories=classes).codes
    yv_index = pd.Categorical(yv, categories=classes).codes
    keep = yv_index >= 0
    model.fit(
        X, y_index, eval_set=[(Xv[keep], yv_index[keep])],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def _proba_full(model, X, n_classes):
    """``predict_proba`` with a column for every class index, even ones unseen in training."""
    proba = model.predict_proba(X)
    out = np.zeros((len(X), n_classes))
    out[:, model.classes_.astype(int)] = proba
    return out


def fit_lgbm_flat(train, val, test, tree):
    leaves = tree.leaf_names
    train = _leaf_rows(train, tree)
    model = _fit_lgbm(train[FEATURES], train.y, val[FEATURES], val.y, leaves)
    proba = _proba_full(model, test[FEATURES], len(leaves))
    return _nodes_from_leaves(tree, proba)


def _nodes_from_leaves(tree, leaf_proba):
    nodes = np.zeros((len(leaf_proba), tree.n_nodes))
    nodes[:, list(tree.leaves)] = leaf_proba
    for node in reversed(range(1, tree.n_nodes)):
        nodes[:, tree.parent[node]] += nodes[:, node] if tree.parent[node] else 0
    nodes[:, 0] = 1.0
    return nodes


def _reaching_child(tree, labels, node):
    """For each label at or below ``node``, the child of ``node`` on its path; else None."""
    out = []
    for label in labels:
        walk = tree.node(label)
        child = None
        while walk != 0:
            if tree.parent[walk] == node:
                child = tree.names[walk]
                break
            walk = tree.parent[walk]
        out.append(child)
    return np.asarray(out, dtype=object)


def fit_lgbm_nodes(train, val, test, tree):
    conditionals = {}
    for node in tree.internal:
        kids = [tree.names[c] for c in tree.children[node]]
        child = _reaching_child(tree, train.y.tolist(), node)
        child_val = _reaching_child(tree, val.y.tolist(), node)
        keep, keep_val = child != None, child_val != None  # noqa: E711
        model = _fit_lgbm(train[FEATURES][keep], child[keep], val[FEATURES][keep_val], child_val[keep_val], kids)
        conditionals[node] = _proba_full(model, test[FEATURES], len(kids))
    return _from_conditionals(tree, conditionals, len(test))


def _ashlar_kwargs(train):
    categorical = [*CATEGORICAL, "previous", *[c for c in FEATURES if c == "season_cat"]]
    return dict(categorical=categorical, random_state=0)


def fit_ashlar_flat(train, val, test, tree):
    from ashlar import classifier

    train = _leaf_rows(train, tree)
    model = classifier(**_ashlar_kwargs(train)).fit(train[FEATURES], train.y.astype(str).to_numpy(), validation=(val[FEATURES], val.y.astype(str).to_numpy()))
    proba = model.predict_proba(test[FEATURES])
    full = np.zeros((len(test), len(tree.leaf_names)))
    for column, name in enumerate(model.classes_):
        full[:, tree.leaf_names.index(name)] = proba[:, column]
    return _nodes_from_leaves(tree, full)


def fit_ashlar_tree(train, val, test, tree):
    from ashlar import outcome_tree

    model = outcome_tree(TREE, **_ashlar_kwargs(train)).fit(
        train[FEATURES], train.y.to_numpy(dtype=object), validation=(val[FEATURES], val.y.to_numpy(dtype=object))
    )
    return model.predict_node_proba(test[FEATURES])


MODELS = {
    "prior": fit_prior,
    "lgbm_flat": fit_lgbm_flat,
    "lgbm_nodes": fit_lgbm_nodes,
    "ashlar_flat": fit_ashlar_flat,
    "ashlar_tree": fit_ashlar_tree,
}


# ---------------------------------------------------------------- scores


def _ece(p, hit, bins=20):
    order = np.argsort(p)
    total = 0.0
    for chunk in np.array_split(order, bins):
        total += len(chunk) * abs(p[chunk].mean() - hit[chunk].mean())
    return total / len(p)


def scores(tree, nodes, labels):
    eps = 1e-12
    index = tree.encode(labels)
    out = {"leaf_logloss": float(-np.mean(np.log(np.clip(nodes[np.arange(len(index)), index], eps, None))))}
    for node in tree.internal:
        child = _reaching_child(tree, labels, node)
        keep = child != None  # noqa: E711
        kids = list(tree.children[node])
        cond = nodes[keep][:, kids] / np.clip(nodes[keep][:, [node]], eps, None)
        slot = np.asarray([kids.index(tree.node(c)) for c in child[keep]])
        name = "root" if node == 0 else tree.names[node]
        out[f"cond_logloss/{name}"] = float(-np.mean(np.log(np.clip(cond[np.arange(len(slot)), slot], eps, None))))
    paths = [tree.path(label) for label in labels]
    for event in ("wicket", "dot", "boundary"):
        hit = np.asarray([event in path for path in paths], dtype=float)
        p = nodes[:, tree.node(event)]
        out[f"ece/{event}"] = float(_ece(p, hit))
        out[f"rate/{event}"] = float(hit.mean())
        out[f"mean_p/{event}"] = float(p.mean())
    return out


# ---------------------------------------------------------------- driver


def _done():
    if not OUT.exists():
        return set()
    return {(r["model"], r["coarse"], r.get("season", "none")) for r in map(json.loads, OUT.read_text().splitlines())}


def run(models, coarse, season="none"):
    global FEATURES
    frame = load_balls()
    train, val, test = split(frame, coarse)
    FEATURES = [*FEATURES, *add_season(frame, train, val, test, season)]
    tree = _tree()
    labels = test.y.astype(str).tolist()
    done = _done()
    print(f"train {len(train):,} val {len(val):,} test {len(test):,} balls", flush=True)
    for name in models:
        if (name, coarse, season) in done:
            print(f"skip {name} coarse={coarse}")
            continue
        start = time.time()
        nodes = MODELS[name](train, val, test, tree)
        record = {"model": name, "coarse": coarse, "season": season, "seconds": round(time.time() - start, 1), **scores(tree, nodes, labels)}
        with OUT.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


def report():
    rows = [json.loads(line) for line in OUT.read_text().splitlines()]
    table = pd.DataFrame(rows)
    table["season"] = table.get("season", pd.Series(index=table.index, dtype=object)).fillna("none")
    table = table.set_index(["coarse", "season", "model"]).sort_index()
    columns = ["leaf_logloss", *[c for c in table.columns if c.startswith("cond_")], "ece/wicket", "ece/dot", "ece/boundary", "seconds"]
    print(table[columns].round(4).to_markdown())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--coarse", type=int, default=None)
    parser.add_argument("--season", choices=["none", "embed", "era", "both"], default="none")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.report:
        report()
    else:
        run(args.models, args.coarse, args.season)
