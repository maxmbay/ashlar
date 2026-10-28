"""Generate the figures embedded in the README.

CPU, small series, a few epochs. Writes PNGs under docs/images/.
Does not fit a benchmark and does not publish the package.

    JAX_PLATFORMS=cpu python examples/figures.py
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

from ashlar import TabularEstimator, TimeSeriesForecaster, ordinal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"

PAPER = "#f4f1ea"
INK = "#1c1917"
MUTED = "#78716c"
RULE = "#d6d1c7"
TEAL = "#0f6e6e"
CLAY = "#c2410c"
SLATE = "#3d4f5f"

EFFECT = LinearSegmentedColormap.from_list("ashlar_effect", ["#0f6e6e", "#5c5346", "#c2410c"])


def _fonts() -> None:
    directory = Path("/usr/share/fonts/truetype/macos")
    for name in ("Inter-Regular.ttf", "Inter-Medium.ttf", "Inter-SemiBold.ttf"):
        path = directory / name
        if path.exists():
            font_manager.fontManager.addfont(str(path))


def apply_style() -> None:
    _fonts()
    plt.rcParams.update(
        {
            "font.family": "Inter",
            "font.size": 11,
            "text.color": INK,
            "axes.titlesize": 14,
            "axes.titleweight": "medium",
            "axes.titlecolor": INK,
            "axes.labelsize": 11,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.6,
            "axes.edgecolor": "#a8a29e",
            "axes.facecolor": PAPER,
            "axes.unicode_minus": False,
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "legend.labelcolor": INK,
        }
    )


def _panel(title: str):
    fig, ax = plt.subplots(figsize=(7.6, 4.45), constrained_layout=True)
    ax.set_title(title, loc="left", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color=RULE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.6, color="#a8a29e")
    return fig, ax


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def _legend(ax, loc: str = "best") -> None:
    ax.legend(loc=loc, handlelength=1.8, borderaxespad=0.15, labelspacing=0.35)


def tabular_fit() -> None:
    rng = np.random.default_rng(0)
    n = 180
    x = rng.uniform(-2.8, 2.8, size=n).astype(np.float32)
    y = np.sin(1.35 * x) + 0.28 * x
    y = (y + rng.normal(0.0, 0.16, size=n)).astype(np.float32)
    model = TabularEstimator(
        max_epochs=24,
        patience=8,
        batch_size=32,
        learning_rate=6e-3,
        backbone_kwargs={"n_blocks": 2, "d_block": 48, "dropout": 0.0},
        validation_size=0.2,
        random_state=0,
    )
    model.fit({"x": x}, y)
    grid = np.linspace(-2.8, 2.8, 240, dtype=np.float32)
    predicted = np.asarray(model.predict({"x": grid}), dtype=np.float64)
    truth = np.sin(1.35 * grid) + 0.28 * grid
    mae = float(np.mean(np.abs(predicted - truth)))
    print(f"tabular curve mae {mae:.4f} epochs {len(model.history_['train_loss'])}", flush=True)

    fig, ax = _panel("The fit follows the curve in the table")
    ax.scatter(x, y, s=22, color=TEAL, alpha=0.72, linewidths=0, label="Rows", zorder=2)
    ax.plot(grid, predicted, color=CLAY, linewidth=2.2, label="Prediction", zorder=3)
    ax.set_xlabel("x")
    ax.set_ylabel("target")
    ax.set_xlim(-3.05, 3.05)
    _legend(ax, "upper left")
    _save(fig, "tabular-fit.png")


def categorical_embeddings() -> None:
    effect = {
        "harbor": 1.7,
        "wharf": 1.45,
        "pier": 1.2,
        "plaza": 0.45,
        "square": 0.2,
        "court": -0.05,
        "mill": -0.4,
        "works": -0.65,
        "grove": -1.25,
        "field": -1.5,
        "meadow": -1.75,
    }
    names = list(effect)
    rng = np.random.default_rng(1)
    per = 40
    city = np.repeat(np.array(names, dtype=object), per)
    rooms = rng.normal(0.0, 1.0, size=len(city)).astype(np.float32)
    signal = np.array([effect[item] for item in city], dtype=np.float32)
    y = (0.3 * rooms + signal + rng.normal(0.0, 0.12, size=len(city))).astype(np.float32)
    order = rng.permutation(len(city))
    model = TabularEstimator(
        max_epochs=36,
        patience=10,
        batch_size=48,
        learning_rate=1e-2,
        weight_decay=1e-5,
        categorical_embedding_dim=2,
        backbone_kwargs={"n_blocks": 1, "d_block": 8, "dropout": 0.0},
        validation_size=0.15,
        random_state=0,
    )
    model.fit({"rooms": rooms[order], "city": city[order]}, y[order])
    column = model.encoder_.embedding_columns[0]
    table = np.asarray(model.core_.categorical_embeddings.tables[0], dtype=np.float64)
    vectors = []
    effects = []
    labels = []
    for index, name in enumerate(column.categories):
        vectors.append(table[index + 1])
        effects.append(effect[name])
        labels.append(name)
    vectors = np.vstack(vectors)
    effects = np.asarray(effects)
    centered = vectors - vectors.mean(axis=0)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    projection = centered @ direction
    if np.corrcoef(projection, effects)[0, 1] < 0:
        projection = -projection
    corr = float(np.corrcoef(projection, effects)[0, 1])
    print(f"embedding effect correlation {corr:.3f} spread {vectors.std():.3f}", flush=True)

    fig, ax = _panel("Categories with the same effect sit together")
    points = ax.scatter(
        vectors[:, 0],
        vectors[:, 1],
        c=effects,
        cmap=EFFECT,
        s=78,
        zorder=3,
        edgecolors="white",
        linewidths=0.8,
    )
    center = vectors.mean(axis=0)
    for name, vector in zip(labels, vectors):
        offset = vector - center
        norm = float(np.linalg.norm(offset)) + 1e-6
        offset = offset / norm
        ax.annotate(
            name,
            vector,
            xytext=(9 * offset[0], 8 * offset[1]),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8.5,
            color=INK,
            zorder=4,
        )
    colorbar = fig.colorbar(points, ax=ax, fraction=0.046, pad=0.03)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0, labelsize=9)
    colorbar.set_label("effect on the target", color=INK)
    ax.set_xlabel("embedding axis 1")
    ax.set_ylabel("embedding axis 2")
    span = np.ptp(vectors, axis=0)
    pad = np.maximum(span * 0.28, 0.15)
    ax.set_xlim(vectors[:, 0].min() - pad[0], vectors[:, 0].max() + pad[0])
    ax.set_ylim(vectors[:, 1].min() - pad[1], vectors[:, 1].max() + pad[1])
    ax.set_aspect("equal", adjustable="box")
    _save(fig, "categorical-embeddings.png")


def ordinal_score() -> None:
    rng = np.random.default_rng(2)
    n = 420
    x = rng.uniform(-2.2, 2.2, size=(n, 1)).astype(np.float32)
    label = (np.digitize(x[:, 0], np.array([-1.3, -0.4, 0.4, 1.3])) + 1).astype(np.int32)
    model = ordinal(
        max_epochs=28,
        patience=8,
        batch_size=48,
        learning_rate=2e-2,
        backbone_kwargs={"n_blocks": 2, "d_block": 32, "dropout": 0.0},
        validation_size=0.2,
        random_state=0,
    )
    model.fit(x, label)
    score = np.asarray(model.predict_score(x), dtype=np.float64)
    mae = float(np.mean(np.abs(score - label)))
    print(f"ordinal mae {mae:.4f} classes {model.classes_.tolist()}", flush=True)

    fig, ax = _panel("Expected score stays with the integer label")
    jitter = rng.uniform(-0.12, 0.12, size=n)
    ax.plot([0.6, 5.4], [0.6, 5.4], color=MUTED, linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
    ax.scatter(label + jitter, score, s=16, color=TEAL, alpha=0.45, linewidths=0, label="Rows", zorder=2)
    classes = np.asarray(model.classes_, dtype=np.float64)
    means = np.array([score[label == int(value)].mean() for value in classes])
    ax.scatter(classes, means, s=70, color=CLAY, zorder=3, label="Mean expected score")
    ax.set_xlabel("integer label")
    ax.set_ylabel("expected score")
    ax.set_xlim(0.5, 5.5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(0.5, 5.5)
    _legend(ax, "upper left")
    _save(fig, "ordinal-score.png")


def synthetic_cycle(n: int, seed: int = 0) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    noise = np.random.default_rng(seed).normal(0.0, 0.18, size=n)
    return 12.0 + 4.0 * np.sin(2.0 * np.pi * t / 24.0) + noise


def quantile_forecast() -> None:
    """One-step quantile forecasts rolled across a held-out cycle.

    Each point uses the observed window that ends at that step. The fit does
    not see the held-out cycle.
    """
    lookback = 48
    holdout = 24
    series = synthetic_cycle(lookback + 24 * 10 + holdout, seed=3)
    cut = len(series) - holdout
    model = TimeSeriesForecaster(
        lookback=lookback,
        horizon=1,
        head="quantile",
        max_epochs=28,
        patience=8,
        batch_size=64,
        random_state=0,
    )
    model.fit(series[:cut])
    origins = np.arange(cut, len(series))
    windows = np.stack([series[origin - lookback : origin] for origin in origins])
    point = np.asarray(model.predict(windows), dtype=np.float64)[:, 0]
    band = np.asarray(model.quantiles(windows), dtype=np.float64)[:, 0, :]
    future = series[cut:]
    covered = float(np.mean((future >= band[:, 0]) & (future <= band[:, 2])))
    width = float(np.mean(band[:, 2] - band[:, 0]))
    mae = float(np.mean(np.abs(point - future)))
    print(f"forecast mae {mae:.3f} coverage {covered:.3f} width {width:.3f} epoch {model.best_epoch_}", flush=True)

    shown = 72
    hist_x = np.arange(-shown, 0)
    fut_x = np.arange(0, holdout)
    fig, ax = _panel("The 0.1–0.9 band follows the next cycle")
    ax.plot(hist_x, series[cut - shown : cut], color=TEAL, linewidth=1.8, label="History", zorder=3)
    ax.fill_between(fut_x, band[:, 0], band[:, 2], color=TEAL, alpha=0.22, linewidth=0, label="0.1–0.9", zorder=2)
    ax.plot(fut_x, future, color=SLATE, linewidth=1.35, linestyle=(0, (2.2, 1.6)), label="Held-out cycle", zorder=3)
    ax.plot(fut_x, point, color=CLAY, linewidth=2.15, label="Point forecast", zorder=4)
    ax.axvline(0, color=MUTED, linewidth=0.7, zorder=1)
    ax.set_xlabel("step from the forecast origin")
    ax.set_ylabel("level")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=4,
        frameon=False,
        columnspacing=1.2,
        handlelength=1.7,
    )
    _save(fig, "quantile-forecast.png")


def mixture_density() -> None:
    rng = np.random.default_rng(4)
    n = 520
    series = rng.choice(np.array([-2.0, 2.0]), size=n) + rng.normal(0.0, 0.32, size=n)
    lookback = 24
    model = TimeSeriesForecaster(
        lookback=lookback,
        horizon=1,
        head="mixture",
        max_epochs=12,
        patience=4,
        batch_size=64,
        random_state=0,
    )
    model.fit(series)
    grid = np.linspace(-5.0, 5.0, 241)
    window = series[-lookback:]
    history = np.repeat(window[None, :], len(grid), axis=0)
    log_density = np.asarray(model.log_prob(history, grid[:, None]), dtype=np.float64)
    density = np.exp(log_density.reshape(-1))
    at = lambda level: float(np.exp(model.log_prob(window, np.array([level]))[0]))
    print(
        f"mixture density at -2 {at(-2):.4f} at 0 {at(0):.4f} at 2 {at(2):.4f} epoch {model.best_epoch_}",
        flush=True,
    )

    fig, ax = _panel("The mixture puts its mass on both modes")
    ax.hist(
        series,
        bins=36,
        density=True,
        color=TEAL,
        alpha=0.38,
        linewidth=0,
        label="Series",
        zorder=2,
    )
    ax.plot(grid, density, color=CLAY, linewidth=2.25, solid_capstyle="round", label="Mixture density", zorder=3)
    ax.set_xlim(-5.0, 5.0)
    ax.set_xlabel("next value")
    ax.set_ylabel("density")
    _legend(ax, "upper right")
    _save(fig, "mixture-density.png")


def histogram_bins() -> None:
    series = synthetic_cycle(24 * 16, seed=5)
    lookback = 36
    # A crest of the 24-step cycle, held out of the fit.
    origin = 366
    observed = series[:origin]
    model = TimeSeriesForecaster(
        lookback=lookback,
        horizon=1,
        head="histogram",
        max_epochs=14,
        patience=5,
        batch_size=64,
        random_state=0,
    )
    model.fit(observed[: origin - 24])
    window = observed[-lookback:]
    probabilities, centers = model.bin_probabilities(window)
    probabilities = np.asarray(probabilities, dtype=np.float64)[0]
    centers = np.asarray(centers, dtype=np.float64)
    target = float(series[origin])
    width = float(centers[1] - centers[0])
    bin_index = int(np.argmin(np.abs(centers - target)))
    print(
        f"histogram target {target:.3f} bin {centers[bin_index]:.3f} mass {probabilities[bin_index]:.3f} "
        f"epoch {model.best_epoch_}",
        flush=True,
    )

    fig, ax = _panel("Histogram probability sits on the next step")
    ax.bar(centers, probabilities, width=width * 0.9, color=TEAL, linewidth=0, zorder=2, label="Bin probability")
    ax.axvline(target, color=CLAY, linewidth=1.6, zorder=3, label="Held-out next step")
    ax.set_xlabel("value")
    ax.set_ylabel("probability")
    finite = centers[probabilities > 0.01]
    if len(finite):
        pad = width * 2.5
        ax.set_xlim(min(finite.min(), target) - pad, max(finite.max(), target) + pad)
    ax.set_ylim(0, 1.0)
    _legend(ax, "upper left")
    _save(fig, "histogram-bins.png")


def main() -> None:
    apply_style()
    tabular_fit()
    categorical_embeddings()
    ordinal_score()
    quantile_forecast()
    mixture_density()
    histogram_bins()


if __name__ == "__main__":
    main()
