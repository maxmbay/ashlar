import warnings

import numpy as np
import pytest
from sklearn.base import clone, is_classifier, is_regressor

import ashlar
from ashlar import (
    CLASSIFIER_RECIPE,
    ORDINAL_RECIPE,
    TREE_RECIPE,
    ASRClassifier,
    ASRModel,
    ASROrdinalClassifier,
    ASROutcomeTree,
    ASRRegressor,
    AshlarError,
)

SPEC = {"a": None, "b": ["x", "y"]}


def test_task_models_carry_their_recipe_defaults():
    assert ASRRegressor().get_params() == ASRModel().get_params()
    for model, recipe in ((ASRClassifier(), CLASSIFIER_RECIPE), (ASROrdinalClassifier(), ORDINAL_RECIPE), (ASROutcomeTree(SPEC), TREE_RECIPE)):
        params = model.get_params()
        for key, value in recipe.items():
            assert params[key] == value, key


def test_task_models_clone_and_tag_like_sklearn():
    assert is_regressor(ASRRegressor())
    assert is_classifier(ASRClassifier())
    assert is_classifier(ASROrdinalClassifier())
    copy = clone(ASRClassifier(max_epochs=3))
    assert type(copy) is ASRClassifier and copy.max_epochs == 3
    tree = clone(ASROutcomeTree(SPEC, patience=2))
    assert tree.tree == SPEC and tree.patience == 2
    assert repr(ASRClassifier(max_epochs=3)) == "ASRClassifier(max_epochs=3)"


def test_task_models_keep_their_task():
    assert ASRRegressor(task="mixture").task == "mixture"
    with pytest.raises(AshlarError):
        ASRClassifier(task="regression")
    with pytest.raises(AshlarError):
        ASRRegressor(task="classification")


@pytest.mark.slow
def test_load_returns_the_saved_class(tmp_path):
    x = np.random.default_rng(0).normal(size=(24, 2))
    model = ASRClassifier(backbone_kwargs={"n_blocks": 1, "d_block": 8, "dropout": 0.0}, max_epochs=1, patience=0, validation_size=0, batch_size=8)
    model.fit(x, (x[:, 0] > 0).astype(int))
    loaded = ASRModel.load(model.save(tmp_path / "model"))
    assert type(loaded) is ASRClassifier
    np.testing.assert_allclose(loaded.predict_proba(x), model.predict_proba(x), atol=1e-6)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("TabularEstimator", "ASRModel"),
        ("TimeSeriesForecaster", "ASRForecaster"),
        ("FullHistoryForecaster", "ASRHistoryForecaster"),
        ("AdditiveModel", "ASRAdditive"),
        ("HistogramBooster", "ASRHistogramBooster"),
        ("GradientSparseNet", "ASRGradientSparseNet"),
    ],
)
def test_old_class_names_warn(old, new):
    with pytest.warns(DeprecationWarning, match=new):
        assert getattr(ashlar, old) is getattr(ashlar, new)


def test_old_recipe_helpers_warn_and_return_the_new_classes():
    with pytest.warns(DeprecationWarning):
        assert type(ashlar.classifier()) is ASRClassifier
    with pytest.warns(DeprecationWarning):
        assert type(ashlar.ordinal()) is ASROrdinalClassifier
    with pytest.warns(DeprecationWarning):
        assert type(ashlar.outcome_tree(SPEC)) is ASROutcomeTree
    with pytest.warns(DeprecationWarning):
        assert type(ashlar.tabular()) is ASRRegressor
    with pytest.warns(DeprecationWarning):
        assert type(ashlar.tabular(task="classification")) is ASRModel


def test_new_names_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ASRRegressor(), ASRClassifier(), ASROrdinalClassifier(), ASROutcomeTree(SPEC)
