# Ashlar on TabArena-Lite

A local test of whether Ashlar is worth entering into [TabArena](https://github.com/autogluon/tabarena).
`ashlar_tabarena.py` wraps Ashlar as an AutoGluon `AbstractModel` with two entries: `Ashlar`
(`ASRRegressor` / `ASRClassifier` defaults) and `AshlarTabM` (`tabm_piecewise`). `run_lite.py`
runs them on TabArena-Lite (first split of each of the 51 datasets, 8-fold bagging) and scores them
against the hosted leaderboard results.

```bash
git clone https://github.com/autogluon/tabarena.git && cd tabarena
git checkout 32452f6fb2c5d14b73b424d14ae5f22c0c17e3e1
uv venv --seed --python 3.12 && source .venv/bin/activate
uv pip install --prerelease=allow -e "./packages/tabarena[benchmark]" -e /path/to/ashlar
cd /path/to/ashlar/examples/tabarena
python run_lite.py --smoke   # 3 small datasets
python run_lite.py           # all 51, default configs
```

Not yet done for a real submission: honouring `time_limit` and `num_cpus`.
