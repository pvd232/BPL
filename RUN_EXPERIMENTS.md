# Task 4

Use the existing `bpl` Conda environment. Dependencies are in `environment.yml`;
VIPER is pinned to the `v0.1.0` source commit.

```bash
conda activate bpl
python -m pip install --no-deps -e .
python pc_experiments.py --smoke
python pc_experiments.py
```

The default run trains nine configurations for 1,000 updates each, sequentially.
Both sweeps share the 20-step, 0.1-learning-rate baseline. Add `--adam` for the
optional tenth configuration. All runs use seed 42 and the same batch order.
The existing MNIST files under `data/MNIST/raw/` supply the images and labels.
The PC equations and architecture come from `pc_in_jax.ipynb`.

Each run saves its model, optimizer state, predictions, and measurements through
VIPER. `results/<timestamp>-task4/` contains `runs.json`, `summary.csv`, and
`accuracy.png`. Runtime includes JAX compilation. Accuracy uses the complete test
set; the notebook's original evaluation drops the final partial batch.

Rebuild the plots from saved measurements:

```bash
python pc_experiments.py --plot-only results/<timestamp>-task4
```

VIPER requires this folder to be a Git repository with committed source and an
HTTP(S) `origin`. The notebook stays editable; the runner uses `pc_model.py`.
Run `python -m unittest test_pc_model test_pc_experiments` to check notebook
parity and the saved checkpoint.
