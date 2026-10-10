"""Check saved training state and accuracy over an incomplete final test batch."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from viper.stages import StageContext

from pc_experiments import (
    PCConfig,
    fit,
    load_arrays,
    load_json,
    load_model,
    load_state,
    read_mnist,
)
from pc_model import make_model, predict, train_step


class ExperimentTests(unittest.TestCase):
    """Exercise the real training function with temporary MNIST-format files."""

    def test_checkpoint_and_full_test_accuracy(self) -> None:
        """A reloaded two-update checkpoint makes the same third update as an uninterrupted run."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rng = np.random.default_rng(7)
            inputs = {}
            for split, count in (("train", 64), ("test", 17)):
                images = rng.integers(0, 256, (count, 28, 28), dtype=np.uint8)
                labels = rng.integers(0, 10, count, dtype=np.uint8)
                image_path = root / f"{split}_images"
                label_path = root / f"{split}_labels"
                image_path.write_bytes(
                    np.array([2051, count, 28, 28], dtype=">u4").tobytes()
                    + images.tobytes()
                )
                label_path.write_bytes(
                    np.array([2049, count], dtype=">u4").tobytes() + labels.tobytes()
                )
                inputs[f"{split}_images"] = image_path
                inputs[f"{split}_labels"] = label_path
            outputs = {
                name: root / filename
                for name, filename in (
                    ("model", "model.eqx"),
                    ("resume_state", "state.pt"),
                    ("predictions", "predictions.npz"),
                    ("measurements", "measurements.json"),
                )
            }
            np.random.seed(42)
            fit(
                StageContext(
                    run_id="test",
                    attempt_id=1,
                    stage_id="train",
                    config=PCConfig(iterations=2, inference_steps=2),
                    inputs=inputs,
                    outputs=outputs,
                    metrics={"energy": Mock()},
                    numpy_generators={},
                )
            )
            saved = load_state(outputs["resume_state"])
            reloaded = load_model(outputs["model"])
            p_optim = optax.adam(1e-3)
            template = p_optim.init(eqx.filter(reloaded, eqx.is_array))
            saved_leaves = saved.optimizer_state["adam_leaves"]
            assert isinstance(saved_leaves, list)
            restored_optimizer = jax.tree.unflatten(
                jax.tree.structure(template),
                [jnp.asarray(leaf) for leaf in saved_leaves],
            )
            images, labels = read_mnist(inputs["train_images"], inputs["train_labels"])
            np.random.seed(42)
            model = make_model()
            p_state = p_optim.init(eqx.filter(model, eqx.is_array))
            x_optim = optax.sgd(0.1)
            for step in range(3):
                order = np.random.permutation(64)
                targets = np.eye(10, dtype=np.float32)[labels[order]]
                if step == 2:
                    resumed = train_step(
                        reloaded,
                        images[order],
                        targets,
                        p_optim,
                        restored_optimizer,
                        x_optim,
                        2,
                        False,
                    )
                reference = train_step(
                    model, images[order], targets, p_optim, p_state, x_optim, 2, False
                )
                model, p_state = reference["model"], reference["p_opt_state"]
            for actual, expected in zip(
                jax.tree.leaves(resumed["model"]), jax.tree.leaves(model), strict=True
            ):
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
            predictions = load_arrays(outputs["predictions"])
            measurements = load_json(outputs["measurements"])
            self.assertEqual(len(predictions["labels"]), 17)
            self.assertEqual(measurements["test_examples"], 17)
            self.assertEqual(
                measurements["accuracy"],
                float(
                    np.mean(
                        predictions["logits"].argmax(axis=1) == predictions["labels"]
                    )
                    * 100
                ),
            )
            test_images, _ = read_mnist(inputs["test_images"], inputs["test_labels"])
            np.testing.assert_allclose(
                predict(reloaded, test_images),
                predictions["logits"],
                rtol=1e-6,
                atol=1e-7,
            )


if __name__ == "__main__":
    unittest.main()
