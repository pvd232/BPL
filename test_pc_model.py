"""Check the extracted model against the notebook and its settling behavior."""

from __future__ import annotations

import ast
import json
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

import pc_model


def notebook_definitions() -> dict[str, Any]:
    """Load only the notebook's model definitions, without executing its tasks."""
    selected = {
        "FeedforwardPCModel",
        "_StaticActivation",
        "energy_fn",
        "batched_energy_fn",
        "batched_energy_sum",
        "batched_energy_mean",
        "init_activities_with_ffwd",
        "compute_activity_grad",
        "compute_param_grad",
        "update_activities",
        "update_params",
        "train_step",
        "predict",
    }
    notebook = Path(__file__).with_name("pc_in_jax.ipynb")
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    definitions = []
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if (
                isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in selected
            ):
                definitions.append(node)
    namespace: dict[str, Any] = {
        "__name__": __name__,
        "Callable": Callable,
        "eqx": eqx,
        "jax": jax,
        "jnp": jnp,
        "jr": jr,
    }
    module = ast.Module(body=definitions, type_ignores=[])
    code = compile(module, str(notebook), "exec", dont_inherit=True)
    exec(code, namespace)  # noqa: S102 - compare against selected local notebook definitions
    return namespace


class PCModelTests(unittest.TestCase):
    """Observe notebook parity, batch scaling, and propagation into both layers."""

    def setUp(self) -> None:
        """Create a fixed two-image batch with distinct one-hot labels."""
        self.model = pc_model.make_model()
        self.images = jnp.linspace(-0.3, 0.3, 2 * 784).reshape(2, 784)
        self.labels = jax.nn.one_hot(jnp.array([1, 7]), 10)
        self.activities = jax.vmap(
            pc_model.init_activities_with_ffwd, in_axes=(None, 0)
        )(self.model, self.images)

    def assert_array_leaves_close(self, left: Any, right: Any) -> None:
        """Compare numeric leaves even when notebook classes have other identities."""
        left_leaves = jax.tree_util.tree_leaves(eqx.filter(left, eqx.is_array))
        right_leaves = jax.tree_util.tree_leaves(eqx.filter(right, eqx.is_array))
        self.assertEqual(len(left_leaves), len(right_leaves))
        for actual, expected in zip(left_leaves, right_leaves, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_matches_notebook_training_step(self) -> None:
        """Preserve initialization and a full SGD- and Adam-activity training step."""
        reference = notebook_definitions()
        model = reference["FeedforwardPCModel"](
            jr.PRNGKey(42),
            input_size=784,
            hidden_size=300,
            output_size=10,
            activation_fn=jax.nn.tanh,
            n_layers=3,
        )
        self.assert_array_leaves_close(self.model, model)
        p_optim = optax.adam(1e-3)
        for name, x_optim in (
            ("sgd", optax.sgd(0.1)),
            ("adam", optax.adam(0.1)),
        ):
            with self.subTest(activity_optimizer=name):
                actual = pc_model.train_step(
                    self.model,
                    self.images,
                    self.labels,
                    p_optim,
                    p_optim.init(eqx.filter(self.model, eqx.is_array)),
                    x_optim,
                    T=3,
                )
                expected = reference["train_step"](
                    model,
                    self.images,
                    self.labels,
                    p_optim,
                    p_optim.init(eqx.filter(model, eqx.is_array)),
                    x_optim,
                    T=3,
                )
                self.assertEqual(actual.keys(), expected.keys())
                self.assert_array_leaves_close(actual, expected)
                np.testing.assert_allclose(
                    pc_model.predict(actual["model"], self.images),
                    reference["predict"](expected["model"], self.images),
                    rtol=1e-6,
                    atol=1e-7,
                )

    def test_shapes_and_two_update_propagation(self) -> None:
        """Change the last hidden state first and the first hidden state second."""
        self.assertEqual(
            [value.shape for value in self.activities],
            [(2, 300), (2, 300), (2, 10)],
        )
        self.assertEqual(pc_model.predict(self.model, self.images).shape, (2, 10))
        grads, _ = pc_model.compute_activity_grad(
            self.model, self.activities, self.images, self.labels
        )
        np.testing.assert_array_equal(grads[0], 0)
        self.assertGreater(float(jnp.linalg.norm(grads[1])), 0)
        np.testing.assert_array_equal(grads[2], 0)

        optimizer = optax.sgd(0.1)
        first, state, _ = pc_model.update_activities(
            self.model,
            self.activities,
            optimizer,
            optimizer.init(self.activities),
            self.images,
            self.labels,
        )
        np.testing.assert_allclose(first[0], self.activities[0], rtol=0, atol=1e-7)
        self.assertGreater(float(jnp.linalg.norm(first[1] - self.activities[1])), 0)
        second, _, _ = pc_model.update_activities(
            self.model, first, optimizer, state, self.images, self.labels
        )
        self.assertGreater(float(jnp.linalg.norm(second[0] - self.activities[0])), 1e-6)
        np.testing.assert_array_equal(second[2], self.activities[2])

    def test_gradient_batch_scaling(self) -> None:
        """Duplicating examples preserves each activity gradient and mean weight gradient."""
        activities = [value + 0.02 for value in self.activities]
        activity_grads, _ = pc_model.compute_activity_grad(
            self.model, activities, self.images, self.labels
        )
        parameter_grads, _ = pc_model.compute_param_grad(
            self.model, activities, self.images, self.labels
        )
        doubled_activities = [jnp.concatenate([value, value]) for value in activities]
        doubled_images = jnp.concatenate([self.images, self.images])
        doubled_labels = jnp.concatenate([self.labels, self.labels])
        doubled_activity_grads, _ = pc_model.compute_activity_grad(
            self.model, doubled_activities, doubled_images, doubled_labels
        )
        doubled_parameter_grads, _ = pc_model.compute_param_grad(
            self.model, doubled_activities, doubled_images, doubled_labels
        )
        for original, doubled in zip(
            activity_grads, doubled_activity_grads, strict=True
        ):
            np.testing.assert_allclose(doubled[:2], original, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(doubled[2:], original, rtol=1e-6, atol=1e-7)
        self.assert_array_leaves_close(parameter_grads, doubled_parameter_grads)


if __name__ == "__main__":
    unittest.main()
