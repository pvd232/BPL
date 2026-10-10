"""Run the predictive-coding model defined in pc_in_jax.ipynb."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

_INPUT_ACTIVATION = eqx.nn.Identity()


class FeedforwardPCModel(eqx.Module):
    """Apply an activation followed by a linear map at each weight layer."""

    layers: list[Any]

    def __init__(
        self,
        key: Any,
        *,
        input_size: int,
        hidden_size: int,
        output_size: int,
        activation_fn: Any,
        input_activation_fn: Any = _INPUT_ACTIVATION,
        n_layers: int = 2,
        use_bias: bool = True,
    ) -> None:
        """Initialize each layer from a separate split of the supplied key."""
        keys = jr.split(key, n_layers)
        widths = [input_size] + [hidden_size] * (n_layers - 1) + [output_size]
        layers = []
        for i in range(n_layers):
            act = input_activation_fn if i == 0 else activation_fn
            act_layer: Any = (
                act if isinstance(act, eqx.Module) else _StaticActivation(act)
            )
            linear = eqx.nn.Linear(
                widths[i], widths[i + 1], key=keys[i], use_bias=use_bias
            )
            layers.append(eqx.nn.Sequential([act_layer, linear]))
        self.layers = layers

    def __call__(self, x: Any) -> Any:
        """Predict one example without settling the hidden activities."""
        for layer in self.layers:
            x = layer(x)
        return x

    def __getitem__(self, i: int) -> Any:
        """Return the layer at the notebook's zero-based index."""
        return self.layers[i]

    def __len__(self) -> int:
        """Return the number of weight layers."""
        return len(self.layers)


class _StaticActivation(eqx.Module):
    """Keep the activation callable outside the model's trainable arrays."""

    fn: Callable[..., Any] = eqx.field(static=True)

    def __call__(self, x: Any, *, key: Any = None) -> Any:
        """Apply the activation using Equinox Sequential's call signature."""
        return self.fn(x)


def make_model(seed: int = 42) -> FeedforwardPCModel:
    """Create the notebook's 784 → 300 → 300 → 10 model."""
    return FeedforwardPCModel(
        jr.PRNGKey(seed),
        input_size=784,
        hidden_size=300,
        output_size=10,
        activation_fn=jax.nn.tanh,
        n_layers=3,
    )


def energy_fn(
    model: Any,
    activities: Any,
    input_: Any,
    output: Any,
    record_aux_details: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Sum one example's layer errors with the output clamped to its label."""
    last_idx = len(model) - 1
    energies = []
    aux = {}
    for layer_idx, layer in enumerate(model.layers):
        if layer_idx == 0:
            pred = layer(input_)
            err = activities[layer_idx] - pred
            key = f"energy/layer_{layer_idx}"
        elif layer_idx == last_idx:
            pred = layer(activities[layer_idx - 1])
            err = output - pred
            key = "energy/output"
        else:
            pred = layer(activities[layer_idx - 1])
            err = activities[layer_idx] - pred
            key = f"energy/layer_{layer_idx}"
        energies.append(0.5 * jnp.sum(err**2))
        if record_aux_details:
            aux[key] = energies[-1]
    return sum(energies), aux


def batched_energy_fn(
    model: Any,
    activities: Any,
    input_: Any,
    output: Any,
    energy_fn_: Callable[..., Any],
    record_aux_details: bool = True,
) -> tuple[Any, Any]:
    """Evaluate energy independently for each example in the batch."""
    return jax.vmap(energy_fn_, in_axes=(None, 0, 0, 0, None))(
        model, activities, input_, output, record_aux_details
    )


def batched_energy_sum(
    model: Any,
    activities: Any,
    input_: Any,
    output: Any,
    energy_fn_: Callable[..., Any],
    record_aux_details: bool = True,
) -> tuple[Any, Any]:
    """Sum energies so activity gradients retain their per-example scale."""
    per_ex, aux = batched_energy_fn(
        model, activities, input_, output, energy_fn_, record_aux_details
    )
    return jnp.sum(per_ex), jax.tree_util.tree_map(jnp.sum, aux)


def batched_energy_mean(
    model: Any,
    activities: Any,
    input_: Any,
    output: Any,
    energy_fn_: Callable[..., Any],
    record_aux_details: bool = True,
) -> tuple[Any, Any]:
    """Average energies before computing the shared parameter gradients."""
    per_ex, aux = batched_energy_fn(
        model, activities, input_, output, energy_fn_, record_aux_details
    )
    return jnp.mean(per_ex), jax.tree_util.tree_map(jnp.mean, aux)


def init_activities_with_ffwd(model: Any, input_: Any) -> list[Any]:
    """Initialize one example's activities with a forward pass."""
    activities = [model[0](input_)]
    for i in range(1, len(model)):
        activities.append(model[i](activities[i - 1]))
    return activities


def compute_activity_grad(
    model: Any,
    activities: Any,
    input_: Any,
    output: Any,
    record_aux_details: bool = True,
) -> tuple[Any, Any]:
    """Differentiate summed energy with respect to the activities only."""
    return jax.grad(batched_energy_sum, argnums=1, has_aux=True)(
        model, activities, input_, output, energy_fn, record_aux_details
    )


def compute_param_grad(
    model: Any,
    activities: Any,
    input_: Any,
    output: Any,
    record_aux_details: bool = True,
) -> tuple[Any, Any]:
    """Differentiate mean energy with the activities held fixed."""
    return eqx.filter_grad(batched_energy_mean, has_aux=True)(
        model, activities, input_, output, energy_fn, record_aux_details
    )


@eqx.filter_jit
def update_activities(
    model: Any,
    activities: Any,
    x_optim: Any,
    x_opt_state: Any,
    input_: Any,
    output: Any,
    record_aux_details: bool = True,
) -> tuple[Any, Any, Any]:
    """Apply one activity update and return energies from before the update."""
    activity_grads, aux = compute_activity_grad(
        model, activities, input_, output, record_aux_details
    )
    updates, x_opt_state = x_optim.update(
        activity_grads, x_opt_state, params=activities
    )
    activities = eqx.apply_updates(activities, updates)
    return activities, x_opt_state, aux


@eqx.filter_jit
def update_params(
    model: Any,
    activities: Any,
    p_optim: Any,
    p_opt_state: Any,
    input_: Any,
    output: Any,
    record_aux_details: bool = True,
) -> tuple[Any, Any, Any]:
    """Apply one parameter update using the current fixed activities."""
    param_grads, aux = compute_param_grad(
        model, activities, input_, output, record_aux_details
    )
    updates, p_opt_state = p_optim.update(param_grads, p_opt_state, params=model)
    model = eqx.apply_updates(model, updates)
    return model, p_opt_state, aux


@eqx.filter_jit
def train_step(
    model: Any,
    input_: Any,
    output: Any,
    p_optim: Any,
    p_opt_state: Any,
    x_optim: Any,
    T: int,
    record_aux_details: bool = True,
) -> dict[str, Any]:
    """Settle for T updates, then update weights once as in the notebook."""
    activities = jax.vmap(init_activities_with_ffwd, in_axes=(None, 0))(model, input_)
    x_opt_state = x_optim.init(activities)

    def settle_body(carry: Any, _: Any) -> tuple[Any, Any]:
        """Advance activity state while retaining each step's initial energy."""
        activities, x_opt_state = carry
        activities, x_opt_state, aux = update_activities(
            model,
            activities,
            x_optim,
            x_opt_state,
            input_,
            output,
            record_aux_details,
        )
        return (activities, x_opt_state), aux

    (activities, _), aux_scan = jax.lax.scan(
        settle_body, (activities, x_opt_state), None, length=T
    )
    energy, _ = batched_energy_mean(model, activities, input_, output, energy_fn, False)
    model, p_opt_state, _ = update_params(
        model,
        activities,
        p_optim,
        p_opt_state,
        input_,
        output,
        record_aux_details=False,
    )
    return {
        "model": model,
        "p_opt_state": p_opt_state,
        "activities": activities,
        "energy": energy,
        "aux": aux_scan,
    }


@eqx.filter_jit
def predict(model: Any, input_: Any) -> Any:
    """Return feedforward predictions for a batch without activity inference."""
    return jax.vmap(model)(input_)
