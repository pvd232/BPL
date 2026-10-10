"""Run the notebook's two MNIST sweeps through local VIPER stages."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
from pydantic import Field
from viper import execution
from viper.authoring import expand, experiment, input, replicate, stage, variant
from viper.config import MetricConfig, TrainConfig
from viper.metrics import MetricContext, measure, metric, min
from viper.outputs import TrainOutputs, output
from viper.randomness import capture_main_process_rng
from viper.references import GitFileRef
from viper.repository import read_source, resolve_root
from viper.restoration import ArtifactRestoreSelector
from viper.resume import (
    DataLoaderConfiguration,
    DataLoaderResumeState,
    ResumeState,
    load_resume_state,
    save_resume_state,
)
from viper.runtime import LocalEnvSpec, observe_python_env
from viper.stages import StageContext, train

from pc_model import make_model, predict, train_step


class PCConfig(TrainConfig):
    """Settings varied by Task 4, with the notebook's other settings fixed."""

    inference_steps: int = Field(
        default=20, gt=0, description="Activity updates per batch."
    )
    activity_lr: float = Field(default=0.1, gt=0, description="Activity learning rate.")
    activity_optimizer: Literal["sgd", "adam"] = Field(
        default="sgd", description="Optimizer used to settle the activities."
    )
    iterations: int = Field(
        default=1000, gt=0, description="Number of training batches."
    )
    seed: int = Field(default=42, ge=0, description="JAX model initialization seed.")


def load_json(path: Path) -> dict[str, Any]:
    """Read one run's saved settings and measurements."""
    return json.loads(path.read_text())


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    """Read numerical arrays without pickle."""
    with np.load(path, allow_pickle=False) as saved:
        return dict(saved)


def load_model(path: Path) -> Any:
    """Restore the fixed notebook architecture from its saved leaves."""
    return eqx.tree_deserialise_leaves(path, make_model())


def load_state(path: Path) -> ResumeState:
    """Load VIPER's checkpoint record containing the Optax array leaves."""
    return load_resume_state(path)


def read_mnist(images_path: Path, labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the notebook's downloaded IDX files and apply its normalization."""
    image_bytes = images_path.read_bytes()
    label_bytes = labels_path.read_bytes()
    image_header = np.frombuffer(image_bytes[:16], dtype=">u4")
    label_header = np.frombuffer(label_bytes[:8], dtype=">u4")
    if (
        image_header.shape != (4,)
        or label_header.shape != (2,)
        or image_header[0] != 2051
        or label_header[0] != 2049
        or tuple(image_header[2:]) != (28, 28)
        or image_header[1] != label_header[1]
    ):
        raise ValueError("Expected matching MNIST image and label files")
    count = int(image_header[1])
    images = np.frombuffer(image_bytes, dtype=np.uint8, offset=16).reshape(count, 784)
    labels = np.frombuffer(label_bytes, dtype=np.uint8, offset=8)
    if len(labels) != count or np.any(labels > 9):
        raise ValueError("MNIST label count or values are invalid")
    images = images.astype(np.float32) / np.float32(255)
    images = (images - np.float32(0.1307)) / np.float32(0.3081)
    return images, labels


@metric(metric_id="energy", mode="stateless")
def energy_metric(_context: MetricContext[MetricConfig], value: float) -> float:
    """Record mean training energy after activity settling."""
    return value


@train(config=PCConfig)
def fit(context: StageContext[PCConfig]) -> None:
    """Train one configuration and save its model, state, and full-test predictions."""
    config = context.config
    images, labels = read_mnist(
        context.inputs["train_images"], context.inputs["train_labels"]
    )
    test_images, test_labels = read_mnist(
        context.inputs["test_images"], context.inputs["test_labels"]
    )
    model = make_model(config.seed)
    p_optim = optax.adam(1e-3)
    p_state = p_optim.init(eqx.filter(model, eqx.is_array))
    x_optim = optax.sgd(config.activity_lr)
    if config.activity_optimizer == "adam":
        x_optim = optax.adam(config.activity_lr)
    # VIPER seeds and captures NumPy's global RNG for every replicate.
    order = np.random.permutation(len(images))
    cursor = 0
    history = []
    started = time.perf_counter()
    for step in range(1, config.iterations + 1):
        if cursor + 64 > len(order):
            order = np.random.permutation(len(images))
            cursor = 0
        indices = order[cursor : cursor + 64]
        cursor += 64
        targets = np.eye(10, dtype=np.float32)[labels[indices]]
        result = train_step(
            model,
            images[indices],
            targets,
            p_optim,
            p_state,
            x_optim,
            config.inference_steps,
            record_aux_details=False,
        )
        model, p_state = result["model"], result["p_opt_state"]
        energy = float(result["energy"])
        if not np.isfinite(energy):
            raise FloatingPointError(f"Non-finite training energy at iteration {step}")
        if step % 100 == 0 or step == config.iterations:
            context.metrics["energy"].record(energy, step=step)
            history.append({"iteration": step, "energy": energy})
            print(f"{step}/{config.iterations}: energy={energy:.5f}", flush=True)
    jax.block_until_ready((model, p_state))
    training_seconds = time.perf_counter() - started
    evaluation_started = time.perf_counter()
    predictions = np.concatenate(
        [
            np.asarray(predict(model, test_images[start : start + 1000]))
            for start in range(0, len(test_images), 1000)
        ]
    )
    if not np.isfinite(predictions).all():
        raise FloatingPointError("Non-finite test predictions")
    accuracy = float(np.mean(predictions.argmax(axis=1) == test_labels) * 100)
    evaluation_seconds = time.perf_counter() - evaluation_started
    for path in context.outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(context.outputs["model"], model)
    save_resume_state(
        context.outputs["resume_state"],
        ResumeState(
            optimizer_state={
                "adam_leaves": [np.asarray(leaf) for leaf in jax.tree.leaves(p_state)],
                "iteration": config.iterations,
            },
            main_process_rng=capture_main_process_rng(
                context.numpy_generators, capture_legacy_global=True
            ),
            dataloader=DataLoaderResumeState(
                configuration=DataLoaderConfiguration(workers=0),
                state_dict={"order": order, "cursor": cursor},
            ),
        ),
    )
    np.savez(context.outputs["predictions"], logits=predictions, labels=test_labels)
    measurements = {
        **config.model_dump(),
        "accuracy": accuracy,
        "test_examples": len(test_labels),
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "timing_includes_compilation": True,
        "history": history,
    }
    context.outputs["measurements"].write_text(
        json.dumps(measurements, indent=2) + "\n"
    )
    print(
        f"Test accuracy: {accuracy:.2f}% ({training_seconds:.1f}s training)", flush=True
    )


def configurations(smoke: bool, adam: bool) -> list[PCConfig]:
    """List the two sweeps once each, sharing the 20-step, 0.1 baseline."""
    if smoke:
        return [PCConfig(iterations=2)]
    configs = [PCConfig(inference_steps=steps) for steps in (1, 5, 20, 50, 100)]
    configs.extend(PCConfig(activity_lr=lr) for lr in (0.001, 0.01, 0.5, 1.0))
    if adam:
        configs.append(PCConfig(activity_optimizer="adam"))
    return configs


def plot_results(directory: Path) -> None:
    """Make the task's plots from saved measurements, without training."""
    records = json.loads((directory / "runs.json").read_text())
    rows = []
    for record in records:
        if record["status"] == "succeeded":
            rows.append(load_json(Path(record["measurements"])))
    if not rows:
        raise ValueError("No completed measurements to plot; inspect runs.json")
    columns = [
        "inference_steps",
        "activity_lr",
        "activity_optimizer",
        "iterations",
        "seed",
        "accuracy",
        "training_seconds",
        "evaluation_seconds",
        "test_examples",
    ]
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, field, fixed_field, fixed_value, label in (
        (axes[0], "inference_steps", "activity_lr", 0.1, "Settling steps"),
        (axes[1], "activity_lr", "inference_steps", 20, "Activity learning rate"),
    ):
        selected = sorted(
            (
                row
                for row in rows
                if row[fixed_field] == fixed_value
                and row["activity_optimizer"] == "sgd"
            ),
            key=lambda row: row[field],
        )
        ax.plot(
            [row[field] for row in selected],
            [row["accuracy"] for row in selected],
            "o-",
        )
        ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("Test accuracy (%)")
        ax.grid(alpha=0.25)
    fig.suptitle(f"MNIST after {rows[0]['iterations']} training updates")
    fig.tight_layout()
    fig.savefig(directory / "accuracy.png", dpi=160)
    plt.close(fig)
    print(f"Results: {directory}")


def main() -> None:
    """Run Task 4, run one smoke check, or replot an existing result directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run two training updates for one configuration.",
    )
    parser.add_argument(
        "--adam",
        action="store_true",
        help="Also run the optional activity-Adam baseline.",
    )
    parser.add_argument("--plot-only", type=Path, metavar="RESULT_DIRECTORY")
    args = parser.parse_args()
    if args.plot_only:
        plot_results(args.plot_only.resolve())
        return
    root = resolve_root()
    source = read_source(root)
    environment = LocalEnvSpec(
        lockfile=GitFileRef(
            repository=source.repository, commit=source.commit, path="environment.yml"
        ),
        python_env=observe_python_env(),
    )
    variants = []
    energy = measure(energy_metric)
    for config in configurations(args.smoke, args.adam):
        name = f"{config.activity_optimizer}_steps_{config.inference_steps}_lr_{config.activity_lr:g}".replace(
            ".", "p"
        )
        inputs = []
        for split, prefix in (("train", "train"), ("test", "t10k")):
            role: Literal["training", "validation"] = (
                "training" if split == "train" else "validation"
            )
            inputs.append(
                input(
                    f"{split}_images",
                    path=f"data/MNIST/raw/{prefix}-images-idx3-ubyte",
                    data_role=role,
                )
            )
            inputs.append(
                input(
                    f"{split}_labels",
                    path=f"data/MNIST/raw/{prefix}-labels-idx1-ubyte",
                    data_role=role,
                )
            )
        training = stage(
            fit,
            stage_id="train",
            config=config,
            inputs=tuple(inputs),
            outputs=TrainOutputs.model_validate(
                {
                    "model": output(
                        path="model.eqx", loader=load_model, data_role="validation"
                    ),
                    "resume_state": output(
                        path="resume_state.pt",
                        loader=load_state,
                        data_role="validation",
                    ),
                    "predictions": output(
                        path="predictions.npz",
                        loader=load_arrays,
                        data_role="validation",
                    ),
                    "measurements": output(
                        path="measurements.json",
                        loader=load_json,
                        data_role="validation",
                    ),
                }
            ),
            metrics=(energy,),
            objective=min(energy),
        )
        variants.append(
            variant(name, stages=(training,), estimator=training.outputs["model"])
        )
    study = experiment(
        experiment_id="pc_smoke" if args.smoke else "pc_task4",
        variants=tuple(variants),
        replicates=(replicate(seed=42),),
    )
    drafts = expand(study, source=source, env=environment)
    directory = (
        root
        / "results"
        / (
            datetime.now().strftime("%Y%m%d-%H%M%S")
            + ("-smoke" if args.smoke else "-task4")
        )
    )
    directory.mkdir(parents=True)
    records = []
    for draft in drafts:
        print(f"Running {draft.variant}", flush=True)
        try:
            result = execution.run(draft)
            record = {
                "variant": draft.variant,
                "status": result.status,
                "run": str(result.path),
            }
            if result.status == "succeeded":
                restored = execution.restore(
                    root,
                    result.reference,
                    artifacts=(
                        ArtifactRestoreSelector(
                            stage_id="train", artifact_name="measurements"
                        ),
                    ),
                    output=directory / f"{draft.run_id}.json",
                )
                record["measurements"] = str(restored.artifacts[0].files[0].path)
        except Exception as error:
            record = {
                "variant": draft.variant,
                "status": "failed",
                "error": str(error),
            }
            print(f"Failed: {error}", flush=True)
        records.append(record)
        (directory / "runs.json").write_text(json.dumps(records, indent=2) + "\n")
    plot_results(directory)
    if any(record["status"] != "succeeded" for record in records):
        raise SystemExit("Some configurations failed; see runs.json")


if __name__ == "__main__":
    main()
