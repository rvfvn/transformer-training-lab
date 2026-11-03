"""Validate and summarize JSON experiment settings without importing tinygrad."""

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive_integer(value):
    return type(value) is int and value > 0


def validate_config(config):
    require(isinstance(config, dict), "configuration must be a JSON object")
    task = config.get("task")
    require(type(task) is int and 0 <= task <= 4, "task must be an integer from 0 to 4")
    require(type(config.get("seed")) is int and config["seed"] >= 0, "seed must be a nonnegative integer")
    model = config.get("model")
    training = config.get("training")
    require(isinstance(model, dict) and isinstance(training, dict), "model and training must be JSON objects")
    require(model.get("vocab_size") == 36 and type(model["vocab_size"]) is int, "vocab_size must be 36")
    require(model.get("seq_len") == (9 if task <= 2 else 10) and type(model["seq_len"]) is int,
            "seq_len must be 9 for tasks 0-2 and 10 for tasks 3-4")
    for field in ("emb_dim", "n_heads", "head_dim", "mlp_size"):
        require(positive_integer(model.get(field)), f"model.{field} must be a positive integer")
    require(type(model.get("n_layers")) is int and model["n_layers"] >= 0, "n_layers must be a nonnegative integer")
    require(type(model.get("position_embeddings")) is bool, "position_embeddings must be true or false")
    layers = model.get("mlp_layers")
    require(isinstance(layers, list), "mlp_layers must be a list of zero-based layer indices")
    require(all(type(index) is int and 0 <= index < model["n_layers"] for index in layers),
            "each MLP layer index must refer to an existing layer")
    require(len(layers) == len(set(layers)), "mlp_layers must not contain duplicate indices")
    require(training.get("optimizer") in ("adam", "sgd"), "optimizer must be adam or sgd")
    rate = training.get("learning_rate")
    require(type(rate) in (int, float) and math.isfinite(rate) and rate > 0, "learning_rate must be finite and positive")
    for field in ("batch_size", "steps", "train_examples", "test_examples"):
        require(positive_integer(training.get(field)), f"training.{field} must be a positive integer")
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Optional JSON files to check; defaults to configs/task0.json through task4.json.")
    args = parser.parse_args(argv)
    paths = args.paths or [ROOT / "configs" / f"task{task}.json" for task in range(5)]
    failed = False
    for index, path in enumerate(paths):
        try:
            config = validate_config(json.loads(path.read_text(encoding="utf-8")))
            if not args.paths:
                require(config["task"] == index, f"{path.name} must describe task {index}")
            model = config["model"]
            mlps = ",".join(str(i + 1) for i in model["mlp_layers"]) or "none"
            print(f"[PASS] {path.name}: task {config['task']}, {model['n_layers']} layers, "
                  f"{model['n_heads']} head(s)/layer, head width {model['head_dim']}, MLP after layer(s): {mlps}")
        except (OSError, ValueError) as error:
            print(f"[FAIL] {path}: {error}", file=sys.stderr)
            failed = True
    if not failed:
        print("CONFIGS OK")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
