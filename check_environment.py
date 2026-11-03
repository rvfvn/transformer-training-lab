"""Verify the local CPU runtime before starting the transformer exercises."""

import argparse
import math
import os
import platform
import shutil
import subprocess
import sys
import traceback
from importlib.metadata import PackageNotFoundError, version

# Choose the backend before importing tinygrad.
os.environ["DEV"] = "CPU"
EXPECTED_VERSION = "0.14.0"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="Include a full traceback if a runtime check fails.")
    args = parser.parse_args()
    print("Python:", platform.python_version(), flush=True)
    print("Interpreter:", sys.executable, flush=True)
    print("System:", platform.system(), platform.machine(), flush=True)
    print("Virtual environment:", "yes" if sys.prefix != sys.base_prefix else "no", flush=True)
    try:
        require(sys.version_info[:2] == (3, 12), "Use the class Python 3.12 interpreter.")
        require(sys.prefix != sys.base_prefix, "Use the project's .venv Python; this is not a virtual environment.")
        compiler = shutil.which("clang")
        require(compiler, "Clang is missing from PATH. See README.md, checkpoint 3, then reopen the terminal.")
        print("Compiler:", compiler, flush=True)
        result = subprocess.run([compiler, "--version"], capture_output=True, text=True)
        require(result.returncode == 0, f"Clang could not run: {result.stderr.strip()}")
        installed_version = version("tinygrad")
        require(installed_version == EXPECTED_VERSION,
                f"Found tinygrad {installed_version}; install requirements.txt with this environment's Python.")
        from tinygrad import Tensor, Context, nn

        # Test actual execution and the derivative of sum(x*x).
        x = Tensor([1.0, 2.0, 3.0])
        value = (x * x).sum()
        value.backward()
        require(abs(value.item() - 14.0) < 1e-5, "Tensor computation returned the wrong sum.")
        require(x.grad is not None and x.grad.tolist() == [2.0, 4.0, 6.0], "The gradient check failed.")
        print("[PASS] Tensor computation and automatic differentiation", flush=True)

        # Exercise the matmul, transpose, causal mask, and softmax used by attention.
        q = Tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
        k = Tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
        v = Tensor([[[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]]])
        mask = Tensor([[0.0, -math.inf, -math.inf], [0.0, 0.0, -math.inf], [0.0, 0.0, 0.0]])
        weights = (q @ k.transpose(-1, -2) / math.sqrt(2) + mask).softmax(axis=-1)
        write = weights @ v
        rows = weights.tolist()[0]
        require(write.shape == (1, 3, 2), "Attention returned the wrong shape.")
        require(all(abs(sum(row) - 1.0) < 1e-5 for row in rows), "Attention rows do not sum to one.")
        require(all(rows[i][j] == 0.0 for i in range(3) for j in range(i + 1, 3)), "The causal mask leaked future positions.")
        require(write.tolist()[0][0] == [1.0, 0.0], "The first position copied the wrong value.")
        print("[PASS] Attention operations, normalization, and causal masking", flush=True)

        # SGD: w=1, loss=(2w-6)^2, gradient=-16, update with lr=0.1 gives w=2.6.
        w = Tensor([1.0])
        optimizer = nn.optim.SGD([w], lr=0.1)
        with Context(TRAINING=1):
            optimizer.zero_grad()
            loss = ((2 * w - 6) ** 2).sum()
            loss.backward()
            optimizer.step()
        require(abs(w.item() - 2.6) < 1e-5, "The optimizer did not apply the expected parameter update.")
        print("[PASS] Optimizer parameter update", flush=True)
        print("tinygrad:", installed_version)
        print("device:", x.device)
        print("sum of squares:", value.item())
        print("gradient:", x.grad.tolist())
        print(f"parameter after update: {w.item():.3f}")
        print("SETUP OK")
        return 0
    except PackageNotFoundError:
        print("\nSETUP FAILED: tinygrad is not installed in this interpreter. Install requirements.txt using the .venv Python.", file=sys.stderr)
    except Exception as error:
        print(f"\nSETUP FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        else:
            print("Rerun check_environment.py with --verbose and show the output to an instructor.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
