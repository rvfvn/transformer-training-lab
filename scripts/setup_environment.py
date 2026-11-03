"""Explain and run the same local setup steps shown in the README."""

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
ENVIRONMENT = ROOT / ".venv"
PYTHON = ENVIRONMENT / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


class SetupError(Exception):
    pass


def child_environment():
    environment = os.environ.copy()
    # The selected interpreter and its environment supply the Python packages.
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["DEV"] = "CPU"
    return environment


def run(command):
    display = subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)
    print(f"\n> {display}", flush=True)
    result = subprocess.run(command, cwd=ROOT, env=child_environment())
    if result.returncode:
        raise SetupError(f"Command exited with code {result.returncode}. Read its message above.")


def verify_existing_environment():
    if ENVIRONMENT.is_symlink():
        raise SetupError(".venv is a symbolic link. Use a real project environment folder.")
    if not (ENVIRONMENT / "pyvenv.cfg").is_file() or not PYTHON.is_file():
        raise SetupError(
            "An incomplete or unrelated .venv already exists. It has been left untouched. "
            "Ask an instructor to help move it aside before creating this project's environment."
        )
    result = subprocess.run(
        [str(PYTHON), "-c", "import json,sys; print(json.dumps([list(sys.version_info[:2]),sys.prefix,sys.base_prefix]))"],
        cwd=ROOT, env=child_environment(), capture_output=True, text=True,
    )
    try:
        version, prefix, base = json.loads(result.stdout)
    except (ValueError, TypeError):
        raise SetupError("The existing .venv Python could not run. No environment files were replaced.") from None
    if result.returncode or version != [3, 12] or prefix == base or Path(prefix).resolve() != ENVIRONMENT.resolve():
        raise SetupError(
            "The existing .venv is not this project's Python 3.12 environment. "
            "It has been left untouched; ask an instructor before replacing it."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Check an existing .venv without creating it or installing packages.")
    args = parser.parse_args()
    try:
        if sys.version_info[:2] != (3, 12):
            raise SetupError(
                f"This class uses Python 3.12; this interpreter is {sys.version.split()[0]}. "
                "Use py -3.12 on Windows or python3.12 on Mac."
            )
        print(f"Project: {ROOT}", flush=True)
        print(f"Python: {sys.executable}", flush=True)
        print("Step 1: validate the five configuration templates.", flush=True)
        run([sys.executable, str(ROOT / "check_configs.py")])

        print("\nStep 2: check the CPU compiler prerequisite.", flush=True)
        compiler = shutil.which("clang")
        if not compiler:
            raise SetupError(
                "Clang was not found in PATH. On Mac, install Apple's command-line tools. "
                "On Windows, install LLVM and enable its bin folder in PATH. "
                "Reopen the terminal afterward. See README.md, Prerequisites."
            )
        run([compiler, "--version"])

        print("\nStep 3: create or reuse the project's isolated .venv.", flush=True)
        if ENVIRONMENT.exists() or ENVIRONMENT.is_symlink():
            verify_existing_environment()
            print("Reusing the existing Python 3.12 environment.", flush=True)
        elif args.check_only:
            raise SetupError("There is no .venv yet. Run setup without --check-only first.")
        else:
            run([sys.executable, "-m", "venv", str(ENVIRONMENT)])
            verify_existing_environment()

        if not args.check_only:
            print("\nStep 4: install the exact package version in requirements.txt.", flush=True)
            run([str(PYTHON), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(ROOT / "requirements.txt")])

        print("\nFinal step: execute the tensor, gradient, attention, and optimizer checks.", flush=True)
        run([str(PYTHON), str(ROOT / "check_environment.py")])
        print("\nSetup complete. Select the .venv interpreter in your editor.", flush=True)
        print("Next: inspect configs/task0.json and configs/task1.json; follow the README checkpoints.", flush=True)
        return 0
    except (SetupError, OSError) as error:
        print(f"\nSETUP FAILED: {error}", file=sys.stderr)
        print("Fix the reported prerequisite or error, then rerun the same command. No existing environment is deleted.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nSetup interrupted. You can rerun the same command.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
