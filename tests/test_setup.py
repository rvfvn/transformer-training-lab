"""Exercise configuration mistakes and setup guards without installing packages."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="lab setup test ")
        self.addCleanup(self.directory.cleanup)
        self.project = Path(self.directory.name) / "project"
        self.project.mkdir()
        shutil.copytree(ROOT / "scripts", self.project / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "configs", self.project / "configs")
        for filename in ("requirements.txt", "check_configs.py", "check_environment.py"):
            shutil.copy2(ROOT / filename, self.project / filename)

    def command(self, script, *args, environment=None):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        if environment:
            env.update(environment)
        return subprocess.run(
            [sys.executable, str(self.project / script), *args],
            cwd=self.directory.name, env=env, capture_output=True, text=True, timeout=30,
        )

    def require_clang(self):
        if not shutil.which("clang"):
            self.skipTest("This guard is reached after the Clang prerequisite check.")

    def test_all_five_configs_validate_from_another_directory(self):
        result = self.command("check_configs.py")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.count("[PASS]"), 5)

    def test_bad_mlp_index_is_explained(self):
        path = self.project / "configs/task3.json"
        config = json.loads(path.read_text())
        config["model"]["mlp_layers"] = [2]
        path.write_text(json.dumps(config))
        result = self.command("check_configs.py")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("existing layer", result.stderr)

    def test_nonfinite_learning_rate_is_rejected(self):
        path = self.project / "configs/task1.json"
        config = json.loads(path.read_text())
        config["training"]["learning_rate"] = float("nan")
        path.write_text(json.dumps(config))
        result = self.command("check_configs.py")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("finite and positive", result.stderr)

    def test_boolean_batch_size_is_not_an_integer_setting(self):
        path = self.project / "configs/task1.json"
        config = json.loads(path.read_text())
        config["training"]["batch_size"] = True
        path.write_text(json.dumps(config))
        result = self.command("check_configs.py")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("batch_size must be a positive integer", result.stderr)

    def test_incorrect_task_file_identity_is_rejected(self):
        path = self.project / "configs/task1.json"
        config = json.loads(path.read_text())
        config["task"] = 0
        path.write_text(json.dumps(config))
        result = self.command("check_configs.py")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must describe task 1", result.stderr)

    def test_missing_compiler_does_not_create_environment(self):
        result = self.command("scripts/setup_environment.py", environment={"PATH": ""})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Clang was not found", result.stderr)
        self.assertFalse((self.project / ".venv").exists())

    def test_existing_unrelated_environment_is_preserved(self):
        self.require_clang()
        folder = self.project / ".venv"
        folder.mkdir()
        sentinel = folder / "keep.txt"
        sentinel.write_text("student's existing files")
        result = self.command("scripts/setup_environment.py")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("left untouched", result.stderr)
        self.assertEqual(sentinel.read_text(), "student's existing files")
        self.assertFalse((folder / "pyvenv.cfg").exists())

    def test_check_only_does_not_create_environment(self):
        self.require_clang()
        result = self.command("scripts/setup_environment.py", "--check-only")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no .venv yet", result.stderr)
        self.assertFalse((self.project / ".venv").exists())

    def test_help_does_not_create_environment(self):
        result = self.command("scripts/setup_environment.py", "--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--check-only", result.stdout)
        self.assertFalse((self.project / ".venv").exists())


if __name__ == "__main__":
    unittest.main()
