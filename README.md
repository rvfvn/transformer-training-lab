# Transformer training lab

Write and train five small transformer models locally using **tinygrad**. Build up from embeddings to attention, multiple layers, an MLP, and multiple heads. Compare loss, held-out accuracy, and attention patterns to understand what each component contributes.

The lab continues across multiple sessions. Start with local setup, then implement and train each model from fresh weights.

## The five models

| Task | Problem | Architecture |
| --- | --- | --- |
| 0 | Find a hidden digit: `qwdk7mzpa -> 7` | Embedding and unembedding, no layers |
| 1 | Solve the same hidden-digit task | One attention layer with one head |
| 2 | Dictionary lookup: `a2f5c9d3f -> 5` | Two attention layers and position embeddings |
| 3 | Look up two digits and add modulo 10: `a2f5c9d3fc -> 4` | Task 2 plus an MLP after layer 2 |
| 4 | Look up two digits and subtract modulo 10: `a2f5c9d3fc -> 6` | Task 3 with two heads per layer |

Each model uses a 36-token vocabulary and a 32-dimensional residual stream. Only the final position is graded. Task 0 is expected to fail at identifying the earlier digit; use it as a baseline.

## Prerequisites

- **[Python 3.12](https://www.python.org/downloads/)** and an editor such as VS Code.
- **Clang** available in your terminal. On Mac, install missing command-line tools with `xcode-select --install`. On Windows, install [LLVM](https://github.com/llvm/llvm-project/releases), add its `bin` folder to PATH, and reopen the terminal.
- Check the compiler with `clang --version`.

Start on CPU. The setup pins **tinygrad 0.14.0** in `requirements.txt`.

## Setup

Clone the repository, or open your existing local copy:

```bash
git clone https://github.com/rvfvn/transformer-training-lab.git
cd transformer-training-lab
```

### Mac Terminal

```bash
bash setup-mac.sh
```

### Windows PowerShell

```powershell
.\setup-windows.bat
```

The helpers print each step, create or reuse `.venv`, install dependencies, validate the configurations, and run `check_environment.py`. Successful setup ends with **`SETUP OK`** after checking computation, gradients, attention, and an optimizer update.

Select the `.venv` Python interpreter in your editor. If setup fails, fix the reported prerequisite and rerun the same command. Append `--check-only` to check an existing environment without installing packages.

## Configuration

The five experiment templates are in `configs/task0.json` through `configs/task4.json`. Inspect the model settings, learning rate, batch size, training steps, and seed. Change one setting at a time and record its effect.

Validate configurations after editing:

```bash
# Mac
.venv/bin/python check_configs.py
```

```powershell
# Windows
.\.venv\Scripts\python.exe check_configs.py
```

**Current status:** setup, diagnostics, and configuration templates are available. Model implementation, data generation, and the training harness are the next stage.

[Lecture slides](output/pdf/transformer-training-lab-intro.pdf) · [Previous lecture](https://boopdotpng.github.io/slide-decks/lec2-interp/#1)
