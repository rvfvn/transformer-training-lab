# Transformer training lab

Implement and train five small transformer models on your own computer. Start with a model that cannot find a hidden digit, then add attention, position information, depth, an MLP, and multiple heads. Explain what each change makes possible.

This lab continues across multiple sessions. The first session introduces the programming model and begins local setup. Work through the checkpoints at your own pace with instructor guidance.

- **Framework:** tinygrad 0.14.0.
- **Classroom Python target:** Python 3.12. The pinned tinygrad package requires Python 3.11 or newer.
- **Initial device:** your computer's CPU.
- **Editor:** your existing editor, such as VS Code.
- **Repository:** [rvfvn/transformer-training-lab](https://github.com/rvfvn/transformer-training-lab).
- **Lecture:** [13-minute introduction and setup reference slides](output/pdf/transformer-training-lab-intro.pdf). Pages 1-9 are the lecture; pages 10-15 are practical references and sources.
- **Previous lecture:** [One head at a time](https://boopdotpng.github.io/slide-decks/lec2-interp/#1).

## Current starting point

This repository supplies the lecture, pinned dependency file, local setup helpers, an executable environment diagnostic, and five JSON experiment templates. You can create the environment, run the diagnostic, and inspect or validate configurations immediately.

The model scaffold, data generator, and training harness are the next materials to add. The training commands below define that intended interface; use them after `model.py` and `train.py` have been provided or implemented. The setup diagnostic does not claim that all five models are already implemented.

The class uses Python 3.12 for a consistent environment. `.python-version` records this choice; the helpers enforce it even though tinygrad itself supports additional Python versions.

## What you will learn

- Create a project-specific Python environment and explain which Python runs your code.
- Read tensor shapes as batch, position, and feature axes.
- Implement attention, residual additions, and an MLP.
- Configure and train each of the five models from fresh initial weights.
- Measure loss and accuracy on held-out examples.
- Change one setting at a time and explain the result.
- Save reproducible experiments and propose your work in a GitHub pull request.

## Checkpoint 1: get a local copy

A **fork** is your copy of the repository on GitHub. A **clone** is a local working copy on your computer. A **branch** records a line of changes within a repository.

1. Sign in to GitHub and open the [class repository](https://github.com/rvfvn/transformer-training-lab).
2. Choose **Fork** to create your own copy. If you already have a fork, use it and sync it with the class repository when instructed.
3. In your fork, choose **Code > HTTPS** and copy the clone URL.
4. Open PowerShell on Windows or Terminal on Mac. Run the commands below, replacing `YOUR-HANDLE` with your GitHub username.

```bash
git --version
git clone https://github.com/YOUR-HANDLE/transformer-training-lab.git
cd transformer-training-lab
git switch -c lab/YOUR-HANDLE
```

`git clone` downloads your fork. `cd` enters its folder. `git switch -c` creates a branch for your lab work.

If you already cloned this project, open that folder instead of cloning another copy. Use `git remote -v` to check whether `origin` is your fork or the class repository before pushing. Ask for help if it points to the class repository and you do not have write access.

If Git or authentication is blocking you, use **Code > Download ZIP**, extract it, and open that folder to begin setup. A ZIP is not a Git working copy. You can transfer your code into a clone of your fork before submission.

Install Git from the [official Git download page](https://git-scm.com/downloads) if it is missing. Do not spend the whole programming session debugging credentials; bring the exact error to an instructor.

## Checkpoint 2: create the Python environment

An environment is a folder containing the project's Python interpreter and installed packages. It keeps this lab's dependencies separate from your other projects. The folder is named `.venv` here.

Run the following commands from the repository folder. You only need the section for your operating system.

### Windows PowerShell

Check whether the class Python version is installed:

```powershell
py -3.12 --version
```

If it is missing, install Python 3.12 using an instructor-provided installer or the [official Python downloads](https://www.python.org/downloads/). Include the Python launcher if offered. Reopen PowerShell, then check again. If a newer Python is already installed, ask the instructor which version to use instead of changing all the commands independently.

Create the environment and install the pinned package:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
```

The printed interpreter path should end in `.venv\Scripts\python.exe`. Use this interpreter in your editor as well.

### macOS Terminal

Check whether the class Python version is installed:

```bash
python3.12 --version
```

If it is missing, install Python 3.12 using the class installer or a distribution agreed with the instructor. The [official Python downloads](https://www.python.org/downloads/) provide installer information. The Python supplied with macOS may be a different version; check rather than assuming.

Create the environment and install the pinned package:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import sys; print(sys.executable)"
```

The printed interpreter path should end in `.venv/bin/python`. Use this interpreter in your editor as well.

### What these commands do

- `-m venv .venv` creates the project's isolated Python environment.
- Calling that environment's Python with `-m pip` installs into the same environment that will run the lab.
- `requirements.txt` pins tinygrad to `0.14.0`. Do not silently upgrade it or copy examples for a different API version.
- Using the full interpreter path means you do not have to activate the environment. You do not need to change PowerShell's execution policy.

If you already know how to activate an environment, that is fine. Always check which interpreter you are actually using. See [Python's virtual environment documentation](https://docs.python.org/3.12/library/venv.html).

### Optional helper: run the same steps with printed explanations

After reading the manual steps, you can use a helper that prints each command and runs the checks:

```powershell
# Windows PowerShell, from the repository folder
.\setup-windows.bat
```

```bash
# Mac Terminal, from the repository folder
bash setup-mac.sh
```

The helper validates the configs, checks Python 3.12 and Clang, creates or reuses `.venv`, installs `requirements.txt`, and executes `check_environment.py`. It does not install Python or a compiler, modify shell profiles, or delete an existing environment. If an unrelated or incomplete `.venv` exists, it stops and explains the issue.

It is safe to rerun after correcting a prerequisite or interrupted installation. A check-only mode reuses a completed environment without installing anything:

```powershell
.\setup-windows.bat --check-only
```

```bash
bash setup-mac.sh --check-only
```

On later runs, the helper can use the existing project's `.venv` interpreter. On a fresh setup it looks for Python 3.12. If that Python has a different executable name, an instructor can set `LAB_PYTHON` to its executable path before running the helper. This selects an interpreter; it does not install one.

## Checkpoint 3: check the compiler and device

tinygrad's CPU backend compiles numerical operations. Its default CPU route requires **Clang available in your terminal's PATH**. `pip install` alone does not establish that this backend can execute. See the [tinygrad runtime requirements](https://docs.tinygrad.org/runtime/).

On either operating system, run:

```text
clang --version
```

### Windows: if Clang is missing

1. Use the official [LLVM releases](https://github.com/llvm/llvm-project/releases) and select a Windows installer appropriate for your machine, with instructor guidance.
2. If offered, enable the option to add LLVM to PATH. Otherwise ask for help adding the installed LLVM `bin` folder to your user PATH.
3. Close and reopen PowerShell, then run `clang --version` again.
4. Complete the numerical diagnostic below. A version string is only a prerequisite check.

Native Windows compiler/backend combinations need an instructor rehearsal. If compilation, linking, or backend execution fails, show the full error and your machine architecture to an instructor. Continue locally with an instructor or a partner while resolving it.

### Mac: if Clang is missing

Install Apple's command-line tools:

```bash
xcode-select --install
```

Follow the installation prompts, then reopen Terminal and run `clang --version` again. See [Apple's command-line tools instructions](https://developer.apple.com/documentation/xcode/installing-the-command-line-tools/).

### Choose CPU before importing tinygrad

The diagnostic below sets `DEV=CPU` before importing tinygrad. Later you can choose the device when launching a program:

Windows PowerShell:

```powershell
$env:DEV = "CPU"
```

Mac Terminal, for a single command:

```bash
DEV=CPU .venv/bin/python check_environment.py
```

Device selection and model configuration are different things. A model's head count does not select a GPU. Start on CPU; hardware acceleration is a later, separately tested extension. See [tinygrad's DEV setting](https://docs.tinygrad.org/env_vars/).

## Checkpoint 4: prove that the environment works

The repository includes [check_environment.py](check_environment.py). Read it to see the checks, then run it using the project's Python. You do not need to create or paste this file.

The checks execute numerical operations, calculate gradients, verify causal attention, and apply an optimizer update using tinygrad 0.14.0. They also report the actual interpreter, virtual environment, operating system, architecture, compiler, and package version.

Run it using the project's Python:

```powershell
# Windows PowerShell
.\.venv\Scripts\python.exe check_environment.py
```

```bash
# Mac Terminal
.venv/bin/python check_environment.py
```

After the interpreter, system, compiler, and passing-check information, the expected values include:

```text
tinygrad: 0.14.0
device: CPU
sum of squares: 14.0
gradient: [2.0, 4.0, 6.0]
parameter after update: 2.600
SETUP OK
```

Formatting or additional compiler messages may differ. The script must finish with exit code zero and `SETUP OK`. It checks actual execution rather than only importing the package. On failure it prints an actionable message. Append `--verbose` to the same command to include a full traceback for an instructor.

## Checkpoint 5: understand the model's tensors

A tensor is a multidimensional array with a shape, data type, and device. For a batch of 16 inputs of length 9:

| Object | Shape | Meaning |
| --- | --- | --- |
| Token IDs | `(16, 9)` | One integer per position in each example |
| Embedded inputs | `(16, 9, 32)` | 32 floating-point features per position |
| Queries, keys, values for one head | `(16, 9, 32)` | The head's three learned projections |
| Attention scores | `(16, 9, 9)` | Every query position compared with every key position |
| Final-position logits | `(16, 36)` | Scores for the 36 possible output tokens |
| Mean loss | `()` | One scalar used for backpropagation |

The batch axis contains independent examples. The position axis identifies tokens within an example. Feature axes contain learned numerical representations. The attention matrix's last axis contains possible source positions.

Inside a head, the core tinygrad operations are:

```python
# q, k, v and the causal mask have already been constructed.
scores = q @ k.transpose(-1, -2)
scores = scores / math.sqrt(q.shape[-1])
weights = (scores + mask).softmax(axis=-1)
write = weights @ v
```

The mask is zero on allowed positions and negative infinity on future positions. Each attention row should sum to one. Do not use the final-axis softmax over the batch or the feature axis by accident.

## Checkpoint 6: model and training configuration

The setup files and configuration templates below are included. `model.py` and `train.py` remain the next implementation stage, and each student creates their own results folder.

```text
transformer-training-lab/
    README.md
    requirements.txt           # pinned dependency
    .python-version            # classroom Python target
    setup-mac.sh               # optional Mac setup helper
    setup-windows.bat          # optional Windows setup helper
    scripts/setup_environment.py
    check_environment.py       # executable numerical diagnostic
    check_configs.py           # JSON validation; no tinygrad required
    model.py                   # to implement: shared model components
    train.py                   # to implement: training entry point
    configs/
        task0.json
        task1.json
        task2.json
        task3.json
        task4.json
    results/
        YOUR-HANDLE/
            report.md
```

Use a configuration file to separate the experiment's choices from its implementation. The following is the actual content of [configs/task3.json](configs/task3.json). The future training program must consume these settings before the training command examples will work.

```json
{
  "task": 3,
  "seed": 0,
  "model": {
    "vocab_size": 36,
    "seq_len": 10,
    "emb_dim": 32,
    "n_layers": 2,
    "n_heads": 1,
    "head_dim": 32,
    "position_embeddings": true,
    "mlp_layers": [1],
    "mlp_size": 64
  },
  "training": {
    "optimizer": "adam",
    "learning_rate": 0.001,
    "batch_size": 64,
    "steps": 1000,
    "train_examples": 10000,
    "test_examples": 1000
  }
}
```

Layer indices are zero-based: `mlp_layers: [1]` means an MLP after the second layer's attention. An empty list means no MLP. Use an empty list for tasks 0-2.

| Setting | What it changes |
| --- | --- |
| `task` | Data generation and the target rule |
| `seed` | Random initialization and reproducible data generation |
| `seq_len` | Number of input positions |
| `emb_dim` | Width of the residual stream |
| `n_layers` | Number of successive attention blocks |
| `n_heads` | Number of parallel attention heads per layer |
| `head_dim` | Width of each head's query, key, and value representations |
| `position_embeddings` | Whether position vectors are added to token embeddings |
| `mlp_layers`, `mlp_size` | Which layers have an MLP and its hidden width |
| `learning_rate` | Scale of optimizer updates |
| `batch_size` | Number of examples in one update |
| `steps` | Number of training updates |
| `train_examples`, `test_examples` | Dataset sizes |

These training values are starting settings, not a measured recipe for reproducing the lecture's accuracy. The instructor will calibrate run budgets on the classroom machines.

Keep `head_dim=32` explicit when adding a second head. Do not automatically change it to `emb_dim // n_heads`, which would make it 16. A simple lecture-compatible design gives each head its own 32-to-32 projections and adds the heads' writes to the stream. Each head computes from the same incoming stream within a layer.

Validate the included templates without running training:

```powershell
# Windows PowerShell
py -3.12 check_configs.py
```

```bash
# Mac Terminal
python3.12 check_configs.py
```

The checker prints a summary of each task and ends with `CONFIGS OK`. You can pass one JSON path to validate a changed configuration, for example `check_configs.py configs/task3.json`. It checks types, positive training settings, vocabulary/sequence length, and valid MLP layer indices. It allows architectural experiments such as changing the number of heads or layers.

Once `train.py` has been provided or implemented, the intended interface is:

```powershell
# Windows PowerShell
$env:DEV = "CPU"
.\.venv\Scripts\python.exe train.py --config configs/task1.json
```

```bash
# Mac Terminal
DEV=CPU .venv/bin/python train.py --config configs/task1.json
```

Use the appropriate config for the task. Validate the config and print it at the beginning of each run. Create a fresh model and optimizer for every experiment. Do not silently resume another task's weights.

## Checkpoint 7: implement and train tasks 0-4

The vocabulary is `abcdefghijklmnopqrstuvwxyz0123456789`. Letters occupy token IDs 0-25 and digits occupy 26-35. If the answer digit is `d`, the correct class ID is `26 + d`.

Only the final input position is graded. Inputs contain no appended answer token. The model returns 36 logits from the final stream, and cross-entropy consumes those logits directly. Do not add a separate softmax before this loss.

| Task | Input length | Layers | Heads/layer | Positions | MLP |
| --- | --- | --- | --- | --- | --- |
| 0: hidden digit | 9 | 0 | unused | No | No |
| 1: hidden digit | 9 | 1 | 1 | No | No |
| 2: dictionary lookup | 9 | 2 | 1 | Yes | No |
| 3: two-lookup sum | 10 | 2 | 1 | Yes | After layer 2, width 64 |
| 4: two-lookup difference | 10 | 2 | 2 | Yes | After layer 2, width 64 |

All models have residual width 32 and vocabulary size 36. For task 0, no attention block is constructed, so the configured head count is unused. Each task's training and test examples are generated separately with reproducible seeds; the held-out set is never used for optimizer updates.

### Task 0: hidden digit without attention

```text
qwdk7mzpa -> 7
```

Use exactly one digit among letters. For tasks 0 and 1, place that digit in one of the first eight positions and keep the final token a letter. Use the same datasets to compare those two architectures.

Implement token embedding lookup and unembedding of the final stream. There are no attention or MLP layers.

**Predict:** which input tokens can affect the output?

**Observe:** the model can learn that the answer is a digit, but cannot identify a random earlier digit from the final letter alone. Uniform guessing over ten digits has loss `ln(10)`, about 2.30, and accuracy about 10%.

**Explain:** why does this failure count as a successful experiment?

### Task 1: add one attention head

Solve the same hidden-digit task with one attention layer and one head. Position embeddings remain off.

Implement query/key/value projections, scaled dot products, causal masking, final-axis softmax, value mixing, and residual addition.

**Check before training:** shapes are correct, future attention is zero, attention rows sum to one, and an optimizer step changes parameters.

**Observe:** compare the final position's attention pattern with the digit's location. Interpret query/key selection separately from value content.

**Explain:** what pathway exists now that task 0 lacked?

### Task 2: dictionary lookup

```text
a2f5c9d3f -> 5
```

The first eight characters encode four letter-digit pairs: `a=2`, `f=5`, `c=9`, `d=3`. Use four distinct letters; digit values may repeat. The final letter is sampled from those dictionary keys.

Add position embeddings and a second attention layer, with one head in each. Keep the MLP off.

**Predict:** how can the first layer make the association available to the second?

**Observe:** look for a digit position attending to the preceding letter, followed by the final position retrieving the appropriate tagged digit. A particular training run may use a less clean strategy.

**Explain:** why can a later layer use information that parallel heads in the same layer have not yet written?

### Task 3: look up two digits and add

```text
a2f5c9d3fc -> 4
(5 + 9) % 10 = 4
```

Append two distinct query letters from the dictionary and output the sum of their digits modulo 10.

Keep two attention layers, one head per layer, and position embeddings. Add an MLP only after the second layer's attention. Its hidden width is 64.

Implement the up-projection, ReLU, down-projection, and residual addition. The same MLP operates independently on each position.

**Predict:** what information must reach the final stream before the MLP?

**Observe:** compare the answer scores before and after the MLP. If you remove ReLU as an extra experiment, distinguish removing it from a trained model from retraining a new model without it.

**Explain:** why does a weighted mixture of digit representations still need a learned computation to produce the answer?

### Task 4: look up two digits and subtract

```text
a2f5c9d3fc -> 6
(5 - 9) % 10 = 6
```

Use the first queried value minus the second, modulo 10. Keep the task 3 architecture but use two heads per layer.

**Predict:** why does an equal blend lose information about operand roles?

**Observe:** inspect the two heads and see whether they retrieve different operands. Swap the associated digits and compare the representation and prediction.

**Explain:** how do distinct learned value transformations help preserve roles even when head writes are added to the same stream?

The previous lecture also showed successful single-head subtraction runs using unequal attention mixtures. Two heads improved reliability across seeds; this is not a proof that subtraction is impossible with one head.

## Checkpoint 8: train, evaluate, and change one setting

When the training harness is implemented, a tinygrad 0.14.0 update has this form:

```python
with Context(TRAINING=1):
    optimizer.zero_grad()
    logits = model(tokens)
    loss = logits.sparse_categorical_crossentropy(targets)
    loss.backward()
    optimizer.step()
```

`backward()` calculates gradients. `step()` updates parameters. `zero_grad()` clears previously accumulated gradients. Imports, model construction, data, batching, and optimizer creation must exist around this excerpt.

Evaluate separately on held-out examples without optimizer updates and outside the training context. Record both training loss and held-out accuracy. A short diagnostic run establishes that training executes; it does not establish convergence.

For one controlled comparison, change only one setting. Predict what will happen, save the changed configuration, rerun from fresh weights, and explain the observed difference. Useful comparisons include one versus two layers, positions on versus off, ReLU versus identity, or different learning rates.

Do not assume your loss must decrease on every batch or that one seed reproduces the lecture's exact accuracy. Do not use the test set repeatedly to select settings; reserve a validation set if you begin tuning extensively.

### Record your results

Create `results/YOUR-HANDLE/report.md`. Keep a copy of each configuration used. Record the Python/framework versions, OS/architecture, device, seeds, optimizer, learning rate, batch size, and step count.

| Task | Steps | Training loss | Held-out accuracy | Example prediction | Status |
| --- | --- | --- | --- | --- | --- |
| 0 | | | | | |
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |

For each task, add your prediction, observed result, explanation, and any help or supplied solution code used. Mark incomplete runs explicitly. Label instructor-provided results separately from your own measurements.

## Troubleshooting

| Symptom | First checks |
| --- | --- |
| `py -3.12` or `python3.12` is not found | Confirm Python is installed; reopen the terminal; ask which interpreter the class uses. |
| `ModuleNotFoundError: tinygrad` | Run pip through the same `.venv` interpreter that runs your script. Print `sys.executable`. |
| Editor runs a different Python | Select `.venv/Scripts/python.exe` on Windows or `.venv/bin/python` on Mac. |
| `clang` is not found | Install the compiler for your OS, check PATH, and reopen the terminal. |
| Compiler/linker/backend error | Save the full traceback, `clang --version`, Python version, OS, architecture, and `DEV` value. Ask an instructor. |
| An example mentions `requires_grad` or `Tensor.train()` | It may target a different tinygrad release. Use the pinned version's examples. |
| Wrong tensor shape | Label batch, position, feature, and head axes. Print shapes immediately before the failing operation. |
| Loss reports an invalid target | Check integer target dtype, the range 0-35, and the `26 + digit` vocabulary mapping. |
| No useful parameter updates | Run the environment diagnostic, register the intended parameters, clear gradients, call backward, and then step inside the training context. |
| Loss stays high | Verify labels, masking, residual additions, last-position readout, optimizer parameters, and learning rate. Task 0 has an intended failure floor. |
| Training is slow | Record startup/compilation separately from later steps. Use the instructor's short diagnostic budget before a longer run. |
| `train.py` is missing | Training is the next implementation stage. Complete setup, inspect the included configs, and ask for the model exercise files. |
| `CONFIGS OK` is missing | Read the `[FAIL]` message. Check JSON syntax, value types, positive settings, and zero-based MLP indices. |

Never paste credentials or access tokens into a report or public issue.

## Checkpoint 9: submit a pull request

A pull request proposes your changes for review. Do this after your work is ready for the submission checkpoint; it is not a first-session deadline.

The included `.gitignore` already excludes local environments, Python caches, common credentials files, and model checkpoint formats. Keep it in the repository. `.gitattributes` preserves the line endings used by the Mac and Windows helpers.

Keep your own source, configuration files, and small results report. Exclude your environment, caches, credentials, and unrequested checkpoints or large generated files.

Check that `origin` points to your fork:

```bash
git remote -v
git status
```

Stage only paths you actually created or changed. For example, once these files exist:

```bash
git add .gitignore model.py train.py check_environment.py configs results/YOUR-HANDLE/report.md
git diff --cached
git commit -m "Implement transformer lab experiments"
git push -u origin lab/YOUR-HANDLE
```

Replace `YOUR-HANDLE` everywhere. If Git asks for an identity, configure your chosen name and verified email or GitHub-provided private commit email. HTTPS pushes require a supported GitHub authentication flow; your account password is not a Git password. Ask for help or use GitHub Desktop rather than placing a token in code.

On GitHub, open a pull request with:

- **Base repository:** `rvfvn/transformer-training-lab`.
- **Base branch:** `main`.
- **Head repository:** your fork.
- **Compare branch:** `lab/YOUR-HANDLE`.

Use this title:

```text
Lab submission: YOUR-HANDLE - tasks 0-4
```

Use this description:

```markdown
## Environment
- Python:
- tinygrad:
- OS and architecture:
- Device:

## Work completed
- Components implemented:
- Tasks trained:
- Tasks still in progress:
- Configurations and results: results/YOUR-HANDLE/report.md

## Explain
1. Why does task 0 fail to identify the earlier digit?
2. What does a second layer make possible?
3. What does the MLP add?
4. Why do multiple heads help preserve operand roles?

## Assistance
- Help, references, or supplied solution code used:
```

If you began from a ZIP, clone your fork and transfer your own source, configs, and report into it before committing. Alternatively, upload the small files through GitHub's browser interface on a branch in your fork, then create the PR. Avoid replacing shared instructor files unnecessarily.

Follow the submission date given by the instructors. There is no additional deadline specified in this README.

## Setup checks for instructors

The setup helper runs the numerical diagnostic automatically. The repository also has tests for invalid configurations, a missing compiler, check-only behavior, and preserving an unrelated existing `.venv`.

```powershell
# Windows PowerShell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

```bash
# Mac Terminal
.venv/bin/python -m unittest discover -s tests -v
```

The [local setup workflow](.github/workflows/setup.yml) runs the same helpers on GitHub-hosted Mac and Windows runners when changes are pushed to `main`, a pull request is opened, or the workflow is started manually. It uses the runners' compiler installations. A passing workflow supplements a rehearsal on the actual student machines; it does not establish compatibility with every Windows toolchain.

## References

- [One head at a time: previous lecture](https://boopdotpng.github.io/slide-decks/lec2-interp/#1).
- [tinygrad 0.14.0 package and Python requirement](https://pypi.org/project/tinygrad/0.14.0/).
- [tinygrad quickstart](https://docs.tinygrad.org/quickstart/).
- [tinygrad v0.14.0 tensor implementation](https://github.com/tinygrad/tinygrad/blob/v0.14.0/tinygrad/tensor.py).
- [tinygrad v0.14.0 optimizers](https://github.com/tinygrad/tinygrad/blob/v0.14.0/tinygrad/nn/optim.py).
- [tinygrad tensor operations](https://docs.tinygrad.org/tensor/ops/) and [movement operations](https://docs.tinygrad.org/tensor/movement/).
- [tinygrad runtimes](https://docs.tinygrad.org/runtime/) and [device selection](https://docs.tinygrad.org/env_vars/).
- [tinygrad scheduling and lowering](https://docs.tinygrad.org/developer/developer/).
- [Python virtual environments](https://docs.python.org/3.12/library/venv.html).
- [Apple command-line tools](https://developer.apple.com/documentation/xcode/installing-the-command-line-tools/) and [LLVM releases](https://github.com/llvm/llvm-project/releases).
- [PyTorch tensor concepts](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) and [automatic differentiation](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html).
- [Attention reference](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html) and [FlashAttention paper](https://arxiv.org/abs/2205.14135).
- [GitHub contribution workflow](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project) and [creating a PR from a fork](https://docs.github.com/en/pull-requests/how-tos/create-pull-requests/creating-a-pull-request-from-a-fork).
