# Senior Capstone Project UTampa

Lung CT lesion segmentation research prototype for a senior Computer Science
capstone. The planned pipeline is:

```text
CT volume -> preprocessing -> 3D U-Net -> lesion mask -> evaluation
```

This is an educational/research prototype, not intended for clinical diagnosis
or treatment decisions.

## Current stage

This branch starts with basic project setup only. Preprocessing code, tests,
and model work will be added in separate steps. No trained model or medical
performance results are included.

## Python setup

Use Python 3.11. Run these commands from the repository folder.

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS Terminal:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Each teammate creates their own `.venv`; do not share the environment folder.
In VS Code, use **Python: Select Interpreter** to select that environment.
Apple Silicon Macs should use native arm64 Python. Requirements automatically
select legacy PyTorch 2.2.2 and NumPy 1.x on Intel Macs; current PyTorch builds
are used on Windows and native Apple Silicon. Intel Mac support is a legacy
compatibility path and still needs testing on a teammate's machine.

## Dataset

Datasets are downloaded separately and are not included in Git. See
[data/README.md](data/README.md) for the CT and expert-mask files needed later.
Keep scans, annotations, and identifying information out of shared logs and
documentation. Future training/validation/test partitions must group all scans
from the same patient together to prevent data leakage.

## Team workflow

- `main`: stable reviewed work.
- `develop`: combines reviewed task branches.
- `feature/project-setup`: this initial setup task.

Make one task-sized change at a time and open a pull request into `develop`.
Create future task branches when their work begins.
