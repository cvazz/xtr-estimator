# Intro to Xtr-Estimator

**xtr-estimator** is a Python-based tool designed to estimate extrapolation factors for time-resolved crystallography. It helps researchers determine the optimal scaling for difference maps by analyzing the relationship between dark states and triggered/difference states.

---

## 🚀 Installation

This package is intended for local development and research. Installing in **editable mode** ensures that any changes you make to the source code are instantly available in your environment.

### 1. Environment Setup
Choose any python version >3.11, we tested 3.12 and 3.14 so far

```bash
# Create a fresh environment
conda create -n xtr-estimator python=3.14 pip
conda activate xtr-estimator
```

### 2. Install the Package
Clone the repository and install it using `pip`:

```bash
git clone https://github.com/your-username/xtr_estimator.git
cd xtr_estimator
pip install -e .
```

---

## 🛠 Configuration Modes

The tool supports two primary workflows, determined automatically by the fields present in your configuration:

1.  **Triggered Mode:** Requires a dark map and a triggered (light) map. The tool calculates the difference internally.
2.  **Difference Mode:** Requires a dark map and a pre-calculated difference map (e.g., `Fo-Fo` or `K-weighted`).

### Configuration Sources
You can configure an experiment in three ways, which are merged in the following priority:
1.  **Command Line:** Highest priority (overrides everything).
2.  **Local YAML:** A user-provided `.yaml` file.
3.  **Global Defaults:** The package-wide `conf/config.yaml`.

---

## 📂 Examples & Usage

The `examples/` directory contains sample data (Photolyase, rsEGFP2) and scripts to help you get started.

### 1. Running via Command Line
The `main.py` entry point handles relative path resolution. If you provide a path to a YAML file, the tool will try to find MTZ/PDB files relative to that YAML's location.

```bash
# Run using a local dataset config
python -m xtr_estimator.main examples/rsEGFP2/local_config.yaml

# Run with command line overrides
python -m xtr_estimator.main examples/rsEGFP2/local_config.yaml masking.sigma=5 general.name_machine="custom_run"
```

### 2. Using in a Script or Notebook
You can use the `get_config` and `execute_main` functions to run the pipeline programmatically:

```python
from xtr_estimator.main import get_config, execute_main

# Load a local config and override specific values
cfg = get_config(
    data_yaml="examples/rsEGFP2/local_config.yaml",
    overrides=["masking.sigma=4"]
)

# Run the estimation pipeline
execute_main(cfg)
```

---

## 📁 Directory Structure
```text
.
├── conf/                # Hydra configuration defaults
├── examples/            # Example datasets (rsEGFP2, Photolyase)
│   ├── data/            # MTZ and PDB files
│   └── scripts/         # Sample scripts for diff/triggered modes
├── xtr_estimator/       # Core package source code
│   ├── main.py          # CLI Entry point and config orchestrator
│   ├── estimation.py    # Math and plotting for extrapolation
│   └── processing.py    # Map handling and scaling
└── pyproject.toml       # Package metadata and dependencies
```

---

## 👥 Authors
* **Sebastian Bielfeldt** ([sebastian.bielfeldt@desy.de](mailto:sebastian.bielfeldt@desy.de))