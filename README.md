# xtr-estimator

xtr-estimator estimates extrapolation factors for time-resolved crystallography maps.
It supports both dark vs triggered map workflows and dark vs precomputed difference-map workflows.

## Installation

Python 3.12+ is required.

```bash
conda create -n xtr-estimator python=3.12 pip
conda activate xtr-estimator
pip install -e .
```

For test dependencies:

```bash
pip install -e ".[tests]"
```

## How the code works

The runtime flow is:

1. Build a validated Settings object.
2. Load maps according to comparison type:
   - triggered mode: map_dark + map_triggered
   - diff mode: map_dark + map_diff
3. Prepare/scale maps and build an inclusion mask.
4. Estimate extrapolation factors and generate a plot.
5. Save the executed config and optional output files.

## Configuration model

Top-level settings groups:

- general
- input_files
- masking
- map_processing
- plot

Comparison mode is inferred from input_files:

- map_diff present -> diff mode
- map_triggered present -> triggered mode

If both are present, the model keeps current comparison_type and emits a warning.

## Using the API in Python

```python
from xtr_estimator.main import parse_settings, execute_as_main

config = parse_settings(
    data_yaml="examples/scripts/pl30ns_meteor.yaml",
    extra_overrides={
        "general": {"name_machine": "demo_run"},
        "plot": {"show_plot": False, "save_to_file": True},
    },
)

prediction = execute_as_main(config, show=False)
print(prediction)
```

Programmatic settings construction is also supported via xtr_estimator.configuration.Settings.

## CLI entry points

Main entry point:

```bash
xtr_estimator.plot_xtr run --from_yaml examples/scripts/pl30ns_meteor.yaml
```

You can add dot-notation overrides as extra args:

```bash
python -m xtr_estimator.main run --from_yaml examples/scripts/pl30ns_meteor.yaml masking.sigma=5 plot.show_plot=false
```

## Outputs

By default the pipeline writes:

- executed configuration YAML in general.output_folder
- several temporary files also in general.output_folder
- extrapolation figure in general.plot_folder

## Authors

- Sebastian Bielfeldt (sebastian.bielfeldt@desy.de)
- Thomas Lane (thomas.joseph.lane@gmail.com)