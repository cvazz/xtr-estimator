import os
from pathlib import Path

from xtr_estimator.main import execute_main, parse_settings, merge_dicts
from xtr_estimator.configuration import Settings


def test_pipeline_example_yaml_meteor(yaml_folder, test_overrides):
    """End-to-end run for examples/scripts/pl30ns_meteor.yaml."""
    yaml_path = os.path.join(yaml_folder, "pl30ns_meteor.yaml")
    cfg = parse_settings(data_yaml=yaml_path, extra_overrides=test_overrides)

    prediction = execute_main(cfg, show=False)

    assert prediction is not None
    plot_file = Path(cfg.general.plot_folder) / (
        f"{cfg.general.name_machine}_extrapolation_estimate.png"
    )
    assert plot_file.exists()


def test_pipeline_example_yaml_x8_diff(yaml_folder, test_overrides):
    """End-to-end run for examples/scripts/pl30ns_x8.yaml."""
    yaml_path = os.path.join(yaml_folder, "pl30ns_x8.yaml")
    cfg = parse_settings(data_yaml=yaml_path, extra_overrides=test_overrides)

    prediction = execute_main(cfg, show=False)

    assert cfg.general.comparison_type == "diff"
    assert prediction is not None
    plot_file = Path(cfg.general.plot_folder) / (
        f"{cfg.general.name_machine}_extrapolation_estimate.png"
    )
    assert plot_file.exists()


def test_pipeline_programmatic_triggered(data_folder, test_overrides):
    """Programmatic triggered pipeline run should complete and write plot."""
    map_dark = f"{data_folder}/1_superdark/superdark_deposit.mtz"
    map_triggered = f"{data_folder}/7_30ns/30ns_deposit.mtz"
    pdb_dark = f"{data_folder}/1_superdark/superdark_deposit.pdb"
    columns_dark = dict(
        amplitude_column="F-obs-filtered",
        phase_column="PHIF-model",
        uncertainty_column="SIGF-obs-filtered",
    )
    columns_triggered = dict(
        amplitude_column="F", phase_column="PHIF-model", uncertainty_column="SIGF"
    )

    all_settings = dict(
        input_files=dict(
            map_dark=map_dark,
            map_triggered=map_triggered,
            pdb_dark=pdb_dark,
            columns_dark=dict(**columns_dark),
            columns_triggered=dict(**columns_triggered),
        ),
        general=dict(high_resolution_limit=2.6, name_machine="PL_30ns"),
    )

    all_settings = merge_dicts(all_settings, test_overrides)
    cfg = Settings(**all_settings)

    prediction = execute_main(cfg, show=False)

    assert prediction is not None
    plot_file = Path(cfg.general.plot_folder) / (
        f"{cfg.general.name_machine}_extrapolation_estimate.png"
    )
    assert plot_file.exists()


def test_pipeline_programmatic_diff(data_folder, test_overrides):
    """Programmatic diff pipeline run should complete and write plot."""
    map_dark = os.path.join(data_folder, "1_superdark/superdark_deposit.mtz")
    map_diff = os.path.join(data_folder, "7_30ns/30ns-dark_kwt_ded.mtz")
    pdb_dark = os.path.join(data_folder, "1_superdark/superdark_deposit.pdb")

    columns_dark = dict(
        amplitude_column="F-obs-filtered",
        phase_column="PHIF-model",
        uncertainty_column="SIGF-obs-filtered",
    )
    columns_diff = dict(
        amplitude_column="KFOFOWT", phase_column="PHIKFOFOWT", uncertainty_column="SIGF"
    )

    all_settings = dict(
        input_files=dict(
            map_dark=map_dark,
            map_diff=map_diff,
            pdb_dark=pdb_dark,
            columns_dark=dict(**columns_dark),
            columns_diff=dict(**columns_diff),
        ),
    )

    all_settings = merge_dicts(all_settings, test_overrides)
    cfg = Settings(**all_settings)

    prediction = execute_main(cfg, show=False)

    assert prediction is not None
    plot_file = Path(cfg.general.plot_folder) / (
        f"{cfg.general.name_machine}_extrapolation_estimate.png"
    )
    assert plot_file.exists()
