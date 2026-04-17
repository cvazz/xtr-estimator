import os
from xtr_estimator.main import parse_settings, merge_dicts
from xtr_estimator.configuration import Settings


def test_parse_settings_from_yaml_applies_overrides(yaml_folder, test_overrides):
    """Configuration parsing should merge YAML values with explicit overrides."""
    yaml_path = os.path.join(yaml_folder, "pl30ns_meteor.yaml")
    cfg = parse_settings(data_yaml=yaml_path, extra_overrides=test_overrides)

    assert cfg.general.comparison_type == "triggered"
    assert cfg.general.high_resolution_limit == 5
    assert cfg.plot.save_to_file is True
    assert cfg.plot.show_plot is False


def test_merge_dicts_overwrites_existing_values():
    base = {"general": {"high_resolution_limit": 2.6, "name_machine": "A"}}
    overrides = {"general": {"high_resolution_limit": 5}}
    merged = merge_dicts(base, overrides)

    assert merged["general"]["high_resolution_limit"] == 5
    assert merged["general"]["name_machine"] == "A"


def test_parse_settings_diff_yaml(yaml_folder, test_overrides):
    """Diff-mode YAML should parse to diff comparison type."""
    yaml_path = os.path.join(yaml_folder, "pl30ns_x8.yaml")
    cfg = parse_settings(data_yaml=yaml_path, extra_overrides=test_overrides)

    assert cfg.general.comparison_type == "diff"


def test_settings_programmatic_diff(data_folder, test_overrides):
    """Programmatic Settings construction for diff mode should validate."""
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
    assert cfg.general.high_resolution_limit == 5
    assert cfg.plot.show_plot is False
