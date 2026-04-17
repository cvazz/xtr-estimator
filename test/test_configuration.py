# tests/test_configuration.py
import os
# import pytest
from xtr_estimator.main import execute_main, parse_settings, merge_dicts
from xtr_estimator.configuration import Settings



def test_option1_yaml_load(yaml_folder, test_overrides):
    """Test loading from YAML (Meteor TV Denoise)."""
    yaml_path = os.path.join(yaml_folder, "pl30ns_meteor.yaml")
    cfg = parse_settings(data_yaml=yaml_path, extra_overrides=test_overrides)
    assert cfg.general.comparison_type == "triggered"
    assert cfg.general.high_resolution_limit == 5
    execute_main(cfg, show=False)


def test_option2_manual_override(yaml_folder, test_overrides):
    """Test loading from YAML and manually overriding a field."""
    yaml_path = os.path.join(yaml_folder, "pl30ns_meteor.yaml")
    cfg = parse_settings(data_yaml=yaml_path, extra_overrides=test_overrides)

    # Manually override
    cfg.map_processing.diffmap_type = "kweighted"

    assert cfg.map_processing.diffmap_type == "kweighted"
    execute_main(cfg, show=False)


def test_option3_diffmap_yaml(yaml_folder, test_overrides):
    """Test loading from a YAML that points directly to a difference map (X8)."""
    yaml_path = os.path.join(yaml_folder, "pl30ns_x8.yaml")
    cfg = parse_settings(data_yaml=yaml_path, extra_overrides=test_overrides)

    assert cfg.general.comparison_type == "diff"
    execute_main(cfg, show=False)




def test_option1_alt_programmatic_triggered(data_folder, test_overrides):
    """Test building triggered config purely through Python."""

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
    high_resolution_limit = 2.6

    name_machine = "PL_30ns"
    all_settings = dict(
        input_files=dict(
            map_dark=map_dark,
            map_triggered=map_triggered,
            pdb_dark=pdb_dark,
            columns_dark=dict(**columns_dark),
            columns_triggered=dict(**columns_triggered),
        ),
        general=dict(
            high_resolution_limit=high_resolution_limit, name_machine=name_machine
        ),
    )
    all_settings = merge_dicts(all_settings, test_overrides)
    cfg = Settings(**all_settings)

    execute_main(cfg, show=False)


def test_option3_alt_programmatic_diff(data_folder, test_overrides):
    """Test building diffmap config purely through Python."""
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
            map_diff = map_diff,
            pdb_dark=pdb_dark,
            columns_dark=dict(**columns_dark),
            columns_diff=dict(**columns_diff),
        ),
    )

    all_settings = merge_dicts(all_settings, test_overrides)
    cfg = Settings(**all_settings)
    # assert cfg.general.name_machine == "PL_30ns_x8_test"
    execute_main(cfg, show=False)
