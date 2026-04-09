# tests/test_configuration.py
import os

from omegaconf import OmegaConf
from xtr_estimator.configuration import merge_overrides
import pytest
from xtr_estimator.main import execute_main
from xtr_estimator.configuration import get_config, get_config_triggered, get_config_diff

# def test_config_merging(test_config):
#     # test_config is provided by the fixture in conftest.py
#     assert test_config.general.name_machine == "rsEGFP2"
#     assert "masking" in test_config
#     assert test_config.masking.sigma == 3  # Check inheritance from base

# def test_path_resolution(sample_data_path, test_config):
#     # Test if the MTZ file name exists in the expected folder
#     mtz_file = test_config.input_files.map_dark
#     assert mtz_file == "scaled-test-data.mtz"

def test_option1_yaml_load(yaml_folder, test_overrides):
    """Test loading from YAML (Meteor TV Denoise)."""
    yaml_path = os.path.join(yaml_folder, "pl30ns_meteor.yaml")
    cfg = get_config(data_yaml=yaml_path, overrides=test_overrides)
    assert cfg.general.comparison_type == "triggered"
    execute_main(cfg, show=False)

def test_option2_manual_override(yaml_folder, test_overrides):
    """Test loading from YAML and manually overriding a field."""
    yaml_path = os.path.join(yaml_folder, "pl30ns_meteor.yaml")
    cfg = get_config(data_yaml=yaml_path, overrides=test_overrides)
    
    # Manually override
    cfg.map_processing.diffmap_type = "kweighted"
    
    assert cfg.map_processing.diffmap_type == "kweighted"
    execute_main(cfg, show=False)

def test_option3_diffmap_yaml(yaml_folder, test_overrides):
    """Test loading from a YAML that points directly to a difference map (X8)."""
    yaml_path = os.path.join(yaml_folder, "pl30ns_x8.yaml")
    cfg = get_config(data_yaml=yaml_path, overrides=test_overrides)
    
    assert cfg.general.comparison_type == "diff"
    execute_main(cfg, show=False)

def test_option1_alt_programmatic_triggered(data_folder, test_overrides):
    """Test building triggered config purely through Python."""
    map_dark = os.path.join(data_folder, "1_superdark/superdark_deposit.mtz")
    map_triggered = os.path.join(data_folder, "7_30ns/30ns_deposit.mtz")
    pdb_dark = os.path.join(data_folder, "1_superdark/superdark_deposit.pdb")
    
    config = get_config_triggered(
        dataloc_dark=map_dark,
        dataloc_light=map_triggered,
        pdbloc_dark=pdb_dark,
        columns_dark={"amplitude_column": "F-obs-filtered", "phase_column": "PHIF-model", "uncertainty_column": "SIGF-obs-filtered"},
        columns_triggered={"amplitude_column": "F", "phase_column": "PHIF-model", "uncertainty_column": "SIGF"},
        high_resolution_limit=2.6,
        name_machine="PL_30ns_test"
    )
    cfg = OmegaConf.create(config)
    cfg = merge_overrides(cfg, test_overrides)
    config = OmegaConf.to_container(cfg, resolve=True)

    # cfg["general"]["high_resolution_limit"] = 5
    # cfg["map_processing"]["diffmap_type"] = "kweighted"
    # cfg["plot"]["show_plot"] = False
    
    execute_main(cfg, show=False)

def test_option3_alt_programmatic_diff(data_folder, test_overrides):
    """Test building diffmap config purely through Python."""
    map_dark = os.path.join(data_folder, "1_superdark/superdark_deposit.mtz")
    map_diff = os.path.join(data_folder, "7_30ns/30ns-dark_kwt_ded.mtz")
    pdb_dark = os.path.join(data_folder, "1_superdark/superdark_deposit.pdb")
    
    config = get_config_diff(
        dataloc_dark=map_dark,
        dataloc_diff=map_diff,
        pdbloc_dark=pdb_dark,
        columns_dark={"amplitude_column": "F-obs-filtered", "phase_column": "PHIF-model", "uncertainty_column": "SIGF-obs-filtered"},
        columns_diff={"amplitude_column": "KFOFOWT", "phase_column": "PHIKFOFOWT", "uncertainty_column": "SIGF"},
        high_resolution_limit=2.6,
        name_machine="PL_30ns_x8_test"
    )
    cfg = OmegaConf.create(config)
    cfg = merge_overrides(cfg, test_overrides)
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # cfg["general"]["high_resolution_limit"] = 5
    # cfg["map_processing"]["diffmap_type"] = "vanilla"
    # cfg["plot"]["show_plot"] = False

    assert config["general"]["name_machine"] == "PL_30ns_x8_test"
    execute_main(config, show=False)