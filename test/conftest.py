# tests/conftest.py
import pytest
import os
from xtr_estimator.configuration import get_config


@pytest.fixture
def sample_data_path():
    """Returns the path to the examples data folder."""
    return os.path.join(os.path.dirname(__file__), "../examples/data/rsEGFP2")

@pytest.fixture
def test_config(sample_data_path):
    """Provides a config merged with the example data YAML."""
    yaml_path = os.path.join(sample_data_path, "conf.yaml")
    return get_config(data_yaml=yaml_path)
# tests/conftest.py

@pytest.fixture
def data_folder():
    """Absolute path to the photolyase data folder."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, "examples", "data", "photolyase")

@pytest.fixture
def yaml_folder():
    """Absolute path to the folder containing example YAMLs."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, "examples", "scripts")


@pytest.fixture
def test_overrides(tmp_path):
    temp_dir = str(tmp_path)
    
    return [
        "general.high_resolution_limit=5",
        "map_processing.diffmap_type=vanilla",
        f"general.output_folder={temp_dir}", 
        "plot.save_to_file=false",
        "plot.show_plot=false",
    ]