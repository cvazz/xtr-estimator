# tests/conftest.py
import pytest
import os


@pytest.fixture
def sample_data_path():
    """Returns the path to the examples data folder."""
    return os.path.join(os.path.dirname(__file__), "../examples/data/rsEGFP2")


# @pytest.fixture
# def test_config(sample_data_path):
#     """Provides a config merged with the example data YAML."""
#     yaml_path = os.path.join(sample_data_path, "conf.yaml")
#     return get_config(data_yaml=yaml_path)


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


# from xtr_estimator.configuration import (
#     GeneralSettingsj,
#     MapProcessingSettings,
#     PlotSettings,
# )


@pytest.fixture
def test_overrides(tmp_path) -> dict:
    """Provides a list of overrides to apply on top of the YAML config during tests."""
    temp_dir = str(tmp_path)

    return dict(
        general=dict(
            high_resolution_limit=5, output_folder=temp_dir, plot_folder=temp_dir
        ),
        map_processing=dict(diffmap_type="vanilla"),
        plot=dict(save_to_file=False, show_plot=False),
    )
