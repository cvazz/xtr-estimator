import os
from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
from .logger import setup_logger
import typer

logger = setup_logger()

def load_homepath():
    # This function returns the path in which t
    current_path = os.getcwd()
    path_parts = current_path.split(os.sep)
    idx = path_parts.index("xtr_estimator")
    homepath = os.sep.join(path_parts[: idx + 1]) + "/"
    return homepath



class BaseModelDictlike(BaseModel):
    def __getitem__(self, item):
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(item)

    def __setitem__(self, key, value):
        # This allows config["key"] = value
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            # Optional: Allow setting new keys even if not in Pydantic model
            # but usually you want to stick to defined fields
            raise KeyError(f"'{type(self).__name__}' has no field '{key}'")

    def keys(self):
        return self.model_dump().keys()

    def __iter__(self):
        return iter(self.keys())


# --- Sub-Models for Columns ---


class ColumnConfig(BaseModelDictlike):
    amplitude_column: str = "F"
    phase_column: str = "PHI"
    uncertainty_column: str = "SIGF"


class IntColumnConfig(BaseModelDictlike):
    ints_column: str = "I"
    int_uncertainty_column: str = "SIGI"


class DiffColumnConfig(BaseModelDictlike):
    amplitude_column: str = "FKOFOWT"
    phase_column: str = "PHIFKOFOWT"
    uncertainty_column: str = "SIGF"


# --- Masking with Presets ---


class MaskingSettings(BaseModelDictlike):
    sigma: Optional[float] = None
    min_blob_size: Optional[float] = None
    blocking_radius: Optional[float] = None
    blocking_percentile: Optional[float] = None
    exclude_solvent: Optional[bool] = None
    dark_size_threshold: Optional[float] = None
    exclude_positive_diffmap: Optional[bool] = None
    exclude_large_occupancy_outliers: Optional[bool] = False

    @classmethod
    def no_mask(cls):
        return cls(
            sigma=0,
            min_blob_size=0.1,
            blocking_radius=1.5,
            blocking_percentile=1e-5,
            exclude_solvent=True,
            dark_size_threshold=0.1,
            exclude_positive_diffmap=True,
            exclude_large_occupancy_outliers=False,
        )

    @classmethod
    def advanced(cls):
        return cls(
            sigma=3.0,
            min_blob_size=3.0,
            blocking_radius=1.5,
            blocking_percentile=95.0,
            exclude_solvent=True,
            dark_size_threshold=0.1,
            exclude_positive_diffmap=True,
            exclude_large_occupancy_outliers=False,
        )

    @classmethod
    def simple(cls):
        return cls(
            sigma=3.0,
            min_blob_size=0.1,
            blocking_radius=0.1,
            blocking_percentile=0.1,
            exclude_solvent=True,
            dark_size_threshold=0.1,
            exclude_positive_diffmap=True,
            exclude_large_occupancy_outliers=False,
        )


# --- Other Grouped Settings ---


class MapProcessingSettings(BaseModelDictlike):
    diffmap_type: Literal["tv", "vanilla", "kweighted"] = "tv"
    dark_mean_correction: bool = True
    simple_dark_correction: bool = True
    calculate_diffmap_before_f000: bool = False
    preprocessing: bool = False


class PlotSettings(BaseModelDictlike):
    show_ignored_voxels: bool = True
    set_ylim: bool = False
    is_composite: bool = False
    std_cutoff: float = 3.0
    solvent_density: float = 0.4
    minimum_datapoints: int = 10
    show_plot: bool = True
    save_to_file: bool = True
    markersize: Optional[float] = 1
    comparison_to_reference: Optional[bool] = False


class InputFileSettings(BaseModelDictlike):
    data_folder: Optional[str] = None
    map_dark: str = Field(..., description="MANDATORY")
    map_triggered: Optional[str] = None
    map_diff: Optional[str] = None
    pdb_dark: str = Field(..., description="MANDATORY")
    pdb_triggered: Optional[str] = None
    columns_dark: ColumnConfig = ColumnConfig()
    columns_dark_ints: IntColumnConfig = IntColumnConfig()
    columns_triggered: ColumnConfig = ColumnConfig()
    columns_triggered_ints: IntColumnConfig = IntColumnConfig()
    columns_diff: DiffColumnConfig = DiffColumnConfig()
    impose_dark_phases: bool = True
    columns_are_ints: bool = False


class GeneralSettings(BaseModelDictlike):
    name_machine: str = "unnamed_experiment"

    # We use an alias or a different name for the 'input'
    # to avoid name collisions with the computed property
    input_name_human: Optional[str] = Field(default=None, alias="name_human")
    input_output_folder: Optional[str] = Field(default=None, alias="output_folder")
    input_plot_folder: Optional[str] = Field(default=None, alias="plot_folder")
    pdbloc_dark: Optional[str] = None

    map_sampling: int = 3
    high_resolution_limit: float = 0.1
    comparison_type: str = "triggered"

    @computed_field
    @property
    def name_human(self) -> str:
        return self.input_name_human or self.name_machine

    @computed_field
    @property
    def output_folder(self) -> str:
        if self.input_output_folder:
            folder = self.input_output_folder.rstrip("/") + "/"
        else:
            folder = f"./tmp/{self.name_machine}/"
        os.makedirs(folder, exist_ok=True)
        return folder

    @computed_field
    @property
    def plot_folder(self) -> str:
        if self.input_plot_folder:
            folder = self.input_plot_folder.rstrip("/") + "/"
        else:
            folder = f"./plots/{self.name_machine}/"
        os.makedirs(folder, exist_ok=True)
        return folder


# --- Main Settings (The Entry Point) ---


class Settings(BaseSettings):
    # Nested Modules
    general: GeneralSettings = GeneralSettings()
    input_files: InputFileSettings
    masking: MaskingSettings = MaskingSettings.advanced()
    map_processing: MapProcessingSettings = MapProcessingSettings()
    plot: PlotSettings = PlotSettings()

    @model_validator(mode="after")
    def sync_general_paths(self) -> "Settings":
        # Injection: Copy pdb_dark from input_files into general
        self.general.pdbloc_dark = self.input_files.pdb_dark
        return self

    @model_validator(mode="after")
    def set_comparison_type(self) -> "Settings":
        # If map_diff is provided, we assume it's a diff comparison
        if self.input_files.map_diff and self.input_files.map_triggered:
            logger.warning(
                "Both map_diff and map_triggered are provided. Choosing {self.general.comparison_type} as comparison type."
            )
        elif self.input_files.map_diff:
            self.general.comparison_type = "diff"
        elif self.input_files.map_triggered:
            self.general.comparison_type = "triggered"
        else:
            raise ValueError(
                "At least one of map_triggered or map_diff must be provided."
            )
        return self

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        # This allows you to pass {"name_human": "..."} to GeneralSettings
        # and have it map to input_name_human automatically
        populate_by_name=True,
    )

    def __getitem__(self, item):
        """Allows config['general'] access patterns."""
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(item)


def config_from_yaml(path: Path | str) -> dict:
    if isinstance(path, str):
        path = Path(path)
    final_payload = {}
    if path and path.exists():
        with open(path) as f:
            final_payload = yaml.safe_load(f) or {}
    else:
        raise FileNotFoundError(f"YAML file {path} not found.")

    settings = Settings(**final_payload)
    return settings

def dump_config(settings: Settings):
    config = settings.model_dump()
    out_dir = Path(settings.general.output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    dump_loc = out_dir / "executed_config.yaml"
    with open(dump_loc, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    typer.secho(f"💾 Config saved to {dump_loc}", fg=typer.colors.BLUE)
    return config
