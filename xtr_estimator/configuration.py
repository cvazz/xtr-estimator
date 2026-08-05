import os
from pathlib import Path
from typing import Optional, Literal, Tuple, List
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    import yaml
except ImportError:
    yaml = None
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
    model_config = SettingsConfigDict(
        extra="forbid",
    )

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
    phase_column: str = "MODEL"
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
    exclude_large_occupancy_outliers: Optional[float] = False
    return_stats: Optional[bool] = False

    @classmethod
    def no_mask(cls):
        return cls(
            sigma=0.1,
            min_blob_size=0.1,
            blocking_radius=0.1,
            blocking_percentile=1e-5,
            exclude_solvent=False,
            dark_size_threshold=0.0,
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
            dark_size_threshold=0.001,
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


# --- Other Grouped Settings ---


class MapProcessingSettings(BaseModelDictlike):
    diffmap_type: Literal["tv", "it_tv", "vanilla", "kweighted"] = "tv"
    simple_dark_correction: bool = False
    dark_mean_correction: bool = True
    calculate_diffmap_before_f000: bool = False
    preprocessing: bool = False
    recalculate_map_from_scratch: bool = False
    fill_NA_with_model: bool = False  # not yet implemented
    enforce_kweight: float | None = None
    enforce_tvweight: float | None = None
    enforce_ittv_weights: List[float] = [0.001, 0.01, 0.1]


class FontMultipliers(BaseModelDictlike):
    title: float = 1.05
    label: float = 1.05
    tick: float = 0.85
    legend: float = 0.95
    annotation: float = 1.0  # the (0.95,0.95) stats box + rho labels
    rhos: float = 1.0  #  rho labels
    outside_title: float = 2.0

class Solvent(BaseModelDictlike):
    large: Tuple[float, float] = (0.25, 0.10)
    small: Tuple[float, float] = (0.95, 0.65)

class Aesthetics(BaseModelDictlike):
    scale: float = 1.0
    base_fontsize: float = 8.0
    font: FontMultipliers = FontMultipliers()

    markersize: float = 3.5
    ignored_markersize_frac: float = 0.5
    optimal_markerarea: float = 200.0
    linewidth: float = 1.1
    grid_linewidth: float = 0.5

    point_alpha: float = 0.5  # moved from plot_points
    fill_alpha: float = 0.18
    anno_alpha: float = 0.8
    grid_alpha: float = 0.55

    textwidth_pt: float = 469.75502
    width_fraction: float = 1.0
    ratio: float = 1.
    panels_per_row: int = 1  # fallback; overridden by the figure-owner
    legend_ncol: int = 3

    solvent: Solvent = Solvent()

    def resolved(self) -> dict:
        s = self.scale
        base = self.base_fontsize * s
        return {
            "title": base * self.font.title,
            "label": base * self.font.label,
            "tick": base * self.font.tick,
            "legend": base * self.font.legend,
            "outside_title": base * self.font.outside_title,
            "annotation": base * self.font.annotation,
            "rhos": base * self.font.rhos,
            "markersize": self.markersize * s,
            "optimal_markerarea": self.optimal_markerarea * s ,
            "ignored_markersize": self.markersize * self.ignored_markersize_frac * s,
            "linewidth": self.linewidth * s,
            "grid_linewidth": self.grid_linewidth * s,
        }

    # def figure_size(self, n_panels: int) -> tuple[float, float]:
    #     width = self.textwidth_pt * self.width_fraction / 72.27
    #     return (width* self.panel_aspect, (width / n_panels) * self.panel_aspect)
    def figure_size(self, panels_per_row=None, n_rows=1):
        width = self.textwidth_pt * self.width_fraction / 72.27
        return (width, (width / panels_per_row) * self.ratio * n_rows)


class PlotSettings(BaseModelDictlike):
    show_ignored_voxels: bool = True
    set_ylim: Tuple[float | None, float | None] | None = None
    std_cutoff: float = 3.0
    solvent_density: float = 0.4
    minimum_datapoints: int = 10
    show_plot: bool = True
    save_to_file: bool = True
    binary_background: bool = False
    legend: bool = True
    aesthetics: Aesthetics = Aesthetics()
    show_rho: bool = False


class InputFileSettings(BaseModelDictlike):
    data_folder: Optional[str] = None
    map_dark: str = Field(..., description="MANDATORY")
    map_triggered: Optional[str] = None
    map_diff: Optional[str] = None
    pdb_dark: str = Field(..., description="MANDATORY")
    pdb_triggered: Optional[str] = None
    cif_file: Optional[str] = None
    columns_dark: ColumnConfig = ColumnConfig()
    columns_dark_ints: IntColumnConfig = IntColumnConfig()
    columns_triggered: ColumnConfig = ColumnConfig()
    columns_triggered_ints: IntColumnConfig = IntColumnConfig()
    columns_diff: DiffColumnConfig = DiffColumnConfig()
    impose_dark_phases: bool = True
    columns_are_ints: bool = False
    high_resolution_limit: Optional[float] = None


class GeneralSettings(BaseModelDictlike):
    model_config = ConfigDict(populate_by_name=True)
    name_machine: str = "unnamed_experiment"

    # We use an alias or a different name for the 'input'
    # to avoid name collisions with the computed property
    input_name_human: Optional[str] = Field(default=None, alias="name_human")
    input_output_folder: Optional[str] = Field(default=None, alias="output_folder")
    input_plot_folder: Optional[str] = Field(default=None, alias="plot_folder")
    pdbloc_dark: Optional[str] = None

    map_sampling: int = 5
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
    def ensure_dark_mean(self) -> "Settings":
        # Injection: Copy pdb_dark from input_files into general
        if self.plot.binary_background:
            self.map_processing.dark_mean_correction = True
            self.map_processing.simple_dark_correction = False
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


def merge_dicts(all_settings, test_overrides):
    for section in test_overrides:
        if section not in all_settings:
            all_settings[section] = {}
        for key in test_overrides[section]:
            all_settings[section][key] = test_overrides[section][key]
    for section in all_settings:
        # get all keys that start with input_
        # delete them
        input_keys = [key for key in all_settings[section] if key.startswith("input_")]
        for key in input_keys:
            del all_settings[section][key]
    return all_settings


def merge_settings(
    base_settings: Settings, extra_settings: Settings | dict
) -> Settings:
    if isinstance(extra_settings, Settings):
        extra_settings = extra_settings.model_dump()
    elif not extra_settings:
        return base_settings
    base_dict = base_settings.model_dump()
    merged_dict = merge_dicts(base_dict, extra_settings)
    return Settings(**merged_dict)
