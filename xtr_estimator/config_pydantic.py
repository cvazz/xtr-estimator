from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class BaseModelDictlike(BaseModel):
    def __getitem__(self, item):
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(item)

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
    def advanced(cls):
        return cls(
            sigma=3.0, min_blob_size=3.0, blocking_radius=1.5,
            blocking_percentile=95.0, exclude_solvent=True,
            dark_size_threshold=0.1, exclude_positive_diffmap=True,
            exclude_large_occupancy_outliers=False
        )

    @classmethod
    def simple(cls):
        return cls(
            sigma=3.0, min_blob_size=0.1, blocking_radius=0.1,
            blocking_percentile=0.1, exclude_solvent=True,
            dark_size_threshold=0.1, exclude_positive_diffmap=True,
            exclude_large_occupancy_outliers=False
        )

# --- Other Grouped Settings ---

class MapProcessingSettings(BaseModelDictlike):
    diffmap_type: str = "tv"
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
            return self.input_output_folder.rstrip("/") + "/"
        return f"./tmp/{self.name_machine}/"
# --- Main Settings (The Entry Point) ---

# class Settings(BaseSettings):
#     masking: MaskingSettings = MaskingSettings.advanced() # Default to advanced
#     map_processing: MapProcessingSettings = MapProcessingSettings()
#     plot: PlotSettings = PlotSettings()
#     input_files: InputFileSettings
    
#     # General section with interpolation logic
#     name_machine: str = "unnamed_experiment"
#     _name_human: Optional[str] = None
#     _output_folder: Optional[str] = None
#     map_sampling: int = 3
#     high_resolution_limit: float = 0.1
#     comparison_type: str = "triggered"

#     @computed_field
#     @property
#     def name_human(self) -> str:
#         return self._name_human or self.name_machine

#     @computed_field
#     @property
#     def output_folder(self) -> str:
#         return self._output_folder or f"./tmp/{self.name_machine}/"

#     @computed_field
#     @property
#     def pdbloc_dark(self) -> str:
#         return self.input_files.pdb_dark

#     # Allow dict-like access for legacy code
#     def __getitem__(self, item):
#         # This allows config["masking"] or config["name_machine"]
#         try:
#             return getattr(self, item)
#         except AttributeError:
#             raise KeyError(item)

#     model_config = SettingsConfigDict(
#         env_nested_delimiter='__',
#         case_sensitive=False
#     )

class Settings(BaseSettings):
    # Nested Modules
    general: GeneralSettings = GeneralSettings()
    input_files: InputFileSettings
    masking: MaskingSettings = MaskingSettings.advanced()
    map_processing: MapProcessingSettings = MapProcessingSettings()
    plot: PlotSettings = PlotSettings()

    @model_validator(mode='after')
    def sync_general_paths(self) -> 'Settings':
        # Injection: Copy pdb_dark from input_files into general
        self.general.pdbloc_dark = self.input_files.pdb_dark
        return self

    model_config = SettingsConfigDict(
        env_nested_delimiter='__',
        # This allows you to pass {"name_human": "..."} to GeneralSettings
        # and have it map to input_name_human automatically
        populate_by_name=True 
    )
    def __getitem__(self, item):
        """Allows config['general'] access patterns."""
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(item)