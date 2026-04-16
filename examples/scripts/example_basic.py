from xtr_estimator.configuration import load_homepath
from xtr_estimator.main import execute_main, get_config

from xtr_estimator.config_pydantic import (
    InputFileSettings,
    GeneralSettings,
    ColumnConfig,
    Settings,
    MaskingSettings,
    MapProcessingSettings,
    PlotSettings,
)


def apply_config_rsEGFP2() -> dict:
    folderloc = f"{load_homepath()}examples/data/rsEGFP2/"

    # Initialize the Settings object
    # This automatically runs all validation and @computed_field logic
    config = Settings(
        general = GeneralSettings(
            name_machine="rsEGFP2",
            high_resolution_limit=1.6,
            comparison_type="triggered",
        ),
        input_files=InputFileSettings(
            map_dark=folderloc + "scaled-test-data.mtz",
            map_triggered=folderloc + "scaled-test-data.mtz",
            pdb_dark=folderloc + "8a6g.pdb",
            columns_dark=ColumnConfig(
                amplitude_column="F_off",
                phase_column="PHIC_nochrom",
                uncertainty_column="SIGF_off",
            ),
            columns_triggered=ColumnConfig(
                amplitude_column="F_on",
                phase_column="PHIC_chrom",
                uncertainty_column="SIGF_on",
            ),
        ),
        # You can specify a preset here if needed:
        # masking=MaskingSettings.advanced()
    )

    # Return as a dictionary for backwards compatibility with your existing pipeline
    return config.model_dump()

def main():
    # Load defaults + local yaml
    cfg = apply_config_rsEGFP2()

    
    # Manual override in code
    # cfg.general.name_human = "Modified_Experiment (Example)"
    
    execute_main(cfg, save2file=True)



if __name__ == "__main__":
    main()
