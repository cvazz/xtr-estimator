from xtr_estimator.configuration import load_homepath
from xtr_estimator.estimation import plot_extrapolation_estimate
from xtr_estimator.main import execute_as_main


from xtr_estimator.configuration import (
    InputFileSettings,
    GeneralSettings,
    ColumnConfig,
    Settings,
    MapProcessingSettings,
    PlotSettings,
    MaskingSettings
)
from xtr_estimator.processing import get_maps, prepare_maps


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
        map_processing=MapProcessingSettings(
                calculate_diffmap_before_f000=False
        ),
        masking=MaskingSettings.simple(
        ),
        plot= PlotSettings(
            solvent_density = 0.3,
        ),
    )
    # config.masking.dark_size_threshold= 0.1
        # You can specify a preset here if needed:
        # masking=MaskingSettings.advanced()


    # Return as a dictionary for backwards compatibility with your existing pipeline
    return config

from xtr_estimator.masking import make_inclusion_mask
from xtr_estimator.xtr_maps import save_to_folder

def simple():
    # Load defaults + local yaml
    cfg = apply_config_rsEGFP2()
    # Manual override in code
    # cfg.general.name_human = "Modified_Experiment (Example)"
    
    prediction_tuple = execute_as_main(cfg, save2file=True)
    return prediction_tuple

def extrapolation():
    config = apply_config_rsEGFP2()
    unscaled_dark, unscaled_triggered = get_maps(config)
    diffmap, map_dark, _ = prepare_maps(unscaled_dark, unscaled_triggered, config)
    inclusion_mask = make_inclusion_mask(diffmap, map_dark, config)
    _, _, prediction_tuple = plot_extrapolation_estimate(
        diffmap, map_dark, inclusion_mask, config, compact=False
    )

    parameters = {
        "folder": config.general.output_folder,
        "xtr_prefix": "extrapolated"
        # "d": config.input_files.map_dark,
    }
    save_to_folder(diffmap, map_dark, 
                   parameters,
                   input_file_config=config.input_files,
                   save_dict={"best_guess": 1/prediction_tuple[0]},
                   )



if __name__ == "__main__":
    extrapolation()
