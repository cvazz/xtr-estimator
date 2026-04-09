import matplotlib.pyplot as plt
import os
import sys

from omegaconf import DictConfig, OmegaConf

from .configuration import get_config
from .masking import make_inclusion_mask
from .processing import get_maps, get_maps_diff, prepare_maps
from .estimation import plot_extrapolation_estimate
from .logger import setup_logger

logger = setup_logger()




def execute_main(config: DictConfig | dict, save2file: bool = False, show: bool = True) -> None:
    """The actual processing logic."""
    # Ensure we have regular dict
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    if config["general"]["comparison_type"] == "diff":
        map_dark, diffmap = get_maps_diff(config)
        diffmap_np = diffmap.to_3d_numpy_map(map_sampling=3)
        map_dark_np = map_dark.to_3d_numpy_map(map_sampling=3)
        print(diffmap_np.shape, map_dark_np.shape)
    elif config["general"]["comparison_type"] == "triggered":
        unscaled_dark, unscaled_triggered = get_maps(config)
        diffmap, map_dark, _ = prepare_maps(unscaled_dark, unscaled_triggered, config)
    else:
        raise ValueError(
            f"Unknown comparison type: {config['general']['comparison_type']}"
        )
    inclusion_mask = make_inclusion_mask(diffmap, map_dark, config)
    fig, axs, _ = plot_extrapolation_estimate(diffmap, map_dark, inclusion_mask, config)
    filename = os.path.join(config["general"]["output_folder"], 
                           f"{config["general"]["name_machine"]}_extrapolation_estimate.png")
    if config["plot"]["save_to_file"]:
        fig.savefig(filename)
    if config["plot"]["show_plot"]:
        plt.show()
    else:
        plt.close(fig)



def main():
    """Entry point for command line: 'python -m xtr_estimator.monster'"""
    # Grab the first arg as the data yaml, the rest as dot-notation overrides
    data_path = sys.argv[1] if len(sys.argv) > 1 and not "=" in sys.argv[1] else None
    cli_overrides = sys.argv[1:] if not data_path else sys.argv[2:]

    cfg = get_config(data_yaml=data_path, overrides=cli_overrides)
    execute_main(cfg)


if __name__ == "__main__":
    main()
