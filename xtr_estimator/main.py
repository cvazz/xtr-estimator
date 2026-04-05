import matplotlib.pyplot as plt
import os

from omegaconf import DictConfig, OmegaConf
import hydra

from .masking import make_inclusion_mask
from .processing import get_maps, prepare_maps
from .estimation import plot_extrapolation_estimate_new
from .logger import setup_logger

logger = setup_logger()

def get_config(data_yaml=None, overrides=None):
    """
    The 'Heavy Lifting' config loader.
    - data_yaml: path to a local conf.yaml
    - overrides: list of dot-notation strings (e.g. ["general.sigma=5"]) 
                 or a dictionary.
    """
    # 1. Load Package Defaults
    # Locate the global config relative to this file
    pkg_conf_path = os.path.join(os.path.dirname(__file__), "..", "conf", "config.yaml")
    cfg = OmegaConf.load(pkg_conf_path)

    # 2. Merge Local YAML if provided
    if data_yaml and os.path.exists(data_yaml):
        local_cfg = OmegaConf.load(data_yaml)
        cfg = OmegaConf.merge(cfg, local_cfg)

    # 3. Merge Overrides (CLI list or Dictionary)
    if overrides:
        if isinstance(overrides, list):
            overrides_cfg = OmegaConf.from_dotlist(overrides)
        else:
            overrides_cfg = OmegaConf.create(overrides)
        cfg = OmegaConf.merge(cfg, overrides_cfg)

    if data_yaml:
        check_paths(cfg, data_yaml)
    print(cfg.input_files)
    # 4. Final Polish
    OmegaConf.resolve(cfg)
    return cfg

def execute_main(config: DictConfig | dict) -> None:
    """The actual processing logic."""
    # Ensure we have regular dict
    config = OmegaConf.to_container(config, resolve=True)
    unscaled_dark, unscaled_triggered = get_maps(config)
    diffmap, map_dark, _ = prepare_maps(unscaled_dark, unscaled_triggered, config)
    inclusion_mask = make_inclusion_mask(diffmap, map_dark, config)
    _ = plot_extrapolation_estimate_new(diffmap, map_dark, inclusion_mask, config)
    plt.show()

def check_paths(cfg: DictConfig, data_path: str) -> DictConfig:
    data_dir = os.path.dirname(os.path.abspath(data_path))
        
    # List of keys that contain file paths to resolve
    path_keys = ["map_dark", "map_triggered", "pdb_dark", "pdb_triggered"]
    
    for key in path_keys:
        # Access the value (e.g., cfg.input_files.map_dark)
        val = cfg.input_files.get(key)
        
        # If the value exists and is NOT an absolute path
        if val and not os.path.isabs(val):
            # Construct path relative to the directory of the YAML file
            potential_path = os.path.join(data_dir, val)
            
            # Update config if that file actually exists there
            if os.path.exists(potential_path):
                cfg.input_files[key] = os.path.abspath(potential_path)
                logger.info(f"Resolved relative path for {key}: {cfg.input_files[key]}")

def main():
    """Entry point for command line: 'python -m xtr_estimator.monster'"""
    # Grab the first arg as the data yaml, the rest as dot-notation overrides
    data_path = sys.argv[1] if len(sys.argv) > 1 and not "=" in sys.argv[1] else None
    cli_overrides = sys.argv[1:] if not data_path else sys.argv[2:]
    
    cfg = get_config(data_yaml=data_path, overrides=cli_overrides)
    execute_main(cfg)


if __name__ == "__main__":
    main()
