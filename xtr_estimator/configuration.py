import os
from hydra import initialize, compose
from omegaconf import OmegaConf, DictConfig
from xtr_estimator.logger import setup_logger
logger = setup_logger()

def merge_overrides(cfg, overrides):
    if isinstance(overrides, list):
        print("Overriding")
        overrides_cfg = OmegaConf.from_dotlist(overrides)
    else:
        overrides_cfg = OmegaConf.create(overrides)
    cfg = OmegaConf.merge(cfg, overrides_cfg)
    return cfg

def load_homepath():
    # This function returns the path in which t
    current_path = os.getcwd()
    path_parts = current_path.split(os.sep)
    idx = path_parts.index("xtr_estimator")
    homepath = os.sep.join(path_parts[: idx + 1]) + "/"
    return homepath


def load_figurepath():
    return "."


def minimal_masking_config():
    return {
        "sigma": 3,
        "min_blob_size": 0.03,  # in A^3
        "blocking_radius": 0.1,
        "blocking_percentile": 1,
        "exclude_solvent": False,
        "dark_size_threshold": 0.0,
        "exclude_large_occupancy_outliers": False,
    }


def get_base_config():
    # load the base config from YAML and return as a dictionary
    path = os.path.join(os.path.dirname(__file__), "../conf/config.yaml")
    with open(path, "r") as f:
        base_cfg = OmegaConf.load(f)
        # create dictionary containing only the "masking", "map_processing", and "plot" sections
        keys = ["masking", "map_processing", "plot"]
        base_dict = {key: base_cfg[key] for key in keys}
        for key in keys:
             if isinstance(base_dict[key], DictConfig):
                base_dict[key] = OmegaConf.to_container(base_dict[key], resolve=True, throw_on_missing=True)
    return base_dict


def get_config_triggered(
    dataloc_dark: str,
    dataloc_light: str,
    pdbloc_dark: str,
    columns_dark: dict,
    columns_triggered: dict,
    high_resolution_limit: float = 0.1,
    pdbloc_triggered: str | None = None,
    name_machine: str = "unnamed_experiment",
    name_human: str | None = None,
    outpath: str | None = None,
):
    config = {
        "general": {
            "name_human": name_human if name_human else name_machine,
            "name_machine": name_machine,
            "output_folder": (
                outpath if outpath else load_homepath() + "tmp/diffmap_data/"
            ),
            "map_sampling": 3,
            "high_resolution_limit": high_resolution_limit,
            "comparison_type": "triggered",
        },
        "input_files": {
            "map_dark": dataloc_dark,
            "map_triggered": dataloc_light,
            "pdb_dark": pdbloc_dark,
            "pdb_triggered": pdbloc_triggered,
            "columns_dark": columns_dark,
            "columns_triggered": columns_triggered,
            "impose_dark_phases": True,
            "columns_are_ints": False,
        },
    } | get_base_config()

    config["general"]["output_folder"] = f"./tmp/{config['general']['name_machine']}/"
    config["general"]["pdbloc_dark"] = pdbloc_dark
    return config


def get_custom_config(
    dataloc_dark: str,
    dataloc_light: str,
    pdbloc_dark: str,
    columns_dark: dict,
    columns_triggered: dict,
    high_resolution_limit: float = 0.1,
    pdbloc_triggered: str | None = None,
    name_machine: str = "unnamed_experiment",
    name_human: str | None = None,
    outpath: str | None = None,
    config_path: str = "../conf",
) -> DictConfig:

    # 1. Load the base config from YAML
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name="config")

    # 2. Define your overrides as a standard Python dictionary
    # Note: We match the structure of your YAML exactly
    
    if columns_dark.get("ints_column", None) and columns_dark.get("amplitude_column", None):
        raise ValueError("Cannot specify both 'ints_column' and 'amplitude_column' in columns_dark")
    columns_are_ints = columns_dark.get("ints_column", None) is not None
    col_name_dark = "columns_dark_ints" if columns_are_ints else"columns_dark"
    col_name_triggered = "columns_triggered_ints" if columns_are_ints else "columns_triggered"
    overrides = {
        "general": {
            "name_machine": name_machine,
            "name_human": name_human or name_machine,
            "high_resolution_limit": high_resolution_limit,
        },
        "input_files": {
            "map_dark": dataloc_dark,
            "map_triggered": dataloc_light,
            "pdb_dark": pdbloc_dark,
            "pdb_triggered": pdbloc_triggered,
            col_name_dark: columns_dark,
            col_name_triggered: columns_triggered,
            "columns_are_ints": columns_are_ints,
        },
    }

    # Add optional outpath logic
    if outpath:
        overrides["general"]["output_folder"] = outpath

    # 3. Merge the dictionary into the Hydra config
    # This replaces values in 'cfg' with values from 'overrides'
    cfg = OmegaConf.merge(cfg, overrides)

    # 4. Validate
    # This ensures all ${interpolations} work and no ??? remain
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    return cfg


def get_config_diff(
    dataloc_dark: str,
    dataloc_diff: str,
    pdbloc_dark: str,
    columns_dark: dict,
    columns_diff: dict,
    pdbloc_light: str | None = None,
    high_resolution_limit: float = 0.1,
    name_machine: str = "unnamed_experiment",
    name_human: str | None = None,
    outpath: str | None = None,
):
    config = {
        "general": {
            "name_human": name_human if name_human else name_machine,
            "name_machine": name_machine,
            "output_folder": (
                outpath if outpath else "./tmp/diffmap_data/"
            ),
            "map_sampling": 3,
            "high_resolution_limit": high_resolution_limit,
            # "data_type": "diff",  # "triggered" or "diffmap"
            "comparison_type": "diff",
        },
        "input_files": {
            "map_dark": dataloc_dark,
            "map_diff": dataloc_diff,
            "pdb_dark": pdbloc_dark,
            "pdb_triggered": pdbloc_light,
            "columns_dark": columns_dark,
            "columns_diff": columns_diff,
        },
    } | get_base_config()

    config["general"]["output_folder"] = f"./tmp/{config['general']['name_machine']}/"
    config["general"]["pdbloc_dark"] = pdbloc_dark

    return config

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



def get_config(data_yaml=None, overrides=None):

    """
    The 'Heavy Lifting' config loader.
    - data_yaml: path to a local conf.yaml
    - overrides: list of dot-notation strings (e.g. ["general.sigma=5"])
                 or a dictionary.
    """

    local_cfg = (
        OmegaConf.load(data_yaml) if data_yaml and os.path.exists(data_yaml) else None
    )
    if local_cfg:
        if local_cfg.input_files.get("map_diff", None) is not None:
            print("hi?")
            mode = "diff"
        elif local_cfg.input_files.get("map_triggered", None) is not None:
            print("hi:(?")
            mode = "triggered"
        else:
            raise ValueError(
                "Could not determine mode from local config. Please ensure either 'map_diff' or 'map_triggered' is specified."
            )
    else:
        mode = "triggered"  # default mode if no local config provided
        logger.info("No local config found")


    with initialize(version_base=None, config_path="../conf"):
        # Load the base + the mode-specific schema
        cfg = compose(
            config_name="config", overrides=[f"general.comparison_type={mode}"]
        )

    # 2. Merge Local YAML if provided

    if local_cfg:
        cfg = OmegaConf.merge(cfg, local_cfg)

    # 3. Merge Overrides (CLI list or Dictionary)
    print("Overrides before merge:", overrides)
    if overrides:
        cfg = merge_overrides(cfg, overrides)

    if data_yaml:
        check_paths(cfg, data_yaml)
    # 4. Final Polish
    OmegaConf.resolve(cfg)
    return cfg