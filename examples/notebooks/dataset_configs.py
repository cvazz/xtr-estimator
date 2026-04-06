import os
import numpy as np

from xtr_estimator.configuration import get_custom_config, get_file_config_diff_only
from xtr_estimator.configuration import load_homepath
from xtr_estimator.logger import setup_logger

logger = setup_logger()


def get_myo_trig(print_it=False):
    return {
        "-0.1ps": "5cn4",
        "0.0ps": "5cn5",
        "0.1ps": "5cn6",
        "0.2ps": "5cn7",
        "0.3ps": "5cn8",
        "0.4ps": "5cn9",
        "0.5ps": "5cnb",
        "0.6ps": "5cnc",
        "3ps": "5cnd",
        "10ps": "5cne",
        "50ps": "5cnf",
        "150ps": "5cng",
    }


def apply_config_myoglobin_general(idx: str | int) -> dict:
    homepath = load_homepath()
    dataloc = homepath + "examples/data/myoglobin/"
    if True:
        dataloc_dark = dataloc + "5cmv.mtz"
        pdbloc_dark = dataloc + "5CMV.pdb"
    else:
        dataloc_dark = dataloc + "5cn4.mtz"
        pdbloc_dark = dataloc + "5CN4.pdb"

    high_resolution_limit = 2.3
    delay, mtzname = None, None

    for ii, (key, value) in enumerate(get_myo_trig().items()):
        if ii == idx or idx == key:
            delay, mtzname = key, value
    if delay is None or mtzname is None:
        raise ValueError(f"Could not find matching delay and mtzname for index {idx}")

    dataloc_light = dataloc + mtzname + ".mtz"
    name_machine = f"myo_{delay}"
    name_human = f"Myoglobin {delay}"
    ints_columns = dict(ints_column="IMEAN", int_uncertainty_column="SIGIMEAN")
    config = get_custom_config(
        dataloc_dark=dataloc_dark,
        dataloc_light=dataloc_light,
        pdbloc_dark=pdbloc_dark,
        columns_dark=ints_columns,
        columns_triggered=ints_columns,
        high_resolution_limit=high_resolution_limit,
        name_machine=name_machine,
        name_human=name_human,
        outpath=None,
    )

    from omegaconf import OmegaConf, Container

    if isinstance(config, Container):
        config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    return config

