import os
import numpy as np
import warnings
from xtr_estimator.configuration import load_homepath
from xtr_estimator.main import execute_main
from xtr_estimator.logger import setup_logger

from xtr_estimator.config_pydantic import (
    InputFileSettings,
    GeneralSettings,
    ColumnConfig,
    DiffColumnConfig,
    IntColumnConfig,
    Settings,
    MaskingSettings,
)

logger = setup_logger()


# --- Helper for Myoglobin ---


def get_myo_trig():
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
    dataloc = f"{load_homepath()}examples/data/myoglobin/"
    dataloc_dark = f"{dataloc}5cmv.mtz"
    pdbloc_dark = f"{dataloc}5CMV.pdb"

    delay, mtzname = None, None
    for ii, (key, value) in enumerate(get_myo_trig().items()):
        if ii == idx or idx == key:
            delay, mtzname = key, value

    if not mtzname:
        raise ValueError(f"Could not find matching delay for index {idx}")

    ints_columns = IntColumnConfig(
        ints_column="IMEAN", int_uncertainty_column="SIGIMEAN"
    )

    config = Settings(
        general=GeneralSettings(
            name_machine=f"myo_{delay}",
            name_human=f"Myoglobin {delay}",
            high_resolution_limit=2.3,
            comparison_type="triggered",
        ),
        input_files=InputFileSettings(
            map_dark=dataloc_dark,
            map_triggered=f"{dataloc}{mtzname}.mtz",
            pdb_dark=pdbloc_dark,
            columns_dark_ints=ints_columns,
            columns_triggered_ints=ints_columns,
            columns_are_ints=True,
        ),
    )
    return config.model_dump()
# --- Helper for rsEGFP2 ---


def apply_config_rsEGFP2() -> dict:
    folderloc = f"{load_homepath()}meteor/test/data/"

    config = Settings(
        general=GeneralSettings(
            name_machine="rsEGFP2",
            high_resolution_limit=1.6,
            comparison_type="triggered",
        ),
        input_files=InputFileSettings(
            map_dark=f"{folderloc}scaled-test-data.mtz",
            map_triggered=f"{folderloc}scaled-test-data.mtz",
            pdb_dark=f"{folderloc}8a6g.pdb",
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
    )
    return config.model_dump()



# --- Helper for Photolyase ---


def apply_config_PL_general(name_ending: str | int, add_light=False) -> dict:
    homepath = load_homepath()
    folderloc = f"{homepath}examples/data/photolyase/"

    folders = [f for f in os.listdir(folderloc) if f[:2] != "1_"]
    out = None
    for ii, f in enumerate(folders):
        if (isinstance(name_ending, str) and f.endswith(name_ending)) or (
            isinstance(name_ending, int) and ii == name_ending
        ):
            out = f
            break

    if not out:
        raise ValueError(f"No folder matching {name_ending} in {folderloc}")

    final = out.split("_")[-1]
    changing_bit = f"{out}/{final}"

    config = Settings(
        general=GeneralSettings(
            name_machine=f"PL_{final}",
            name_human=f"Photolyase {final}",
            high_resolution_limit=2.6,
        ),
        input_files=InputFileSettings(
            map_dark=f"{folderloc}1_superdark/superdark_deposit.mtz",
            map_triggered=f"{folderloc}{changing_bit}_deposit.mtz",
            pdb_dark=f"{folderloc}1_superdark/superdark_deposit.pdb",
            pdb_triggered=(
                f"{folderloc}{changing_bit}_deposit.pdb" if add_light else None
            ),
            columns_dark=ColumnConfig(
                amplitude_column="F-obs-filtered",
                uncertainty_column="SIGF-obs-filtered",
                phase_column="PHIF-model",
            ),
            columns_triggered=ColumnConfig(
                amplitude_column="F",
                uncertainty_column="SIGF",
                phase_column="PHIF-model",
            ),
        ),
    )
    return config.model_dump()


# --- Helper for B12 (Difference Map Path) ---


def apply_config_B12_general(idx: int) -> dict:
    homepath = load_homepath()
    folderloc = f"{homepath}data/b12_sacla/"

    diffmap_locs = sorted([f for f in os.listdir(folderloc) if "qFoFo" in f])
    # Reproduce your delete logic: np.delete(diffmap_locs, [1])
    diffmap_locs = [f for i, f in enumerate(diffmap_locs) if i != 1]
    diffmap_name = diffmap_locs[idx]

    b12_pdb = {
        "100us_30mJ.cm-2_SACLA_qFoFo.mtz": "9S0B",
        "10ns_12mJ.cm-2_SACLA_qFoFo.mtz": "9S0C",
        "10ns_30mJ.cm-2_SACLA_qFoFo.mtz": "9S08",
        "300ns_30mJ.cm-2_SACLA_qFoFo.mtz": "9S09",
        "3ms_30mJ.cm-2_SACLA_qFoFo.mtz": "9S0B",
        "3us_12mJ.cm-2_SACLA_qFoFo.mtz": "9S0D",
        "3us_30mJ.cm-2_SACLA_qFoFo.mtz": "9S0E",
    }.get(diffmap_name)

    config = Settings(
        general=GeneralSettings(
            name_machine=f"B12_{diffmap_name.split('mJ')[0]}",
            high_resolution_limit=2.3,
            comparison_type="diff",
        ),
        input_files=InputFileSettings(
            map_dark=f"{folderloc}9S06_dark.mtz",
            map_diff=f"{folderloc}{diffmap_name}",
            pdb_dark=f"{folderloc}9S06.pdb",
            pdb_triggered=f"{folderloc}{b12_pdb}.pdb" if b12_pdb else None,
            columns_dark=ColumnConfig(
                amplitude_column="FP",
                phase_column="PHIF-model",
                uncertainty_column="SIGFP",
            ),
            columns_diff=DiffColumnConfig(
                amplitude_column="QFOFOWT",
                uncertainty_column="SIGF",
                phase_column="PHIQFOFOWT",
            ),
        ),
    )
    return config.model_dump()


# --- Main Logic ---


def main():
    cfg = apply_config_rsEGFP2()
    execute_main(cfg, save2file=True)


if __name__ == "__main__":
        # print(get_folders_B12_diff_with_pdb())
    # print(get_folders_B12())
    # print(apply_config_B12_general_light(0))
    main()
