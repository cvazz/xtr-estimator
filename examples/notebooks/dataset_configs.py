import os
import numpy as np
from xtr_estimator.configuration import load_homepath
from xtr_estimator.main import execute_as_main
from xtr_estimator.logger import setup_logger

from xtr_estimator.configuration import (
    InputFileSettings,
    GeneralSettings,
    ColumnConfig,
    DiffColumnConfig,
    Settings,
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

    ints_columns = dict(
        ints_column="IMEAN", int_uncertainty_column="SIGIMEAN"
    )

    config = Settings(
        general=dict(
            name_machine=f"myo_{delay}",
            name_human=f"Myoglobin {delay}",
            high_resolution_limit=2.3,
            comparison_type="triggered",
        ),
        input_files=dict(
            map_dark=dataloc_dark,
            map_triggered=f"{dataloc}{mtzname}.mtz",
            pdb_dark=pdbloc_dark,
            columns_dark_ints=ints_columns,
            columns_triggered_ints=ints_columns,
            columns_are_ints=True,
        ),
    )
    return config


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


def apply_config_PL_general(
    name_ending: str | int, add_light=False, diff=False
) -> dict:
    homepath = load_homepath()
    folderloc = f"{homepath}examples/data/photolyase/"
    folders = [f for f in os.listdir(folderloc) if (f[:2] != "1_") and f[0].isdigit()]
    folders = sorted(folders, key=lambda x: int(x.split('_')[0]))

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
    gen_settings = GeneralSettings(
        name_machine=f"PL_{final}",
        name_human=f"Photolyase {final}",
        high_resolution_limit=2.6,
    )
    input_files = InputFileSettings(
        map_dark=f"{folderloc}1_superdark/superdark_deposit.mtz",
        pdb_dark=f"{folderloc}1_superdark/superdark_deposit.pdb",
        pdb_triggered=f"{folderloc}{changing_bit}_deposit.pdb" if add_light else None,
        columns_dark=ColumnConfig(
            amplitude_column="F-obs-filtered",
            uncertainty_column="SIGF-obs-filtered",
            phase_column="PHIF-model",
        ),
    )
    if diff:
        input_files.map_diff = f"{folderloc}{changing_bit}-dark_kwt_ded.mtz"
        input_files.columns_diff = ColumnConfig(
            amplitude_column="KFOFOWT",
            uncertainty_column="SIGF",
            phase_column="PHIKFOFOWT",
        )
        gen_settings.comparison_type = "diff"
    else:
        input_files.map_triggered = f"{folderloc}{changing_bit}_deposit.mtz"
        input_files.columns_triggered = ColumnConfig(
            amplitude_column="F",
            uncertainty_column="SIGF",
            phase_column="PHIF-model",
        )
    config = Settings(
        general=gen_settings,
        input_files=input_files,
    )
    return config.model_dump()


# --- Helper for B12 (Difference Map Path) ---
def get_b12_diffmap_paths(folderloc: str, idx: int | str):
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
    return diffmap_name, b12_pdb


def get_folders_B12_light_with_pdb(folderloc: str, idx: int | str):
    lightmap_locs = os.listdir(folderloc)
    lightmap_locs = [f for f in lightmap_locs if "light_FPFree" in f]
    lightmap_locs = np.sort(lightmap_locs)
    lightmap_locs = np.delete(lightmap_locs, [1])
    lightmap_name = str(lightmap_locs[idx])
    logger.info(f"Selected lightmap: {lightmap_name}")
    b12_pdb = {
        "100us_30mJ.cm-2_light_FPFree.mtz": "9S0B",
        "10ns_12mJ.cm-2_light_FPFree.mtz": "9S0C",
        "10ns_30mJ.cm-2_light_FPFree.mtz": "9S08",
        "300ns_30mJ.cm-2_light_FPFree.mtz": "9S09",
        "3ms_30mJ.cm-2_light_FPFree.mtz": "9S0B",
        "3us_120mJ.cm-2_light_FPFree.mtz": None,
        "3us_12mJ.cm-2_light_FPFree.mtz": "9S0D",
        "3us_30mJ.cm-2_light_FPFree.mtz": "9S0E",
        "3us_60mJ.cm-2_light_FPFree.mtz": None,
    }.get(lightmap_name)
    return lightmap_name, b12_pdb


def apply_config_B12_general(idx: int, diff=False) -> dict:
    homepath = load_homepath()
    folderloc = f"{homepath}examples/data/b12_sacla/"

    if diff:
        othermap_name, b12_pdb = get_b12_diffmap_paths(folderloc, idx)
    else:
        othermap_name, b12_pdb = get_folders_B12_light_with_pdb(folderloc, idx)

    general = GeneralSettings(
        name_machine=f"B12_{othermap_name.split('mJ')[0]}",
        high_resolution_limit=2.3,
    )
    input_files_preset = dict(
        map_dark=f"{folderloc}9S06_dark.mtz",
        pdb_dark=f"{folderloc}9S06.pdb",
        pdb_triggered=f"{folderloc}{b12_pdb}" if b12_pdb else None,
        columns_dark=ColumnConfig(
            amplitude_column="FP", phase_column="MODEL", uncertainty_column="SIGFP"
        ),
    )
    if diff:
        general.comparison_type = "diff"
        input_files = InputFileSettings(
        map_diff = f"{folderloc}{othermap_name}",
        columns_diff = DiffColumnConfig(
            amplitude_column="QFOFOWT",
            uncertainty_column="SIGF",
            phase_column="PHIQFOFOWT",
        ),
        **input_files_preset
        )
    else:
        general.comparison_type = "triggered"
        input_files.map_triggered = f"{folderloc}{othermap_name}"
        input_files.columns_triggered = ColumnConfig(
            amplitude_column="F", phase_column="MODEL", uncertainty_column="SIGF"
        )
    config = Settings(
        general=general,
        input_files=input_files,
    )
    return config.model_dump()


# --- Main Logic ---


def main():
    cfg = apply_config_rsEGFP2()
    execute_as_main(cfg, save2file=True)


if __name__ == "__main__":
    main()
