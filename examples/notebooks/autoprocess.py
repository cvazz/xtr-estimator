from functools import partial
from multiprocessing import Pool
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import reciprocalspaceship as rs

from xtr_estimator.estimation import plot_extrapolation_estimate
from xtr_estimator.logger import setup_logger

# from xtr_estimator.main import execute_main
from xtr_estimator.masking import make_inclusion_mask
from xtr_estimator.processing import get_maps, prepare_maps
from xtr_estimator.refinement import run_command, run_single_refinement
from xtr_estimator.refinement import combine_and_weight_structures
from xtr_estimator.refinement import extract_simple_stats
from xtr_estimator.xtr_maps import find_rfree_column

from dataset_configs import apply_config_B12_general_light, apply_config_PL_general
from xtr_estimator.xtr_maps import save_to_folder

logger = setup_logger()


def make_folder_name(config, diffmap_type=""):
    if config["input_files"]["pdb_triggered"] is None:
        raise ValueError("Triggered PDB must be provided in config for auto processing")
    general_config = config["general"]
    folder = f"./tmp/{general_config['name_machine']}_xtr/"
    os.makedirs(folder, exist_ok=True)
    parameters = dict()
    parameters["folder"] = folder
    parameters["xtr_prefix"] = general_config["name_machine"] + f"_{diffmap_type}"
    parameters["diffmap_prefix"] = (
        general_config["name_machine"] + f"_{diffmap_type}_diff.mtz"
    )
    parameters.update(
        {
            "dark_map": config["input_files"]["map_dark"],
            "triggered_map": config["input_files"]["map_triggered"],
            "dark_model": config["input_files"]["pdb_dark"],
            "triggered_model": config["input_files"]["pdb_triggered"],
        }
    )
    output_pdb = parameters["folder"] + "combined_weighted.pdb"
    parameters["combined_model"] = output_pdb
    return parameters





def extrapolation(config, parameters):
    unscaled_dark, unscaled_triggered = get_maps(config)
    diffmap, map_dark, _ = prepare_maps(unscaled_dark, unscaled_triggered, config)
    inclusion_mask = make_inclusion_mask(diffmap, map_dark, config)
    fig, ax, prediction_tuple = plot_extrapolation_estimate(
        diffmap, map_dark, inclusion_mask, config
    )
    filename = os.path.join(
        config["general"]["output_folder"],
        f"{config["general"]["name_machine"]}_extrapolation_estimate.png",
    )
    # if config["plot"]["save_to_file"]:
    fig.savefig(filename)

    dataloc_dark = config["input_files"]["map_dark"]
    ds_dark = rs.read_mtz(dataloc_dark)
    rfree_column = find_rfree_column(ds_dark)
    rfree = ds_dark[rfree_column] 
    filelocs = save_to_folder(
        diffmap,
        map_dark,
        parameters,
        config["input_files"],
        {"best_vacuum": 1 / prediction_tuple[0]},
        rfree_flags=rfree,
    )
    return filelocs[0], prediction_tuple


# def refine_xtr(parameters, prediction_tuple, datafile):


# parameters["xtr_model"] = pdb_name
# new_st = combine_and_weight_structures(
#     parameters["files"]["dark_model"],
#     parameters["files"]["combined_model"],
#     st1weight=1 - xtr_estimate,
#     threshold=0.3,
# )
# new_st.write_pdb(parameters["combined_model"])


def combine_and_refine(occ_val, structure1, structure2, parameters, run_id_base):
    run_id_comb = f"{run_id_base}_occ{occ_val:.2f}"

    base_out = Path(parameters["folder"]).resolve()
    folderloc = base_out / str(run_id_comb)
    output_pdb = parameters["folder"] + f"combined_occ{occ_val:.2f}.pdb"

    if folderloc.exists():
        try:
            prefix = f"refine_{run_id_comb}"
            log_file = folderloc / f"{prefix}.log"
            stats = extract_simple_stats(log_file)
            if stats.get("r_work", None) is None and stats.get("r_free", None) is None:
                raise ValueError(f"Stats do not contain r_work or r_free: {stats}")

            print(f"Already processed run: {run_id_comb}, stats: {stats}")
            return stats
        except Exception as e:
            print(f"Error processing existing run: {run_id_comb}, error: {e}")
            print("Re-running refinement for this run_id...")

    new_st = combine_and_weight_structures(
        structure1, structure2, st1weight=1 - occ_val, threshold=0.3
    )
    new_st.write_pdb(output_pdb)
    stats, _, _ = run_single_refinement(
        output_pdb, parameters["triggered_map"], run_id_comb, parameters["folder"]
    )
    return stats
def comprehensive_xtr_analysis(config):
    parameters = make_folder_name(config)
    base_out = Path(parameters["folder"]).resolve()
    run_id_comb = "vacuum"
    pdb_name = base_out / f"{run_id_comb}_final.pdb"
    if not pdb_name.exists():
        print("pdb name does not exist, running extrapolation and refinement...")
        datafile, prediction_tuple = extrapolation(config, parameters)
        parameters["xtr_map"] = datafile
        (parameters, prediction_tuple, datafile)
        if parameters.get("shake_triggered_model", False):
            triggered_model = parameters["triggered_model"][:-4] + "_minimized.pdb"
            if not Path(triggered_model).exists():
                print(f"Minimizing triggered model and saving to {triggered_model}...")
                cmd = ["phenix.minimize_geometry", parameters["triggered_model"]]
                log_name = "minimize_geometry+"+triggered_model[:-4] + ".log"
                run_command(cmd,  log_name, parameters["folder"])
            parameters["triggered_model"] = triggered_model

        stats, mtz_name, pdb_name = run_single_refinement(
            parameters["triggered_model"],
            datafile,
            run_id_comb,
            parameters["folder"],
        )
        parameters["xtr_model"] = pdb_name
    else:
        print("pdb name exists, skipping extrapolation and refinement...")
        parameters["xtr_model"] = pdb_name

    comb_ref_xtr = partial(
        combine_and_refine,
        structure1=parameters["dark_model"],
        structure2=str(parameters["xtr_model"]),
        run_id_base="combined_model_vs_triggered_amplitudes",
        parameters=parameters,
    )

    comb_ref_model = partial(
        combine_and_refine,
        structure1=parameters["dark_model"],
        structure2=str(parameters["triggered_model"]),
        run_id_base="author_model_vs_triggered_amplitudes",
        parameters=parameters,
    )
    occ_values = np.arange(0.1, 0.9, 0.05)
    with Pool() as pool:
        results = pool.map(comb_ref_xtr, occ_values)
        results = pool.map(comb_ref_model, occ_values)
    print(results)
    # comb_ref(occ_val=occ_val)
def main():
    if False:
        config = apply_config_B12_general_light(3)
        config.general.high_resolution_limit = 2.3
    else:
        config = apply_config_PL_general("30ns", add_light=True)
    comprehensive_xtr_analysis(config)


if __name__ == "__main__":
    main()
