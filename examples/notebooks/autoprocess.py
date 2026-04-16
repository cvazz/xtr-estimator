import argparse
from copy import deepcopy
from functools import partial
from multiprocessing import Pool
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# import hydra
# from omegaconf import DictConfig
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


def make_folder_name(config):
    if config["input_files"]["pdb_triggered"] is None:
        raise ValueError("Triggered PDB must be provided in config for auto processing")
    general_config = config["general"]
    diffmap_type = config["map_processing"]["diffmap_type"]
    if config["prescribe_xtr"]:
        folder = f"./tmp/{general_config['name_machine']}_{diffmap_type}_xtr_{config['prescribe_xtr']}/"
    else:
        folder = f"./tmp/{general_config['name_machine']}_{diffmap_type}_xtr/"
    os.makedirs(folder, exist_ok=True)
    parameters = dict()
    parameters["name_machine"] = (
        general_config["name_machine"] + f"_{diffmap_type}_{config['prescribe_xtr']}"
    )
    parameters["name_human"] = general_config["name_human"]
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
    parameters["number_iterations_refinement"] = 6
    return parameters


def extrapolation(config, parameters):
    unscaled_dark, unscaled_triggered = get_maps(config)
    diffmap, map_dark, _ = prepare_maps(unscaled_dark, unscaled_triggered, config)
    if config.get("prescribe_xtr", None) is None:
        inclusion_mask = make_inclusion_mask(diffmap, map_dark, config)
        fig, ax, prediction_tuple = plot_extrapolation_estimate(
            diffmap, map_dark, inclusion_mask, config
        )
        img_name = f'{config["general"]["name_machine"]}_{config["map_processing"]["diffmap_type"]}_xtr{prediction_tuple[0]:.2f}.png'
        folders = [config["general"]["output_folder"], parameters["folder"]]
        for folder in folders:
            filename = os.path.join(folder, img_name)
            print(f"Saving extrapolation estimate plot to {filename}...")
            fig.savefig(filename)
        xtr_prescribe = ({"best_vacuum": 1 / prediction_tuple[0]},)
    else:
        xtr_prescribe = {"prescribe": 1 / config["prescribe_xtr"]}
        prediction_tuple = (config["prescribe_xtr"], None, None)

    # if config["plot"]["save_to_file"]:

    dataloc_dark = config["input_files"]["map_dark"]
    ds_dark = rs.read_mtz(dataloc_dark)
    rfree_column = find_rfree_column(ds_dark)
    rfree = ds_dark[rfree_column]
    filelocs = save_to_folder(
        diffmap,
        map_dark,
        parameters,
        config["input_files"],
        xtr_prescribe,
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
            stats["occ_val"] = occ_val

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
        output_pdb,
        parameters["triggered_map"],
        run_id_comb,
        parameters["folder"],
        number_iterations=parameters["number_iterations_refinement"],
    )
    stats["occ_val"] = occ_val
    return stats


def comprehensive_xtr_analysis(config):
    parameters = make_folder_name(config)
    base_out = Path(parameters["folder"]).resolve()
    run_id_comb = "vacuum"
    if "prescribe_xtr" in config:
        occ_val = config["prescribe_xtr"]
        pdb_name = base_out / f"{run_id_comb}_{occ_val:.2f}.pdb"
    else:
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
                log_name = "minimize_geometry+" + triggered_model[:-4] + ".log"
                run_command(cmd, log_name, parameters["folder"])
            parameters["triggered_model"] = triggered_model

        stats, mtz_name, pdb_name = run_single_refinement(
            parameters["triggered_model"],
            datafile,
            run_id_comb,
            parameters["folder"],
            number_iterations=parameters["number_iterations_refinement"],
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
        results_xtr = pool.map(comb_ref_xtr, occ_values)
        results_model = pool.map(comb_ref_model, occ_values)
    return results_xtr, results_model, parameters


def evaluate_models(results_xtr, results_model, parameters):
    print(parameters)
    # in parameters folder look for all files that contain paramteres["xtr_prefix"] and print
    occus = []
    expected_occu = retrieve_occupancies_from_folder(parameters)

    occ_vals_xtr = np.array([res["occ_val"] for res in results_xtr], dtype=float)
    r_work_xtr = np.array([res["r_work"] for res in results_xtr], dtype=float)
    r_free_xtr = np.array([res["r_free"] for res in results_xtr], dtype=float)
    occ_vals_model = np.array([res["occ_val"] for res in results_model], dtype=float)
    r_work_model = np.array([res["r_work"] for res in results_model], dtype=float)
    r_free_model = np.array([res["r_free"] for res in results_model], dtype=float)
    fig = plt.figure(figsize=(10, 5))
    if len(occus) == 1:
        plt.axvline(
            x=expected_occu, color="green", linestyle="--", label="Best Vacuum Estimate"
        )
    plt.plot(occ_vals_xtr, r_work_xtr, label="XTR R-work", marker="o", color="blue")
    plt.plot(
        occ_vals_model, r_work_model, label="Model R-work", marker="s", color="red"
    )
    plt.plot(occ_vals_xtr, r_free_xtr, label="XTR R-free", marker="o", color="cyan")
    plt.plot(
        occ_vals_model, r_free_model, label="Model R-free", marker="s", color="magenta"
    )
    # plt.axvline(x=parameters["best_vacuum"], color="green", linestyle="--", label="Best Vacuum Estimate")
    plt.legend()
    fig.savefig(
        os.path.join(
            parameters["folder"],
            f"{parameters['xtr_prefix']}_rwork_rfree_comparison.png",
        )
    )


def evaluate_models_double(
    results_tv, results_k, results_model1, results_model2, parameters, exp_occus
):

    occ_vals_tv = np.array([res["occ_val"] for res in results_tv], dtype=float)
    r_work_tv = np.array([res["r_work"] for res in results_tv], dtype=float)
    r_free_tv = np.array([res["r_free"] for res in results_tv], dtype=float)
    occ_vals_model1 = np.array([res["occ_val"] for res in results_model1], dtype=float)
    r_work_model1 = np.array([res["r_work"] for res in results_model1], dtype=float)
    r_free_model1 = np.array([res["r_free"] for res in results_model1], dtype=float)
    occ_vals_k = np.array([res["occ_val"] for res in results_k], dtype=float)
    r_work_k = np.array([res["r_work"] for res in results_k], dtype=float)
    r_free_k = np.array([res["r_free"] for res in results_k], dtype=float)
    occ_vals_model2 = np.array([res["occ_val"] for res in results_model2], dtype=float)
    r_work_model2 = np.array([res["r_work"] for res in results_model2], dtype=float)
    r_free_model2 = np.array([res["r_free"] for res in results_model2], dtype=float)

    fig = plt.figure(figsize=(10, 5))

    plt.axvline(x=exp_occus[0], color="blue", linestyle="--", label="Best TV Estimate")
    plt.axvline(
        x=exp_occus[1], color="green", linestyle="--", label="Best K-weighted Estimate"
    )
    plt.plot(occ_vals_tv, r_work_tv, label="TV R-work", marker="o", color="blue")
    plt.plot(occ_vals_tv, r_free_tv, label="TV R-free", marker="o", color="cyan")

    plt.plot(occ_vals_k, r_work_k, label="K-weighted R-work", marker="o", color="green")
    plt.plot(
        occ_vals_k, r_free_k, label="K-weighted R-free", marker="o", color="darkgreen"
    )
    model_plot_args_rfree = dict(label="Model R-free", marker="s", color="red")
    model_plot_args_work = dict(label="Model R-work", marker="s", color="magenta")
    plt.plot(occ_vals_model1, r_work_model1, **model_plot_args_work)
    plt.plot(occ_vals_model1, r_free_model1, **model_plot_args_rfree)
    plt.plot(occ_vals_model2, r_work_model2, marker="s", color="red")
    plt.plot(occ_vals_model2, r_free_model2, marker="s", color="magenta")
    # plt.axvline(x=parameters["best_vacuum"], color="green", linestyle="--", label="Best Vacuum Estimate")
    plt.title(
        f"Comparison of TV and K-weighted Diffmap Types of {parameters['name_human']}"
    )
    plt.legend()
    fig.savefig(
        os.path.join(
            parameters["folder"],
            f"{parameters['xtr_prefix']}_rwork_rfree_comparison_double.png",
        )
    )

    # comb_ref(occ_val=occ_val)


def int_or_str(value):
    try:
        return int(value)
    except ValueError:
        return str(value)


def parsing():
    parser = argparse.ArgumentParser(
        description="Autoprocess X-ray crystallography data."
    )

    # 2. Define the expected arguments
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["b12", "pl"],  # Automatically enforces valid inputs
        help="Type of configuration to apply ('b12' or 'pl').",
    )

    parser.add_argument(
        "--specifier",
        type=int_or_str,
        default=1,
        help="Specifier ID for the dataset.",
    )

    parser.add_argument(
        "--dmin",
        type=float,
        default=None,
        help="High resolution limit (optional).",
    )

    parser.add_argument(
        "--diffmap_type",
        type=str,
        default=None,
        help="Type of difference map (e.g., 'tv') (optional).",
    )

    parser.add_argument(
        "--prescribe_xtr",
        type=float,
        default=None,
        help="Skip automatic extrapolation and refinement, and directly prescribe an extrapolated occupancy for evaluation.",
    )

    # 3. Parse the arguments from the command line
    return parser.parse_args()


def main_single(args, config):

    results_xtr, results_model, parameters = comprehensive_xtr_analysis(config)
    evaluate_models(results_xtr, results_model, parameters)


def retrieve_occupancies_from_folder(parameters):
    occus = []
    for file in Path(parameters["folder"]).glob(f"*{parameters['xtr_prefix']}*xtr*"):
        print(f"File in output folder: {file}")
        # cut file between last xtr and .mtz and print
        # if file.suffix == ".mtz":

        if file.suffix != ".png":
            continue
        try:
            real_occu = float(file.stem.split("xtr")[-1])
        except ValueError:
            print(f"Could not convert {file} to float")
            continue
        print(f"MTZ file: {file}, real_occu: {real_occu}")
        occus.append(real_occu)
    return occus[0] if occus else np.nan


def main_double(config):
    config_tv = deepcopy(config)
    config_tv["map_processing"]["diffmap_type"] = "tv"
    out_tv = comprehensive_xtr_analysis(config)
    expected_occu_tv = retrieve_occupancies_from_folder(out_tv[2])
    config_k = deepcopy(config)
    config_k["map_processing"]["diffmap_type"] = "kweighted"
    out_k = comprehensive_xtr_analysis(config_k)
    expected_occu_k = retrieve_occupancies_from_folder(out_k[2])

    expected_occus = (expected_occu_tv, expected_occu_k)
    evaluate_models_double(
        results_tv=out_tv[0],
        results_k=out_k[0],
        results_model1=out_tv[1],
        results_model2=out_k[1],
        parameters=out_tv[2],
        exp_occus=expected_occus,
    )


def main():
    args = parsing()
    if args.type == "b12":
        config = apply_config_B12_general_light(args.specifier)
    elif args.type == "pl":
        config = apply_config_PL_general(args.specifier, add_light=True)
    else:
        raise ValueError(f"Unknown config type: {args.type}")

    if args.dmin:
        print(config.general.high_resolution_limit)
        config.general.high_resolution_limit = args.dmin

    config = dict(config)
    if args.prescribe_xtr is not None:
        config["prescribe_xtr"] = args.prescribe_xtr

    if args.diffmap_type in ["tv", "kweighted"]:
        config["map_processing"]["diffmap_type"] = args.diffmap_type
    elif args.diffmap_type in ["both"]:
        main_double(config)
        return
    elif args.diffmap_type is not None:
        raise ValueError(f"Unknown diffmap type: {args.diffmap_type}")

    main_single(args, config)


if __name__ == "__main__":
    main()
