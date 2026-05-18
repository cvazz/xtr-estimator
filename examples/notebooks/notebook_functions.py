import logging

import matplotlib.pyplot as plt
import numpy as np
from contextlib import redirect_stdout
from xtr_estimator.configuration import merge_settings
from xtr_estimator.masking import make_inclusion_mask
from xtr_estimator.processing import get_maps, prepare_maps
from xtr_estimator.main import xtr_logic


def plot_extrapolation_results(
    means_list, stds_list, labels_list, timepoints, title="Extrapolation Estimates"
):
    """
    Plots multiple extrapolation datasets with error bars.

    """
    # 1. Setup Style
    plt.style.use("seaborn-v0_8-muted")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    # 2. Define visual constants
    markers = ["o", "s", "^", "D", "v"]
    linestyles = ["-", "--", "-.", ":", "-"]
    x_axis = np.arange(len(timepoints))

    # 3. Iterate through data and plot
    for i, (means, stds, label) in enumerate(zip(means_list, stds_list, labels_list)):
        ax.errorbar(
            x_axis,
            means,
            yerr=stds,
            label=label,
            capsize=4,
            marker=markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)],
            elinewidth=1.5,
            alpha=0.8,
            markersize=6,
        )

    # 4. Refine Axes and Labels
    ax.set_xticks(x_axis)
    ax.set_xticklabels(timepoints, rotation=35, ha="right")

    ax.set_xlabel("Timepoints", fontsize=12, fontweight="bold")
    ax.set_ylabel("Estimate Value", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, pad=15)

    # 5. Aesthetic Clean-up
    ax.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, None)

    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="best")

    plt.tight_layout()
    return fig, ax


def create_global_inclusion_mask(
    loading_function: callable, global_mask: bool, global_overrides: dict = {}
):
    if global_mask:
        print("Calculating global inclusion mask across all configs...")
        inclusion_masks = []
        for config in loading_function(diff=False, global_overrides=global_overrides):
            unscaled_dark, unscaled_triggered = get_maps(
                config.input_files,
                high_resolution_limit=config.general.high_resolution_limit,
            )
            diffmap, map_dark, map_triggered = prepare_maps(
                unscaled_dark, unscaled_triggered, config
            )
            inclusion_mask = make_inclusion_mask(diffmap, map_dark, config)
            inclusion_masks.append(inclusion_mask)
        # inclusion_mask_avg = np.mean(inclusion_masks, axis=0)
        inclusion_mask_thresh = np.mean(inclusion_masks, axis=0) > 0.5
    else:
        print("Using individual inclusion masks for each config.")
        inclusion_mask_thresh = None
    return inclusion_mask_thresh


def make_all_plots(
    loading_function: callable, global_mask: bool, global_overrides: dict = {}
):
    inclusion_mask_thresh = create_global_inclusion_mask(
        loading_function, global_mask, global_overrides
    )

    def plot_all_configs(configs, override_dict={}, map_dark_base=None, verbose=False):
        """
        Plots all configs in a grid

        Also returns the figure and a list of interpolation tuples (mean, std) for each config
        """
        # suppress logging from xtr_estimator, verbose when processing many configs
        # logging.getLogger("xtr_estimator").setLevel(logging.ERROR)

        fig, axs = plt.subplots(5, 2, figsize=(12, 20), tight_layout=True)
        print(
            f"inclusion_mask_thresh {isinstance(inclusion_mask_thresh, np.ndarray)} and \nglobal_overrides {global_overrides} included!"
        )
        interpolation_tuples = []
        for i, (ax, config) in enumerate(zip(axs.flatten(), configs)):

            print(
                f"Processing config: {config['general']['name_human']}, \t({i+1}/{len(configs)})"
            )
            ax.set_title(config["general"]["name_human"])
            merged_config = merge_settings(config, global_overrides)
            merged_config = merge_settings(merged_config, override_dict)

            # this will suppress the output of the following function, which is quite verbose
            if verbose:
                _, _, interpolation_tuple, map_dark_base = xtr_logic(
                    merged_config,
                    ax=ax,
                    map_dark_base=map_dark_base,
                    prescribe_mask=inclusion_mask_thresh,
                )
            else:
                with redirect_stdout(None):
                    _, _, interpolation_tuple, map_dark_base = xtr_logic(
                        merged_config,
                        ax=ax,
                        map_dark_base=map_dark_base,
                        prescribe_mask=inclusion_mask_thresh,
                    )
            interpolation_tuples.append(interpolation_tuple)

        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,  # Adjust ncol to spread them horizontally
            bbox_to_anchor=(0.5, 1.03),
        )  # Moves it above the subplots

        plt.tight_layout()
        return fig, interpolation_tuples, map_dark_base

    return plot_all_configs


def compare(loading_function: callable, global_mask: bool, global_overrides: dict = {}):
    inclusion_mask_thresh = create_global_inclusion_mask(
        loading_function, global_mask, global_overrides
    )

    def plot_all_configs(configs, override_dict={}, map_dark_base=None, verbose=False):
        """
        Plots all configs in a grid

        Also returns the figure and a list of interpolation tuples (mean, std) for each config
        """
        # suppress logging from xtr_estimator, verbose when processing many configs
        # logging.getLogger("xtr_estimator").setLevel(logging.ERROR)

        fig, axs = plt.subplots(5, 2, figsize=(12, 20), tight_layout=True)
        print(
            f"inclusion_mask_thresh {isinstance(inclusion_mask_thresh, np.ndarray)} and \nglobal_overrides {global_overrides} included!"
        )
        interpolation_tuples = []
        for i, (ax, config) in enumerate(zip(axs.flatten(), configs)):

            print(
                f"Processing config: {config['general']['name_human']}, \t({i+1}/{len(configs)})"
            )
            ax.set_title(config["general"]["name_human"])
            merged_config = merge_settings(config, global_overrides)
            merged_config = merge_settings(merged_config, override_dict)

            # this will suppress the output of the following function, which is quite verbose
            if verbose:
                _, _, interpolation_tuple, map_dark_base = xtr_logic(
                    merged_config,
                    ax=ax,
                    map_dark_base=map_dark_base,
                    prescribe_mask=inclusion_mask_thresh,
                )
            else:
                with redirect_stdout(None):
                    _, _, interpolation_tuple, map_dark_base = xtr_logic(
                        merged_config,
                        ax=ax,
                        map_dark_base=map_dark_base,
                        prescribe_mask=inclusion_mask_thresh,
                    )
            interpolation_tuples.append(interpolation_tuple)

        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,  # Adjust ncol to spread them horizontally
            bbox_to_anchor=(0.5, 1.03),
        )  # Moves it above the subplots

        plt.tight_layout()
        return fig, interpolation_tuples, map_dark_base

    return plot_all_configs


def make_overview_plots(
    loading_function: callable, global_mask: bool, global_overrides: dict = {}
):
    inclusion_mask_thresh = create_global_inclusion_mask(
        loading_function, global_mask, global_overrides
    )

    def plot_overview_pl_configs(
        config, override_dict={}, ax=None, map_dark_base=None, verbose=False
    ):
        """
        Plots all configs in a grid

        Also returns the figure and a list of interpolation tuples (mean, std) for each config
        """
        # suppress logging from xtr_estimator, verbose when processing many configs
        if ax is None:
            raise ValueError("An axis must be provided for plotting.")

        ax.set_title(config["general"]["name_human"])
        merged_config = merge_settings(config, global_overrides)
        merged_config = merge_settings(merged_config, override_dict)

        # this will suppress the output of the following function, which is quite verbose
        if verbose:
            _, _, _, map_dark_base = xtr_logic(
                merged_config,
                ax=ax,
                map_dark_base=map_dark_base,
                prescribe_mask=inclusion_mask_thresh,
            )
        else:
            xtr_logger = logging.getLogger("xtr_estimator")
            original_level = xtr_logger.level
            xtr_logger.setLevel(logging.CRITICAL)
            # logging.disable(logging.CRITICAL)
            with redirect_stdout(None):
                _, _, _, map_dark_base = xtr_logic(
                    merged_config,
                    ax=ax,
                    map_dark_base=map_dark_base,
                    prescribe_mask=inclusion_mask_thresh,
                )
            xtr_logger.setLevel(original_level)
            # logging.disable(logging.NOTSET)

        return map_dark_base

    def overview_plots(xtr_setups, config_indices, map_dark_base=None, verbose=False):
        len_i = len(xtr_setups)
        len_j = len(config_indices)
        fig, axs = plt.subplots(
            len_i, len_j, figsize=(5 * len_j, 4 * len_i), tight_layout=True
        )
        for ii, setup in enumerate(xtr_setups):
            for jj, idx in enumerate(config_indices):
                print(f"Plotting config {idx} for setup {setup.prefix}")
                # Unpack everything directly from the helper method
                ax = axs.flat[ii * len_j + jj]
                map_dark_base = plot_overview_pl_configs(
                    *setup.get_plotting_data(idx),
                    map_dark_base=map_dark_base,
                    ax=ax,
                    verbose=verbose,
                )
                if jj:
                    ax.set_ylabel("")
                else:
                    ax.text(
                        -0.15,
                        0.5,
                        f"{setup.title}",
                        transform=ax.transAxes,  # Positions relative to the axes (0 to 1)
                        fontsize=20,  # Increased size
                        rotation=90,  # Rotates text to run parallel to the y-axis
                        va="center",  # Vertically centers the text at the 0.5 mark
                        ha="right",
                    )  # Aligns it properly outside the axis
                    ax.set_ylabel("Extrapolation Estimate", fontsize=10)
                if ii < len_i - 1:
                    ax.set_xlabel("")
                if ii:
                    ax.set_title("")
        return fig, map_dark_base

    return overview_plots


# def make_compare_diffmaps(
#     loading_function: callable, global_mask: bool, global_overrides: dict = {}
# ):


def compare_diffmaps(configs, override_dict_A={}, override_dict_B={}, verbose=False):
    """
    Plots all configs in a grid

    Also returns the figure and a list of interpolation tuples (mean, std) for each config
    """

    # suppress logging from xtr_estimator, verbose when processing many configs
    # logging.getLogger("xtr_estimator").setLevel(logging.ERROR)
    def calc_amplitude_similarity(amp_A, amp_B):
        return np.mean((amp_A + amp_B) ** 2 / 2 / (amp_A**2 + amp_B**2 + 1e-8))

    def calc_cosine_phase_similarity(phases_A, phases_B):
        rad1 = np.radians(phases_A)
        rad2 = np.radians(phases_B)

        # Calculate phase difference
        phase_diff = rad1 - rad2

        # Return cosine of the difference
        return np.mean(np.cos(phase_diff))

    def inner_action(config_A, config_B):
        unscaled_dark, unscaled_triggered = get_maps(
            config_A.input_files,
            high_resolution_limit=config_A.general.high_resolution_limit,
        )
        unscaled_dark, unscaled_triggered = get_maps(
            config_B.input_files,
            high_resolution_limit=config_B.general.high_resolution_limit,
        )

        diffmap_A, _, _ = prepare_maps(unscaled_dark, unscaled_triggered, config_A)
        diffmap_B, _, _ = prepare_maps(unscaled_dark, unscaled_triggered, config_B)

        amplitude_similarity = calc_amplitude_similarity(
            diffmap_A.amplitudes, diffmap_B.amplitudes
        )
        phase_similarity = calc_cosine_phase_similarity(
            diffmap_A.phases, diffmap_B.phases
        )
        return amplitude_similarity, phase_similarity

    out_tuple_list = []
    for i, (config) in enumerate(configs):

        print(
            f"Processing config: {config['general']['name_human']}, \t({i+1}/{len(configs)})"
        )
        merged_config_A = merge_settings(config, override_dict_A)
        merged_config_B = merge_settings(config, override_dict_B)
        # this will suppress the output of the following function, which is quite verbose
        if verbose:
            out_tuple = inner_action(merged_config_A, merged_config_B)
        else:
            with redirect_stdout(None):
                out_tuple = inner_action(merged_config_A, merged_config_B)
        out_tuple_list.append((merged_config_A, *out_tuple))
    return out_tuple_list


# def compare(loading_function: callable, global_mask: bool, global_overrides: dict = {}):
#     inclusion_mask_thresh = create_global_inclusion_mask(
#         loading_function, global_mask, global_overrides
#     )

#     def plot_all_configs(configs, override_dict={}, map_dark_base=None, verbose=False):
#         """
#         Plots all configs in a grid

#         Also returns the figure and a list of interpolation tuples (mean, std) for each config
#         """
#         # suppress logging from xtr_estimator, verbose when processing many configs
#         # logging.getLogger("xtr_estimator").setLevel(logging.ERROR)

#         fig, axs = plt.subplots(5, 2, figsize=(12, 20), tight_layout=True)
#         print(
#             f"inclusion_mask_thresh {isinstance(inclusion_mask_thresh, np.ndarray)} and \nglobal_overrides {global_overrides} included!"
#         )
#         interpolation_tuples = []
#         for i, (ax, config) in enumerate(zip(axs.flatten(), configs)):

#             print(
#                 f"Processing config: {config['general']['name_human']}, \t({i+1}/{len(configs)})"
#             )
#             ax.set_title(config["general"]["name_human"])
#             merged_config = merge_settings(config, global_overrides)
#             merged_config = merge_settings(merged_config, override_dict)

#             # this will suppress the output of the following function, which is quite verbose
#             if verbose:
#                 _, _, interpolation_tuple, map_dark_base = xtr_logic(
#                     merged_config,
#                     ax=ax,
#                     map_dark_base=map_dark_base,
#                     prescribe_mask=inclusion_mask_thresh,
#                 )
#             else:
#                 with redirect_stdout(None):
#                     _, _, interpolation_tuple, map_dark_base = xtr_logic(
#                         merged_config,
#                         ax=ax,
#                         map_dark_base=map_dark_base,
#                         prescribe_mask=inclusion_mask_thresh,
#                     )
#             interpolation_tuples.append(interpolation_tuple)

#         handles, labels = axs[0, 0].get_legend_handles_labels()
#         fig.legend(
#             handles,
#             labels,
#             loc="upper center",
#             ncol=3,  # Adjust ncol to spread them horizontally
#             bbox_to_anchor=(0.5, 1.03),
#         )  # Moves it above the subplots

#         plt.tight_layout()
#         return fig, interpolation_tuples, map_dark_base

#     return compare_diffmaps


from typing import NamedTuple, List, Any


class XtrSetup(NamedTuple):
    config_list: List[Any]
    overrides: dict
    xtr_means: List[Any]
    prefix: str
    title: str = ""

    def get_saving_data(self, idx: int):
        """Returns the specific config, mean, and formatted name for a given index."""
        cfg = self.config_list[idx]
        mean = self.xtr_means[idx]
        name = f"{self.prefix}_{cfg['general']['name_machine']}"
        return cfg, self.overrides, mean, name

    def get_plotting_data(self, idx: int):
        """Returns the specific config, mean, and formatted name for a given index."""
        cfg = self.config_list[idx]
        # mean = self.xtr_means[idx]
        # name = f"{self.prefix}_{cfg['general']['name_machine']}"
        return cfg, self.overrides
