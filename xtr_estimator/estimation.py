import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from meteor import rsmap

from .logger import setup_logger

logger = setup_logger()


def weighted_std(values: np.ndarray, weights: np.ndarray) -> tuple:
    """
    Calculate the weighted standard deviation.
    """
    weighted_mean = np.average(values, weights=weights)

    # This is the weighted average of the squared deviations from the weighted mean
    variance = np.average((values - weighted_mean) ** 2, weights=weights)

    return weighted_mean, np.sqrt(variance)


def _calculate_statistics(
    diffmap_np: np.ndarray,
    map_dark_np: np.ndarray,
    mask_np: np.ndarray,
) -> dict:
    """
    Extracts voxel values for masked and unmasked regions and calculates
    basic weights and divisions.
    """
    # Division (Occupancy Factor proxy)
    not_zero = map_dark_np != 0
    mask_inv = np.logical_and(~mask_np, not_zero)

    pseudo_occupancy = -diffmap_np[mask_np] / map_dark_np[mask_np]
    pseudo_occupancy_inv = -diffmap_np[mask_inv] / map_dark_np[mask_inv]

    weight = np.abs(diffmap_np[mask_np])

    sigma_level = np.sqrt(
        np.sum((diffmap_np - np.mean(diffmap_np)) ** 2) / diffmap_np.size
    )

    diffmap_sigma = (diffmap_np - np.mean(diffmap_np)) / sigma_level
    diffmap_sigma = (np.abs(diffmap_sigma)) * np.sign(diffmap_sigma)
    return {
        "diffmap_raw": diffmap_np[mask_np],
        "pseudo_occupancy": pseudo_occupancy,
        "weight": weight,
        "diffmap_sigma": diffmap_sigma[mask_np],
        "diffmap_inv": diffmap_sigma[mask_inv],
        "pseudo_occupancy_inv": pseudo_occupancy_inv,
    }


def _calculate_statistics2(
    diffmap_np: np.ndarray,
    map_dark_np: np.ndarray,
    mask_np: np.ndarray,
    solvent_vacuum: np.ndarray,
) -> dict:
    """
    Extracts voxel values for masked and unmasked regions and calculates
    basic weights and divisions.
    """
    sigma_level = np.sqrt(
        np.sum((diffmap_np - np.mean(diffmap_np)) ** 2) / diffmap_np.size
    )

    # Division (Occupancy Factor proxy)
    divisor = map_dark_np - solvent_vacuum
    legal_mask = np.logical_and.reduce(
        [divisor > 0, diffmap_np < 0, (-diffmap_np / divisor) < 2]
    )
    # diffmap_np = np.where(divisor <= 0, np.nan, diffmap_np)  # Avoid division by zero
    # divisor = np.where(divisor <= 0, np.nan, divisor)  # Avoid division by zero
    pseudo_occupancy = -diffmap_np[legal_mask] / divisor[legal_mask]

    mask_inv = np.logical_and(~mask_np, ~np.isnan(diffmap_np))
    mask_np = np.logical_and(mask_np, ~np.isnan(diffmap_np))
    mask_solvent = np.logical_and(mask_np, solvent_vacuum > 0)
    mask_vacuum = np.logical_and(mask_np, solvent_vacuum < 1e-4)

    diffmap_sigma = (diffmap_np - np.mean(diffmap_np)) / sigma_level
    diffmap_sigma = (np.abs(diffmap_sigma)) * np.sign(diffmap_sigma)
    print(diffmap_sigma[mask_np])
    print(np.unique(diffmap_sigma[mask_np]))
    print(np.unique(diffmap_sigma[mask_inv]))
    print(np.unique(diffmap_sigma[~mask_np]))
    return {
        "diffmap_raw": diffmap_np[legal_mask],
        "pseudo_occupancy": pseudo_occupancy,  # already filtered
        "diffmap_sigma": diffmap_sigma[legal_mask],
        "mask_solvent": mask_solvent[legal_mask],
        "mask_vacuum": mask_vacuum[legal_mask],
        "mask_inv": mask_inv[legal_mask],
        "mask_np": mask_np[legal_mask],
    }


def cummean_and_errors2(stats_data, leng_shown=None, number_sym_ops=1, plot_config={}):
    mask_np = stats_data["mask_np"]
    pseudo = stats_data["pseudo_occupancy"][mask_np]
    diff2 = -stats_data["diffmap_raw"][mask_np]
    diff_sigma = -stats_data["diffmap_sigma"][mask_np]
    leng_shown = len(diff2) if leng_shown is None else leng_shown
    argsorted = np.argsort(diff2)[::-number_sym_ops][:leng_shown]
    pseudo_sort = np.cumsum(pseudo[argsorted]) / (np.arange(len(argsorted)) + 1)
    # 1. Your existing sorted data
    sorted_vals = pseudo[argsorted]
    weights = 1  # diff2[argsorted]
    cum_mean = np.cumsum(sorted_vals) / (np.arange(len(sorted_vals)) + 1)
    cum_weighted = np.cumsum(sorted_vals * weights) / (np.cumsum(weights) + 1e-8)
    cum_mean_sq = np.cumsum(sorted_vals**2) / (np.arange(len(sorted_vals)) + 1)

    # 4. Cumulative standard deviation
    # We use np.maximum to avoid tiny negative numbers due to floating point error
    cum_std = np.sqrt(np.maximum(cum_mean_sq - cum_mean**2, 0))
    pseudo_std = cum_std
    pseudo_ste = pseudo_std / np.sqrt(np.arange(1, len(argsorted) + 1)) * 2
    bias_term = [
        np.abs(pseudo_sort[i] - pseudo_sort[i // 2]) for i in range(len(pseudo_sort))
    ]

    solvent_density = plot_config["solvent_density"]
    thresh_line = diff2[argsorted] / solvent_density

    return {
        "pseudo_sort": pseudo_sort,
        "pseudo_weighted": cum_weighted,
        "pseudo_ste": pseudo_ste,
        "bias_term": bias_term,
        "diff_sorted": diff2[argsorted],
        "diff_sigma": diff_sigma[argsorted],
        "thresh_line": thresh_line,
        "pseudo_std": pseudo_std,
        "number_sym_ops": number_sym_ops,
    }


def create_plot2(
    stats, cummean_dict, extra_info={}, ax=None, plot_config={}, return_both_ax=False
) -> tuple[Figure, Axes, tuple[float, float]]:
    std_cutoff = plot_config["std_cutoff"]
    markersize = plot_config["markersize"]
    thresh_line = cummean_dict["thresh_line"]
    average_distance_mask = (
        cummean_dict["pseudo_sort"] + std_cutoff * cummean_dict["pseudo_std"]
        > thresh_line
    )
    bottom_index = 0
    if np.any(average_distance_mask):
        bottom_index = np.where(average_distance_mask)[0][0]

    min_size_middle = 5 * cummean_dict["number_sym_ops"]

    marker = "."
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ########################## Data ###################################

    mask_solv = stats["mask_solvent"]
    mask_vac = stats["mask_vacuum"]
    mask_inv = stats["mask_inv"]
    diff_sigma = stats["diffmap_sigma"]
    xtr_fac = stats["pseudo_occupancy"]

    # plotting kwargs
    base_dot_kwargs = dict(marker=marker, linestyle="", alpha=0.5)
    vac_color = "royalblue"
    solv_color = "darkgoldenrod"
    comb_color = "forestgreen"
    vac_color = "royalblue"
    solv_color = "green"
    comb_color = "purple"
    inv_color = "gray"
    dot_kwargs_vac = base_dot_kwargs | dict(markersize=markersize, color=vac_color)
    dot_kwargs_solv = base_dot_kwargs | dict(markersize=markersize, color=solv_color)
    inv_dot_kwargs = dot_kwargs_vac | dict(markersize=markersize / 2, color=inv_color)
    dot_kwargs_solv["label"] = ("Solvent Voxels",)
    dot_kwargs_vac["label"] = ("Vacuum Voxels",)
    inv_dot_kwargs["label"] = ("Ignored Voxels",)

    ax.plot(diff_sigma[mask_inv], xtr_fac[mask_inv], **inv_dot_kwargs)
    ax.plot(diff_sigma[mask_vac], xtr_fac[mask_vac], **dot_kwargs_vac)
    ax.plot(diff_sigma[mask_solv], xtr_fac[mask_solv], **dot_kwargs_solv)
    ax.fill_between(
        -cummean_dict["diff_sigma"],
        cummean_dict["pseudo_sort"] - cummean_dict["pseudo_std"],
        cummean_dict["pseudo_sort"] + cummean_dict["pseudo_std"],
        color=inv_color,
        alpha=0.2,
    )
    ax.plot(
        -cummean_dict["diff_sigma"][: bottom_index + 1],
        cummean_dict["pseudo_sort"][: bottom_index + 1],
        color=comb_color,
        label="Cumulative Mean",
    )
    ax.plot(
        -cummean_dict["diff_sigma"][bottom_index:],
        cummean_dict["pseudo_sort"][bottom_index:],
        color=inv_color,
        # label="Cumulative Mean",
    )
    ########################## Thresholds ###################################
    pref = 1.1
    lowest_sigma = -cummean_dict["diff_sigma"][0] * pref
    lowest_thresh = thresh_line[0] * pref

    max_xtr = 11
    ax.fill_between(
        [lowest_sigma, 0],
        [lowest_thresh, 0],
        [max_xtr, max_xtr],
        color="red",
        alpha=0.2,
    )

    ax.plot(
        [lowest_thresh, 0],
        [lowest_sigma, 0],
        "--",
        color=inv_color,
        label="Reference Density Cutoff = Solvent",
    )
    if bottom_index > 0:
        ax.plot(
            [
                -cummean_dict["diff_sigma"][bottom_index],
            ]
            * 2,
            [0, thresh_line[bottom_index]],
            "r--",
            label="3x Std Dev. Cutoff",
        )
        ymax = (
            cummean_dict["pseudo_sort"] + std_cutoff * 2 * cummean_dict["pseudo_std"]
        )[bottom_index]
        ax.set_ylim(0.0, ymax)

    ############## Optimimum
    stats_kwarg = dict(
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    is_there_a_middle = bottom_index > 0 and min_size_middle < bottom_index

    chi = "$\\chi$"

    if is_there_a_middle:
        middle_diff = (
            cummean_dict["diff_sigma"][bottom_index] + cummean_dict["diff_sigma"][0]
        ) / 2
        print(f"Middle diff: {middle_diff}")
        print(f"Middle diff: {cummean_dict['diff_sigma']}")
        middle_index = np.where(middle_diff > cummean_dict["diff_sigma"])[0][0]
        middle_index = max(middle_index, min_size_middle)
        pseudo_range = cummean_dict["pseudo_sort"][: bottom_index + 1]
        middle_mean = cummean_dict["pseudo_sort"][middle_index]
        middle_diff = -cummean_dict["diff_sigma"][middle_index]
        middle_std_range = np.array([-1, 1]) * cummean_dict["pseudo_std"][middle_index]
        ax.plot(
            middle_diff * np.ones(2), middle_mean - middle_std_range, color=comb_color
        )
        optimal_kwargs = dict(
            s=200,
            facecolor="none",
            color="brown",
            label="Optimal",
        )
        ax.scatter(middle_diff, middle_mean, **optimal_kwargs)  # type: ignore

        prediction_tuple = middle_mean, cummean_dict["pseudo_std"][middle_index]
        plot_stats = dict(
            middle_mean=middle_mean,
            middle_std=cummean_dict["pseudo_std"][middle_index],
            middle_std_rel=cummean_dict["pseudo_std"][middle_index] / middle_mean,
            estimation_range=cummean_dict["diff_sigma"][0]
            / cummean_dict["diff_sigma"][bottom_index]
            - 1,
            variation_range=(np.max(pseudo_range) - np.min(pseudo_range)) / middle_mean,
        )
        text = ""
        text += f" {chi} =  {middle_mean:.3f}"
        text += f"\nSt. Dev.: {plot_stats['middle_std']:.3f} ({plot_stats['middle_std_rel']:.1%})"
        # if plot_config["comparison_to_reference"]:
        #     text += f"\nReference: {plot_config['comparison_to_reference']['value']:.3f} (1/{1/plot_config['comparison_to_reference']['value']:.1f})"
        if np.min(pseudo_range) < middle_mean - plot_stats["middle_std"] and False:
            text += "\nWarning: Min. est. less than\n1 std. dev. than reported est."
        if np.max(pseudo_range) > middle_mean + plot_stats["middle_std"] and False:
            text += "\nWarning: Max. est. more than\n1 std. dev. than reported est."
        skip_first_idcs = cummean_dict["number_sym_ops"]
        if (
            np.min(pseudo_range[skip_first_idcs:])
            / np.max(pseudo_range[skip_first_idcs:])
            < 2 / 3
        ):
            warning_text = (
                "Warning: Large variation in estimates\nCheck the plot for details."
            )
            text += f"\n{warning_text}"
            logger.warning(warning_text)
        if (
            cummean_dict["diff_sigma"][0] / cummean_dict["diff_sigma"][bottom_index]
            < 1.2
        ):

            warning_text = (
                "Warning: Very small estimation range\nCheck the plot for details."
            )
            text += f"\n{warning_text}"
            logger.warning(warning_text)
        # text += f"\nMin-Max Variation: {plot_stats['variation_range']:.1%}"
        # text += f"\n Estimation Range: {plot_stats['estimation_range']:.0%}"
        # place legend like box in top right corner
        ax.text(0.95, 0.95, text, **stats_kwarg)  # type: ignore

    else:
        text = "No Prediction because \n estimation range is too small"
        ax.text(0.95, 0.95, text, **stats_kwarg)  # type: ignore
        prediction_tuple = (np.nan, np.nan)

    # 4. Formatting
    if not plot_config["is_composite"]:
        ax.set_ylabel(f"Extrapolation factor  {chi} = " + r"$-\Delta\rho/\rho_{0}$")
        ax.set_xlabel("Difference Map " + r"$-\Delta \rho$ (standard deviations)")
        # ax.legend(loc="upper left")

        text_kwargs = dict(
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.4),
        )
        ax.text(0.25, 0.10, r"$\rho_0>\rho_{solvent}$", **text_kwargs)  # type: ignore
        ax.text(0.95, 0.65, r"$\rho_0<\rho_{solvent}$", **text_kwargs)  # type: ignore

    ax.grid()
    ax.set_xlim((lowest_sigma, 0.0))

    if True:
        ax2 = ax.twiny()
        ax2.plot(
            [-cummean_dict["diff_sorted"][0], 0],
            [thresh_line[0], 0],
            "--",
            color="gray",
        )
        ax2.set_xlim((-cummean_dict["diff_sorted"][0]) * 1.1, 0.0)
        ax2.set_xlabel(r"$-\Delta \rho$ (absolute units)")

    if plot_config["set_ylim"]:
        ax.set_ylim(*plot_config["set_ylim"])
    if return_both_ax:
        return fig, (ax, ax2), prediction_tuple
    return fig, ax, prediction_tuple


def cummean_and_errors(stats_data, leng_shown=None, number_sym_ops=1, plot_config={}):
    pseudo = stats_data["pseudo_occupancy"]
    diff2 = -stats_data["diffmap_raw"]
    diff_sigma = -stats_data["diffmap_sigma"]
    leng_shown = len(diff2) if leng_shown is None else leng_shown
    argsorted = np.argsort(diff2)[::-number_sym_ops][:leng_shown]
    pseudo_sort = np.cumsum(pseudo[argsorted]) / (np.arange(len(argsorted)) + 1)
    # 1. Your existing sorted data
    sorted_vals = pseudo[argsorted]
    weights = diff2[argsorted]
    cum_mean = np.cumsum(sorted_vals) / (np.arange(len(sorted_vals)) + 1)
    cum_weighted = np.cumsum(sorted_vals * weights) / (np.cumsum(weights) + 1e-8)
    cum_mean_sq = np.cumsum(sorted_vals**2) / (np.arange(len(sorted_vals)) + 1)

    # 4. Cumulative standard deviation
    # We use np.maximum to avoid tiny negative numbers due to floating point error
    cum_std = np.sqrt(np.maximum(cum_mean_sq - cum_mean**2, 0))
    pseudo_std = cum_std
    pseudo_ste = pseudo_std / np.sqrt(np.arange(1, len(argsorted) + 1)) * 2
    bias_term = [
        np.abs(pseudo_sort[i] - pseudo_sort[i // 2]) for i in range(len(pseudo_sort))
    ]

    solvent_density = plot_config["solvent_density"]
    thresh_line = diff2[argsorted] / solvent_density

    return {
        "pseudo_sort": pseudo_sort,
        "pseudo_weighted": cum_weighted,
        "pseudo_ste": pseudo_ste,
        "bias_term": bias_term,
        "diff_sorted": diff2[argsorted],
        "diff_sigma": diff_sigma[argsorted],
        "thresh_line": thresh_line,
        "pseudo_std": pseudo_std,
        "number_sym_ops": number_sym_ops,
    }


def compact_v3(cummean_dict, plot_config={}):
    std_cutoff = plot_config["std_cutoff"]
    thresh_line = cummean_dict["thresh_line"]
    average_distance_mask = (
        cummean_dict["pseudo_sort"] + std_cutoff * cummean_dict["pseudo_std"]
        > thresh_line
    )
    bottom_index = 0
    if np.any(average_distance_mask):
        bottom_index = np.where(average_distance_mask)[0][0]

    min_size_middle = 5 * cummean_dict["number_sym_ops"]
    is_there_a_middle = bottom_index > 0 and min_size_middle < bottom_index
    if is_there_a_middle:
        middle_diff = (
            cummean_dict["diff_sigma"][bottom_index] + cummean_dict["diff_sigma"][0]
        ) / 2
        middle_index = np.where(middle_diff > cummean_dict["diff_sigma"])[0][0]
        middle_index = max(middle_index, min_size_middle)
        middle_mean = cummean_dict["pseudo_sort"][middle_index]
        middle_diff = -cummean_dict["diff_sigma"][middle_index]
        return middle_mean, cummean_dict["pseudo_std"][middle_index]
    else:
        return np.nan, np.nan


def create_plot(
    stats, cummean_dict, extra_info={}, ax=None, plot_config={}, return_both_ax=False
) -> tuple[Figure, Axes, tuple[float, float]]:
    std_cutoff = plot_config["std_cutoff"]
    markersize = plot_config["markersize"]
    thresh_line = cummean_dict["thresh_line"]
    average_distance_mask = (
        cummean_dict["pseudo_sort"] + std_cutoff * cummean_dict["pseudo_std"]
        > thresh_line
    )
    bottom_index = 0
    if np.any(average_distance_mask):
        bottom_index = np.where(average_distance_mask)[0][0]

    min_size_middle = 5 * cummean_dict["number_sym_ops"]

    marker = "."
    if ax is None:
        fig, ax = plt.subplots(
        layout="constrained",
        squeeze=False,
        )
        ax = ax.flat[0]

    else:
        fig = ax.get_figure()
    ########################## Data ###################################
    base_dot_kwargs = dict(marker=marker, linestyle="", alpha=0.5)
    dot_kwargs = base_dot_kwargs | dict(markersize=markersize, color="blue")
    inv_dot_kwargs = dot_kwargs | dict(markersize=markersize / 2, color="gray")
    ax.plot(stats["diffmap_inv"], stats["pseudo_occupancy_inv"], **inv_dot_kwargs)
    ax.plot(stats["diffmap_sigma"], stats["pseudo_occupancy"], **dot_kwargs)
    del_items = ["markersize", "alpha"]
    dot_kwargs = {k:v for k, v in dot_kwargs.items() if k not in del_items}
    inv_dot_kwargs = {k:v for k, v in inv_dot_kwargs.items() if k not in del_items}
    ax.plot([], [], **dot_kwargs, label=r"Relevant Estimates $\chi$")
    ax.plot([], [], **inv_dot_kwargs, label="Discarded Estimates")
    ax.fill_between(
        -cummean_dict["diff_sigma"],
        cummean_dict["pseudo_sort"] - cummean_dict["pseudo_std"],
        cummean_dict["pseudo_sort"] + cummean_dict["pseudo_std"],
        color="grey",
        alpha=0.2,
    )
    ax.plot(
        -cummean_dict["diff_sigma"][: bottom_index + 1],
        cummean_dict["pseudo_sort"][: bottom_index + 1],
        color="blue",
        label="Cumulative Mean",
    )
    ax.plot(
        -cummean_dict["diff_sigma"][bottom_index:],
        cummean_dict["pseudo_sort"][bottom_index:],
        color="gray",
        # label="Cumulative Mean",
    )
    ########################## Thresholds ###################################
    pref = 1.1
    lowest_sigma = -cummean_dict["diff_sigma"][0] * pref
    lowest_thresh = thresh_line[0] * pref

    max_xtr = 11
    ax.fill_between(
        [lowest_sigma, 0],
        [lowest_thresh, 0],
        [max_xtr, max_xtr],
        color="red",
        alpha=0.2,
    )


    ############## Optimimum
    stats_kwarg = dict(
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    is_there_a_middle = bottom_index > 0 and min_size_middle < bottom_index

    chi = "$\\chi$"

    if is_there_a_middle:
        middle_diff = (
            cummean_dict["diff_sigma"][bottom_index] + cummean_dict["diff_sigma"][0]
        ) / 2
        middle_index = np.where(middle_diff > cummean_dict["diff_sigma"])[0][0]
        middle_index = max(middle_index, min_size_middle)
        pseudo_range = cummean_dict["pseudo_sort"][: bottom_index + 1]
        middle_mean = cummean_dict["pseudo_sort"][middle_index]
        middle_diff = -cummean_dict["diff_sigma"][middle_index]
        middle_std_range = np.array([-1, 1]) * cummean_dict["pseudo_std"][middle_index]
        ax.plot(middle_diff * np.ones(2), middle_mean - middle_std_range, color="blue")
        optimal_kwargs = dict(
            s=200,
            facecolor="none",
            color="brown",
            label="Optimal",
        )
        ax.scatter(middle_diff, middle_mean, **optimal_kwargs)  # type: ignore

        prediction_tuple = middle_mean, cummean_dict["pseudo_std"][middle_index]
        plot_stats = dict(
            middle_mean=middle_mean,
            middle_std=cummean_dict["pseudo_std"][middle_index],
            middle_std_rel=cummean_dict["pseudo_std"][middle_index] / middle_mean,
            estimation_range=cummean_dict["diff_sigma"][0]
            / cummean_dict["diff_sigma"][bottom_index]
            - 1,
            variation_range=(np.max(pseudo_range) - np.min(pseudo_range)) / middle_mean,
        )
        text = ""
        text += f" {chi} =  {middle_mean:.3f}"
        text += f"\nSt. Dev.: {plot_stats['middle_std']:.3f} ({plot_stats['middle_std_rel']:.1%})"
        # if plot_config["comparison_to_reference"]:
        #     text += f"\nReference: {plot_config['comparison_to_reference']['value']:.3f} (1/{1/plot_config['comparison_to_reference']['value']:.1f})"
        if np.min(pseudo_range) < middle_mean - plot_stats["middle_std"] and False:
            text += "\nWarning: Min. est. less than\n1 std. dev. than reported est."
        if np.max(pseudo_range) > middle_mean + plot_stats["middle_std"] and False:
            text += "\nWarning: Max. est. more than\n1 std. dev. than reported est."
        skip_first_idcs = cummean_dict["number_sym_ops"]
        if (
            np.min(pseudo_range[skip_first_idcs:])
            / np.max(pseudo_range[skip_first_idcs:])
            < 2 / 3
        ):
            warning_text = (
                "Warning: Large variation in estimates\nCheck the plot for details."
            )
            text += f"\n{warning_text}"
            logger.warning(warning_text)
        if (
            cummean_dict["diff_sigma"][0] / cummean_dict["diff_sigma"][bottom_index]
            < 1.2
        ):

            warning_text = (
                "Warning: Very small estimation range\nCheck the plot for details."
            )
            text += f"\n{warning_text}"
            logger.warning(warning_text)
        # text += f"\nMin-Max Variation: {plot_stats['variation_range']:.1%}"
        # text += f"\n Estimation Range: {plot_stats['estimation_range']:.0%}"
        # place legend like box in top right corner
        ax.text(0.95, 0.95, text, **stats_kwarg)  # type: ignore

    else:
        text = "No Prediction because \n estimation range is too small"
        ax.text(0.95, 0.95, text, **stats_kwarg)  # type: ignore
        prediction_tuple = (np.nan, np.nan)

    # 4. Formatting
    if not plot_config["is_composite"]:
        ax.set_ylabel(f"Extrapolation factor  {chi} = " + r"$-\Delta\rho/\rho_{0}$")
        ax.set_xlabel("Difference Map " + r"$-\Delta \rho$ (standard deviations)")
        # ax.legend(loc="upper left")

        text_kwargs = dict(
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.4),
        )
        ax.text(0.25, 0.10, r"$\rho_0>\rho_{solvent}$", **text_kwargs)  # type: ignore
        ax.text(0.95, 0.65, r"$\rho_0<\rho_{solvent}$", **text_kwargs)  # type: ignore

    ax.grid()
    ax.set_xlim((lowest_sigma, 0.0))

    ax.plot(
        [lowest_thresh, 0],
        [lowest_sigma, 0],
        "--",
        color="gray",
        label="Reference Density Cutoff",
    )

    if bottom_index > 0:
        # line_x = [-cummean_dict["diff_sigma"][bottom_index]] * 2
        # line_y = [0, thresh_line[bottom_index]]
        ax.axvline(thresh_line[bottom_index], linestyle="--", color="red", label="3x Std Dev. Cutoff")
        ymax = (
            cummean_dict["pseudo_sort"] + std_cutoff * 2 * cummean_dict["pseudo_std"]
        )[bottom_index]
        ax.set_ylim(0.0, ymax)

    if True:
        ax2 = ax.twiny()
        ax2.plot(
            [-cummean_dict["diff_sorted"][0], 0],
            [thresh_line[0], 0],
            "--",
            color="gray",
        )
        ax2.set_xlim((-cummean_dict["diff_sorted"][0]) * 1.1, 0.0)
        ax2.set_xlabel(r"$-\Delta \rho$ (absolute units)")

    if plot_config["set_ylim"]:
        ax.set_ylim(*plot_config["set_ylim"])

    if plot_config["legend"]:
        fig.legend(loc="outside lower center", ncol=3)
    if return_both_ax:
        return fig, (ax, ax2), prediction_tuple
    return fig, ax, prediction_tuple


def plot_extrapolation_estimate(
    diffmap: rsmap.Map,
    map_dark: rsmap.Map,
    inclusion_mask: np.ndarray,
    config: dict,
    ax: Axes | None = None,
    compact: bool = False,
) -> tuple[Figure | None, Axes | None, tuple[float, float]]:
    general_config = config["general"]

    diffmap_np = diffmap.to_3d_numpy_map(map_sampling=general_config["map_sampling"])
    map_dark_np = map_dark.to_3d_numpy_map(map_sampling=general_config["map_sampling"])
    logger.warning(
        f"Mean of diffmap_np: {np.mean(diffmap_np)}, Mean of map_dark_np: {np.mean(map_dark_np)}"
    )
    stats_data = _calculate_statistics(diffmap_np, map_dark_np, inclusion_mask)
    number_sym_ops = 1
    cummean_dict = cummean_and_errors(
        stats_data, number_sym_ops=number_sym_ops, plot_config=config["plot"]
    )
    number_sym_ops = len(map_dark.spacegroup.operations())
    cummean_dict["number_sym_ops"] = number_sym_ops
    # trend_data = _analyze_threshold_trends(
    #     stats_data["diffmap_masked"], #     stats_data["pseudo_occupancy"],
    #     stats_data["weight"],
    # )
    # 5. Visualization
    if compact:
        return None, None, compact_v3(cummean_dict, plot_config=config["plot"])
    return create_plot(stats_data, cummean_dict, plot_config=config["plot"], ax=ax)


def plot_extrapolation_estimate_binary(
    diffmap: rsmap.Map,
    map_dark: rsmap.Map,
    inclusion_mask: np.ndarray,
    rho_floor: dict,
    config: dict,
    ax: Axes | None = None,
    compact: bool = False,
) -> tuple[Figure | None, Axes | None, tuple[float, float]]:
    general_config = config["general"]

    diffmap_np = diffmap.to_3d_numpy_map(map_sampling=general_config["map_sampling"])
    map_dark_np = map_dark.to_3d_numpy_map(map_sampling=general_config["map_sampling"])
    stats_data = _calculate_statistics2(
        diffmap_np, map_dark_np, inclusion_mask, solvent_vacuum=rho_floor
    )
    number_sym_ops = 1
    cummean_dict = cummean_and_errors2(
        stats_data, number_sym_ops=number_sym_ops, plot_config=config["plot"]
    )
    number_sym_ops = len(map_dark.spacegroup.operations())
    cummean_dict["number_sym_ops"] = number_sym_ops
    # trend_data = _analyze_threshold_trends(
    #     stats_data["diffmap_masked"], #     stats_data["pseudo_occupancy"],
    #     stats_data["weight"],
    # )
    # 5. Visualization
    if compact:
        return None, None, compact_v3(cummean_dict, plot_config=config["plot"])
    return create_plot2(stats_data, cummean_dict, plot_config=config["plot"], ax=ax)
