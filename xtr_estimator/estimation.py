import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from meteor import rsmap
from .configuration import Settings, PlotSettings

from .logger import setup_logger
from .utils import map_to_array

logger = setup_logger()

CHI = r"$\chi$"


# =========================================================================== #
# statistics  (single extractor; legal_mask identical in both modes)
# =========================================================================== #
def _legal_mask(diffmap_np: np.ndarray, divisor: np.ndarray) -> np.ndarray:
    """Voxels we are allowed to divide by. Also keeps the returned arrays small.

    errstate suppresses the divide/invalid warnings from the full-array ratio;
    the result is only *used* where divisor > 0 anyway.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = -diffmap_np / divisor
    return (divisor > 0) & (diffmap_np < 0) & (ratio < 2)


def _sigma_scale(arr: np.ndarray) -> np.ndarray:
    """Z-score against the *whole* map (mean/std over every voxel)."""
    sigma = np.sqrt(np.sum((arr - arr.mean()) ** 2) / arr.size)
    return (arr - arr.mean()) / sigma  # old abs()*sign() was a no-op


def _calculate_statistics(
    diffmap_np: np.ndarray,
    map_dark_np: np.ndarray,
    inclusion_mask: np.ndarray,
    *,
    solvent_vacuum: np.ndarray | None = None,
) -> dict:
    """One extractor for both modes. Pass solvent_vacuum for the binary mode.

    All returned arrays are length == #legal voxels; all masks are booleans
    over that same legal subset, so they index each other freely.
    """
    divisor = map_dark_np if solvent_vacuum is None else map_dark_np - solvent_vacuum
    legal = _legal_mask(diffmap_np, divisor)

    incl = inclusion_mask[legal]
    out = {
        "diffmap_raw": diffmap_np[legal],
        "diffmap_sigma": _sigma_scale(diffmap_np)[legal],
        "pseudo_occupancy": -diffmap_np[legal] / divisor[legal],
        "mask_relevant": incl,  # voxels the cumulative mean is built from
        "mask_inv": ~incl,  # discarded / ignored
    }
    if solvent_vacuum is not None:
        out["mask_solvent"] = (inclusion_mask & (solvent_vacuum > 0))[legal]
        out["mask_vacuum"] = (inclusion_mask & (solvent_vacuum < 1e-4))[legal]
    return out


# =========================================================================== #
# cumulative mean  (mode-agnostic; no number_sym_ops, no weights)
# =========================================================================== #
def cummean_and_errors(
    pseudo: np.ndarray,
    diff_raw: np.ndarray,
    diff_sigma: np.ndarray,
    *,
    solvent_density: float,
    leng_shown: int | None = None,
) -> dict:
    diff2 = -diff_raw
    diff_sigma = -diff_sigma
    n = len(diff2) if leng_shown is None else leng_shown
    order = np.argsort(diff2)[::-1][:n]

    vals = pseudo[order]
    k = np.arange(len(order)) + 1
    cum_mean = np.cumsum(vals) / k
    cum_mean_sq = np.cumsum(vals**2) / k
    cum_std = np.sqrt(np.maximum(cum_mean_sq - cum_mean**2, 0))

    return {
        "pseudo_sort": cum_mean,
        "pseudo_std": cum_std,
        "diff_sorted": diff2[order],
        "diff_sigma": diff_sigma[order],
        "thresh_line": diff2[order] / solvent_density,
    }


# =========================================================================== #
# shared index / prediction logic  (used by both compact and the plot)
# =========================================================================== #
@dataclass
class Indices:
    bottom: int
    min_middle: int

    @property
    def has_middle(self) -> bool:
        return self.bottom > 0 and self.min_middle < self.bottom


def locate_indices(cummean: dict, std_cutoff: float) -> Indices:
    mask = (
        cummean["pseudo_sort"] + std_cutoff * cummean["pseudo_std"]
        > cummean["thresh_line"]
    )
    bottom = int(np.where(mask)[0][0]) if np.any(mask) else 0
    return Indices(bottom=bottom, min_middle=5 * cummean["number_sym_ops"])


def _middle_index(cummean: dict, idx: Indices) -> int:
    diff = cummean["diff_sigma"]
    middle_diff = (diff[idx.bottom] + diff[0]) / 2
    mi = int(np.where(middle_diff > diff)[0][0])
    return max(mi, idx.min_middle)


def locate_prediction(cummean: dict, std_cutoff: float) -> dict :
    """Replaces compact_v3 — same math the plot reports, so they can't diverge."""
    idx = locate_indices(cummean, std_cutoff)
    if not idx.has_middle:
        return dict(
            estimate=np.nan,
            std=np.nan,
            relative_std=np.nan,
            sigma_at_estimate=np.nan,
            ratio_estimation_range=np.nan,
            variation_range=np.nan,
        )
    mi = _middle_index(cummean, idx)

    diff = cummean["diff_sigma"]
    middle_x = -diff[mi]
    std = cummean["pseudo_std"]
    ratio_estimation_range = diff[0] / diff[b]
    skip = cummean["number_sym_ops"]
    b = idx.bottom
    y = cummean["pseudo_sort"]
    pseudo_range = y[: b + 1]
    variation_range = np.min(pseudo_range[skip:]) / np.max(pseudo_range[skip:])
    middle_mean = y[mi]
    return dict(
        estimate=middle_mean,
        std=std[mi],
        relative_std=std[mi] / middle_mean,
        sigma_at_estimate=middle_x,
        ratio_estimation_range=ratio_estimation_range,
        variation_range=variation_range,
    )



# =========================================================================== #
# mode object  (the only place the simple/binary distinction survives)
# =========================================================================== #
@dataclass(frozen=True)
class ColorScheme:
    points: str  # main scatter (simple mode)
    vacuum: str
    solvent: str
    cumulative: str  # cumulative-mean line + optimum std-bar
    ignored: str
    optimal: str = "brown"
    cutoff: str = "red"


@dataclass
class PointGroup:
    x: np.ndarray
    y: np.ndarray
    color: str
    size: float
    label: str


@dataclass(frozen=True)
class Mode:
    name: str
    colors: ColorScheme
    make_point_groups: Callable[[dict, ColorScheme, float], list[PointGroup]]
    needs_solvent: bool = False


def _groups_simple(stats, c, sizes) -> list[PointGroup]:
    x, y = stats["diffmap_sigma"], stats["pseudo_occupancy"]
    return [
        PointGroup(
            x[stats["mask_inv"]],
            y[stats["mask_inv"]],
            c.ignored,
            sizes["ignored_markersize"],
            "Discarded Estimates",
        ),
        PointGroup(
            x[stats["mask_relevant"]],
            y[stats["mask_relevant"]],
            c.points,
            sizes["markersize"],
            r"Relevant Estimates $\chi$",
        ),
    ]


def _groups_binary(stats, c, sizes) -> list[PointGroup]:
    x, y = stats["diffmap_sigma"], stats["pseudo_occupancy"]
    return [
        PointGroup(
            x[stats["mask_inv"]],
            y[stats["mask_inv"]],
            c.ignored,
            sizes["ignored_markersize"],
            "Ignored Voxels",
        ),
        PointGroup(
            x[stats["mask_vacuum"]],
            y[stats["mask_vacuum"]],
            c.vacuum,
            sizes["markersize"],
            "Vacuum Voxels",
        ),
        PointGroup(
            x[stats["mask_solvent"]],
            y[stats["mask_solvent"]],
            c.solvent,
            sizes["markersize"],
            "Solvent Voxels",
        ),
    ]


SIMPLE = Mode(
    name="simple",
    colors=ColorScheme(
        points="blue",
        vacuum="royalblue",
        solvent="green",
        cumulative="blue",
        ignored="gray",
    ),
    make_point_groups=_groups_simple,
)

BINARY = Mode(
    name="binary",
    colors=ColorScheme(
        points="blue",
        vacuum="royalblue",
        solvent="green",
        cumulative="purple",
        ignored="gray",
    ),
    make_point_groups=_groups_binary,
    needs_solvent=True,
)


# =========================================================================== #
# drawing primitives  (all mode-agnostic — colors come in via ColorScheme)
# =========================================================================== #
@dataclass
class Bounds:
    lowest_sigma: float
    lowest_thresh: float
    lowest_abs: float
    max_xtr: float = 11.0


def compute_bounds(cummean: dict, pref: float = 1.1) -> Bounds:
    return Bounds(
        lowest_sigma=-cummean["diff_sigma"][0] * pref,
        lowest_thresh=cummean["thresh_line"][0] * pref,
        lowest_abs=-cummean["diff_sorted"][0] * pref,
    )


def ensure_axes(ax, figsize):
    """Constrained layout is required for loc='outside ...' legends."""
    if ax is None:
        fig, axarr = plt.subplots(figsize=figsize, layout="constrained", squeeze=False)
        return fig, axarr.flat[0]
    return ax.get_figure(), ax


def _stats_box(ax, aesthetics):
    sizes = aesthetics.resolved()
    return dict(
        transform=ax.transAxes,
        fontsize=sizes["annotation"],  # was aesthetics.font.annotation -> 1.0
        verticalalignment="top",
        horizontalalignment="right",
        ma="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )


def plot_points(ax, groups: list[PointGroup], alpha: float = 0.5):
    """Real points unlabeled; one opaque, full-size proxy per group for the
    legend (uniform across modes)."""
    general_kwargs = dict(
        marker=".",
        linestyle="",
    )
    for g in groups:
        point_kwargs = dict(markersize=g.size, color=g.color, alpha=alpha)
        ax.plot(g.x, g.y, **general_kwargs, **point_kwargs)
        ax.plot([], [], color=g.color, label=g.label, **general_kwargs)


def plot_cumulative_mean(ax, cummean, idx, colors, aesthetics):
    sizes = aesthetics.resolved()
    x = -cummean["diff_sigma"]
    y = cummean["pseudo_sort"]
    std = cummean["pseudo_std"]
    b = idx.bottom
    ax.fill_between(
        x, y - std, y + std, color=colors.ignored, alpha=aesthetics.fill_alpha
    )
    ax.plot(
        x[: b + 1],
        y[: b + 1],
        color=colors.cumulative,
        linewidth=sizes["linewidth"],
        label="Cumulative Mean",
    )
    ax.plot(x[b:], y[b:], color=colors.ignored, linewidth=sizes["linewidth"])


def plot_thresholds(ax, bounds, colors, aesthetics):
    # red region: lower edge is the diagonal (lowest_sigma, lowest_thresh)->(0,0)
    # logger.info("PLOT THRESHOLDS")
    ax.fill_between(
        [bounds.lowest_sigma, 0],
        [bounds.lowest_thresh, 0],
        [bounds.max_xtr, bounds.max_xtr],
        color=colors.cutoff,
        alpha=aesthetics.fill_alpha,
    )
    # reference cutoff = that SAME diagonal edge (x/y no longer swapped)
    ax.plot(
        [bounds.lowest_sigma, 0],
        [bounds.lowest_thresh, 0],
        # linestyle=(0, (6, 4)),
        linestyle="--",
        linewidth=aesthetics.resolved()["linewidth"],
        color=colors.ignored,
        label="Reference density cutoff",
    )


def annotate_optimum(
    ax, cummean, idx, colors, aesthetics, owns_figure=False
) -> dict:
    box_kwargs = _stats_box(ax, aesthetics)

    if not idx.has_middle:
        error_msg = "     No Prediction \n"
        error_msg += "estimation range \n"
        error_msg += "      too small"
        ax.text(0.95, 0.95, error_msg, **box_kwargs)
        return dict(
            estimate=np.nan,
            std=np.nan,
            relative_std=np.nan,
            sigma_at_estimate=np.nan,
            ratio_estimation_range=np.nan,
            variation_range=np.nan,
        )

    diff = cummean["diff_sigma"]
    y = cummean["pseudo_sort"]
    std = cummean["pseudo_std"]
    b = idx.bottom
    mi = _middle_index(cummean, idx)

    pseudo_range = y[: b + 1]
    middle_mean = y[mi]
    middle_x = -diff[mi]
    std_bar = np.array([-1, 1]) * std[mi]

    sizes = aesthetics.resolved()
    pos_x = middle_x * np.ones(2)
    pos_y = middle_mean - std_bar
    ax.plot(pos_x, pos_y, color=colors.cumulative, linewidth=sizes["linewidth"])

    circ_args = dict(facecolor="none", color=colors.optimal)
    ax.scatter(middle_x, middle_mean, s=sizes["optimal_markerarea"], **circ_args)
    ax.scatter([], [], label="Optimal", **circ_args)

    text = rf" $\chi\, = {middle_mean:.3f}$"
    sig_chi = r"$\sigma_\chi$"
    text += "\n" + rf"$\sigma_\chi = {std[mi]:.3f}$ ({std[mi] / middle_mean:.1%})"
    skip = cummean["number_sym_ops"]
    variation_range = np.min(pseudo_range[skip:]) / np.max(pseudo_range[skip:])
    if variation_range < 2 / 3:
        msg = "Warning: Large variation in estimates\nCheck the plot for details."
        logger.warning(msg)
        msg_short = "Large variation \n"
        msg_short += "   in estimates"
        text += f"\n{msg_short}"  # if owns_figure else ""
    ratio_estimation_range = diff[0] / diff[b]
    if ratio_estimation_range < 1.2:
        msg = "Warning: Very small estimation range\nCheck the plot for details."
        logger.warning(msg)
        msg_short = "      Very small \n"
        msg_short += "estimation range"
        text += f"\n{msg_short}"  # if owns_figure else ""
    ax.text(0.95, 0.95, text, **box_kwargs)
    return dict(
        estimate=middle_mean,
        std=std[mi],
        relative_std=std[mi] / middle_mean,
        sigma_at_estimate=middle_x,
        ratio_estimation_range=ratio_estimation_range,
        variation_range=variation_range,
    )


def format_axes(ax, aesthetics, show_rho):
    sizes = aesthetics.resolved()
    ax.set_ylabel(f"Extrapolation factor  {CHI}", fontsize=sizes["label"])
    ax.set_xlabel(r"Difference Map $\Delta \rho$ [RMSD]", fontsize=sizes["label"])
    tk = dict(
        transform=ax.transAxes,
        fontsize=sizes["rhos"],
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=aesthetics.anno_alpha),
    )
    if show_rho:
        0.25, 0.10
        0.95, 0.65
        x, y = aesthetics.solvent.large
        # aesthetics.rho_solvent.large
        ax.text(x, y, r"$\rho_0>\rho_{solvent}$", **tk)
        x, y = aesthetics.solvent.small
        ax.text(x, y, r"$\rho_0<\rho_{solvent}$", **tk)
    ax.tick_params(labelsize=sizes["tick"])
    ax.grid(linewidth=sizes["grid_linewidth"], alpha=aesthetics.grid_alpha)


def add_secondary_axis(ax, bounds, aesthetics):
    sizes = aesthetics.resolved()
    ax2 = ax.twiny()
    ax2.plot([], [], "--", color="gray", linewidth=sizes["linewidth"])
    ax2.set_xlim(bounds.lowest_abs, 0.0)
    ax2.set_xlabel(r"$\Delta \rho$ [e$^-$/Å$^3$]", fontsize=sizes["label"])
    ax2.tick_params(labelsize=sizes["tick"])
    return ax2


def apply_limits(ax, cummean, idx, bounds, plot_config):
    ax.set_xlim((bounds.lowest_sigma, 0.0))
    if idx.bottom > 0:
        ymax = (
            cummean["pseudo_sort"]
            + plot_config["std_cutoff"] * 2 * cummean["pseudo_std"]
        )[idx.bottom]
        ax.set_ylim(0.0, ymax)
    if plot_config["set_ylim"]:
        ax.set_ylim(*plot_config["set_ylim"])  # explicit override wins, applied last


# =========================================================================== #
# orchestrator
# =========================================================================== #
def create_plot(
    stats, cummean, mode: Mode, ax=None, plot_config={}, return_both_ax=False
) -> tuple[Figure, Axes, dict]:
    idx = locate_indices(cummean, plot_config["std_cutoff"])
    bounds = compute_bounds(cummean)

    owns_figure = ax is None  # A) single plot -> we draw the legend

    aesthetics = plot_config.aesthetics
    sizes = aesthetics.resolved()
    fig, ax = ensure_axes(ax, plot_config.aesthetics.figure_size(1))

    plot_points(
        ax,
        mode.make_point_groups(stats, mode.colors, sizes),
        alpha=aesthetics.point_alpha,
    )
    plot_cumulative_mean(ax, cummean, idx, mode.colors, aesthetics)
    plot_thresholds(ax, bounds, mode.colors, aesthetics)
    prediction = annotate_optimum(
        ax, cummean, idx, mode.colors, aesthetics, owns_figure=owns_figure
    )
    # print(aesthetics)
    format_axes(ax, aesthetics, show_rho=plot_config["show_rho"])
    ax2 = add_secondary_axis(ax, bounds, aesthetics)

    # 3x std-dev cutoff. Both land at the same screen-x; pick the axis to draw on.
    if idx.bottom > 0:
        # Option A — on the sigma (bottom) axis:
        ax.axvline(
            -cummean["diff_sigma"][idx.bottom],
            linestyle="--",
            linewidth=sizes["linewidth"],
            color=mode.colors.cutoff,
            label="3x Std Dev. Cutoff",
        )
        # Option B — same position, on the absolute-unit (top) axis:
        # ax2.axvline(-cummean["diff_sorted"][idx.bottom], linestyle="--",
        #             color=mode.colors.cutoff, label="3x Std Dev. Cutoff")

    apply_limits(ax, cummean, idx, bounds, plot_config)

    if owns_figure:  # B) caller passed ax -> caller owns legend
        fig.legend(
            loc="outside lower center",
            ncol=aesthetics.legend_ncol,
            fontsize=sizes["legend"],
            frameon=False,
        )
    print(prediction)
    if return_both_ax:
        return fig, (ax, ax2), prediction
    return fig, ax, prediction


# =========================================================================== #
# entry point  (one function; pick the mode)
# =========================================================================== #
# Furthermore, I only want to pass plottingConfig into plot_extrapolation_estimate, so i will move map_sampling as a quantity into plotting_config.
def plot_extrapolation_estimate(
    diffmap: rsmap.Map,
    map_dark: rsmap.Map,
    inclusion_mask: np.ndarray,
    config: dict,
    mode: Mode = SIMPLE,
    rho_floor: np.ndarray | None = None,
    ax: Axes | None = None,
    compact: bool = False,
    return_both_ax: bool = False,
) -> tuple[Figure | None, Axes | None, dict]:
    sampling = config["general"]["map_sampling"]
    diffmap_np = diffmap.to_3d_numpy_map(map_sampling=sampling)
    map_dark_np = map_to_array(map_dark, diffmap_np.shape)

    stats = _calculate_statistics(
        diffmap_np,
        map_dark_np,
        inclusion_mask,
        solvent_vacuum=rho_floor if mode.needs_solvent else None,
    )

    rel = stats["mask_relevant"]
    cummean = cummean_and_errors(
        stats["pseudo_occupancy"][rel],
        stats["diffmap_raw"][rel],
        stats["diffmap_sigma"][rel],
        solvent_density=config["plot"]["solvent_density"],
    )
    cummean["number_sym_ops"] = len(map_dark.spacegroup.operations())

    if compact:
        return None, None, locate_prediction(cummean, config["plot"]["std_cutoff"])
    return create_plot(
        stats,
        cummean,
        mode=mode,
        plot_config=config["plot"],
        ax=ax,
        return_both_ax=return_both_ax,
    )


def outside_legend(fig, axes, ncol=3, fontsize=None):
    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    by_label = dict(zip(labels, handles))  # last-wins dedupe
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="outside lower center",
        ncol=ncol,
        fontsize=fontsize,
        frameon=False,
    )


def decorate_cell(
    ax,
    cell,
    i,
    j,
    nrows,
    ncols,
    sizes,
    *,
    edge_labels_only=True,
    ax2=None,
    sharey=True,
    fig=None,
):
    is_top = i == 0
    is_bottom = i == nrows - 1
    is_left = j == 0
    if edge_labels_only:
        if not is_bottom:  # bottom axis only on last row
            ax.set_xlabel("")
            # ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel(r"$\Delta \rho$ [RMSD]", fontsize=sizes["label"])
        if ax2 is not None and not is_top:  # top (secondary) axis only on first row
            ax2.set_xlabel("")
            # ax2.tick_params(axis="x", labeltop=False)
            # if you want it gone entirely: ax2.axis("off")
        if not is_left:  # y label only on first column
            ax.set_ylabel("")
            if sharey:
                ax.tick_params(axis="y", labelleft=False, left=False)
        else:
            ax.set_ylabel(
                f"Extrapolation factor  {CHI}",
                fontsize=sizes["label"],
            )
    if cell.title and is_top:
        ax.set_title(
            cell.title, fontsize=sizes["outside_title"], pad=sizes["tick"] * 1.5
        )
    elif not is_top:
        ax.set_title("")
    if cell.row_label and is_left:
        kwargs = dict(
            rotation=90, va="center", ha="right", fontsize=sizes["outside_title"]
        )
        # if fig is None:
        t = ax.text(-0.4, 0.5, cell.row_label, transform=ax.transAxes, **kwargs)
        t.set_in_layout(True)
        ax.set_ylabel(r"Extr. factor $\chi$")
        # else:
        #     y = (ax.get_position().y0 + ax.get_position().y1) * 0.6
        #     fig.text(0.00, y, cell.row_label, **kwargs)


@dataclass
class PlottingUnit:
    diffmap: rsmap.Map
    map_dark: rsmap.Map
    inclusion_mask: np.ndarray
    config: dict | Settings
    title: str | None = None
    row_label: str | None = None

    def draw(self, ax):
        return plot_extrapolation_estimate(
            self.diffmap,
            self.map_dark,
            self.inclusion_mask,
            self.config,
            ax=ax,
            return_both_ax=True,
        )


def composite_grid(
    plotting_unit,
    nrows,
    ncols,
    config_plot,
    *,
    sharey=True,
    sharex=False,
    edge_labels_only=True,
    legend=True,
):
    aesthetics = config_plot.aesthetics
    sizes = aesthetics.resolved()
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=aesthetics.figure_size(panels_per_row=ncols, n_rows=nrows),
        sharey=sharey,
        sharex=sharex,
        layout="constrained",
        squeeze=False,
    )
    results = []
    for k, cell in enumerate(plotting_unit):
        i, j = divmod(k, ncols)
        ax = axs[i, j]
        cell.config.plot = config_plot  # inject the aesthetics into the cell's config for use in plotting
        fig, (ax, ax2), result = cell.draw(ax)
        results.append(result)
        decorate_cell(
            ax,
            cell,
            i,
            j,
            nrows,
            ncols,
            sizes,
            sharey=sharey,
            edge_labels_only=edge_labels_only,
            ax2=ax2,
            # fig=fig,
        )
    if legend:
        outside_legend(fig, axs, ncol=aesthetics.legend_ncol, fontsize=sizes["legend"])
    return fig, results
