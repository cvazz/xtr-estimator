import numpy as np
import gemmi
from pathlib import Path

from scipy.ndimage import label, generate_binary_structure
from scipy.ndimage import convolve

from meteor import rsmap
from .logger import setup_logger
from .configuration import Settings, MaskingSettings

logger = setup_logger()

###############################################################################
# masking.py
# Exposed functions:
# - make_inclusion_mask(diffmap: rsmap.Map, map_dark: rsmap.Map, config: dict) -> np.ndarray
# - support_from_masker(pdb_file: str | Path | gemmi.Structure, grid_shape: tuple, radii_set: gemmi.Atomic


def _format_exclusion_overview_table(rows: list[dict], base_count: int) -> str:
    """Build a compact ASCII table for mask exclusion statistics."""
    headers = ["Step", "Control option", "Excluded voxels", "% of base"]

    formatted_rows = []
    for row in rows:
        excluded = int(row["excluded"])
        percent = (excluded / base_count * 100.0) if base_count else 0.0
        formatted_rows.append(
            [
                str(row["name"]),
                str(row["option"]),
                f"{excluded}",
                f"{percent:.2f}%",
            ]
        )

    widths = [len(h) for h in headers]
    for row in formatted_rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    def _line(values: list[str]) -> str:
        return (
            "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(values)) + " |"
        )

    separator = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    lines = [_line(headers), separator]
    lines.extend(_line(row) for row in formatted_rows)
    return "\n".join(lines)


def support_from_masker(
    pdb_file: str | Path | gemmi.Structure,
    grid_shape: tuple,
    radii_set: gemmi.AtomicRadiiSet = gemmi.AtomicRadiiSet.Cctbx,
    remove_waters: bool = True,
    options: dict = {},
):

    if isinstance(pdb_file, gemmi.Structure):
        st = pdb_file
    else:
        st = gemmi.read_structure(str(pdb_file))
    model = st[0]
    if remove_waters:
        model.remove_waters()

    grid = gemmi.Int8Grid()

    # carry over crystallographic metadata so the size check & masking work correctly
    grid.set_unit_cell(st.cell)
    try:
        # If the structure has a space group, use it; otherwise we stay in P1.
        sg_name = getattr(st, "spacegroup_hm", None)
        if sg_name:
            grid.spacegroup = gemmi.SpaceGroup(sg_name)
    except Exception:
        # safe fallback: no space group set (P1)
        pass

    # set the exact target grid size (Gemmi will validate it vs the space group)
    nu, nv, nw = map(int, grid_shape)
    grid.set_size(nu, nv, nw)

    # write the solvent mask into our pre-sized grid
    masker = gemmi.SolventMasker(radii_set)
    if options.get("rprobe", None) is not None:
        masker.rprobe = options["rprobe"]
    if options.get("rshrink", None) is not None:
        masker.rshrink = options["rshrink"]
    # masker.rprobe = 0.5
    # masker.rshrink = 0.5
    masker.put_mask_on_int8_grid(grid, model)

    # return as boolean numpy array
    return np.asarray(grid.array, dtype=bool)


def radius_mask_minibox_from_ccp4(
    ccp4: gemmi.Ccp4Map, unit_cell: gemmi.UnitCell, radius_A: float, dtype=np.uint8
) -> np.ndarray:
    """
    Create a minimal-size 3D NumPy array that is 1 for all voxels whose CARTESIAN
    distance from the origin is <= radius_A, 0 otherwise. The box is the tightest
    rectangular box that must contain the sphere in voxel space, accounting for skew.

    Inputs
    ------
    ccp4       : gemmi.Ccp4Map (already read via gemmi.read_ccp4_map)
    unit_cell  : gemmi.UnitCell associated with the map
    radius_A   : float, radius in Å
    dtype      : output dtype (default np.uint8; bool is fine too)

    Returns
    -------
    mask : np.ndarray with shape (2*ri+1, 2*rj+1, 2*rk+1)
           where (0,0,0) corresponds to the origin voxel (the center of the box).
    """
    grid = ccp4.grid
    nx, ny, nz = grid.nu, grid.nv, grid.nw
    uc = unit_cell

    # Cartesian step vectors (Å) for a +1 step in voxel i/j/k:
    # r = B @ (i/nx, j/ny, k/nz). So the basis vectors are:
    ei = uc.orthogonalize(gemmi.Fractional(1.0 / nx, 0.0, 0.0))  # Vec3 in Å
    ej = uc.orthogonalize(gemmi.Fractional(0.0, 1.0 / ny, 0.0))
    ek = uc.orthogonalize(gemmi.Fractional(0.0, 0.0, 1.0 / nz))

    # Convert to NumPy 3-vectors
    ei = np.array([ei.x, ei.y, ei.z], dtype=float)
    ej = np.array([ej.x, ej.y, ej.z], dtype=float)
    ek = np.array([ek.x, ek.y, ek.z], dtype=float)

    # Conservative half-widths along each voxel axis (tight for single-axis moves).
    # Any combination (i,j,k) uses r = i*ei + j*ej + k*ek; this box contains the sphere.
    ri = int(np.ceil(radius_A / np.linalg.norm(ei)))
    rj = int(np.ceil(radius_A / np.linalg.norm(ej)))
    rk = int(np.ceil(radius_A / np.linalg.norm(ek)))

    # Build coordinate grids of voxel indices centered at 0, with minimal shape.
    # Use ogrid for memory efficiency, then broadcast.
    II = np.arange(-ri, ri + 1, dtype=float)[:, None, None]
    JJ = np.arange(-rj, rj + 1, dtype=float)[None, :, None]
    KK = np.arange(-rk, rk + 1, dtype=float)[None, None, :]

    # Map voxel index triples -> Cartesian positions using the step basis
    # r(i,j,k) = i*ei + j*ej + k*ek
    # Broadcast multiply-and-sum along the last axis.
    # Shape: (2*ri+1, 2*rj+1, 2*rk+1, 3)
    RR = np.stack(
        [
            II * ei[0] + JJ * ej[0] + KK * ek[0],
            II * ei[1] + JJ * ej[1] + KK * ek[1],
            II * ei[2] + JJ * ej[2] + KK * ek[2],
        ],
        axis=-1,
    )

    # Squared distances in Å^2 and spherical inclusion test
    d2 = np.sum(RR * RR, axis=-1)
    # slice_3d(d2,startval=0, imkwargs=dict(vmin=0))

    mask_bool = d2 <= (radius_A * radius_A + 1e-12)  # small tolerance for FP

    # Return 1/0 array as requested (or bool if user sets dtype=bool)
    return mask_bool.astype(dtype)


def calculate_all_pos_blobs(diffmap_np: np.ndarray, sigma: float):

    diffmap_np -= diffmap_np.mean()
    threshold = np.std(diffmap_np) * sigma

    # Create masks for positive and negative blobs
    pos_mask = diffmap_np >= threshold

    # Use 3D connectivity for labeling
    structure = generate_binary_structure(3, 3)

    # Label positive and negative blobs
    pos_labeled, pos_num = label(pos_mask, structure=structure)  # type: ignore
    text = f"Used threshold for posmask: {threshold/np.max(diffmap_np):.3f}, found {pos_num} blobs"
    logger.debug(text)
    return pos_labeled


def voxel_volume(unit_cell_shape: tuple, unit_cell: gemmi.UnitCell) -> float:
    return unit_cell.volume / np.prod(unit_cell_shape)


def minimum_blob_size(
    all_neg_blobs: np.ndarray, min_blob_size: float, cell: gemmi.UnitCell
) -> np.ndarray:
    uniq, counts = np.unique(all_neg_blobs, return_counts=True)
    one_voxel_volume = voxel_volume(all_neg_blobs.shape, cell)
    min_count = min_blob_size / one_voxel_volume

    logger.debug(f"Using blobs with more than {min_count}({min_blob_size} A^3) voxels")
    logger.debug(f"Maximum Blob sizes found: {np.max(counts[1:])}")

    if np.max(counts[1:]) < min_count:
        error_message = ""
        error_message += f"No blobs found with size > {min_blob_size} A^3. Please decrease 'min_blob_size' or 'sigma'."
        raise ValueError(error_message)
    mask_np = np.zeros(all_neg_blobs.shape, dtype=bool)
    for uu in uniq[counts > min_count]:
        if uu == 0:
            continue

        mask_np += all_neg_blobs == uu
    return mask_np


def positive_density_blocking(
    diffmap: rsmap.Map, mask_np: np.ndarray, masking_config: dict, map_sampling: int
) -> np.ndarray:

    blocking_percentile = masking_config["blocking_percentile"]
    blocking_radius = masking_config["blocking_radius"]

    ccp4diff = diffmap.to_ccp4_map(map_sampling=map_sampling)
    neighborhood_kernel = radius_mask_minibox_from_ccp4(
        ccp4diff, diffmap.cell, radius_A=blocking_radius, dtype=np.uint8  # type: ignore
    )
    neighborhood_kernel[tuple(np.array(neighborhood_kernel.shape) // 2 + 1)] = 0

    log_text = f"Applying blocking radius of {blocking_radius} A around positive density regions"
    log_text += f" creating a mask of shape {neighborhood_kernel.shape}, considering a total of {np.sum(neighborhood_kernel)} voxels"
    log_text += " (control via 'blocking_radius' parameter)"
    logger.debug(log_text)

    diffmap_np = diffmap.to_3d_numpy_map(map_sampling=map_sampling)
    positive_data = np.maximum(diffmap_np, 0)
    result = convolve(positive_data, neighborhood_kernel, mode="wrap")
    result_perc = np.percentile(result, blocking_percentile)
    pos_mask = result > result_perc

    excluded = np.logical_and(mask_np, pos_mask)
    excluded_share = np.sum(excluded) / np.sum(mask_np)

    log_text = f"Excluding {np.sum(excluded)} ({excluded_share:.2%}) voxels from mask"
    log_text += f" due to positive density within {masking_config['blocking_radius']} A and above the {masking_config['blocking_percentile']} percentile"
    log_text += " (control via 'blocking_radius' and 'blocking_percentile' parameters)"
    logger.debug(log_text)
    mask_np = np.logical_and(mask_np, ~pos_mask)
    return mask_np


def make_inclusion_mask(diffmap: rsmap.Map, map_dark: rsmap.Map, config: Settings | dict):
    """_summary_

    Parameters
    ----------
    diffmap : _type_
        _description_
    map_dark_np : _type_
        _description_
    cell : _type_
        _description_
    config requires the following entries:
        general: dict
            - map_sampling: int

        mask: dict
            - sigma: float
                number of standard deviations above which to consider blobs
            - min_blob_size: float
                minimum size of blobs in Å^3
            - blocking_radius: float
                radius around positive density regions to block in Å
            - exclude_solvent: bool, optional
                whether to exclude solvent regions (default: True)
            - exclude_negative_dark: bool, optional
                whether to exclude regions with negative dark map values (default: True)
            - exclude_large_occupancy_outliers: float, optional
                threshold for excluding large occupancy outliers (default: False)
                If set, any blob with an occupancy greater than this value will be excluded.


    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    try:
        return make_inclusion_mask_real(
            diffmap,
            map_dark,
            map_sampling=config["general"]["map_sampling"],
            masking_config=config["masking"],
            pdbloc_dark=config["input_files"]["pdb_dark"],
        )
    except ValueError as e:
        logger.error("Error in make_inclusion_mask: " + str(e))
        logger.error(f"{np.min(map_dark.compute_dHKL())=}")
        logger.error(f"{np.min(diffmap.compute_dHKL())=}")
        logger.error(f"{config=}")

        raise


def make_inclusion_mask_real(
    diffmap: rsmap.Map,
    map_dark: rsmap.Map,
    map_sampling: int,
    masking_config: MaskingSettings | dict,
    pdbloc_dark: str | None = None,
):
    if isinstance(masking_config, dict):
        masking_config = MaskingSettings(**masking_config)
    elif not isinstance(masking_config, MaskingSettings):
        raise ValueError("masking_config must be either a dict or MaskingSettings instance")

    if pdbloc_dark is None:
        masking_config["exclude_solvent"] = False

    dark_size_std_threshold = masking_config["dark_size_threshold"]
    diffmap_np = diffmap.to_3d_numpy_map(map_sampling=map_sampling)

    ### Find all negative blobs ###
    sigma = masking_config["sigma"]

    all_neg_blobs = calculate_all_pos_blobs(-diffmap_np, sigma=sigma)

    ### Impose minimum blob size ###
    mask_np = minimum_blob_size(
        all_neg_blobs, masking_config["min_blob_size"], diffmap.cell  # type: ignore
    )
    base_mask_voxels = int(np.sum(mask_np))
    exclusion_rows: list[dict] = []

    before_blocking = int(np.sum(mask_np))
    mask_np = positive_density_blocking(diffmap, mask_np, masking_config, map_sampling)
    exclusion_rows.append(
        {
            "name": "Positive density neighborhood blocking",
            "option": "blocking_radius, blocking_percentile",
            "excluded": before_blocking - int(np.sum(mask_np)),
        }
    )

    if masking_config["exclude_solvent"]:
        before_solvent = int(np.sum(mask_np))
        only_solvent = support_from_masker(
            pdbloc_dark, mask_np.shape, gemmi.AtomicRadiiSet.Cctbx
        )
        solvent_mask = ~only_solvent
        excluded_by_solvent = np.logical_and(mask_np, only_solvent)
        log_text = f"Excluding an addtional {np.sum(excluded_by_solvent)} voxels from mask due to being in solvent"
        log_text += " (deactivate via 'exclude_solvent' parameter)"
        logger.debug(log_text)
        mask_np = np.logical_and.reduce([mask_np, solvent_mask])
        exclusion_rows.append(
            {
                "name": "Solvent exclusion",
                "option": "exclude_solvent",
                "excluded": before_solvent - int(np.sum(mask_np)),
            }
        )

    else:
        log_text = "Solvent blocking deactivated."
        log_text += " (activate via 'exclude_solvent' parameter)"
        logger.debug(log_text)
        exclusion_rows.append(
            {
                "name": "Solvent exclusion",
                "option": "exclude_solvent",
                "excluded": 0,
            }
        )

    if dark_size_std_threshold:
        mask_total_before = np.sum(mask_np)
        map_dark_np = map_dark.to_3d_numpy_map(map_sampling=map_sampling)
        mask_np = np.logical_and(mask_np, map_dark_np > dark_size_std_threshold)
        number_negative_darks = mask_total_before - np.sum(mask_np)
        log_text = ""
        log_text += f"Excluding an addtional {number_negative_darks} voxels from mask "
        log_text += f"due dark map smaller than mean + {dark_size_std_threshold} "
        log_text += "* sigma (deactivate via 'exclude_negative_dark' parameter)"
        if number_negative_darks:
            logger.debug(log_text)
        exclusion_rows.append(
            {
                "name": "Dark-map threshold filter",
                "option": "dark_size_threshold",
                "excluded": int(number_negative_darks),
            }
        )
    else:
        mask_total_before = np.sum(mask_np)
        map_dark_np = map_dark.to_3d_numpy_map(map_sampling=map_sampling)
        map_dark_threshold = 0
        mask_np = np.logical_and(mask_np, map_dark_np > map_dark_threshold)
        number_negative_darks = mask_total_before - np.sum(mask_np)
        log_text = ""
        log_text += (
            f"Excluding an addtional {number_negative_darks} voxels from mask due"
        )
        log_text += f"dark map smaller than mean + {dark_size_std_threshold} * sigma"
        log_text += " (deactivate via 'exclude_negative_dark' parameter)"
        if number_negative_darks:
            logger.debug(log_text)
        exclusion_rows.append(
            {
                "name": "Dark-map threshold filter",
                "option": "dark_size_threshold",
                "excluded": int(number_negative_darks),
            }
        )

    if masking_config["exclude_positive_diffmap"]:
        before_positive = int(np.sum(mask_np))
        log_text = ""
        log_text += "Excluding voxels with positive difference density from mask"
        log_text += " (activate via 'exclude_positive_diffmap' parameter)"
        logger.debug(log_text)
        mask_np = np.logical_and(mask_np, diffmap_np < 0)
        exclusion_rows.append(
            {
                "name": "Positive diffmap exclusion",
                "option": "exclude_positive_diffmap",
                "excluded": before_positive - int(np.sum(mask_np)),
            }
        )
    else:
        exclusion_rows.append(
            {
                "name": "Positive diffmap exclusion",
                "option": "exclude_positive_diffmap",
                "excluded": 0,
            }
        )

    if masking_config["exclude_large_occupancy_outliers"]:
        map_dark_np = map_dark.to_3d_numpy_map(map_sampling=map_sampling)
        mask_np_before = np.sum(mask_np)
        outliers = (
            np.where(map_dark_np != 0, -diffmap_np / map_dark_np, 0)
            < masking_config["exclude_large_occupancy_outliers"]
        )
        mask_np = np.logical_and(mask_np, outliers)
        log_text = f"Excluding an addtional {mask_np_before - np.sum(mask_np)} voxels from mask due to large occupancy outliers (threshold: {masking_config['exclude_large_occupancy_outliers']})"
        log_text += f" most negative voxel excluded: {-diffmap_np[~outliers].min():.3f}"
        log_text += " (activate via 'exclude_large_occupancy_outliers' parameter)"
        logger.debug(log_text)
        exclusion_rows.append(
            {
                "name": "Large occupancy outlier exclusion",
                "option": "exclude_large_occupancy_outliers",
                "excluded": int(mask_np_before - np.sum(mask_np)),
            }
        )
    else:
        exclusion_rows.append(
            {
                "name": "Large occupancy outlier exclusion",
                "option": "exclude_large_occupancy_outliers",
                "excluded": 0,
            }
        )

    final_voxels = int(np.sum(mask_np))
    total_excluded = base_mask_voxels - final_voxels
    header = (
        "Mask exclusion overview "
        f"(base voxels after min_blob_size: {base_mask_voxels}, \n"
        f"final included: {final_voxels}, total excluded: {total_excluded}"
        f"share kept: {final_voxels / base_mask_voxels:.1%}"
        ")"
    )
    table = _format_exclusion_overview_table(exclusion_rows, base_mask_voxels)
    logger.info("\n" + header + "\n" + table)

    # mask_ccp4 =
    return mask_np
