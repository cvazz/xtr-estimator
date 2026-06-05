# ============================================================================
# Background-density (rho_floor) estimation for floor-corrected occupancy.
#
# Intended to be merged into the preprocessing module shown, so the following
# names are assumed to already be in scope:
#     logger, voxel_volume, generate_masks, minimum_blob_size
# The only new top-level import required is:
#     from scipy import ndimage
#
# Convention, as elsewhere in the module: the "true" map stays an rsmap.Map,
# but everything here is a real-space operation and therefore works on numpy
# arrays obtained via Map.to_3d_numpy_map(map_sampling=...). Functions stay
# independent of concrete voxel size by taking the gemmi.UnitCell and using
# voxel_volume() to convert physical lengths (A) into voxel counts.
# ============================================================================

import numpy as np
import gemmi
from scipy import ndimage

from .masking import voxel_volume, minimum_blob_size
from .logger import setup_logger
from .processing import generate_masks
logger = setup_logger()  # ensure logger is configured for this module

# ----------------------------------------------------------------------------
# small voxel-size-aware helpers
# ----------------------------------------------------------------------------
def _voxel_length(shape: tuple, cell: gemmi.UnitCell) -> float:
    """Cube-root of the voxel volume, i.e. an isotropic voxel edge length (A).

    NOTE: index-space isotropy only equals real-space isotropy for orthogonal
    cells. For monoclinic/triclinic cells the dilation/smoothing neighbourhoods
    below are skewed; resample to an orthogonal grid (or apply the metric
    tensor) first if that matters for your space group.
    """
    return float(voxel_volume(shape, cell)) ** (1.0 / 3.0)


def _periodic_binary_dilation(mask: np.ndarray, iterations: int) -> np.ndarray:
    """binary_dilation that respects the periodic unit cell via a wrap pad."""
    if iterations < 1:
        return mask.copy()
    p = iterations
    padded = np.pad(mask, p, mode="wrap")
    dil = ndimage.binary_dilation(padded, iterations=iterations)
    return dil[p:-p, p:-p, p:-p]


def _periodic_binary_propagation(
    seed: np.ndarray, fillable: np.ndarray, pad: int
) -> np.ndarray:
    """Morphological reconstruction (flood of `seed` within `fillable`) with a
    wrap pad so connectivity across the cell boundary is captured locally."""
    p = max(int(pad), 1)
    seed_p = np.pad(seed, p, mode="wrap")
    fill_p = np.pad(fillable, p, mode="wrap")
    flooded = ndimage.binary_propagation(seed_p, mask=fill_p)
    return flooded[p:-p, p:-p, p:-p]


# ----------------------------------------------------------------------------
# step 1: significant negative regions of the difference map
# ----------------------------------------------------------------------------
def label_negative_regions(
    diffmap_np: np.ndarray,
    cell: gemmi.UnitCell,
    sigma_level: float = 3.0,
    min_blob_size: float = 10.0,
    denoise: bool = False,
) -> np.ndarray:
    """Boolean mask of the negative difference-map regions worth classifying.

    Voxels below -sigma_level * RMSD(diffmap) are kept, optionally opened to
    remove single-voxel noise, then filtered to blobs larger than
    `min_blob_size` (A^3) via the existing minimum_blob_size().
    """
    sigma = float(diffmap_np.std())
    neg = diffmap_np < -sigma_level * sigma
    logger.debug(f"sigma_delta={sigma:.4f}; {neg.sum()} voxels below -{sigma_level} sigma")

    if denoise:
        neg = ndimage.binary_opening(neg)

    labels, n_found = ndimage.label(neg, structure=np.ones((3, 3, 3)))
    logger.info(f"Found {n_found} raw negative regions at {sigma_level} sigma_delta")

    # minimum_blob_size raises if nothing survives the size cut, which is the
    # right behaviour: it signals the user to lower sigma_level / min_blob_size.
    neg_mask = minimum_blob_size(labels, min_blob_size, cell)
    return neg_mask


# ----------------------------------------------------------------------------
# step 2: split negative regions into solvent-connected (wet) vs isolated (dry)
# ----------------------------------------------------------------------------
def classify_solvent_connectivity(
    neg_mask: np.ndarray,
    solvent_mask: np.ndarray,
    cell: gemmi.UnitCell,
    connection_distance: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify significant negative voxels by reachability from bulk solvent.

    A negative region counts as 'wet' (relaxes toward rho_solvent) if it lies
    within `connection_distance` (A) of the reference-state bulk-solvent mask,
    allowing the flood to bridge protein walls up to that thickness. Everything
    else is 'dry' (relaxes toward vacuum, floor ~ 0).

    connection_distance is the dominant tuning knob and a sharp one: it is
    effectively the maximum wall thickness deemed water-accessible. Validate it
    on a known-buried case (should stay dry) and a surface case (should go wet).

    LIMITATION: connectivity is measured on the *reference* (ground) state, so
    it cannot see channels the reaction opens/closes, nor whether water has had
    time to equilibrate at the relevant delay. Treat 'wet' as an upper bound on
    filling, not a determination.
    """
    solvent_bool = np.asarray(solvent_mask).astype(bool)
    neg_mask = np.asarray(neg_mask).astype(bool)

    vox = _voxel_length(neg_mask.shape, cell)
    iterations = max(1, int(round(connection_distance / vox)))

    reach = _periodic_binary_dilation(solvent_bool, iterations)
    fillable = reach | neg_mask
    wet_reach = _periodic_binary_propagation(reach, fillable, pad=iterations + 1)

    wet_mask = wet_reach & neg_mask
    dry_mask = neg_mask & ~wet_mask
    logger.info(
        f"Connectivity (D={connection_distance} A -> {iterations} vox): "
        f"{int(wet_mask.sum())} wet / {int(dry_mask.sum())} dry negative voxels"
    )
    return wet_mask, dry_mask


# ----------------------------------------------------------------------------
# floor value for the wet voxels
# ----------------------------------------------------------------------------
def _local_solvent_field(
    rho0_np: np.ndarray,
    solvent_bool: np.ndarray,
    cell: gemmi.UnitCell,
    smoothing_radius: float,
) -> np.ndarray:
    """Mask-weighted (Shepard) smoothing of rho0 over solvent voxels only, so
    protein/feature voxels do not leak in. Captures the first-hydration-shell
    gradient that a single global rho_bulk flattens away. mode='wrap' keeps it
    periodic."""
    sigma_vox = smoothing_radius / _voxel_length(rho0_np.shape, cell)
    w = solvent_bool.astype(np.float32)
    num = ndimage.gaussian_filter(rho0_np.astype(np.float32) * w, sigma_vox, mode="wrap")
    den = ndimage.gaussian_filter(w, sigma_vox, mode="wrap")
    with np.errstate(invalid="ignore", divide="ignore"):
        field = np.where(den > 1e-6, num / den, np.nan)
    return field


# ----------------------------------------------------------------------------
# orchestrator
# ----------------------------------------------------------------------------
def estimate_rho_floor(
    diffmap_np: np.ndarray,
    rho0_np: np.ndarray,
    solvent_mask: np.ndarray,
    cell: gemmi.UnitCell,
    *,
    sigma_level: float = 3.0,
    min_blob_size: float = 10.0,
    connection_distance: float = 2.0,
    floor_mode: str = "constant",  # "constant" | "smooth"
    rho_bulk: float | None = None,
    smoothing_radius: float = 3.0,
) -> tuple[np.ndarray, dict]:
    """Estimate the per-voxel background density rho_floor(r) a vacated region
    relaxes to, for the floor-corrected per-voxel estimate

        chi(r) = -delta_rho(r) / (rho0(r) - rho_floor(r)).

    Wet (solvent-connected) negative voxels are floored at the bulk solvent
    level; dry (isolated) ones at 0 (vacuum). This recovers the standard Vacuum
    Matching estimate exactly where rho_floor == 0.

    PREREQUISITE: rho0_np must be on an absolute scale with F000 applied (e.g.
    via estimate_absolute_densities), otherwise both floors are displaced by a
    global constant and the correction is biased.

    Parameters
    ----------
    diffmap_np, rho0_np : real-space arrays (Map.to_3d_numpy_map), same grid.
    solvent_mask        : 1.0/True in bulk solvent (as generate_masks returns).
    floor_mode          : "constant" -> wet voxels get the global rho_bulk
                          (simplest defensible first pass);
                          "smooth"   -> wet voxels get the local solvent field
                          (captures the hydration-shell gradient).
    rho_bulk            : global bulk level; if None, median of rho0 over solvent.

    Returns
    -------
    rho_floor : np.ndarray   (vacuum = 0, wet = rho_bulk or local field)
    info      : dict         (neg/wet/dry masks + bookkeeping)

    KNOWN BLIND SPOT: if a vacated site refills with an *ordered* water in the
    triggered state the true floor is a peak, not bulk; a reference-state
    estimate will under-call it. The median/smoothing keeps that from polluting
    the estimate but cannot predict it. Atom-shift contamination (a negative
    lobe paired with a nearby positive lobe) is *not* removed here; rely on
    min_blob_size and, if needed, a separate paired-lobe guard upstream.
    """
    solvent_bool = np.asarray(solvent_mask).astype(bool)

    neg_mask = label_negative_regions(diffmap_np, cell, sigma_level, min_blob_size)
    wet_mask, dry_mask = classify_solvent_connectivity(
        neg_mask, solvent_bool, cell, connection_distance
    )

    if rho_bulk is None:
        rho_bulk = float(np.median(rho0_np[solvent_bool])) if solvent_bool.any() else 0.0
        logger.info(f"rho_bulk not supplied; median rho0 over solvent = {rho_bulk:.4f}")

    rho_floor = np.zeros_like(rho0_np, dtype=np.float32)  # dry / vacuum default

    if floor_mode == "constant":
        rho_floor[wet_mask] = rho_bulk
    elif floor_mode == "smooth":
        field = _local_solvent_field(rho0_np, solvent_bool, cell, smoothing_radius)
        wet_vals = field[wet_mask]
        wet_vals = np.where(np.isfinite(wet_vals), wet_vals, rho_bulk)  # fallback
        rho_floor[wet_mask] = wet_vals
    else:
        raise ValueError(f"Unknown floor_mode '{floor_mode}'. Use 'constant' or 'smooth'.")

    info = {
        "neg_mask": neg_mask,
        "wet_mask": wet_mask,
        "dry_mask": dry_mask,
        "rho_bulk": rho_bulk,
        "n_wet": int(wet_mask.sum()),
        "n_dry": int(dry_mask.sum()),
        "floor_mode": floor_mode,
        "connection_distance": connection_distance,
    }
    return rho_floor, info


# ----------------------------------------------------------------------------
# usage sketch (drop into the existing pipeline; not executed here)
# ----------------------------------------------------------------------------
# ms        = config["general"]["map_sampling"]
# diff_np   = diffmap.to_3d_numpy_map(map_sampling=ms)
# rho0_np   = map_dark.to_3d_numpy_map(map_sampling=ms)          # F000-corrected
# _, solv   = generate_masks(pdb_file, rho0_np.shape, map_dark.cell, map_dark.spacegroup)
# rho_bulk  = estimate_absolute_densities(...)["rho_bulk"]       # already available
#
# rho_floor, info = estimate_rho_floor(
#     diff_np, rho0_np, solv, map_dark.cell,
#     floor_mode="constant", rho_bulk=rho_bulk,
# )
# chi = np.where(info["neg_mask"], -diff_np / (rho0_np - rho_floor), np.nan)

def get_rho_floor(map_dark, diffmap, config, solvent_level):
    if solvent_level is np.nan:
        raise ValueError("Cannot estimate rho_floor: solvent level is NaN. Check that the input map has valid densities and that the solvent mask is correctly generated.")
    map_dark_np = map_dark.to_3d_numpy_map(map_sampling=config.general.map_sampling)
    diffmap_np = diffmap.to_3d_numpy_map(map_sampling=config.general.map_sampling)
    _, solv = generate_masks(
        config.input_files.pdb_dark,
        map_dark_np.shape,
        map_dark.cell,
        map_dark.spacegroup,
    )
    rho_floor, info = estimate_rho_floor(
        diffmap_np.copy(),
        map_dark_np.copy(),
        solv,
        map_dark.cell,
        min_blob_size=config.masking.min_blob_size,
        floor_mode="constant",
        rho_bulk=solvent_level,
    )
    return rho_floor