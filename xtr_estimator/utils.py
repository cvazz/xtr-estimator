import numpy as np
from meteor import rsmap

GridShape = tuple[int, int, int]

def map_to_array(m: rsmap.Map, grid_shape: GridShape) -> np.ndarray:
    """Sample `m` onto exactly `grid_shape`, bypassing gemmi's per-map auto-sizing."""
    mtz = m.to_gemmi()
    grid = mtz.transform_f_phi_to_map(
        m.amplitude_column_name,
        m.phase_column_name,
        exact_size=[int(n) for n in grid_shape],
    )
    return np.array(grid)