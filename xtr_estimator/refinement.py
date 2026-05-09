import reciprocalspaceship as rs
import os
import shutil
import subprocess
import logging
from pathlib import Path
import gemmi
import numpy as np
from itertools import combinations
from typing import Union

from xtr_estimator.xtr_maps import find_rfree_column

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_command(cmd, log_name, cwd):
    """Utility to run shell commands and log output."""
    log_path = Path(cwd) / log_name
    with open(log_path, "w") as log_file:
        result = subprocess.run(
            cmd, cwd=cwd, stdout=log_file, stderr=subprocess.STDOUT, text=True
        )
    return result.returncode == 0


def extract_simple_stats(log_path):
    """
    Placeholder for extracting R-factors from the log.
    In a real scenario, you'd parse the 'Final R-work' and 'R-free' lines.
    """
    stats = {"r_work": None, "r_free": None}
    if not os.path.exists(log_path):
        return stats

    with open(log_path, "r") as f:
        for line in f:
            if "Final R-work =" in line:
                stats["r_work"] = line.split("=")[1].split(",")[0].strip()
                stats["r_free"] = line.split("=")[2].strip()
    return stats


def run_single_refinement(
    pdb_file, mtz_file, run_id, output_dir=".", number_iterations=3, add_arguments=[], cif_file=None
):
    """
    Refines a single PDB against an MTZ, isolates work in run_id folder,
    and returns refinement stats and the final MTZ path.
    """
    # 1. Path Setup
    base_out = Path(output_dir).resolve()
    sandbox = base_out / str(run_id)
    sandbox.mkdir(parents=True, exist_ok=True)

    pdb_abs = Path(pdb_file).resolve()
    mtz_abs = Path(mtz_file).resolve()

    if not (pdb_abs.exists() and mtz_abs.exists()):
        logger.error("Input files missing.")
        raise FileNotFoundError(
            f"PDB or MTZ file not found. Inputs were: {pdb_file}, {mtz_file}"
        )
    ds_temp = rs.read_mtz(mtz_file)
    if cif_file is not None:
        cif_abs = Path(cif_file).resolve()
        if cif_abs.exists():
            add_arguments += [cif_file]

    try:
        find_rfree_column(ds_temp)
    except ValueError:
        add_arguments += [
            "refinement.input.xray_data.r_free_flags.generate=True",
            "refinement.input.xray_data.r_free_flags.fraction=0.05",
        ]

    # 2. Refinement Command
    # We use the prefix to easily identify output files
    prefix = f"refine_{run_id}"
    refine_cmd = (
        [
            "phenix.refine",
            str(pdb_abs),
            str(mtz_abs),
        ]
        + add_arguments
        + [
            "strategy=individual_sites+individual_adp",
            f"main.number_of_macro_cycles={number_iterations}",
            f"output.prefix={prefix}",
            "output.serial=1",
            "--overwrite",
        ]
    )

    logger.info(f"Running refinement, for logs see {sandbox / f'{prefix}_001.log'} ...")
    success = run_command(refine_cmd, f"{prefix}.log", cwd=sandbox)

    if not success:
        logger.error("Phenix refinement failed.")
        logger.info("Refinement command arguments:\n" + "\n".join(f"  {arg}" for arg in refine_cmd))
        raise RuntimeError("Refinement failed.")

    # 3. Validation (CC)
    expected_pdb = sandbox / f"{prefix}_001.pdb"
    expected_mtz = sandbox / f"{prefix}_001.mtz"

    val_log = f"validate_{run_id}.log"
    run_command(
        ["phenix.get_cc_mtz_pdb", expected_pdb.name, expected_mtz.name],
        val_log,
        cwd=sandbox,
    )

    # 4. Cleanup and File Movement
    # Requirement: All files in run_id folder EXCEPT for the output MTZ.
    final_mtz_destination = base_out / f"{run_id}_final.mtz"
    final_pdb_destination = base_out / f"{run_id}_final.pdb"

    if expected_pdb.exists():
        shutil.move(str(expected_pdb), str(final_pdb_destination))
        logger.info(f"Moved output PDB to: {final_pdb_destination}")
    else:
        logger.warning("Output PDB not found.")

    if expected_mtz.exists():
        shutil.move(str(expected_mtz), str(final_mtz_destination))
        logger.info(f"Moved output MTZ to: {final_mtz_destination}")
    else:
        logger.warning("Output MTZ not found.")

    # 5. Extract Stats
    stats = extract_simple_stats(sandbox / f"{prefix}.log")

    return stats, str(final_mtz_destination), str(final_pdb_destination)


def get_origin_code(struct_num, altloc):
    """
    Maps structure number and original altloc to a fixed letter A-F.
    """
    alt = altloc.strip().upper()

    if struct_num == 1:
        if not alt:
            return "A"
        elif alt == "A":
            return "B"
        elif alt == "B":
            return "C"

    elif struct_num == 2:
        if not alt:
            return "D"
        elif alt == "A":
            return "E"
        elif alt == "B":
            return "F"

    return f"{struct_num}{alt}"




def merge_structures_model_level(
    st1: Union[str, gemmi.Structure],
    st2: Union[str, gemmi.Structure],
    st1_weight: float = 0.5,
) -> gemmi.Structure:
    """
    Model-level merge:
    - Entire structure 1 → altloc A
    - Entire structure 2 → altloc B
    - No coordinate averaging
    - Chemically safe for Phenix refinement
    """

    if isinstance(st1, str):
        st1 = gemmi.read_structure(st1)
    if isinstance(st2, str):
        st2 = gemmi.read_structure(st2)

    assert 0.0 <= st1_weight <= 1.0, "st1_weight must be between 0 and 1"
    st2_weight = 1.0 - st1_weight

    # --- Copy header safely ---
    out = gemmi.Structure()
    out.name = st1.name
    out.cell = st1.cell
    out.spacegroup_hm = st1.spacegroup_hm
    out.ncs = st1.ncs
    out.entities = st1.entities
    out.raw_remarks = st1.raw_remarks

    # We create ONE model
    new_model = gemmi.Model("1")

    # Index chains in st2
    st2_chains = {}
    for m in st2:
        for c in m:
            st2_chains.setdefault(c.name, []).append(c)

    for m1 in st1:
        for c1 in m1:
            new_chain = gemmi.Chain(c1.name)

            # Find matching chain in st2 (first occurrence)
            c2_list = st2_chains.get(c1.name, [])
            c2 = c2_list[0] if c2_list else None

            # Build residue map for st2
            c2_res_map = {}
            if c2:
                for r2 in c2:
                    key = (r2.seqid.num, r2.seqid.icode, r2.name)
                    c2_res_map[key] = r2

            for r1 in c1:
                key = (r1.seqid.num, r1.seqid.icode, r1.name)

                r2 = c2_res_map.get(key)

                new_res = gemmi.Residue()
                new_res.name = r1.name
                new_res.seqid = r1.seqid
                new_res.entity_type = r1.entity_type
                new_res.subchain = r1.subchain

                # --- Add structure 1 atoms (altloc A) ---
                for atom in r1:
                    a = gemmi.Atom()
                    a.name = atom.name
                    a.element = atom.element
                    a.charge = atom.charge
                    a.pos = atom.pos
                    a.b_iso = atom.b_iso
                    a.occ = st1_weight
                    a.altloc = "A"
                    new_res.add_atom(a)

                # --- Add structure 2 atoms (altloc B) ---
                if r2:
                    for atom in r2:
                        a = gemmi.Atom()
                        a.name = atom.name
                        a.element = atom.element
                        a.charge = atom.charge
                        a.pos = atom.pos
                        a.b_iso = atom.b_iso
                        a.occ = st2_weight
                        a.altloc = "B"
                        new_res.add_atom(a)

                new_chain.add_residue(new_res)

            new_model.add_chain(new_chain)

    out.add_model(new_model)
    return out




def combine_and_weight_structures_refined(st1, st2, st1weight=0.5, threshold=0.5):
    if isinstance(st1, str):
        st1 = gemmi.read_structure(st1)
    if isinstance(st2, str):
        st2 = gemmi.read_structure(st2)

    st2weight = 1.0 - st1weight

    # 1. NEW: Clone the first structure to preserve all REMARKs, CRYST1, SCALE, etc.
    new_st = st1.clone()

    # Remove existing models so we can rebuild them with our merged coordinates
    new_st.models.clear()

    # Phenix-friendly AltLoc map
    ALT_MAP = {1: "A", 2: "B"}
    BACKBONE = {"N", "CA", "C", "O"}

    for m_idx, (m1, m2) in enumerate(zip(st1, st2)):
        new_model = gemmi.Model(getattr(m1, "name", str(m_idx + 1)))

        for c1, c2 in zip(m1, m2):
            new_chain = gemmi.Chain(c1.name)
            c2_residues = {(r.seqid.num, r.seqid.icode, r.name): r for r in c2}

            for r1 in c1:
                new_res = gemmi.Residue()
                new_res.name = r1.name
                new_res.seqid = r1.seqid

                r2 = c2_residues.get((r1.seqid.num, r1.seqid.icode, r1.name))
                atom_groups = {}

                # Helper to pool atoms
                def add_to_pool(res, weight, struct_idx):
                    if not res:
                        return
                    for atom in res:
                        atom_groups.setdefault(atom.name, []).append(
                            {
                                "src": struct_idx,
                                "occ": atom.occ * weight,
                                "pos": atom.pos,
                                "b_iso": atom.b_iso,
                                "element": atom.element,
                                "charge": atom.charge,
                            }
                        )

                add_to_pool(r1, st1weight, 1)
                add_to_pool(r2, st2weight, 2)

                for atom_name, pool in atom_groups.items():
                    # If it's a backbone atom, we are much more aggressive about merging
                    active_threshold = 1.5 if atom_name in BACKBONE else threshold

                    while len(pool) > 1:
                        dists = [
                            pool[i]["pos"].dist(pool[j]["pos"])
                            for i, j in combinations(range(len(pool)), 2)
                        ]
                        idx_pairs = list(combinations(range(len(pool)), 2))
                        min_idx = np.argmin(dists)

                        if dists[min_idx] < active_threshold:
                            i, j = idx_pairs[min_idx]
                            a1, a2 = pool.pop(max(i, j)), pool.pop(min(i, j))

                            total_occ = a1["occ"] + a2["occ"]
                            w1, w2 = (
                                (a1["occ"] / total_occ, a2["occ"] / total_occ)
                                if total_occ > 0
                                else (0.5, 0.5)
                            )

                            pool.append(
                                {
                                    "src": 0,  # 0 indicates merged
                                    "occ": total_occ,
                                    "pos": gemmi.Position(
                                        a1["pos"].x * w1 + a2["pos"].x * w2,
                                        a1["pos"].y * w1 + a2["pos"].y * w2,
                                        a1["pos"].z * w1 + a2["pos"].z * w2,
                                    ),
                                    "b_iso": a1["b_iso"] * w1 + a2["b_iso"] * w2,
                                    "element": a1["element"],
                                    "charge": a1["charge"],
                                }
                            )
                        else:
                            break

                    # Add finalized atoms to residue
                    for i, state in enumerate(pool):
                        new_atom = gemmi.Atom()
                        new_atom.name = atom_name
                        new_atom.pos = state["pos"]
                        new_atom.occ = state["occ"]
                        new_atom.b_iso = state["b_iso"]
                        new_atom.element = state["element"]
                        new_atom.charge = state["charge"]

                        # Assign AltLocs only if multiple states exist
                        if len(pool) > 1:
                            new_atom.altloc = ALT_MAP.get(state["src"], chr(65 + i))
                        else:
                            # 2. FIX: gemmi requires the null character for "no altloc"
                            new_atom.altloc = "\0"
                            # Force full occupancy for single-state atoms to satisfy Phenix
                            new_atom.occ = 1.00

                        new_res.add_atom(new_atom)

                new_chain.add_residue(new_res)
            new_model.add_chain(new_chain)
        new_st.add_model(new_model)

    return new_st


def combine_and_weight_structures(st1, st2, st1weight=0.5, threshold=0.5):
    if isinstance(st1, str):
        st1 = gemmi.read_structure(st1)
    if isinstance(st2, str):
        st2 = gemmi.read_structure(st2)

    st2weight = 1.0 - st1weight

    new_st = gemmi.Structure()
    new_st.cell = st1.cell
    new_st.spacegroup_hm = st1.spacegroup_hm

    print(f"Merging with threshold {threshold} Å")
    print(f"Weights: Struct 1 = {st1weight:.2f}, Struct 2 = {st2weight:.2f}\n")

    for i, (m1, m2) in enumerate(zip(st1, st2)):
        model_name = getattr(m1, "name", str(i + 1))
        new_model = gemmi.Model(model_name)

        for c1, c2 in zip(m1, m2):
            new_chain = gemmi.Chain(c1.name)

            # --- THE FIX: Create a dictionary of Struct 2 residues ---
            # This indexes them by Sequence Number and Name (e.g., (42, 'ALA'))
            # This completely prevents the "slipped gear" frameshift bug.
            c2_residues = {}
            for r2 in c2:
                res_id = (r2.seqid.num, r2.name)
                c2_residues[res_id] = r2

            for r1 in c1:
                new_res = gemmi.Residue()
                new_res.name = r1.name
                new_res.seqid = r1.seqid

                # Ask Structure 2 if it has this exact residue
                r2 = c2_residues.get((r1.seqid.num, r1.name), None)

                atom_groups = {}

                # Always add atoms from our reference structure (Struct 1)
                for atom in r1:
                    code = get_origin_code(1, atom.altloc)
                    atom_groups.setdefault(atom.name, []).append(
                        {
                            "origins": [code],
                            "occ": atom.occ * st1weight,
                            "pos": gemmi.Position(atom.pos.x, atom.pos.y, atom.pos.z),
                            "b_iso": atom.b_iso,
                            "element": atom.element,
                            "charge": atom.charge,
                        }
                    )

                # Only add atoms from Struct 2 if the matching residue actually exists
                if r2:
                    for atom in r2:
                        code = get_origin_code(2, atom.altloc)
                        atom_groups.setdefault(atom.name, []).append(
                            {
                                "origins": [code],
                                "occ": atom.occ * st2weight,
                                "pos": gemmi.Position(
                                    atom.pos.x, atom.pos.y, atom.pos.z
                                ),
                                "b_iso": atom.b_iso,
                                "element": atom.element,
                                "charge": atom.charge,
                            }
                        )

                # Iteratively cluster and average based on threshold
                for atom_name, pool in atom_groups.items():
                    while len(pool) > 1:
                        dists = []
                        idx_pairs = list(combinations(range(len(pool)), 2))

                        for idx_i, idx_j in idx_pairs:
                            dist = pool[idx_i]["pos"].dist(pool[idx_j]["pos"])
                            dists.append(dist)

                        dists = np.array(dists)
                        min_idx = np.argmin(dists)
                        min_dist = dists[min_idx]

                        if min_dist < threshold:
                            idx1, idx2 = idx_pairs[min_idx]
                            a1, a2 = pool[idx1], pool[idx2]

                            new_occ = a1["occ"] + a2["occ"]

                            if new_occ > 0:
                                w1 = a1["occ"] / new_occ
                                w2 = a2["occ"] / new_occ
                            else:
                                w1, w2 = 0.5, 0.5

                            new_pos = gemmi.Position(
                                a1["pos"].x * w1 + a2["pos"].x * w2,
                                a1["pos"].y * w1 + a2["pos"].y * w2,
                                a1["pos"].z * w1 + a2["pos"].z * w2,
                            )
                            new_b = a1["b_iso"] * w1 + a2["b_iso"] * w2

                            merged_item = {
                                "origins": a1["origins"] + a2["origins"],
                                "occ": new_occ,
                                "pos": new_pos,
                                "b_iso": new_b,
                                "element": a1["element"],
                                "charge": a1["charge"],
                            }

                            pool.pop(max(idx1, idx2))
                            pool.pop(min(idx1, idx2))
                            pool.append(merged_item)
                        else:
                            break

                    # Finalize AltLocs and add to the new residue
                    for state in pool:
                        new_atom = gemmi.Atom()
                        new_atom.name = atom_name
                        new_atom.pos = state["pos"]
                        new_atom.occ = state["occ"]
                        new_atom.b_iso = state["b_iso"]
                        new_atom.element = state["element"]
                        new_atom.charge = state["charge"]

                        if len(pool) == 1:
                            new_atom.altloc = "\0"
                        else:
                            origin_str = state["origins"][0]
                            new_atom.altloc = (
                                origin_str[0] if len(origin_str) > 0 else "\0"
                            )

                        new_res.add_atom(new_atom)

                new_chain.add_residue(new_res)
            new_model.add_chain(new_chain)
        new_st.add_model(new_model)

    return new_st




def collapse_to_main_conformer(st: Union[str, gemmi.Structure]) -> gemmi.Structure:
    """
    For each atom name in each residue, keep only the altloc
    with the highest occupancy.
    """

    if isinstance(st, str):
        st = gemmi.read_structure(st)

    out = gemmi.Structure()
    out.name = st.name
    out.cell = st.cell
    out.spacegroup_hm = st.spacegroup_hm
    out.ncs = st.ncs
    out.entities = st.entities
    out.raw_remarks = st.raw_remarks

    for i, model in enumerate(st, start=1):
        new_model = gemmi.Model(str(i))

        for chain in model:
            new_chain = gemmi.Chain(chain.name)

            for res in chain:
                new_res = gemmi.Residue()
                new_res.name = res.name
                new_res.seqid = res.seqid
                new_res.entity_type = res.entity_type
                new_res.subchain = res.subchain

                # --- group atoms by name ---
                atom_map = {}
                for atom in res:
                    name = atom.name
                    if name not in atom_map:
                        atom_map[name] = atom
                    else:
                        if atom.occ > atom_map[name].occ:
                            atom_map[name] = atom

                # --- keep only best atoms ---
                for atom in atom_map.values():
                    a = gemmi.Atom()
                    a.name = atom.name
                    a.element = atom.element
                    a.charge = atom.charge
                    a.pos = atom.pos
                    a.b_iso = atom.b_iso
                    a.occ = 1.0  # normalize
                    a.altloc = "\0"  # remove altloc
                    new_res.add_atom(a)

                new_chain.add_residue(new_res)

            new_model.add_chain(new_chain)
        out.add_model(new_model)

    return out


def merge_structures_model_greedy(
    st1: Union[str, gemmi.Structure],
    st2: Union[str, gemmi.Structure],
    st1_weight: float = 0.5,
) -> gemmi.Structure:

    st1 = collapse_to_main_conformer(st1)
    st2 = collapse_to_main_conformer(st2)

    st2_weight = 1.0 - st1_weight

    out = gemmi.Structure()
    out.name = st1.name
    out.cell = st1.cell
    out.spacegroup_hm = st1.spacegroup_hm
    out.ncs = st1.ncs
    out.entities = st1.entities
    out.raw_remarks = st1.raw_remarks

    new_model = gemmi.Model("1")

    # index chains in st2
    st2_chains = {c.name: c for m in st2 for c in m}

    for m1 in st1:
        for c1 in m1:
            new_chain = gemmi.Chain(c1.name)
            c2 = st2_chains.get(c1.name)

            # residue map for st2
            c2_res_map = {}
            if c2:
                for r in c2:
                    key = (r.seqid.num, r.seqid.icode, r.name)
                    c2_res_map[key] = r

            for r1 in c1:
                key = (r1.seqid.num, r1.seqid.icode, r1.name)
                r2 = c2_res_map.get(key)

                new_res = gemmi.Residue()
                new_res.name = r1.name
                new_res.seqid = r1.seqid
                new_res.entity_type = r1.entity_type
                new_res.subchain = r1.subchain

                # --- structure 1 → altloc A ---
                for atom in r1:
                    a = gemmi.Atom()
                    a.name = atom.name.strip().rjust(4)
                    a.element = atom.element
                    a.charge = atom.charge
                    a.pos = atom.pos
                    a.b_iso = atom.b_iso
                    a.occ = st1_weight
                    a.altloc = "A"
                    new_res.add_atom(a)

                # --- structure 2 → altloc B ---
                if r2:
                    for atom in r2:
                        a = gemmi.Atom()
                        a.name = atom.name.strip().rjust(4)
                        a.element = atom.element
                        a.charge = atom.charge
                        a.pos = atom.pos
                        a.b_iso = atom.b_iso
                        a.occ = st2_weight
                        a.altloc = "B"
                        new_res.add_atom(a)

                new_chain.add_residue(new_res)

            new_model.add_chain(new_chain)

    out.add_model(new_model)
    return out
