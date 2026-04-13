import os
import shutil
import subprocess
import logging
from pathlib import Path
import gemmi
import numpy as np
from itertools import combinations

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
    pdb_file, mtz_file, run_id, output_dir=".", number_iterations=3
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
        return None, None

    # 2. Refinement Command
    # We use the prefix to easily identify output files
    prefix = f"refine_{run_id}"
    refine_cmd = [
        "phenix.refine",
        str(pdb_abs),
        str(mtz_abs),
        "strategy=individual_sites+individual_adp",
        f"main.number_of_macro_cycles={number_iterations}",
        f"output.prefix={prefix}",
        "output.serial=1",
        "--overwrite",
    ]

    logger.info(f"Running refinement in {sandbox}...")
    success = run_command(refine_cmd, f"{prefix}.log", cwd=sandbox)

    if not success:
        logger.error("Phenix refinement failed.")
        return None, None, None

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
