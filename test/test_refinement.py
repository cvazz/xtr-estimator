from pathlib import Path
from types import SimpleNamespace

from xtr_estimator import refinement


def test_run_command_writes_log_and_returns_success(monkeypatch, tmp_path):
    def fake_run(cmd, cwd, stdout, stderr, text):
        stdout.write("mock output\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(refinement.subprocess, "run", fake_run)

    ok = refinement.run_command(["echo", "x"], "job.log", tmp_path)
    assert ok is True
    assert (tmp_path / "job.log").exists()
    assert "mock output" in (tmp_path / "job.log").read_text(encoding="utf-8")


def test_extract_simple_stats_parses_r_values(tmp_path):
    log_file = tmp_path / "refine.log"
    log_file.write_text(
        "something\nFinal R-work = 0.210, R-free = 0.250\n",
        encoding="utf-8",
    )

    stats = refinement.extract_simple_stats(log_file)
    assert stats["r_work"] == "0.210"
    assert stats["r_free"] == "0.250"


def test_extract_simple_stats_missing_file_returns_none_values(tmp_path):
    stats = refinement.extract_simple_stats(tmp_path / "nope.log")
    assert stats == {"r_work": None, "r_free": None}


def test_get_origin_code_mapping_examples():
    assert refinement.get_origin_code(1, "") == "A"
    assert refinement.get_origin_code(1, "A") == "B"
    assert refinement.get_origin_code(2, "B") == "F"
    assert refinement.get_origin_code(3, "z") == "3Z"


def test_run_single_refinement_returns_none_when_inputs_missing(tmp_path):
    stats, mtz = refinement.run_single_refinement(
        pdb_file=tmp_path / "missing.pdb",
        mtz_file=tmp_path / "missing.mtz",
        run_id="r1",
        output_dir=tmp_path,
    )
    assert stats is None
    assert mtz is None


def test_run_single_refinement_happy_path_with_mocked_tools(monkeypatch, tmp_path):
    pdb_file = tmp_path / "in.pdb"
    mtz_file = tmp_path / "in.mtz"
    pdb_file.write_text("ATOM\n", encoding="utf-8")
    mtz_file.write_text("MTZ\n", encoding="utf-8")

    created_prefix = "refine_run42"

    def fake_run_command(cmd, log_name, cwd):
        cwd = Path(cwd)
        if log_name == f"{created_prefix}.log":
            (cwd / f"{created_prefix}_001.pdb").write_text("PDB\n", encoding="utf-8")
            (cwd / f"{created_prefix}_001.mtz").write_text("MTZ\n", encoding="utf-8")
            (cwd / log_name).write_text(
                "Final R-work = 0.200, R-free = 0.240\n", encoding="utf-8"
            )
        else:
            (cwd / log_name).write_text("cc ok\n", encoding="utf-8")
        return True

    monkeypatch.setattr(refinement, "run_command", fake_run_command)

    stats, out_mtz, out_pdb = refinement.run_single_refinement(
        pdb_file=pdb_file,
        mtz_file=mtz_file,
        run_id="run42",
        output_dir=tmp_path,
        number_iterations=2,
    )

    assert stats == {"r_work": "0.200", "r_free": "0.240"}
    assert Path(out_mtz).exists()
    assert Path(out_pdb).exists()
    assert Path(out_mtz).name == "run42_final.mtz"
    assert Path(out_pdb).name == "run42_final.pdb"


def test_run_single_refinement_returns_triple_none_on_refine_failure(monkeypatch, tmp_path):
    pdb_file = tmp_path / "in.pdb"
    mtz_file = tmp_path / "in.mtz"
    pdb_file.write_text("ATOM\n", encoding="utf-8")
    mtz_file.write_text("MTZ\n", encoding="utf-8")

    monkeypatch.setattr(refinement, "run_command", lambda *args, **kwargs: False)

    result = refinement.run_single_refinement(
        pdb_file=pdb_file,
        mtz_file=mtz_file,
        run_id="run_fail",
        output_dir=tmp_path,
    )
    assert result == (None, None, None)
