from pathlib import Path

import pytest

from xtr_estimator import xtr_maps


def test_find_rfree_column_single_match(monkeypatch):
    class DummyIntType:
        pass

    monkeypatch.setattr(xtr_maps.rs, "MTZIntDtype", DummyIntType)

    ds = type("DummyDS", (), {})()
    ds.columns = ["F", "Rfree_flag", "OTHER"]
    ds.dtypes = {
        "F": object(),
        "Rfree_flag": DummyIntType(),
        "OTHER": object(),
    }

    assert xtr_maps.find_rfree_column(ds) == "Rfree_flag"


def test_find_rfree_column_raises_when_not_found(monkeypatch):
    class DummyIntType:
        pass

    monkeypatch.setattr(xtr_maps.rs, "MTZIntDtype", DummyIntType)

    ds = type("DummyDS", (), {})()
    ds.columns = ["F", "SIGF"]
    ds.dtypes = {"F": object(), "SIGF": object()}

    with pytest.raises(ValueError):
        xtr_maps.find_rfree_column(ds)


def test_find_rfree_column_multiple_candidates_picks_first(monkeypatch):
    class DummyIntType:
        pass

    monkeypatch.setattr(xtr_maps.rs, "MTZIntDtype", DummyIntType)

    ds = type("DummyDS", (), {})()
    ds.columns = ["Rfree1", "Rfree2", "F"]
    ds.dtypes = {
        "Rfree1": DummyIntType(),
        "Rfree2": DummyIntType(),
        "F": object(),
    }

    assert xtr_maps.find_rfree_column(ds) == "Rfree1"


def test_save_to_folder_copies_inputs_and_collects_filelocs(monkeypatch, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for fname in ["dark.pdb", "trig.pdb", "dark.mtz", "trig.mtz", "diff.mtz"]:
        (src / fname).write_text(fname, encoding="utf-8")

    copied = []
    monkeypatch.setattr(xtr_maps.shutil, "copy", lambda src, dst: copied.append((Path(src).name, Path(dst))))

    emitted = []

    def fake_save_extrapolated_map(*args, **kwargs):
        emitted.append(kwargs.get("name_prefix"))
        folder = args[4]
        return str(folder / f"{kwargs['name_prefix']}.mtz")

    monkeypatch.setattr(xtr_maps, "save_extrapolated_map", fake_save_extrapolated_map)

    params = {
        "folder": str(tmp_path / "out"),
        "xtr_prefix": "job",
        "diffmap_prefix": "_d",
    }
    input_cfg = {
        "pdb_dark": str(src / "dark.pdb"),
        "pdb_triggered": str(src / "trig.pdb"),
        "map_dark": str(src / "dark.mtz"),
        "map_triggered": str(src / "trig.mtz"),
        "map_diff": str(src / "diff.mtz"),
    }

    filelocs = xtr_maps.save_to_folder(
        diffmap=None,
        map_dark=None,
        parameters=params,
        input_file_config=input_cfg,
        save_dict={"a": 1.2, "b": 2.3},
    )

    out_dir = tmp_path / "out"
    assert out_dir.exists()
    assert len(copied) == 5
    assert emitted == ["job_a", "job_b"]
    assert len(filelocs) == 2
    assert filelocs[0].endswith("job_a.mtz")


def test_save_to_folder_raises_for_non_directory_target(tmp_path):
    out_file = tmp_path / "not_a_dir"
    out_file.write_text("x", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        xtr_maps.save_to_folder(
            diffmap=None,
            map_dark=None,
            parameters={"folder": str(out_file), "xtr_prefix": "x"},
            input_file_config={},
            save_dict={},
        )
