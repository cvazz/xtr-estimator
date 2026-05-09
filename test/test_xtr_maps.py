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
