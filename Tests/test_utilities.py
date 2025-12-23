from pathlib import Path
import os
import duckdb
import pytest
import pandas as pd

repo = Path("/home/runner/work/open-urbex/open-urbex")
os.chdir(repo)

from IO.Scripts.Modules.utilities import (  # noqa
    create_dd_con,
    delete_file_if,
    move_pathlib,
)


def test_create_dd_con():
    # test type
    c = create_dd_con()
    assert isinstance(c, duckdb.DuckDBPyConnection)

    with pytest.raises(TypeError):
        c = create_dd_con("FRED")


def make_test_file(outfp, name):
    if not outfp.exists():
        Path(outfp).mkdir(parents=True, exist_ok=True)
    data = {"calories": [420, 380, 390], "duration": [50, 40, 45]}
    df = pd.DataFrame(data)
    df.to_csv(outfp / f"{name}.csv")
    return outfp / f"{name}.csv"


def test_delete_file_if():
    outfp = repo / "Tests" / "Test_Data" / "test_utilities" / "delete_file_if"
    make_test_file(outfp, "data")
    # false, real file
    fr = delete_file_if(outfp / "data.csv", False)
    assert "delete not selected." in fr
    assert Path(outfp / "data.csv").exists()
    # true, real file
    tr = delete_file_if(outfp / "data.csv", True)
    assert not Path(outfp / "data.csv").exists()
    assert tr is None
    # false, fake file
    ff = delete_file_if(outfp / "data34.csv", False)
    assert "does not exist" in ff
    assert not Path(outfp / "data34.csv").exists()
    # true, folder
    with pytest.raises(IsADirectoryError):
        delete_file_if(outfp, True)


def test_move_pathlib():
    # test if file exists in new and old locs
    startfp = repo / "Tests" / "Test_Data" / "test_utilities" / "delete_file_if"
    endfp = repo / "Tests" / "Test_Data" / "test_utilities" / "out"

    if startfp.exists():
        for x in startfp.rglob("*"):
            x.unlink()
    if endfp.exists():
        for x in endfp.rglob("*"):
            x.unlink()

    a = make_test_file(startfp, "start")
    b = make_test_file(endfp, "end")

    move_pathlib(a, endfp / "start.csv")
    assert not a.exists()
    assert Path(endfp / "start.csv").exists()

    with pytest.raises(FileNotFoundError):
        move_pathlib(a, endfp / "start.csv")

    # if in/out same path
    move_pathlib(b, endfp / "end.csv")
    assert b.exists()
    assert Path(endfp / "end.csv").exists()
