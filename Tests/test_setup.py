import os  # noqa
import math  # noqa
from pathlib import Path  # noqa
import pandas as pd
import geopandas as gpd  # noqa
import requests  # noqa
from zipfile import ZipFile  # noqa
from shapely.geometry import Polygon  # noqa
import shutil  # noqa
import tempfile
import random
from pyproj import CRS
import pytest
import pyogrio

repo = Path("/home/runner/work/open-urbex/open-urbex")
os.chdir(repo)

from IO.Scripts.Modules.setup import (  # noqa
    dwnld_import,  # noqa
    local_import,  # noqa
    create_extents,  # noqa
    wgs84_to_utm,  # noqa
    folder_set_up,  # noqa
    test_set_up,  # noqa
)  # noqa


# Required Files:
"""
In: "Tests" / "Test_Data"
/ test_setup
    / out
"""

base_path = repo / "Tests" / "Test_Data" / "test_setup"
out_path = base_path / "out"

# def test_dwnld_import():
#     dwnld_import()


def test_local_import():
    in_path = base_path / "local_import"
    uc_local = Path(
        in_path,
        "GHS_UCDB_THEME_GEOGRAPHY_GLOBE_R2024A_V1_0",
        "GHS_UCDB_THEME_GEOGRAPHY_GLOBE_R2024A.gpkg",
    )
    cit = random.choice(range(1, 11686))
    uc = local_import(cit, uc_local)
    assert len(uc) == 1

    uc = local_import(1234567, uc_local)
    assert len(uc) == 0

    with pytest.raises(pyogrio._io.DataSourceError):  # type: ignore
        uc = local_import(cit, in_path)


# def test_create_extents():
#     create_extents()


def test_wgs84_to_utm():
    y = random.choice(range(-90, 90))
    x = random.choice(range(-180, 180))
    df = pd.DataFrame({"id": ["fred"], "long": [x], "lat": [y]})
    geo = gpd.points_from_xy(df.long, df.lat, crs="EPSG:4326")
    ggg = gpd.GeoDataFrame(df, geometry=geo)
    UTMZone, wkid = wgs84_to_utm(ggg)  # type: ignore
    assert UTMZone[-1:] in ["S", "N"]
    assert 0 <= int(UTMZone[:-1]) <= 601
    assert isinstance(CRS.from_user_input(wkid), CRS)
    bigstr = CRS.from_user_input(wkid).to_wkt(pretty=True)
    assert "UTM zone" in bigstr

    # test dif crs change pt loc (switch lat/long)
    geo2 = gpd.points_from_xy(df.lat, df.long, crs=3857)
    ggg2 = gpd.GeoDataFrame(df, geometry=geo2)
    UTMZone2, wkid2 = wgs84_to_utm(ggg2)  # type: ignore
    assert UTMZone != UTMZone2
    assert wkid != wkid2

    # test dif crs same loc
    ggg3 = ggg.to_crs(3857)
    UTMZone3, wkid3 = wgs84_to_utm(ggg3)  # type: ignore
    assert UTMZone == UTMZone3
    assert wkid == wkid3


def test_folder_set_up():
    temp_dir = Path(tempfile.mkdtemp())
    outfp, fpdict = folder_set_up(temp_dir, "Test_City", "Test_Country")
    for fp in fpdict:
        fpx = Path(fpdict[fp], "test.txt")
        open(fpx, "w").close()
        assert Path(fpdict[fp]).exists()
        assert Path(outfp).exists()


def test_test_set_up():
    outfp, fpdict = test_set_up()
    print([x for x in Path(outfp).glob("*")])
    for fp in fpdict:
        fpx = Path(fpdict[fp], "test.txt")
        open(fpx, "w").close()
        assert Path(fpdict[fp]).exists()
        assert Path(outfp).exists()


# def test_is_req_file_check():
#     is_req_file_check()
