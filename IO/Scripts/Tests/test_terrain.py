import os
from pathlib import Path
import random
import mercantile
import pytest
import rasterio
import pandas as pd
import tempfile
import geopandas as gpd  # noqa
import numpy as np

repo = Path("/home/runner/work/urbex/urbex/IO")
os.chdir(repo)

from IO.Scripts.Modules.terrain import (  # noqa
    decode_terrain_png,
    get_terrain_data,
    download_terrain_tile,
)
from IO.Scripts.Modules.setup import dwnld_import, create_extents  # noqa


# Required Files:
"""
In: "Scripts" / "Tests" / "Test_Data"
/ test_terrain
    / out
    / decode_terrain_png
        / Test_Random_Raster.tif
"""


def test_get_terrain_data():
    out_path = Path(repo / r"Scripts/Tests/Test_Data/test_terrain/out")
    cities = pd.read_table(
        Path(
            repo,
            "Data",
            "GHS_UCDB_THEME_GEOGRAPHY_GLOBE_R2024A_V1_0",
            "GHS_UCDB_THEME_GEOGRAPHY_GLOBE_R2024A.csv",
        ),
        delimiter=",",
    )["ID_UC_G0"].to_list()
    rcit = random.sample(cities, 3)
    for r in rcit:
        uc = dwnld_import(r, out_path)
        extentPoly, zoom, bounds, xmin, xmax, ymin, ymax = create_extents(
            uc["geometry"]
        )  # noqa
        city_name = uc["GC_UCN_MAI_2025"].tolist()[0]
        print(city_name)
        elvfp = out_path / f"{city_name}_elevation.tif"

        # good
        get_terrain_data(bounds, zoom, out_path, city_name, delfile=True)
        assert Path(elvfp).exists()

        # bad zoom?
        with pytest.raises(mercantile.InvalidZoomError):
            zoom = -5
            get_terrain_data(bounds, zoom, out_path, city_name, delfile=True)

        # test bad bounds
        zoom = 12
        with pytest.raises(TypeError):
            bounds2 = ["-11", 35, 0, 50]
            get_terrain_data(bounds2, zoom, out_path, city_name, delfile=True)

        # test bad output path
        with pytest.raises(TypeError):
            out_path = "fred"
            get_terrain_data(bounds, zoom, out_path, city_name, delfile=True)
        with pytest.raises(rasterio.RasterioIOError):
            out_path = Path("fred")
            get_terrain_data(bounds, zoom, out_path, city_name, delfile=True)


def test_decode_terrain_png():
    base_path = Path(repo / r"Scripts/Tests/Test_Data")
    # test good file
    out_path = Path(base_path / r"/test_terrain/out")
    zoom = 12
    cities = pd.read_table(
        Path(
            repo,
            "Data",
            "GHS_UCDB_THEME_GEOGRAPHY_GLOBE_R2024A_V1_0",
            "GHS_UCDB_THEME_GEOGRAPHY_GLOBE_R2024A.csv",
        ),
        delimiter=",",
    )["ID_UC_G0"].to_list()
    rcit = random.sample(cities, 3)
    temp_dir = Path(tempfile.mkdtemp())
    # Get list of tiles that cover the area
    for r in rcit:
        uc = dwnld_import(r, out_path)
        extentPoly, zoom, bounds, xmin, xmax, ymin, ymax = create_extents(
            uc["geometry"]
        )  # noqa
        tiles = list(
            mercantile.tiles(bounds[0], bounds[1], bounds[2], bounds[3], zooms=[zoom])
        )
        for tile in tiles:
            tile_path = temp_dir / f"{tile.x}_{tile.y}_{tile.z}.png"
            # Download tile
            if download_terrain_tile(tile.x, tile.y, tile.z, str(tile_path)):
                # Decode elevation data
                elevation = decode_terrain_png(str(tile_path))
        assert isinstance(elevation, np.ndarray)
        assert elevation.min() >= -440
        assert elevation.max() <= 8849

    # test bad file (only 1 band)
    tile_path = Path(
        base_path / r"test_terrain/decode_terrain_png/Test_Random_Raster.tif"
    )
    with pytest.raises(IndexError):
        decode_terrain_png(str(tile_path))
    # test absent file
    tile_path = Path(base_path / "fred.gpkg")
    with pytest.raises(rasterio.RasterioIOError):
        decode_terrain_png(str(tile_path))
