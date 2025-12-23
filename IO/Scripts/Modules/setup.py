from IO.Scripts.Modules.vectors import roads_dwnld_filt
from IO.Scripts.Modules.terrain import get_terrain_data
from IO.Scripts.Modules.rasters import vec_2_rast, distance_accumulation

import math
from pathlib import Path
import geopandas as gpd
import requests
from zipfile import ZipFile
from shapely.geometry import Polygon
import shutil
import numpy as np
import duckdb


def dwnld_import(ucid: int, outfp: str | Path) -> gpd.GeoDataFrame:
    """
    Download Global Human Settlement Layer data and filter to requested city.

    Parameters
    ----------
    ucid : int, id number for city
    outfp : str or PurePath, location to download data to

    Returns
    -------
    uc : geopandas geodataframe, GHS filtered to requested city
    """
    uc_local = Path(
        outfp,
        "GHS_UCDB_THEME_GEOGRAPHY_GLOBE_R2024A_V1_0",
        "GHS_UCDB_THEME_GEOGRAPHY_GLOBE_R2024A.gpkg",
    )
    uc_download = (
        "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata"
        + "/GHSL/GHS_UCDB_GLOBE_R2024A/GHS_UCDB_THEME_GLOBE_R2024A"
        + "/GHS_UCDB_THEME_GEOGRAPHY_GLOBE_R2024A/V1-0"
        + "/GHS_UCDB_THEME_GEOGRAPHY_GLOBE_R2024A_V1_0.zip"
    )

    if not Path(uc_local).exists():
        ziploc = str(Path(uc_local).parent) + ".zip"
        r = requests.get(uc_download, stream=True)
        with open(ziploc, "wb") as fd:
            for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)
        with ZipFile(ziploc, "r") as zobj:
            zobj.extractall(ziploc[:-4])
    uc = local_import(ucid, uc_local)
    return uc


def local_import(ucid: int, uc_local: str | Path) -> gpd.GeoDataFrame:
    """
    Filter GHS to requested city if data already downloaded.

    Parameters
    ----------
    ucid : int, city id
    uc_local : str or PurePath, where GHS data is downloaded

    Returns
    -------
    uc : geopandas geodataframe, filtered to requested city
    """
    uc = (
        gpd.read_file(
            uc_local,
            layer="GHS_UCDB_THEME_GEOGRAPHY_GLOBE_R2024A",
            columns=["geometry", "GC_UCN_MAI_2025", "GC_CNT_GAD_2025", "ID_UC_G0"],
        )
        .query("ID_UC_G0==@ucid")
        .to_crs("EPSG:4326")
    )
    uc["GC_UCN_MAI_2025"] = uc["GC_UCN_MAI_2025"].str.replace(" ", "_")
    uc["GC_CNT_GAD_2025"] = uc["GC_CNT_GAD_2025"].str.replace(" ", "_")
    return uc


def alt_poly(fp: Path | str, id: str, city: str, country: str) -> gpd.GeoDataFrame:
    """
    Use a geospatial file to create a fake FUA as basis for urbex analysis.

    Parameters
    ----------
    fp : Path or str
        input file path of vector data - one polygon, if more than one, dissolve
    id : str
        "fake id number"
    city : str
        Name of area
    country : str
        Name of country area is in

    Returns
    -------
    gdf : gpd.GeoDataFrame
    """
    if ".parquet" in str(fp):
        gdf = gpd.read_parquet(fp)
    else:
        gdf = gpd.read_file(fp)
    gdf["IC_UC_G0"] = id.replace(" ", "_")
    gdf["GC_UCN_MAI_2025"] = city.replace(" ", "_")
    gdf["GC_CNT_GAD_2025"] = country.replace(" ", "_")

    if len(gdf) > 1:
        gdf = gdf.dissolve("IC_UC_G0").reset_index()
    return gdf[["IC_UC_G0", "GC_UCN_MAI_2025", "GC_CNT_GAD_2025", "geometry"]]


def create_extents(shape: gpd.GeoSeries) -> tuple:  # TODO: Write Test!
    """
    Make a Bounding Box for Shape (city extent).

    Parameters
    ----------
    shape : geopandas geoseries (one shape)

    Returns
    -------
    extentPoly : shapely.geometry.polygon.Polygon, bounding box
    zoom : int, set to 12 to get lots of detail
    bounds : list of numpy.float64, represents corners of extentPoly
    xmin : numpy.float64, minimum x value of bounding box
    xmax : numpy.float64, maximum x value of bounding box
    ymin : numpy.float64, minimum y value of bounding box
    ymax : numpy.float64, maximum y value of bounding box
    """
    # make bounding box
    # Get the extent of the current feature
    extent = shape.total_bounds

    # Extract the min and max x/y coordinates
    xmin = extent[0] - 0.125
    xmax = extent[2] + 0.125
    ymin = extent[1] - 0.125
    ymax = extent[3] + 0.125

    # Set parameters
    zoom = 12  # Zoom level (1-14, higher means more detail)
    bounds = [xmin, ymin, xmax, ymax]  # [west, south, east, north]

    extentpoly = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
    return extentpoly, zoom, bounds, xmin, xmax, ymin, ymax


def wgs84_to_utm(geo: gpd.GeoSeries) -> tuple[str, str]:
    """
    Calculate the UTM Zone for a geometry that is spatially referenced.

    Parameters
    ----------
    geo : Geopandas GeoSeries, one shape

    Returns
    -------
    UTMZone : str
    wkid : int
    """
    geo = geo.to_crs("EPSG:4326")
    pt = geo.centroid.get_coordinates().iloc[0]
    longitude = pt["x"]
    latitude = pt["y"]

    # Calculate UTM zone (each zone is 6 degrees wide)
    adjusted_lon = (longitude + 180) % 360 - 180
    utm_zone = math.floor((adjusted_lon + 180) / 6) + 1

    # Calculate UTM Hemisphere
    if latitude >= 0:
        hemisphere = "N"
    else:
        hemisphere = "S"

    UTMZone = f"{utm_zone}{hemisphere}"
    wkid = 32600 + utm_zone if hemisphere == "N" else 32700 + utm_zone
    return UTMZone, wkid


def folder_set_up(out_path: str | Path, city_name: str, country_name: str) -> tuple:
    """
    Create folders required for urbex to create data
    and run maxent.

    Parameters
    ----------
    out_path : str or PurePath, parent folder location
    city_name : str, name of city
    country_name: str, name of country

    Returns
    -------
    outfp : PurePath, analysis parent path
    fpdict : dict, all subfolders required by urbex
        within outfp
    """
    outfp = Path(out_path) / country_name / city_name
    Path(outfp).mkdir(parents=True, exist_ok=True)
    fpdict = {}
    for f in [
        "Downloads",
        "Intermediate",
        "Model_Inputs",
        "Presence_Data",
        "Model_Outputs",
        "Model_Inputs_ASCII",
    ]:
        Path(outfp, f).mkdir(parents=True, exist_ok=True)
        fpdict[f] = outfp / f
    return outfp, fpdict



def test_set_up():
    """
    Create folders required for urbex to run unit tests.

    Parameters
    ----------

    Returns
    -------
    outfp : PurePath, analysis parent path
    fpdict : Dictionary of str: Path

    """
    outfp = Path("Tests", "Test_Data")
    Path(outfp).mkdir(parents=True, exist_ok=True)
    fpdict = {}
    for f in [
        "test_netx",
        "test_rasters",
        "test_setup",
        "test_terrain",
        "test_utilities",
        "test_vectors",
    ]:
        Path(outfp, f).mkdir(parents=True, exist_ok=True)
        Path(outfp, f, "out").mkdir(parents=True, exist_ok=True)
        fpdict[f] = outfp / f / "out"

    return outfp, fpdict
