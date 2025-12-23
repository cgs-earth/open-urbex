from IO.Scripts.Modules.utilities import delete_file_if, create_dd_con
import numpy as np
from pathlib import Path
import pandas as pd
import geopandas as gpd
import osmnx as ox
import shapely
from arcgis.gis import GIS
from arcgis.features import FeatureLayer
from arcgis.geometry.filters import overlaps
import duckdb


def ox_trans_load(
    extentpoly: shapely.geometry.polygon.Polygon,
    transPath: str | Path,
    city_name: str,
    delfile: bool = False,
) -> None:  # TODO: Write Test!
    """
    Download transportation data from OpenStreetMap (OSM) with osmnx.
    Select only point data and export.

    Parameters
    ----------
    extentPoly : shapely.geometry.polygon.Polygon, clip extent
    transPath : str or PurePath, folder to write out to
    city_name : str, Name of analysis city
    delfile : True/False, default is False
        Whether you want to delete a previous
        version of the file if it exists.

    Returns
    -------
    None
    """
    outPath = Path(transPath) / f"{city_name}_trans.shp"
    delete_file_if(outPath, delfile=delfile)
    if not Path(outPath).exists():
        gdf = ox.features.features_from_polygon(
            polygon=extentpoly,
            tags={
                "public_transport": "station",
                "railway": ["stop", "halt", "station", "tram_stop"],
                "aerialway": "station",
                "highway": "bus_stop",
                "amenity": ["taxi", "ferry_terminal", "bus_station"],
            },
        )
        pointsgdf = gdf[gdf.geometry.geom_type == "Point"].copy()
        pointsgdf = pointsgdf.to_crs("EPSG:4326")
        pointsgdf.geometry = pointsgdf.geometry.centroid  # deal with multi points
        vec_out(pointsgdf, outPath, delfile)
    else:
        print(f"{outPath} was not deleted and recreated.")


def ovrtr_dwnld(
    dpath: str | Path,
    cols: str,
    fpath: str | Path,
    xmin: np.float64,
    xmax: np.float64,
    ymin: np.float64,
    ymax: np.float64,
    delfile: bool = False,
    quer: str | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> str | None:  # TODO: Write Test!
    """
    Read and export data from parquet to shapefile,
    filtered to bounding box. Parquet can be stored in cloud,
    like Overture: "azure://release/2024-09-18.0/theme=X/type=X/*"

    Parameters
    ----------
    dpath : str, Parquet Data Path (location)
    cols : str, in format: "x, y, z" representing columns to keep
    fpath : str or PurePath, output path
    xmin : numpy.float64, minimum x value of bounding box
    xmax : numpy.float64, maximum x value of bounding box
    ymin : numpy.float64, minimum y value of bounding box
    ymax : numpy.float64, maximum y value of bounding box
    delfile : True/False, default False
        Whether you want to delete a previous
        version of the file if it exists.
    quer : str, like "ST_GeometryType(geometry)='LINESTRING'"
        SQL, DuckDB flavor - WHERE clause
    con : DuckDB Connection Object

    Returns
    -------
        : str, download/delete success
    """

    delete_file_if(fpath, delfile=delfile)
    if not Path(fpath).exists():
        # create query & run
        if quer:
            quer = quer + " AND "
        else:
            quer = ""
        if not con:
            con = create_dd_con()
        rq = f"COPY \
            (SELECT {cols} FROM \
                read_parquet('{dpath}', filename=true) \
                WHERE {quer} \
                    bbox.xmin BETWEEN {xmin} AND {xmax} \
                    AND bbox.ymin BETWEEN {ymin} AND {ymax}) \
            TO '{fpath}' WITH (FORMAT GDAL, DRIVER 'ESRI Shapefile', \
                SRS 'EPSG:4326');"
        con.sql(rq)
        return f"Data Download Successful: {fpath}"
    else:
        return f"{fpath} was not deleted and recreated."


def ovrtr_memry(
    cols: str,
    dpath: str | Path,
    quer: str,
    xmin: np.float64,
    xmax: np.float64,
    ymin: np.float64,
    ymax: np.float64,
    con: duckdb.DuckDBPyConnection | None = None,
) -> gpd.GeoDataFrame:  # TODO: Write Test!
    """
    Read a parquet file into memory, filtered by bounding box.

    Parameters
    ----------
    cols : str, in format: "x, y, z" representing columns to keep
    dpath : str, Parquet Data Path (location)
        Parquet can be stored in cloud, like Overture:
        "azure://release/2024-09-18.0/theme=X/type=X/*"
    quer : str, like "ST_GeometryType(geometry)='LINESTRING'"
        SQL, DuckDB flavor - WHERE clause
    xmin : numpy.float64, minimum x value of bounding box
    xmax : numpy.float64, maximum x value of bounding box
    ymin : numpy.float64, minimum y value of bounding box
    ymax : numpy.float64, maximum y value of bounding box
    con : DuckDB Connection Object

    Returns
    -------
    gdf : Geopandas Geodataframe with 'geometry' column
    """
    if quer:
        quer = quer + " AND "
    else:
        quer = ""
    if not con:
        con = create_dd_con()
    rq = f"SELECT {cols}, ST_AsText(geometry) as wkt FROM \
        read_parquet('{dpath}', filename=true) \
        WHERE {quer} \
            bbox.xmin BETWEEN {xmin} AND {xmax} \
            AND bbox.ymin BETWEEN {ymin} AND {ymax};"
    dt = con.sql(rq).df()
    gdf = df_2_gdf(dt)
    return gdf


def df_2_gdf(
    df: pd.DataFrame, espg: str = "EPSG:4326"
) -> gpd.GeoDataFrame:  # TODO: Write Test!
    """
    Transform Pandas Dataframe with WKT column (named wkt)
    to a Geopandas GeoDataframe.

    Parameters
    -----------
    df : Pandas dataframe with wkt column
    espg : str, default="EPSG:4326", represents CRS

    Returns
    -------
    gdf : Geopandas Geodataframe with 'geometry' column
    """
    if len(df) > 0:
        shapes = gpd.GeoSeries.from_wkt(df.wkt)
    else:
        shapes = []
    gdf = gpd.GeoDataFrame(
        df.drop(["geometry", "wkt"], axis=1), geometry=shapes, crs=espg
    )
    return gdf


def fl_2_gdf(
    fl: FeatureLayer, cols: list[str], espg: str = "EPSG:4326"
) -> gpd.GeoDataFrame:  # TODO: Write Test!
    """
    ArcGIS (Esri) Feature Layer to Geopandas Geodataframe.

    Parameters
    ----------
    fl : arcgis.features.FeatureLayer object (Feature Layer)
    cols : list of str, columns to keep in output
    espg : str, default="EPSG:4326", represents CRS

    Returns
    -------
    gdf : Geopandas Geodataframe with 'geometry' column
    """
    if len(fl) > 0:  # type: ignore
        shapes = gpd.GeoSeries.from_wkt(fl.SHAPE.geom.WKT)  # type: ignore
    else:
        shapes = []
    gdf = gpd.GeoDataFrame(fl[cols], geometry=shapes, crs=espg)  # type: ignore
    return gdf


def filt_load_bldgs(
    dloc: str | Path,
    samp_out: str | Path | None,
    pt_out: str | Path | None,
    big_out: str | Path | None,
    cols: str,
    xmin: np.float64,
    xmax: np.float64,
    ymin: np.float64,
    ymax: np.float64,
    con: duckdb.DuckDBPyConnection | None = None,
    delfile: bool = False,
) -> str | None:  # TODO: Write Test!
    """
    Create a table in memory from parquet, filtered by bounding box.
    Depending on output desired, will execute queries to filter,
    modify, and export the required data. Specifically written
    with building footprint data from Overture in mind.

    Parameters
    ----------
    dloc : str, Parquet Data Path (location)
        Specifically designed for:
        "azure://release/2024-09-18.0/theme=buildings/type=building/*"
    samp_out : str or PurePath, for a sample of buildings as points
        used as input presence data for MaxEnt
    pt_out : str or PurePath, for all buildings as points
        used for informal settlement analysis
    big_out : str or PurePath, for buildings with area > 25 sqm
        used for old atlas calc (for comparison)
    cols : str, in format: "x, y, z" representing columns to keep
    xmin : numpy.float64, minimum x value of bounding box
    xmax : numpy.float64, maximum x value of bounding box
    ymin : numpy.float64, minimum y value of bounding box
    ymax : numpy.float64, maximum y value of bounding box
    con : DuckDB Connection Object
    delfile : True/False, default False
        Whether you want to delete a previous
        version of the file if it exists.

    Returns
    -------
        : str, message if files not created.
    """
    files = [f for f in [samp_out, pt_out, big_out] if f is not None]
    nf = []
    for f in files:
        delete_file_if(f, delfile=delfile)
        if not Path(f).exists():
            nf.append(f)

    if len(nf) > 0:
        if not con:
            con = create_dd_con()
        # read in all buildings in the bounding box
        con.sql(
            f"CREATE TEMP TABLE bldgs AS SELECT {cols}, \
                ROUND(ST_Area_Spheroid(ST_FlipCoordinates(geometry))::DOUBLE, 2) \
                    as area_sqm FROM read_parquet('{dloc}') \
                        WHERE (bbox.xmin > {xmin} AND bbox.xmax < {xmax} \
                            AND bbox.ymin > {ymin} AND bbox.ymax< {ymax});"
        )
        if big_out is not None and not Path(big_out).exists():
            # create big buildings polygons dataset
            con.sql(
                f"COPY(SELECT {cols}, area_sqm \
                    FROM bldgs WHERE area_sqm > 25) \
                    TO '{big_out}' WITH (FORMAT GDAL, \
                        DRIVER 'ESRI Shapefile', SRS 'EPSG:4326');"
            )
        if pt_out is not None and not Path(pt_out).exists():
            con.sql(
                f"COPY(SELECT area_sqm, \
                    {cols.replace(', geometry', '')}, \
                    ST_Centroid(geometry)::geometry \
                    as geometry FROM bldgs) TO '{pt_out}' \
                    WITH (FORMAT GDAL, \
                    DRIVER 'ESRI Shapefile', SRS 'EPSG:4326')"
            )
        if samp_out is not None and not Path(samp_out).exists():
            # create sample buildings point set
            con.sql("CREATE TEMP SEQUENCE id_seq START 1;")
            con.sql(
                "ALTER TABLE bldgs ADD \
                    COLUMN id INTEGER DEFAULT nextval('id_seq');"
            )
            leg = len(con.sql("SELECT id FROM bldgs;"))
            if leg > 1000000:
                con.sql("ALTER TABLE bldgs ADD COLUMN samp_ids INTEGER;")

                if leg < 10000000:
                    con.sql(
                        "UPDATE bldgs \
                            SET samp_ids = \
                            CAST(right(CAST(id AS VARCHAR), 1) AS INTEGER); \
                            DELETE FROM bldgs WHERE samp_ids != 5;"
                    )

                else:
                    con.sql(
                        "UPDATE bldgs SET samp_ids = \
                            CAST(right(CAST(id AS VARCHAR), 2) AS INTEGER); \
                            DELETE FROM bldgs WHERE samp_ids != 15;"
                    )
                con.sql("ALTER TABLE bldgs DROP samp_ids;")
            con.sql(
                f"COPY(SELECT area_sqm, {cols.replace(', geometry', '')}, \
                    ST_Centroid(geometry)::geometry \
                    as geometry FROM bldgs) TO '{samp_out}' \
                    WITH (FORMAT GDAL, \
                    DRIVER 'ESRI Shapefile', SRS 'EPSG:4326')"
            )
        con.sql("DROP TABLE bldgs;")
    else:
        print("Either no selected files or not able to delete selected files.")


def con_rds_filt(
    roadfp: str | Path, crs: int = 4326
) -> gpd.GeoDataFrame:  # TODO: Write Test!
    """
    Filter roads to just connected roads (remove roads that
    do not connect to other roads).

    Parameters
    ----------
    roadfp : str or PurePath, location of line data
    crs : int, default=4326

    Returns
    -------
    filt_rds : Geopandas Geodataframe, containing
        just lines representing connected roads
    """
    road_lines = vec_in(roadfp)
    road_lines["start_point"] = shapely.get_point(road_lines.geometry, 0)
    road_lines["end_point"] = shapely.get_point(road_lines.geometry, -1)

    starts = road_lines[["id", "class", "subtype", "start_point"]].rename(
        columns={"start_point": "geometry"}
    )
    ends = road_lines[["id", "class", "subtype", "end_point"]].rename(
        columns={"end_point": "geometry"}
    )
    rd_pts = pd.concat([starts, ends])
    rd_pts.crs = crs

    multipts = (
        rd_pts.groupby(by="geometry")["id"]
        .count()
        .reset_index()
        .query("id > 1")
        .rename(columns={"id": "id_count"})
    )
    multilist = pd.merge(rd_pts, multipts, how="left", on="geometry").query(
        "id_count > 1"
    )
    multilist = multilist["id"].to_list()

    filt_rds = road_lines[["id", "class", "subtype", "geometry"]].query(
        "id.isin(@multilist)"
    )
    return filt_rds


def water_dwnld_clean(
    city_name: str,
    out_folder: str | Path,
    xmin: np.float64,
    xmax: np.float64,
    ymin: np.float64,
    ymax: np.float64,
    wkid: str,
    oloc: str,
    relnum: str,
    con: duckdb.DuckDBPyConnection | None = None,
    delfile: bool = True,
) -> str | None:  # TODO: Write Test!
    """
    Runner function to download and combine multiple types of
    water data within a bounding box. Outputs all water as polygon
    layer.

    Parameters
    ----------
    city_name : str, name of city
    out_folder : str or PurePath, output folder
    xmin : numpy.float64, minimum x value of bounding box
    xmax : numpy.float64, maximum x value of bounding box
    ymin : numpy.float64, minimum y value of bounding box
    ymax : numpy.float64, maximum y value of bounding box
    wkid : str, represents UTM Zone (or projected CRS)
    oloc : str, represents overture location (for data)
    relnum : str, represents release number id
    con : DuckDB Connection Object
    delfile : True/False, default False
        Whether you want to delete a previous
        version of the file if it exists.

    Returns
    -------
        : str, location of data and success or not message

    """
    outPath = Path(out_folder) / f"{city_name}_all_water.shp"
    delete_file_if(outPath, delfile=delfile)
    if not Path(outPath).exists():

        # rivers/streams
        rsCols = "id, subtype, geometry"
        rsQuer = "ST_GeometryType(geometry)='LINESTRING'"
        rsDPath = f"{oloc}/release/{relnum}/theme=base/type=water/*"  # noqa
        rivers = ovrtr_memry(rsCols, rsDPath, rsQuer, xmin, xmax, ymin, ymax, con)

        # lakes etc
        lkCols = "id, subtype, geometry"
        lkQuer = "ST_GeometryType(geometry)='POLYGON'"
        lkDPath = f"{oloc}/release/{relnum}/theme=base/type=water/*"  # noqa
        lakes = ovrtr_memry(lkCols, lkDPath, lkQuer, xmin, xmax, ymin, ymax, con)

        # esri water
        ewCols = ["objectid", "name1", "type", "iso_cc"]
        ewDPath = "https://maps.nccs.nasa.gov/mapping/rest/services/base_layers/esri_world_water_bodies/FeatureServer/0"  # noqa
        agol = GIS()  # noqa
        query_extent: dict[str, np.float64 | dict[str, int]] = {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "spatialReference": {"wkid": 4326},
        }
        water = FeatureLayer(ewDPath).query(
            where="SHAPE__Area > 0.001",
            geometry_filter=overlaps(query_extent, sr=4326),  # type: ignore
            as_df=True,
        )
        water = fl_2_gdf(water, cols=ewCols, espg="EPSG:4326")  # type: ignore

        # buffer rivers
        rivers = rivers.to_crs(wkid)
        rivers["geometry"] = rivers["geometry"].buffer(5).to_crs("EPSG:4326")

        # merge water
        all_water_poly = pd.concat([rivers, lakes, water])
        vec_out(all_water_poly, outPath, delfile)
        print(str(outPath))

    else:
        print(f"{outPath} was not deleted nor recreated.")


def roads_dwnld_filt(
    fdict: dict[str, Path],
    city_name: str,
    xmin: np.float64,
    xmax: np.float64,
    ymin: np.float64,
    ymax: np.float64,
    oloc: str,
    relnum: str,
    con: duckdb.DuckDBPyConnection | None = None,
    delfile: bool = True,
) -> str | None:  # TODO: Write Test!
    """
    Runner function for road data - download and filter road data.

    Parameters
    ----------
    fdict : dict, contains PurePath of folder paths
    city_name : str, name of city
    xmin : numpy.float64, minimum x value of bounding box
    xmax : numpy.float64, maximum x value of bounding box
    ymin : numpy.float64, minimum y value of bounding box
    ymax : numpy.float64, maximum y value of bounding box
    oloc : str, represents overture location (for data)
    relnum : str, represents release number id
    con : DuckDB Connection Object
    delfile : True/False, default False
        Whether you want to delete a previous
        version of the file if it exists.

    Returns
    -------
        : str, location of data and success or not message
    """
    rdFPath = fdict["Downloads"] / f"{city_name}_rds.shp"

    rdCols = "id, class, subtype, geometry"
    rdQuer = "subtype='road' AND \
        (class = 'trunk' OR class = 'primary' OR class = 'secondary' \
            OR class = 'tertiary'OR class = 'residential' \
                OR class = 'unclassified' OR class = 'motorway')"
    rdDPath = f"{oloc}/release/{relnum}/theme=transportation/type=segment/*"  # noqa
    q = ovrtr_dwnld(
        rdDPath,
        rdCols,
        rdFPath,
        xmin,
        xmax,
        ymin,
        ymax,
        quer=rdQuer,
        con=con,
        delfile=delfile,
    )
    print(q)

    # filter out disconnected roads
    # deal with file overwrite
    conPath = fdict["Intermediate"] / f"{city_name}_con_rds.shp"
    delete_file_if(conPath, delfile=delfile)
    if not Path(conPath).exists():

        filt_rds = con_rds_filt(rdFPath, 4326)
        vec_out(filt_rds, conPath, delfile)
        print(f"{conPath} successfully created.")
    else:
        print(f"{conPath} was not deleted nor recreated.")


def places_dwnld(
    fdict: dict[str, Path],
    city_name: str,
    xmin: np.float64,
    xmax: np.float64,
    ymin: np.float64,
    ymax: np.float64,
    oloc: str | Path,
    relnum: str,
    con: duckdb.DuckDBPyConnection | None = None,
    delfile: bool = False,
) -> str | None:  # TODO: Write Test!
    """
    Runner function for downloading place data from Overture.

    Parameters
    ----------
    fdict : dict, contains PurePath of folder paths
    city_name : str, name of city
    xmin : numpy.float64, minimum x value of bounding box
    xmax : numpy.float64, maximum x value of bounding box
    ymin : numpy.float64, minimum y value of bounding box
    ymax : numpy.float64, maximum y value of bounding box
    oloc : str, represents overture location (for data)
    relnum : str, represents release number id
    con : DuckDB Connection Object
    delfile : True/False, default False
        Whether you want to delete a previous
        version of the file if it exists.

    Returns
    -------
        : str or PurePath, location where data is exported
    """
    plFPath = fdict["Downloads"] / f"{city_name}_plcs.shp"
    plCols = "id, categories.primary, brand.names.primary, geometry"
    plDPath = f"{oloc}/release/{relnum}/theme=places/type=*/*"  # noqa
    q = ovrtr_dwnld(
        plDPath, plCols, plFPath, xmin, xmax, ymin, ymax, con=con, delfile=delfile
    )
    print(q)


def points_2_csv(
    fp: str | Path,
    outfp: str | Path,
    swd: bool = False,
    species: str = "Building",
    layers: list = [],
    delfile: bool = False,
) -> str | None:  # TODO: Write Test!
    """
    Convert Point data to XY table and export as CSV.
    Specifically formatted to match MaxEnt import requirements
    for presence-only species data.

    Parameters
    ----------
    fp : str or PurePath, input point data file location
    outfp : str or PurePath, output location for csv
    swd : boolean (True/False)
        whether to set up with sample-with-data csv output or not
    species : str, default=Building - represents what is at location
    layers : list of str
        column names representing input rasters - keep
    delfile : True/False, default=False
        Whether you want to delete a previous
        version of the file if it exists.

    Returns
    -------
        : str, message saying data not exported
    """
    delete_file_if(outfp, delfile=delfile)
    if swd:
        if (species not in ["Building", None]) and (len(layers) > 0):
            raise Exception(
                "Please list layers and have your species column prepared \
                    with labels already."
            )
    else:
        if species is not None and len(layers) == 0:
            raise Exception(
                "Please confirm that you have a species parameter and no layers in the \
                    layer list parameter if you are just exporting sample data."
            )
    if not Path(outfp).exists():
        bldgs = vec_in(fp).to_crs(4326)
        bldgs[["Long", "Lat"]] = bldgs.get_coordinates()
        if "species" not in bldgs.columns:
            bldgs["species"] = species
        cols = ["species", "Long", "Lat"]
        if swd:
            for la in layers:
                if la not in bldgs.columns:
                    raise Exception(
                        "Strings in Layers list are not already in the point dataset."
                    )
            cols = cols + layers
        bldgs[cols].to_csv(outfp, index=False)
    else:
        return f"{outfp} was not deleted nor recreated."


def vec_out(data, fp: str | Path, delfile=True) -> str | None:
    """
    Write out vector files in either parquet or gpkg or shp formats.

    Parameters
    ----------
    data : geopandas dataframe
    fp : str / Path
        output file path
    delfile : boolean
        default True - for delete file if fxn

    Returns
    -------
    str noting failure or None
    """

    delete_file_if(fp, delfile)

    types = Path(fp).suffixes
    if len(types) > 0:
        tp = types[-1]
    else:
        return f"{str(fp)} has no path suffix."

    if tp in [".gpkg", ".shp"]:
        data.to_file(fp)
    elif tp == ".parquet":
        data.to_parquet(fp)
    else:
        return f"File type of {tp} not available - \
        please choose .parquet, .gpkg, or .shp"


def vec_in(fp: str | Path) -> gpd.GeoDataFrame:
    """
    Read vector data into geopandas geodataframe.

    Parameters
    ----------
    fp : str / Path

    Returns
    -------
    gdf : geopandas geodataframe
    """
    if Path(fp).is_file():
        tp = Path(fp).suffixes
        if len(tp) > 0:
            tp = tp[-1]
        else:
            raise ValueError(f"{str(fp)} has no path suffix.")

        if tp in [".gpkg", ".shp"]:
            gdf = gpd.read_file(fp)
            return gdf
        elif tp == ".parquet":
            gdf = gpd.read_parquet(fp)
            return gdf
        else:
            raise ValueError(
                f"File type of {tp} not available - \
            please choose .parquet, .gpkg, or .shp"
            )
    else:
        raise ValueError(f"File Path: {str(fp)} is not valid.")


def big_roads_only(
    fp: str | Path, fdict: dict[str, Path], city_name: str
) -> str | Path:
    """
    Filter OSM Roads to only four categories:
    1. Primary
    2. Secondary
    3. Tertiary
    4. Trunk
    The idea is to constrain residential roads that are not signifiers of density.

    Parameters
    ----------
    fp : str / Path
        path to the vector line file representing roads
        must have column "class"

    Returns
    -------
    out_fp : str / Path
        output file path
    """
    out_fp = fdict["Intermediate"] / f"{city_name}_big_rds.shp"

    rds = gpd.read_file(Path(fp))
    rds = rds.rename(columns={"class": "type"})
    rds = rds.query('type.isin(["primary", "secondary", "trunk", "tertiary"])')

    vec_out(rds, out_fp)
    return out_fp
