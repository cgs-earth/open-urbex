from IO.Scripts.Modules.utilities import warning_handler
from IO.Scripts.Modules.vectors import vec_in
import numpy as np
from numpy import copy
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio import errors, features
from rasterio.transform import Affine, xy
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.crs import CRS
from rasterio.shutil import delete as rdel
from rasterio.stack import stack
from shapely.geometry import shape
from sklearn.neighbors import KernelDensity
from scipy import ndimage
from scipy.interpolate import griddata
from pathlib import Path
import xarray as xr
from xrspatial.focal import focal_stats
from xrspatial.convolution import circle_kernel
from xrspatial.zonal import stats, regions
import osgeo_utils.gdal_merge


def run_kde(
    pts: gpd.GeoDataFrame,
    xmin: np.float64,
    ymin: np.float64,
    xmax: np.float64,
    ymax: np.float64,
    bw: float = 0.005,
):
    """
    Create Grid, Put Points into Grid, Estimate Kernel Density,
    Reshape Estimated Data to Grid.

    Parameters
    ----------
    pts : geopandas geodataframe of points
    xmin : numpy.float64, minimum x value of bounding box
    xmax : numpy.float64, maximum x value of bounding box
    ymin : numpy.float64, minimum y value of bounding box
    ymax : numpy.float64, maximum y value of bounding box
    bw : float, default=0.005 - bandwidth for kernel

    Returns
    -------
    eval : numpy mesh grid, evaluated model
    xct : x cell count (in grid) (width)
    yct : y cell count (in grid) (height)
    """
    # Get X and Y coordinates of places points
    xpts = pts["geometry"].x
    ypts = pts["geometry"].y
    # Create a cell mesh grid
    # Horizontal and vertical cell counts should be the same
    xct, yct = np.mgrid[xmin:xmax:2500j, ymin:ymax:2500j]
    # Create 2-D array of the coordinates (paired) of each cell in the mesh
    pos_xy = np.vstack([xct.ravel(), yct.ravel()]).T
    # Create 2-D array of the coordinate values of the points
    xy_coords = np.vstack([xpts, ypts]).T
    # Get kernel density estimator (can change parameters as desired)
    kde = KernelDensity(
        bandwidth=bw, metric="euclidean", kernel="cosine", algorithm="auto"
    )
    # Fit kernel density estimator to coordinates
    kde.fit(xy_coords)
    # Evaluate the estimator on coordinate pairs
    eval = np.exp(kde.score_samples(pos_xy))
    # Reshape the data to fit mesh grid
    eval = eval.reshape(xct.shape)
    return eval, xct, yct


def export_kde_raster(
    kde_out: np.ndarray,
    XX: np.ndarray,
    YY: np.ndarray,
    xmin: np.float64,
    ymin: np.float64,
    xmax: np.float64,
    ymax: np.float64,
    proj: int,
    filename: str | Path,
) -> None:
    """
    Export and save a kernel density raster.

    Parameters
    ----------
    kde_out : numpy mesh grid, fit model
    XX : x cell count (in grid) (width)
    YY : y cell count (in grid) (height)
    xmin : numpy.float64, minimum x value of bounding box
    xmax : numpy.float64, maximum x value of bounding box
    ymin : numpy.float64, minimum y value of bounding box
    ymax : numpy.float64, maximum y value of bounding box
    proj : int, crs
    filename : str or PurePath, output file path

    Returns
    -------
    N/A
    """
    if ".tif" not in str(filename):
        raise TypeError("Rasterio can only write out correctly to .tif files!")
    # Flip array vertically and rotate 270 degrees
    kde_out = np.rot90(np.flip(kde_out, 0), 3)
    # Get resolution
    xres = (xmax - xmin) / len(XX)
    yres = (ymax - ymin) / len(YY)
    # Set transform
    transform = Affine.translation(
        float(xmin - xres / 2), float(ymin - yres / 2)
    ) * Affine.scale(xres, yres)

    # Export array as raster
    with rasterio.open(
        filename,
        mode="w",
        driver="GTiff",
        height=kde_out.shape[0],
        width=kde_out.shape[1],
        count=1,
        dtype=kde_out.dtype,
        crs=proj,
        transform=transform,
        nodata=-9999,
    ) as new_dataset:
        new_dataset.write(kde_out, 1)


def kde_to_city_center(
    kde_places,
    xct,
    yct,
    city_name: str,
    country_name: str,
    xmin: np.float64,
    ymin: np.float64,
    xmax: np.float64,
    ymax: np.float64,
    cc_one: bool = True,
) -> gpd.GeoDataFrame:
    """
    Calculate City Center Point from KDE of POI data.

    Parameters
    ----------
    kde_places : output of run_kde
    xct : x coordinate value for every cel
    yct : y coordinate value for every cel
    city_name : str
    country_name : str
    xmin : numpy.float64, minimum x value of bounding box
    xmax : numpy.float64, maximum x value of bounding box
    ymin : numpy.float64, minimum y value of bounding box
    ymax : numpy.float64, maximum y value of bounding box

    Returns
    -------
    cc_gdf : geopandas geodataframe, city center point(s)
    """
    # Get resolution
    xres = (xmax - xmin) / len(xct)
    yres = (ymax - ymin) / len(yct)
    # Set transform
    trnsfrm = Affine.translation(
        float(xmin - xres / 2), float(ymin - yres / 2)
    ) * Affine.scale(xres, yres)
    # calculate point
    cc_indices = np.where(kde_places == kde_places.max())
    cc_coords = {"lat": [], "lon": []}
    for p in range(0, len(cc_indices[0])):
        ltln = xy(trnsfrm, cc_indices[1][p], cc_indices[0][p])
        cc_coords["lon"].append(ltln[0])
        cc_coords["lat"].append(ltln[1])

    cc_df = pd.DataFrame(
        {
            "city": f"{city_name}, {country_name}",
            "longitude": cc_coords["lon"],
            "latitude": cc_coords["lat"],
        }
    )
    geom = gpd.points_from_xy(cc_df.longitude, cc_df.latitude, crs="EPSG:4326")
    cc_gdf = gpd.GeoDataFrame(cc_df, geometry=geom)
    if cc_one is True:
        cc_gdf = cc_gdf.loc[[0]]

    return cc_gdf


def distance_accumulation(
    inrast: str | Path, outrast: str | Path
) -> None:  # TODO: Write Test!
    """
    Calculate distance from values for each cell.

    Parameters
    ----------
    inrast : str or PurePath, input file location
    outrast : str or PurePath, output file location

    Returns
    -------
    N/A
    """
    rast = rasterio.open(inrast)
    ary = rast.read(1)

    # reverse features in raster (0 is location)
    ary[ary > 0] = 2
    ary[ary == 0] = 1
    ary[ary == 2] = 0

    distacc = ndimage.distance_transform_edt(ary)
    rast.close()

    raster_out(distacc, outrast, inrast)


def vec_2_rast(
    vec: str | Path | gpd.GeoDataFrame, exrast: str | Path, outrast: str | Path
) -> np.ndarray:
    """
    Transform a vector dataset to a raster dataset.

    Parameters
    ----------
    vec : str / Path object / GeoPandas GeoDataFrame
        Input vector dataset.
    exrast : str / Path object
        The example raster dataset file path to derive parameters from.
    outrast : str / Path object
        The file path to export the new raster dataset to.

    Returns
    -------
    rasterized : numpy array
        Shapes reflected in cel values.
    """
    if isinstance(vec, (Path, str)):
        locs = vec_in(vec)
    else:
        locs = vec
    # Get list of geometries for all features in vector file
    geom = [shapes for shapes in locs.geometry]

    samprast = rasterio.open(exrast)
    rasterized = features.rasterize(
        geom,
        out_shape=samprast.shape,
        fill=-9999,
        out=None,
        transform=samprast.transform,
        all_touched=True,
        default_value=1,
        dtype=rasterio.int32,
    )
    samprast.close()

    raster_out(rasterized, outrast, exrast)

    return rasterized


def tif_2_ascii(in_fp: str | Path, out_fp: str | Path) -> str:
    """
    Convert TIF raster file to ASCII file in format
    required for MaxEnt.

    Parameters
    ----------
    in_fp : str or PurePath, TIF file path
    out_fp : str or PurePath, ASCII output

    Returns
    -------
        : str, message if success or if need different format
    """
    tif = rasterio.open(in_fp)
    arr = np.nan_to_num(tif.read(1), nan=0)
    if ".asc" in str(out_fp):
        with rasterio.open(
            out_fp,
            "w",
            driver="AAIGrid",
            height=arr.shape[0],
            width=arr.shape[1],
            count=1,
            dtype=arr.dtype,
            crs=tif.crs,
            transform=tif.transform,
            nodata=-9999,
            force_cellsize=True,
        ) as dst:
            dst.write(arr, 1)
        return f"{in_fp} Converted to {out_fp}"
    else:
        return "Output File Path Must End in .asc"


def coreg_raster(
    infile: str | Path,
    match: str | Path,
    outfile: str | Path,
    resample_method=Resampling.nearest,
) -> None:
    """
    Reproject a file to match the shape and projection of existing raster.

    Parameters
    ----------
    infile : (string) path to input tif file to reproject
    match : (string) path to raster with desired shape and projection
    outfile : (string) path to output file tif
    resample_method : (rasterio.enums.Resampling class object)
        see: https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling  # noqa
            for more method options - defaults to nearest

    Returns
    -------
    N/A

    Source: https://pygis.io/docs/e_raster_resample.html July 8th, 2025
    """
    # open input
    with rasterio.open(infile) as src:

        # open input to match
        with rasterio.open(match) as matche:

            # calculate the output transform matrix
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs,  # input CRS
                matche.crs,  # output CRS
                matche.width,  # input width
                matche.height,  # input height
                *matche.bounds,  # unpacks input outer boundaries
                # (left, bottom, right, top)
            )
            dst_crs = matche.crs

        # set properties for output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update(
            {
                "crs": dst_crs,
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height,
                "nodata": -9999,
            }
        )
        print(
            "Coregistered to shape:", dst_height, dst_width, "\n Affine", dst_transform
        )
        # open output
        with rasterio.open(outfile, "w", **dst_kwargs) as dst:
            # iterate through bands and write using reproject function
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resample_method,
                )


def node_dist_grid(
    nodes: gpd.GeoDataFrame,
    xmin: np.float64,
    ymin: np.float64,
    xmax: np.float64,
    ymax: np.float64,
) -> tuple:
    """
    Interpolate node distance to city center over a grid.

    Parameters
    ----------
    nodes : GeoPandas GeoDataFrame
        Point dataset
    xmin : numpy.float64, minimum x value of bounding box
    xmax : numpy.float64, maximum x value of bounding box
    ymin : numpy.float64, minimum y value of bounding box
    ymax : numpy.float64, maximum y value of bounding box

    Returns
    -------
    grid_z2 : numpy array
        Grid of point locations with point values.
    grid_x : numpy array
        X Coordinates of each point
    grid_y : numpy array
        Y Coordinates of each point
    xy_coords : numpy array
        Point Coordinates in XY of each point
    """
    grid_x, grid_y = np.mgrid[xmin:xmax:5000j, ymin:ymax:5000j]

    # Get X and Y coordinates of places points
    xpts = nodes["geometry"].x
    ypts = nodes["geometry"].y

    # Create 2-D array of the coordinate values of the points
    xy_coords = np.vstack([xpts, ypts]).T

    grid_z2 = griddata(
        xy_coords,
        nodes["cc_dist"].values,
        (grid_x, grid_y),
        method="linear",
        fill_value=0,
    )

    return grid_z2, grid_x, grid_y, xy_coords


def array_2_tif(
    grid: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    fdict: dict[str, Path],
    city_name: str,
    xmin: np.float64,
    ymin: np.float64,
    xmax: np.float64,
    ymax: np.float64,
) -> Path:  # TODO: Write Test!
    """
    Export Numpy Array (representing raster data) to a TIF.

    Parameters
    ----------
    grid : numpy array
        Grid of point locations with point values.
    grid_x : numpy array
        X Coordinates of each point
    grid_y : numpy array
        Y Coordinates of each point
    fdict : dict, dictionary of file paths
    city_name : str, name of city
    xmin : numpy.float64, minimum x value of bounding box
    xmax : numpy.float64, maximum x value of bounding box
    ymin : numpy.float64, minimum y value of bounding box
    ymax : numpy.float64, maximum y value of bounding box

    Returns
    -------
    out_fp : PurePath, location of TIF file
    """
    # Prepare array data for export
    filename = fdict["Intermediate"] / f"{city_name}_nodes2cc_dist_idw.tif"

    grid_out = np.rot90(np.flip(grid, 0), 3)

    # Get resolution
    xres = (xmax - xmin) / len(grid_x)
    yres = (ymax - ymin) / len(grid_y)

    # Set transform
    transform = Affine.translation(
        float(xmin - xres / 2), float(ymin - yres / 2)
    ) * Affine.scale(xres, yres)

    # replace no data with -9999 (nan --> -9999)
    grid_out = np.nan_to_num(grid_out, nan=-9999)

    # Export array as raster
    with rasterio.open(
        filename,
        mode="w",
        driver="GTiff",
        height=grid_out.shape[0],
        width=grid_out.shape[1],
        count=1,
        dtype=grid_out.dtype,
        crs=4326,
        transform=transform,
        nodata=-9999,
    ) as new_dataset:
        new_dataset.write(grid_out, 1)

    # co-register idw raster to match elevation raster
    out_fp = fdict["Model_Inputs"] / f"{city_name}_nodes2cc_dist_idw_coreg.tif"
    reference_fp = fdict["Downloads"] / f"{city_name}_elevation.tif"  # must be 4326
    coreg_raster(filename, reference_fp, out_fp)

    return out_fp


def get_resolution_in_meters(
    rast: str | Path, outfp: str | Path, wkid: str | int, scrs: str = "EPSG:4326"
) -> int | float:
    """
    Calculate the area of a raster cel in meters.
    Note: Destination CRS (wkid) must be have units that are meters
    or this will just return the area of a cell in whatever unit the
    destination crs is in.

    Parameters
    ----------
    rast : str / Path object
        Input Raster dataset file path
    outfp : str / Path object
        Output Re-Projected dataset file path
    wkid : str
        Well Known ID for UTM Zone
    scrs : str
        Source CRS (geographic projection)

    Returns
    -------
    h * w : numeric
        cel height times weight = area
    """
    rcrs = CRS.from_user_input(wkid)
    if rcrs.is_projected:
        if rcrs.linear_units in ["m", "meter", "metre"]:
            print("Units are Meters in Projected Coordinate System.")
        else:
            raise errors.CRSError(
                "Use a coordinate system that has meters as its unit."
            )
    else:
        raise errors.CRSError("Use a Projected Coordinate System.")
    if CRS.from_user_input(scrs) == CRS.from_user_input(wkid):
        wkid_rast = rasterio.open(rast)  # open original
    else:
        orig_rast = rasterio.open(rast)
        reparry, aff = reproject(
            source=rasterio.band(orig_rast, 1),
            src_transform=orig_rast.transform,
            src_crs=scrs,
            dst_crs=wkid,
            src_nodata=-9999,
            dst_nodata=-9999,
        )

        with rasterio.open(
            outfp,
            "w",
            driver="GTiff",
            crs=wkid,
            transform=aff,
            dtype=rasterio.float64,
            count=1,
            width=reparry.shape[2],
            height=reparry.shape[1],
            nodata=-9999,
        ) as dst:
            dst.write(reparry[0], indexes=1)

        wkid_rast = rasterio.open(outfp)  # open reprojected raster

    h, w = wkid_rast.res
    return h * w


def basic_point_density(
    points_fp: str | Path | gpd.GeoDataFrame,
    exrast_fp: str | Path,
    outrast_fp: str | Path,
    arm: int | float,
    fld1: str | None = None,
    k: np.ndarray = circle_kernel(1, 1, 3),
) -> None:  # TODO: Write test!
    """
    Calculate the Point Density (by neighborhood) per cel.

    Parameters
    ----------
    points_fp : str / Path object / gpd.GeoDataFrame
        File path for point dataset (shapefile etc.)
    exrast_fp : str / Path object
        Sample raster file path (for export params)
    outrast_fp : str / Path object
        Output raster file path
    arm : numeric
        area of cel in meters
    fld1 : str (default None)
        field name for points column containing weighting values
    k : numpy array
        representing neighborhood kernel
        defaults to circle_kernel(1, 1, 3)

    Returns
    -------
    N/A
    """
    if isinstance(points_fp, (str, Path)):
        points = vec_in(points_fp)
    elif not isinstance(points_fp, gpd.GeoDataFrame):
        raise TypeError(
            "Points must be a file path pointing to a spatial file \
                or a GeoPandas GeoDataFrame."
        )
    else:
        points = points_fp

    if fld1 not in points.columns:
        warning_handler(
            "Requested Field not in Point Dataset Columns.\
                Creating new field and weighting all points as 1."
        )
        fld1 = "one"
        points[fld1] = 1

    pts = list(zip(points.geometry, points[fld1]))
    # sum of bldg area per cell
    samprast = rasterio.open(exrast_fp)
    br = features.rasterize(
        shapes=pts,
        out_shape=samprast.shape,
        transform=samprast.transform,
        merge_alg=rasterio.enums.MergeAlg.add,  # type: ignore
        fill=0,  # no data is 0 here, focal stats treats as data
    )
    samprast.close()

    # focal stats - sum of cell value
    fs = focal_stats(xr.DataArray(br), k, stats_funcs=["sum"])
    # divide each cell by neighborhood size (in sqm)
    fs.values = fs.values / (arm * sum(sum(k)))  # type: ignore
    # reset no data value
    fs.values[fs.values == 0] = -9999

    raster_out(fs[0], outrast_fp, exrast_fp)


def rescale_raster(
    i_fp: str | Path, o_fp: str | Path, scale_factor: int | float
) -> None:  # TODO: Write Test!
    """
    Rescale a raster by a factor and write out data to a new dataset.
    Uses bilinear sampling to resample and the factor to change
    the resolution. To lower the resolution (bigger cels),
    the scale_factor should be < 1, to increase the resolution
    (smaller cells), the scale_factor should be > 1.

    Parameters
    ----------
    i_fp : str or Path object (input file path)
    o_fp : str or Path object (output file path)
    scale_factor : numeric (int or float)

    Returns
    -------
    """
    with rasterio.open(i_fp) as dataset:
        # calculate new height and width
        h = int(dataset.height * scale_factor)
        w = int(dataset.width * scale_factor)

        # resample data to target shape
        data = dataset.read(
            out_shape=(dataset.count, h, w), resampling=Resampling.bilinear
        )

        # scale image transform
        t = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]), (dataset.height / data.shape[-2])
        )

        with rasterio.open(
            o_fp,
            "w",
            driver="GTiff",
            crs=dataset.crs,
            transform=t,
            dtype=rasterio.float64,
            count=1,
            width=w,
            height=h,
            nodata=-9999,
        ) as dst:
            dst.write(data[0], indexes=1)


def raster_out(
    rast: (
        np.ndarray
        | xr.DataArray
        | rasterio.io.DatasetReader  # type: ignore
        | rasterio.io.DatasetWriter  # type: ignore
    ),
    ofp: str | Path,
    exfp: str | Path,
    rty: str = rasterio.float64,
    nd: int = -9999,
    drvr: str = "GTiff",
) -> None:  # TODO: Write Test!
    """
    Write raster dataset out to file based on an example raster's properties.

    Parameters
    ----------
    rast : np.array / xarray / rasterio object
        Data to be written out on disk.
    ofp : str or Path object
        File Path to write file out to.
    exfp : str or Path object
        File path of example raster to derive file properties from.
    rty : rasterio data type
        Defaults to rasterio.float64 but can be any of the following -
            rasterio.int16, rasterio.int32, rasterio.float32, etc.
    nd : int / float
        No Data value - defaults to -9999
    drvr : str
        Driver for reading/writing file type. Defaults to "GTiff".

    Returns
    -------
    N/A
    """
    if not isinstance(rast, np.ndarray):
        rast = np.array(rast)
    ex = rasterio.open(exfp)
    with rasterio.open(
        ofp,
        "w",
        driver=drvr,
        crs=ex.crs,
        transform=ex.transform,
        dtype=rty,
        count=1,
        width=ex.width,
        height=ex.height,
        nodata=nd,
    ) as dst:
        dst.write(rast, indexes=1)
    ex.close()


def zonal_stats(zones_fp: str | Path, values_fp: str | Path, sf: str = "max") -> tuple[
    np.ndarray,
    pd.DataFrame | xr.DataArray,
]:  # TODO: Write Test!
    """
    Calculate Zonal Statistics!

    Parameters
    ----------
    zones_fp : str / Path
        File path representing raster of zones.
    values_fp : str / Path
        File path representing raster of data to be calculated per zone.
    sf : str
        "stats_funcs" parameter for stats - one of the following options:
        ['mean', 'max', 'min', 'sum', 'std', 'var', 'count']

    Returns
    -------
    zs_zz : numpy array
        Zonal Stats Grid Output - zone represented by selected stat
    zs : pandas DataFrame
        Zonal Stats Table Output - zone and stats values
    """
    zz = rasterio.open(zones_fp).read(1)
    zz[zz == -9999] = np.NaN
    zz = xr.DataArray(zz)

    vv = rasterio.open(values_fp).read(1)
    vv[vv == -9999] = np.NaN
    vv = xr.DataArray(vv)

    zs = stats(zones=zz, values=vv)
    dzs = zs[["zone", sf]].set_index("zone").to_dict()  # type: ignore

    zs_zz = copy(zz)
    for z, m in dzs[sf].items():
        zs_zz[zz == z] = m

    zs_zz = np.nan_to_num(zs_zz, nan=-9999)

    return zs_zz, zs  # type: ignore


def rast_2_poly(
    rast_fp: str | Path, col_name: str, nv: int = -9999
) -> gpd.GeoDataFrame:
    """
    Raster dataset to polygons based on values.

    Parameters
    ----------
    rast_fp : str / Path object
        File path pointing to raster to convert to polygons.
    col_name : str
        Name of the column that you want the values to be represented in.
    nv : int (default = -9999)
        represents null value/no data

    Returns
    -------
    gdf : GeoPandas GeoDataFrame
        Polygonized output of input raster. In same crs as input raster.
    """
    with rasterio.open(rast_fp) as src:
        shapes = features.shapes(rasterio.band(src, 1), transform=src.transform)
        shplist = [s for s in shapes]
        cors = src.crs
    src.close()
    infset = []
    geo = []
    for g, i in shplist:
        if i != nv:
            infset.append(i)
            geo.append(shape(g))

    gdf = gpd.GeoDataFrame({col_name: infset, "geometry": geo}, crs=cors)
    return gdf


def region_group(
    data: str | Path | np.ndarray,
    out: str | Path,
    exrast: str | Path,
    n: int = 8,
    nv: int = -9999,
) -> np.ndarray:
    """
    Creates zones of touching cels of identical values -
    Intended to be an equivalent to Esri's Region Group tool.
    Writes out the raster to a tif as its needed later.

    Parameters
    ----------
    data : str / Path object OR numpy array
        Numpy Array Grid representing raster data
        or File Path to raster file with the data you want to region group.
    out_fp : str / Path object
        Output file path for region group raster.
    exrast : str / Path object
        Example raster for export information
            - can be the same as data if data input is also a file path
    n : int (default == 8, can only be 8 or 4)
    nv : int (default = -9999)

    Returns
    -------
    rgs : numpy array
        Array with the zone id as the value for each cel.
    """
    if isinstance(data, np.ndarray):
        d = data
    else:
        d = rasterio.open(data).read(1)
    if len(d[d == nv]) > 0:
        d[d == nv] = np.NaN
    dx = xr.DataArray(d)

    rgs = regions(raster=dx, neighborhood=n)
    rgs = np.nan_to_num(np.array(rgs), nan=nv)
    raster_out(rast=rgs, ofp=out, exfp=exrast, nd=nv)
    return rgs


def background_points(samp_rast: str | Path, out_shp: str | Path) -> gpd.GeoDataFrame:
    """
    Create background points from sample raster.

    Parameters
    ----------
    samp_rast : Path object or str
        File Path to sample raster (w/ crs and resolution)

    Returns
    -------
    bps : geopandas geodataframe (points)
    """
    with rasterio.open(samp_rast) as src:
        crs = src.crs
        # get dimenstions
        band = src.read(1)
        h = band.shape[0]
        w = band.shape[1]
        # create meshgrid
        cols, rows = np.meshgrid(np.arange(w), np.arange(h))
        xv, yv = xy(src.transform, rows, cols)
    src.close()

    bps = gpd.GeoDataFrame(
        {
            "building": 0,
            "x": xv,
            "y": yv,
            "geometry": gpd.GeoSeries.from_xy(xv, yv, crs=crs),
        },
        crs="EPSG:4326",
    )
    bps["id"] = bps.index + 1
    bps["species"] = "background"
    bps = bps.loc[::4]

    bps.to_file(out_shp)

    return bps


def samples_with_data(
    samples: str | Path, lnames: list[str], rstack_fp: str | Path, s_out_fp: str | Path
) -> None:
    """
    Use background points, sample points (species),
    and rasters to create sample-with-data format.
    Ex.
        Species, X, Y, Var1, Var2, Var3
        Blue-headed Vireo, 310186, 8243704, 1, 19.5, 0.91
        Blue-headed Vireo, 300243, 8173341, 2, 18.3, 1.04
        Loggerhead Shrike, 290434, 8192276, 4, 20.7, 0.88
    OR
        background, 320268, 8428840, 1, 17.5, 0.55
        background, 301886, 8432739, 2, 18.1, 0.65

    Parameters
    ----------
    samples : Path / str
        species sample or background points
    lnames : list of str
        raster layer names in raster stack
    rstack_fp : Path / str
        raster stack tif
    s_out_fp : Path / str
        Output filepath for sample data
    """
    s = vec_in(samples)
    if "x" not in s.columns.tolist() and "y" not in s.columns.tolist():
        s[["x", "y"]] = s.geometry.get_coordinates()
    s_coord_list = [(x, y) for x, y in zip(s["x"], s["y"])]

    with rasterio.open(rstack_fp) as rast:
        s[lnames] = [x for x in rast.sample(s_coord_list)]

    s.to_file(s_out_fp)


def stack_rasters(rdict: dict[str, str | Path], out_fp: str | Path) -> list[str]:
    """
    Combine one band raster files to create one output file with one band per raster.
    Rasters must have the same CRS, cel size, location, etc.
    e.g. identical in every way except cel values.

    Parameters
    -----------
    rdict : dictionary with
        keys = str
        values = Path obj (of raster file)
    out_fp : Path obj (or str)
        Output path for raster stack.


    Returns
    -------
    rnames : list of str
        Names of rasters in order of combined
    """
    stack(
        [rdict[x] for x in rdict],
        nodata=-9999,
        dst_path=out_fp,
    )
    rnames = [x for x in rdict]
    return rnames
