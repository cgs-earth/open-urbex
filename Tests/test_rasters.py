import os
from pathlib import Path
import geopandas as gpd
import numpy as np
import pytest
import rasterio

repo = Path("/home/runner/work/urbex/urbex")
os.chdir(repo)
test_data_path = repo / "Tests" / "Test_Data" / "test_rasters"

from IO.Scripts.Modules.rasters import (  # noqa
    run_kde,
    kde_to_city_center,
    export_kde_raster,
    vec_2_rast,
    tif_2_ascii,
    coreg_raster,
    node_dist_grid,
    array_2_tif,
    get_resolution_in_meters,
    region_group,
    rast_2_poly,
    # raster_out,
    rescale_raster,
    # all_bldg_point_density,
    # basic_point_density,
    # zonal_stats,
    # distance_accumulation,
)

# Required Files:
# In: "Tests" / "Test_Data"
"""
In: "Tests" / "Test_Data"
/ test_rasters
    / array_2_tif
        / array_2_tiff_test_Points.shp
    / coreg_rast
        / Test_nodes.shp
        / Test_Random_Raster.tif
        / Test_Random_Raster_BigCell.tif
        / Test_Random_Raster_BigShape.tif
        / Test_Random_Raster_Nulls.tif
        / Test_Random_Raster_SmallCell.tif
        / Test_Random_Raster_SmallShape.tif
    / export_kde_rast
        / Test_nodes.shp
    / get_resolution_in_meters
        / Test_Random_Raster.tif
    / kde_to_city_center
        / Test_nodes.shp
    / node_dist_grid
        / Test_nodes.shp
        / Test_lines.shp
        / Test_Random_Raster.tif
    / rast2poly
        / Test_Rast2Poly_Six.tif
        / Test_Rast2Poly_Three.tif
    / raster_out
    / region_group
        / Test_elevation_3857.tif
        / Test_Random_Raster_RG.tif
        / Test_Random_Raster_RG_Example.tif
    / run_kde
        / Test_nodes.shp
        / Test_lines.shp
        / Test_multipoints.shp
        / Test_polygons.shp
    / tif2ascii
        / Test_Random_Raster.tif
        / Test_Random_Raster_Nulls.tif
    / vec2rast
        / Test_multipoints.shp
        / Test_Random_Raster.tif
        / vec2rast_test_lines.shp
        / vec2rast_test_points.shp
        / vec2rast_test_polygons.shp
    / out
"""


def test_run_kde():
    itd = "run_kde"
    lines = gpd.read_file(test_data_path / itd / "Test_lines.shp").to_crs(4326)
    points = gpd.read_file(test_data_path / itd / "Test_nodes.shp").to_crs(4326)
    multipoints = gpd.read_file(test_data_path / itd / "Test_multipoints.shp").to_crs(
        4326
    )
    polygons = gpd.read_file(test_data_path / itd / "Test_polygons.shp").to_crs(4326)
    bbx = points.total_bounds

    # Extract the min and max x/y coordinates
    xmin = bbx[0] - 0.125
    xmax = bbx[2] + 0.125
    ymin = bbx[1] - 0.125
    ymax = bbx[3] + 0.125
    # test ideal points
    a, yct, xct = run_kde(points, xmin, ymin, xmax, ymax)
    assert (
        isinstance(a, np.ndarray)
        and isinstance(yct, np.ndarray)
        and isinstance(xct, np.ndarray)
    )
    assert len(np.nonzero(a)[0]) > 0
    assert a.shape == (2500, 2500)
    assert yct[0][0] != xct[0][0]

    # test one point
    a, yct, xct = run_kde(points.iloc[[0]], xmin, ymin, xmax, ymax)
    assert (
        isinstance(a, np.ndarray)
        and isinstance(yct, np.ndarray)
        and isinstance(xct, np.ndarray)
    )
    assert len(np.nonzero(a)[0]) > 0
    assert a.shape == (2500, 2500)
    assert yct[0][0] != xct[0][0]

    with pytest.raises(ValueError):
        # test multipoints
        a, yct, xct = run_kde(multipoints, xmin, ymin, xmax, ymax)
        # test lines
        a, yct, xct = run_kde(lines, xmin, ymin, xmax, ymax)
        # test polygons
        a, yct, xct = run_kde(polygons, xmin, ymin, xmax, ymax)
    # bad bounding box
    a, yct, xct = run_kde(points, 1.0, 2.0, 2.0, 1.0)  # type: ignore
    assert (
        isinstance(a, np.ndarray)
        and isinstance(yct, np.ndarray)
        and isinstance(xct, np.ndarray)
    )
    assert len(np.nonzero(a)[0]) == 0
    assert a.shape == (2500, 2500)
    assert yct[0][0] != xct[0][0]


def test_kde_to_city_center():
    itd = "kde_to_city_center"
    points = gpd.read_file(test_data_path / itd / "Test_nodes.shp").to_crs(4326)
    bbx = points.total_bounds
    # Extract the min and max x/y coordinates
    xmin = bbx[0] - 0.125
    xmax = bbx[2] + 0.125
    ymin = bbx[1] - 0.125
    ymax = bbx[3] + 0.125
    # test ideal points
    kde_out, YY, XX = run_kde(points, xmin, ymin, xmax, ymax)
    city_name = "Test"
    country_name = "TestTest"
    # test good inputs
    gdf = kde_to_city_center(
        kde_out, XX, YY, city_name, country_name, xmin, xmax, ymin, ymax
    )
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf["city"][0] == f"{city_name}, {country_name}"
    assert len(gdf) == 1
    # test bad kde_places values (too many max)
    kde_bad = np.where(kde_out > kde_out.mean(), 100000, kde_out)
    gdf = kde_to_city_center(
        kde_bad, XX, YY, city_name, country_name, xmin, xmax, ymin, ymax
    )
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf["city"][0] == f"{city_name}, {country_name}"
    assert len(gdf) == 1

    # one is false
    gdf = kde_to_city_center(
        kde_bad,
        XX,
        YY,
        city_name,
        country_name,
        xmin,
        xmax,
        ymin,
        ymax,
        cc_one=False,
    )
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf["city"].unique() == f"{city_name}, {country_name}"
    assert len(gdf) > 1
    # test bad kde_places grid (aka subset)
    kde_bad = kde_bad[kde_bad == kde_bad.max()]
    with pytest.raises(IndexError):
        gdf = kde_to_city_center(
            kde_bad, XX, YY, city_name, country_name, xmin, xmax, ymin, ymax
        )


def test_export_kde_raster():
    itd = test_data_path / "export_kde_rast"
    print([x for x in Path(test_data_path).glob("*")])
    otd = test_data_path / "out"
    points = gpd.read_file(itd / "Test_nodes.shp").to_crs(4326)
    bbx = points.total_bounds
    # Extract the min and max x/y coordinates
    xmin = bbx[0] - 0.125
    xmax = bbx[2] + 0.125
    ymin = bbx[1] - 0.125
    ymax = bbx[3] + 0.125
    # test ideal points
    kde_out, YY, XX = run_kde(points, xmin, ymin, xmax, ymax)
    proj = 4326
    filename = otd / "TestKDE.tif"
    # test good fp, crs, bounding box, xct/yct, grid
    export_kde_raster(kde_out, XX, YY, xmin, xmax, ymin, ymax, proj, filename)
    assert Path(itd / filename).exists()
    # test bad fp
    bad_fp = otd / "BadTestKDE.txt"
    with pytest.raises(TypeError):
        export_kde_raster(kde_out, XX, YY, xmin, xmax, ymin, ymax, proj, bad_fp)
    # test bad crs
    bad_crs = 111111111
    with pytest.raises(rasterio.errors.CRSError):  # type: ignore
        export_kde_raster(kde_out, XX, YY, xmin, xmax, ymin, ymax, bad_crs, filename)


def test_vec_2_rast():
    itd = test_data_path / "vec2rast"
    otd = test_data_path / "out"
    pts_fp = itd / "vec2rast_test_points.shp"
    polys_fp = itd / "vec2rast_test_polygons.shp"
    lines_fp = itd / "vec2rast_test_lines.shp"
    multipts_fp = itd / "Test_multipoints.shp"
    samp_fp = itd / "Test_Random_Raster.tif"
    out_fp = otd / "vec2rast.tif"
    # points
    rast_pts = vec_2_rast(pts_fp, samp_fp, out_fp)
    assert Path(out_fp).exists()
    assert isinstance(rast_pts, np.ndarray)
    assert len(gpd.read_file(pts_fp)) == rast_pts[rast_pts > 0].size
    # polys
    rast_polys = vec_2_rast(polys_fp, samp_fp, out_fp)
    assert Path(out_fp).exists()
    assert isinstance(rast_polys, np.ndarray)
    assert len(gpd.read_file(polys_fp)) < rast_polys[rast_polys > 0].size
    # lines
    rast_lines = vec_2_rast(lines_fp, samp_fp, out_fp)
    assert Path(out_fp).exists()
    assert isinstance(rast_lines, np.ndarray)
    assert len(gpd.read_file(lines_fp)) < rast_lines[rast_lines > 0].size
    # multipoints
    rast_mps = vec_2_rast(multipts_fp, samp_fp, out_fp)
    assert Path(out_fp).exists()
    assert isinstance(rast_mps, np.ndarray)
    assert len(gpd.read_file(multipts_fp)) < rast_mps[rast_mps > 0].size


def test_tif_2_ascii():
    itd = test_data_path / "tif2ascii"
    otd = test_data_path / "out"
    tif_fp = itd / "Test_Random_Raster.tif"
    out_fp = otd / "Test_Random_Raster.asc"
    bad_fp = itd / "Test_Random_Raster2.tif"
    null_fp = itd / "Test_Random_Raster_Nulls.tif"
    nout_fp = otd / "Test_Random_Raster_Nulls.asc"

    b = tif_2_ascii(tif_fp, bad_fp)
    assert isinstance(b, str)
    assert b == "Output File Path Must End in .asc"

    o = tif_2_ascii(tif_fp, out_fp)
    assert isinstance(o, str)
    assert Path(out_fp).exists()
    assert Path(otd / (Path(out_fp).name[:-4] + ".prj")).exists()
    with open(Path(otd / (Path(out_fp).name[:-4] + ".prj"))) as prj:
        assert "WGS_1984_Web_Mercator_Auxiliary_Sphere" in prj.read()
    with open(out_fp) as asc:
        assert "NODATA_value -9999\n" in asc.readlines(150)
    with open(out_fp) as asc:
        f = asc.read()
        assert f.count("-9999") == 1
        assert f.count("None") == 0
        assert f.count("np.") == 0

    o = tif_2_ascii(null_fp, nout_fp)
    assert isinstance(o, str)
    assert Path(nout_fp).exists()
    with open(nout_fp) as asc:
        assert "NODATA_value -9999\n" in asc.readlines(150)
    with open(nout_fp) as asc:
        f = asc.read()
        assert f.count("-9999") == 1
        assert f.count("None") == 0
        assert f.count("np.") == 0


def test_coreg_raster():
    itd = test_data_path / "coreg_rast"
    otd = test_data_path / "out"
    bc_in_fp = itd / "Test_Random_Raster_BigCell.tif"
    bs_in_fp = itd / "Test_Random_Raster_BigShape.tif"
    sc_in_fp = itd / "Test_Random_Raster_SmallCell.tif"
    ss_in_fp = itd / "Test_Random_Raster_SmallShape.tif"
    asc_in_fp = itd / "Test_Random_Raster_Nulls.asc"
    shp_in_fp = itd / "Test_nodes.shp"
    out_fp = otd / "Test_Random_Raster_CoregDelete.tif"
    match_fp = itd / "Test_Random_Raster.tif"

    # test different crs

    match = rasterio.open(match_fp)
    for in_fp in [bc_in_fp, bs_in_fp, sc_in_fp, ss_in_fp, asc_in_fp]:
        print(in_fp)
        coreg_raster(in_fp, match_fp, out_fp)
        assert Path(out_fp).exists()
        assert rasterio.open(out_fp).crs == match.crs
        assert rasterio.open(out_fp).height == match.height
        assert rasterio.open(out_fp).width == match.width
        assert rasterio.open(out_fp).bounds == match.bounds
    for bad in [shp_in_fp]:
        with pytest.raises(rasterio.RasterioIOError):
            coreg_raster(bad, match_fp, out_fp)


def test_node_dist_grid():
    itd = test_data_path / "node_dist_grid"
    nodes = itd / "Test_nodes.shp"
    lines_fp = itd / "Test_lines.shp"
    rast_fp = itd / "Test_Random_Raster.tif"

    points = gpd.read_file(nodes).to_crs(4326)
    points["cc_dist"] = 21
    bbx = points.total_bounds
    # Extract the min and max x/y coordinates
    xmin = bbx[0] - 0.125
    xmax = bbx[2] + 0.125
    ymin = bbx[1] - 0.125
    ymax = bbx[3] + 0.125
    xpts = points["geometry"].x
    ypts = points["geometry"].y
    g, gx, gy, xy = node_dist_grid(points, xmin, xmax, ymin, ymax)

    assert isinstance(g, np.ndarray)
    assert isinstance(gx, np.ndarray)
    assert isinstance(gy, np.ndarray)
    assert isinstance(xy, np.ndarray)

    assert len(gx) == len(gy) == 5000
    assert abs(gx.max()) - abs(max(xpts)) < 53
    assert abs(gx.min()) - abs(min(xpts)) < 0.15
    assert abs(gy.max()) - abs(max(ypts)) < 0.15
    assert abs(gy.min()) - abs(min(ypts)) < 53
    assert len(g[g == 21]) > len(points)
    assert len(points) == len(xy)

    # no distance field
    pts_shp = points[["geometry"]]
    with pytest.raises(KeyError):
        g, gx, gy, xy = node_dist_grid(pts_shp, xmin, xmax, ymin, ymax)

    # bad inputs lines
    lines = gpd.read_file(lines_fp).to_crs(4326)
    with pytest.raises(ValueError):
        g, gx, gy, xy = node_dist_grid(lines, xmin, xmax, ymin, ymax)
    # bad inputs rasters
    rast = rasterio.open(rast_fp)
    with pytest.raises(TypeError):
        g, gx, gy, xy = node_dist_grid(rast, xmin, xmax, ymin, ymax)

    # bad bounding box
    xmin = 0
    xmax = 10
    ymin = -10
    ymax = 25
    g, gx, gy, xy = node_dist_grid(points, xmin, xmax, ymin, ymax)  # type: ignore
    assert isinstance(g, np.ndarray)
    assert isinstance(gx, np.ndarray)
    assert isinstance(gy, np.ndarray)
    assert isinstance(xy, np.ndarray)

    assert xmax < max(xpts) or xmin > min(xpts)
    assert ymax < max(ypts) or ymin > min(ypts)
    assert len(gx) == len(gy) == 5000
    assert len(g[g == 21]) == 0


def test_array_2_tif():
    itd = test_data_path / "array_2_tif"
    otd = test_data_path / "out"
    nodes = itd / "array_2_tiff_test_Points.shp"
    city_name = "Test"
    fdict = {
        "Intermediate": otd,
        "Model_Inputs": otd,
        "Downloads": itd,  # elevation must also be 4326
    }

    points = gpd.read_file(nodes)  # fails if not 4326
    points["cc_dist"] = 21
    points["cc_dist"] = 21
    bbx = points.total_bounds
    # Extract the min and max x/y coordinates
    xmin = bbx[0] - 0.125
    xmax = bbx[2] + 0.125
    ymin = bbx[1] - 0.125
    ymax = bbx[3] + 0.125
    g, gx, gy, xy = node_dist_grid(points, xmin, xmax, ymin, ymax)  # noqa

    array_2_tif(g, gx, gy, fdict, city_name, xmin, xmax, ymin, ymax)


def test_get_resolution_in_meters():
    itd = test_data_path / "get_resolution_in_meters"
    otd = test_data_path / "out"
    rast = itd / "Test_Random_Raster.tif"
    outfp = otd / "get_resolution_in_meters_Test_Rast.tif"
    wkid = 27700
    scrs = "EPSG:3857"
    # test good inputs
    res0 = get_resolution_in_meters(rast, outfp, wkid, scrs)
    assert isinstance(res0, float)
    assert Path(outfp).exists()
    # test bad scrs
    with pytest.raises(rasterio.errors.CRSError):  # type: ignore
        get_resolution_in_meters(rast, outfp, wkid, 111111)
    # test wrong scrs
    with pytest.raises(rasterio._err.CPLE_AppDefinedError):  # type: ignore
        get_resolution_in_meters(rast, outfp, wkid, "EPSG:4326")
    # test bad wkid
    with pytest.raises(rasterio.errors.CRSError):  # type: ignore
        get_resolution_in_meters(rast, outfp, 111111, scrs)
    # test same wkid as scsrs
    outfp = otd / "no_get_resolution_in_meters_Test_Rast.tif"
    res1 = get_resolution_in_meters(rast, outfp, wkid, wkid)
    assert isinstance(res1, float)
    assert not Path(outfp).exists()
    assert res1 != res0
    # test bad input
    with pytest.raises(rasterio.errors.RasterioIOError):  # type: ignore
        get_resolution_in_meters(str(rast) + "fred", outfp, wkid, scrs)


def test_region_group():
    itd = test_data_path / "region_group"
    otd = test_data_path / "out"
    input1 = itd / "Test_Random_Raster_RG.tif"

    outfp = otd / "Test_Random_Raster_RG_Out.tif"
    exrast = itd / "Test_Random_Raster_RG_Example.tif"
    one = region_group(input1, outfp, exrast)
    assert Path(outfp).exists()
    input2 = rasterio.open(input1).read(1)
    two = region_group(input2, outfp, exrast)
    assert np.array_equal(one, two)
    three = region_group(input1, outfp, exrast, n=4)
    assert not np.array_equal(one, three)
    # bad n value
    with pytest.raises(ValueError):
        region_group(input1, outfp, exrast, n=6)
    # bad input data string fp
    with pytest.raises(rasterio.errors.RasterioIOError):  # type: ignore
        region_group(str(input1) + "fred", outfp, exrast)
    # mismatch with projection of example and input raster
    exrast = itd / "Test_elevation_3857.tif"
    outfp = otd / "Test_Random_Raster_RG_Out3857.tif"
    x = region_group(input1, outfp, exrast)
    assert np.array_equal(one, x)
    assert Path(outfp).exists()
    assert str(rasterio.open(outfp).crs) != str(rasterio.open(input1).crs)


def test_rast_2_poly():
    itd = test_data_path / "rast2poly"
    in3 = itd / "Test_Rast2Poly_Three.tif"
    in6 = itd / "Test_Rast2Poly_Six.tif"
    col = "fred"
    gdf3 = rast_2_poly(in3, col)
    gdf6 = rast_2_poly(in6, col)

    assert gdf6.crs == gdf3.crs == rasterio.open(in3).crs
    assert col in gdf3.columns
    assert col in gdf6.columns
    assert len(gdf3) == 3
    assert len(gdf6) == 6


# def test_raster_out():
#     raster_out()


# def test_rescale_raster():
#     rescale_raster()


# def test_basic_point_density():
#     basic_point_density()


# def test_distance_accumulation():
#     distance_accumulation()


# def test_zonal_stats()
#     zonal_stats()
