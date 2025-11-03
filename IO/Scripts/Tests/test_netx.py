from IO.Scripts.Modules.netx import node2cc
from IO.Scripts.Modules.utilities import BadGeoError

from pathlib import Path
import pandas as pd
import geopandas as gpd
import pytest

repo = Path("/home/runner/work/urbex/urbex/IO")


def test_node2cc():
    """
    Test Cases:

    Inputs
    1. Roads
        a. Ideal Input
        b. Correct Type, Data Issue (Island Roads)
        c. Incorrect Geometry Type
        d. No Geometry
    2. City Center
        a. Ideal Input
        b. Correct Type, Data Issue (Point Out of Bounding Box)
        c. Too Many Points
        d. Incorrect Geometry Type
        e. No Geometry
    3. WKID
        a. Ideal Input
        b. Not Planar (Geodesic)
        c. Not Valid Input

    Output
    1. Assert Pandas Dataframe
    2. Assert Columns == ["tup_node", "cc_dist"]
    3. Assert cc_dist is numeric and positive
    """
    test_data = repo / "Scripts" / "Tests" / "Test_Data" / "test_netx"
    lines = gpd.read_file(test_data / "node2cc_test_lines.shp").to_crs(4326)
    points = gpd.read_file(test_data / "node2cc_test_nodes.shp").to_crs(4326)
    multipoints = gpd.read_file(test_data / "node2cc_test_multipoints.shp").to_crs(4326)
    polygons = gpd.read_file(test_data / "node2cc_test_polygons.shp").to_crs(4326)

    rds = {
        "ideal": lines.query("problem == 'Ideal'"),  # ideal
        "discon": lines.query("problem == 'Disjoint'"),  # disconnected roads
        "polygons": polygons,  # polygons
        "points": points,  # points
        "mixed": pd.concat([lines, points, polygons]),  # mixed
        "nogeo": lines.drop(columns="geometry"),  # no geometry
        "one": lines.iloc[0],  # only one line
    }
    cc = {
        "ideal": points.query("problem == 'Ideal'"),  # ideal
        "far": points.query("problem == 'Far Point'"),  # far point
        "many": points.query("problem == 'Too Many'"),  # too many points
        "multipoint": multipoints,  # multipoint
        "polygon": polygons.iloc[0],  # polygon
        "line": lines.iloc[0],  # line
        "nogeo": points.query("problem == 'Ideal'").drop(
            columns="geometry"
        ),  # no geometry
    }
    wkid = ["27700", "ESPG:4326", "77HUNGARY88"]

    rd_failures = ["polygons", "points", "mixed", "nogeo"]
    cc_failures = ["far", "many", "multipoint", "polygon", "line", "nogeo"]

    for r in rds:
        if r in rd_failures:
            with pytest.raises(TypeError):
                n = node2cc(rds[r], cc["ideal"], wkid[0])
        else:
            n = node2cc(rds[r], cc["ideal"], wkid[0])
            assert isinstance(n, pd.DataFrame)
            assert list(n.columns) == ["tup_node", "cc_dist"]
            assert str(n["cc_dist"].type).str.contains("float")
    for c in cc:
        if c in cc_failures:
            with pytest.raises(TypeError, BadGeoError):
                n = node2cc(rds["ideal"], cc[c], wkid[0])
        else:
            n = node2cc(rds["ideal"], cc[c], wkid[0])
            assert isinstance(n, pd.DataFrame)
            assert list(n.columns) == ["tup_node", "cc_dist"]
            assert str(n["cc_dist"].type).str.contains("float")
    for w in wkid:
        if w != wkid[0]:
            with pytest.raises(TypeError):
                n = node2cc(rds["ideal"], cc["ideal"], wkid[w])
        else:
            n = node2cc(rds["ideal"], cc["ideal"], wkid[w])
