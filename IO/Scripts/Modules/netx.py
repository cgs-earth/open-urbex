from IO.Scripts.Modules.utilities import BadGeoError
import momepy as mp
import pandas as pd
import networkx as nx
import geopandas as gpd  # noqa
from pyproj import CRS


def node2cc(
    rds: gpd.GeoDataFrame,
    cc: gpd.GeoDataFrame,
    wkid: str,
) -> gpd.GeoDataFrame:  # TODO: Write Test!  # TODO: fix types!
    """
    Calculate distance from every node in network to city center.

    Parameters
    ----------
    rds : geopandas geodataframe, represents road lines
    cc : geopandas geodataframe, city center point
    wkid : string or int, planar crs (projected)

    Returns
    -------
    nodes : pandas dataframe, distance to city center per node
    """
    # input check
    if not (
        (
            rds.geom_type.iloc[0] == "LineString"
            or rds.geom_type.iloc[0] == "MultiLineString"
        )
        and (cc.geom_type.iloc[0] == "Point" or cc.geom_type.iloc[0] == "MultiPoint")
    ):
        raise TypeError("Check your input geometry.")
    if not CRS.from_user_input(str(wkid)).is_projected:
        raise BadGeoError("Not a Valid Projected CRS.")
    # if not rds.union_all().envelope.contains(cc.geometry).iloc[0]:
    # raise BadGeoError("Point not within Road Bounding Box.")

    # wkid should be the utm - need planar
    # make graph and dfs from graph
    cc = cc.to_crs(wkid)
    G = mp.gdf_to_nx(rds.to_crs(wkid), approach="primal")
    nodes, edges = mp.nx_to_gdf(G)  # type: ignore

    # get nearest node for city center
    cc["edge_index"] = mp.get_nearest_street(cc, edges)  # type: ignore
    city_center = mp.get_nearest_node(  # noqa
        cc, nodes, edges, cc["edge_index"]  # type: ignore
    )
    citcen = [
        tuple(x)
        for x in nodes.query("nodeID.isin(@city_center)")[  # type: ignore
            ["x", "y"]
        ].values.tolist()
    ]
    # solve graph
    dj = nx.single_source_dijkstra_path_length(
        G, source=citcen[0], weight="mm_len"  # type: ignore
    )  # distance is in meters
    # attach distances to nodes dataframe
    nodes["tup_node"] = [  # type: ignore
        tuple(x) for x in nodes[["x", "y"]].values.tolist()  # type: ignore
    ]
    dj_df = pd.DataFrame.from_dict(dj, orient="index").reset_index()
    dj_df.columns = ["tup_node", "cc_dist"]
    nodes = pd.merge(nodes, dj_df, how="left", on="tup_node")  # type: ignore
    return nodes  # type: ignore
