from IO.Scripts.Modules.utilities import delete_file_if
import rasterio
import requests
import tempfile
import mercantile
from pathlib import Path
from rasterio.transform import from_bounds
from rasterio.merge import merge
import numpy as np


def download_terrain_tile(x: int, y: int, z: int, output_path: str | Path) -> bool:
    """
    Download a terrain tile from AWS Terrain Tiles.

    Args:
        x (int): Tile x coordinate
        y (int): Tile y coordinate
        z (int): Zoom level
        output_path (str): Path to save the terrain tile
    """
    url = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/"
    url2 = f"{z}/{x}/{y}.png"
    response = requests.get(url + url2)

    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        return True
    return False


def decode_terrain_png(png_path: str | Path) -> np.ndarray:
    """
    Decode terrain PNG file into elevation data.

    Args:
        png_path (str): Path to terrain PNG file

    Returns:
        numpy.ndarray: Elevation data
    """
    with rasterio.open(png_path) as src:
        red = src.read(1)
        green = src.read(2)
        blue = src.read(3)

    # Convert RGB values to elevation in meters
    elevation = (red * 256 + green + blue / 256) - 32768
    return elevation


def get_terrain_data(
    bounds: tuple[np.float64, ...],
    zoom: int,
    output_path: str | Path,
    city_name: str,
    delfile: bool = False,
) -> str | None:
    """
    Download and merge terrain data for a given bounding box.

    Args:
        bounds (tuple): (min_lon, min_lat, max_lon, max_lat)
        zoom (int): Zoom level
        output_path (str): Path to save the merged GeoTIFF
        city_name : str,
        delfile : bool - default to False
    """
    elves = Path(output_path) / f"{city_name}_elevation.tif"
    delete_file_if(elves, delfile=delfile)
    if not Path(elves).exists():
        # Create temporary directory for downloaded tiles
        temp_dir = Path(tempfile.mkdtemp())

        # Get list of tiles that cover the area
        tiles = list(
            mercantile.tiles(bounds[0], bounds[1], bounds[2], bounds[3], zooms=[zoom])
        )

        # Download and process each tile
        to_merge = []
        for tile in tiles:
            tile_path = temp_dir / f"{tile.x}_{tile.y}_{tile.z}.png"

            # Download tile
            if download_terrain_tile(tile.x, tile.y, tile.z, str(tile_path)):
                # Get tile bounds
                tile_bounds = mercantile.bounds(tile)

                # Decode elevation data
                elevation = decode_terrain_png(str(tile_path))

                # Create temporary GeoTIFF for the tile
                temp_tiff = temp_dir / f"{tile.x}_{tile.y}_{tile.z}.tif"

                # Calculate transform for the tile
                transform = from_bounds(
                    tile_bounds.west,
                    tile_bounds.south,
                    tile_bounds.east,
                    tile_bounds.north,
                    elevation.shape[1],
                    elevation.shape[0],
                )

                # Save as GeoTIFF
                with rasterio.open(
                    str(temp_tiff),
                    "w",
                    driver="GTiff",
                    height=elevation.shape[0],
                    width=elevation.shape[1],
                    count=1,
                    dtype=elevation.dtype,
                    crs="EPSG:4326",
                    transform=transform,
                ) as dst:
                    dst.write(elevation, 1)

                to_merge.append(temp_tiff)

        # Merge all tiles
        if to_merge:
            src_files = [rasterio.open(str(p)) for p in to_merge]
            mosaic, out_transform = merge(src_files)

            # Get metadata from first file
            out_meta = src_files[0].meta.copy()
            out_meta.update(
                {
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_transform,
                }
            )

            # Write merged file
            with rasterio.open(elves, "w", **out_meta) as dest:
                dest.write(mosaic)

            # Close all files
            for src in src_files:
                src.close()
        print(f"Data Download Successful: {elves}")
    else:
        print(f"{elves} was not deleted nor recreated.")
