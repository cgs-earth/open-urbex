from pathlib import Path
import duckdb


def create_dd_con():
    """
    Create DuckDB Connection to in-memory database mode
    with spatial extension and azure storage connection string.

    Parameters
    ----------
    None

    Returns
    -------
    con : Connection Object
    """
    con = duckdb.connect()
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial;")
    con.execute(
        "SET azure_storage_connection_string = \
                'DefaultEndpointsProtocol=https;\
                AccountName=overturemapswestus2;\
                AccountKey=;EndpointSuffix=core.windows.net';"
    )
    return con


def move_pathlib(src_path, dest_path):
    """
    Move file from one directory to another.
    Does not make a copy.

    Parameters
    ----------
    src_path : str or PurePath, current file location
    dest_path : str or PurePath, destination location

    Returns
    -------
        : str, message whether file has been moved
    """
    # Create Path objects
    src_path = Path(src_path)
    dest_path = Path(dest_path)

    # Copy file
    src_path.rename(dest_path)
    return f"{dest_path} exists? {dest_path.exists()}"


def delete_file_if(src_path, delfile=True):
    """
    Delete a file if it exists and if the user wants
    to default to deleting and writing new files.

    Parameters
    ----------
    src_path : str or PurePath, file location

    Returns
    -------
        : str, message whether file has been deleted
    """
    src_path = Path(src_path)
    if src_path.exists():
        if delfile is True:
            src_path.unlink()
        else:
            return f"{src_path} already exists, delete not selected."
    else:
        return f"{src_path} does not exist, delete not necessary."


class BadGeoError(Exception):
    """
    Exception raised for custom error scenarios including:
    1. Point is required to be within a bounding box.

    Attributes:
        message -- explanation of the error

    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def warning_handler(w):
    print(f"Warning: Function running with caveat - {w}")
