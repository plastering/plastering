"""Helper function that deal with paths
"""

import pathlib


def create_dir(dir_name):
    """Create a dir if it does not already exist

    Args:
        dir_name (str): the directory name

    Raises:
        FileExistsError: If the path specified already exists and is a file
    """

    path = pathlib.Path(dir_name)

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    elif path.is_file():
        raise FileExistsError('Log Path already exists and is not a dir')
