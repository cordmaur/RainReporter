"""Util functions"""

import io
from pathlib import Path
from typing import Union, Dict

import pyjson5


def open_json_file(config_file: Union[str, Path, io.StringIO]) -> Dict:
    """
    Try to locate the configuration file and return it as a dictionary
    If a file is not passed, try to locate in current directory
    """

    # if the config file is a file-like (StringIO) obj, parse it automatically
    if config_file is not None and isinstance(config_file, io.StringIO):
        report_config = pyjson5.decode_io(config_file)  # pylint: disable=no-member

    else:
        config_file = Path(config_file)

        # check if it exists
        if not config_file.exists():
            raise FileNotFoundError(
                f"Config file not found {str(config_file.absolute())}"
            )

        # open the config file
        with open(config_file, "r", encoding="utf-8") as file:
            report_config = pyjson5.decode_io(file)  # pylint: disable=no-member

    return report_config
