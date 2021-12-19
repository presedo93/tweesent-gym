import os
import json


def open_conf(conf_path: str) -> dict:
    """Loads the config JSON.
    Args:
        conf_path (str): config file path.
    Returns:
        dict: config values as dict.
    """
    with open(os.path.join(os.getcwd(), conf_path), "r") as f:
        conf = json.load(f)

    return conf
