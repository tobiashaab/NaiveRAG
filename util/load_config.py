from dotenv import load_dotenv
import yaml
import os


def load_config(config_path: str = "./config.yaml") -> dict:
    """Loads the config into a dictionary for later usage.

    Args:
        config_path (str, optional): The file path to the dictionary.. Defaults to "./config.yaml".

    Returns:
        dict: The dictionary containing the config parameters.
    """
    load_dotenv()

    with open(config_path, "r") as file:
        config = file.read()

    config = os.path.expandvars(config)
    config = yaml.safe_load(config)

    return config
