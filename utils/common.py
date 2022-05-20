import yaml
import json
from from_root import from_root
import os


def read_config(config_path):
    with open(config_path) as config_file:
        content = yaml.safe_load(config_file)

    return content