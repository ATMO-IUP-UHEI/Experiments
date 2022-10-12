"""
"""

import numpy as np
from pathlib import Path


from context import Transport, data_path

def setup_data_path(data_path):
    (data_path / "meteopgt.all").touch()

def test_transport(data_path):
    setup_data_path(data_path)
    config = {
        "catalog_path": data_path, 
        "config_path": data_path,
        "simulation_path": data_path,
        }

    config_with_optional = config | {"seed": 1, "time": 2, "mode": "random_realistic"}

    transport = Transport(config)
    transport = Transport(config_with_optional)