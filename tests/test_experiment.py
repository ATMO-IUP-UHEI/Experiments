"""
"""

import numpy as np
from pathlib import Path
import yaml

from context import *
import test_transport


def test_experiments(data_path):
    test_transport.setup_data_path(data_path)
    config = {
        "seed": 2,
        "time": 10,
        "transport": {
            "mode": "random",
        },
        "sensors": {
            "n_sensors": 20,
            "height": 1,  # 0-4
            "noise": 1.0,
        },
        "emissions": {
            "constant": False,
            "prior": "mean_TNO",
            "prior_variance": "TNO_variance",
            "truth": "TNO",
            "emission_path": str(data_path),
            "correlation": {
                "tau_h": 0.001,  # 1/e correlation after 1h
                "tau_d": 0.001,  # 2. # 1/e correlation after 2d
                "tau_l": None,  # 1/e correlation for each km in mean distance
            },
        },
        "reader": {
            "catalog_path": str(data_path),
            "config_path": str(data_path),
            "simulation_path": str(data_path),
        },
    }

    yml_name = "config.yml"
    config_path = data_path / yml_name
    with open(config_path, "w") as file:
        yaml.dump(config, file)
   
    experiment = Experiment(config_path)
