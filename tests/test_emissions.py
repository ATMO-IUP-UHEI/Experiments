"""
"""

import numpy as np
from pathlib import Path


from context import *


def test_emission(data_path):
    config = {
        "prior": "TNO",
        "prior_mode": "diurnal",
        "prior_variance": "TNO",
        "truth": "combined_emissions",
        "truth_mode": "diurnal",
        "emission_path": data_path,
    }

    config_with_optional = config | {
        "time": 10,
        "tau_h": 1.0,
        "tau_d": 1.0,
    }
    emissions = Emissions(config)
    emissions = Emissions(config_with_optional)
    emission_list = [
        "TNO",
        "TNO_variance",
        "mean_TNO",
        "mean_TNO_variance",
        "heat",
        "traffic",
        "combined_emissions",
    ]
    mode_list = ["single_time", "constant", "diurnal"]

    for emission_name in emission_list:
        for mode in mode_list:
            print(emission_name, mode)
            emissions.get(emission_name, mode)
