"""
"""

import numpy as np
from pathlib import Path


from context import *


def test_emission(data_path):
    config = {
        "prior": "TNO",
        "prior_variance": "TNO",
        "truth": "combined_emissions",
        "emission_path": data_path,
    }

    config_with_optional = config | {
        "time": 10,
        "mode": "diurnal",
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

    for emission_name in emission_list:
        emissions.get(emission_name)
