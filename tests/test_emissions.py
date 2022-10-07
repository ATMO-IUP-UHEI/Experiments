"""
"""

import numpy as np
from pathlib import Path


from context import Emissions

config = {
    "emissions": {
        "prior": "TNO",
        "prior_uncertainty": "TNO",
        "truth": "combined_emissions",
    }
}


def test_emission():
    emissions = Emissions(config)

    emission_list = [
        "TNO",
        "TNO_uncertainty",
        # "mean_TNO",
        # "mean_TNO_uncertainty",
        "heat",
        "traffic",
        "combined_emissions",
    ]

    for emission_name in emission_list:
        emissions.get(emission_name)
        emissions.get(emission_name + "_diurnal")
