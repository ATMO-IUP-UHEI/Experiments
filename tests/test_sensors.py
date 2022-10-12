"""
"""

import numpy as np
from pathlib import Path


from context import *


def test_sensors():
    config = {"n_sensors": 20, "noise": 1.0}

    config_with_optional = config | {"seed": 1, "time": 2, "height": 1}

    sensors = Sensors(config)
    sensors.get_index()
    sensors.get_noise()

    sensors = Sensors(config_with_optional)
    sensors.get_index()
    sensors.get_noise()
