"""
"""

import numpy as np
from pathlib import Path


from context import *

config = {"sensors": {"n_sensors": 20, "noise": 1.0}}


def test_sensors():
    sensors = Sensors(config)

    sensors.get_index()
    sensors.get_noise()
