"""
"""

import numpy as np
from pathlib import Path


from context import *


def test_sensors():
    config = {"n_sensors": 20, "noise": 1.0}

    config_with_optional = config | {"seed": 1, "time": 2, "height": 1}

    for config in [config, config_with_optional]:
        sensors = Sensors(config)
        sensors.get_index()
        sensors.get_noise()

        new_n_sensors = 12
        sensors.set_n_sensors(n_sensors=new_n_sensors)
        index = sensors.get_index()
        assert len(index[0]) == new_n_sensors
        noise = sensors.get_noise()
        assert len(noise["sensor"]) == new_n_sensors

        n_sub_sample = 5
        index = sensors.get_index(n_sub_sample)
        assert len(index[0]) == n_sub_sample
        noise = sensors.get_noise(n_sub_sample)
        assert len(noise["sensor"]) == n_sub_sample
