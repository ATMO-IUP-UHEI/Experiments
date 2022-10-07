"""
"""

import numpy as np
from pathlib import Path


from context import *


""" def test_base_experiment():
    experiment = Experiment()
    experiment.get_config() """

def test_experiments():
    Experiments = [BasicSetup]
    config_paths = [
        Path(__file__).resolve().parent.parent / "experiments/basic_setup/config.yaml"
    ]

    for config_path, ExperimentClass in zip(config_paths, Experiments):
        experiment = ExperimentClass(config_path)
        experiment.sensors.get_index()

        assert "run" in dir(experiment)
