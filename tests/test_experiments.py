"""
"""

import numpy as np

from context import *


def test_base_experiment():
    experiment = Experiment()
    experiment.get_config()

def test_experiments():
    experiments = [BasicSetup()]

    for experiment in experiments:
        experiment.get_config()
        assert "run" in dir(experiment)
