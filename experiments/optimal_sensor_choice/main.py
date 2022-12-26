import numpy as np


from bayesinverse import Regression


from ..shared import Experiment
from ..shared import utilities as utils


class OptimalSensorChoice(Experiment):
    def __init__(self, config_path):
        super(OptimalSensorChoice, self).__init__(config_path)
