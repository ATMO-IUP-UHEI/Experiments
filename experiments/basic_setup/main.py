from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time

from bayesinverse import Regression


from experiments.shared import CONSTANTS, Experiment
from experiments.shared import utilities as utils


CONSTANTS.TEST_INT


class BasicSetup(Experiment):
    def __init__(self, config_path):
        super(BasicSetup, self).__init__(config_path)

    def run(self):
        self.emissions.prior
        self.emissions.prior_variance
        self.emissions.truth

        self.K = self.transport.get_transport(self.sensors, self.emissions)

        self.reg = Regression(
            y=utils.stack_xr(
                self.K @ self.emissions.truth + self.sensors.get_noise()
            ).values,
            K=utils.stack_xr(self.K).values,
            x_prior=utils.stack_xr(self.emissions.prior).values,
            x_covariance=utils.stack_xr(self.emissions.prior_covariance).values,
            y_covariance=utils.stack_xr(self.sensors.noise).values,
        )
        x_est, res, rank, s = self.reg.fit()
        posterior = self.emissions.to_xr(x_est)
        print(self.emissions.truth - posterior)
        print(self.reg.get_averaging_kernel())

    
if __name__ == "__main__":
    config_path = config_path = Path(__file__).resolve().parent / "config.yaml"

    basic_setup = BasicSetup(config_path)
    basic_setup.run()
