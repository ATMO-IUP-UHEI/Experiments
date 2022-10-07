from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time

from bayesinverse import Regression


from experiments.shared import CONSTANTS, Experiment


CONSTANTS.TEST_INT


class BasicSetup(Experiment):
    def __init__(self, config_path):
        super(BasicSetup, self).__init__(config_path)

    def run(self):
        self.emissions.prior
        self.emissions.prior_uncertainty
        self.emissions.truth

        self.K = self.transport.get_K(800, self.sensors, self.emissions)
      
        self.reg = Regression(
            y=self.K @ self.emissions.truth + self.sensors.get_noise(),
            K=self.K,
            x_prior=self.emissions.prior,
            x_covariance=self.emissions.prior_uncertainty,
            y_covariance=np.repeat(self.sensors.noise, self.sensors.n_sensors),
        )
        x_est, res, rank, s = self.reg.fit()
        print(self.emissions.truth - x_est)
        print(self.reg.get_averaging_kernel())

    
if __name__ == "__main__":
    config_path = config_path = Path(__file__).resolve().parent / "config.yaml"

    basic_setup = BasicSetup(config_path)
    basic_setup.run()
