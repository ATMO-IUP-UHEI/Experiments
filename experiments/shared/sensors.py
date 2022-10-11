import numpy as np

import xarray as xr


from ggpymanager import GRAL


class Sensors:
    def __init__(self, config) -> None:
        """

        Parameters
        ----------
        config : dict
            Required config keys:
                n_sensors: number of sensors.
                noise: std of measurements from sensor.
            Optional config keys (default):
                seed: (1) seed for random number generator
                time: (1) number of hourly measurements
                height: (0) height level of the sensors
        """
        self.rng = np.random.default_rng(config.get("seed", 1))

        self.time = config.get("time", 1)

        self.n_sensors = config["n_sensors"]
        self.height = config.get("height", 0)
        noise = config["noise"]
        if isinstance(noise, dict):
            pass
        else:
            self.noise = xr.DataArray(
                data=np.tile(noise, [self.n_sensors,self.time]),
                dims=["sensor", "time_measurement"],
            )

        # Add support for MC
        self.create_grid(self.n_sensors)

        self.index = None

    def create_grid(self, n_sensors):
        ny = int(np.sqrt(n_sensors))
        nx = int(np.ceil(n_sensors / ny))

        dx = GRAL.nx // nx
        dy = GRAL.ny // ny
        self.xmesh, self.ymesh = np.meshgrid(
            np.arange(0.5 * dx, dx * nx, dx, dtype=int),
            np.arange(0.5 * dy, dy * ny, dy, dtype=int),
        )

    def get_index(self, compute_new=False):
        if (self.index is None) or compute_new:
            if self.n_sensors == (self.xmesh.shape[0] * self.xmesh.shape[1]):
                self.index = (
                    self.xmesh.flatten(),
                    self.ymesh.flatten(),
                    np.repeat(self.height, self.n_sensors),
                )
            else:
                print("Error: not same shape")
        return self.index

    def get_noise(self):
        mean = np.zeros_like(self.noise)
        std = self.noise
        noise = self.rng.normal(
            loc=mean, scale=std #, size=(self.n_sensors, self.time)
        )
        xr_noise = xr.DataArray(
            data=noise,
            dims=self.noise.dims
        )
        return xr_noise
