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

        Examples
        --------
        For new number of sensors

        >>> sensors = Sensors(config)
        >>> sensors.create_grid(new_n_sensors)
        >>> sensors.get_index(compute_new=True)
        """
        self.rng = np.random.default_rng(config.get("seed", 1))

        self.time = config.get("time", 1)

        self.n_sensors = config["n_sensors"]
        self.height = config.get("height", 0)
        self.std_config = config["noise"]
        self.set_std(self.std_config)

        # Add support for MC
        self.create_grid(self.n_sensors)

        self.index = None

    def set_std(self, std):
        self.std = std

    def set_n_sensors(self, n_sensors):
        self.n_sensors = n_sensors
        self.create_grid(n_sensors)
        self.index = None

    def create_grid(self, n_sensors):
        ny = int(np.ceil(np.sqrt(n_sensors)))
        nx = int(np.ceil(n_sensors / ny))

        dx = GRAL.nx // nx
        dy = GRAL.ny // ny
        self.xmesh, self.ymesh = np.meshgrid(
            np.arange(0.5 * dx, dx * nx, dx, dtype=int),
            np.arange(0.5 * dy, dy * ny, dy, dtype=int),
        )

    def get_sample_ids(self, n_sub_sample=None):
        sample_ids = np.arange(self.n_sensors)
        # Sample from sensors
        if n_sub_sample is not None:
            self.rng.shuffle(sample_ids)
            sample_ids = np.sort(sample_ids[:n_sub_sample])
        return sample_ids

    def get_index(self, n_sub_sample=None):
        if (self.index is None) or (n_sub_sample is not None):
            sample_ids = self.get_sample_ids(n_sub_sample)

            self.index = (
                self.xmesh.flatten()[sample_ids[:n_sub_sample]],
                self.ymesh.flatten()[sample_ids[:n_sub_sample]],
                np.repeat(self.height, self.n_sensors)[sample_ids],
            )
        return self.index

    def get_noise(self, n_sub_sample=None):
        if n_sub_sample is None:
            n_sub_sample = self.n_sensors
        mean = np.zeros_like(self.std)
        noise = self.rng.normal(
            loc=mean, scale=self.std, size=(self.n_sensors, self.time)
        )
        xr_noise = xr.DataArray(data=noise, dims=["sensor", "time_measurement"])
        sample_ids = self.get_sample_ids(n_sub_sample)
        return xr_noise.isel(sensor=sample_ids)
