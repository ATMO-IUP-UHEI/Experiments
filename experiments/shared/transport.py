import numpy as np
import xarray as xr
from ggpymanager import Reader, Status


class Transport:
    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Required config keys:
                catalog_path: 
                config_path: 
                simulation_path: 
            Optional config keys (default):
                seed: (0)
                time: (1) number of hourly measurements.
                mode: (random) {random, random_realistic, slice}
        """
        self.rng = np.random.default_rng(config.get("seed", 0))
        
        self.reader = Reader(
            config["catalog_path"],
            config["config_path"],
            config["simulation_path"],
        )
        self.time = config.get("time", 1)
        self.mode = config.get("mode", "random")

        self.unfinished_meteo_numbers = self.reader.get_gral_missing()

    def check_available(self, meteo_id):
        meteo_number = meteo_id + 1
        if meteo_number in self.unfinished_meteo_numbers:
            return False
        else:
            return True

    def get_K(self, meteo_id, sensors, emissions):
        con_dict = self.reader.get_concentration(meteo_id)

        # make forward model K
        sensor_index = sensors.get_index()
        height_list = np.unique(sensor_index[2])
        xr_K = xr.DataArray(
            data=np.zeros((sensors.n_sensors, len(emissions.prior))),
            dims=["sensor", "source_group"],
            coords={"source_group": emissions.prior.coords["source_group"]}
        ) 
        for key, val in con_dict.items():
            height = int(key[0]) - 1
            if height in height_list:
                source_group = int(key[1:])
                if source_group < len(emissions.mask):
                    if emissions.mask.sel(source_group=source_group):
                        if val.size > 1:
                            index = sensor_index[2] == height
                            xr_K.loc[index, source_group] = val[
                                (
                                    sensor_index[0][index],
                                    sensor_index[1][index],
                                )
                            ]

        # Convert from mu g/m^3 to ppm
        # At 273.15 K and 1 bar
        Vm = 22.71108  # standard molar volume of ideal gas [l/mol]
        m = 44.01  # molecular weight mass [g/mol]
        cubic_meter_to_liter = 1000
        mu_g_to_g = 1e-6
        to_ppm = 1e6
        factor = Vm / m / cubic_meter_to_liter
        xr_K *= factor
        return xr_K

    def get_transport(self, sensors, emissions):
        """Return a the finished forward model K for the inversion"""
        # Get random wind situations
        if self.mode == "random":
            finished_meteo_numbers = [i + 1 for i in range(self.reader.total_sim)]
            for meteo_number in self.unfinished_meteo_numbers:
                finished_meteo_numbers.remove(meteo_number)
            # Choose meteo_ids from the finished simulations
            ids = self.rng.integers(
                low=0, high=len(finished_meteo_numbers), size=self.time
            )
            meteo_ids = np.array(finished_meteo_numbers)[ids] - 1
        # Construct the transport model matrix K
        time_measurement = self.time
        time_state = self.time if not emissions.constant else 1
        K = np.zeros(
            (sensors.n_sensors, time_measurement, len(emissions.prior), time_state)
        )
        xr_K = xr.DataArray(
            data=K, dims=["sensor", "time_measurement", "source_group", "time_state"]
        )
        for ti in range(self.time):
            ti_measurement = ti
            ti_state = ti if not emissions.constant else 0
            ti_meteo_id = meteo_ids[ti]
            xr_K.loc[:, ti_measurement, :, ti_state] = self.get_K(
                ti_meteo_id, sensors, emissions
            )
        xr_K.coords["time_measurement"] = meteo_ids
        return xr_K
