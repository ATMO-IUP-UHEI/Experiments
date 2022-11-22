import numpy as np
import xarray as xr
import time
from multiprocessing import Pool
from pathlib import Path
import pandas as pd

from ggpymanager import Reader, Status
import ggpymanager.utils


from ..shared import utilities as utils


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
                mode: (random) {random, realistic}
        """
        self.rng = np.random.default_rng(config.get("seed", 0))

        self.reader = Reader(
            config["catalog_path"],
            config["config_path"],
            config["simulation_path"],
        )
        self.time = config.get("time", 1)
        self.mode = config.get("mode", "random")

    def get_finished_meteo_numbers(self):
        finished_meteo_numbers = []
        for meteo_number in range(1, self.reader.total_sim + 1):
            meteo_id = meteo_number - 1
            if self.reader.simulations[meteo_id].status == Status.finished:
                finished_meteo_numbers.append(meteo_number)
        return finished_meteo_numbers

    def get_meteo_ids(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        finished_meteo_numbers = self.get_finished_meteo_numbers()
        # Get random wind situations
        if self.mode == "random":
            # Choose meteo_ids from the finished simulations
            ids = self.rng.integers(
                low=0, high=len(finished_meteo_numbers), size=self.time
            )
            meteo_ids = np.array(finished_meteo_numbers)[ids] - 1
        if self.mode == "realistic":
            path = Path(
                "/mnt/data/users/rmaiwald/GRAMM-GRAL/wind_logs/used_weather_sits.csv"
            )
            df = pd.read_csv(path, index_col=0)
            meteo_ids = df.index[: self.time]
        return meteo_ids

    def init_K(self, n_sensors, emissions):
        # Construct the transport model matrix K
        n_time_measurement = self.time
        n_time_state = self.time if not emissions.prior_mode == "single_time" else 1
        K = np.zeros(
            (n_sensors, n_time_measurement, len(emissions.prior), n_time_state)
        )
        xr_K = xr.DataArray(
            data=K,
            dims=["sensor", "time_measurement", "source_group", "time_state"],
            coords={"source_group": emissions.prior.coords["source_group"]},
        )
        return xr_K

    def get_xr_K_list(
        self,
        meteo_ids,
        n_processes,
        sensors_index,
        n_sensors,
        emissions_mask,
        n_emissions,
    ):
        path_list = []
        for meteo_id in meteo_ids:
            path_list.append(self.reader.simulations[meteo_id].sim_sub_path / "con.npz")

        if n_processes > 1:
            # Create args as list
            args_list = []
            for path in path_list:
                args_list.append(
                    [
                        path,
                        sensors_index,
                        n_sensors,
                        emissions_mask,
                        n_emissions,
                    ]
                )
            with Pool(n_processes) as pool:
                xr_K_list = pool.starmap(utils.get_K_kernel, args_list)
        else:
            xr_K_list = []
            for path in path_list:
                xr_K_list.append(
                    utils.get_K_kernel(
                        path, sensors_index, n_sensors, emissions_mask, n_emissions
                    )
                )
        return xr_K_list

    def get_transport(
        self, n_sensors, sensors_index, emissions, n_processes=1, seed=None
    ):
        """Return a the finished forward model K for the inversion"""
        # Get meteo_ids for the time steps
        start = time.perf_counter()
        meteo_ids = self.get_meteo_ids(seed=None)
        # Initialize the forward model matrix with as xr DataArray
        xr_K = self.init_K(n_sensors, emissions)
        # Get a list of the concentration as xr DataArray for each time step
        emissions_mask = emissions.mask
        xr_K_list = self.get_xr_K_list(
            meteo_ids=meteo_ids,
            n_processes=n_processes,
            sensors_index=sensors_index,
            n_sensors=n_sensors,
            emissions_mask=emissions_mask,
            n_emissions=len(emissions.prior),
        )
        print(f"loop start {time.perf_counter()-start}")
        for ti, xr_K_ti in enumerate(xr_K_list):
            ti_measurement = ti
            ti_state = ti if not emissions.prior_mode == "single_time" else 0
            # ti_meteo_id = meteo_ids[ti]
            xr_K.loc[:, ti_measurement, :, ti_state] = xr_K_ti
        print(f"loop end {time.perf_counter() - start}")
        # xr_K.coords["time_measurement"] = meteo_ids
        xr_K = utils.convert_to_ppm(xr_K)
        xr_K.attrs["units"] = "ppm"
        return xr_K

    def get_time_ids(self, time):
        time_ids = np.arange(self.time)
        self.rng.shuffle(time_ids)
        return time_ids[:time]
