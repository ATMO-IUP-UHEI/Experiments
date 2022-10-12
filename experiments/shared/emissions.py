import numpy as np


import xarray as xr
import pandas as pd
from pathlib import Path

from ..shared import utilities as utils

point_source_ids = [0, 1]
line_source_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
tno_cadastre_source_ids = [
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
]
heat_cadastre_source_ids = [
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    58,
]

n_sources = 59
point_list = None

class Emissions:
    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Required config keys:
                prior: prior emission name {TNO, mean_TNO, combined_emissions}.
                prior_variance: prior variance name {TNO_variance, mean_TNO_variance}.
                truth: prior emission name {TNO, mean_TNO, combined_emissions}.
                emission_path: path to GRAL emission files.
            Optional config keys (default):
                time: (1) number of hourly measurements.
                mode: (single_time) {single_time, constant, diurnal}
                tau_h: correlation length in hours.
                tau_d: correlation length in days.
        """
        self.time = config.get("time", 1)
        self.mode = config.get("mode", "single_time")

        self.config_path = Path(config["emission_path"])

        # Relative emission values
        self.prior = self.get(config["prior"])
        self.prior_variance = self.get(config["prior_variance"])
        self.truth = self.get(config["truth"])
        # Absolute emission values
        self.prior_absolute = self.prior * self.get_absolute()
        self.prior_absolute_variance = self.prior_variance * self.get_absolute()
        self.truth_absolute = self.truth * self.get_absolute()

        self.mask = (self.prior.isel(time_state=0) != 0.0) | (
            self.truth.isel(time_state=0) != 0.0
        )

        # Filter only used source groups
        self.prior = self.prior[self.mask]
        self.prior_variance = self.prior_variance[self.mask]
        if "correlation" in config:
            self.prior_covariance = utils.compute_prior_covariance(
                xr_prior_var=self.prior_variance, 
                tau_h=config["correlation"]["tau_h"], 
                tau_d=config["correlation"]["tau_d"],
            )
        else:
            self.prior_covariance = self.prior_variance
        self.truth = self.truth[self.mask]
        # Absolute emission values
        self.prior_absolute = self.prior_absolute[self.mask]
        self.prior_absolute_variance = self.prior_absolute_variance[self.mask]
        self.truth_absolute = self.truth_absolute[self.mask]

    def get_source_group_dict(self):
        pass

    def get(self, name):
        xr_time_factor = self.get_time_factor()

        func = {
            "TNO": self.get_TNO,
            "TNO_variance": self.get_TNO_variance,
            "mean_TNO": self.get_mean_TNO,
            "mean_TNO_variance": self.get_mean_TNO_variance,
            "heat": self.get_heat_emissions,
            "traffic": self.get_traffic_emissions,
            "combined_emissions": self.get_combined_emissions,
        }
        xr_emission = xr.DataArray(
            data=func[name](),
            dims=["source_group"],
            coords={"source_group": [i + 1 for i in range(n_sources)]},
        )
        xr_emission = xr_emission * xr_time_factor
        # xr_emission = xr_emission.stack(state=["time_state", "source_group"])
        return xr_emission

    def get_time_factor(self):
        if self.mode == "diurnal":
            time_factor = self.get_diurnal_factors(self.time)
        elif self.mode == "single_time":
            time_factor = [1.0]
        elif self.mode == "constant":
            time_factor = np.ones(self.time)
        xr_time_factor = xr.DataArray(time_factor, dims=["time_state"])
        return xr_time_factor

    def get_absolute(self):
        emissions = np.zeros(n_sources)
        file_list = ["point.dat", "line.dat", "cadastre.dat"]
        col_list = ["emission [kg/h]", "emission [kg/h/km]", "emission [kg/h]"]
        for file_name, col_name in zip(file_list, col_list):
            file_path = self.config_path / file_name
            if file_path.exists():
                # Only "cadastre.dat" has no comments
                skiprows = 0 if (file_name == "cadastre.dat") else 1
                df = pd.read_csv(file_path, skiprows=skiprows, index_col=False)
                for source_group in df["source group"].unique():
                    source_id = source_group - 1
                    # Sum all sources from source group
                    if file_name == "line.dat":
                        # Multiply emissions with street length
                        m_to_km = 1 / 1000
                        length = (
                            np.sqrt(
                                (df["x1"] - df["x2"]) ** 2
                                + (df["y1"] - df["y2"]) ** 2
                                + (df["z1"] - df["z2"]) ** 2
                            )
                            * m_to_km
                        )
                        emissions[source_id] = (
                            df[df["source group"] == source_group][col_name] * length
                        ).sum()
                    else:
                        # Just sum
                        emissions[source_id] = df[df["source group"] == source_group][
                            col_name
                        ].sum()
        xr_absolute_emissions = xr.DataArray(emissions, dims="source_group")
        return xr_absolute_emissions

    def get_TNO(self):
        # 1. get geopandas dataframe
        # 2. normalize heating with TNO values?
        emissions = np.zeros(n_sources)
        emissions[point_source_ids + tno_cadastre_source_ids] = 1
        return emissions

    def get_TNO_variance(self):
        return self.get_TNO()**2

    def get_mean_TNO(self):
        # 1. get geopandas dataframe
        # 2. compute TNO emissions per m^2
        # 3. give factor to mean
        emissions = np.zeros(n_sources)
        emissions_abs = self.get_absolute()
        mean_TNO = emissions_abs[tno_cadastre_source_ids].mean()
        emissions[tno_cadastre_source_ids] = mean_TNO
        # Normalize
        emissions[tno_cadastre_source_ids] /= emissions_abs[tno_cadastre_source_ids]
        return emissions

    def get_mean_TNO_variance(self):
        emissions = self.get_mean_TNO() ** 2
        emissions[point_source_ids] = 1
        return emissions

    def get_heat_emissions(self):
        emissions = np.zeros(n_sources)
        emissions[heat_cadastre_source_ids] = 1
        return emissions

    def get_traffic_emissions(self):
        emissions = np.zeros(n_sources)
        emissions[line_source_ids] = 1
        return emissions

    def get_combined_emissions(self):
        return np.ones(n_sources)

    def get_diurnal_factors(self, time):
        return np.ones(time)  # 24 hours

    def to_xr(self, array):
        return xr.DataArray(
            data=array.reshape(self.prior.shape),
            coords=self.prior.coords,
        )
