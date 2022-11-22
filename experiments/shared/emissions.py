import numpy as np
import pickle
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
                prior_mode: (single_time) {single_time, constant, diurnal}
                truth_mode: (single_time) {single_time, constant, diurnal}
                tau_h: correlation length in hours.
                tau_d: correlation length in days.
        """
        self.time = config.get("time", 1)
        self.prior_mode = config.get("prior_mode", "single_time")
        self.truth_mode = config.get("truth_mode", "single_time")

        self.config_path = Path(config["emission_path"])

        # Relative emission values
        self.prior = self.get(config["prior"], config["prior_mode"])
        self.prior_variance = self.get(config["prior_variance"], config["prior_mode"])
        self.truth = self.get(config["truth"], config["truth_mode"])
        # Absolute emission values
        absolute_emissions = self.get_absolute()
        self.prior_absolute = self.prior * absolute_emissions
        self.prior_absolute_variance = self.prior_variance * absolute_emissions
        self.truth_absolute = self.truth * absolute_emissions

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

    def get(self, name, mode):
        xr_time_factor = self.get_time_factor(mode)

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

    def get_time_factor(self, mode):
        if mode == "diurnal":
            time_factor = self.get_diurnal_factors(self.time)
        elif mode == "single_time":
            time_factor = np.ones((1, n_sources))
        elif mode == "constant":
            time_factor = np.ones((self.time, n_sources))
        xr_time_factor = xr.DataArray(time_factor, dims=["time_state", "source_group"])
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
        xr_absolute_emissions = xr.DataArray(
            emissions,
            dims="source_group",
            coords={"source_group": [i + 1 for i in range(n_sources)]},
        )
        return xr_absolute_emissions

    def get_TNO(self):
        # 1. get geopandas dataframe
        # 2. normalize heating with TNO values?
        emissions = np.zeros(n_sources)
        emissions[point_source_ids + tno_cadastre_source_ids] = 1
        return emissions

    def get_TNO_variance(self):
        return self.get_TNO() ** 2

    def get_mean_TNO(self):
        # 1. get geopandas dataframe
        # 2. compute TNO emissions per m^2
        # 3. give factor to mean
        emissions = np.zeros(n_sources)
        emissions_abs = self.get_absolute() # kg/h
        mean_TNO = emissions_abs[tno_cadastre_source_ids].mean() # kg/h
        emissions[tno_cadastre_source_ids] = mean_TNO # kg/h
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
        path = Path("/mnt/data/users/rmaiwald/")
        gnfr_to_snap_df = pd.read_csv(
            path / "TNO/gnfr_to_snap.csv", header=0, index_col=0
        )
        snap_temporal_profiles_df = pd.read_csv(
            path / "TNO/snap_temporal_profiles.csv", header=[0, 1], index_col=0
        )
        with open(
            path / "GRAMM-GRAL/emissions/heat_traffic/tno_points_gdf.pickle",
            "rb",
        ) as file:
            tno_points_gdf = pickle.load(file)
        with open(
            path / "GRAMM-GRAL/emissions/heat_traffic/tno_districts_gdf.pickle",
            "rb",
        ) as file:
            tno_districts_gdf = pickle.load(file)

        point_GNFR_codes = [
            em_cat
            for em_cat in gnfr_to_snap_df["GNFR_Category"]
            if em_cat in tno_points_gdf.columns.levels[0]
        ]
        point_index_co2 = pd.MultiIndex.from_product(
            [point_GNFR_codes, ["co2_ff", "co2_bf"]]
        )
        point_base_sum = tno_points_gdf[point_index_co2].sum(axis=1)

        district_GNFR_codes = [
            em_cat
            for em_cat in gnfr_to_snap_df["GNFR_Category"]
            if em_cat in tno_districts_gdf.columns.levels[0]
        ]
        district_index_co2 = pd.MultiIndex.from_product(
            [district_GNFR_codes, ["co2_ff", "co2_bf"]]
        )
        district_base_sum = tno_districts_gdf[district_index_co2].sum(axis=1)

        time_factor = np.ones((time, n_sources))
        for t in range(time):
            hour = 1 + (t % 24)
            # Multiply emission categories with the time factors
            time_factor_snap = snap_temporal_profiles_df[("Hour", str(hour))]
            time_factor_GNFR = time_factor_snap[gnfr_to_snap_df["SNAP_link"]]
            time_factor_GNFR.index = gnfr_to_snap_df["GNFR_Category"]

            # Point sources
            # Multiply point sources with time factor
            mpoint = tno_points_gdf[point_GNFR_codes].multiply(
                time_factor_GNFR[point_GNFR_codes], level=0
            )
            # Sum over co2 emissions
            point_sum = mpoint[point_index_co2].sum(axis=1)
            time_factor[t, :2] = point_sum / point_base_sum

            # Line sources
            index = slice(line_source_ids[0], line_source_ids[-1] + 1)
            time_factor[t, index] = time_factor_GNFR["F"]

            # District sources
            # Multiply district sources with time factor
            mdistrict = tno_districts_gdf[district_GNFR_codes].multiply(
                time_factor_GNFR[district_GNFR_codes], level=0
            )
            # Sum over co2 emissions
            district_sum = mdistrict[district_index_co2].sum(axis=1)
            index = slice(tno_cadastre_source_ids[0], tno_cadastre_source_ids[-1] + 1)
            time_factor[t, index] = district_sum / district_base_sum

            # Heating sources
            index = slice(heat_cadastre_source_ids[0], heat_cadastre_source_ids[-1] + 1)
            time_factor[t, index] = time_factor_GNFR["C"]

        return time_factor

    def get_emission_factors(self, time=None):
        path = Path("/mnt/data/users/rmaiwald/")

        with open(
            path / "GRAMM-GRAL/emissions/heat_traffic/tno_points_gdf.pickle",
            "rb",
        ) as file:
            tno_points_gdf = pickle.load(file)
        with open(
            path / "GRAMM-GRAL/emissions/heat_traffic/tno_districts_gdf.pickle",
            "rb",
        ) as file:
            tno_districts_gdf = pickle.load(file)

        em_factor_df = pd.read_csv(
            path / "TNO/emission_factors.csv",
            index_col=0,
        )
        em_factor_gfnr = em_factor_df["Emission_Factor"]

        point_index_co2 = pd.MultiIndex.from_product(
            [em_factor_gfnr.index, ["co2_ff", "co2_bf"]]
        )

        point_base_sum = tno_points_gdf[point_index_co2].sum(axis=1)
        district_index_co2 = pd.MultiIndex.from_product(
            [em_factor_gfnr.index, ["co2_ff", "co2_bf"]]
        )
        district_base_sum = tno_districts_gdf[district_index_co2].sum(axis=1)

        em_factor = np.zeros(n_sources)
        # Point sources
        # Multiply point sources with emission factor
        mpoint = tno_points_gdf[em_factor_gfnr.index].multiply(em_factor_gfnr, level=0)
        # Sum over co2 emissions
        point_sum = mpoint[point_index_co2].sum(axis=1)
        index = slice(point_source_ids[0], point_source_ids[-1] + 1)
        em_factor[index] = point_sum / point_base_sum

        # Line sources
        index = slice(line_source_ids[0], line_source_ids[-1] + 1)
        em_factor[index] = em_factor_gfnr["F1"]

        # District sources
        # Multiply district sources with emission factor
        mdistrict = tno_districts_gdf[em_factor_gfnr.index].multiply(
            em_factor_gfnr, level=0
        )
        # Sum over co2 emissions
        district_sum = mdistrict[district_index_co2].sum(axis=1)
        index = slice(tno_cadastre_source_ids[0], tno_cadastre_source_ids[-1] + 1)
        em_factor[index] = district_sum / district_base_sum

        # Heating sources
        index = slice(heat_cadastre_source_ids[0], heat_cadastre_source_ids[-1] + 1)
        em_factor[index] = em_factor_gfnr["C"]

        return em_factor

    def to_xr(self, array):
        if len(array.shape) == 1:
            xr_array = xr.DataArray(
                data=array,
                coords=utils.stack_xr(self.prior).coords,
            )
        elif len(array.shape) == 2:
            xr_array = xr.DataArray(
                data=array,
                dims=["state", "state_2"],
                coords=utils.stack_xr(self.prior).coords,
            )
            # Add coordinates to second dimension
            multi_index = xr_array.coords["state"]
            xr_array.coords["state_2"] = pd.MultiIndex.from_tuples(
                multi_index.values, names=["source_group_2", "time_state_2"]
            )
        else:
            xr_array = None
        return xr_array
