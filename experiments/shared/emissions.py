import numpy as np
import pickle
import xarray as xr
import pandas as pd
from pathlib import Path

from ..shared import utilities as utils


# Indices with the source group number in GRAL (1-based indexing)
point_source_index = np.array(range(1, 3))
# 13-15 have no emissions
line_source_index = np.array(range(3, 25))
# 35-37 have no emissions
tno_cadastre_source_index = np.concatenate([range(25, 35), range(38, 47)])
heat_cadastre_source_index = np.array(range(47, 69))
all_index = np.array(range(1, 69))

# Indices for 0-based indexing
point_source_ids = point_source_index - 1
line_source_ids = line_source_index - 1
tno_cadastre_source_ids = tno_cadastre_source_index - 1
heat_cadastre_source_ids = heat_cadastre_source_index - 1
all_ids = all_index - 1

n_sources = len(all_ids)
point_list = None


class Emissions:
    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Required config keys:
                prior: prior emission name 
                    {
                        TNO, 
                        mean_TNO,
                        mean_TNO_with_points,
                        combined_emissions,
                    }
                prior_variance: prior variance name 
                    {
                        TNO_variance, 
                        mean_TNO_variance,
                    }
                truth: true emission name 
                    {
                        TNO, 
                        mean_TNO, 
                        mean_TNO_with_points,
                        combined_emissions,
                    }
                emission_path: path to GRAL emission files.
            Optional config keys (default):
                time: (1) number of hourly measurements.
                prior_mode: (single_time) {
                        single_time,
                        constant,
                        diurnal,
                        single_diurnal,
                        constant_diurnal,
                    }
                truth_mode: (single_time) {
                        single_time,
                        constant,
                        diurnal,
                        single_diurnal,
                        constant_diurnal,
                    }
                tau_h: correlation length in hours.
                tau_d: correlation length in days.
        """
        self.time = config.get("time", 1)
        self.prior_mode = config.get("prior_mode", "single_time")
        self.truth_mode = config.get("truth_mode", "single_time")

        self.config_path = Path(config["emission_path"])

        # Move to config
        source_group_path = Path(
            "/mnt/data/users/rmaiwald/GRAMM-GRAL/emissions/pickle_jar/source_groups_infos.csv"
        )
        self.source_group_df = pd.read_csv(
            source_group_path,
            index_col=0,
        )
        self.absolute_emissions = self.get_absolute()

        # Relative emission values
        self.prior = self.get(config["prior"], config["prior_mode"])
        self.prior_variance = self.get(config["prior_variance"], config["prior_mode"])
        self.truth = self.get(config["truth"], config["truth_mode"])
        # Absolute emission values
        self.prior_absolute = self.prior * self.absolute_emissions
        self.prior_absolute_variance = self.prior_variance * self.absolute_emissions
        self.truth_absolute = self.truth * self.absolute_emissions

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
            "mean_TNO_with_points": self.get_mean_TNO_with_points,
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
        elif mode == "single_diurnal":
            time_factor = self.get_diurnal_factors(np.min([24, self.time]))
        elif mode == "single_time":
            time_factor = np.ones((1, n_sources))
        elif mode == "constant":
            time_factor = np.ones((self.time, n_sources))
        elif mode == "constant_diurnal":
            time_factor = np.ones((np.min([24, self.time]), n_sources))
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
        emissions[point_source_ids] = 1
        emissions[tno_cadastre_source_ids] = 1
        return emissions

    def get_TNO_variance(self):
        return self.get_TNO() ** 2

    def get_mean_TNO(self):
        # 1. get geopandas dataframe
        path = Path("/mnt/data/users/rmaiwald/")
        with open(
            path / "GRAMM-GRAL/emissions/heat_traffic/tno_districts_gdf.pickle",
            "rb",
        ) as file:
            tno_districts_gdf = pickle.load(file)

        # 2. compute TNO emissions per m^2
        area = tno_districts_gdf["area"]
        emissions_tno_cadastre = self.absolute_emissions.loc[tno_cadastre_source_index]
        emissions_tno_cadastre = emissions_tno_cadastre.sum().values * area / area.sum()
        # Normalize
        emissions_tno_cadastre = (
            emissions_tno_cadastre
            / self.absolute_emissions.loc[tno_cadastre_source_index]
        )
        # 3. Create values for all other source groups
        emissions = np.zeros(n_sources)
        emissions[tno_cadastre_source_ids] = emissions_tno_cadastre
        return emissions

    def get_mean_TNO_with_points(self):
        emissions = self.get_mean_TNO()
        emissions[point_source_ids] = 1
        return emissions

    def get_mean_TNO_variance(self):
        emissions = self.get_mean_TNO() ** 2
        emissions[point_source_ids] = 1
        emissions[line_source_ids] = 1 # Make reasonable var
        emissions[heat_cadastre_source_ids] = 1 # Make reasonable var
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
        # Replace TNO traffic and stationary combustion with the data from
        # Geoinformatics and from the Stadtwerke
        emissions = np.zeros(n_sources)
        emissions[point_source_ids] = 1
        emissions[line_source_ids] = 1
        emissions[heat_cadastre_source_ids] = 1
        # Compute new emissions for TNO area sources
        path = Path("/mnt/data/users/rmaiwald/")
        with open(
            path / "GRAMM-GRAL/emissions/heat_traffic/tno_districts_gdf.pickle",
            "rb",
        ) as file:
            tno_districts_gdf = pickle.load(file)
        heating_emission_categories = ["A", "B", "C"]
        traffic_emission_categories = ["F1", "F2", "F3"]
        tno_emission_categories = ["D", "E", "G", "H", "I", "J", "L"]
        categories = (
            heating_emission_categories
            + traffic_emission_categories
            + tno_emission_categories
        )

        tno_index = pd.MultiIndex.from_product(
            [tno_emission_categories, ["co2_ff", "co2_bf"]]
        )
        total_index = pd.MultiIndex.from_product([categories, ["co2_ff", "co2_bf"]])
        # tno_emission
        tno_sum = tno_districts_gdf[tno_index].sum(axis="columns")
        total_sum = tno_districts_gdf[total_index].sum(axis="columns")
        emissions[tno_cadastre_source_ids] = tno_sum / total_sum
        return emissions

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
            time_factor[t, line_source_ids] = time_factor_GNFR["F"]

            # District sources
            # Multiply district sources with time factor
            mdistrict = tno_districts_gdf[district_GNFR_codes].multiply(
                time_factor_GNFR[district_GNFR_codes], level=0
            )
            # Sum over co2 emissions
            district_sum = mdistrict[district_index_co2].sum(axis=1)
            time_factor[t, tno_cadastre_source_ids] = district_sum / district_base_sum

            # Heating sources
            time_factor[t, heat_cadastre_source_ids] = time_factor_GNFR["C"]

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
        em_factor[point_source_ids] = point_sum / point_base_sum

        # Line sources
        em_factor[line_source_ids] = em_factor_gfnr["F1"]

        # District sources
        # Multiply district sources with emission factor
        mdistrict = tno_districts_gdf[em_factor_gfnr.index].multiply(
            em_factor_gfnr, level=0
        )
        # Sum over co2 emissions
        district_sum = mdistrict[district_index_co2].sum(axis=1)
        em_factor[tno_cadastre_source_ids] = district_sum / district_base_sum

        # Heating sources
        em_factor[heat_cadastre_source_ids] = em_factor_gfnr["C"]

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
