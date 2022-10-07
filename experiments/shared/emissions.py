import numpy as np


import xarray as xr
import pandas as pd


from ggpymanager import Reader


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


class Emissions:
    def __init__(self, config=None):
        self.reader = Reader(
            config["reader"]["catalog_path"],
            config["reader"]["config_path"],
            config["reader"]["simulation_path"],
        )

        self.prior = self.get(config["emissions"]["prior"])
        self.prior_absolute = self.prior * self.get_absolute()
        self.prior_uncertainty = self.get(config["emissions"]["prior_uncertainty"])
        self.prior_absolute_uncertainty = self.prior_uncertainty * self.get_absolute()
        self.truth = self.get(config["emissions"]["truth"])
        self.truth_absolute = self.truth * self.get_absolute()

    def get_source_group_dict(self):
        pass

    def get(self, name):
        position = name.find("_diurnal")
        if position >= 0:
            diurnal = self.get_diurnal_factors()
            name = name[:position]
        else:
            diurnal = 1.0

        func = {
            "TNO": self.get_TNO,
            "TNO_uncertainty": self.get_TNO_uncertainty,
            "mean_TNO": self.get_mean_TNO,
            "mean_TNO_uncertainty": lambda: 1.0,
            "heat": self.get_heat_emissions,
            "traffic": self.get_traffic_emissions,
            "combined_emissions": self.get_combined_emissions,
        }

        return diurnal * func[name]()

    def get_absolute(self):
        emissions = np.zeros(n_sources)
        file_list = ["point.dat", "line.dat", "cadastre.dat"]
        col_list = ["emission [kg/h]", "emission [kg/h/km]", "emission [kg/h]"]
        for file_name, col_name in zip(file_list, col_list):
            file_path = self.reader.config_path / file_name
            # Only "cadastre.dat" has no comments 
            skiprows = 0 if (file_name == "cadastre.dat") else 1
            df = pd.read_csv(file_path, skiprows=skiprows, index_col=False)
            for source_group in df["source group"].unique():
                source_id = source_group - 1
                # Sum all sources from source group
                if file_name == "line.dat":
                    # Multiply emissions with street length
                    m_to_km = 1 / 1000
                    length = np.sqrt(
                        (df["x1"] - df["x2"]) ** 2
                        + (df["y1"] - df["y2"]) ** 2
                        + (df["z1"] - df["z2"]) ** 2
                    ) * m_to_km
                    emissions[source_id] = (
                        df[df["source group"] == source_group][col_name] * length
                    ).sum()
                else:
                    # Just sum
                    emissions[source_id] = df[df["source group"] == source_group][
                        col_name
                    ].sum()
        return emissions

    def get_TNO(self):
        # 1. get geopandas dataframe
        # 2. normalize heating with TNO values?
        emissions = np.zeros(n_sources)
        emissions[point_source_ids + tno_cadastre_source_ids] = 1
        return emissions

    def get_TNO_uncertainty(self):
        return np.ones(n_sources)

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

    def get_diurnal_factors(self):
        return np.ones((24, n_sources))  # 24 hours
