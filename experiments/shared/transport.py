import numpy as np


from ggpymanager import Reader


class Transport:
    def __init__(self, config):
        self.reader = Reader(
            config["reader"]["catalog_path"],
            config["reader"]["config_path"],
            config["reader"]["simulation_path"],
        )

    def get_K(self, meteo_id, sensors, emissions):
        con_dict = self.reader.get_concentration(meteo_id)

        # make forward model K
        sensor_index = sensors.get_index()
        height_list = np.unique(sensor_index[2])
        K = np.zeros((sensors.n_sensors, len(emissions.prior)))
        for key, val in con_dict.items():
            height = int(key[0]) - 1
            if height in height_list:
                col_id = int(key[1:]) - 1  # Correct because source groups start from 1
                if col_id < len(emissions.prior):
                    if (emissions.prior[col_id] != 0.0) or (
                        emissions.truth[col_id] != 0.0
                    ):
                        if val.size > 1:
                            index = sensor_index[2] == height
                            K[index, col_id] = val[
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
        K *= factor
        return K
