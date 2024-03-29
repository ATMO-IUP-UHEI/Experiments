from pathlib import Path

import yaml
import pickle

from ggpymanager import Reader

from .sensors import Sensors
from .emissions import Emissions
from .transport import Transport


class Experiment:
    def __init__(self, config_path) -> None:
        self.config = self.get_config(config_path)

        self.sensors_config = self.config["sensors"] | {
            "seed": self.config.get("seed", 0),
            "time": self.config.get("time", 1),
        }
        self.sensors = Sensors(self.sensors_config)

        self.emissions_config = self.config["emissions"] | {
            "emission_path": self.config["reader"]["config_path"],
            "time": self.config["time"],
        }
        self.emissions = Emissions(self.emissions_config)

        self.transport_config = (
            self.config["transport"]
            | self.config["reader"]
            | {
                "seed": self.config.get("seed", 0),
                "time": self.config.get("time", 1),
            }
        )
        self.transport = Transport(self.transport_config)

        # self.figs = {}
        self.data = {}

    def get_config(self, config_path):
        # parent_path = Path(__file__).resolve().parent
        # config_path = parent_path / "config.yaml"
        with open(config_path) as file:
            config = yaml.safe_load(file)
        return config

    """ def pickle_figs(self):
        dir_name = self.__class__.__name__
        jar_path = Path(self.config["paths"]["fig_jar"]) / dir_name
        if not jar_path.exists():
            jar_path.mkdir()
        self.pickle_objects(jar_path, self.figs) """

    def pickle_data(self):
        dir_name = self.__class__.__name__
        jar_path = Path(self.config["paths"]["data_jar"]) / dir_name
        if not jar_path.exists():
            jar_path.mkdir()
        self.pickle_objects(jar_path, self.data)

    def pickle_objects(self, jar_path, obj_dict):
        for obj_name, obj in obj_dict.items():
            with open(jar_path / f"{obj_name}.pickle", "xb") as file:
                pickle.dump(obj, file=file)

    """ def load_figs(self):
        dir_name = self.__class__.__name__
        jar_path = Path(self.config["paths"]["fig_jar"]) / dir_name
        self.figs = self.unpickle_objects(jar_path) """

    def load_data(self):
        dir_name = self.__class__.__name__
        jar_path = Path(self.config["paths"]["data_jar"]) / dir_name
        self.data = self.unpickle_objects(jar_path)

    def unpickle_objects(self, jar_path):
        obj_dict = {}
        for file_path in jar_path.iter_dir():
            obj_name = file_path.stem
            with open(file_path, "rb") as file:
                obj = pickle.load(file=file)
            obj_dict[obj_name] = obj
        return obj_dict
