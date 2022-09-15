from pathlib import Path

import yaml
import pickle


class Experiment():
    def __init__(self) -> None:
        self.config = self.get_config()

        self.figs = {}
        self.data = {}

    def get_config(self):
        parent_path = Path(__file__).resolve().parent
        config_path = parent_path / "config.yaml"
        with open(config_path) as file:
            config = yaml.safe_load(file)  
        return config

    def pickle_figs(self):
        dir_name = self.__class__.__name__
        jar_path = Path(self.config["paths"]["fig_jar"]) / dir_name
        if not jar_path.exists():
            jar_path.mkdir()
        self.pickle_objects(jar_path, self.data)

    def pickle_data(self):
        dir_name = self.__class__.__name__
        jar_path = Path(self.config["paths"]["data_jar"]) / dir_name
        if not jar_path.exists():
            jar_path.mkdir()
        self.pickle_objects(jar_path, self.figs)
   
    def pickle_objects(self, jar_path, obj_dict):
        for obj_name, obj in obj_dict.items():
            with open( jar_path / f"{obj_name}.pickle", "xb") as file:
                pickle.dump(obj, file=file)

    def load_figs(self):
        dir_name = self.__class__.__name__
        jar_path = Path(self.config["paths"]["fig_jar"]) / dir_name
        self.figs = self.unpickle_objects(jar_path)

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
