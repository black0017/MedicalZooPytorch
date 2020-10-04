import os
from enum import Enum

from omegaconf import OmegaConf

RELATIVE_PATH = "./medzoo."
MODEL_PATH = "./medzoo/models/"
DATASET_PATH ="./medzoo/datasets/"
DEFAULT_CONFIG = "defaults.yaml"
PATHS_MAPPING_PATH = "./medzoo/utils/paths_config.yaml"

class ConfigType(Enum):
    MODEL_CONFIG = 1
    DATASET_CONFIG = 2

    @staticmethod
    def get_path(config_type):
        if config_type == ConfigType.MODEL_CONFIG:
            return MODEL_PATH
        else:
            return DATASET_PATH

class ConfigReader:

    @staticmethod
    def read_config(model_config_path: str, dataset_config_path: str, model_name: str, dataset_name: str):
        """

        Args:
            model_name:
            config_path:

        Returns:

        """
        dataset_config = ConfigReader.load_config(dataset_config_path, dataset_name, ConfigType.DATASET_CONFIG)
        model_config = ConfigReader.load_config(model_config_path, model_name, ConfigType.MODEL_CONFIG)

        config =  OmegaConf.merge(dataset_config, model_config)
        print(config)
        return config

    @staticmethod
    def load_config(config_path: str, model_name: str, config_type: ConfigType):
        mapping = None

        if not config_path :
            return ConfigReader.load_default_config(model_name, config_type)

        try:
            mapping = OmegaConf.load(config_path)
        except FileNotFoundError as e:
            # Check if this file might be relative to root?
            relative_path = os.path.abspath(os.path.join(RELATIVE_PATH, config_path))
            if not os.path.exists(relative_path):
                raise e
            else:
                mapping = OmegaConf.load(relative_path)

        if mapping is None:
            return ConfigReader.load_default_config(model_name,config_type)

        default_mapping = ConfigReader.load_default_config(model_name, config_type)

        mapping = OmegaConf.merge(default_mapping,mapping)

        return mapping

    @staticmethod
    def load_default_config(name: str, config_type: ConfigType):

        default_config_path = os.path.join(ConfigReader.get_config_path(name,config_type), DEFAULT_CONFIG)
        return OmegaConf.load(default_config_path)

    @staticmethod
    def get_config_path(name:str, config_type: ConfigType):
        default_path = ConfigType.get_path(config_type)
        path_mapping = OmegaConf.load(PATHS_MAPPING_PATH)
        return os.path.join(default_path + path_mapping[name])


