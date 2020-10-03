import os

from omegaconf import OmegaConf

RELATIVE_PATH = "./medzoo."
MODEL_PATH = "./medzoo/models/"
DEFAULT_CONFIG = "defaults.yaml"


class ConfigReader:

    @staticmethod
    def read_config(config_path: str, model_name: str):
        """

        Args:
            model_name:
            config_path:

        Returns:

        """
        return ConfigReader.load_config(config_path, model_name)

    @staticmethod
    def load_config(config_path: str, model_name: str):
        mapping = {}

        if not config_path :
            return ConfigReader.load_default_config(model_name)

        try:
            mapping = OmegaConf.load(config_path)
        except FileNotFoundError as e:
            print("aaaa")
            # Check if this file might be relative to root?
            relative_path = os.path.abspath(os.path.join(RELATIVE_PATH, config_path))
            if not os.path.exists(relative_path):
                raise e
            else:
                mapping = OmegaConf.load(relative_path)

        if mapping is None:
            return ConfigReader.load_default_config(model_name)

        default_mapping = ConfigReader.load_default_config(model_name)

        mapping = OmegaConf.merge(default_mapping,mapping)

        return mapping

    @staticmethod
    def load_default_config(model_name: str):

        default_config_path = os.path.join(MODEL_PATH, model_name, DEFAULT_CONFIG)
        return OmegaConf.load(default_config_path)
