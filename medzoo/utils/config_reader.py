from omegaconf import OmegaConf


class ConfigReader:

    @staticmethod
    def read_config(config_path: str):
        return OmegaConf.load(config_path)
