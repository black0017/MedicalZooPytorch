from omegaconf import OmegaConf


class ConfigReader:

    @staticmethod
    def read_config(config_path: str):
        """

        Args:
            config_path:

        Returns:

        """
        return OmegaConf.load(config_path)
