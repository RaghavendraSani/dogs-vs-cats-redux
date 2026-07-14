from pathlib import Path
import yaml


class ConfigNode:
    """
    allows dictionary values to be accessed using dot notation.
    example:
        config.training.epochs
        config.model.name
    """
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                value = ConfigNode(value)
            setattr(self, key, value)

    def __repr__(self):
        return str(self.__dict__)


class ConfigLoader:
    """
    loads and merges all YAML configuration files.
    """
    CONFIG_FILES = [
        "paths.yaml",
        "dataset.yaml",
        "model.yaml",
        "training.yaml",
        "inference.yaml",
        "experiment.yaml",
    ]

    def __init__(self, config_dir="configs"):
        self.config_dir = Path(config_dir)

    def load_yaml(self, filename):
        filepath = self.config_dir / filename
        with open(filepath, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def deep_merge(self, base, new):
        """
        recursively merges dictionaries.
        """
        for key, value in new.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                self.deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def validate(self, config):
        """
        validates that all required keys are present.
        """
        required_keys = [
            "experiment.name",
            "model.name",
            "training.epochs",
            "dataset.image_size",
            "dataloader.batch_size",
        ]
        for key in required_keys:
            current = config
            for part in key.split("."):
                if part not in current:
                    raise KeyError(f"missing required configuration key: '{key}'")
                current = current[part]

    def load(self):
        config = {}
        for file in self.CONFIG_FILES:
            yaml_data = self.load_yaml(file)
            config = self.deep_merge(config, yaml_data)
        return ConfigNode(config)


def load_config():
    """
    convenience function used throughout the project.
    """
    loader = ConfigLoader()
    return loader.load()