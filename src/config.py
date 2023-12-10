import yaml


class TrainingConfig(dict):
    def __init__(self, config_file):
        super().__init__(self._read_config(config_file))

    def _read_config(self, config_file):
        with open(config_file, 'r') as file:
            try:
                return yaml.safe_load(file) or {}
            except yaml.YAMLError as exc:
                print(exc)
                return {}


if __name__ == '__main__':
    config = TrainingConfig('config.yaml')
    print(config)
    print(config.get("batch_size"))
    print(type(config))