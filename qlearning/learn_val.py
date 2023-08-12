import os
import time

from deepq.playground.configs.manager import ConfigManager


def run(config_name, model_name=None):
    cfg = ConfigManager.load(config_name)

    if model_name is None:
        model_name = '-'.join([
            cfg.env_name.lower(),
            cfg.policy_name.replace('_', '-'),
            os.path.splitext(os.path.basename(config_name))[
                0] if config_name else 'default',
            str(int(time.time()))
        ])

    model_name = model_name.lower()
    cfg.start_training(model_name)
    cfg.save("val-v5")


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))

    config_name = f"{dir_path}/setting/val_high.json"
    run(config_name, model_name="valkyrie-v5")
