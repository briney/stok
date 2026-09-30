import sys
from importlib.resources import as_file, files

from hydra import compose, initialize_config_dir

from stok.cli.train import run_training


def main():
    overrides = list(sys.argv[1:])
    with as_file(files("stok").joinpath("configs")) as cfg_dir:
        with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
            cfg = compose(config_name="config", overrides=overrides)
    run_training(cfg)


if __name__ == "__main__":
    main()
