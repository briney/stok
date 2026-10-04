"""The same training command for python -m and distributed launchers."""

from stok.cli.cli import train_cmd


def main():
    train_cmd(prog_name="python -m stok.train")


if __name__ == "__main__":
    main()
