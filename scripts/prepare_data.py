import argparse
import subprocess
import sys


def run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main(dataset_config: str, train_config: str) -> None:
    run_cmd(
        [
            sys.executable,
            "-m",
            "src.data.build_dataset_index",
            "--config",
            dataset_config,
        ]
    )

    run_cmd(
        [
            sys.executable,
            "-m",
            "src.data.build_boundary_masks",
            "--config",
            dataset_config,
        ]
    )

    run_cmd(
        [
            sys.executable,
            "-m",
            "src.data.build_patch_index",
            "--dataset-config",
            dataset_config,
            "--train-config",
            train_config,
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="configs/dataset.yaml",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default="configs/train.yaml",
    )
    args = parser.parse_args()

    main(
        dataset_config=args.dataset_config,
        train_config=args.train_config,
    )