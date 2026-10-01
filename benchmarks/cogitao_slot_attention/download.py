from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "yassinetb/COGITAO"
REVISION = "25e347f9873acbaba98bdaef7e64ba430a2890c2"
EXPECTED_FILES = 125


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the five COGITAO CompGen benchmark settings."
    )
    parser.add_argument("--output", type=Path, default=Path("data/cogitao/files"))
    args = parser.parse_args()

    destination = args.output.resolve()
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        local_dir=destination,
        allow_patterns=["CompGen/exp_setting_*/*/*.parquet"],
    )
    files = sorted(
        (destination / "CompGen").glob("exp_setting_*/experiment_*/*.parquet")
    )
    if len(files) != EXPECTED_FILES:
        raise RuntimeError(
            f"Expected {EXPECTED_FILES} parquet files, found {len(files)} "
            f"in {destination}."
        )
    size = sum(path.stat().st_size for path in files)
    print(f"COGITAO CompGen ready: {len(files)} files, {size / 2**20:.2f} MiB")
    print(destination / "CompGen")


if __name__ == "__main__":
    main()
