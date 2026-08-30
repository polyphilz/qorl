from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.utils.protocol_dataset import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate every transcript and artifact in protocol-sft-v1."
    )
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("outputs/sft/protocol-sft-v1"),
    )
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    dataset = arguments.dataset
    if not dataset.is_absolute():
        dataset = repository / dataset
    print(json.dumps(validate_dataset(repository, dataset), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
