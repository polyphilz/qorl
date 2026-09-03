from __future__ import annotations

import argparse
import json
from pathlib import Path

from qorl.sft.assemble import validate_dataset as validate_v1
from qorl.sft.build_protocol_dataset import validate_dataset as validate_v2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate every transcript and artifact in a protocol SFT dataset."
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
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    validator = (
        validate_v2 if manifest.get("dataset_id") == "protocol-sft-v2" else validate_v1
    )
    print(json.dumps(validator(repository, dataset), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
