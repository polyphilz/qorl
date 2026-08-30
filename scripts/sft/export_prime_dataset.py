from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.utils.protocol_demo import validate_protocol_demo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the validated QORL demonstration to Prime-RL SFT data."
    )
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--demo",
        type=Path,
        default=Path("outputs/sft/protocol-demo-v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/sft/prime-protocol-demo"),
    )
    arguments = parser.parse_args()

    repository = arguments.repository.resolve()
    demo_path = arguments.demo
    output_dir = arguments.output
    if not demo_path.is_absolute():
        demo_path = repository / demo_path
    if not output_dir.is_absolute():
        output_dir = repository / output_dir

    document = json.loads(demo_path.read_text())
    validation = validate_protocol_demo(document, repository)
    row = {
        "messages": document["messages"],
        "tools": json.dumps(document["tools"], sort_keys=True),
    }
    encoded = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train.jsonl").write_bytes(encoded)
    manifest = {
        "schema_version": 1,
        "format": "prime-rl-messages-tools",
        "examples": 1,
        "source": str(demo_path.relative_to(repository)),
        "source_canonical_sha256": validation["canonical_sha256"],
        "train_jsonl_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"output": str(output_dir), **manifest}, indent=2))


if __name__ == "__main__":
    main()
