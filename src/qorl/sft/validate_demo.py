from __future__ import annotations

import argparse
import json
from pathlib import Path

from qorl.sft.validate import validate_protocol_demo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mechanically validate one QORL demonstration."
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path("outputs/sft/protocol-demo-v1.json"),
    )
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    path = arguments.path
    if not path.is_absolute():
        path = repository / path
    document = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(validate_protocol_demo(document, repository), indent=2))


if __name__ == "__main__":
    main()
