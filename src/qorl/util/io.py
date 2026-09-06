from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from pydantic import JsonValue


def write_json[Value: JsonValue](
    path: Path, value: Value | list[Value] | dict[str, Value]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temporary:
        temporary.write(content)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def display_path(repository: Path, path: Path) -> str:
    try:
        return str(path.relative_to(repository))
    except ValueError:
        return str(path)
