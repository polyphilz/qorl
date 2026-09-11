"""Persist native binary records without losing sampling masks or float precision."""

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import msgpack
from pydantic import BaseModel
from verifiers.v1.serve import encoding


class NativeCodec(Protocol):
    def packb(
        self, o: object, /, *, default: Callable[[object], object], use_bin_type: bool
    ) -> bytes | None: ...


class NativeEncoding(Protocol):
    def msgpack_encoder(self, obj: object) -> object: ...


CODEC: NativeCodec = msgpack
ENCODING: NativeEncoding = encoding


def write_native(path: Path, record: BaseModel) -> None:
    """Use the same serialization as the native EnvServer response."""
    payload = CODEC.packb(
        record.model_dump(mode="python"),
        default=ENCODING.msgpack_encoder,
        use_bin_type=True,
    )
    if payload is None:
        raise RuntimeError("native record encoding returned no bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        staged = Path(temporary.name)
    staged.replace(path)
