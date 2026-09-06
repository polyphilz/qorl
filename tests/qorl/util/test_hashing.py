import hashlib
from pathlib import Path

import pytest

from qorl.util.hashing import sha256_bytes, sha256_file, sha256_json


def test_sha256_bytes_matches_known_digest() -> None:
    assert sha256_bytes(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


@pytest.mark.parametrize(
    "content", [b"", b"abc" * (1024 * 1024)], ids=["empty", "multiple-blocks"]
)
def test_sha256_file_hashes_all_bytes(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(content)

    assert sha256_file(path) == hashlib.sha256(content).hexdigest()


def test_sha256_json_preserves_canonical_encoding() -> None:
    expected = sha256_bytes(b'{"a":[null,true,1,2.5],"z":"\\u00e9"}')

    assert sha256_json({"z": "é", "a": [None, True, 1, 2.5]}) == expected
    assert sha256_json({"a": [None, True, 1, 2.5], "z": "é"}) == expected


def test_sha256_json_accepts_typed_containers() -> None:
    values: list[str] = ["a", "b"]
    counts: dict[str, int] = {"a": 1}

    assert sha256_json(values) == sha256_bytes(b'["a","b"]')
    assert sha256_json(counts) == sha256_bytes(b'{"a":1}')
