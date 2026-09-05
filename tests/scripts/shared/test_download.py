import inspect
import io
from email.message import Message
from http.client import HTTPResponse
from pathlib import Path
from unittest.mock import Mock
from urllib.response import addinfourl

import pytest

from scripts.shared import download as downloader


def http_response(contents: bytes) -> HTTPResponse:
    wire = (
        f"HTTP/1.1 200 OK\r\nContent-Length: {len(contents)}\r\n\r\n".encode()
        + contents
    )
    socket = Mock()
    socket.makefile.return_value = io.BytesIO(wire)
    response = HTTPResponse(socket)
    response.begin()
    return response


def test_cached_file_is_preserved_without_network(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "archive.tgz"
    contents = b"archive"
    target.write_bytes(contents)
    network = Mock(side_effect=AssertionError("cached file must not be downloaded"))
    monkeypatch.setattr(downloader.urllib.request, "urlopen", network)

    downloader.download("https://example.invalid/archive", target)

    network.assert_not_called()
    assert target.read_bytes() == contents


@pytest.mark.parametrize("contents", [b"archive", b""], ids=["nonempty", "empty"])
@pytest.mark.parametrize("response_kind", ["http", "file"])
def test_download_publishes_the_file(
    tmp_path: Path, monkeypatch, contents: bytes, response_kind: str
) -> None:
    target = tmp_path / "nested/archive.tgz"
    response = (
        http_response(contents)
        if response_kind == "http"
        else addinfourl(io.BytesIO(contents), Message(), "file:///archive")
    )
    network = Mock(return_value=response)
    monkeypatch.setattr(downloader.urllib.request, "urlopen", network)

    downloader.download("https://example.invalid/archive", target)

    assert target.read_bytes() == contents
    assert list(target.parent.iterdir()) == [target]
    assert network.call_args.args[0].full_url == "https://example.invalid/archive"
    assert response.closed


@pytest.mark.parametrize("failure", ["connection", "read"])
def test_network_failure_removes_partial_downloads(
    tmp_path: Path, monkeypatch, failure: str
) -> None:
    target = tmp_path / "archive.tgz"
    if failure == "connection":
        network = Mock(side_effect=OSError("connection failed"))
    else:
        response = http_response(b"archive")
        monkeypatch.setattr(
            response, "read", Mock(side_effect=[b"partial", OSError("read failed")])
        )
        network = Mock(return_value=response)
    monkeypatch.setattr(downloader.urllib.request, "urlopen", network)

    with pytest.raises(OSError, match=f"{failure} failed"):
        downloader.download("https://example.invalid/archive", target)

    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("print_progress", [False, True])
def test_progress_uses_kib_and_is_opt_in(
    tmp_path: Path, monkeypatch, capsys, print_progress: bool
) -> None:
    contents = b"x" * (5 * 1024)
    monkeypatch.setattr(downloader, "TRANSFER_BLOCK_BYTES", 1024)
    monkeypatch.setattr(
        downloader.urllib.request, "urlopen", Mock(return_value=http_response(contents))
    )

    downloader.download(
        "https://example.invalid/archive",
        tmp_path / "archive.tgz",
        print_progress=print_progress,
        progress_interval_in_kib=2,
    )

    progress = [
        line for line in capsys.readouterr().out.splitlines() if " bytes from " in line
    ]
    assert progress == (
        [
            "downloaded 2048 bytes from https://example.invalid/archive",
            "downloaded 4096 bytes from https://example.invalid/archive",
        ]
        if print_progress
        else []
    )


def test_default_progress_interval_is_128_000_kib() -> None:
    parameters = inspect.signature(downloader.download).parameters
    assert parameters["print_progress"].default is False
    assert parameters["progress_interval_in_kib"].default == 128_000
    assert (
        parameters["progress_interval_in_kib"].default * downloader.BYTES_PER_KIB
        == 125 * 1024 * 1024
    )


@pytest.mark.parametrize("interval", [0, -1])
def test_nonpositive_progress_interval_is_rejected(
    tmp_path: Path, monkeypatch, interval: int
) -> None:
    network = Mock()
    monkeypatch.setattr(downloader.urllib.request, "urlopen", network)
    with pytest.raises(
        ValueError, match="progress_interval_in_kib must be greater than zero"
    ):
        downloader.download(
            "https://example.invalid/archive",
            tmp_path / "archive.tgz",
            progress_interval_in_kib=interval,
        )
    network.assert_not_called()


def test_unsupported_response_is_closed_and_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    response = io.StringIO("not binary")
    monkeypatch.setattr(
        downloader.urllib.request, "urlopen", Mock(return_value=response)
    )

    with pytest.raises(TypeError, match="HTTP or binary URL response"):
        downloader.download(
            "https://example.invalid/archive",
            tmp_path / "archive",
        )

    assert response.closed
    assert not list(tmp_path.iterdir())
