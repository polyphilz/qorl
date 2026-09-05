"""Download and extract the pinned IMDb CSV archive."""

from __future__ import annotations

import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from scripts.imdb.schemas import ImdbManifest
from scripts.shared.download import download
from scripts.shared.verify import verify_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
IMDB_MANIFEST = REPOSITORY_ROOT / "scripts/imdb/manifest.json"
IMDB_RAW_DATA_PATH = REPOSITORY_ROOT / "data/raw"
TRANSFER_BLOCK_BYTES = 1024 * 1024
PROGRESS_INTERVAL_IN_KIB = 128_000


def extract_dataset(archive: Path, target: Path) -> None:
    if target.exists():
        print(f"using existing dataset: {target}")
        return

    temporary = Path(tempfile.mkdtemp(prefix=".imdb.extract.", dir=target.parent))
    try:
        with tarfile.open(archive, "r:gz") as source:
            for member in source.getmembers():
                if not member.isfile():
                    continue
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts:
                    raise RuntimeError(f"unsafe archive member: {member.name}")
                extracted = source.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"cannot read archive member: {member.name}")
                output_path = temporary / path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("xb") as output:
                    shutil.copyfileobj(extracted, output, length=TRANSFER_BLOCK_BYTES)
        temporary.replace(target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"extracted dataset: {target}")


def main() -> None:
    manifest = ImdbManifest.model_validate_json(
        IMDB_MANIFEST.read_text(encoding="utf-8")
    )
    IMDB_RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    dataset = manifest.dataset
    archive = IMDB_RAW_DATA_PATH / dataset.archive.filename
    download(
        dataset.source_url,
        archive,
        print_progress=True,
        progress_interval_in_kib=PROGRESS_INTERVAL_IN_KIB,
    )
    verify_file(
        archive,
        expected_size_in_bytes=dataset.archive.bytes,
        expected_checksum=dataset.archive.sha256,
    )
    extract_dataset(archive, IMDB_RAW_DATA_PATH / "tables")


if __name__ == "__main__":
    main()
