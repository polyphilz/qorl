from __future__ import annotations

from pathlib import Path

import pytest

from qorl.model.model import model_snapshot


@pytest.mark.parametrize("cache_source", ["hub_cache", "hf_home", "default"])
def test_model_snapshot_finds_only_the_requested_cached_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache_source: str
) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    if cache_source == "hub_cache":
        cache = tmp_path / "hub-cache"
        monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache))
        monkeypatch.setenv("HF_HOME", str(tmp_path / "unused-home"))
    elif cache_source == "hf_home":
        hf_home = tmp_path / "hf-home"
        cache = hf_home / "hub"
        monkeypatch.setenv("HF_HOME", str(hf_home))
    else:
        cache = home / ".cache/huggingface/hub"
    snapshot = cache / "models--organization--model/snapshots/pinned-revision"
    snapshot.mkdir(parents=True)

    assert model_snapshot("organization/model", "pinned-revision") == snapshot.resolve()
    with pytest.raises(RuntimeError, match="pinned model snapshot is missing"):
        model_snapshot("organization/model", "missing-revision")
