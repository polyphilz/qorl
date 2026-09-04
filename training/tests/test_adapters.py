from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import save_file

from qorl.adapters.verify import verify_merged_model
from qorl_training.adapters.merge import merge


class AdapterMergeTest(unittest.TestCase):
    def test_merge_uses_recorded_base_and_replaces_inherited_manifest(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base"
            adapter = root / "adapter"
            output = root / "output"
            base.mkdir()
            adapter.mkdir()
            save_file({"layer.weight": torch.eye(2)}, base / "model.safetensors")
            (base / "tokenizer.json").write_text("{}")
            (base / "qorl-merge.json").write_text('{"stale": true}')
            save_file(
                {
                    "layer.lora_A.weight": torch.ones((1, 2)),
                    "layer.lora_B.weight": torch.ones((2, 1)),
                },
                adapter / "adapter_model.safetensors",
            )
            (adapter / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "base_model_name_or_path": str(base),
                        "peft_type": "LORA",
                        "bias": "none",
                        "r": 1,
                        "lora_alpha": 1.0,
                    }
                )
            )

            manifest_path = merge(base, adapter, output, root)

            manifest = json.loads(manifest_path.read_text())
            assert manifest.get("stale") is None
            assert all(
                artifact["path"] != "qorl-merge.json"
                for artifact in manifest["artifacts"]
            )
            verify_merged_model(base, adapter, output, root)


if __name__ == "__main__":
    unittest.main()
