import hashlib
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from lora_catalog import LoraAsset


class DownloadModelsTests(unittest.TestCase):
    def setUp(self):
        hub = types.ModuleType("huggingface_hub")
        hub.hf_hub_download = self.hub_download
        self.dependencies = patch.dict(sys.modules, {"huggingface_hub": hub})
        self.dependencies.start()
        self.addCleanup(self.dependencies.stop)
        path = Path(__file__).resolve().parents[1] / "scripts" / "download_models.py"
        spec = importlib.util.spec_from_file_location("download_models_test", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.payload = b"test weights"
        self.requests = []

    def hub_download(self, *, repo_id, filename, revision, local_dir, token):
        self.requests.append((repo_id, revision, filename))
        file = Path(local_dir) / filename
        file.write_bytes(self.payload)
        return str(file)

    def test_batch_downloads_all_assets_with_pinned_revision(self):
        digest = hashlib.sha256(self.payload).hexdigest()
        assets = tuple(LoraAsset(key, f"{key}.safetensors", digest, 0)
                       for key in ("nsfw", "penis", "vagina"))
        with tempfile.TemporaryDirectory() as tmp, \
                patch("lora_catalog.BUILTIN_LORAS", assets), \
                patch.object(sys, "argv", ["download_models.py", "--builtin-loras", "--target", tmp]):
            self.module.main()
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()),
                             ["nsfw.safetensors", "penis.safetensors", "vagina.safetensors"])
            for file in Path(tmp).iterdir():
                self.assertEqual(file.read_bytes(), self.payload)
        self.assertEqual(self.requests, [
            ("sintecs/Qwen2.1_loras", "9181d64c28d06ce05425c741a2e551c0d812ae41", f"{key}.safetensors")
            for key in ("nsfw", "penis", "vagina")
        ])

    def test_bad_checksum_does_not_install_corrupt_asset(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "weights.safetensors"
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                self.module.download("repo", "file", "revision", str(target), "0" * 64)
            self.assertFalse(target.exists())

    def test_existing_single_asset_cli_remains_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "weights.safetensors"
            with patch.object(sys, "argv", ["download_models.py", "--repo", "repo",
                    "--file", "file", "--target", str(target)]):
                self.module.main()
            self.assertEqual(target.read_bytes(), self.payload)
        self.assertEqual(self.requests, [("repo", "main", "file")])
