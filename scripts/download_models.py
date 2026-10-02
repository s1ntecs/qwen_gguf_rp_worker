from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    repo: str,
    filename: str,
    revision: str,
    target: str,
    expected_sha256: str | None = None,
) -> None:
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(
            hf_hub_download(
                repo_id=repo,
                filename=filename,
                revision=revision or None,
                local_dir=tmp,
                token=os.getenv("HF_TOKEN") or None,
            )
        )

        if expected_sha256:
            actual = sha256_file(source)
            if actual.lower() != expected_sha256.lower():
                raise RuntimeError(
                    f"SHA256 mismatch for {repo}:{filename}: "
                    f"expected {expected_sha256}, got {actual}"
                )

        shutil.move(str(source), target_path)

    print(
        f"Downloaded {repo}@{revision}:{filename} -> {target_path}"
        + (f" (sha256={expected_sha256})" if expected_sha256 else "")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--repo")
    mode.add_argument("--builtin-loras", action="store_true",
                      help="Download all three pinned adapters; --target is their directory")
    parser.add_argument("--file")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--target", required=True)
    parser.add_argument("--sha256")
    args = parser.parse_args()

    if args.builtin_loras:
        if args.file or args.sha256 or args.revision != "main":
            parser.error("Built-in LoRAs use the filenames, revision and checksums in lora_catalog.py")
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from lora_catalog import BUILTIN_LORAS, LORA_REPO, LORA_REVISION

        for asset in BUILTIN_LORAS:
            download(LORA_REPO, asset.filename, LORA_REVISION,
                     str(Path(args.target) / asset.filename), asset.sha256)
        return
    if not args.file:
        parser.error("--file is required with --repo")

    download(
        repo=args.repo,
        filename=args.file,
        revision=args.revision,
        target=args.target,
        expected_sha256=args.sha256,
    )


if __name__ == "__main__":
    main()
