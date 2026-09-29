from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download


def download(repo: str, filename: str, revision: str, target: str) -> None:
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
        shutil.move(str(source), target_path)
    print(f"Downloaded {repo}@{revision}:{filename} -> {target_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    download(args.repo, args.file, args.revision, args.target)


if __name__ == "__main__":
    main()
