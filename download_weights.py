#!/usr/bin/env python3
"""Download pretrained weights and topology assets from Hugging Face."""

import argparse
from pathlib import Path

from runtime_paths import HF_REPO_ID

ALLOW_PATTERNS = ["ckpts/**", "assets/topo/**"]


def download_weights(repo_id: str = HF_REPO_ID, local_dir: str = "."):
    """Download model checkpoints and topology assets to `local_dir`."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        )

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading weights from {repo_id} to {local_dir.resolve()} ...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=ALLOW_PATTERNS,
    )
    print("Done. Expected directories: ./ckpts and ./assets/topo")


def main():
    parser = argparse.ArgumentParser(
        description="Download MARCUS-Avatar pretrained weights from Hugging Face."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=HF_REPO_ID,
        help=f"Hugging Face model repo id (default: {HF_REPO_ID}).",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=".",
        help="Project directory where ckpts/ and assets/topo/ will be restored (default: .).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Deprecated alias for --local-dir, kept for compatibility.",
    )
    args = parser.parse_args()
    download_weights(args.repo_id, args.cache_dir or args.local_dir)


if __name__ == "__main__":
    main()
