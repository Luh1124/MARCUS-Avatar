"""Shared runtime paths for MARCUS-Avatar.

Environment variables can override these defaults when models live outside the
repository checkout.
"""

import os


HF_REPO_ID = os.environ.get("HF_REPO_ID", "luh0502/MARCUS-Avatar")
CKP_DIR = os.environ.get("CKP_DIR", "ckpts")
TOPO_DIR = os.environ.get("TOPO_DIR", "assets/topo")
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH", "meituan-longcat/LongCat-Image-Edit")
JOY_CAPTION_MODEL = os.environ.get(
    "JOY_CAPTION_MODEL",
    "fancyfeast/llama-joycaption-beta-one-hf-llava",
)


def ckpt_path(*parts: str) -> str:
    return os.path.join(CKP_DIR, *parts)


def topo_path(*parts: str) -> str:
    return os.path.join(TOPO_DIR, *parts)


def is_local_model_path(value: str) -> bool:
    """Return True when a model id should be validated as a local filesystem path."""
    expanded = os.path.expanduser(value)
    if os.path.isabs(expanded) or value.startswith((".", "~")):
        return True
    first_part = value.replace("\\", "/").split("/", 1)[0]
    if first_part in {CKP_DIR, "ckpts", "weights", "models"}:
        return True
    if os.path.exists(expanded):
        return True
    return "/" not in value
