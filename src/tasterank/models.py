"""Model folders the box may load (P4 security review, SECURITY.md T-4 and T-6).

The host downloads each pinned model's `files` (models.lock) into a folder at
the pinned revision. The sandbox gets that folder read-only and no Hub access.
Before any loader touches it, `verify_model_dir` checks that the folder holds
exactly those files, each with its pinned sha256, and nothing else: no Python
file (remote code), no pickle checkpoint (.bin, .pth, .pt, .pkl), no extra
weights a loader might pick up instead.

    python -m tasterank.models <model id> <folder>
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from tasterank.seams import REPO_ROOT

MODELS_LOCK = REPO_ROOT / "models.lock"
ALLOWED_SUFFIXES = frozenset({".json", ".safetensors"})
_CHUNK = 1 << 20


def load_lock(path: Path | str = MODELS_LOCK) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def model_entry(model_id: str, lock: dict[str, Any] | None = None) -> dict[str, Any]:
    """The models.lock entry for `model_id`; KeyError if it is not pinned."""
    lock = load_lock() if lock is None else lock
    for m in lock["models"]:
        if m["id"] == model_id:
            return m
    raise KeyError(f"{model_id} is not pinned in models.lock")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_model_dir(folder: Path | str, entry: dict[str, Any]) -> dict[str, Path]:
    """name -> path for every pinned file; ValueError on any extra, missing, linked or changed file."""
    want: dict[str, str] = dict(entry.get("files") or {})
    if not want:
        raise ValueError(f"{entry.get('id')}: models.lock lists no files to verify")
    for name in want:
        if Path(name).suffix not in ALLOWED_SUFFIXES or "/" in name:
            raise ValueError(f"{entry.get('id')}: models.lock pins {name}, which is not a loadable data file")
    root = Path(folder)
    if not root.is_dir():
        raise ValueError(f"model folder {root} does not exist")
    found: dict[str, Path] = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if p.is_symlink():
            raise ValueError(f"{root}: {rel} is a symlink; copy the pinned files in")
        if p.is_dir():
            raise ValueError(f"{root}: {rel} is a subfolder; the model folder holds the pinned files only")
        if rel not in want:
            raise ValueError(f"{root}: {rel} is not pinned in models.lock (only {sorted(want)} may be present)")
        found[rel] = p
    missing = sorted(set(want) - set(found))
    if missing:
        raise ValueError(f"{root}: missing pinned file(s) {missing}")
    for name, p in found.items():
        got = sha256_file(p)
        if got != want[name]:
            raise ValueError(f"{root}: {name} has sha256 {got}, models.lock pins {want[name]}")
    return found


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("usage: python -m tasterank.models <model id> <folder>", file=sys.stderr)
        return 2
    try:
        verify_model_dir(args[1], model_entry(args[0]))
    except (KeyError, ValueError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 1
    print(f"ok: {args[1]} holds exactly the pinned files of {args[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
