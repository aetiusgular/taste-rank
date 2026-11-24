"""P4: tasterank.models refuses any model folder that is not exactly the pinned files (SECURITY.md T-4, T-6).

No real model is downloaded here; the folders are made-up bytes with their own hashes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import pytest

from tasterank import models
from tasterank.seams import REPO_ROOT


def _entry(tmp_path, files: dict[str, bytes]):
    d = tmp_path / "m"
    d.mkdir()
    for name, body in files.items():
        (d / name).write_bytes(body)
    return d, {"id": "test/model", "files": {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}}


FILES = {"open_clip_config.json": b"{}", "open_clip_model.safetensors": b"weights"}


def test_exact_folder_passes(tmp_path):
    d, entry = _entry(tmp_path, FILES)
    assert set(models.verify_model_dir(d, entry)) == set(FILES)


@pytest.mark.parametrize("extra", ["marqo_fashionSigLIP.py", "open_clip_pytorch_model.bin", "model.pth", "notes.txt"])
def test_any_extra_file_is_refused(tmp_path, extra):
    d, entry = _entry(tmp_path, FILES)
    (d / extra).write_bytes(b"x")
    with pytest.raises(ValueError, match="not pinned"):
        models.verify_model_dir(d, entry)


def test_changed_missing_linked_and_nested_are_refused(tmp_path):
    d, entry = _entry(tmp_path, FILES)
    (d / "open_clip_model.safetensors").write_bytes(b"other")
    with pytest.raises(ValueError, match="sha256"):
        models.verify_model_dir(d, entry)
    (d / "open_clip_model.safetensors").unlink()
    with pytest.raises(ValueError, match="missing"):
        models.verify_model_dir(d, entry)
    (tmp_path / "w").write_bytes(b"weights")
    (d / "open_clip_model.safetensors").symlink_to(tmp_path / "w")
    with pytest.raises(ValueError, match="symlink"):
        models.verify_model_dir(d, entry)
    (d / "open_clip_model.safetensors").unlink()
    (d / "open_clip_model.safetensors").write_bytes(b"weights")
    (d / ".cache").mkdir()
    with pytest.raises(ValueError, match="subfolder"):
        models.verify_model_dir(d, entry)


def test_lock_pins_only_data_files_for_every_model():
    lock = models.load_lock()
    for m in lock["models"]:
        assert m["files"], m["id"]
        for name, sha in m["files"].items():
            assert name.rsplit(".", 1)[-1] in ("json", "safetensors"), (m["id"], name)
            assert len(sha) == 64 and int(sha, 16) >= 0
        assert "trust_remote_code" not in m["loader"].replace("never trust_remote_code", ""), m["id"]
    with pytest.raises(ValueError, match="not a loadable"):
        models.verify_model_dir(REPO_ROOT, {"id": "x", "files": {"code.py": "0" * 64}})
    with pytest.raises(KeyError):
        models.model_entry("nobody/none", lock)


def test_cli_exit_codes(tmp_path):
    lock = json.loads((REPO_ROOT / "models.lock").read_text())
    mid = lock["models"][0]["id"]
    p = subprocess.run([sys.executable, "-m", "tasterank.models", mid, str(tmp_path)], capture_output=True, text=True)
    assert p.returncode == 1 and "refused" in p.stderr
    p = subprocess.run([sys.executable, "-m", "tasterank.models"], capture_output=True, text=True)
    assert p.returncode == 2
