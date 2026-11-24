"""
File-integrity helpers used to protect trained checkpoints and recorded results.
"""
import hashlib
import json
import os
from typing import Dict, List, Tuple


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """Returns the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(manifest_path: str) -> Dict[str, str]:
    """Parses a `sha256sum`-format manifest into {relative_path: digest}."""
    entries: Dict[str, str] = {}
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            digest, path = line.split(None, 1)
            entries[path.lstrip("*").strip()] = digest
    return entries


def verify_manifest(manifest_path: str, root: str = ".") -> Tuple[List[str], List[str]]:
    """
    Verifies every file listed in a sha256 manifest.
    Returns (changed_paths, missing_paths); both empty means all files are intact.
    """
    changed, missing = [], []
    for rel_path, digest in read_manifest(manifest_path).items():
        full = os.path.join(root, rel_path)
        if not os.path.exists(full):
            missing.append(rel_path)
        elif sha256_file(full) != digest:
            changed.append(rel_path)
    return changed, missing


def write_json_if_absent(path: str, payload) -> None:
    """Writes JSON to `path`, refusing to replace an existing file."""
    if os.path.exists(path):
        raise FileExistsError(f"Refusing to overwrite existing file '{path}'.")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "x") as f:
        json.dump(payload, f, indent=2)
