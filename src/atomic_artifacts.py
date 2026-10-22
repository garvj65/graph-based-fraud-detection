"""Atomic, fsynced experiment artifacts prevent partial files becoming evidence."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np


def _finish(temporary: Path, destination: Path) -> None:
    """Publish a complete file and persist its parent directory entry on Unix."""
    temporary.replace(destination)
    if os.name == "posix":
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def write_json(path: Path, payload: Any) -> None:
    """Write strict JSON atomically, preserving the earlier JSON representation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        _finish(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_npz(path: Path, **arrays: Any) -> None:
    """Publish a complete compressed NumPy archive without appending an extension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        _finish(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
