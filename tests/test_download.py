"""Test the dataset acquisition boundary without network access or real metrics."""

import io
import zipfile
from pathlib import Path

import pytest

from src.data import RAW_FILES, download_csvs


def _archive(name: str, content: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, content)
    return buffer.getvalue()


def test_download_writes_expected_members_only(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def open_url(url: str, timeout: int) -> io.BytesIO:
        calls.append(url)
        name = url.rsplit("/", 1)[-1].removesuffix(".zip")
        assert timeout > 0
        return io.BytesIO(_archive("nested/" + name, b"expected-data"))

    monkeypatch.setattr("urllib.request.urlopen", open_url)
    download_csvs(tmp_path)
    assert len(calls) == 3
    for name in RAW_FILES:
        assert (tmp_path / name).read_bytes() == b"expected-data"
    assert not list(tmp_path.glob("*.part"))
    # Existing data is not silently overwritten or downloaded again.
    download_csvs(tmp_path)
    assert len(calls) == 3


def test_wrong_archive_member_fails_without_partial_raw_file(tmp_path: Path, monkeypatch) -> None:
    def open_url(url: str, timeout: int) -> io.BytesIO:
        return io.BytesIO(_archive("unexpected.csv", b"not-the-dataset"))

    monkeypatch.setattr("urllib.request.urlopen", open_url)
    with pytest.raises(ValueError, match="exactly one"):
        download_csvs(tmp_path)
    assert not list(tmp_path.iterdir())
