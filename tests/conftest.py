"""Tiny artificial CSV fixtures are for correctness tests, never result claims."""

from pathlib import Path

import pandas as pd
import pytest

from src.data import NUMERIC_FEATURE_COUNT, RAW_FILES, EllipticData, load_csv


@pytest.fixture
def raw_csvs(tmp_path: Path) -> Path:
    """49 time steps, each with licit, illicit and unknown nodes; shuffled labels."""
    rows, labels, edges = [], [], []
    for time in range(1, 50):
        ids = [str(1000 + 3 * time + i) for i in range(3)]
        for offset, (tx_id, label) in enumerate(zip(ids, ("2", "1", "unknown"))):
            values = [float(time + offset + i / 1000) for i in range(NUMERIC_FEATURE_COUNT)]
            rows.append([tx_id, time, *values])
            labels.append([tx_id, label])
        edges.extend([[ids[0], ids[1]], [ids[1], ids[2]]])
    pd.DataFrame(rows).to_csv(tmp_path / RAW_FILES[0], index=False, header=False)
    pd.DataFrame(edges, columns=["txId1", "txId2"]).to_csv(tmp_path / RAW_FILES[1], index=False)
    pd.DataFrame(labels[::-1], columns=["txId", "class"]).to_csv(
        tmp_path / RAW_FILES[2], index=False
    )
    return tmp_path


@pytest.fixture
def data(raw_csvs: Path) -> EllipticData:
    return load_csv(raw_csvs)
