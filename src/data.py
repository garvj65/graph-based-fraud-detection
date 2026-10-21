"""Load raw Elliptic files without adopting PyG's incompatible default masks.

Raw labels: 1=illicit, 2=licit, unknown=unlabeled.
Internal labels: 1=illicit, 0=licit, -1=unlabeled. IDs are joined explicitly.
Time is always stored separately; including it as a predictor requires a choice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

UNKNOWN = -1
ILLICIT = 1
LICIT = 0
LOCAL_NUMERIC_COUNT = 93
AGGREGATE_COUNT = 72
NUMERIC_FEATURE_COUNT = LOCAL_NUMERIC_COUNT + AGGREGATE_COUNT
TOTAL_TIME_STEPS = 49
EXPECTED_NODE_COUNT = 203_769
EXPECTED_EDGE_COUNT = 234_355
MIRROR_URL = "https://data.pyg.org/datasets/elliptic"
RAW_FILES = (
    "elliptic_txs_features.csv",
    "elliptic_txs_edgelist.csv",
    "elliptic_txs_classes.csv",
)
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class TemporalSplit:
    """The fixed, inclusive temporal partition specified for this project."""

    train: tuple[int, int] = (1, 24)
    validation: tuple[int, int] = (25, 34)
    test: tuple[int, int] = (35, 49)

    def __post_init__(self) -> None:
        if (self.train, self.validation, self.test) != ((1, 24), (25, 34), (35, 49)):
            raise ValueError("Split is fixed: train 1–24, validation 25–34, test 35–49.")


@dataclass(frozen=True)
class EllipticData:
    """Aligned raw data. `x` contains 165 numeric features, excluding time."""

    tx_ids: NDArray[np.str_]
    time_step: IntArray
    x: NDArray[np.float32]
    y: IntArray
    edge_index: IntArray

    @property
    def num_nodes(self) -> int:
        return len(self.tx_ids)

    def features(
        self, group: Literal["local", "all"], *, include_time: bool
    ) -> tuple[NDArray[np.float32], list[str]]:
        """Return a named feature matrix; make the time-feature choice explicit."""
        if group not in {"local", "all"}:
            raise ValueError(f"Unknown feature group: {group}")
        count = LOCAL_NUMERIC_COUNT if group == "local" else NUMERIC_FEATURE_COUNT
        names = [f"local_{i:03d}" for i in range(1, LOCAL_NUMERIC_COUNT + 1)]
        if group == "all":
            names += [f"aggregate_{i:03d}" for i in range(1, AGGREGATE_COUNT + 1)]
        matrix = self.x[:, :count].copy()
        if include_time:
            matrix = np.column_stack((self.time_step.astype(np.float32), matrix))
            names = ["time_step", *names]
        return matrix, names


@dataclass(frozen=True)
class TemporalMasks:
    """Separate graph membership (all nodes) from supervision (known labels)."""

    train_nodes: BoolArray
    validation_nodes: BoolArray
    test_nodes: BoolArray
    train_labeled: BoolArray
    validation_labeled: BoolArray
    test_labeled: BoolArray

    def node_mask(self, split: str) -> BoolArray:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")
        return getattr(self, f"{split}_nodes")

    def labeled_mask(self, split: str) -> BoolArray:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")
        return getattr(self, f"{split}_labeled")


@dataclass(frozen=True)
class GraphView:
    """An induced graph with local edge indices and no exposed labels.

    `node_indices` maps local indices back to the aligned original node table.
    No edge to an unavailable node is retained.
    """

    node_indices: IntArray
    edge_index: IntArray


def validate_data(data: EllipticData, *, require_full_dataset: bool = False) -> None:
    """Fail on ambiguous labels/IDs, invalid features, times or edge endpoints."""
    n = data.num_nodes
    if n == 0 or len(np.unique(data.tx_ids)) != n:
        raise ValueError("Transaction IDs must be nonempty and unique.")
    if any(str(tx).strip() == "" for tx in data.tx_ids):
        raise ValueError("Transaction IDs cannot be blank.")
    if data.time_step.shape != (n,) or data.y.shape != (n,):
        raise ValueError("Time/label arrays must align with node rows.")
    if not np.issubdtype(data.time_step.dtype, np.integer):
        raise ValueError("Time steps must be integers.")
    if np.any((data.time_step < 1) | (data.time_step > TOTAL_TIME_STEPS)):
        raise ValueError("Time steps must lie in 1–49.")
    if data.x.shape != (n, NUMERIC_FEATURE_COUNT) or not np.isfinite(data.x).all():
        raise ValueError("Expected 165 finite numeric features per node (time separate).")
    if not np.issubdtype(data.y.dtype, np.integer) or not np.isin(data.y, [-1, 0, 1]).all():
        raise ValueError("Labels must be integers: -1=unknown, 0=licit, 1=illicit.")
    edges = data.edge_index
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, number_of_edges].")
    if not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("Edge indices must be integers.")
    if edges.size and (edges.min() < 0 or edges.max() >= n):
        raise ValueError("Edge endpoint is not a node index.")
    if edges.size and np.any(data.time_step[edges[0]] != data.time_step[edges[1]]):
        raise ValueError("Cross-time edge found: dataset violates the snapshot contract.")
    if require_full_dataset:
        if n != EXPECTED_NODE_COUNT or edges.shape[1] != EXPECTED_EDGE_COUNT:
            raise ValueError("Raw node/edge counts do not match the published Elliptic dataset.")
        if not np.array_equal(np.unique(data.time_step), np.arange(1, TOTAL_TIME_STEPS + 1)):
            raise ValueError("Full dataset must contain all 49 time steps.")


def load_csv(raw_dir: Path, *, require_full_dataset: bool = False) -> EllipticData:
    """Read Kaggle/PyG raw CSVs and align labels and edges by transaction ID.

    Features have no header: ID, time, 93 local numeric, 72 aggregate numeric.
    CSV class-row order is never used as an alignment assumption.
    """
    missing = [name for name in RAW_FILES if not (raw_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing CSVs in {raw_dir}: {missing}. Use --download.")
    dtypes: dict[int, Any] = {0: str}
    dtypes.update({i: np.float32 for i in range(1, NUMERIC_FEATURE_COUNT + 2)})
    # Check timestamps before narrowing precision: 24.0000001 rounds to 24 in float32.
    dtypes[1] = np.float64
    features = pd.read_csv(raw_dir / RAW_FILES[0], header=None, dtype=dtypes)
    if features.shape[1] != NUMERIC_FEATURE_COUNT + 2:
        raise ValueError("Feature CSV must have 167 columns: ID + time + 165 numeric.")
    ids = features.iloc[:, 0]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("Feature CSV has missing or duplicate transaction IDs.")
    times = features.iloc[:, 1].to_numpy()
    if not np.isfinite(times).all() or np.any(times != np.floor(times)):
        raise ValueError("Raw time steps must be finite integers.")
    classes = pd.read_csv(raw_dir / RAW_FILES[2], dtype=str)
    if set(classes.columns) != {"txId", "class"}:
        raise ValueError("Class CSV must have txId,class columns.")
    if classes.isna().any().any() or classes["txId"].duplicated().any():
        raise ValueError("Class CSV has missing values or duplicate IDs.")
    if set(classes["txId"]) != set(ids):
        raise ValueError("Class CSV and feature CSV must contain exactly the same IDs.")
    mapping = {"1": ILLICIT, "2": LICIT, "unknown": UNKNOWN}
    if not classes["class"].isin(mapping).all():
        raise ValueError("Unrecognized raw class: expected 1, 2, or unknown.")
    aligned_y = classes.set_index("txId")["class"].map(mapping).reindex(ids)
    edges = pd.read_csv(raw_dir / RAW_FILES[1], dtype=str)
    if list(edges.columns) != ["txId1", "txId2"]:
        raise ValueError("Edge CSV must have ordered txId1,txId2 columns.")
    index = pd.Index(ids)
    source = index.get_indexer(edges["txId1"])
    target = index.get_indexer(edges["txId2"])
    if np.any(source < 0) or np.any(target < 0):
        raise ValueError("An edge references an ID missing from the feature CSV.")
    data = EllipticData(
        tx_ids=ids.to_numpy(dtype=str),
        time_step=times.astype(np.int64),
        x=features.iloc[:, 2:].to_numpy(dtype=np.float32, copy=True),
        y=aligned_y.to_numpy(dtype=np.int64),
        edge_index=np.stack((source, target)).astype(np.int64),
    )
    validate_data(data, require_full_dataset=require_full_dataset)
    return data


def download_csvs(raw_dir: Path) -> None:
    """Download the public PyG mirror's raw CSVs using atomic file replacement.

    Only the expected archive member is copied; arbitrary ZIP paths are never
    extracted. Existing raw files are not replaced. SHA-256 is logged later.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES:
        destination = raw_dir / name
        if destination.exists():
            continue
        archive = raw_dir / f"{name}.zip.part"
        temporary = raw_dir / f"{name}.part"
        try:
            with urllib.request.urlopen(f"{MIRROR_URL}/{name}.zip", timeout=60) as source:
                with archive.open("wb") as target:
                    shutil.copyfileobj(source, target)
            with zipfile.ZipFile(archive) as zipped:
                members = [m for m in zipped.infolist() if Path(m.filename).name == name]
                if len(members) != 1:
                    raise ValueError(f"Expected exactly one {name} in archive.")
                with zipped.open(members[0]) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target)
            temporary.replace(destination)
        finally:
            archive.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)


def load_via_pyg(root: Path, *, require_full_dataset: bool = False) -> EllipticData:
    """Acquire via EllipticBitcoinDataset, then recover timestamps from its raw CSV.

    Ignore PyG masks (train=1–34) and reconstruct ID-aligned labels ourselves.
    PyG is optional for stage 1; absence is a clear error, never a silent fallback.
    """
    try:
        from torch_geometric.datasets import EllipticBitcoinDataset
    except ImportError as error:
        raise RuntimeError(
            "Install requirements-gnn.txt or use --source csv --download."
        ) from error
    dataset = EllipticBitcoinDataset(root=str(root))
    return load_csv(Path(dataset.raw_dir), require_full_dataset=require_full_dataset)


def make_temporal_masks(
    data: EllipticData, split: TemporalSplit = TemporalSplit()
) -> TemporalMasks:
    """Build fixed chronological graph and supervised masks, retaining unknowns."""
    validate_data(data)
    node_masks = [
        (data.time_step >= start) & (data.time_step <= stop)
        for start, stop in (split.train, split.validation, split.test)
    ]
    labeled = [mask & (data.y != UNKNOWN) for mask in node_masks]
    masks = TemporalMasks(*node_masks, *labeled)
    assert_no_split_leakage(data, masks, split)
    return masks


def assert_no_split_leakage(
    data: EllipticData, masks: TemporalMasks, split: TemporalSplit = TemporalSplit()
) -> None:
    """Reject overlap, moved boundaries, missing nodes and unknown supervision."""
    coverage = np.zeros(data.num_nodes, dtype=np.int64)
    previous_max = 0
    for name, (start, stop) in zip(
        ("train", "validation", "test"), (split.train, split.validation, split.test)
    ):
        nodes, labeled = masks.node_mask(name), masks.labeled_mask(name)
        for mask in (nodes, labeled):
            if mask.shape != (data.num_nodes,) or mask.dtype != np.bool_:
                raise ValueError("Split masks must be boolean vectors aligned with nodes.")
        expected = (data.time_step >= start) & (data.time_step <= stop)
        if not np.array_equal(nodes, expected) or not nodes.any():
            raise ValueError(f"{name} node mask violates the fixed temporal boundary.")
        if not np.array_equal(labeled, nodes & (data.y != UNKNOWN)):
            raise ValueError(f"{name} supervision must include only its known labels.")
        if not labeled.any():
            raise ValueError(f"{name} has no labeled examples.")
        if data.time_step[nodes].min() <= previous_max:
            raise ValueError("Split time ranges overlap or are out of order.")
        previous_max = int(data.time_step[nodes].max())
        coverage += nodes
    if not np.all(coverage == 1):
        raise ValueError("Every node must belong to exactly one temporal split.")


def training_labels(data: EllipticData, masks: TemporalMasks) -> IntArray:
    """Expose only train labels to future neighbor-label features.

    Even these labels are not safe to aggregate naively for training examples:
    stage 3 must address leave-one-out/cross-fitting and label availability.
    """
    assert_no_split_leakage(data, masks)
    labels = np.full(data.num_nodes, UNKNOWN, dtype=np.int64)
    labels[masks.train_labeled] = data.y[masks.train_labeled]
    return labels


def _induced_view(data: EllipticData, mask: BoolArray) -> GraphView:
    nodes = np.flatnonzero(mask).astype(np.int64)
    keep = mask[data.edge_index[0]] & mask[data.edge_index[1]]
    remap = np.full(data.num_nodes, -1, dtype=np.int64)
    remap[nodes] = np.arange(len(nodes))
    return GraphView(nodes, remap[data.edge_index[:, keep]])


def training_graph(data: EllipticData, masks: TemporalMasks) -> GraphView:
    """An induced graph of steps 1–24, including unknown training-period nodes."""
    assert_no_split_leakage(data, masks)
    return _induced_view(data, masks.train_nodes)


def snapshot_graph(data: EllipticData, time_step: int) -> GraphView:
    """Build one completed snapshot; this API makes no transaction-time claim."""
    if isinstance(time_step, bool) or not isinstance(time_step, (int, np.integer)):
        raise ValueError("Snapshot time must be an integer.")
    if not 1 <= time_step <= TOTAL_TIME_STEPS:
        raise ValueError("Snapshot time must lie in 1–49.")
    return _induced_view(data, data.time_step == time_step)


def sha256_file(path: Path) -> str:
    """Hash raw input bytes for reproducible dataset provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """Write strict JSON; NaN and Infinity cannot quietly enter evidence logs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def environment_versions() -> dict[str, str | None]:
    """Capture the running interpreter and stage-relevant package versions."""
    packages: dict[str, str | None] = {"python": platform.python_version()}
    for package in (
        "numpy",
        "pandas",
        "matplotlib",
        "PyYAML",
        "pytest",
        "scikit-learn",
        "xgboost",
        "xgboost-cpu",
        "joblib",
        "networkx",
        "scipy",
        "torch",
        "torch-geometric",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return packages


def main() -> None:
    """Run stage 1 from one command; log config, input hashes and audit results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--source", choices=("csv", "pyg"))
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("stage") != 1:
        raise ValueError("This entry point implements stage 1 only.")
    split = TemporalSplit(**{name: tuple(bounds) for name, bounds in config["split"].items()})
    if args.source:
        config["data"]["source"] = args.source
    if args.raw_dir:
        config["data"]["raw_dir"] = str(args.raw_dir)
    if args.output_dir:
        config["output_dir"] = str(args.output_dir)
    source = config["data"]["source"]
    raw_dir = Path(config["data"]["raw_dir"])
    root = Path(config["data"]["root"])
    require_full = config["data"]["require_full_dataset"]
    if not isinstance(require_full, bool):
        raise ValueError("require_full_dataset must be a YAML boolean.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output = Path(config["output_dir"])
    log_path = output / "runs" / f"{run_id}.json"
    record: dict[str, Any] = {
        "run_id": run_id,
        "stage": 1,
        "status": "running",
        "seed": None,
        "seed_note": "No stochastic training in stage 1; configured seeds reserved for stage 2+.",
        "config": config,
        "argv": sys.argv,
        "versions": environment_versions(),
        "source_files_sha256": {
            str(path.relative_to(Path(__file__).parent.parent)): sha256_file(path)
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "model_metrics": None,
    }
    write_json(log_path, record)
    try:
        if source == "pyg":
            if args.raw_dir is not None:
                raise ValueError("--raw-dir is for CSV input; PyG uses data.root/raw.")
            raw_dir = root / "raw"
            data = load_via_pyg(root, require_full_dataset=require_full)
        elif source == "csv":
            if args.download:
                download_csvs(raw_dir)
            data = load_csv(raw_dir, require_full_dataset=require_full)
        else:
            raise ValueError(f"Unsupported data source: {source}")
        masks = make_temporal_masks(data, split)
        from src.eda import build_stage1_report

        audit = build_stage1_report(data, masks, output)
        record.update(
            status="completed",
            raw_dir=str(raw_dir.resolve()),
            input_files={
                name: {
                    "sha256": sha256_file(raw_dir / name),
                    "bytes": (raw_dir / name).stat().st_size,
                }
                for name in RAW_FILES
            },
            audit=audit,
        )
        write_json(log_path, record)
        write_json(output / "latest_run.json", record)
        print(json.dumps({"status": "completed", "run_id": run_id, "audit": audit}, indent=2))
    except Exception as error:
        record.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        write_json(log_path, record)
        raise


if __name__ == "__main__":
    main()
