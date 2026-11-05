"""Independent loaders for historical market-data extensions.

This module intentionally does not import or alter ``src.loaders``. Extended
files are written under ``data/raw/extended`` and are never overwritten.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Final

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTENDED_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "extended"
OHLCV_COLUMNS: Final[tuple[str, ...]] = ("Open", "High", "Low", "Close", "Volume")
NIFTY_SP500_START: Final[str] = "1990-01-01"
MANIFEST_FILENAME: Final[str] = "download_manifest.json"

EXTENDED_DATASETS: Final[dict[str, dict[str, str | None]]] = {
    "nifty50": {
        "ticker": "^NSEI",
        "filename": "nifty50_1990_onward.csv",
        "start": NIFTY_SP500_START,
    },
    "sp500": {
        "ticker": "^GSPC",
        "filename": "sp500_1990_onward.csv",
        "start": NIFTY_SP500_START,
    },
    "india_vix": {
        "ticker": "^INDIAVIX",
        "filename": "india_vix_available_history.csv",
        "start": NIFTY_SP500_START,
    },
}


def cache_path(name: str) -> Path:
    """Return the dedicated cache path for a configured extended dataset."""
    try:
        filename = EXTENDED_DATASETS[name]["filename"]
    except KeyError as exc:
        raise KeyError(f"Unknown extended dataset: {name!r}") from exc
    assert isinstance(filename, str)
    return EXTENDED_RAW_DIR / filename


def _dataset_config(name: str) -> dict[str, str | None]:
    try:
        return EXTENDED_DATASETS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown extended dataset: {name!r}") from exc


def _normalise_download(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("Yahoo Finance returned no observations")

    data = raw.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    missing_columns = [column for column in OHLCV_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Downloaded data is missing columns: {missing_columns}")

    data = data.loc[:, list(OHLCV_COLUMNS)].copy()
    data.index.name = "Date"
    return data


def _download_dataset(name: str, end: str | None) -> pd.DataFrame:
    config = _dataset_config(name)
    raw = yf.download(
        config["ticker"],
        start=config["start"],
        end=end,
        progress=False,
        auto_adjust=False,
    )
    return _normalise_download(raw)


def _write_csv_exclusive(name: str, data: pd.DataFrame) -> None:
    destination = cache_path(name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as handle:
        data.to_csv(handle)


def manifest_path() -> Path:
    return EXTENDED_RAW_DIR / MANIFEST_FILENAME


def build_manifest(
    datasets: dict[str, pd.DataFrame],
    end: str | None,
    downloaded_at: str | None = None,
) -> dict[str, object]:
    """Build validation metadata for a completed batch download."""
    manifest: dict[str, object] = {
        "source": "Yahoo Finance via yfinance",
        "requested_start": NIFTY_SP500_START,
        "requested_end": end,
        "download_timestamp_utc": downloaded_at or datetime.now(timezone.utc).isoformat(),
        "datasets": {},
    }
    dataset_metadata = manifest["datasets"]
    assert isinstance(dataset_metadata, dict)
    for name, data in datasets.items():
        destination = cache_path(name)
        try:
            recorded_path = str(destination.relative_to(PROJECT_ROOT))
        except ValueError:
            recorded_path = str(destination)
        dataset_metadata[name] = {
            "ticker": EXTENDED_DATASETS[name]["ticker"],
            "path": recorded_path,
            "downloaded_successfully": True,
            "first_date": data.index.min().date().isoformat(),
            "last_date": data.index.max().date().isoformat(),
            "row_count": len(data),
            "columns": list(data.columns),
            "missing_values": {column: int(count) for column, count in data.isna().sum().items()},
            "duplicate_date_count": int(data.index.duplicated().sum()),
            "dates_sorted": bool(data.index.is_monotonic_increasing),
        }
    return manifest


def write_manifest_exclusive(manifest: dict[str, object]) -> None:
    """Write the batch manifest once without replacing an existing file."""
    destination = manifest_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def download_extended_dataset(name: str, end: str | None = None) -> pd.DataFrame:
    """Download one extended dataset and create its cache exactly once.

    ``end`` is passed unchanged to yfinance. Existing destinations are checked
    before downloading and are opened in exclusive-create mode when written.
    """
    config = _dataset_config(name)
    destination = cache_path(name)
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing extended dataset: {destination}"
        )

    data = _download_dataset(name, end)
    _write_csv_exclusive(name, data)
    return data


def load_extended_dataset(name: str) -> pd.DataFrame:
    """Load an existing extended CSV without downloading or changing it."""
    destination = cache_path(name)
    if not destination.exists():
        raise FileNotFoundError(f"Extended dataset does not exist: {destination}")
    return pd.read_csv(destination, index_col=0, parse_dates=True)


def download_all_extended(end: str | None = None) -> dict[str, pd.DataFrame]:
    """Download all extended datasets using one shared end-date argument."""
    destinations = {name: cache_path(name) for name in EXTENDED_DATASETS}
    existing = [path for path in destinations.values() if path.exists()]
    existing.append(manifest_path()) if manifest_path().exists() else None
    if existing:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing extended file(s): {paths}")

    datasets = {name: _download_dataset(name, end=end) for name in EXTENDED_DATASETS}
    for name, data in datasets.items():
        _write_csv_exclusive(name, data)
    write_manifest_exclusive(build_manifest(datasets, end=end))
    return datasets
