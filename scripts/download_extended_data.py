"""Explicitly download the non-destructive historical data extension."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extended_loaders import build_manifest, download_all_extended


def main() -> None:
    datasets = download_all_extended()
    for name, data in datasets.items():
        metadata = build_manifest(datasets, end=None)["datasets"][name]
        print(
            f"{name}: {len(data)} rows, "
            f"{data.index.min().date()} to {data.index.max().date()}"
        )
        print(f"  columns: {list(data.columns)}")
        print(f"  missing: {data.isna().sum().to_dict()}")
        print(f"  duplicate dates: {metadata['duplicate_date_count']}")
        print(f"  dates sorted: {metadata['dates_sorted']}")
        print(f"  downloaded successfully: {metadata['downloaded_successfully']}")


if __name__ == "__main__":
    main()
