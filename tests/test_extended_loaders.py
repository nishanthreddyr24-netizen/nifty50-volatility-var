from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import extended_loaders


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(extended_loaders, "EXTENDED_RAW_DIR", tmp_path)
    return tmp_path


def sample_download() -> pd.DataFrame:
    index = pd.DatetimeIndex(["1990-01-02", "1990-01-03"], name="Date")
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["^NSEI"]]
    )
    values = np.array(
        [
            [100.0, 102.0, 99.0, 101.0, 1000.0],
            [101.0, 103.0, 100.0, 102.0, 1100.0],
        ]
    )
    return pd.DataFrame(values, index=index, columns=columns)


def test_download_normalises_columns_and_uses_unadjusted_data(
    isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    def fake_download(ticker: str, **kwargs: object) -> pd.DataFrame:
        calls.append({"ticker": ticker, **kwargs})
        return sample_download()

    monkeypatch.setattr(extended_loaders.yf, "download", fake_download)

    data = extended_loaders.download_extended_dataset("nifty50", end="2026-11-01")

    assert calls == [
        {
            "ticker": "^NSEI",
            "start": "1990-01-01",
            "end": "2026-11-01",
            "progress": False,
            "auto_adjust": False,
        }
    ]
    assert list(data.columns) == list(extended_loaders.OHLCV_COLUMNS)
    assert data.index.name == "Date"
    assert extended_loaders.cache_path("nifty50").exists()


def test_india_vix_uses_no_fabricated_start_date(
    isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    def fake_download(ticker: str, **kwargs: object) -> pd.DataFrame:
        calls.append({"ticker": ticker, **kwargs})
        return sample_download()

    monkeypatch.setattr(extended_loaders.yf, "download", fake_download)
    extended_loaders.download_extended_dataset("india_vix")

    assert calls[0]["ticker"] == "^INDIAVIX"
    assert calls[0]["start"] == "1990-01-01"
    assert calls[0]["end"] is None


def test_batch_download_passes_one_end_argument_to_nifty_and_sp500(
    isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, object, object]] = []

    def fake_download(ticker: str, **kwargs: object) -> pd.DataFrame:
        calls.append((ticker, kwargs["start"], kwargs["end"]))
        return sample_download()

    monkeypatch.setattr(extended_loaders.yf, "download", fake_download)
    extended_loaders.download_all_extended(end="2026-11-01")

    assert calls == [
        ("^NSEI", "1990-01-01", "2026-11-01"),
        ("^GSPC", "1990-01-01", "2026-11-01"),
        ("^INDIAVIX", "1990-01-01", "2026-11-01"),
    ]


def test_batch_download_waits_for_all_downloads_before_writing(
    isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_download(ticker: str, **kwargs: object) -> pd.DataFrame:
        calls.append(ticker)
        if ticker == "^GSPC":
            raise RuntimeError("provider failure")
        return sample_download()

    monkeypatch.setattr(extended_loaders.yf, "download", fake_download)

    with pytest.raises(RuntimeError, match="provider failure"):
        extended_loaders.download_all_extended()

    assert calls == ["^NSEI", "^GSPC"]
    assert not list(isolated_cache.iterdir())


def test_manifest_contains_validation_metadata(
    isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(extended_loaders.yf, "download", lambda *args, **kwargs: sample_download())

    datasets = extended_loaders.download_all_extended(end="2026-11-01")
    manifest = extended_loaders.build_manifest(datasets, end="2026-11-01")
    india = manifest["datasets"]["india_vix"]

    assert manifest["requested_start"] == "1990-01-01"
    assert manifest["requested_end"] == "2026-11-01"
    assert india["row_count"] == 2
    assert india["columns"] == list(extended_loaders.OHLCV_COLUMNS)
    assert india["missing_values"] == {column: 0 for column in extended_loaders.OHLCV_COLUMNS}
    assert india["duplicate_date_count"] == 0
    assert india["dates_sorted"] is True
    assert india["downloaded_successfully"] is True


def test_existing_destination_is_not_overwritten(
    isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = extended_loaders.cache_path("nifty50")
    destination.write_text("original\n", encoding="utf-8")

    def unexpected_download(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("download must not run for an existing destination")

    monkeypatch.setattr(extended_loaders.yf, "download", unexpected_download)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        extended_loaders.download_extended_dataset("nifty50")

    assert destination.read_text(encoding="utf-8") == "original\n"


def test_load_extended_dataset_reads_existing_csv(
    isolated_cache: Path,
) -> None:
    data = sample_download().droplevel(1, axis=1)
    destination = extended_loaders.cache_path("nifty50")
    destination.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(destination)

    loaded = extended_loaders.load_extended_dataset("nifty50")

    pd.testing.assert_frame_equal(loaded, data)
