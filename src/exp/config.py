"""Immutable configuration for the historical extension experiments."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping

HISTORICAL_START: Final[str] = "1990-07-03"
BASE_DATE: Final[str] = "1995-11-03"
BASE_VALUE: Final[int] = 1000
OFFICIAL_LAUNCH_DATE: Final[str] = "1996-04-22"
FREE_FLOAT_START: Final[str] = "2009-06-26"
FINAL_DATE: Final[str] = "2026-10-30"
DOCUMENTED_BASELINE_START: Final[str] = "2008-01-01"
MIN_TRAINING_RETURNS: Final[int] = 500
VAR_CONFIDENCE: Final[float] = 0.99
FORECAST_HORIZON: Final[int] = 1
GBM_REFIT_EVERY: Final[int] = 1
HMM_REFIT_EVERY: Final[int] = 63
GARCH_REFIT_EVERY: Final[int] = 21
HMM_GARCH_HMM_REFIT_EVERY: Final[int] = 63
HMM_GARCH_GARCH_REFIT_EVERY: Final[int] = 21
MSGARCH_REFIT_EVERY: Final[int] = 63
HISTORICAL_SIMULATION_WINDOW: Final[int] = 500

EXPECTED_NSE_PATH: Final[str] = "data/processed/nifty50_nse_1990_2026.csv"
EXPECTED_YAHOO_PATH: Final[str] = "data/raw/extended/nifty50_1990_onward.csv"

REGIME_DESCRIPTIONS: Final[Mapping[int, str]] = MappingProxyType({
    0: "Before 1995-11-03: pre-base retrospective history",
    1: "1995-11-03 to 1996-04-19: post-base, pre-launch historical series",
    2: "1996-04-22 to 2009-06-25: launched index, pre-free-float methodology",
    3: "2009-06-26 onward: free-float methodology",
})


@dataclass(frozen=True)
class VariantConfig:
    variant_id: str
    nominal_start: str
    description: str


EXPERIMENT_1_VARIANTS: Final[tuple[VariantConfig, ...]] = (
    VariantConfig(
        "BASELINE_CONFIGURED_2008",
        DOCUMENTED_BASELINE_START,
        "Documented original loader start; not a verified reconstruction",
    ),
    VariantConfig(
        "EXTENDED_1990",
        HISTORICAL_START,
        "Complete published NSE historical series",
    ),
)

EXPERIMENT_2_VARIANTS: Final[tuple[VariantConfig, ...]] = (
    VariantConfig("PRELAUNCH_1990", HISTORICAL_START, "Complete published historical series"),
    VariantConfig("LAUNCHED_1996", OFFICIAL_LAUNCH_DATE, "Officially launched NIFTY 50"),
    VariantConfig("FREEFLOAT_2009", FREE_FLOAT_START, "Free-float methodology period"),
)

MODEL_CONFIGURATION: Final[Mapping[str, object]] = MappingProxyType({
    "GBM": {"refit_every": GBM_REFIT_EVERY},
    "HMM": {"refit_every": HMM_REFIT_EVERY},
    "GARCH": {"refit_every": GARCH_REFIT_EVERY},
    "HMM_GARCH": {
        "hmm_refit_every": HMM_GARCH_HMM_REFIT_EVERY,
        "garch_refit_every": HMM_GARCH_GARCH_REFIT_EVERY,
    },
    "MS_GARCH": {"refit_every": MSGARCH_REFIT_EVERY},
    "Historical Simulation": {"window": HISTORICAL_SIMULATION_WINDOW},
})
