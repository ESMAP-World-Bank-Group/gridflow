import pandas as pd
from pathlib import Path
from types import SimpleNamespace

import pytest

from gridflow.epm_input_generator import generate_epm_inputs

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_EPM_INPUTS = REPO_ROOT / "data" / "epm_inputs_raw"


def _make_region():
    zones = pd.DataFrame(
        {
            "country": ["AGO", "AGO", "CMR"],
            "population": [100, 200, 150],
        },
        index=["AGO_Z0", "AGO_Z1", "CMR_Z0"],
    )
    return SimpleNamespace(zones=zones)


def _write_csv(root: Path, relative: str, df: pd.DataFrame):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _copy_subset(relative: str, root: Path, limit=None, mutate=None):
    df = pd.read_csv(BASE_EPM_INPUTS / relative)
    if limit is not None:
        df = df.head(limit)
    if mutate is not None:
        df = mutate(df)
    return _write_csv(root, relative, df)


def _ensure_country_column(df: pd.DataFrame) -> pd.DataFrame:
    if "country" not in df.columns:
        df = df.copy()
        df["country"] = df["zone"]
    return df


def _seed_base_inputs(root: Path):
    data = {}
    data["pDemandProfile"] = _copy_subset(
        "load/pDemandProfile.csv", root, limit=4
    )
    data["pDemandForecast"] = _copy_subset(
        "load/pDemandForecast.csv",
        root,
        limit=4,
        mutate=_ensure_country_column,
    )
    data["pDemandData"] = _copy_subset(
        "load/pDemandData.csv", root, limit=4
    )
    # Supply-side defaults (only a few rows copied to minimize fixture size)
    _copy_subset("supply/pAvailabilityDefault.csv", root, limit=4)
    _copy_subset("supply/pGenDataInputDefault.csv", root, limit=4)
    _copy_subset("supply/pCapexTrajectoriesDefault.csv", root, limit=4)
    _copy_subset("supply/pFuelPrice.csv", root, limit=4)
    _copy_subset("supply/pVREProfile.csv", root, limit=4)
    _copy_subset("constraint/pMaxFuellimit.csv", root, limit=4)
    # Placeholder template expected by the generator
    _write_csv(
        root,
        "supply/pGenDataExcelDefault.csv",
        data["pDemandProfile"][["zone"]]
        .drop_duplicates()
        .assign(tech="GEN", value=1),
    )
    data["config"] = _copy_subset("config.csv", root)
    return data


def test_generate_epm_inputs_processes_zonal_files(tmp_path):
    """Ensure real raw inputs get replicated/distributed correctly and non-zonal files copy through."""
    region = _make_region()
    input_root = tmp_path / "inputs"
    output_root = tmp_path / "outputs"
    originals = _seed_base_inputs(input_root)

    generate_epm_inputs(region, input_root, output_root)

    profile_out = pd.read_csv(output_root / "load/pDemandProfile.csv")
    assert len(profile_out) == len(originals["pDemandProfile"]) * len(region.zones)
    assert set(profile_out["zone"]) == set(region.zones.index)

    forecast_out = pd.read_csv(output_root / "load/pDemandForecast.csv")
    scale = region.zones["population"] / region.zones["population"].sum()
    angola_2025 = (
        originals["pDemandForecast"]
        .loc[originals["pDemandForecast"]["country"] == "Angola", "2025"]
        .iloc[0]
    )
    zone_val = (
        forecast_out.loc[
            (forecast_out["zone"] == "AGO_Z0") & (forecast_out["type"] == "Energy"),
            "2025",
        ]
        .iloc[0]
    )
    assert zone_val == pytest.approx(angola_2025 * scale.loc["AGO_Z0"])

    assert (
        output_root / "config.csv"
    ).read_text() == (input_root / "config.csv").read_text()


def test_generate_epm_inputs_flags_unhandled_zonal_files(tmp_path):
    """Guard against accidentally skipping zonal CSVs when scanning raw inputs."""
    region = _make_region()
    input_root = tmp_path / "inputs"
    output_root = tmp_path / "outputs"
    _seed_base_inputs(input_root)
    _write_csv(
        input_root,
        "load/pUnhandled.csv",
        pd.DataFrame({"zone": ["Angola"], "value": [1]}),
    )

    with pytest.raises(ValueError, match="pUnhandled.csv"):
        generate_epm_inputs(region, input_root, output_root)
