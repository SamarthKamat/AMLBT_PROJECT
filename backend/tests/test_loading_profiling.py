"""CSV / Excel loading, type coercion, profiling and the Gemini-bound profile view."""
import io
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from config import DOMAINS, SAMPLE_DIR
from data_profiler import load_dataset, profile_dataframe, profile_for_llm
from errors import UDIError
from tests import datasets


def _write(tmp_path, name, text, mode="w", encoding="utf-8"):
    p = tmp_path / name
    if mode == "w":
        p.write_text(text, encoding=encoding)
    else:
        p.write_bytes(text)
    return p


def test_csv_loading_types(tmp_path):
    df = datasets.delivery_df()
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    out = load_dataset(p)
    assert out.shape == df.shape
    assert pd.api.types.is_datetime64_any_dtype(out["Order_Date"])
    assert pd.api.types.is_numeric_dtype(out["Distance"])
    assert out["City"].dtype == object


def test_csv_semicolon_and_currency_and_missing(tmp_path):
    text = "Name;Amount;Joined\nA;\"1,200.50\";2024-01-05\nB;$900;2024-02-07\nC;N/A;2024-03-09\nD;(50);2024-04-01\n"
    out = load_dataset(_write(tmp_path, "s.csv", text))
    assert list(out.columns) == ["Name", "Amount", "Joined"]
    assert pd.api.types.is_numeric_dtype(out["Amount"])
    assert out["Amount"].iloc[0] == 1200.5 and out["Amount"].iloc[3] == -50 and pd.isna(out["Amount"].iloc[2])
    assert pd.api.types.is_datetime64_any_dtype(out["Joined"])


def test_day_first_dates_detected(tmp_path):
    text = "d,v\n" + "\n".join(f"{d:02d}/03/2024,{d}" for d in range(13, 28))
    out = load_dataset(_write(tmp_path, "dm.csv", text))
    assert out["d"].iloc[0] == pd.Timestamp("2024-03-13")


def test_header_offset_title_rows(tmp_path):
    text = "Quarterly report,,\n,,\nRegion,Sales,Units\nNorth,100,5\nSouth,200,9\nEast,150,7\n"
    out = load_dataset(_write(tmp_path, "t.csv", text))
    assert list(out.columns) == ["Region", "Sales", "Units"] and len(out) == 3


def test_excel_loading_and_sheet_choice(tmp_path):
    df = datasets.hr_df(60)
    p = tmp_path / "hr.xlsx"
    with pd.ExcelWriter(p) as w:
        pd.DataFrame().to_excel(w, sheet_name="Empty", index=False)
        df.to_excel(w, sheet_name="Data", index=False)
    out = load_dataset(p)
    assert out.shape == df.shape
    assert out.attrs["sheet"] == "Data"
    assert pd.api.types.is_numeric_dtype(out["Salary"])


def test_excel_dates(tmp_path):
    df = datasets.retail_df(40)
    p = tmp_path / "r.xlsx"
    df.to_excel(p, index=False)
    out = load_dataset(p)
    assert pd.api.types.is_datetime64_any_dtype(out["Day"])


@pytest.mark.parametrize("name,payload", [("empty.csv", ""), ("one.csv", "a,b\n"), ("bad.txt", "x")])
def test_bad_files_raise_parse_error(tmp_path, name, payload):
    with pytest.raises(UDIError) as e:
        load_dataset(_write(tmp_path, name, payload))
    assert e.value.code == "DATASET_PARSE_ERROR"


def test_corrupt_excel_raises(tmp_path):
    p = _write(tmp_path, "x.xlsx", b"not a workbook", mode="b")
    with pytest.raises(UDIError) as e:
        load_dataset(p)
    assert e.value.code == "DATASET_PARSE_ERROR"


def test_profile_roles_on_unseen_schema():
    df = datasets.delivery_df()
    prof = profile_dataframe(df, "delivery.csv")
    roles = {c["name"]: c["role"] for c in prof["columns"]}
    assert roles["Order_ID"] == "identifier"
    assert roles["Order_Date"] == "datetime"
    assert roles["Distance"] == "numeric" and roles["Delivery_Time"] == "numeric"
    assert roles["City"] == "categorical" and roles["Status"] == "binary"
    assert prof["primary_time_column"] == "Order_Date" and prof["date_range"]["frequency"] == "D"
    assert prof["rows"] == len(df) and prof["identifier_columns"] == ["Order_ID"]


def test_high_cardinality_numeric_business_variables_are_not_dropped():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({"Revenue": rng.normal(1000, 200, n).round(2), "Customers": rng.integers(1, 5000, n),
                       "Inventory": rng.uniform(0, 1e5, n), "Customer_ID": np.arange(n) + 1})
    roles = {c["name"]: c["role"] for c in profile_dataframe(df)["columns"]}
    assert roles["Revenue"] == roles["Customers"] == roles["Inventory"] == "numeric"
    assert roles["Customer_ID"] == "identifier"


def test_duplicates_and_invalid_values_reported(tmp_path):
    rows = ["a,b"] + [f"{i},{i * 2}" for i in range(20)] + ["x,5", "3,6", "3,6"]
    prof = profile_dataframe(load_dataset(_write(tmp_path, "w.csv", "\n".join(rows))), "w.csv")
    assert prof["duplicate_rows"] >= 1
    assert any("could not be parsed" in w for w in prof["warnings"])


def test_llm_profile_never_contains_the_dataset():
    df = datasets.delivery_df(2000)
    prof = profile_dataframe(df, "d.csv")
    view = profile_for_llm(prof, df, sample_rows=3)
    assert len(view["sample_rows"]) == 3 and view["rows"] == 2000
    text = str(view)
    assert len(text) < 12000                                   # schema + stats, not 2000 rows
    assert "100500" not in text                                # an ID from deep inside the data
    assert "sample_rows" not in profile_for_llm(prof, df, sample_rows=0)


@pytest.mark.parametrize("domain", list(DOMAINS))
def test_seven_domain_samples_load_and_profile(domain):
    df = load_dataset(SAMPLE_DIR / f"{domain}.csv")
    prof = profile_dataframe(df, f"{domain}.csv")
    assert len(df) == 900 and prof["primary_time_column"] == "Date"
    assert len(prof["numeric_columns"]) >= 3


def test_seven_domains_configured_as_context_only():
    assert sorted(DOMAINS) == sorted(["company", "hospital", "college", "municipality", "ngo", "community", "event_organizer"])
    for cfg in DOMAINS.values():
        assert {"label", "icon", "description", "context"} <= set(cfg)
        assert "target" not in cfg and "forecast_target" not in cfg      # no domain-fixed targets
