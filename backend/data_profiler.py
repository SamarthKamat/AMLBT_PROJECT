"""Dataset loading (CSV / XLSX / XLS), type coercion and profiling.

Nothing here knows about any particular business domain or column name. Column *roles*
(numeric, categorical, datetime, identifier, ...) are inferred from the data itself.
"""
from __future__ import annotations

import csv
import io
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from errors import UDIError

NA_VALUES = ["", " ", "NA", "N/A", "n/a", "NaN", "nan", "NULL", "null", "None", "none", "-", "--", "#N/A", "#NA", "?"]
_ID_NAME = re.compile(r"(^|[^a-z])(id|uuid|guid|key|ref|idx|index|sku)([^a-z]|$)", re.I)
_CURRENCY = re.compile(r"[,\s₹$€£¥]")
FREQ_LABEL = {"H": "hourly", "D": "daily", "W": "weekly", "M": "monthly", "Q": "quarterly", "Y": "yearly"}


# --------------------------------------------------------------------------- loading
def _detect_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except Exception:
        return ","


def _read_csv_raw(path: Path) -> pd.DataFrame:
    last_err: Optional[Exception] = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            with open(path, "r", encoding=enc, newline="") as fh:
                head = fh.read(20000)
            sep = _detect_delimiter(head)
            return pd.read_csv(path, header=None, dtype=str, sep=sep, encoding=enc,
                               keep_default_na=False, skip_blank_lines=True, engine="c")
        except UnicodeDecodeError as exc:
            last_err = exc
            continue
        except pd.errors.EmptyDataError:
            raise UDIError("DATASET_PARSE_ERROR", "The CSV file is empty.")
    raise UDIError("DATASET_PARSE_ERROR", f"Could not decode the CSV file: {last_err}")


def _read_excel_raw(path: Path) -> pd.DataFrame:
    engine = "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"
    try:
        sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object, engine=engine)
    except ImportError as exc:
        raise UDIError("DATASET_PARSE_ERROR", f"Missing Excel reader dependency: {exc}")
    for name, frame in sheets.items():       # first sheet that actually contains data
        frame = frame.dropna(how="all")
        if not frame.empty:
            frame.attrs["sheet"] = name
            return frame
    raise UDIError("DATASET_PARSE_ERROR", "The workbook contains no data.")


def _detect_header_row(raw: pd.DataFrame) -> int:
    """First row (within the first 10) that looks like column names: mostly filled, mostly non-numeric."""
    head = raw.head(10)
    width = max(1, raw.shape[1])
    fills = [(row.notna() & (row.astype(str).str.strip() != "")).sum() / width for _, row in head.iterrows()]
    if not fills:
        return 0
    best_fill = max(fills)
    for i, (_, row) in enumerate(head.iterrows()):
        vals = [str(v).strip() for v in row if pd.notna(v) and str(v).strip() != ""]
        if not vals or fills[i] < 0.6 * best_fill or fills[i] < 0.5:
            continue
        numeric_like = sum(1 for v in vals if _is_number(v))
        if numeric_like / len(vals) <= 0.3:
            return i
    return 0


def _is_number(text: str) -> bool:
    try:
        float(_CURRENCY.sub("", text))
        return True
    except ValueError:
        return False


def _unique_names(values) -> List[str]:
    names, seen = [], {}
    for j, v in enumerate(values):
        base = str(v).strip() if pd.notna(v) and str(v).strip() else f"column_{j + 1}"
        if base in seen:
            seen[base] += 1
            names.append(f"{base}_{seen[base]}")
        else:
            seen[base] = 1
            names.append(base)
    return names


def _to_numeric_series(s: pd.Series) -> Optional[pd.Series]:
    """Convert a text/object series to numeric when >=90% of non-null values parse. Returns None otherwise."""
    nn = s.dropna()
    if nn.empty:
        return None
    txt = nn.astype(str).str.strip()
    txt = txt.str.replace(r"^\((.*)\)$", r"-\1", regex=True)       # (123) -> -123
    txt = txt.str.replace("%", "", regex=False)
    txt = txt.str.replace(_CURRENCY, "", regex=True)
    parsed = pd.to_numeric(txt, errors="coerce")
    if parsed.notna().mean() < 0.9:
        return None
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out.loc[nn.index] = parsed.values
    out = out.replace([np.inf, -np.inf], np.nan)
    if out.notna().all() and np.all(np.mod(out.to_numpy(), 1) == 0) and out.abs().max() < 2**53:
        return out.astype("int64")
    return out


_DATE_HINT = re.compile(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{1,2}[\s-][A-Za-z]{3,9}[\s,-]+\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}")


def _to_datetime_series(s: pd.Series) -> Optional[pd.Series]:
    nn = s.dropna()
    if nn.empty:
        return None
    if all(isinstance(v, (pd.Timestamp, np.datetime64)) or hasattr(v, "year") and hasattr(v, "month") for v in nn.head(50)):
        out = pd.to_datetime(s, errors="coerce")
        return out.dt.tz_localize(None) if getattr(out.dt, "tz", None) is not None else out
    sample = nn.astype(str).str.strip().head(300)
    if sample.map(lambda v: bool(_DATE_HINT.search(v))).mean() < 0.9:
        return None
    dayfirst = False
    for v in sample:
        m = re.match(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})", v)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > 12:
                dayfirst = True
                break
            if b > 12:
                dayfirst = False
                break
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(s.astype("string").str.strip(), errors="coerce", dayfirst=dayfirst, format="mixed")
    if parsed.notna().sum() / max(1, len(nn)) < 0.9 or parsed.nunique() < 3:
        return None
    return parsed.dt.tz_localize(None) if getattr(parsed.dt, "tz", None) is not None else parsed


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    coerced_invalid: Dict[str, int] = {}
    for col in df.columns:
        s = df[col]
        if s.dtype == object or str(s.dtype).startswith("string") or str(s.dtype) == "str":
            s = s.map(lambda v: v.strip() if isinstance(v, str) else v)
            s = s.replace({v: np.nan for v in NA_VALUES}) if not s.empty else s
            num = _to_numeric_series(s)
            if num is not None:
                bad = int(s.notna().sum() - num.notna().sum())
                if bad:
                    coerced_invalid[col] = bad
                out[col] = num
                continue
            dt = _to_datetime_series(s)
            if dt is not None:
                bad = int(s.notna().sum() - dt.notna().sum())
                if bad:
                    coerced_invalid[col] = bad
                out[col] = dt
                continue
            out[col] = s.astype(object)
        else:
            out[col] = s
    res = pd.DataFrame(out, index=df.index)
    res.attrs["coerced_invalid"] = coerced_invalid
    return res


def load_dataset(path) -> pd.DataFrame:
    """Load a CSV/XLSX/XLS file into a typed DataFrame. Raises UDIError(DATASET_PARSE_ERROR)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in (".csv", ".xlsx", ".xls"):
        raise UDIError("DATASET_PARSE_ERROR", "Only CSV, XLSX and XLS files are supported.")
    try:
        raw = _read_csv_raw(path) if suffix == ".csv" else _read_excel_raw(path)
        raw = raw.dropna(how="all").reset_index(drop=True)
        if raw.empty or raw.shape[0] < 2:
            raise UDIError("DATASET_PARSE_ERROR", "The file has no data rows below the header.")
        sheet = raw.attrs.get("sheet")
        header_idx = _detect_header_row(raw)
        names = _unique_names(raw.iloc[header_idx].tolist())
        body = raw.iloc[header_idx + 1:].copy()
        body.columns = names
        body = body.reset_index(drop=True)
        body = body.dropna(axis=1, how="all").dropna(how="all")
        body = body.loc[:, [c for c in body.columns if not (re.match(r"^column_\d+$", c) and body[c].isna().all())]]
        if body.empty:
            raise UDIError("DATASET_PARSE_ERROR", "The file has a header but no usable data rows.")
        df = coerce_types(body).reset_index(drop=True)
        df.attrs["sheet"] = sheet
        df.attrs["header_row"] = int(header_idx)
        return df
    except UDIError:
        raise
    except Exception as exc:  # corrupted workbook, bad quoting, ...
        raise UDIError("DATASET_PARSE_ERROR", f"Could not parse the dataset: {exc}")


# --------------------------------------------------------------------------- column roles
def _is_integer_valued(s: pd.Series) -> bool:
    nn = s.dropna()
    return len(nn) > 0 and bool(np.all(np.mod(nn.to_numpy(dtype=float), 1) == 0))


def infer_roles(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Role of every column, inferred from values (and weakly from the name for identifiers)."""
    n = len(df)
    roles: Dict[str, Dict[str, Any]] = {}
    for col in df.columns:
        s = df[col]
        nn = s.dropna()
        nun = int(nn.nunique())
        info: Dict[str, Any] = {"role": "categorical", "high_cardinality": False, "reason": ""}
        if len(nn) == 0:
            info.update(role="empty", reason="all values missing")
        elif pd.api.types.is_datetime64_any_dtype(s):
            info.update(role="datetime")
        elif pd.api.types.is_bool_dtype(s):
            info.update(role="binary", reason="boolean")
        elif pd.api.types.is_numeric_dtype(s):
            unique_ratio = nun / max(1, len(nn))
            steps = nn.diff().dropna().abs()
            sequential = (_is_integer_valued(s) and unique_ratio > 0.98 and n >= 30
                          and (nn.is_monotonic_increasing or nn.is_monotonic_decreasing)
                          and len(steps) > 0 and float((steps == 1).mean()) >= 0.95)       # 1,2,3,... style row index
            if nun <= 1:
                info.update(role="constant", reason="single distinct value")
            elif _ID_NAME.search(str(col)) and unique_ratio >= 0.5 and _is_integer_valued(s) and n >= 20:
                info.update(role="identifier", reason="identifier-like name with mostly unique integer values")
            elif sequential:
                info.update(role="identifier", reason="sequential integer index")
            elif nun == 2:
                info.update(role="binary")
            else:
                info.update(role="numeric")
        else:
            lengths = nn.astype(str).str.len()
            unique_ratio = nun / max(1, len(nn))
            if nun <= 1:
                info.update(role="constant", reason="single distinct value")
            elif n >= 20 and ((_ID_NAME.search(str(col)) and (unique_ratio >= 0.5 or nun > 50)) or (unique_ratio >= 0.9 and nun > 50 and lengths.mean() <= 40)):
                info.update(role="identifier", reason="identifier-like column (names/IDs are not predictive features)")
            elif lengths.mean() > 40 and unique_ratio > 0.5:
                info.update(role="text", reason="free text")
            else:
                info.update(role="binary" if nun == 2 else "categorical", high_cardinality=nun > 50)
                if nun > 50:
                    info["reason"] = f"{nun} distinct values"
        roles[col] = info
    return roles


def _infer_frequency(ts: pd.Series) -> Dict[str, Any]:
    uniq = pd.Series(pd.to_datetime(ts.dropna().unique())).sort_values()
    if len(uniq) < 3:
        return {"frequency": None, "periods": int(len(uniq))}
    delta_days = uniq.diff().dropna().dt.total_seconds().median() / 86400.0
    if delta_days < 0.9:
        f = "H"
    elif delta_days < 1.5:
        f = "D"
    elif 5.5 <= delta_days <= 8.5:
        f = "W"
    elif 26 <= delta_days <= 33:
        f = "M"
    elif 80 <= delta_days <= 100:
        f = "Q"
    elif 350 <= delta_days <= 380:
        f = "Y"
    else:
        f = "D" if delta_days < 26 else "M"
    return {"frequency": f, "periods": int(len(uniq)), "median_gap_days": round(float(delta_days), 3)}


# --------------------------------------------------------------------------- profile
def profile_dataframe(df: pd.DataFrame, name: str = "dataset") -> Dict[str, Any]:
    roles = infer_roles(df)
    n = len(df)
    cols: List[Dict[str, Any]] = []
    time_info: Dict[str, Any] = {}
    for c in df.columns:
        s = df[c]
        nn = s.dropna()
        r = roles[c]
        entry: Dict[str, Any] = {
            "name": c,
            "dtype": "datetime" if r["role"] == "datetime" else ("numeric" if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) else "categorical" if r["role"] in ("categorical", "binary") else r["role"]),
            "role": r["role"],
            "unique_values": int(nn.nunique()),
            "missing": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "identifier_like": r["role"] == "identifier",
        }
        if r.get("high_cardinality"):
            entry["high_cardinality"] = True
        if r.get("reason"):
            entry["note"] = r["reason"]
        if r["role"] in ("numeric", "binary") and pd.api.types.is_numeric_dtype(s) and len(nn):
            x = nn.astype(float)
            entry.update(min=float(x.min()), max=float(x.max()), mean=float(x.mean()), median=float(x.median()),
                         std=float(x.std()) if len(x) > 1 else None, integer_valued=_is_integer_valued(s))
        elif r["role"] == "datetime" and len(nn):
            entry.update(start=str(nn.min()), end=str(nn.max()))
            time_info[c] = _infer_frequency(s)
            entry["inferred_frequency"] = time_info[c]["frequency"]
            entry["distinct_timestamps"] = time_info[c]["periods"]
            entry["rows_per_timestamp"] = round(len(nn) / max(1, time_info[c]["periods"]), 2)
        elif r["role"] in ("categorical", "binary") and len(nn):
            vc = nn.astype(str).value_counts()
            entry["top_values"] = [{"value": str(k)[:60], "count": int(v)} for k, v in vc.head(25 if entry["unique_values"] <= 30 else 10).items()]
        cols.append(entry)

    datetime_cols = [c["name"] for c in cols if c["role"] == "datetime"]
    primary_time = None
    if datetime_cols:
        primary_time = max(datetime_cols, key=lambda c: time_info.get(c, {}).get("periods", 0))
    date_range = None
    if primary_time:
        d = df[primary_time].dropna()
        date_range = {"column": primary_time, "start": str(d.min()), "end": str(d.max()),
                      "frequency": time_info[primary_time]["frequency"]}
    warnings_list: List[str] = []
    invalid = df.attrs.get("coerced_invalid") or {}
    for col, k in invalid.items():
        warnings_list.append(f"{k} value(s) in '{col}' could not be parsed and were treated as missing.")
    dup = int(df.duplicated().sum())
    if dup:
        warnings_list.append(f"{dup} exact duplicate row(s) detected.")
    return {
        "dataset_name": name,
        "rows": int(n),
        "columns": cols,
        "column_names": list(df.columns),
        "numeric_columns": [c["name"] for c in cols if c["role"] in ("numeric", "binary") and pd.api.types.is_numeric_dtype(df[c["name"]]) and not pd.api.types.is_bool_dtype(df[c["name"]])],
        "categorical_columns": [c["name"] for c in cols if c["role"] in ("categorical", "binary")],
        "datetime_columns": datetime_cols,
        "text_columns": [c["name"] for c in cols if c["role"] == "text"],
        "identifier_columns": [c["name"] for c in cols if c["role"] == "identifier"],
        "primary_time_column": primary_time,
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": dup,
        "date_range": date_range,
        "sheet": df.attrs.get("sheet"),
        "warnings": warnings_list,
    }


def profile_for_llm(profile: Dict[str, Any], df: pd.DataFrame, sample_rows: int = 3) -> Dict[str, Any]:
    """Compact, schema-level view of the dataset for Gemini. The dataset itself is never sent."""
    cols = []
    for c in profile["columns"][:200]:
        item = {k: v for k, v in c.items() if k in (
            "name", "role", "unique_values", "missing_pct", "min", "max", "mean", "median", "start", "end",
            "inferred_frequency", "rows_per_timestamp", "integer_valued", "high_cardinality")}
        if "top_values" in c:
            item["values"] = [t["value"] for t in c["top_values"]]
        cols.append(item)
    out = {
        "rows": profile["rows"],
        "columns": cols,
        "primary_time_column": profile.get("primary_time_column"),
        "date_range": profile.get("date_range"),
    }
    if sample_rows > 0:
        head = df.head(sample_rows).copy()
        for c in head.columns:
            head[c] = head[c].astype(str).str.slice(0, 40)
        out["sample_rows"] = head.to_dict("records")
    return out
