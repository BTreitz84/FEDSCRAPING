from __future__ import annotations

import io
import os
import re
from collections import Counter
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


def load_soup(filepath: str) -> BeautifulSoup:
    with open(filepath, "r", encoding="utf-8") as f:
        return BeautifulSoup(f, "html.parser")


def date_from_filename(filepath: str) -> pd.Timestamp:
    m = re.search(r"(\d{8})", os.path.basename(filepath))
    if not m:
        raise ValueError(f"Could not find an 8-digit date in filename: {filepath}")
    return pd.Timestamp(datetime.strptime(m.group(1), "%Y%m%d"))


def _preceding_heading_text(table_tag, max_chars: int = 200, node_limit: int = 300) -> str:
    texts = []
    total = 0
    for i, el in enumerate(table_tag.find_all_previous(string=True)):
        if i >= node_limit:
            break
        t = el.strip()
        if t:
            texts.append(t)
            total += len(t)
        if total >= max_chars:
            break
    return " ".join(reversed(texts))[-max_chars:]


def find_tables_by_summary(soup: BeautifulSoup, summary_substring: str) -> list:
    target = summary_substring.lower()
    matches = []
    for table in soup.find_all("table"):
        summary = (table.get("summary", "") or "").lower()
        if target in summary:
            matches.append(table)
        elif not summary:
            preceding = _preceding_heading_text(table).lower()
            if target in preceding:
                matches.append(table)
    return [t for t in matches if not _looks_like_footnotes_table(t)]


_FOOTNOTE_NUMBER_ONLY = re.compile(r"^\d{1,2}\.?$")


def _looks_like_footnotes_table(table_tag) -> bool:
    first_col_cells = [tr.find(["td", "th"]) for tr in table_tag.find_all("tr")]
    texts = [c.get_text(strip=True) for c in first_col_cells if c is not None and c.get_text(strip=True)]
    if len(texts) < 2:
        return False
    return sum(bool(_FOOTNOTE_NUMBER_ONLY.match(t)) for t in texts) / len(texts) > 0.5


def _looks_numeric(cell: str) -> bool:
    c = cell.strip().replace(",", "").replace("\xa0", "")
    c = c.replace("−", "-")
    if c == "" or re.fullmatch(r"-+", c) or re.fullmatch(r"\.+", c):


        return True
    c = re.sub(r"\s*;\s*$", "", c)
    if re.fullmatch(r"\(\d+(?:\.\d+)?\)", c):
        return True
    c = re.sub(r"^[+-]\s*", "", c)
    return bool(re.fullmatch(r"\d+(\.\d+)?", c))


def table_to_grid(table_tag) -> pd.DataFrame:
    rows = table_tag.find_all("tr")


    widths = [sum(int(c.get("colspan", 1)) for c in tr.find_all(["td", "th"])) for tr in rows]
    width_counts = Counter(w for w in widths if w > 0)
    n_cols = width_counts.most_common(1)[0][0] if width_counts else 0
    n_rows = len(rows)

    grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]

    occupied = [[False] * n_cols for _ in range(n_rows)]

    for r, tr in enumerate(rows):
        c = 0
        for cell in tr.find_all(["td", "th"]):
            while c < n_cols and occupied[r][c]:
                c += 1
            if c >= n_cols:
                break
            raw_text = cell.get_text(strip=False)


            headers_attr = cell.get("headers", "")
            if not isinstance(headers_attr, str):
                headers_attr = " ".join(headers_attr)


            style_attr = (cell.get("style", "") or "").lower()


            is_indented = c == 0 and (
                raw_text.startswith("  ")
                or len(headers_attr.split()) > 1
                or ("padding-left" in style_attr and "text-indent" not in style_attr)
            )
            text = cell.get_text(separator=" ", strip=True)
            text = text.replace(" ", " ").replace(" ", " ").replace(" ", " ")
            text = re.sub(r"\s+", " ", text).strip()
            if is_indented and text:
                text = "__INDENT__" + text
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))
            for dr in range(rowspan):
                for dc in range(colspan):
                    rr, cc = r + dr, c + dc
                    if rr < n_rows and cc < n_cols:
                        grid[rr][cc] = text
                        occupied[rr][cc] = True
            c += colspan

    return pd.DataFrame(grid)


_MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
_DATE_PATTERN = re.compile(rf"{_MONTHS}\s+\d{{1,2}},?\s+\d{{2,4}}", re.IGNORECASE)


def strip_dates_with_qualifier(text: str, release_date: pd.Timestamp, column_is_ambiguous: bool = True) -> str:
    match = _DATE_PATTERN.search(text)
    if not match:
        return text.strip()
    ref_date = pd.to_datetime(match.group(0), errors="coerce")
    out = _DATE_PATTERN.sub("", text)
    out = re.sub(r"[\s,-]+$", "", out).strip()
    if pd.isna(ref_date) or pd.isna(release_date):
        return out
    if not column_is_ambiguous:
        return out
    diff_days = (release_date - ref_date).days
    if 3 <= diff_days <= 10:
        qualifier = "(1wk ago)"
    elif 350 <= diff_days <= 380:
        qualifier = "(1yr ago)"
    elif -2 <= diff_days <= 2:
        qualifier = ""
    else:
        qualifier = f"({diff_days}d ago)"
    return f"{out} {qualifier}".strip()


_ASOF_WEDNESDAY_PATTERN = re.compile(rf"^Wednesday\s*,?\s+({_MONTHS}\s+\d{{1,2}},?\s+\d{{2,4}})", re.IGNORECASE)


_ASOF_WEDNESDAY_PATTERN_REVERSED = re.compile(rf"^({_MONTHS}\s+\d{{1,2}},?\s+\d{{2,4}})\s+Wednesday$", re.IGNORECASE)


def extract_true_asof_date(grid: pd.DataFrame, fallback: pd.Timestamp) -> pd.Timestamp:
    naive_wednesday = fallback - pd.Timedelta(days=1)
    header_rows = min(3, len(grid))
    for r in range(header_rows):
        for c in range(grid.shape[1]):
            cell = str(grid.iat[r, c]).strip()
            m = _ASOF_WEDNESDAY_PATTERN.match(cell) or _ASOF_WEDNESDAY_PATTERN_REVERSED.match(cell)
            if m:
                parsed = pd.to_datetime(m.group(1), errors="coerce")
                if pd.notna(parsed) and abs((parsed - naive_wednesday).days) <= 5:
                    return parsed


            if cell.lower() == "wednesday" and r + 1 < len(grid):
                below = str(grid.iat[r + 1, c]).strip()
                date_m = _DATE_PATTERN.search(below)
                if date_m:
                    parsed = pd.to_datetime(date_m.group(0), errors="coerce")
                    if pd.notna(parsed) and abs((parsed - naive_wednesday).days) <= 5:
                        return parsed
    return naive_wednesday


def strip_dates(text: str) -> str:
    out = _DATE_PATTERN.sub("", text)
    out = re.sub(r"[\s,-]+$", "", out).strip()
    return out


_FOOTNOTE_PATTERN = re.compile(r"\s*\(\d+(?:,\s*\d+)*\)\s*$|\s+\d{1,2}$")
_CONTINUATION_ENDING = re.compile(
    r"(,|:|\b(?:to|of|for|than|through|with|and|or|in|by|on|at|from|under|as))\s*$",
    re.IGNORECASE,
)


def strip_footnote_marker(label: str) -> str:
    return _FOOTNOTE_PATTERN.sub("", label).strip()


def merge_continuation_rows(rows: list[dict]) -> list[dict]:
    if not rows:
        return rows
    by_row_position: dict[int, list[dict]] = {}
    for r in rows:
        by_row_position.setdefault(r["_row_idx"], []).append(r)

    positions = sorted(by_row_position)
    pending_prefix = None
    out = []
    for pos in positions:
        cells = by_row_position[pos]
        label = cells[0]["row_label_raw"]
        all_blank = all(not c["value_raw"].strip() for c in cells)
        if all_blank and _CONTINUATION_ENDING.search(label):
            pending_prefix = (pending_prefix + " " if pending_prefix else "") + label
            continue
        if pending_prefix:
            label = f"{pending_prefix} {label}".strip()
            pending_prefix = None
        for c in cells:
            c["row_label_raw"] = label
        out.extend(cells)
    return out


def grid_to_tidy(grid: pd.DataFrame, date: pd.Timestamp, source: str = "", enable_parent_prefix: bool = False) -> pd.DataFrame:
    n_rows, n_cols = grid.shape
    if n_cols < 2 or n_rows < 2:
        return pd.DataFrame(columns=["date", "row_label_raw", "col_label_raw", "value_raw", "source"])

    header_row_count = 0
    for i in range(n_rows):
        data_cells = grid.iloc[i, 1:]
        non_empty = [c for c in data_cells if c.strip()]
        if non_empty and sum(_looks_numeric(c) for c in non_empty) / len(non_empty) > 0.5:
            header_row_count = i
            break
    else:
        header_row_count = 1

    header_rows = grid.iloc[:header_row_count, 1:]
    data = grid.iloc[header_row_count:, :].reset_index(drop=True)


    col_labels = []
    for col_idx in range(header_rows.shape[1]):
        col_cells = [str(v).strip() for v in header_rows.iloc[:, col_idx] if str(v).strip()]


        column_is_ambiguous = any("change from" in c.lower() for c in col_cells)
        parts = [strip_dates_with_qualifier(c, date, column_is_ambiguous) for c in col_cells]
        parts = [p for p in parts if p]
        seen = []
        for p in parts:
            if p not in seen:
                seen.append(p)
        col_labels.append(" - ".join(seen) if seen else f"col_{col_idx}")

    rows = []
    current_parent = None
    current_subsection = None
    for row_idx, (_, row) in enumerate(data.iterrows()):
        row_label = str(row.iloc[0]).strip()
        if not row_label:
            continue


        row_has_data = any(str(v).strip() for v in row.iloc[1:])


        is_indented = row_label.startswith("__INDENT__")
        bare_for_subsection = row_label[len("__INDENT__"):] if is_indented else row_label
        if not is_indented:
            current_subsection = None
        elif not row_has_data:
            current_subsection = strip_footnote_marker(bare_for_subsection).strip().lower()
        elif (current_subsection == "federal agency obligations"
              and bare_for_subsection.strip().lower() in ("bought outright", "held under repurchase agreements")):
            row_label = "__INDENT__" + f"Federal agency obligations - {bare_for_subsection.strip()}"

        if enable_parent_prefix:


            bare_label = row_label[len("__INDENT__"):] if row_label.startswith("__INDENT__") else row_label
            if bare_label.strip().lower() in ("holdings", "weekly changes"):
                row_label = "__INDENT__" + bare_label.strip()
            if row_label.startswith("__INDENT__"):
                row_label = row_label[len("__INDENT__"):]
                if current_parent and row_has_data:
                    row_label = f"{current_parent} - {row_label}"
            else:


                current_parent = strip_footnote_marker(row_label)
        elif row_label.startswith("__INDENT__"):


            row_label = row_label[len("__INDENT__"):]
        for col_idx, col_label in enumerate(col_labels, start=1):
            if col_idx >= len(row):
                continue
            value_raw = str(row.iloc[col_idx]).strip()
            rows.append(
                {
                    "_row_idx": row_idx,
                    "date": date,
                    "row_label_raw": row_label,
                    "col_label_raw": col_label,
                    "value_raw": value_raw,
                    "source": source,
                }
            )
    rows = merge_continuation_rows(rows)


    cutoff = next((i for i, r in enumerate(rows) if "memo (off-balance-sheet" in r["row_label_raw"].lower()), None)
    if cutoff is not None:
        rows = rows[:cutoff]

    for r in rows:
        r["row_label_raw"] = strip_footnote_marker(r["row_label_raw"])
        del r["_row_idx"]
    return pd.DataFrame(rows)


def extract_tidy(filepath: str, summary_substring: str, enable_parent_prefix: bool = False) -> pd.DataFrame:
    soup = load_soup(filepath)
    fallback_date = date_from_filename(filepath)
    tables = find_tables_by_summary(soup, summary_substring)
    grids = [table_to_grid(t) for t in tables]


    date = fallback_date
    for g in grids:
        date = extract_true_asof_date(g, fallback_date)
        if date != fallback_date:
            break
    frames = [
        grid_to_tidy(g, date=date, source=os.path.basename(filepath), enable_parent_prefix=enable_parent_prefix)
        for g in grids
    ]
    if not frames:
        return pd.DataFrame(columns=["date", "row_label_raw", "col_label_raw", "value_raw", "source"])
    combined = pd.concat(frames, ignore_index=True)


    combined["_has_value"] = combined["value_raw"].astype(str).str.strip().ne("")
    combined = (
        combined.sort_values("_has_value", ascending=False)
        .drop_duplicates(subset=["date", "row_label_raw", "col_label_raw"], keep="first")
        .drop(columns="_has_value")
        .sort_index()
    )
    return combined


def extract_tidy_many(filepaths: list[str], summary_substring: str, verbose: bool = True, enable_parent_prefix: bool = False) -> pd.DataFrame:
    frames = []
    failures = []
    for fp in filepaths:
        try:
            frames.append(extract_tidy(fp, summary_substring, enable_parent_prefix=enable_parent_prefix))
        except Exception as exc:
            failures.append((fp, str(exc)))
    if verbose and failures:
        print(f"{len(failures)} file(s) failed to parse:")
        for fp, msg in failures[:10]:
            print(f"  {os.path.basename(fp)}: {msg}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def swap_row_col_before_date(tidy: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    out = tidy.copy()
    mask = out["date"] < cutoff_date
    out.loc[mask, ["row_label_raw", "col_label_raw"]] = out.loc[mask, ["col_label_raw", "row_label_raw"]].values
    return out


def clean_value(raw: str) -> float:
    if raw is None:
        return np.nan
    s = str(raw).strip().replace("\xa0", "").replace(",", "")
    s = s.replace("−", "-")
    if s == "" or re.fullmatch(r"-+", s) or re.fullmatch(r"\.+", s):
        return 0.0


    s = re.sub(r"\s*;\s*$", "", s)

    paren = re.fullmatch(r"\((\d+(?:\.\d+)?)\)", s)
    if paren:
        s = "-" + paren.group(1)
    s = re.sub(r"^\+\s*", "", s)
    s = re.sub(r"^-\s*", "-", s)
    try:
        return float(s)
    except ValueError:
        return np.nan


def add_clean_value(tidy: pd.DataFrame) -> pd.DataFrame:
    out = tidy.copy()
    out["value"] = out["value_raw"].map(clean_value)
    out["parse_failed"] = out["value"].isna() & ~out["value_raw"].astype(str).str.fullmatch(r"-*|\.*").fillna(False)
    return out


def label_inventory(tidy: pd.DataFrame, label_col: str) -> pd.DataFrame:
    g = tidy.groupby(label_col)["date"].agg(["min", "max", "count"])
    g.columns = ["first_seen", "last_seen", "n_obs"]
    return g.reset_index().sort_values("first_seen")


def unmapped_labels(inventory: pd.DataFrame, crosswalk: pd.DataFrame, label_col: str, raw_col: str = "raw") -> pd.DataFrame:
    known = set(crosswalk[raw_col].astype(str))
    mask = ~inventory[label_col].astype(str).isin(known)
    return inventory.loc[mask].reset_index(drop=True)


def load_or_init_crosswalk(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_excel(path)
    return pd.DataFrame(columns=["raw", "standardized"])


def apply_crosswalk(tidy: pd.DataFrame, crosswalk: pd.DataFrame, raw_label_col: str, out_col: str = "standardized") -> pd.DataFrame:
    mapping = crosswalk.dropna(subset=["standardized"]).set_index("raw")["standardized"].to_dict()
    out = tidy.copy()
    out[out_col] = out[raw_label_col].map(mapping)
    return out


def check_components_sum_to_total(
    wide: pd.DataFrame,
    component_cols: list[str],
    total_col: str,
    tolerance: float = 1.0,
) -> pd.DataFrame:
    diffs = wide[component_cols].sum(axis=1) - wide[total_col]
    bad = wide.loc[diffs.abs() > tolerance].copy()
    bad["reconciliation_diff"] = diffs.loc[bad.index]
    return bad


def check_week_over_week_jumps(series: pd.Series, n_std: float = 8.0) -> pd.Series:
    diffs = series.diff()
    threshold = n_std * diffs.std()
    return diffs.abs() > threshold


def save_wide_excel(
    tidy: pd.DataFrame,
    index_cols: list[str],
    out_path: str,
    date_col: str = "date",
    value_col: str = "value",
    extra_col: Optional[pd.Series] = None,
) -> pd.DataFrame:
    wide = tidy.pivot_table(index=index_cols, columns=date_col, values=value_col, aggfunc="last").sort_index()
    wide.columns = [c.date() if hasattr(c, "date") else c for c in wide.columns]
    if extra_col is not None:
        wide.insert(0, extra_col.name, extra_col.reindex(wide.index))
    wide.to_excel(out_path)
    print(f"Saved {wide.shape[0]:,} rows x {wide.shape[1]:,} date columns to: {out_path}")
    return wide


def coverage_report(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in wide.columns:
        s = wide[col]
        non_na = s.notna()
        rows.append(
            {
                "column": col,
                "first_seen": s[non_na].index.min() if non_na.any() else pd.NaT,
                "last_seen": s[non_na].index.max() if non_na.any() else pd.NaT,
                "coverage": non_na.mean(),
            }
        )
    return pd.DataFrame(rows)
