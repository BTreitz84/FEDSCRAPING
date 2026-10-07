from __future__ import annotations

import re
import time
from typing import Iterable, Optional, Sequence

import pandas as pd
import requests

FISCAL_SERVICE_BASE = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"

AUCTIONS_ENDPOINT = f"{FISCAL_SERVICE_BASE}/v1/accounting/od/auctions_query"
MSPD_TABLE_3_ENDPOINT = f"{FISCAL_SERVICE_BASE}/v1/debt/mspd/mspd_table_3"
MSPD_TABLE_5_ENDPOINT = f"{FISCAL_SERVICE_BASE}/v1/debt/mspd/mspd_table_5"

AUCTIONS_DEFAULT_FIELDS: Sequence[str] = (
    "auction_date", "issue_date", "maturity_date",
    "security_type", "security_term", "cusip",
    "offering_amt", "total_accepted", "announcemt_date",
)

AUCTIONS_EXTRA_FIELDS: set[str] = {
    "total_tendered", "bid_to_cover_ratio",
    "avg_med_yield", "avg_med_price", "avg_med_discnt_rate", "avg_med_investment_rate",
    "high_yield", "high_price", "high_discnt_rate", "high_investment_rate",
    "low_yield", "low_price", "low_discnt_rate", "low_investment_rate",
    "price_per100", "unadj_price", "adj_price",
    "direct_bidder_accepted", "indirect_bidder_accepted", "primary_dealer_accepted",
    "comp_accepted", "comp_tendered", "noncomp_accepted", "noncomp_tenders_accepted",
    "soma_accepted", "soma_tendered", "soma_holdings", "soma_included",
    "security_term_day_month", "security_term_week_year",
    "pdf_filenm_announcemt", "xml_filenm_announcemt",
}

_NUMERIC_PAT = re.compile(
    r"(?:^|_)(amt|accepted|tendered|tenders|price|yield|rate|ratio|margin|pct|decimals|holdings|spread|max|min|int$|int_|sum)(?:_|$)"
)

DATE_COLUMNS = (
    "auction_date", "issue_date", "maturity_date", "announcemt_date",
    "dated_date", "mat_date", "original_issue_date", "record_date",
    "interest_pay_date_1", "interest_pay_date_2", "interest_pay_date_3", "interest_pay_date_4",
)


def _peek_keys(endpoint: str, date_col: str, start: Optional[str], end: Optional[str]) -> list[str]:
    filt = []
    if start:
        filt.append(f"{date_col}:gte:{start}")
    if end:
        filt.append(f"{date_col}:lte:{end}")
    params = {"format": "json", "page[size]": 1, "page[number]": 1}
    if filt:
        params["filter"] = ",".join(filt)
    r = requests.get(endpoint, params=params, timeout=90)
    r.raise_for_status()
    data = r.json().get("data", [])
    return list(data[0].keys()) if data else []


def _resolve_fields(desired: Iterable[str], keys_available: Iterable[str]) -> list[str]:
    avail = set(keys_available)
    fields = sorted(set(desired) & avail)
    if not fields:
        fields = sorted(avail)
    return fields


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    updates = {}
    for col in DATE_COLUMNS:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            s = df[col].astype("string", copy=False).str.slice(0, 10)
            parsed = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce", utc=True)
            updates[col] = parsed.dt.tz_convert(None).dt.normalize()
    for col in df.columns:
        if _NUMERIC_PAT.search(col):
            updates[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("security_type", "security_term", "security_term_day_month",
                "security_term_week_year", "cusip", "series", "series_cd",
                "security_type_desc", "security_class1_desc", "security_class2_desc", "security_class3_desc",
                "noncomp_tenders_accepted", "comp_tenders_accepted", "treas_retail_tenders_accepted",
                "floating_rate", "soma_included"):
        if col in df.columns:
            updates[col] = df[col].astype("string", copy=False).str.strip()
    if updates:
        df = df.assign(**updates)
    return df.copy()


def fetch_fiscaldata_paginated(
    endpoint: str,
    date_col: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    desired_fields: Optional[Iterable[str]] = None,
    extra_fields: Optional[Iterable[str]] = None,
    sort: Optional[str] = None,
    page_size: int = 10_000,
    pause: float = 0.08,
) -> pd.DataFrame:
    keys = _peek_keys(endpoint, date_col, start, end)
    if desired_fields is not None:
        want = set(desired_fields)
        if extra_fields:
            want |= set(extra_fields)
        fields = _resolve_fields(want, keys)
    else:
        fields = sorted(keys)

    filt = []
    if start:
        filt.append(f"{date_col}:gte:{start}")
    if end:
        filt.append(f"{date_col}:lte:{end}")

    params = {
        "format": "json",
        "fields": ",".join(fields),
        "page[size]": page_size,
        "page[number]": 1,
    }
    if sort:
        params["sort"] = sort
    if filt:
        params["filter"] = ",".join(filt)

    frames: list[pd.DataFrame] = []
    max_retries = 4
    page_number = 1

    with requests.Session() as session:
        while True:
            params["page[number]"] = page_number

            for attempt in range(max_retries):
                r = session.get(endpoint, params=params, timeout=120)
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(min(0.3 * (2 ** attempt), 2.4))
                    continue
                break

            if r.status_code != 200:
                try:
                    err = r.json()
                    if isinstance(err, dict) and "Page #" in str(err.get("message", "")):
                        break
                except Exception:
                    pass
                r.raise_for_status()

            j = r.json()
            data = j.get("data", [])
            if not data:
                break

            frames.append(pd.DataFrame(data))

            if len(data) < page_size:
                break

            page_number += 1
            if pause:
                time.sleep(pause)

    if not frames:
        return pd.DataFrame(columns=fields)

    df = pd.concat(frames, ignore_index=True)
    df = _coerce_types(df)
    if date_col in df.columns:
        df = df.sort_values(date_col).reset_index(drop=True)
    return df


def fetch_auctions(start: str = "1979-01-01", end: Optional[str] = None, include_extra: bool = False, page_size: int = 10_000, pause: float = 0.08) -> pd.DataFrame:
    extra = AUCTIONS_EXTRA_FIELDS if include_extra else None
    return fetch_fiscaldata_paginated(
        AUCTIONS_ENDPOINT, date_col="issue_date", start=start, end=end,
        desired_fields=AUCTIONS_DEFAULT_FIELDS, extra_fields=extra,
        sort="issue_date", page_size=page_size, pause=pause,
    )


def fetch_mspd(table: str = "table_3", start: Optional[str] = None, end: Optional[str] = None, page_size: int = 10_000, pause: float = 0.08) -> pd.DataFrame:
    endpoint = MSPD_TABLE_3_ENDPOINT if table == "table_3" else MSPD_TABLE_5_ENDPOINT
    return fetch_fiscaldata_paginated(
        endpoint, date_col="record_date", start=start, end=end,
        desired_fields=None, sort="record_date", page_size=page_size, pause=pause,
    )
