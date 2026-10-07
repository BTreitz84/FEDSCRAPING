from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.util.retry import Retry

BASE_URL = "https://www.federalreserve.gov/releases/h41/{date}/h41.htm"
FIRST_RELEASE = datetime(1997, 3, 6)
MANIFEST_COLUMNS = ["date", "status", "filepath", "fetched_at"]


def make_session(user_agent: str = "research-scraper (contact: set-your-email-here)") -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({"User-Agent": user_agent})
    return session


def candidate_dates(start: datetime, end: datetime, buffer_days: int = 3) -> list[datetime]:
    dates: set[datetime] = set()
    d = start
    while d <= end:
        days_ahead = (3 - d.weekday()) % 7
        thursday = d + timedelta(days=days_ahead)
        for offset in range(-buffer_days, buffer_days + 1):
            cand = thursday + timedelta(days=offset)
            if start <= cand <= end:
                dates.add(cand)
        d = thursday + timedelta(days=7)
    return sorted(dates)


def load_manifest(manifest_path: str) -> pd.DataFrame:
    if os.path.exists(manifest_path):
        return pd.read_csv(manifest_path, parse_dates=["date"])
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def save_manifest(df: pd.DataFrame, manifest_path: str) -> None:
    df.to_csv(manifest_path, index=False)


def fetch_one(session: requests.Session, date: datetime, out_dir: str, timeout: int = 15) -> tuple[str, Optional[str]]:
    date_str = date.strftime("%Y%m%d")
    filepath = os.path.join(out_dir, f"h41_{date_str}.html")
    if os.path.exists(filepath):
        return "cached", filepath

    url = BASE_URL.format(date=date_str)
    try:
        resp = session.get(url, timeout=timeout)
    except requests.RequestException as exc:
        return f"error:{exc}", None

    if resp.status_code != 200:
        return "not_found", None

    resp.encoding = resp.apparent_encoding or "utf-8"
    text = resp.text


    if "factors affecting reserve balances" not in text.lower():
        return "unexpected_content", None

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    return "downloaded", filepath


def run_scrape(
    out_dir: str,
    start: datetime = FIRST_RELEASE,
    end: Optional[datetime] = None,
    buffer_days: int = 3,
    sleep_sec: float = 1.0,
    max_requests: Optional[int] = None,
) -> pd.DataFrame:
    end = end or datetime.today()
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.csv")
    manifest = load_manifest(manifest_path)

    resolved_dates = set(
        manifest.loc[manifest["status"].isin(["downloaded", "cached", "not_found"]), "date"]
    )

    session = make_session()
    candidates = candidate_dates(start, end, buffer_days=buffer_days)

    new_rows = []
    n_requests = 0
    pbar = tqdm(candidates, unit="release")
    for d in pbar:
        if pd.Timestamp(d) in resolved_dates:
            continue
        if max_requests is not None and n_requests >= max_requests:
            pbar.close()
            print(f"Stopping early: hit max_requests={max_requests}")
            break

        status, filepath = fetch_one(session, d, out_dir)
        new_rows.append(
            {"date": d, "status": status, "filepath": filepath, "fetched_at": datetime.now().isoformat()}
        )
        pbar.set_description(f"{d.date()} {status}")

        if not status.startswith("cached"):
            n_requests += 1
            time.sleep(sleep_sec)

    if new_rows:
        manifest = pd.concat([manifest, pd.DataFrame(new_rows)], ignore_index=True)
        manifest = manifest.drop_duplicates(subset="date", keep="last").sort_values("date")
        save_manifest(manifest, manifest_path)

    return manifest


def summarize_manifest(manifest: pd.DataFrame) -> pd.Series:
    return manifest["status"].str.replace(r"^error:.*", "error", regex=True).value_counts()


def last_known_date(out_dir: str) -> Optional[datetime]:
    manifest_path = os.path.join(out_dir, "manifest.csv")
    if not os.path.exists(manifest_path):
        return None
    manifest = load_manifest(manifest_path)
    downloaded = manifest.loc[manifest["status"] == "downloaded", "date"]
    if downloaded.empty:
        return None
    return pd.Timestamp(downloaded.max()).to_pydatetime()


def update_h41_archive(out_dir: str, lookback_days: int = 10, sleep_sec: float = 1.0) -> pd.DataFrame:
    last = last_known_date(out_dir)
    start = (last - timedelta(days=lookback_days)) if last else FIRST_RELEASE
    end = datetime.today()
    print(f"Last known release on disk: {last.date() if last else 'none'}")
    print(f"Checking {start.date()} -> {end.date()}")
    return run_scrape(out_dir=out_dir, start=start, end=end, sleep_sec=sleep_sec)


def last_output_date(path: str) -> Optional[pd.Timestamp]:
    if not os.path.exists(path):
        return None
    dates = pd.read_excel(path, usecols=["date"])["date"]
    dates = pd.to_datetime(dates)
    return dates.max() if len(dates) else None


def report_archive_status(html_dir: str, output_paths: dict[str, str]) -> None:
    html_date = last_known_date(html_dir)
    print(f"Most recent H.4.1 release on disk: {html_date.date() if html_date else 'none'}")
    for label, path in output_paths.items():
        d = last_output_date(path)
        print(f"{label}: last date = {d.date() if d is not None else 'file not found'}")


def rerun_notebook(notebook_path: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", notebook_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{notebook_path} failed:\n{result.stdout}\n{result.stderr}")
    print(f"Re-ran {notebook_path}")
