from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil.parser import parse as dateparse

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

FRB_HOST = "www.federalreserve.gov"
HISTORICAL_INDEX = "https://www.federalreserve.gov/monetarypolicy/fomc_historical_year.htm"
CAL_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

PRESS_DIR_PAT = re.compile(r"^/newsevents/pressreleases/", re.I)

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4, "may": 5,
    "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12,
}

RE_HEADER = re.compile(
    r"(?P<m1>[A-Za-z]{3,9})(?:\s*/\s*(?P<m2>[A-Za-z]{3,9}))?\s+"
    r"(?P<d1>\d{1,2})(?:\s*(?:–|-|to|–|—)\s*(?P<d2>\d{1,2}))?"
    r"(?:,\s*(?P<y>\d{4}))?",
    re.I,
)

LABEL_MAP = {
    "Statement": ["statement", "press release", "policy statement"],
    "Minutes": ["minutes of the federal open market committee", "minutes"],
    "Transcripts": ["transcripts", "meeting transcript"],
    "Agenda": ["agenda"],
    "Record of Policy Actions": ["record of policy action", "record of policy actions"],
    "Tealbook": ["tealbook", "greenbook*tealbook"],
    "Greenbook": ["greenbook"],
    "Bluebook": ["bluebook"],
    "Redbook": ["redbook"],
    "Implementation Note": ["implementation note"],
    "Projections/SEP": ["summary of economic projections", "sep", "economic projections"],
    "Meeting Calendar": ["meeting information"],
}

MAIN_SELECTORS = [
    "#article", "[role='main']", ".pressContent", ".col-xs-12.col-sm-8",
    ".col-sm-8", ".content", "main", "article",
]


def norm(s: Optional[str]) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", s).strip()


def get_soup(url: str, **kwargs) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=30, **kwargs)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def filetype_of(href: str) -> Optional[str]:
    p = urlparse(href)
    path = p.path.lower()
    if path.endswith(".pdf"):
        return "pdf"
    if path.endswith(".htm") or path.endswith(".html") or path.endswith("/") or path == "":
        return "html"
    if PRESS_DIR_PAT.search(path):
        return "html"
    return None


def is_frb(href: str) -> bool:
    netloc = urlparse(href).netloc.lower()
    return netloc == "" or netloc == FRB_HOST


def is_doc_href(href: str) -> bool:
    if not href:
        return False
    if not is_frb(href):
        return False
    return filetype_of(href) in {"pdf", "html"}


def absolute_url(base: str, href: str) -> str:
    return urljoin(base, href)


def parse_meeting_header_date(header_text: str, fallback_year: Optional[int] = None) -> Optional[pd.Timestamp]:
    txt = norm(header_text)
    m = RE_HEADER.search(txt)
    if not m:
        return None

    m1 = m.group("m1")
    m2 = m.group("m2")
    d1 = m.group("d1")
    d2 = m.group("d2")
    y = m.group("y")
    try:
        d1 = int(d1) if d1 else None
        d2 = int(d2) if d2 else None
    except Exception:
        d1, d2 = None, None

    if m2 and d2:
        mon = MONTHS.get(m2.lower())
        day = d2
    else:
        mon = MONTHS.get((m2 or m1).lower()) if (m2 and d2) else MONTHS.get(m1.lower())
        day = d2 or d1

    if y is None and fallback_year is not None:
        year = fallback_year
    elif y is None:
        y2 = re.search(r"(\d{4})", txt)
        year = int(y2.group(1)) if y2 else None
    else:
        year = int(y)

    if not (mon and day and year):
        return None
    try:
        return pd.Timestamp(year=year, month=mon, day=day)
    except Exception:
        return None


def contains_any(s: str, patterns: List[str]) -> bool:
    for pat in patterns:
        if "*" in pat:
            rx = re.compile(pat.replace("*", ".*"), re.I)
            if rx.search(s):
                return True
        if pat in s:
            return True
    return False


def classify_doc_type(anchor_label: str, url: str = "") -> Tuple[str, str]:
    al = norm(anchor_label).lower()
    path = urlparse(url).path.lower() if url else ""

    for canon, variants in LABEL_MAP.items():
        if contains_any(al, [v.lower() for v in variants]):
            return canon, canon
    if "implementation note" in al:
        return "Implementation Note", "Implementation Note"

    if PRESS_DIR_PAT.search(path):
        return "Statement", "Statement"
    if "minutes" in path:
        return "Minutes", "Minutes"
    return "Unknown", "Unknown"


@dataclass
class DocRow:
    year: int
    date: Optional[pd.Timestamp]
    doc_type_raw: str
    doc_type: str
    title_text: str
    anchor_label: str
    url: str
    filetype: str
    year_page: str


def get_year_index_pages(start_url: str = HISTORICAL_INDEX, pause: float = 0.0) -> List[Tuple[int, str]]:
    soup = get_soup(start_url)
    out = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "fomchistorical" in href.lower() and href.lower().endswith(".htm"):
            url = absolute_url(start_url, href)
            year_txt = norm(a.get_text()) or ""
            m = re.search(r"(\d{4})", year_txt)
            if m:
                out.append((int(m.group(1)), url))
    out = sorted(list({(y, u) for (y, u) in out}), key=lambda x: x[0])
    if pause:
        time.sleep(pause)
    return out


def extract_docs_from_year(year: int, year_url: str, pause: float = 0.0, verbose: bool = False) -> pd.DataFrame:
    soup = get_soup(year_url)
    rows: List[DocRow] = []
    for section in soup.select("div, section"):
        header_el = None
        for sel in ["h2", "h3", "h4", ".panel-heading", ".fomc-meeting__date"]:
            header_el = section.select_one(sel)
            if header_el:
                break
        if not header_el:
            continue

        header_text = norm(header_el.get_text(" "))
        mtg_date = parse_meeting_header_date(header_text, fallback_year=year)

        links = section.find_all("a", href=True)
        if not links:
            continue

        for a in links:
            label = norm(a.get_text(" "))
            href = a["href"]
            abs_url = absolute_url(year_url, href)
            if not is_doc_href(abs_url):
                continue
            ftype = filetype_of(abs_url)
            raw, canon = classify_doc_type(label, url=abs_url)

            rows.append(DocRow(
                year=year, date=mtg_date, doc_type_raw=raw, doc_type=canon,
                title_text=label, anchor_label=label, url=abs_url,
                filetype=ftype or "html", year_page=year_url,
            ))

    df = pd.DataFrame([r.__dict__ for r in rows])
    if not df.empty:
        df.loc[df["doc_type_raw"].str.contains("record of policy action", case=False, na=False), "doc_type"] = "Record of Policy Actions"
        df = (df.sort_values("date", na_position="last")
                .drop_duplicates(subset=["url"], keep="first")
                .sort_values(["year", "date", "doc_type", "url"]).reset_index(drop=True))
    if pause:
        time.sleep(pause)
    return df


def scrape_all_fomc_docs(pause: float = 0.4, verbose: bool = False) -> pd.DataFrame:
    idx = get_year_index_pages(HISTORICAL_INDEX, pause=0.0)
    frames = []
    for year, url in idx:
        if verbose:
            print(f"[Historical] {year} -> {url}")
        dfy = extract_docs_from_year(year, url, pause=0.0, verbose=verbose)
        frames.append(dfy)
        if pause:
            time.sleep(pause)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[
        "year", "date", "doc_type_raw", "doc_type", "title_text", "anchor_label", "url", "filetype", "year_page",
    ])
    return df[df["date"].notna()].reset_index(drop=True) if not df.empty else df


def scrape_calendar(cal_url: str = CAL_URL) -> pd.DataFrame:
    soup = get_soup(cal_url)
    rows: List[DocRow] = []
    for sec in soup.select("div.row.fomc-meeting"):
        full_text = norm(sec.get_text(" "))
        panel = sec.find_parent("div", class_="panel")
        year_el = panel.select_one(".panel-heading") if panel else None
        year_txt = norm(year_el.get_text(" ")) if year_el else ""
        ym = re.search(r"(\d{4})", year_txt)
        fallback_year = int(ym.group(1)) if ym else None
        mtg_date = parse_meeting_header_date(full_text[:40], fallback_year=fallback_year)

        links = sec.find_all("a", href=True)
        if not links:
            continue

        for a in links:
            label = norm(a.get_text(" "))
            href = a["href"]
            abs_url = absolute_url(cal_url, href)
            if not is_doc_href(abs_url):
                continue
            ftype = filetype_of(abs_url)
            raw, canon = classify_doc_type(label, url=abs_url)

            rows.append(DocRow(
                year=fallback_year, date=mtg_date, doc_type_raw=raw, doc_type=canon,
                title_text=label, anchor_label=label, url=abs_url,
                filetype=ftype or "html", year_page=cal_url,
            ))

    df = pd.DataFrame([r.__dict__ for r in rows])
    if not df.empty:
        df["source"] = "Recent"
        df = (df[df["date"].notna()]
                .sort_values("date")
                .drop_duplicates(subset=["url"], keep="first")
                .reset_index(drop=True))
    return df


@dataclass
class StatementText:
    url: str
    http_status: Optional[int]
    page_title: str
    text: str
    pub_date_from_url: Optional[str]
    word_count: int
    error: Optional[str]


def fetch_statement_text(url: str, pause: float = 0.0) -> StatementText:
    status = None
    title = ""
    body_text = ""
    pub_date_guess = None
    err = None
    try:
        resp = requests.get(url, headers=HEADERS, timeout=45)
        status = resp.status_code
        if status == 200:
            soup = BeautifulSoup(resp.text, "lxml")
            if soup.title and soup.title.string:
                title = norm(soup.title.string)
            main = None
            for sel in MAIN_SELECTORS:
                main = soup.select_one(sel)
                if main:
                    break
            if not main:
                main = soup.body or soup
            lines = [re.sub(r"[ \t\r]+", " ", ln).strip() for ln in main.get_text("\n").split("\n")]
            body_text = "\n".join(ln for ln in lines if ln)
            m = re.search(r"/(20\d{2}|19\d{2})/([01]\d)/([0-3]\d)/", url)
            if m:
                pub_date_guess = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        else:
            err = f"HTTP {status}"
    except Exception as e:
        err = repr(e)
    if pause:
        time.sleep(pause)
    wc = len(body_text.split()) if body_text else 0
    return StatementText(url, status, title, body_text, pub_date_guess, wc, err)


BOILERPLATE_TOP = [
    r"for immediate release",
    r"release date[:\s]",
    r"press release",
    r"for media inquiries",
    r"implementation note.*$",
    r"for release at.*[ap]\.m\..*$",
]

FOOTER_PATTERNS = [
    r"voting for.*?were[:\s].*",
    r"voting against.*?were[:\s].*",
    r"Home\s*\|.*Accessibility.*Last update:.*",
]


def _strip_first_lines(text: str, n: int = 6) -> str:
    lines = text.splitlines()
    head = "\n".join(lines[:n])
    tail = "\n".join(lines[n:])
    for pat in BOILERPLATE_TOP:
        head = re.sub(pat, "", head, flags=re.I | re.M)
    head = re.sub(r"^\s*share\s*$", "", head, flags=re.I | re.M)
    head = re.sub(r"\n{2,}", "\n", head, flags=re.M)
    return (head + ("\n" if head and tail else "") + tail).strip()


def _strip_footers(text: str) -> str:
    t = text
    for pat in FOOTER_PATTERNS:
        t = re.sub(pat, "", t, flags=re.I | re.S)
    return t


def _fix_unicode(text: str) -> str:
    t = html.unescape(text)
    t = t.replace("â€”", "—").replace("â€“", "–").replace("â€˜", "‘").replace("â€™", "’").replace("â€œ", "“").replace("â€\x9d", "”")
    return t


def clean_fomc_statement(txt: str, min_chars: int = 30, flatten: bool = False) -> str:
    if not txt:
        return ""
    t = txt.replace("<br>", "\n").replace("<br/>", "\n").replace("</p>", "\n")
    t = re.sub(r"\n{2,}", "\n", t)
    t = _fix_unicode(t)
    t = _strip_first_lines(t, n=8)
    t = _strip_footers(t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if len(t) < min_chars:
        return txt.strip()
    if flatten:
        t = re.sub(r"\s+", " ", t).strip()
    return t


def fetch_statement_texts(df_statements_html: pd.DataFrame, pause: float = 0.0) -> pd.DataFrame:
    recs = []
    for _, row in df_statements_html.reset_index(drop=True).iterrows():
        s = fetch_statement_text(row["url"], pause=pause)
        recs.append({
            "url": s.url, "http_status": s.http_status, "page_title": s.page_title,
            "text": s.text, "pub_date_from_url": s.pub_date_from_url,
            "word_count": s.word_count, "error": s.error,
        })
    df_txt = pd.DataFrame(recs)
    return df_statements_html.merge(df_txt, on="url", how="left")


def ensure_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["year", "date", "doc_type_raw", "doc_type", "title_text", "anchor_label", "url", "filetype", "year_page", "source"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def concat_historical_and_calendar(df_hist: pd.DataFrame, df_cal: pd.DataFrame) -> pd.DataFrame:
    df_hist = df_hist.copy()
    df_cal = df_cal.copy()
    df_hist["source"] = "Historical"
    df_cal["source"] = df_cal.get("source", "Recent")
    df_hist2 = ensure_canonical_columns(df_hist)
    df_cal2 = ensure_canonical_columns(df_cal)
    out = pd.concat([df_hist2, df_cal2], ignore_index=True)
    out = out.sort_values(["date", "doc_type", "url"], na_position="last").reset_index(drop=True)
    return out


def filter_statement_html(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df[(df["doc_type"] == "Statement") & (df["filetype"] == "html")].copy()
    df2 = df2.drop_duplicates(subset=["url"]).reset_index(drop=True)
    return df2


DOC_LETTER_RE = re.compile(r"monetary\d{8}([a-z]?)\d*\.htm")


def url_doc_letter(url: str) -> str:
    m = DOC_LETTER_RE.search(url)
    return m.group(1) if m and m.group(1) else "a"
