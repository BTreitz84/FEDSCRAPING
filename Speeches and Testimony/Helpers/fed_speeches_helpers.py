from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import pdfplumber
import requests
from dateutil import parser as dateparse

API_KEY = os.environ["FRASER_API_KEY"]
BASE = "https://fraser.stlouisfed.org/api"
RPM = 60
SLEEP = 60.0 / RPM

CHAIR_IDS = {
    "Alan_Greenspan": 452,
    "Janet_L._Yellen": 930,
    "bernanke": 453,
    "Paul_A._Volcker": 451,
    "Jerome_H._Powell": 1164,
    "Roger_Walton_Ferguson,_Jr": 950,
    "Lael_Brainard": 3777,
    "Donald_L._Kohn": 464,
    "Preston_Martin": 947,
    "Stanley_Fischer": 3778,
    "Manuel_H._Johnson": 940,
    "Richard_H._Clarida": 5997,
    "Alice_M._Rivlin": 907,
    "Frederick_Henry_Schultz": 920,
    "Philip_N._Jefferson": 6860,
    "David_W._Mullins": 912,
    "Alan_S._Blinder": 906,
    "Paul_M._Warburg": 454,
}

DOCTYPE_MAP = {
    "transcript": "transcript",
    "press conference": "press_conference",
    "testimon": "testimony",
    "hearing": "testimony",
    "lecture": "lecture",
    "remarks": "speech",
    "address": "speech",
    "statement": "speech",
    "speech": "speech",
}

ID_RE = re.compile(r"/(?:api/)?(title|item)/(\d+)", re.I)


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "X-API-Key": API_KEY,
        "Accept": "application/json",
        "User-Agent": "curl/8.4.0",
    })
    return s


def rl():
    time.sleep(SLEEP + 0.1)


def get_json(session: requests.Session, url: str, params: Optional[dict] = None, max_retries: int = 6) -> dict:
    tries = 0
    while True:
        tries += 1
        try:
            r = session.get(url, params=params or {}, timeout=60)
        except requests.exceptions.RequestException:
            if tries >= max_retries:
                raise
            time.sleep(min(60, 2 ** tries))
            continue
        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            wait = float(ra) if ra else min(90, 2 ** tries)
            time.sleep(wait)
            continue
        if r.status_code in (500, 502, 503, 504):
            if tries >= max_retries:
                r.raise_for_status()
            time.sleep(min(60, 2 ** tries))
            continue
        r.raise_for_status()
        return r.json()


def record_type_ids(rec) -> list[tuple[str, int]]:
    hits = []

    def scan(x):
        if isinstance(x, dict):
            for v in x.values():
                scan(v)
        elif isinstance(x, list):
            for v in x:
                scan(v)
        elif isinstance(x, str):
            for m in ID_RE.finditer(x):
                hits.append((m.group(1).lower(), int(m.group(2))))

    scan(rec)
    seen = set()
    out = []
    for t, i in hits:
        if (t, i) not in seen:
            out.append((t, i))
            seen.add((t, i))
    return out


def list_title_records(session: requests.Session, title_id: int, limit: int = 200) -> list[dict]:
    page, out = 1, []
    while True:
        j = get_json(session, f"{BASE}/title/{title_id}/items", params={"page": page, "limit": limit})
        recs = j.get("records", [])
        if not recs:
            break
        out.extend(recs)
        total = int(j.get("total", 0))
        start = int(j.get("start", 0))
        if start + len(recs) >= total:
            break
        page += 1
        rl()
    return out


def discover_item_ids(session: requests.Session, title_id: int) -> set[int]:
    item_ids = set()
    sub_titles = []
    for rec in list_title_records(session, title_id):
        for typ, rid in record_type_ids(rec):
            if typ == "item":
                item_ids.add(rid)
            elif typ == "title":
                sub_titles.append(rid)
    for tid in sub_titles:
        for rec in list_title_records(session, tid):
            for typ, rid in record_type_ids(rec):
                if typ == "item":
                    item_ids.add(rid)
        rl()
    return item_ids


def first(x, default="NA"):
    if x is None:
        return default
    if isinstance(x, list):
        return first(x[0], default) if x else default
    return x


def safe_get(rec, path: str, default="NA"):
    cur = rec
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, list):
            cur = cur[0] if cur else None
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return default
    return first(cur, default)


def extract_pdfs(obj) -> list[str]:
    pdfs = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str) and x.lower().endswith(".pdf"):
            pdfs.append(x)

    walk(obj)
    out, seen = [], set()
    for u in pdfs:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def guess_date_iso(rec: dict) -> str:
    sort_date = (rec.get("originInfo") or {}).get("sortDate")
    if sort_date and re.match(r"^\d{4}-\d{2}-\d{2}$", str(sort_date)):
        return str(sort_date)
    txt = json.dumps(rec, ensure_ascii=False)
    m = re.search(r"\b\d{4}-\d{2}-\d{2}\b", txt) or re.search(r"\b\d{4}\b", txt)
    return m.group(0) if m else ""


def infer_doc_type(title: str, genre) -> str:
    t = (title or "").lower()
    for key, label in DOCTYPE_MAP.items():
        if key in t:
            return label
    if isinstance(genre, list) and genre:
        return str(genre[0]).lower()
    if isinstance(genre, str) and genre:
        return genre.lower()
    return "speech"


def extract_speaker(rec: dict, fallback_label: str) -> str:
    rel = rec.get("relatedItem", [])
    if isinstance(rel, list):
        for entry in rel:
            if isinstance(entry, dict) and entry.get("@type") == "parent":
                for nm in entry.get("name", []) or []:
                    if isinstance(nm, dict) and str(nm.get("role", "")).lower() == "creator":
                        np = nm.get("namePart")
                        if isinstance(np, list):
                            parts = [p for p in np if isinstance(p, str) and p.strip()]
                            if parts:
                                return parts[0]
                        elif isinstance(np, str) and np.strip():
                            return np
    for nm in rec.get("name", []) or []:
        if isinstance(nm, dict) and str(nm.get("role", "")).lower() == "creator":
            np = nm.get("namePart")
            if isinstance(np, list):
                parts = [p for p in np if isinstance(p, str) and p.strip()]
                if parts:
                    return parts[0]
            elif isinstance(np, str) and np.strip():
                return np
    return fallback_label.replace("_", " ").replace(" ,", ",")


def fetch_item_record(session: requests.Session, item_id: int) -> dict:
    j = get_json(session, f"{BASE}/item/{item_id}")
    return j["records"][0] if isinstance(j, dict) and "records" in j and j["records"] else j


def download_file(session: requests.Session, url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = session.get(url, timeout=90, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 15):
            if chunk:
                f.write(chunk)


def scrape_speaker(speaker_label: str, title_id: int, downloads_dir: Path, force_redownload: bool = False) -> pd.DataFrame:
    session = make_session()
    item_ids = discover_item_ids(session, title_id)
    rows = []
    dl_ok = dl_skip = dl_fail = 0

    for iid in sorted(item_ids):
        rec = fetch_item_record(session, iid)
        rl()
        title = first(safe_get(rec, "titleInfo.title") or rec.get("title") or rec.get("name") or f"Item {iid}")
        date_iso = guess_date_iso(rec)
        year = date_iso[:4] if date_iso else "0000"
        pdf_urls = extract_pdfs(rec)
        genre = safe_get(rec, "genre", default=None)
        speaker = extract_speaker(rec, speaker_label)
        doc_type = infer_doc_type(title, genre)

        year_dir = downloads_dir / speaker_label / year
        for j, u in enumerate(pdf_urls, 1):
            dest = year_dir / f"Item {iid}_{j}.pdf"
            if dest.exists() and not force_redownload:
                dl_skip += 1
            else:
                try:
                    download_file(session, u, dest)
                    rl()
                    dl_ok += 1
                except Exception:
                    dl_fail += 1

            rows.append({
                "item_id": iid, "chair_label": speaker_label, "speaker": speaker,
                "title": title, "date_iso": date_iso, "year": year,
                "doc_type": doc_type, "pdf_url": u, "path": str(dest),
            })

    manifest = pd.DataFrame(rows)
    manifest.attrs["dl_ok"] = dl_ok
    manifest.attrs["dl_skip"] = dl_skip
    manifest.attrs["dl_fail"] = dl_fail
    return manifest


def extract_pdf_text(path: str) -> tuple[str, int, Optional[str]]:
    try:
        with pdfplumber.open(path) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        text = "\n".join(pages)
        return text, len(pdf.pages), None
    except Exception as e:
        return "", 0, str(e)


PART_RE = re.compile(r"Item\s*(\d+)_(\d+)\.pdf$", re.I)


def compile_manifest_texts(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in manifest.iterrows():
        text, n_pages, err = extract_pdf_text(r["path"])
        needs_ocr = err is None and n_pages > 0 and len(text.strip()) == 0
        rows.append({
            "item_id": r["item_id"], "chair_label": r["chair_label"], "speaker": r["speaker"],
            "title": r["title"], "date_iso": r["date_iso"], "year": r["year"],
            "doc_type": r["doc_type"], "path": r["path"], "pdf_url": r["pdf_url"],
            "raw_text": text, "n_pages": n_pages, "error": err, "needs_ocr": needs_ocr,
        })
    return pd.DataFrame(rows)


def part_num(path: str) -> int:
    m = PART_RE.search(str(path))
    return int(m.group(2)) if m else 1


FRASER_WATERMARK_RE = re.compile(
    r"Digitized for FRASER\s*\n\s*https?://fraser\.stlouisfed\.org/?\s*\n\s*Federal Reserve Bank of St\.?\s*Louis\s*\n?",
    re.I,
)


def clean_speech_text(text: str) -> str:
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    joined = "\n".join(lines)
    joined = FRASER_WATERMARK_RE.sub("", joined)
    return joined.strip()


def build_speeches_clean(compiled: pd.DataFrame) -> pd.DataFrame:
    df = compiled.copy()
    df["part_num"] = df["path"].map(part_num)
    df = df.sort_values(["item_id", "part_num"])

    grouped = df.groupby("item_id").agg(
        chair_label=("chair_label", "first"), speaker=("speaker", "first"),
        title=("title", "first"), date_iso=("date_iso", "first"), year=("year", "first"),
        doc_type=("doc_type", "first"), path=("path", lambda s: "; ".join(s)),
        pdf_url=("pdf_url", lambda s: "; ".join(s)),
        raw_text=("raw_text", lambda s: "\n".join(e for e in s if pd.notna(e))),
        n_pages=("n_pages", "sum"), needs_ocr=("needs_ocr", "any"),
        error=("error", lambda s: "; ".join(e for e in s if pd.notna(e)) or None),
    ).reset_index()

    grouped["clean_text"] = grouped["raw_text"].map(clean_speech_text)
    grouped["word_count"] = grouped["clean_text"].str.split().str.len()
    grouped["date"] = pd.to_datetime(grouped["date_iso"], errors="coerce")
    grouped["year"] = grouped["date"].dt.year

    has_text = grouped["clean_text"].str.len() > 0
    deduped_text = grouped[has_text].drop_duplicates(subset=["clean_text"], keep="first")
    grouped = pd.concat([deduped_text, grouped[~has_text]], ignore_index=True)

    tidy = pd.DataFrame({
        "date": grouped["date"], "year": grouped["year"], "speaker": grouped["speaker"],
        "doc_type": grouped["doc_type"], "title": grouped["title"],
        "source_corpus": "fed_speeches", "path": grouped["path"], "url": grouped["pdf_url"],
        "word_count": grouped["word_count"], "clean_text": grouped["clean_text"],
        "raw_text": grouped["raw_text"], "needs_ocr": grouped["needs_ocr"],
        "item_id": grouped["item_id"], "chair_label": grouped["chair_label"],
    })
    return tidy.sort_values(["date", "item_id"]).reset_index(drop=True)
