import os
import re
import time
import io
import subprocess
import threading
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse, urljoin

import requests
import pandas as pd
import pdfplumber
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from playwright.sync_api import sync_playwright

# ============================================================
# Config & FastAPI setup
# ============================================================

load_dotenv()
QUIZ_SECRET = os.getenv("QUIZ_SECRET")

print(f"[startup] QUIZ_SECRET loaded: {repr(QUIZ_SECRET)}")

app = FastAPI()


class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str


# ============================================================
# Playwright runtime-install fallback
# ============================================================

_playwright_install_lock = threading.Lock()
_playwright_installed = False


def ensure_playwright_browsers_installed() -> None:
    """
    Ensure Chromium is installed for Playwright.
    Safe to call multiple times; only installs once per process.
    """
    global _playwright_installed
    if _playwright_installed:
        return

    with _playwright_install_lock:
        if _playwright_installed:
            return

        print("[playwright] Installing Chromium (runtime fallback)...")
        subprocess.run(
            ["python", "-m", "playwright", "install", "chromium"],
            check=True,
        )
        _playwright_installed = True
        print("[playwright] Chromium install complete.")


def load_rendered_page(url: str) -> (str, str):
    """
    Open the given URL in a headless Chromium browser and return
    (rendered_html, final_url_after_redirects).

    If Chromium is not installed, install it once at runtime and retry.
    """
    print(f"[browser] Opening URL in Playwright: {url}")

    def _open() -> (str, str):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle")
            html = page.content()
            final_url = page.url
            browser.close()
        print(f"[browser] Finished loading. Final URL: {final_url}")
        return html, final_url

    try:
        # First attempt
        return _open()
    except Exception as e:
        msg = str(e)
        # Typical error when browser binary is missing or outdated
        if "Executable doesn't exist" in msg or "playwright install" in msg:
            print("[browser] Chromium not found. Running runtime installer...")
            ensure_playwright_browsers_installed()
            # Retry once after installing browsers
            return _open()
        # Different error → bubble up
        raise


# ============================================================
# Parsing & data helpers
# ============================================================

def extract_quiz_details(html: str, page_url: str) -> Dict[str, Any]:
    """
    Extract:
      - raw text of the page
      - submit URL (absolute)
      - file URLs (absolute) with interesting extensions
    """
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")

    # 1) Absolute submit URL in text
    abs_match = re.search(r"https?://\S*submit\S*", text)
    submit_url = abs_match.group(0) if abs_match else None

    # 2) Relative /submit style
    if not submit_url:
        rel_match = re.search(r"\s(/?submit\S*)", text)
        if rel_match:
            rel_path = rel_match.group(1).strip()
            parsed = urlparse(page_url)
            base = f"{parsed.scheme}://{parsed.netloc}"
            submit_url = urljoin(base, rel_path)

    # 3) File URLs in href/src
    exts = [".pdf", ".csv", ".xlsx", ".xls", ".json", ".txt"]
    file_urls: List[str] = []

    for tag in soup.find_all(["a", "audio", "video", "source", "link"]):
        candidate = tag.get("href") or tag.get("src")
        if not candidate:
            continue
        if any(candidate.lower().endswith(ext) for ext in exts):
            abs_url = urljoin(page_url, candidate)
            file_urls.append(abs_url)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in file_urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)

    print(f"[parser] Detected submit URL: {submit_url}")
    print(f"[parser] Detected file URLs: {deduped}")

    return {
        "raw_text": text,
        "submit_url": submit_url,
        "file_urls": deduped,
    }


def download_bytes(url: str) -> bytes:
    print(f"[data] Downloading file: {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def extract_page_number(raw_text: str, default_page: int = 2) -> int:
    """
    Try to find 'page N' in the question text; otherwise use default_page.
    """
    m = re.search(r"page\s+(\d+)", raw_text, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return default_page


def sum_value_column_from_pdf(pdf_bytes: bytes, page_number: int) -> float:
    """
    On the given page in the PDF, find a column named 'value' and sum it.
    """
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        index = page_number - 1
        if index < 0 or index >= len(pdf.pages):
            raise ValueError(f"PDF has only {len(pdf.pages)} pages; asked for page {page_number}")

        page = pdf.pages[index]
        table = page.extract_table()
        if not table:
            raise ValueError("No table found on requested page")

        header = table[0]
        rows = table[1:]

        df = pd.DataFrame(rows, columns=header)

        target_col = None
        for col in df.columns:
            if col and col.strip().lower() == "value":
                target_col = col
                break

        if target_col is None:
            raise ValueError(f"'value' column not found in headers: {df.columns.tolist()}")

        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        return float(df[target_col].sum())


def load_table_df(file_bytes: bytes, ext: str) -> pd.DataFrame:
    """
    Load CSV or Excel into a DataFrame.
    """
    if ext == ".csv":
        return pd.read_csv(io.BytesIO(file_bytes))
    return pd.read_excel(io.BytesIO(file_bytes))


def find_numeric_column(df: pd.DataFrame) -> Optional[str]:
    """
    Prefer a 'value' column; otherwise first numeric column if any.
    """
    # Explicit 'value' column
    for col in df.columns:
        if isinstance(col, str) and col.strip().lower() == "value":
            return col

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        return numeric_cols[0]

    return None


def infer_cutoff_mode(raw_text: str) -> str:
    """
    Infer whether the question wants values ABOVE or BELOW the cutoff.

    Returns:
      - "lt"  → use < cutoff
      - "gte" → use >= cutoff (default)
    """
    t = raw_text.lower()

    below_keywords = ["less than", "below", "under", "strictly less"]
    above_keywords = ["greater than", "above", "at least", ">= ", "more than"]

    if any(k in t for k in below_keywords):
        return "lt"
    if any(k in t for k in above_keywords):
        return "gte"

    # Default guess: sum of values >= cutoff
    return "gte"


# ============================================================
# Core solver logic
# ============================================================

def compute_answer(raw_text: str, page_url: str, file_urls: List[str]) -> Any:
    """
    Given the question text, page URL, and any data file URLs,
    compute the answer object (number/string/etc.).
    """
    print("=== QUIZ RAW TEXT PREVIEW (first 600 chars) ===")
    print(raw_text[:600])
    print("=== END PREVIEW ===")

    text_lower = raw_text.lower()

    # 0) Demo: "anything you want"
    if "anything you want" in text_lower:
        print("[compute_answer] Detected demo quiz → returning 'demo-answer'")
        return "demo-answer"

    # 1) Demo scrape: look for /demo-scrape-data?...
    if "demo-scrape-data" in raw_text:
        print("[compute_answer] Detected demo-scrape task")
        m = re.search(r"(/demo-scrape-data[^\s\"']*)", raw_text)
        if m:
            rel = m.group(1).strip()
            data_url = urljoin(page_url, rel)
            print(f"[compute_answer] Scraping secret from (rendered): {data_url}")

            html2, _ = load_rendered_page(data_url)
            soup2 = BeautifulSoup(html2, "html.parser")
            txt = soup2.get_text(separator="\n").strip()
            print("[compute_answer] demo-scrape rendered text:", txt[:300])

            # Pattern: "Secret code is 62749 ..."
            m2 = re.search(r"secret code\s+is\s+([0-9A-Za-z_\-]+)", txt, flags=re.IGNORECASE)
            if m2:
                secret = m2.group(1)
                print(f"[compute_answer] Parsed secret (secret code is ...): {secret}")
                return secret

            # Fallback: first number in the text
            m3 = re.search(r"(\d+)", txt)
            if m3:
                secret = m3.group(1)
                print(f"[compute_answer] Parsed first numeric secret: {secret}")
                return secret

            print("[compute_answer] Could not parse secret from rendered page, returning text.")
            return txt or "missing-demo-scrape-secret"

        print("[compute_answer] Could not find demo-scrape-data URL, returning fallback.")
        return "missing-demo-scrape-secret"

    # 2) Cutoff-based numeric question (e.g. demo-audio)
    cutoff_match = re.search(r"Cutoff:\s*([\d\.]+)", raw_text, flags=re.IGNORECASE)
    cutoff = float(cutoff_match.group(1)) if cutoff_match else None
    if cutoff is not None:
        print(f"[compute_answer] Detected cutoff: {cutoff}")

    # Collect candidate file URLs (HTML-derived + explicit URLs in text)
    candidate_files = list(file_urls)

    m_file = re.search(
        r"(https?://\S+\.(?:pdf|csv|xlsx?|xls))",
        raw_text,
        flags=re.IGNORECASE,
    )
    if m_file:
        candidate_files.append(m_file.group(1).strip())

    # Deduplicate
    seen = set()
    files: List[str] = []
    for u in candidate_files:
        if u not in seen:
            seen.add(u)
            files.append(u)

    if not files:
        print("[compute_answer] No file URL found. Returning fallback.")
        return "fallback-answer"

    print(f"[compute_answer] Candidate data files: {files}")
    file_url = files[0]
    print(f"[compute_answer] Using file URL: {file_url}")

    ext_match = re.search(r"\.(pdf|csv|xlsx?|xls)\b", file_url, flags=re.IGNORECASE)
    ext = ext_match.group(0).lower() if ext_match else ""

    file_bytes = download_bytes(file_url)

    # PDF case: sum "value" column on given page
    if ext == ".pdf":
        page_num = extract_page_number(raw_text, default_page=2)
        print(f"[compute_answer] PDF detected; using page {page_num}")
        s = sum_value_column_from_pdf(file_bytes, page_num)
        return int(s) if float(s).is_integer() else float(s)

    # CSV / Excel case: numeric aggregation with possible cutoff
    if ext in (".csv", ".xls", ".xlsx"):
        print(f"[compute_answer] Table file detected ({ext})")
        df = load_table_df(file_bytes, ext)

        col = find_numeric_column(df)
        if col is None:
            print("[compute_answer] No numeric/value column found, returning fallback.")
            return "no-value-column"

        print(f"[compute_answer] Using numeric column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

        if cutoff is not None:
            mode = infer_cutoff_mode(raw_text)
            if mode == "lt":
                print(f"[compute_answer] Applying cutoff < {cutoff} on column {col}")
                s = df.loc[df[col] < cutoff, col].sum()
            else:
                print(f"[compute_answer] Applying cutoff >= {cutoff} on column {col}")
                s = df.loc[df[col] >= cutoff, col].sum()
        else:
            s = df[col].sum()

        s = float(s)
        return int(s) if s.is_integer() else s

    print("[compute_answer] Unhandled file type. Returning fallback.")
    return "unhandled-file-type"


def solve_quiz(email: str, secret: str, url: str, start_time: float) -> None:
    """
    Synchronous solver loop:
      - Load quiz URL
      - Parse question & submit URL
      - Compute answer
      - Submit
      - Follow chained URLs
      - Stop when correct & no new URL, or after 3 minutes
    """
    print("\n[solver] ===============================")
    print(f"[solver] Starting solve_quiz for URL: {url}")
    print(f"[solver] Email: {email}")
    print("[solver] --------------------------------")

    while True:
        elapsed = time.time() - start_time
        print(f"[solver] Elapsed time: {elapsed:.1f}s")
        if elapsed > 180:
            print("[solver] ⏰ Time limit exceeded. Stopping.")
            return

        # 1) Load page
        html, final_url = load_rendered_page(url)

        # 2) Parse details
        quiz = extract_quiz_details(html, final_url)
        submit_url = quiz.get("submit_url")
        if not submit_url:
            print("[solver] ❌ No submit URL found. Stopping.")
            return

        # 3) Compute answer
        answer = compute_answer(
            quiz["raw_text"],
            final_url,
            quiz.get("file_urls", []),
        )
        print(f"[solver] Computed answer: {answer}")

        # 4) Submit answer
        payload = {
            "email": email,
            "secret": secret,
            "url": url,
            "answer": answer,
        }
        print(f"[solver] Submitting to {submit_url} with payload: {payload}")

        try:
            resp = requests.post(submit_url, json=payload, timeout=60)
            resp.raise_for_status()
        except Exception as e:
            print(f"[solver] ❌ Error submitting answer: {e}")
            return

        try:
            result = resp.json()
        except Exception as e:
            print(f"[solver] ❌ Could not parse JSON: {e}")
            print("[solver] Raw response:", resp.text[:500])
            return

        print("[solver] ✅ Submission response:", result)

        # 5) Handle chaining
        if result.get("correct") and not result.get("url"):
            print("[solver] 🎉 Quiz completed, no further URLs.")
            return

        if result.get("url"):
            url = result["url"]
            print(f"[solver] ➡️ Moving to next quiz URL: {url}")
            continue

        print("[solver] ℹ️ No 'url' in response, stopping.")
        return


# ============================================================
# FastAPI endpoint
# ============================================================

@app.post("/")
def receive_quiz(req: QuizRequest):
    print("\n[endpoint] Received POST /")
    print(f"[endpoint] email={req.email}, url={req.url}")

    if QUIZ_SECRET is None:
        print("[endpoint] ERROR: QUIZ_SECRET is not set")
        raise HTTPException(status_code=500, detail="Server secret not configured")

    if req.secret != QUIZ_SECRET:
        print("[endpoint] Invalid secret provided")
        raise HTTPException(status_code=403, detail="Invalid secret")

    start_time = time.time()
    solve_quiz(req.email, req.secret, req.url, start_time)

    # Per spec: respond 200 on valid secret; quiz solver runs synchronously here.
    return {"status": "accepted", "message": "Quiz solving completed (sync)"}
