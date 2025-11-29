# main.py
import os
import re
import time
import io
import subprocess
import threading
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse, urljoin

import requests
import pandas as pd
import pdfplumber
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Use Playwright sync API with runtime-install fallback
from playwright.sync_api import sync_playwright

# -------- Config & FastAPI --------
load_dotenv()
QUIZ_SECRET = os.getenv("QUIZ_SECRET")

print(f"[startup] QUIZ_SECRET loaded: {repr(QUIZ_SECRET)}")

app = FastAPI()


class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str


# -------- Playwright runtime-install helpers --------
_playwright_install_lock = threading.Lock()
_playwright_installed = False


def ensure_playwright_browsers_installed() -> None:
    """
    Install Chromium if not already present. Safe to call multiple times.
    """
    global _playwright_installed
    if _playwright_installed:
        return

    with _playwright_install_lock:
        if _playwright_installed:
            return

        print("[playwright] Installing Chromium (runtime fallback)...")
        # Use subprocess to call playwright install
        try:
            subprocess.run(
                ["python", "-m", "playwright", "install", "chromium"],
                check=True,
            )
            _playwright_installed = True
            print("[playwright] Chromium install complete.")
        except Exception as e:
            print("[playwright] Runtime install failed:", e)
            raise


def load_rendered_page(url: str) -> Tuple[str, str]:
    """
    Return (rendered_html, final_url) for the given URL.
    If browsers are missing, try to install them at runtime and retry once.
    """
    print(f"[browser] Opening URL in Playwright: {url}")

    def _open() -> Tuple[str, str]:
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
        return _open()
    except Exception as e:
        msg = str(e)
        # Typical missing-browser error -> attempt runtime install then retry
        if "Executable doesn't exist" in msg or "playwright install" in msg or "Chromium not found" in msg:
            print("[browser] Chromium not found or error detected. Running runtime installer...")
            ensure_playwright_browsers_installed()
            # retry
            return _open()
        raise


# -------- Audio transcription stub (AIPipe) --------
def transcribe_with_aipipe(audio_bytes: bytes) -> str:
    """
    Placeholder for audio transcription integration (AIPipe/Whisper).
    Replace with actual API call if you have keys.
    """
    print("[audio] transcribe_with_aipipe: stub used.")
    return "transcription placeholder"


def is_audio_url(url: str) -> bool:
    url_l = url.lower()
    return url_l.endswith(".mp3") or url_l.endswith(".wav") or url_l.endswith(".ogg") or url_l.endswith(".m4a")


# -------- Parsing helpers (robust) --------
def extract_quiz_details(html: str, page_url: str) -> Dict[str, Any]:
    """
    Aggressively extract:
      - raw_text (rendered)
      - submit_url (absolute) — many heuristics
      - file_urls (absolute)
    """
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")

    submit_url: Optional[str] = None

    # 1) Check HTML for absolute /submit links
    m = re.search(r"(https?://[^\s'\"<>]+/submit[^\s'\"<>]*)", html, flags=re.IGNORECASE)
    if m:
        submit_url = m.group(1).strip()

    # 2) form[action]
    if not submit_url:
        form = soup.find("form", action=True)
        if form:
            submit_url = urljoin(page_url, form["action"].strip())

    # 3) buttons/links with href/data-* pointing to submit
    if not submit_url:
        for tag in soup.find_all(["a", "button", "input"]):
            candidate = (tag.get("href") or tag.get("data-href") or tag.get("data-url")
                         or tag.get("data-target") or tag.get("onclick") or "")
            if candidate and "/submit" in str(candidate).lower():
                # extract the /...submit... piece
                m2 = re.search(r"(/?[^'\"\s>]*submit[^'\"\s>]*)", str(candidate), flags=re.IGNORECASE)
                if m2:
                    submit_url = urljoin(page_url, m2.group(1).strip())
                    break

    # 4) <pre> blocks (often instruction JSON)
    if not submit_url:
        for pre in soup.find_all("pre"):
            txt = pre.get_text()
            m = re.search(r"(https?://[^\s'\"<>]+/submit[^\s'\"<>]*)", txt, flags=re.IGNORECASE)
            if m:
                submit_url = m.group(1).strip()
                break
            m2 = re.search(r'"\s*(/submit[^"\s]*)\s*"', txt, flags=re.IGNORECASE)
            if m2:
                submit_url = urljoin(page_url, m2.group(1).strip())
                break
            m3 = re.search(r"post\s+(?:to\s+)?(\/?submit[^\s\.,>]*)", txt, flags=re.IGNORECASE)
            if m3:
                submit_url = urljoin(page_url, m3.group(1).strip())
                break

    # 5) scan <script> blocks for fetch/XHR
    if not submit_url:
        for script in soup.find_all("script"):
            stext = script.string or script.get_text() or ""
            m = re.search(r"fetch\(\s*['\"]([^'\"]*?/submit[^'\"]*)['\"]", stext, flags=re.IGNORECASE)
            if m:
                submit_url = urljoin(page_url, m.group(1).strip())
                break
            m2 = re.search(r'open\(\s*[\'"]post[\'"]\s*,\s*[\'"]([^\'"]*?/submit[^\'"]*)[\'"]', stext, flags=re.IGNORECASE)
            if m2:
                submit_url = urljoin(page_url, m2.group(1).strip())
                break

    # 6) plain text search for "POST to https://.../submit"
    if not submit_url:
        m = re.search(r"(?:POST|post)\s*(?:this\s+JSON\s+to|to)\s*[:\-]?\s*(https?://[^\s'\"<>]+/submit[^\s'\"<>]*)", text, flags=re.IGNORECASE)
        if m:
            submit_url = m.group(1).strip()

    # 7) fallback: relative /submit in visible text
    if not submit_url:
        m = re.search(r"(\/[^\s'\"<>]*submit[^\s'\"<>]*)", text, flags=re.IGNORECASE)
        if m:
            submit_url = urljoin(page_url, m.group(1).strip())

    # file URLs discovery
    exts = [".pdf", ".csv", ".xlsx", ".xls", ".json", ".txt", ".mp3", ".wav", ".ogg", ".m4a"]
    file_urls: List[str] = []

    for tag in soup.find_all(["a", "audio", "video", "source", "link"]):
        candidate = tag.get("href") or tag.get("src")
        if not candidate:
            continue
        for ext in exts:
            if candidate.lower().endswith(ext):
                file_urls.append(urljoin(page_url, candidate))
                break

    # also scan raw HTML for full URLs to files
    for ext in exts:
        pattern = r"(https?://[^\s'\"<>]+%s)" % re.escape(ext)
        for m in re.finditer(pattern, html, flags=re.IGNORECASE):
            url_c = m.group(1).strip().strip('"')
            if url_c not in file_urls:
                file_urls.append(url_c)

    # dedupe files preserving order
    seen = set()
    deduped_files = []
    for u in file_urls:
        if u not in seen:
            seen.add(u)
            deduped_files.append(u)

    print(f"[parser] Detected submit URL: {submit_url}")
    print(f"[parser] Detected file URLs: {deduped_files}")

    return {
        "raw_text": text,
        "submit_url": submit_url,
        "file_urls": deduped_files,
    }


# -------- Data helpers --------
def download_bytes(url: str) -> bytes:
    print(f"[data] Downloading file: {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def extract_page_number(raw_text: str, default_page: int = 2) -> int:
    m = re.search(r"page\s+(\d+)", raw_text, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return default_page


def sum_value_column_from_pdf(pdf_bytes: bytes, page_number: int) -> float:
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
            if col and isinstance(col, str) and col.strip().lower() == "value":
                target_col = col
                break
        if target_col is None:
            raise ValueError(f"'value' column not found in headers: {df.columns.tolist()}")
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        return float(df[target_col].sum())


def load_table_df(file_bytes: bytes, ext: str) -> pd.DataFrame:
    if ext == ".csv":
        # read via pandas from bytes
        try:
            return pd.read_csv(io.BytesIO(file_bytes))
        except Exception:
            # fallback: try without header
            return pd.read_csv(io.BytesIO(file_bytes), header=None)
    else:
        return pd.read_excel(io.BytesIO(file_bytes))


def find_numeric_column(df: pd.DataFrame) -> Optional[str]:
    # Prefer 'value' column
    for col in df.columns:
        if isinstance(col, str) and col.strip().lower() == "value":
            return col
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        return numeric_cols[0]
    # fallback: coerce and pick column with most numeric values
    best_col = None
    best_nonnull = -1
    for c in df.columns:
        coerced = pd.to_numeric(df[c], errors="coerce")
        nonnull = int(coerced.notna().sum())
        if nonnull > best_nonnull:
            best_nonnull = nonnull
            best_col = c
    if best_nonnull > 0:
        return best_col
    return None


def infer_cutoff_mode(raw_text: str) -> str:
    t = raw_text.lower()
    below_keywords = ["less than", "below", "under", "strictly less"]
    above_keywords = ["greater than", "above", "at least", ">=", "more than"]
    if any(k in t for k in below_keywords):
        return "lt"
    if any(k in t for k in above_keywords):
        return "gte"
    return "gte"


# -------- Core compute logic --------
def compute_answer(raw_text: str, page_url: str, file_urls: List[str]) -> Any:
    print("=== QUIZ RAW TEXT PREVIEW (first 600 chars) ===")
    print(raw_text[:600])
    print("=== END PREVIEW ===")

    text_lower = raw_text.lower()

    # Demo simple case
    if "anything you want" in text_lower:
        print("[compute_answer] Detected demo quiz → returning 'demo-answer'")
        return "demo-answer"

    # demo-scrape like tasks
    if "demo-scrape-data" in raw_text or "demo-scrape" in raw_text:
        print("[compute_answer] Detected demo-scrape task")
        m = re.search(r"(/demo-scrape-data[^\s\"']*)", raw_text)
        if m:
            data_url = urljoin(page_url, m.group(1).strip())
            print(f"[compute_answer] Scraping secret from (rendered): {data_url}")
            html2, _ = load_rendered_page(data_url)
            soup2 = BeautifulSoup(html2, "html.parser")
            txt = soup2.get_text(separator="\n").strip()
            m2 = re.search(r"secret code\s+is\s+([0-9A-Za-z_\-]+)", txt, flags=re.IGNORECASE)
            if m2:
                return m2.group(1)
            m3 = re.search(r"(\d+)", txt)
            if m3:
                return m3.group(1)
            return txt or "missing-demo-scrape-secret"

    # audio tasks
    audio_candidates = [u for u in file_urls if is_audio_url(u)]
    if audio_candidates and ("audio" in text_lower or "listen" in text_lower or "transcribe" in text_lower):
        audio_url = audio_candidates[0]
        audio_bytes = download_bytes(audio_url)
        transcript = transcribe_with_aipipe(audio_bytes)
        m_ans = re.search(r"answer\s+is\s+([0-9A-Za-z_\-]+)", transcript, flags=re.IGNORECASE)
        if m_ans:
            extracted = m_ans.group(1)
            if re.fullmatch(r"\d+", extracted):
                return int(extracted)
            return extracted
        return transcript

    # cutoff numeric tasks (CSV/Excel/PDF)
    cutoff_match = re.search(r"Cutoff:\s*([\d\.]+)", raw_text, flags=re.IGNORECASE)
    cutoff = float(cutoff_match.group(1)) if cutoff_match else None
    if cutoff is not None:
        print(f"[compute_answer] Detected cutoff: {cutoff}")

    # collect candidate data files (from file_urls and inline links)
    candidate_files = list(file_urls)
    m_file = re.search(r"(https?://\S+\.(?:pdf|csv|xlsx?|xls))", raw_text, flags=re.IGNORECASE)
    if m_file:
        candidate_files.append(m_file.group(1).strip())

    # dedupe
    seen = set()
    files = []
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

    # PDF with 'value' column on a page
    if ext == ".pdf":
        page_num = extract_page_number(raw_text, default_page=2)
        print(f"[compute_answer] PDF detected; using page {page_num}")
        s = sum_value_column_from_pdf(file_bytes, page_num)
        return int(s) if float(s).is_integer() else float(s)

    # CSV / Excel handling (robust numeric detection)
    if ext in (".csv", ".xls", ".xlsx"):
        print(f"[compute_answer] Table file detected ({ext})")
        df = load_table_df(file_bytes, ext)

        # if no header (pandas created numeric headers), try to re-read with header=0
        if all(isinstance(c, int) for c in df.columns):
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), header=0)
            except Exception:
                pass

        value_col = None
        for c in df.columns:
            if isinstance(c, str) and c.strip().lower() == "value":
                value_col = c
                break

        if value_col is None:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if len(numeric_cols) == 0:
                # try coercion heuristic
                best_col = None
                best_nonnull = -1
                for c in df.columns:
                    coerced = pd.to_numeric(df[c], errors="coerce")
                    nonnull = int(coerced.notna().sum())
                    if nonnull > best_nonnull:
                        best_nonnull = nonnull
                        best_col = c
                if best_col is None or best_nonnull <= 0:
                    print("[compute_answer] No numeric column found, returning fallback.")
                    return "no-value-column"
                value_col = best_col
            else:
                value_col = numeric_cols[0]

        print(f"[compute_answer] Using numeric column: {value_col}")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

        if cutoff is not None:
            mode = infer_cutoff_mode(raw_text)
            if mode == "lt":
                print(f"[compute_answer] Applying cutoff < {cutoff} on column {value_col}")
                s = df.loc[df[value_col] < cutoff, value_col].sum()
            else:
                print(f"[compute_answer] Applying cutoff >= {cutoff} on column {value_col}")
                s = df.loc[df[value_col] >= cutoff, value_col].sum()
        else:
            s = df[value_col].sum()

        s = float(s)
        return int(s) if s.is_integer() else s

    print("[compute_answer] Unhandled file type. Returning fallback.")
    return "unhandled-file-type"


# -------- Solver loop --------
def solve_quiz(email: str, secret: str, url: str, start_time: float) -> None:
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

        # 1) Load page (rendered)
        try:
            html, final_url = load_rendered_page(url)
        except Exception as e:
            print("[solver] Error loading page:", e)
            return

        # debug dump the HTML for problematic pages (writes to /tmp on Linux/Render)
        try:
            fname = f"/tmp/last_page_{int(time.time())}.html"
            with open(fname, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"[debug] Wrote last page HTML to {fname}")
        except Exception:
            pass

        # 2) Parse
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

        if result.get("correct") and not result.get("url"):
            print("[solver] 🎉 Quiz completed, no further URLs.")
            return

        if result.get("url"):
            url = result["url"]
            print(f"[solver] ➡️ Moving to next quiz URL: {url}")
            continue

        print("[solver] ℹ️ No 'url' in response, stopping.")
        return


# -------- FastAPI endpoint --------
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
    # Run synchronously (this keeps code simpler and aligns with earlier behavior)
    solve_quiz(req.email, req.secret, req.url, start_time)

    return {"status": "accepted", "message": "Quiz solving completed (sync)"}
