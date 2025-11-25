# LLM Analysis Quiz – IITM BS Project

This project implements an HTTP endpoint that can automatically solve a sequence of data-related quiz tasks using:

- FastAPI (for the API endpoint)
- Playwright (for JavaScript-rendered web pages)
- Pandas / pdfplumber (for data extraction and analysis)
- Requests + BeautifulSoup (for HTTP + parsing)

It is designed for the **LLM Analysis Quiz** project in the IITM BS Data Science programme.

---

## 🚀 Features

- **Secret-verified API endpoint**  
  - Accepts `POST /` with JSON body:
    ```json
    {
      "email": "you@example.com",
      "secret": "your-secret",
      "url": "https://example.com/quiz-123"
    }
    ```
  - Valid secret → HTTP 200 with JSON.  
  - Invalid secret → HTTP 403.

- **Headless browser with Playwright**  
  - Renders JavaScript-heavy quiz pages.
  - Follows redirects and dynamic content (e.g., script-injected HTML).

- **Quiz solving pipeline**
  - Loads the quiz page and extracts:
    - Raw text
    - Submit URL (absolute or relative `/submit`)
    - Data file URLs (.csv, .xlsx, .xls, .pdf, etc.)
  - Interprets the instructions in the page and computes an answer.
  - Submits the answer as JSON to the submit URL.
  - Follows chained quiz URLs from the server’s JSON response.
  - Stops when:
    - There is no further URL, or
    - 3 minutes have passed since the first request.

- **Supported task types (examples)**
  - “Anything you want” demo task.
  - Scraping a separate endpoint for a **secret code** (JS-rendered).
  - Downloading CSV / Excel data and:
    - Selecting a numeric / `value` column,
    - Applying a cutoff condition (e.g. values below a threshold),
    - Computing sums.
  - Downloading PDF files and summing the `"value"` column on a specific page.

---

## 🗂 Project structure

```text
llm-analysis-quiz/
├── main.py            # FastAPI app + solver logic
├── requirements.txt   # Python dependencies
├── .env.example       # Example environment variables (no secrets)
├── README.md
├── LICENSE            # MIT License
└── venv/              # (local virtual environment – not committed)
