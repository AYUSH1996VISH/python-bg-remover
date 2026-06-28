@echo off
REM ─────────────────────────────────────────────────
REM  BG Remover Pro  –  Start Server (Windows)
REM ─────────────────────────────────────────────────

REM Optional: set your API keys (comma-separated)
REM set BG_API_KEYS=mykey1,mykey2

REM Optional: change default quality  (fast|balanced|premium|ultra|portrait)
REM set BG_QUALITY=premium

REM Optional: max parallel heavy jobs (default 3)
REM set BG_MAX_CONCURRENT=3

echo Starting BG Remover API v5...
echo Open http://127.0.0.1:8000/docs for interactive API docs
echo Open frontend\index.html in your browser for the web UI
echo.

python bg_remover_api_v5.py api --host 0.0.0.0 --port 8000
pause
