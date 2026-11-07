"""The app + an `async def` copy of /search, ONLY for the Lab 7 def-vs-async measurement.

INDEX_PATH=index.bin uvicorn lab07_variants:app --app-dir scripts
GET /search        -> def        (FastAPI runs it in the thread pool)
GET /search_async  -> async def  (runs ON the event loop: the CPU work blocks it)
"""

from __future__ import annotations

from fastapi import Request

from findex.web.app import create_app, get_index, run_search
from findex.web.models import SearchResponse

app = create_app()


@app.get("/search_async", response_model=SearchResponse)
async def search_async(request: Request, q: str, k: int = 10) -> SearchResponse:
    return run_search(get_index(request), q, k, "bm25", 1)  # no await inside: blocks the loop
