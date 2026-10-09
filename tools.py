"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import ssl
import threading
import time
import xml.etree.ElementTree as ET

import httpx  # noqa: F401
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    PSEUDO-CODE:
      for attempt in 0 .. attempts-1:
          try: return fn()
          except RetryableError as e:
              if this was the last attempt: raise
              delay = e.retry_after if the server told us, else exponential backoff base * 2**attempt
              cap the delay at `cap` seconds; add random jitter to the exponential case
              sleep(delay)
    Use it to wrap EVERY network call below. Also treat these as retryable: HTTP 429/500/502/503/504,
    httpx.TransportError (timeouts, connection resets). Read the Retry-After header when present.
    """
    attempts = max(1, int(attempts))
    cap = max(0.0, float(cap))
    for attempt in range(attempts):
        try:
            return fn()
        except httpx.TransportError as exc:
            error = RetryableError(str(exc))
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in {429, 500, 502, 503, 504}:
                raise
            error = RetryableError(str(exc), _retry_after(exc.response))
        except RetryableError as exc:
            error = exc
        if attempt == attempts - 1:
            raise error
        if error.retry_after is not None:
            delay = min(cap, max(0.0, float(error.retry_after)))
        else:
            raw = max(0.0, float(base)) * (2 ** attempt)
            delay = min(cap, raw + random.uniform(0.0, min(raw * 0.25, max(0.0, cap - raw))))
        time.sleep(delay)


def _retry_after(response):
    value = response.headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _request(method, url, **kwargs):
    global _force_direct_http
    try:
        if _force_direct_http:
            raise httpx.ConnectError("environment proxy previously failed")
        response = httpx.request(method, url, timeout=30.0, follow_redirects=True, **kwargs)
    except httpx.TransportError:
        # Some managed Windows environments inject an unreachable proxy while their TLS inspection certificate is
        # available only through the OS trust store. Fall back to a direct, still certificate-verified connection.
        _force_direct_http = True
        response = httpx.request(
            method,
            url,
            timeout=30.0,
            follow_redirects=True,
            trust_env=False,
            verify=ssl.create_default_context(),
            **kwargs,
        )
    if response.status_code in {429, 500, 502, 503, 504}:
        raise RetryableError(f"HTTP {response.status_code}", _retry_after(response))
    response.raise_for_status()
    return response


def _compact(value, limit=600):
    return " ".join(str(value or "").split())[:limit]


def _clamp_int(value, low, high, default):
    try:
        return max(low, min(int(value), high))
    except (TypeError, ValueError):
        return default


def _error_text(exc, secret=""):
    message = f"{type(exc).__name__}: {exc}"
    if secret:
        message = message.replace(secret, "***REDACTED***")
    return f"ERROR: {message}"


_arxiv_lock = threading.Lock()
_last_arxiv_call = 0.0
_force_direct_http = False


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    # PSEUDO-CODE:
    #   keep only word characters of `query` -> terms; no terms -> "NO RESULTS" (do not call the network)
    #   respect arXiv etiquette: at least 3 seconds between two arXiv calls (remember the time of the last call)
    #   GET ARXIV_URL params: search_query="all:t1 AND all:t2 ...", sortBy=submittedDate, sortOrder=descending,
    #       max_results=clamp(max_results, 1, 30)           (wrap in with_retry)
    #   parse the Atom XML: each <entry> -> {id (last part of <id> after /abs/), url, published[:10], title, summary}
    #       collapse whitespace/newlines in title and summary; cut summary to ~600 chars
    #   no entries -> "NO RESULTS"; else json.dumps(records, ensure_ascii=False)
    #   any exception -> "ERROR: <type>: <message>"
    terms = re.findall(r"[\w-]+", str(query or ""), flags=re.UNICODE)
    if not terms:
        return "NO RESULTS"
    params = {
        "search_query": " AND ".join(f"all:{term}" for term in terms),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": _clamp_int(max_results, 1, 30, 10),
        "start": 0,
    }

    def fetch():
        global _last_arxiv_call
        with _arxiv_lock:
            wait = 3.0 - (time.monotonic() - _last_arxiv_call)
            if wait > 0:
                time.sleep(wait)
            try:
                return _request("GET", ARXIV_URL, params=params)
            finally:
                _last_arxiv_call = time.monotonic()

    try:
        response = with_retry(fetch, attempts=7, base=2.0, cap=60.0)
        root = ET.fromstring(response.text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        records = []
        for entry in root.findall("a:entry", ns):
            raw_id = _compact(entry.findtext("a:id", default="", namespaces=ns))
            paper_id = raw_id.rsplit("/", 1)[-1]
            paper_id = re.sub(r"v\d+$", "", paper_id)
            if not paper_id:
                continue
            records.append({
                "id": paper_id,
                "url": f"https://arxiv.org/abs/{paper_id}",
                "published": _compact(entry.findtext("a:published", default="", namespaces=ns))[:10],
                "title": _compact(entry.findtext("a:title", default="", namespaces=ns)),
                "summary": _compact(entry.findtext("a:summary", default="", namespaces=ns), 600),
            })
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # tools never leak exceptions to the agent
        return _error_text(exc)


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    # PSEUDO-CODE:
    #   GET HF_DAILY_URL params: limit (clamp 1..100) and date (only when given)      (with_retry)
    #   response = list of items {"paper": {id, title, summary, upvotes, githubRepo, githubStars, publishedAt}, ...}
    #   map every item to the record shape above (skip items without paper.id); url = https://huggingface.co/papers/<id>
    #   keyword -> keep records whose title+summary contains it (case-insensitive); sort by upvotes descending
    params = {"limit": _clamp_int(limit, 1, 100, 30)}
    if date:
        params["date"] = date
    try:
        data = with_retry(lambda: _request("GET", HF_DAILY_URL, params=params).json())
        records = _hf_records(data, prefer_ai=False)
        if keyword:
            needle = str(keyword).casefold()
            records = [r for r in records if needle in f"{r['title']} {r['summary']}".casefold()]
        records.sort(key=lambda r: r["upvotes"], reverse=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error_text(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    # PSEUDO-CODE:
    #   GET HF_SEARCH_URL params: q=query, limit (clamp 1..50)                         (with_retry)
    #   same item shape as the daily endpoint; prefer paper["ai_summary"] over paper["summary"] when present
    query = str(query or "").strip()
    if not query:
        return "NO RESULTS"
    try:
        data = with_retry(lambda: _request(
            "GET", HF_SEARCH_URL, params={"q": query, "limit": _clamp_int(limit, 1, 50, 10)}
        ).json())
        records = _hf_records(data, prefer_ai=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error_text(exc)


def _hf_records(items, *, prefer_ai):
    records = []
    for item in items if isinstance(items, list) else []:
        paper = item.get("paper") if isinstance(item, dict) else None
        if not isinstance(paper, dict):
            continue
        paper_id = str(paper.get("id") or "").strip()
        if not paper_id:
            continue
        summary = paper.get("ai_summary") if prefer_ai and paper.get("ai_summary") else paper.get("summary")
        published = paper.get("publishedAt") or item.get("publishedAt") or ""
        records.append({
            "id": paper_id,
            "url": f"https://huggingface.co/papers/{paper_id}",
            "published": str(published)[:10],
            "title": _compact(paper.get("title") or item.get("title")),
            "summary": _compact(summary or item.get("summary"), 600),
            "upvotes": int(paper.get("upvotes") or item.get("upvotes") or 0),
            "github": paper.get("githubRepo") or item.get("githubRepo") or "",
            "stars": int(paper.get("githubStars") or item.get("githubStars") or 0),
        })
    return records


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    # PSEUDO-CODE:
    #   call the MCP tool "web_search_exa" with arguments {query, objective, numResults}
    #       (objective is REQUIRED by Exa: when empty, build one from the query)
    #   see GUIDE.md part 1.4 for how to call an MCP server over plain HTTP (JSON-RPC "tools/call") and read the answer
    #   read optional env EXA_API_KEY; when present it is sent to the Exa endpoint.
    #       (see GUIDE.md 1.4 for where it goes) => the key then appears in exception text: redact it before returning "ERROR: ..."
    #   WATCH OUT: read GUIDE.md 1.4 about how Exa signals "rate limited" on the free tier, and retry on it
    query = str(query or "").strip()
    if not query:
        return "NO RESULTS"
    arguments = {
        "query": query,
        "objective": str(objective or "").strip() or f"Find reliable sources about {query}",
        "numResults": _clamp_int(num_results, 1, 10, 5),
    }
    return _exa_tool("web_search_exa", arguments)


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    # PSEUDO-CODE: MCP tool "web_fetch_exa" with arguments {"urls": [url]}; truncate the text to ~12000 chars
    url = str(url or "").strip()
    if not re.match(r"^https?://", url, flags=re.I):
        return "ERROR: ValueError: url must start with http:// or https://"
    result = _exa_tool("web_fetch_exa", {"urls": [url]})
    return result if result.startswith("ERROR:") or result == "NO RESULTS" else result[:12000]


def _exa_tool(name, arguments):
    api_key = (os.getenv("EXA_API_KEY") or "").strip()
    endpoint_params = {"exaApiKey": api_key} if api_key else None
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

    def call():
        response = _request("POST", EXA_URL, params=endpoint_params, headers=headers, json=payload)
        envelope = _mcp_envelope(response)
        if "error" in envelope:
            raise RuntimeError(json.dumps(envelope["error"], ensure_ascii=False))
        result = envelope.get("result") or {}
        content = "\n".join(
            str(part.get("text", "")) for part in result.get("content", [])
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
        meta_text = json.dumps(result.get("_meta", {}), ensure_ascii=False).casefold()
        content_lower = content.casefold()
        if any(marker in meta_text for marker in ("rate_limit", "rate limit", "ratelimit", "throttl")) or (
            "rate limit" in content_lower and ("exa" in content_lower or "api key" in content_lower)
        ):
            raise RetryableError("Exa rate limit")
        if result.get("isError"):
            raise RuntimeError(content or "Exa MCP tool error")
        return content

    try:
        text = with_retry(call, attempts=5, base=2.0, cap=60.0)
        return text if text else "NO RESULTS"
    except Exception as exc:
        return _error_text(exc, api_key)


def _mcp_envelope(response):
    content_type = response.headers.get("content-type", "").lower()
    if "text/event-stream" not in content_type:
        return response.json()
    envelopes = []
    for line in response.text.splitlines():
        if line.startswith("data:"):
            raw = line[5:].strip()
            if raw and raw != "[DONE]":
                envelopes.append(json.loads(raw))
    if not envelopes:
        raise ValueError("Exa returned an empty event stream")
    return envelopes[-1]


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
