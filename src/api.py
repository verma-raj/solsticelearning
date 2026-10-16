"""FastAPI app: the search page, the JSON search endpoint, full-document pages and a health check.

Run with:  uvicorn api:app --reload      (then open http://localhost:8000)
Nothing here generates text: every word shown comes from the stored documents.
"""
from __future__ import annotations

import html

import markdown
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

import config
import gate
import retrieval
from embedder import OllamaError, check_ollama
from ingest import parse_front_matter

app = FastAPI(title="Solstice document finder")

NOT_FOUND_MESSAGE = "No document available"  # the exact text required when nothing relevant exists
MAX_QUERY_CHARS = 300                        # protects the embedding call from absurdly long input

# ------------------------------------------------------------------ HTML ----
# One self-contained page: inline CSS + JS, no CDN. All text from the server is put on the
# page with textContent (never innerHTML), so nothing in a document or query can inject HTML.
SEARCH_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Solstice document finder</title>
<style>
  body { font-family: Segoe UI, Arial, sans-serif; background: #f4f6f8; color: #1f2933; margin: 0; }
  main { max-width: 760px; margin: 0 auto; padding: 32px 16px; }
  h1 { font-size: 1.6rem; margin-bottom: 4px; }
  .sub { color: #52606d; margin-top: 0; }
  form { display: flex; gap: 8px; margin: 20px 0; }
  input[type=text] { flex: 1; padding: 12px; font-size: 1rem; border: 1px solid #9aa5b1; border-radius: 6px; }
  button { padding: 12px 22px; font-size: 1rem; border: 0; border-radius: 6px; background: #2b6cb0; color: #fff; cursor: pointer; }
  button:disabled { background: #9aa5b1; cursor: default; }
  .card { background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 16px; margin-bottom: 14px; }
  .card h2 { font-size: 1.15rem; margin: 0 0 2px; }
  .id { color: #52606d; font-size: 0.85rem; margin-bottom: 10px; }
  .excerpt { white-space: pre-wrap; background: #f0f4f8; border-left: 4px solid #2b6cb0; padding: 10px 12px; margin: 10px 0; font-size: 0.95rem; }
  .none { font-size: 1.2rem; font-weight: 600; color: #52606d; padding: 20px 0; }
  .error { color: #b83232; padding: 12px 0; }
  a.open { color: #2b6cb0; font-weight: 600; }
</style>
</head>
<body>
<main>
  <h1>Solstice document finder</h1>
  <p class="sub">Type a question or a few keywords. You will get the matching help document, or nothing at all.</p>
  <form id="search-form">
    <input type="text" id="q" placeholder="e.g. How long is the password reset link valid?" autofocus autocomplete="off">
    <button type="submit" id="go">Search</button>
  </form>
  <div id="results"></div>
</main>
<script>
  const form = document.getElementById("search-form");   // submitting the form also handles the Enter key
  const input = document.getElementById("q");
  const button = document.getElementById("go");
  const box = document.getElementById("results");

  // Small helper: create an element with plain text (textContent escapes everything).
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function showMessage(className, text) {
    box.replaceChildren(el("div", className, text));
  }

  function renderResults(results) {
    box.replaceChildren();
    for (const r of results) {
      const card = el("div", "card");
      card.appendChild(el("h2", null, r.title));
      card.appendChild(el("div", "id", r.doc_id));
      card.appendChild(el("div", "excerpt", r.excerpt));
      const link = el("a", "open", "Open document");
      link.href = r.link;          // always /doc/<known id>, built by the server
      link.target = "_blank";
      link.rel = "noopener";
      card.appendChild(link);
      box.appendChild(card);
    }
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const q = input.value.trim();
    if (!q) { showMessage("none", "Please type a question or some keywords."); return; }
    button.disabled = true;
    showMessage("none", "Searching...");
    try {
      const response = await fetch("/search?q=" + encodeURIComponent(q));
      const data = await response.json();
      if (!response.ok) {
        showMessage("error", data.message || "Search is unavailable right now.");
      } else if (data.found) {
        renderResults(data.results);
      } else {
        showMessage("none", data.message);   // exactly: No document available
      }
    } catch (err) {
      showMessage("error", "Could not reach the server. Is it still running?");
    } finally {
      button.disabled = false;
    }
  });
</script>
</body>
</html>
"""

DOC_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ font-family: Segoe UI, Arial, sans-serif; background: #f4f6f8; color: #1f2933; margin: 0; }}
  main {{ max-width: 800px; margin: 0 auto; padding: 24px 16px 48px; }}
  article {{ background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 8px 24px 24px; }}
  table {{ border-collapse: collapse; margin: 12px 0; }}
  th, td {{ border: 1px solid #bcccdc; padding: 6px 12px; text-align: left; }}
  th {{ background: #f0f4f8; }}
  a {{ color: #2b6cb0; }}
  .meta {{ color: #52606d; font-size: 0.85rem; }}
</style>
</head>
<body>
<main>
  <p><a href="/">&larr; Back to search</a></p>
  <article>
    <p class="meta">{doc_id} &middot; last updated {updated}</p>
    {body}
  </article>
</main>
</body>
</html>
"""


# -------------------------------------------------------------- endpoints ----
@app.get("/", response_class=HTMLResponse)
def home() -> str:
    """The search page."""
    return SEARCH_PAGE


@app.get("/search")
def search(q: str = Query("", description="Plain-English question or keywords")) -> JSONResponse:
    """Search the documents and return them only if the gate says they are relevant."""
    query = q.strip()[:MAX_QUERY_CHARS]
    if not query:
        return JSONResponse({
            "query": q, "found": False, "results": [],
            "message": "Please type a question or some keywords to search.",
            "reason": "The query was empty.",
        })
    try:
        results = retrieval.search(query)
        decision = gate.decide(query, results)
    except (OllamaError, retrieval.IndexNotBuiltError) as exc:
        # A broken setup is reported as an error, never disguised as "No document available".
        return JSONResponse(status_code=503, content={
            "query": query, "found": False, "message": str(exc), "results": [], "reason": str(exc),
        })

    shown = decision.accepted[: config.MAX_DOCS_SHOWN] if decision.found else []
    return JSONResponse({
        "query": query,
        "found": decision.found,
        "message": f"Found {len(shown)} matching document(s)" if decision.found else NOT_FOUND_MESSAGE,
        "results": [{
            "doc_id": d["doc_id"],
            "title": d["title"],
            "link": f"/doc/{d['doc_id']}",
            "excerpt": d["excerpt"],            # verbatim text from the document
            "score": d["cosine_similarity"],
        } for d in shown],
        "reason": decision.reason,
    })


@app.get("/doc/{doc_id}", response_class=HTMLResponse)
def document(doc_id: str) -> HTMLResponse:
    """Show one full document as HTML."""
    try:
        documents = retrieval.get_retriever().documents
    except retrieval.IndexNotBuiltError as exc:
        return HTMLResponse(f"<p>{html.escape(str(exc))}</p>", status_code=503)

    # Only ids that exist in the index are served, and the file name comes from the index,
    # never from the URL, so "../" tricks cannot reach any other file.
    info = documents.get(doc_id)
    if info is None:
        return HTMLResponse(
            DOC_PAGE.format(title="Not found", doc_id="404", updated="-",
                            body="<h1>Document not found</h1><p>There is no document with that id.</p>"),
            status_code=404,
        )
    _, body_md = parse_front_matter((config.DOCS_DIR / info["file"]).read_text(encoding="utf-8"))
    # The docs are our own trusted files, so their Markdown is rendered as-is.
    body_html = markdown.markdown(body_md, extensions=["tables"])
    return HTMLResponse(DOC_PAGE.format(
        title=html.escape(info["title"]), doc_id=html.escape(doc_id),
        updated=html.escape(info["last_updated"]), body=body_html,
    ))


@app.get("/health")
def health() -> JSONResponse:
    """Is Ollama reachable, is the model installed, does the index exist?"""
    ollama = check_ollama()
    index_ready = config.CHUNKS_JSON.exists() and config.EMBEDDINGS_NPY.exists()
    ok = ollama["reachable"] and ollama["model_installed"] and index_ready
    hints = []
    if not ollama["reachable"]:
        hints.append("Start Ollama (run `ollama serve` or open the Ollama app).")
    elif not ollama["model_installed"]:
        hints.append(f"Run: ollama pull {config.EMBED_MODEL}")
    if not index_ready:
        hints.append("Run: python ingest.py")
    return JSONResponse(
        status_code=200 if ok else 503,
        content={
            "status": "ok" if ok else "degraded",
            "ollama_reachable": ollama["reachable"],
            "embedding_model_installed": ollama["model_installed"],
            "index_built": index_ready,
            "hints": hints,
        },
    )
