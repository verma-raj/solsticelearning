# Solstice document finder

A local search page over the Markdown help docs in `kb/docs`. It shows the matching
document's title, a link and a verbatim excerpt - or exactly **"No document available"**.
No LLM runs at query time and nothing is generated; every word shown is copied from a document.

## Setup (once)

```powershell
cd FDE\solsticelearning\src
..\..\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull nomic-embed-text
```

## Run

All commands are run from the `src` folder.

```powershell
python ingest.py                 # build the index (re-run whenever kb/docs changes)
uvicorn api:app --reload         # then open http://localhost:8000
python evaluate.py               # score against kb/eval_questions.csv -> output/results.csv
python sweep_threshold.py        # COS_MIN sweep -> output/sweep.csv
```

Options: `python evaluate.py --retrieval bm25|vector|hybrid --cos-min 0.7 --bm25-min 0.15`
Health check: http://localhost:8000/health

## Files

| File | Job |
|---|---|
| `config.py` | every path, model name and threshold |
| `textutils.py` | tokenising / stopwords shared by the other files |
| `embedder.py` | text -> vectors through Ollama (`/api/embed`) |
| `ingest.py` | parse docs, chunk, build BM25 + ChromaDB (+ NumPy fallback) |
| `retrieval.py` | BM25 + vector search merged with Reciprocal Rank Fusion, one result per document |
| `gate.py` | found / not found decision with a written reason |
| `api.py` | web page, `/search`, `/doc/{id}`, `/health` |
| `evaluate.py`, `sweep_threshold.py` | measurement and threshold selection |

## How the gate decides

The top document is accepted only if **both** hold:
1. cosine similarity >= `COS_MIN` (0.70), and
2. normalised BM25 >= `BM25_MIN` (0.20) **or** at least `MIN_OVERLAP_TERMS` (3) meaningful query words
   appear in the document (stopwords and words found in over half of all documents are ignored).

Current results on the 40 eval questions: false-found 0/10 (group 3), miss rate 19/20 (groups 1-2; was 6/20 at COS_MIN 0.64),
top-1 / top-3 retrieval hit rate 75% / 90%. These thresholds were tuned on the same 40 questions,
so treat them as a starting point and re-run `sweep_threshold.py` after adding documents or questions.
