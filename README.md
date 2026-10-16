# solsticelearning 

This is case study for FDE assignment
------------------------------

=========================
Forward Deployed Engineer 
==========================

CASE STUDY
-----------------------------------------

SOLSTICE LEARNING

Sells an online- course platform to corporate training a documentation site admins rarely search successfully.

“ New admins ask us the same handful of questions over and over — password resets, export limits, billing cycles — and it’ s all 
already in our docs, they just don’t want to dig through pages to find it. I want some thing where an admin can type a plain -English 
question and get a real answer pulled from our documentation. The one thing I really care about: if the docs genuinely don’t cover something, 
I’d much rather it say ‘I don’t have an answer for that’ than make something up.”

THE ASK  IS  DELIBERATELY
--------------------------------------------- 
You decide — and document — how to 
across documents for the right passage, 
confident an answer must be before 
rather than saying you don’ t know, and 
to use an LLM at all.


WHAT WE ARE EVALUATING
--------------------------------------------
• Handling ambiguity
• Soundness of design judgment
• Building under real constraints
• Clarity of your reasoning
• Response when challenged

******************************************************************************************************************
******************************************************************************************************************


<!-- ===================== Technical stack (added) ===================== -->
<h2>Document finder: technical stack</h2>
<p>A fully local search app over the Solstice help documents. No LLM is used at query time and nothing is generated: every word shown is copied from a stored document.</p>

<h3>Tech layers</h3>
<table border="1" cellpadding="6" cellspacing="0">
  <thead>
    <tr><th>#</th><th>Layer</th><th>What it does</th><th>Technology</th><th>Files</th></tr>
  </thead>
  <tbody>
    <tr><td>1</td><td>Data</td><td>40 Markdown help docs with a front-matter header, plus the evaluation questions</td><td>Markdown, CSV</td><td><code>kb/docs</code>, <code>kb/eval_questions.csv</code></td></tr>
    <tr><td>2</td><td>Ingestion</td><td>Parses each document, splits it into chunks by heading, builds every index</td><td>Python, <code>re</code>, <code>json</code></td><td><code>src/ingest.py</code></td></tr>
    <tr><td>3</td><td>Embedding</td><td>Turns text into 768-number vectors (document and query prefixes, L2-normalised)</td><td>Ollama + nomic-embed-text, <code>requests</code>, <code>numpy</code></td><td><code>src/embedder.py</code></td></tr>
    <tr><td>4</td><td>Storage</td><td>Persists vectors, chunks and documents; BM25 is rebuilt in memory at start-up</td><td>ChromaDB, NumPy <code>.npy</code>, JSON</td><td><code>index/</code></td></tr>
    <tr><td>5</td><td>Retrieval</td><td>Keyword search (BM25) and vector search merged with Reciprocal Rank Fusion, one result per document</td><td><code>rank_bm25</code>, ChromaDB, <code>numpy</code></td><td><code>src/retrieval.py</code>, <code>src/textutils.py</code></td></tr>
    <tr><td>6</td><td>Decision gate</td><td>Returns "No document available" unless cosine similarity and a lexical check both pass</td><td>Python rules and thresholds</td><td><code>src/gate.py</code></td></tr>
    <tr><td>7</td><td>API and UI</td><td>Search page, JSON search, full-document pages, health check</td><td>FastAPI, Uvicorn, <code>markdown</code>, HTML/CSS/JavaScript</td><td><code>src/api.py</code></td></tr>
    <tr><td>8</td><td>Evaluation</td><td>Scores the finder per question group and sweeps the threshold</td><td><code>pandas</code>, <code>numpy</code></td><td><code>src/evaluate.py</code>, <code>src/sweep_threshold.py</code></td></tr>
    <tr><td>9</td><td>Configuration</td><td>Paths, model name, chunk size and all thresholds in one place</td><td>Python, <code>pathlib</code></td><td><code>src/config.py</code></td></tr>
  </tbody>
</table>

<h3>Packages</h3>
<table border="1" cellpadding="6" cellspacing="0">
  <thead>
    <tr><th>Package</th><th>Used for</th><th>Used in</th></tr>
  </thead>
  <tbody>
    <tr><td><code>requests</code></td><td>HTTP calls to Ollama (no-proxy session)</td><td>embedder.py</td></tr>
    <tr><td><code>numpy</code></td><td>Vectors, cosine similarity, saved embeddings</td><td>embedder.py, ingest.py, retrieval.py, sweep_threshold.py</td></tr>
    <tr><td><code>rank_bm25</code></td><td>BM25 keyword scoring</td><td>ingest.py, retrieval.py</td></tr>
    <tr><td><code>chromadb</code></td><td>Persistent vector store</td><td>ingest.py, retrieval.py</td></tr>
    <tr><td><code>fastapi</code></td><td>Web framework and endpoints</td><td>api.py</td></tr>
    <tr><td><code>uvicorn</code></td><td>Web server that runs the app</td><td>command line</td></tr>
    <tr><td><code>markdown</code></td><td>Renders a full document to HTML (tables extension)</td><td>api.py</td></tr>
    <tr><td><code>pandas</code></td><td>Evaluation tables and CSV output</td><td>evaluate.py, sweep_threshold.py</td></tr>
    <tr><td>Python standard library</td><td><code>json</code>, <code>re</code>, <code>html</code>, <code>argparse</code>, <code>pathlib</code>, <code>dataclasses</code>, <code>functools</code>, <code>sys</code></td><td>all files</td></tr>
    <tr><td>Ollama (external program)</td><td>Serves the nomic-embed-text embedding model locally</td><td>embedder.py</td></tr>
  </tbody>
</table>

<h3>Run</h3>
<p>From the <code>src</code> folder: <code>python ingest.py</code>, then <code>uvicorn api:app --reload</code> (open http://localhost:8000), <code>python evaluate.py</code>, <code>python sweep_threshold.py</code>.</p>
