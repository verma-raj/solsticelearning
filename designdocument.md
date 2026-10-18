<p><em>Classification: Internal</em></p>

<h1>Case Study: SOLSTICE LEARNING</h1>

<p>Sells an online-course platform to corporate training. Its documentation site is one that admins rarely search successfully.</p>

<blockquote>
  <p>"New admins ask us the same handful of questions over and over — password resets, export limits, billing cycles — and it's all already in our docs, they just don't want to dig through pages to find it. I want something where an admin can type a plain-English question and get a real answer pulled from our documentation. The one thing I really care about: if the docs genuinely don't cover something, I'd much rather it says, 'I don't have an answer for that' than make something up."</p>
</blockquote>

<hr>

<h1>Tech Stack</h1>

<p><strong>Language and Runtime:</strong> Python 3.14, Windows 11, VSCode</p>

<h3>Search and AI Components</h3>
<ul>
  <li><strong><em>Ollama</em></strong> (local) serving the nomic-embed-text embedding model. It turns text into 768-number vectors and generates no text.</li>
  <li><strong><em>BM25 keyword search</em></strong> through rank_bm25</li>
  <li><strong><em>Vector search</em></strong> through ChromaDB (persistent store), with a NumPy fallback</li>
  <li><strong><em>Reciprocal Rank Fusion</em></strong> to merge the keyword and vector rankings</li>
  <li><strong><em>A rule-based relevance gate</em></strong> (cosine and keyword thresholds, plus the platform rule) that decides between "found" and "No document available"</li>
</ul>

<h3>Python packages</h3>
<ul>
  <li>requests, numpy, pandas, rank_bm25, chromadb</li>
  <li>fastapi and uvicorn for the web server</li>
  <li>markdown for rendering full documents</li>
  <li>standard library: json, re, html, argparse, pathlib, dataclasses, functools, sys</li>
</ul>

<h3>Web front end</h3>
<ul>
  <li>A single HTML page with inline CSS and JavaScript, with no framework and no CDN</li>
</ul>

<h3>Code Version management</h3>
<ul>
  <li>GitHub</li>
</ul>

<hr>

<h1>Design &amp; Considerations</h1>

<h2>Solstice Document Finder: Design</h2>

<h3>Problem and goals</h3>
<p>Solstice wants admins to type a plain-English question and get a real answer from the documentation, and to be told "I don't have an answer" when the docs do not cover it. Wrongly returning a document is worse than returning nothing.</p>
<p>The app is a local web page over 40 help documents on four topics: password reset (PWD), data export (EXP), billing (BIL) and purchasing (PUR). For each search it shows up to three documents, each with a title, a link to the full document and a verbatim excerpt. Otherwise it shows exactly "No document available".</p>
<ul>
  <li><strong>Nothing generated.</strong> Every word on the page is copied from a stored document.</li>
  <li><strong>Fully local.</strong> No cloud APIs and no Hugging Face downloads, because Hugging Face is blocked on the network.</li>
</ul>

<h3>Key design decisions</h3>
<p>The design avoids hallucination by never generating text and by refusing to answer unless two independent signals agree.</p>
<table>
  <thead>
    <tr><th>Decision</th><th>Choice</th><th>Why</th></tr>
  </thead>
  <tbody>
    <tr><td>LLM at runtime</td><td>None</td><td>Nothing is generated, so nothing can be made up. Every word shown is copied from a stored document.</td></tr>
    <tr><td>Finding documents</td><td>Keyword search (BM25) plus vector search, merged with Reciprocal Rank Fusion</td><td>Keywords catch exact terms such as limits and error names. Vectors catch paraphrases such as "spreadsheet download cut off".</td></tr>
    <tr><td>Embeddings</td><td>nomic-embed-text through local Ollama</td><td>Hugging Face is blocked and cloud APIs are ruled out.</td></tr>
    <tr><td>Saying "no"</td><td>A rule-based gate with written reasons</td><td>Each decision can be explained and each threshold tuned and justified.</td></tr>
    <tr><td>Chunking</td><td>One chunk per heading, split at paragraph breaks above 350 words</td><td>Keeps a section whole. The document title is added to the searched text only, so the excerpt stays verbatim.</td></tr>
    <tr><td>Output</td><td>Up to 3 documents, best first, each with a verbatim excerpt and a link</td><td>The admin can open the full document and check the answer.</td></tr>
  </tbody>
</table>

<h3>Architecture</h3>
<p>The app has two parts that share one index: a one-off build step and a search path that ends in the gate.</p>

<p><strong>Every search passes a gate before anything is shown</strong></p>
<table>
  <tbody>
    <tr><td colspan="7"><em>Build once: python ingest.py</em></td></tr>
    <tr align="center">
      <td><strong>Help documents</strong><br>40 Markdown files<br>in kb/docs</td>
      <td>&rarr;</td>
      <td><strong>Ingest</strong><br>parse and chunk<br>ingest.py</td>
      <td>&rarr;</td>
      <td><strong>Embedder</strong><br>Ollama<br>nomic-embed-text</td>
      <td colspan="2"></td>
    </tr>
    <tr align="center">
      <td></td><td></td>
      <td>&darr; chunks</td>
      <td></td>
      <td>&darr; vectors</td>
      <td colspan="2"></td>
    </tr>
    <tr align="center" bgcolor="#eeeeee">
      <td colspan="7"><strong>Index store</strong><br>ChromaDB vectors, embeddings.npy, chunks.json (BM25 built at start-up)</td>
    </tr>
    <tr align="center">
      <td colspan="7">&darr; reads (Retrieval)</td>
    </tr>
    <tr><td colspan="7"><em>Each search</em></td></tr>
    <tr align="center">
      <td><strong>Search page</strong><br>type a question<br>or keywords</td>
      <td>&rarr;</td>
      <td><strong>API</strong><br>FastAPI<br>/search and /doc</td>
      <td>&rarr;</td>
      <td><strong>Retrieval</strong><br>BM25 + vector<br>merged by RRF</td>
      <td>&rarr;</td>
      <td bgcolor="#dbe9fb"><strong>Gate</strong><br>three checks decide<br>found or not</td>
    </tr>
    <tr align="center">
      <td colspan="7">&larr; result back to the page: documents with verbatim excerpts, or No document available</td>
    </tr>
  </tbody>
</table>
<p><em>architecture &middot; 7 components, 1 index store</em></p>

<p>The Embedder is used twice: at ingest for the documents and at search for the query. Retrieval reads the index and the Gate decides what the API returns to the page. All settings, such as paths, model name and thresholds, live in one config file.</p>

<h3>How a search works</h3>
<p>An index is built once from the documents, then every search runs the same five steps.</p>

<p><strong>Once, at ingest (<code>python ingest.py</code>)</strong></p>
<ol>
  <li>Read each Markdown file and split its header (doc_id, title) from the body.</li>
  <li>Cut the body into chunks at headings. Keep the text exactly as written for display, and a second copy with the title added for searching.</li>
  <li>Turn each chunk into a 768-number vector with Ollama, then store the vectors in ChromaDB and a NumPy file. Save the chunks as JSON. The BM25 keyword index is rebuilt in memory when the app starts.</li>
</ol>

<p><strong>On every search</strong></p>
<ol>
  <li>Split the query into lowercase words with plurals trimmed and turn it into a vector.</li>
  <li>Score every chunk with BM25. The score is divided by the best score this query could get, so it falls between 0 and 1. Keep the top 20 chunks that match at least one word.</li>
  <li>Rank chunks by cosine similarity to the query vector and keep the top 20.</li>
  <li>Merge the two lists. A chunk earns 1 / (60 + rank) from each list it appears in, and the points are added.</li>
  <li>Keep each document's best chunk, return the top 5 documents, and pass them to the gate.</li>
</ol>

<h3>The gate: found or not found</h3>
<p>Retrieval always returns something, so the gate decides whether the top document is good enough, and the default answer is "No document available". The top document must pass all three checks.</p>
<table>
  <thead>
    <tr><th>Check</th><th>Rule</th><th>Why</th></tr>
  </thead>
  <tbody>
    <tr><td>Platform</td><td>If the query names a platform such as Linux, Mac or Ubuntu, the document must mention it</td><td>"Reset my Linux password" is close in meaning to every password document, and the matching words hide that Linux is not documented anywhere.</td></tr>
    <tr><td>Meaning</td><td>Cosine similarity at least 0.70</td><td>The query must be about the same thing as the document.</td></tr>
    <tr><td>Keywords</td><td>BM25 at least 0.20, or at least 3 meaningful query words appear in the document</td><td>A second signal must agree. Stopwords and words found in over half of all documents, such as "solstice", do not count.</td></tr>
  </tbody>
</table>
<p>Lower-ranked documents are shown only if they pass the same checks themselves. Every decision comes back with a written reason, for example: "NOT FOUND - top document PWD-03: the query mentions 'linux' but the document never does."</p>
