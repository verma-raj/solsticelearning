<h1 align="center">Solstice Learning: Document Finder</h1>
<p align="center"><em>One-time technical setup</em></p>

<hr>

<h2>1. Setup requirements</h2>
<table>
  <thead>
    <tr><th>Component</th><th>Version</th></tr>
  </thead>
  <tbody>
    <tr><td>VS Code</td><td>1.142.0</td></tr>
    <tr><td>Python</td><td>3.14.6</td></tr>
    <tr><td>Operating system</td><td>Windows 11</td></tr>
    <tr><td>Shell</td><td>PowerShell</td></tr>
  </tbody>
</table>

<h2>2. Create and activate the virtual environment</h2>
<p>Run these in PowerShell, from the project folder:</p>
<pre><code>python --version
python -m pip install --upgrade pip
python -m venv .venv
.venv\Scripts\Activate.ps1</code></pre>

<blockquote>
  <p><strong>If you get this error:</strong> <code>Activate.ps1 cannot be loaded because running scripts is disabled on this system</code></p>
  <p>Run the command below once, then activate again:</p>
</blockquote>
<pre><code>Set-ExecutionPolicy -Scope CurrentUser RemoteSigned</code></pre>

<h2>3. Install the Ollama model and check it responds</h2>
<pre><code>ollama pull llama3.2:3b</code></pre>
<p>Check that Ollama responds:</p>
<pre><code>ollama run llama3.2:3b "Say hello in one sentence"</code></pre>

<h2>4. Install the Python packages</h2>
<blockquote>
  <p>Install the CPU-only version of PyTorch first, so pip doesn't download a large CUDA build you don't need.</p>
</blockquote>
<pre><code>pip install torch --index-url https://download.pytorch.org/whl/cpu</code></pre>
<p>Then install the remaining packages, either one by one:</p>
<pre><code>pip install sentence-transformers rank_bm25 chromadb fastapi uvicorn streamlit pandas python-docx requests</code></pre>
<p><strong>or</strong> from the requirements file:</p>
<pre><code>pip install -r requirements.txt</code></pre>

<h2>5. Install the embedding model</h2>
<pre><code>ollama pull nomic-embed-text</code></pre>

<h2>6. Check that everything imports</h2>
<pre><code>python -c "import sentence_transformers, rank_bm25, chromadb, fastapi, streamlit, pandas, docx; print('All good')"</code></pre>
<p>You should see <code>All good</code>.</p>

<h2>7. Start the application</h2>
<p>To start the application (run from the <code>src</code> folder):</p>
<pre><code>Start-Process -WindowStyle Hidden -FilePath ..\.venv\Scripts\python.exe -ArgumentList '-m','uvicorn','api:app','--reload'</code></pre>

<table>
  <thead>
    <tr><th>What</th><th>URL</th><th>Expected</th></tr>
  </thead>
  <tbody>
    <tr><td>Application health check</td><td><a href="http://localhost:8000/health">http://localhost:8000/health</a></td><td>If the application is up, it shows <code>"status": "ok"</code></td></tr>
    <tr><td>Application</td><td><a href="http://localhost:8000/">http://localhost:8000/</a></td><td>The search page opens</td></tr>
  </tbody>
</table>

<hr>

<h3>Setup checklist</h3>
<table>
  <thead>
    <tr><th>#</th><th>Step</th><th>Done when</th></tr>
  </thead>
  <tbody>
    <tr><td>1</td><td>Requirements installed</td><td><code>python --version</code> shows 3.14.6</td></tr>
    <tr><td>2</td><td>Virtual environment active</td><td>The prompt shows <code>(.venv)</code></td></tr>
    <tr><td>3</td><td>Ollama model pulled</td><td>The test prompt returns a one-sentence hello</td></tr>
    <tr><td>4</td><td>Packages installed</td><td><code>pip</code> finishes without errors</td></tr>
    <tr><td>5</td><td>Embedding model pulled</td><td><code>ollama pull nomic-embed-text</code> completes</td></tr>
    <tr><td>6</td><td>Import check</td><td>Prints <code>All good</code></td></tr>
    <tr><td>7</td><td>Application started</td><td><code>/health</code> shows <code>"status": "ok"</code></td></tr>
  </tbody>
</table>
