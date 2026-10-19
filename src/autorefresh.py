"""Keeps the search index in step with kb/docs while the app is running.

A background thread checks the documents every few seconds. When a document is added,
changed or removed it rebuilds the index and switches the app to the new copy, so no
restart and no manual `python ingest.py` is needed.
"""
from __future__ import annotations

import threading
from datetime import datetime

import config
import ingest
import retrieval

# Shown by /health so you can see whether the last refresh worked.
status: dict = {"enabled": config.AUTO_REFRESH, "last_rebuild": None, "documents": None, "last_error": None}

_stop = threading.Event()
_thread: threading.Thread | None = None


def _step(state: dict) -> None:
    """One check: rebuild the index if the documents changed, then load the newest index."""
    current = ingest.docs_fingerprint()
    on_disk = ingest.stored_fingerprint()

    if on_disk != current:
        # Wait for the change to be seen twice in a row, so a file that is still being saved
        # (or copied) is not indexed half-written.
        if state["pending"] != current:
            state["pending"] = current
            return
        info = ingest.build_index()
        state["pending"] = None
        status["last_rebuild"] = datetime.now().isoformat(timespec="seconds")
        status["documents"] = info["documents"]
        on_disk = ingest.stored_fingerprint()
    else:
        state["pending"] = None

    # Also covers an index built by `python ingest.py` in another window.
    if on_disk is not None and retrieval.loaded_fingerprint() != on_disk:
        retrieval.reload_retriever()


def _run() -> None:
    state = {"pending": None}
    wait = 0  # the first check runs straight away at start-up
    while not _stop.wait(wait):
        wait = config.AUTO_REFRESH_SECONDS
        try:
            _step(state)
            status["last_error"] = None
        except Exception as exc:  # for example Ollama is down: keep the old index and try again later
            status["last_error"] = str(exc)
            state["pending"] = None
            wait = config.AUTO_REFRESH_RETRY_SECONDS  # do not hammer a server that is down


def start() -> None:
    """Start the watcher thread (does nothing if disabled in config or already running)."""
    global _thread
    if not config.AUTO_REFRESH or (_thread is not None and _thread.is_alive()):
        return
    _stop.clear()
    _thread = threading.Thread(target=_run, name="index-auto-refresh", daemon=True)
    _thread.start()


def stop() -> None:
    _stop.set()
