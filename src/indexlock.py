"""One shared lock so the index files are never read while they are being rewritten."""
import threading

# RLock: the same thread may take it twice (for example a rebuild that also reloads).
INDEX_LOCK = threading.RLock()
