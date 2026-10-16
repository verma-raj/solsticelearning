"""All settings for the document finder in one place.

Every other file imports from here, so to change a path, a model name or a
threshold you only edit this file.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths ----
# All paths are built from this file's location, so the project can live anywhere.
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent                      # FDE\solsticelearning

KB_DIR = ROOT_DIR / "kb"
DOCS_DIR = KB_DIR / "docs"                     # the 40 Markdown help docs
EVAL_CSV = KB_DIR / "eval_questions.csv"       # questions used by evaluate.py

INDEX_DIR = ROOT_DIR / "index"                 # everything ingest.py builds
CHUNKS_JSON = INDEX_DIR / "chunks.json"        # documents + chunks cache
EMBEDDINGS_NPY = INDEX_DIR / "embeddings.npy"  # one vector per chunk (NumPy fallback)
CHROMA_DIR = INDEX_DIR / "chroma"              # ChromaDB persistent store
CHROMA_COLLECTION = "kb_chunks"

OUTPUT_DIR = ROOT_DIR / "output"               # results.csv, sweep.csv
RESULTS_CSV = OUTPUT_DIR / "results.csv"
SWEEP_CSV = OUTPUT_DIR / "sweep.csv"

# --------------------------------------------------------------- Ollama ----
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
EMBED_BATCH_SIZE = 16          # texts per request: small enough to stay fast, big enough to be efficient
EMBED_TIMEOUT_SECONDS = 120    # the first call can be slow while Ollama loads the model

# nomic-embed-text was trained with these task prefixes; leaving them out hurts quality.
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

# ------------------------------------------------------------- chunking ----
# ~350 words is roughly 500 tokens: long enough to keep a whole section together,
# short enough that one chunk is about one topic (sharper embeddings and excerpts).
CHUNK_MAX_WORDS = 350

# ------------------------------------------------------------ retrieval ----
TOP_K = 20           # candidates taken from each retriever before fusion
RRF_K = 60           # the standard Reciprocal Rank Fusion constant; damps the gap between ranks 1 and 2
MAX_DOCS_RETURNED = 5  # documents returned by retrieval.search()
MAX_DOCS_SHOWN = 3     # documents the web page shows at most

# BM25 scores are divided by a per-query upper bound (see retrieval.py) so they land in 0..1.
BM25_K1 = 1.5
BM25_B = 0.75

# ----------------------------------------------------------------- gate ----
# "Found / not found" thresholds, chosen with sweep_threshold.py on the 40 eval questions.
# A wrong document is worse than "No document available", so we sit on the plateau where
# no not-covered question gets through. Cosine alone could not do this: questions about
# uncovered features of the SAME product score 0.62-0.77, overlapping genuinely covered
# paraphrases (0.65-0.85), so a second, lexical signal has to agree.
COS_MIN = 0.64          # top document's cosine similarity must be at least this...
BM25_MIN = 0.20         # ...AND either its normalised BM25 score is at least this...
MIN_OVERLAP_TERMS = 3   # ...OR at least this many meaningful query words appear in the document.
                        # (1 lets generic words like "support" or "report" wave wrong documents through.)

# A query word that appears in more than this share of all documents (for example
# "solstice") says nothing about which document is right, so it is not "meaningful".
COMMON_TERM_MAX_DOC_RATIO = 0.5

# Words that never count as meaningful, even if rare. Kept short on purpose.
STOPWORDS = {
    "a", "about", "after", "again", "all", "also", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "but", "by", "can", "could", "did", "do", "does", "doing",
    "for", "from", "get", "got", "had", "has", "have", "he", "her", "here", "how", "i", "if",
    "in", "into", "is", "it", "its", "just", "me", "more", "most", "my", "need", "no", "not",
    "of", "on", "one", "or", "our", "out", "please", "should", "so", "some", "than", "that",
    "the", "their", "them", "then", "there", "these", "they", "this", "to", "too", "up", "us",
    "use", "very", "want", "was", "we", "were", "what", "when", "where", "which", "who", "why",
    "will", "with", "would", "you", "your",
}
