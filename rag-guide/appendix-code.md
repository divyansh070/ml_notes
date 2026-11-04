# Appendix: Reference Implementation Sketches

This appendix provides executable reference implementations and concrete algorithmic code sketches corresponding to each phase of the RAG Study Guide. While the main chapters emphasize architecture, trade-offs, and first-principles scaling intuition, this reference catalog provides self-contained Python implementations.

---

## Phase 1: Ingestion & Document Processing

### §P1.1 Fixed-Length Token Chunking with Overlap
```python
from typing import List
import tiktoken

def fixed_token_chunking(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """Token-accurate sliding window chunker.
    Note: 'cl100k_base' is used here as a clean stand-in; in production, always replace
    it with the exact tokenizer paired with your downstream embedding or language model.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    stride = chunk_size - overlap
    chunks = []
    for start in range(0, len(tokens), stride):
        chunk_tokens = tokens[start : start + chunk_size]
        chunks.append(enc.decode(chunk_tokens))
        if start + chunk_size >= len(tokens):
            break
    return chunks
```

---

### §P1.2 Structure-Aware Markdown Header Chunking
```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_document = """# OmniFlow Architecture
## Hot Memory Tier
In-memory HNSW index structures built per active partition.
## Cold Persistent Tier
Immutable Parquet-backed columnar files in object storage."""

headers_to_split_on = [
    ("#", "Header_1"),
    ("##", "Header_2"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
splits = markdown_splitter.split_text(markdown_document)
# Each split contains page_content plus metadata={'Header_1': '...', 'Header_2': '...'}
```

---

### §P1.3 Semantic Chunking via Consecutive Sentence Distance Thresholding
```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
sentences = [
    "OmniFlow uses HNSW with M=64.",
    "Each vector maintains bidirectional graph edges.",
    "Client timeouts caused retries that exhausted the gateway token bucket, emitting 429 errors.",
    "Subsequent retries caused duplicate partial partition graphs concurrently."
]

embeddings = model.encode(sentences, normalize_embeddings=True)
# Cosine distance for normalized vectors: d = 1 - dot_product
distances = [1.0 - np.dot(embeddings[i], embeddings[i+1]) for i in range(len(embeddings)-1)]

# ddof=1 ensures sample standard deviation matches math (dividing by N-1)
threshold = np.mean(distances) + 1.0 * np.std(distances, ddof=1)
chunks, current_chunk = [], [sentences[0]]

for i, dist in enumerate(distances):
    if dist > threshold:
        chunks.append(" ".join(current_chunk))
        current_chunk = [sentences[i+1]]
    else:
        current_chunk.append(sentences[i+1])
chunks.append(" ".join(current_chunk))
```

---

### §P1.4 Late Chunking via Long-Context Embedding Span Pooling
```python
import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)

text = "OmniFlow API Rate Limiting... Upgrades take effect in 15 seconds."
char_spans = [(0, 29), (30, 65)]  # Exact character slices for chunks

inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
offsets = inputs.pop("offset_mapping").squeeze(0)  # Shape: (seq_len, 2)

with torch.no_grad():
    # Pass task="retrieval.passage" for task-adapted jina-embeddings-v3 representations
    outputs = model(**inputs, task="retrieval.passage")
    token_embeddings = outputs.last_hidden_state.squeeze(0)

chunk_embeddings = []
for start_char, end_char in char_spans:
    # Exclude special tokens (where offset is (0, 0)) and match character span
    mask = (offsets[:, 0] >= start_char) & (offsets[:, 1] <= end_char) & (offsets[:, 1] > offsets[:, 0])
    span_vec = token_embeddings[mask].mean(dim=0)
    chunk_embeddings.append(torch.nn.functional.normalize(span_vec, p=2, dim=0))
```

---

### §P1.5 Small-to-Big (Parent-Child) Indexing and Retrieval
```python
import uuid
from typing import Dict, List, Tuple

class SmallToBigRetriever:
    def __init__(self):
        self.doc_store: Dict[str, str] = {}  # Key-Value store for Parents
        self.child_index: List[Tuple[str, str]] = []  # (child_text, parent_id)

    def index(self, parent_text: str, child_texts: List[str]):
        parent_id = str(uuid.uuid4())
        self.doc_store[parent_id] = parent_text
        for child in child_texts:
            self.child_index.append((child, parent_id))

    def resolve_context(self, retrieved_child_matches: List[str]) -> List[str]:
        """Resolves retrieved child chunks to unique parent documents."""
        matched_parent_ids = set()
        for child, parent_id in self.child_index:
            if child in retrieved_child_matches:
                matched_parent_ids.add(parent_id)
        return [self.doc_store[pid] for pid in matched_parent_ids]
```

---

### §P1.6 Summary Indexing and Metadata Enrichment
```python
from typing import Dict, Any

def enrich_chunk_metadata(chunk_text: str, doc_metadata: Dict[str, Any], summary: str) -> Dict[str, Any]:
    """Appends structural and summary lineage directly into chunk payload."""
    return {
        "text": chunk_text,
        "metadata": {
            "source_doc_id": doc_metadata.get("id"),
            "section": doc_metadata.get("current_section"),
            "tenant_tier": doc_metadata.get("tier", "all"),
            "doc_summary": summary,
            "char_length": len(chunk_text),
            "is_code": "def " in chunk_text or "class " in chunk_text
        }
    }
```

---

## Phase 2: Retrieval Primitives & Indexing

### §P2.1 BM25 Inverted Index & Scoring (SimpleBM25)
```python
import math
from collections import Counter

class SimpleBM25:
    """Minimal BM25 implementation with Lucene smoothed IDF."""
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.N = k1, b, len(corpus)
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lens) / self.N
        self.doc_freqs = Counter(term for doc in corpus for term in set(doc))
        self.corpus = [Counter(doc) for doc in corpus]

    def score(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.N
        for term in query:
            n = self.doc_freqs.get(term, 0)
            if n == 0:
                continue
            idf = math.log(1.0 + (self.N - n + 0.5) / (n + 0.5))
            for i, doc in enumerate(self.corpus):
                f = doc.get(term, 0)
                if f > 0:
                    K = 1.0 - self.b + self.b * (self.doc_lens[i] / self.avgdl)
                    scores[i] += idf * (f * (self.k1 + 1.0)) / (f + self.k1 * K)
        return scores
```

---

### §P2.2 Dense Bi-Encoder Runtime Inference & Pooling
```python
from sentence_transformers import SentenceTransformer

# Load modern asymmetric bi-encoder (BGE uses [CLS] pooling internally)
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

query = "Why did our servers crash while building the search index for a big data import?"
docs = [
    "OmniFlow buffers writes in WAL and flushes to Parquet.",
    "Incident 4092: Worker OOM During Large Batch Ingestion. HNSW M=64 raw vectors + Layer-0 edges exhausted 64 GB instances."
]

# BGE requires explicit instruction prefix for queries
instruction = "Represent this sentence for searching relevant passages: "
q_emb = model.encode(instruction + query, normalize_embeddings=True)
doc_embs = model.encode(docs, normalize_embeddings=True)

similarities = q_emb @ doc_embs.T  # Cosine similarities
# similarities[1] is significantly higher than similarities[0]
```

---

### §P2.3 Unsupervised Contrastive Pretraining with Contriever
```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained unsupervised Contriever
tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
model = AutoModel.from_pretrained("facebook/contriever")

sentences = [
    "Why did our servers crash while building the search index for a big data import?",
    "Incident 4092: Ingestion worker OOM during large HNSW batch backfill."
]

inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    # Contriever uses mean pooling across token states
    mask = inputs["attention_mask"].unsqueeze(-1)
    embeddings = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

sim = (embeddings[0] @ embeddings[1]).item()
# sim captures high semantic similarity without needing supervised QA training data
```

---

### §P2.4 Vector Metric Computations & MIPS-to-L2 Reduction
```python
import numpy as np

def fast_batch_cosine_search(q: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """Computes exact cosine similarity via SIMD dot-product on pre-normalized vectors."""
    # Offline: Ensure doc_matrix is L2-normalized:
    # doc_matrix = doc_matrix / np.linalg.norm(doc_matrix, axis=1, keepdims=True)
    # Online: Normalize query once:
    q_norm = q / np.linalg.norm(q)
    # Pure matrix-vector multiplication (BLAS GEMV):
    scores = np.dot(doc_matrix, q_norm)
    return scores  # Exactly equal to cosine similarity

def mips_to_l2_transform(query: np.ndarray, docs: np.ndarray):
    """Reduces Maximum Inner Product Search (MIPS) to Euclidean L2 search.
    Implements Bachrach et al. (2014) norm augmentation.
    """
    doc_norms = np.linalg.norm(docs, axis=1)
    phi = np.max(doc_norms)
    
    # Augment docs: extra dimension sqrt(phi^2 - ||x||^2) makes all augmented docs have norm phi
    extra_dim = np.sqrt(np.maximum(0.0, phi**2 - doc_norms**2))
    docs_augmented = np.hstack([docs, extra_dim[:, np.newaxis]])
    
    # Augment query: extra dimension 0
    query_augmented = np.append(query, 0.0)
    return query_augmented, docs_augmented
```

---

### §P2.5 Inverted File with Product Quantization (IVF-PQ with Faiss)
```python
import faiss
import numpy as np

d = 128                          # Vector dimension
nlist = 32                       # Number of Voronoi coarse cells
m = 16                           # Number of sub-vectors (8 dims per sub-vector, 1 byte code)
quantizer = faiss.IndexFlatL2(d) # Coarse quantizer
# 8 bits per sub-vector code = 256 centroids per sub-space
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)

# Generate dummy training and database vectors
np.random.seed(42)
data = np.random.random((10000, d)).astype(np.float32)
index.train(data)                # Trains coarse centroids and PQ codebooks
index.add(data)                  # Quantizes residuals and builds posting lists

index.nprobe = 4                 # Search the 4 closest Voronoi cells out of 32
query = np.random.random((1, d)).astype(np.float32)
distances, indices = index.search(query, k=5)
```

---

### §P2.6 Hierarchical Navigable Small World (HNSW with hnswlib)
```python
import hnswlib
import numpy as np

dim = 128
num_elements = 10000

# Generating dummy data
data = np.random.random((num_elements, dim)).astype(np.float32)

# Initializing HNSW index with cosine distance
p = hnswlib.Index(space="cosine", dim=dim)

# M: Max outgoing edges per node (heuristic: 16 to 64)
# ef_construction: Depth of candidate search during construction (heuristic: 100 to 200)
p.init_index(max_elements=num_elements, ef_construction=100, M=16)

# Add vectors (construction phase)
p.add_items(data)

# efSearch: Depth of candidate queue during runtime search (heuristic: 50 to 100)
p.set_ef(50)

# Querying
query = np.random.random((1, dim)).astype(np.float32)
labels, distances = p.knn_query(query, k=5)
```
