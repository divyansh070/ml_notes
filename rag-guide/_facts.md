# The Canonical RAG Guide Facts & Constants (`_facts.md`)
## Authoritative Source of Truth for Numbers, Formulas, Corpus, and Worked Examples

> **CRITICAL DIRECTIVE**: This file is the single canonical source of truth for the entire RAG Study Guide (`00` through `07`, `98`, `99`, and `appendix-code.md`). 
> Every phase file must copy values, formulas, text snippets, and numerical results **verbatim** from this file and must **never alter them**.
> 
> - **Units Policy**: Decimal gigabytes everywhere ($1\text{ GB} = 10^9\text{ bytes}$, $1\text{ MB} = 10^6\text{ bytes}$). Binary $\text{GiB}$ is reserved exclusively for literal quotations of Document $D_1$'s explicit text (`4 GiB`).
> - **Verification Policy**: Every numerical constant in this file has been verified with Python.

---

## 1. The Canonical Corpus & Queries (Verbatim from `00-index.md`)

### 1.1 The Running Corpus ($\mathcal{D} = \{D_1, D_2, D_3, D_4, D_5, D_6\}$)

#### **Document $D_1$: OmniFlow Engine Architecture & Storage Tier**
```markdown
# OmniFlow Architecture Specification: Storage & Memory Hierarchy
OmniFlow is a distributed streaming vector database engine. The storage subsystem 
is decoupled into two primary tiers:
1. Hot Memory Tier: In-memory HNSW index structures built per active partition for 
   sub-millisecond approximate nearest neighbor search.
2. Cold Persistent Tier: Immutable Parquet-backed columnar files residing in object storage, 
   indexed via sparse lexical inverted indices and inverted file postings lists (IVF).
During data ingestion, OmniFlow buffers vector writes in a memory-mapped write-ahead 
log (WAL). Partitions flush to persistent object storage once their uncompressed memory 
footprint exceeds 4 GiB or when the flush timer reaches 300 seconds. OmniFlow partitions 
maintain isolated concurrency locks: readers acquire shared read leases, while compactions 
require exclusive table-level locks.
```

#### **Document $D_2$: OmniFlow Rate Limits, Quota Policies & Tier Upgrades**
```markdown
# OmniFlow API Rate Limiting and Quota Enforcement
All external API access to the OmniFlow distributed gateway is rate-limited via a 
distributed token-bucket algorithm keyed on tenant API credentials:
- Developer Tier: 100 requests per minute (RPM), bursting up to 150 RPM; 10 MB/sec payload bandwidth.
- Production Tier: 2,500 requests per minute (RPM), bursting up to 3,500 RPM; 100 MB/sec payload bandwidth.
- Enterprise Tier: 20,000 requests per minute (RPM), bursting up to 30,000 RPM; 1 GB/sec payload bandwidth.
When a client exceeds its token allowance, the gateway rejects the request with HTTP status code 429 
(Too Many Requests), returning a JSON error payload containing:
  {"error": "rate_limited", "retry_after_ms": 350, "quota_tier": "Developer"}
Tenants can initiate an instantaneous quota tier upgrade via the administrative API endpoint:
  POST /api/v1/tenant/upgrade?target_tier=Production
Upgrades take effect globally across all edge routing nodes within 15 seconds without cluster restarts.
```

#### **Document $D_3$: OmniFlow Python SDK: Streaming Client Configuration**
```markdown
# OmniFlow Python SDK Developer Manual: Client Lifecycle
The `OmniFlowClient` class encapsulates connection pooling, automatic token renewal, 
and resilient network fault tolerance for high-throughput batch ingest.
Key configuration parameters:
- `connection_pool_size`: Default is 32 HTTP/2 concurrent multiplexed sessions.
- `max_retries`: Maximum number of retry attempts for retryable HTTP status codes (default: 5).
- `retry_codes`: Tuple of HTTP error codes subject to exponential retry logic. 
  Defaults to (429, 502, 503, 504).
- `backoff_factor`: Multiplier for truncated exponential backoff delay calculation:
  sleep_time = min(backoff_factor * (2 ** (attempt - 1)) + jitter, max_backoff_seconds).
- `enable_adaptive_batching`: Dynamically resizes client-side micro-batches based on 
  observed round-trip latency (RTT) and downstream 429 response frequency.
```

#### **Document $D_4$: Research Excerpt: Late-Interaction & Dense-Sparse Alignment**
```markdown
# Dense-Sparse Alignment in Neural Information Retrieval: The ColBERT Paradigm
Traditional bi-encoder architectures map entire query and document sequences into isolated 
single-vector dense embeddings, creating an informational bottleneck that degrades retrieval 
precision on fine-grained lexical entities. Conversely, cross-encoders preserve all token-level 
interactions via all-to-all quadratic attention, yet exhibit an inference complexity of O(N * L^2), 
rendering top-K retrieval computationally intractable over massive corpora.
The ColBERT architecture introduces token-level late interaction: queries and documents are independently 
encoded into sequences of contextualized embeddings. Relevance score computation is formulated as:
  Score(Q, D) = \sum_{i \in Q} \max_{j \in D} (E_{q, i} \cdot E_{d, j})
This MaxSim operator allows pre-computing and indexing document token vectors offline, pruning candidate 
sets via inverted token centroids while retaining fine-grained cross-token matching semantics.
```

#### **Document $D_5$: Source Code: `omniflow/client.py`**
```python
import time
import random
from typing import Dict, Any, Optional
import requests

class OmniFlowClient:
    """Production client for OmniFlow streaming ingestion and query APIs."""
    def __init__(self, api_key: str, base_url: str, backoff_factor: float = 0.5, max_retries: int = 5):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.backoff_factor = backoff_factor
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def ingest_vectors(self, collection: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Ingests a vector micro-batch with exponential backoff on HTTP 429 rate limit."""
        url = f"{self.base_url}/api/v1/collections/{collection}/ingest"
        for attempt in range(1, self.max_retries + 1):
            response = self.session.post(url, json=payload)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                retry_after_ms = response.json().get("retry_after_ms", 250)
                jitter = random.uniform(0.01, 0.1)
                sleep_time = (retry_after_ms / 1000.0) + (self.backoff_factor * (2 ** (attempt - 1))) + jitter
                time.sleep(sleep_time)
            else:
                response.raise_for_status()
        raise TimeoutError(f"Exceeded max retries ({self.max_retries}) due to persistent rate limiting.")
```

#### **Document $D_6$: Incident Postmortem: OOM in HNSW Graph Construction**
```markdown
# Incident Postmortem: Incident 4092 — Worker OOM During Large Batch Ingestion
Date: 2024-03-14 | Severity: P1 | Root Cause: Memory Exhaustion in HNSW Graph Construction

Summary:
During a high-throughput vector backfill of 10,000,000 embeddings (dimension d=1536), multiple 
OmniFlow ingestion workers experienced Linux Out-Of-Memory (OOM) killer terminations on 64 GB RAM instances.

Root Cause Analysis:
1. Ingestion workers were provisioned with 64 GB RAM. Raw uncompressed vectors alone consumed:
   10,000,000 * 1536 * 4 bytes = 61.44 GB.
2. The ingestion collection was configured with HNSW parameters M=64 and efConstruction=200. At the base 
   layer (Layer 0), each vector node maintains up to 2 * M = 128 outgoing neighbor IDs using uint32 (4 bytes each):
   10,000,000 * 128 * 4 bytes = 5.12 GB.
3. Raw vectors plus Layer-0 graph edges alone totaled 61.44 GB + 5.12 GB = 66.56 GB, exceeding the 64 GB worker 
   memory ceiling before accounting for upper-layer edges, OS overhead, or active process buffers.
4. Compounding the memory pressure: ingestion workers slowed down under heavy allocation load, causing client requests to time out. Clients retried, rapidly exhausting the API gateway token bucket which returned HTTP 429 rate limits. Subsequent client retries created a thundering herd where each retried batch initiated a new, duplicate partial partition graph in worker RAM before uncompacted graphs could be reclaimed.

Resolution:
Reduced M to 16 (dropping Layer-0 edge overhead to 1.28 GB). However, because raw vectors alone (61.44 GB) plus M=16 edges and metadata still consume 62.84 GB—leaving under 1.2 GB of headroom on a 64 GB instance—the team sharded the collection across workers (e.g., 2 × 5M vectors ≈ 31 GB each) and/or evaluated storing compressed vectors in the graph (HNSW+PQ/SQ or DiskANN-style on-disk graph), while adding gateway token-bucket rate limiting and exponential backoff to prevent duplicate partition allocations.
```

---

### 1.2 The Running Queries & Baseline Failure Modes

| Query ID | Query Text | Query Archetype | Why Baseline / Naive RAG Fails |
| :--- | :--- | :--- | :--- |
| **$Q_1$** | `"retry_after_ms max_backoff_seconds OmniFlowClient"` | **Identifier-Heavy Search** | **Dense Bi-Encoders Fail**: Subword tokenization fragments rare identifiers (e.g., `retry_after_ms` $\to$ `retry`, `_after`, `_ms`; `OmniFlowClient` $\to$ `Omni`, `Flow`, `Client`), and single-vector pooling dilutes them across smooth latent dimensions. Dense retrieval returns generic client connection text rather than the exact source code in $D_5$ and configuration in $D_3$. |
| **$Q_2$** | `"Why did our servers crash while building the search index for a big data import?"` | **Conceptual / Vocabulary Mismatch** | **Sparse BM25 Fails**: Content words in $Q_2$ are: `servers`, `crash`, `building`, `search`, `index`, `big`, `data`, `import`. <br/>• In $D_1$: `search` (1), `index` (1), `data` (1) appear.<br/>• In $D_5$: `import` (4) appears.<br/>• In $D_6$: **zero** content words appear ($D_6$ uses `workers`, `OOM killer terminations`, `construction`, `HNSW`, `backfill`).<br/>Consequently, BM25 scores $D_1$ highest (Lucene: 5.25 / Robertson: 4.43) and ranks $D_1$ substantially above $D_6$ (score 0.00 on content words), completely missing the incident postmortem! |
| **$Q_3$** | `"How do enterprise quota tier upgrades interact with client retry configuration to resolve ingestion batch failures?"` | **Multi-Hop / Compositional Reasoning** | **Single-Step Retrieval Fails**: No single document contains the complete answer. Answering requires reasoning across $D_2$ (tier upgrades to Production/Enterprise), $D_3$ / $D_5$ (client retry logic on 429), and $D_6$ (thundering herd failure during backfills).<br/>*(Note: $D_3$ lists retry codes as `(429, 502, 503, 504)`, but $D_5$ only implements retries for status code `429`—an intentional cross-document inconsistency that multi-hop retrieval must reconcile).* |

---

## 2. Sparse Lexical Retrieval Constants (BM25)

### 2.1 Hyperparameters & Scoring Formulations
- **Standard Hyperparameters**: $k_1 = 1.5$, $b = 0.75$ (k1 = 1.5, b = 0.75 / k1=1.5, b=0.75).
- **Lucene Smoothed IDF Formulation** (Default):
  $$\text{IDF}_{\text{Lucene}}(q_i) = \ln \left( 1 + \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} \right)$$
  - Always non-negative ($\text{IDF} \ge 0$) for any document frequency $n(q_i) \le N$.
- **Robertson Unsmoothed IDF Formulation** (`rank_bm25` / academic baseline):
  $$\text{IDF}_{\text{Robertson}}(q_i) = \ln \left( \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} \right)$$
  - Becomes negative when a term appears in more than half the corpus ($n(q_i) > N/2$). `rank_bm25` floors negative values to $\epsilon \times \text{avg-idf}$.

### 2.2 Corpus Query BM25 Scores
- **Query $Q_2$ (Q2: `"Why did our servers crash while building the search index for a big data import?"`)**:
  - **Doc $D_1$**: **5.25** (Lucene IDF) / **4.43** (Robertson IDF). Matches content words `search` (1), `index` (1), `data` (1).
  - **Doc $D_6$**: **0.00**. Contains zero content words from $Q_2$ (uses `workers`, `OOM killer terminations`, `HNSW`, `backfill`).
  - *Outcome*: BM25 ranks $D_1$ substantially above $D_6$, failing completely on semantic synonymy and missing the incident postmortem.
- **Query $Q_1$ (Q1: `"retry_after_ms max_backoff_seconds OmniFlowClient"`)**:
  - **Doc $D_3$**: **3.1077** (matches `OmniFlowClient` and `max_backoff_seconds`).
  - **Doc $D_5$**: **2.7636** (matches `OmniFlowClient` and `retry_after_ms`).
  - **Doc $D_2$**: **1.0739** (matches `retry_after_ms`).
  - **Docs $D_1, D_4, D_6$**: **0.00** (contain zero matching terms; **must not appear in the sparse candidate list**).

### 2.3 Hand-Calculated Worked Example
- **Setup**: Corpus $N = 6$ documents, average document length $\text{avgdl} = 150\text{ tokens}$ *(illustrative)*.
- **Query Term**: Single term $q$ (`retry_after_ms`) appearing in $n(q) = 2$ documents ($D_2$ and $D_5$).
- **Candidates**:
  - Doc A ($D_2$): $|D_A| = 100\text{ tokens}$, $f(q, D_A) = 1$.
  - Doc B ($D_5$, source code): $|D_B| = 300\text{ tokens}$, $f(q, D_B) = 3$.
- **Step 1: Lucene IDF**:
  $$\text{IDF}(q) = \ln \left( 1 + \frac{6 - 2 + 0.5}{2 + 0.5} \right) = \ln(2.8) \approx \mathbf{1.0296}$$
- **Step 2: Length Penalty Factor $K(D) = 1 - b + b \cdot \frac{|D|}{\text{avgdl}}$**:
  - Doc A: $K(D_A) = 1 - 0.75 + 0.75 \cdot (100 / 150) = 0.25 + 0.50 = \mathbf{0.75}$
  - Doc B: $K(D_B) = 1 - 0.75 + 0.75 \cdot (300 / 150) = 0.25 + 1.50 = \mathbf{1.75}$
- **Step 3: Term Frequency Component**:
  - Doc A ($f=1$): $\frac{1 \cdot 2.5}{1 + 1.5 \cdot 0.75} = \frac{2.5}{2.125} \approx \mathbf{1.1765}$
  - Doc B ($f=3$): $\frac{3 \cdot 2.5}{3 + 1.5 \cdot 1.75} = \frac{7.5}{5.625} \approx \mathbf{1.3333}$
- **Step 4: Final BM25 Scores**:
  - $\text{Score}(D_A) = 1.0296 \cdot 1.1765 \approx \mathbf{1.211}$
  - $\text{Score}(D_B) = 1.0296 \cdot 1.3333 \approx \mathbf{1.373}$
  - **Score Delta**: Doc B holds an advantage of only $\approx \mathbf{13\%}$ ($13.33\%$, ~13%) over Doc A despite mentioning the term $3\times$ as often, demonstrating length penalty and asymptotic saturation.

---

## 3. Reciprocal Rank Fusion (RRF) Constants ($k = 60$)

### 3.1 Setup on Query $Q_1$
- **RRF Constant**: $k = 60$.
- **Scoring Formula**:
  $$\text{RRF}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
- **Dense Ranking** *(illustrative)*:
  $$[D_3, D_1, D_4, D_6, D_5, D_2]$$
- **Sparse Ranking** (from BM25 above):
  $$[D_3, D_5, D_2]$$
  *(Note: $D_1, D_4, D_6$ score 0.00 and are unranked in the sparse list).*

### 3.2 Canonical RRF Scores & Rankings
| Document | Dense Rank ($r_{\text{dense}}$) | Dense Component | Sparse Rank ($r_{\text{sparse}}$) | Sparse Component | Final RRF Score | Final Rank |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Doc $D_3$** (SDK Manual) | Rank 1 | $1 / (60 + 1) \approx 0.016393$ | Rank 1 | $1 / (60 + 1) \approx 0.016393$ | $\mathbf{0.032787}$ | **Rank 1** |
| **Doc $D_5$** (Client Code) | Rank 5 | $1 / (60 + 5) \approx 0.015385$ | Rank 2 | $1 / (60 + 2) \approx 0.016129$ | $\mathbf{0.031514}$ | **Rank 2** |
| **Doc $D_2$** (Rate Limits) | Rank 6 | $1 / (60 + 6) \approx 0.015152$ | Rank 3 | $1 / (60 + 3) \approx 0.015873$ | $\mathbf{0.031025}$ | **Rank 3** |
| **Doc $D_1$** (Architecture) | Rank 2 | $1 / (60 + 2) \approx 0.016129$ | Unranked | $0.0$ | $\mathbf{0.016129}$ | **Rank 4** |
| **Doc $D_4$** (ColBERT Paper) | Rank 3 | $1 / (60 + 3) \approx 0.015873$ | Unranked | $0.0$ | $\mathbf{0.015873}$ | **Rank 5** |
| **Doc $D_6$** (Postmortem) | Rank 4 | $1 / (60 + 4) \approx 0.015625$ | Unranked | $0.0$ | $\mathbf{0.015625}$ | **Rank 6** |

### 3.3 Margins and Takeaways
- $D_3$ legitimately wins (score $0.032787$) because it is the only document that defines `max_backoff_seconds`.
- $D_3$ beats $D_5$ by a margin of:
  $$\frac{0.032787 - 0.031514}{0.031514} \approx \mathbf{4.04\%}$$
- $D_5$ beats $D_2$ by a margin of:
  $$\frac{0.031514 - 0.031025}{0.031025} \approx \mathbf{1.58\%}$$
- **Architectural Impact**: Hybrid RRF successfully rescues Document $D_5$ (lifted from dense rank 5 into top-2) and Document $D_2$ (lifted from dense rank 6 into top-3), overcoming dense embedding dilution on exact identifiers.

---

## 4. Dense Retrieval & Contrastive Training Constants

### 4.1 Pooling Mechanisms
- **`BAAI/bge-large-en-v1.5`**: Uses **`[CLS]` token pooling**; requires query instruction prefix:
  `"Represent this sentence for searching relevant passages: "`
- **`intfloat/e5-large-v2` & SBERT**: Use **mean pooling** across contextualized token hidden states.

### 4.2 InfoNCE Loss Worked Example
- **Hyperparameters**: Temperature $\tau = 0.1$.
- **Similarities**:
  - Positive passage: $\langle \mathbf{q}, \mathbf{d}^+ \rangle = 0.85$
  - Easy negative: $\langle \mathbf{q}, \mathbf{d}_{\text{easy}}^- \rangle = 0.30$
  - Hard negative: $\langle \mathbf{q}, \mathbf{d}_{\text{hard}}^- \rangle = 0.65$
- **Scaled Logits & Exponentials**:
  - Positive: $\exp(0.85 / 0.1) = \exp(8.5) \approx \mathbf{4914.77}$
  - Easy negative: $\exp(0.30 / 0.1) = \exp(3.0) \approx \mathbf{20.09}$
  - Hard negative: $\exp(0.65 / 0.1) = \exp(6.5) \approx \mathbf{665.14}$
- **Softmax Denominator**:
  $$\sum = 4914.77 + 20.09 + 665.14 = 5599.998 \approx \mathbf{5600.0}$$
- **Positive Softmax Probability**:
  $$P(\text{positive}) = \frac{4914.77}{5600.0} \approx \mathbf{0.8776}$$
- **Final InfoNCE Loss**:
  $$\mathcal{L}_{\text{InfoNCE}} = -\ln(0.8776) \approx \mathbf{0.1306}$$
- **Hard Negative Mass Dominance**:
  $$\frac{665.14}{665.14 + 20.09} = \frac{665.14}{685.23} \approx \mathbf{97.07\%} \approx \mathbf{97\%}$$
  The single hard negative constitutes over $97\%$ of the total negative denominator mass, proving that gradient updates are driven almost entirely by hard negatives.

### 4.3 Contriever Unsupervised Contrastive Loss (TMLR 2022)
- **Setup**: Temperature $\tau = 0.1$, positive crop pair from $D_6$ similarity $0.80$, negatives from $D_1$ ($0.20$) and $D_4$ ($0.10$).
- **Scaled Logits**: $\exp(8.0) \approx 2980.96$, $\exp(2.0) \approx 7.39$, $\exp(1.0) \approx 2.72$.
- **Denominator**: $2980.96 + 7.39 + 2.72 = 2991.07$.
- **Final Loss**:
  $$\mathcal{L}_{\text{Contriever}} = -\ln \left( \frac{2980.96}{2991.07} \right) = -\ln(0.9966) \approx \mathbf{0.0034}$$
- **Status**: Historically important proof that dense retrievers can be trained without supervised pairs; in production use `contriever-msmarco`, `BAAI/bge-large-en-v1.5`, or `intfloat/e5-large-v2`.

---

## 5. Metric Geometry & Vector Distance Constants

### 5.1 Geometric Equivalence Theorem
On the unit hypersphere where $\|\mathbf{u}\|_2 = 1$ and $\|\mathbf{v}\|_2 = 1$:
$$\|\mathbf{u} - \mathbf{v}\|_2^2 = \|\mathbf{u}\|_2^2 + \|\mathbf{v}\|_2^2 - 2 \langle \mathbf{u}, \mathbf{v} \rangle = 1 + 1 - 2 \cos(\mathbf{u}, \mathbf{v}) = 2 - 2 \cos(\mathbf{u}, \mathbf{v})$$
$$\langle \mathbf{u}, \mathbf{v} \rangle = 1 - \frac{1}{2} \|\mathbf{u} - \mathbf{v}\|_2^2$$
- Maximizing Cosine Similarity, maximizing Dot Product, and minimizing Euclidean Distance are **strictly monotonically equivalent**.
- Sorting unit-normalized vectors by Dot Product yields the exact same top-K ranking as Euclidean Distance while executing up to $3\times$ faster.

### 5.2 Bachrach-Neyshabur MIPS-to-Euclidean Reduction
For unnormalized database vectors $\mathbf{x} \in \mathbb{R}^d$ bounded by maximum norm $\phi = \max_i \|\mathbf{x}_i\|_2$:
- Augment database vectors: $\mathbf{x}' = \left[ \mathbf{x}, \; \sqrt{\phi^2 - \|\mathbf{x}\|_2^2} \right] \in \mathbb{R}^{d+1} \implies \|\mathbf{x}'\|_2^2 = \phi^2$.
- Augment query vector: $\mathbf{q}' = [\mathbf{q}, \; 0] \in \mathbb{R}^{d+1}$.
- Metric expansion:
  $$\|\mathbf{q}' - \mathbf{x}'\|_2^2 = \|\mathbf{q}\|_2^2 + \phi^2 - 2 \langle \mathbf{q}, \mathbf{x} \rangle$$
- Minimizing Euclidean distance in $\mathbb{R}^{d+1}$ strictly maximizes unnormalized inner product $\langle \mathbf{q}, \mathbf{x} \rangle$.

---

## 6. Approximate Nearest Neighbor (ANN) Indexing Constants

### 6.1 Product Quantization (IVF-PQ) Constants
- **Vector Dimension**: $d = 1536$.
- **Coarse Voronoi Centroids**: $n_{\text{list}} = 4096$.
- **Subvectors**: $m = 96$ ($d^* = 1536 / 96 = 16\text{ dimensions per subvector}$).
- **Codebook Size**: $k^* = 256$ ($8\text{ bits} = 1\text{ byte per code}$).
- **Postings Entry Breakdown**:
  $$\text{Size per Entry} = 96\text{ B (PQ code)} + 8\text{ B (uint64 Doc ID)} = \mathbf{104\text{ bytes}}$$
  Each postings entry is exactly 104 bytes (104 B).
- **10M Vector Inverted Index Memory**:
  - Postings memory: $10{,}000{,}000 \times 104\text{ B} = 1{,}040{,}000{,}000\text{ B} = \mathbf{1.04\text{ GB}}$ (1.04 GB).
  - Coarse centroids: $4096 \times 1536 \times 4\text{ B} \approx 25.17\text{ MB} = 0.025\text{ GB}$.
  - Codebook centroids: $96 \times 256 \times 16 \times 4\text{ B} \approx 1.57\text{ MB} = 0.0016\text{ GB}$.
  - Total fixed overhead: $\approx \mathbf{0.027\text{ GB}}$ (0.027 GB).
  - Total Index Memory: $1.04\text{ GB} + 0.027\text{ GB} = \mathbf{1.067\text{ GB}}$ (1.067 GB).
  - **Compression Ratio vs. Raw `float32` ($61.44\text{ GB}$)**:
    $$\text{Compression} = \frac{61.44}{1.067} \approx \mathbf{57.6\times \text{ reduction in RAM}}$$
- **1M Vector Inverted Index Memory**:
  $$\text{Memory} = 0.104\text{ GB} + 0.027\text{ GB} = 0.131\text{ GB} \implies \frac{6.144}{0.131} \approx \mathbf{47\times \text{ reduction}}$$

### 6.2 HNSW & Document $D_6$ Incident Constants
- **Scale**: $N = 10{,}000{,}000$ vectors ($10\text{M}$), dimension $d = 1536$, format `float32` ($4\text{ B/dim}$).
- **Host Hardware Ceiling**: $64\text{ GB}$ physical RAM worker instances.
- **Raw Vector Footprint**:
  $$10{,}000{,}000 \times 1536 \times 4\text{ B} = 61{,}440{,}000{,}000\text{ B} = \mathbf{61.44\text{ GB}}$$
- **Original Configuration ($M = 64, efConstruction = 200$)**:
  - Layer-0 bidirectional edges per node: $2M = 128$ edges.
  - Neighbor ID storage: `uint32` ($4\text{ bytes}$ each) $\implies 128 \times 4\text{ B} = \mathbf{512\text{ bytes/node}}$.
  - Layer-0 total edge allocation: $10{,}000{,}000 \times 512\text{ B} = \mathbf{5.12\text{ GB}}$ (5.12 GB).
  - Base index memory:
    $$61.44\text{ GB} + 5.12\text{ GB} = \mathbf{66.56\text{ GB}} > 64\text{ GB physical RAM!}$$
    Total = 66.56 GB > 64 GB.
  - *Failure Mode*: Linux kernel OOM killer terminates worker process with signal 9.
- **Fixed Overhead Allocations**:
  - Upper-layer graph edges ($l \ge 1$): $\approx \mathbf{0.04\text{ GB}}$ (0.04 GB).
  - Node internal vector IDs (64-bit uint64): $10{,}000{,}000 \times 8\text{ B} = \mathbf{0.08\text{ GB}}$ (0.08 GB).
- **Remediated Configuration ($M = 16$)**:
  - Layer-0 edges per node: $2 \times 16 \times 4\text{ B} = 128\text{ bytes/node} \implies \mathbf{1.28\text{ GB}}$ (1.28 GB).
  - Total static index memory:
    $$61.44\text{ GB} + 1.28\text{ GB} + 0.04\text{ GB} + 0.08\text{ GB} = \mathbf{62.84\text{ GB}}$$
    Total index memory = 62.84 GB.
  - **Operating Headroom on 64 GB Host**:
    $$64.00\text{ GB} - 62.84\text{ GB} = \mathbf{1.16\text{ GB} \text{ headroom (insufficient!)}}$$
    Leaves only 1.16 GB headroom on a 64 GB host.
  - Leaving only $1.16\text{ GB}$ leaves zero operating margin for OS kernel caching, dynamic $efConstruction$ priority queues, or query execution buffers.
- **Production Remediations**:
  1. Distributed collection sharding (e.g., $2 \times 5\text{M}$ vectors $\approx 31.42\text{ GB}$ index footprint per worker).
  2. In-graph vector quantization (scalar SQ8 `int8` reducing raw vectors to $15.36\text{ GB}$, total index $\approx 16.76\text{ GB}$).
  3. DiskANN-style on-disk graph navigation.
- **Hyperparameter Architectural Roles**:
  - **M drives memory; efConstruction drives build time and quality**:
    - **$M$**: Drives **graph memory footprint** (number of outgoing edges per node).
    - **$efConstruction$**: Drives **index build time and graph connectivity quality** (candidate beam search size during construction). Does not alter static index memory.
    - **$efSearch$**: Dynamically trades **runtime query latency for retrieval recall** without rebuilding the index.
- **Level Multiplier Math**:
  - Level multiplier: $m_L = 1 / \ln(M)$.
  - Probability of reaching layer $l$: $P(\text{level} \ge l) = M^{-l}$.
  - Upper-layer node summation forms geometric series: $\sum_{l=1}^\infty N \cdot M^{-l} = \frac{N}{M - 1}$. For $M=16$, $\frac{N}{15} \approx 0.067N$ (Layer 0 accounts for $> 93\%$ of all graph memory).

---

## 7. Multi-Tier Vector Memory Scaling Table ($d = 1024$, Illustrative)

Exact computed memory footprints across data types at $100\text{M}$ and $1\text{B}$ vectors ($d = 1024$, assuming 8-byte uint64 doc ID and HNSW $M=16$ adding $\approx 132\text{ B/vector}$ [$128\text{ B Layer-0} + \approx 4\text{ B upper}$]):

| Scale ($N$) | Representation Format | Raw Bytes/Vec | Raw Vector Footprint | + Doc ID (8 B/vec) | + HNSW Graph ($M=16$, 132 B/vec) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **100M** | **`float32`** ($4\text{ B/dim}$) | $4096\text{ B}$ | **$409.60\text{ GB}$** | $410.40\text{ GB}$ | $423.60\text{ GB}$ |
| **100M** | **`float16` / `bfloat16`** ($2\text{ B/dim}$) | $2048\text{ B}$ | **$204.80\text{ GB}$** | $205.60\text{ GB}$ | $218.80\text{ GB}$ |
| **100M** | **`int8` (SQ8 Scalar Quantization)** | $1024\text{ B}$ | **$102.40\text{ GB}$** | $103.20\text{ GB}$ | $116.40\text{ GB}$ |
| **100M** | **`PQ64` ($m=64$, 1 B code)** | $64\text{ B}$ | **$6.40\text{ GB}$** | $7.20\text{ GB}$ | $20.40\text{ GB}$ |
| **100M** | **`Binary` ($1\text{ bit/dim}$)** | $128\text{ B}$ | **$12.80\text{ GB}$** | $13.60\text{ GB}$ | $26.80\text{ GB}$ |
| **1B** | **`float32`** ($4\text{ B/dim}$) | $4096\text{ B}$ | **$4096.00\text{ GB}$** ($4.10\text{ TB}$) | $4104.00\text{ GB}$ | $4236.00\text{ GB}$ |
| **1B** | **`float16` / `bfloat16`** ($2\text{ B/dim}$) | $2048\text{ B}$ | **$2048.00\text{ GB}$** ($2.05\text{ TB}$) | $2056.00\text{ GB}$ | $2188.00\text{ GB}$ |
| **1B** | **`int8` (SQ8 Scalar Quantization)** | $1024\text{ B}$ | **$1024.00\text{ GB}$** ($1.02\text{ TB}$) | $1032.00\text{ GB}$ | $1164.00\text{ GB}$ |
| **1B** | **`PQ64` ($m=64$, 1 B code)** | $64\text{ B}$ | **$64.00\text{ GB}$** | $72.00\text{ GB}$ | $204.00\text{ GB}$ |
| **1B** | **`Binary` ($1\text{ bit/dim}$)** | $128\text{ B}$ | **$128.00\text{ GB}$** | $136.00\text{ GB}$ | $268.00\text{ GB}$ |

---

## 8. Generation, Evaluation & Faithfulness Constants (Query $Q_3$ / Q3)

### 8.1 Faithful Answer Evaluation (Ragas Metric)
- **Extracted Claims (9 total)**:
  1. OmniFlow API access is rate-limited via token-bucket algorithm keyed on tenant credentials (supported by $D_2$).
  2. Developers can upgrade quota tier via `POST /api/v1/tenant/upgrade?target_tier=Production` taking effect in 15s (supported by $D_2$).
  3. Upgrading to Enterprise tier increases limit to 20,000 RPM (30,000 burst) (supported by $D_2$).
  4. Client `OmniFlowClient` implements exponential backoff on HTTP 429 status codes (supported by $D_5$).
  5. Backoff delay incorporates `retry_after_ms` from gateway payload, backoff factor, and jitter (supported by $D_5$).
  6. $D_3$ documents retry codes `(429, 502, 503, 504)` with `max_backoff_seconds` ceiling (supported by $D_3$).
  7. Client 5xx errors raise immediate unhandled exceptions in $D_5$ because $D_5$ calls `raise_for_status()` for non-429 codes (supported by $D_5$).
  8. Incident 4092 postmortem identified uncoordinated client retries creating thundering herds during memory pressure (supported by $D_6$).
  9. Higher throughput from quota upgrades intensifies memory pressure unless HNSW graph construction memory is remediated *(labeled: inference, outside retrieved context)*.
- **Scoring**:
  - **Strict Ragas Policy**: Extracts every claim regardless of author labeling: faithful 8/9 ≈ 0.889 strict.
    $$\text{Faithfulness} = \frac{8}{9} \approx \mathbf{0.889}$$
  - **Custom Exclusion Policy**: Explicitly configured to filter claims labeled as outside inferences: 8/8 custom exclusion.
    $$\text{Faithfulness} = \frac{8}{8} = \mathbf{1.00} \quad \text{(8/8 custom exclusion)}$$

### 8.2 Unfaithful Answer Evaluation
- **Extracted Claims (5 total)**:
  1. Rate limits return 429 status codes (supported: 1).
  2. Tier upgrades are performed via administrative API (supported: 1).
  3. Upgrading to Enterprise tier completely eliminates memory crashes during vector backfills (unsupported: 0; crashes caused by $M=64$ graph edges, not quota).
  4. Exponential backoff prevents thundering-herd retry storms (unsupported: 0; in $D_6$, retry storms originated from worker timeouts, not 429s).
  5. API gateway automatically rolls back failed batch ingestions (unsupported: 0; not stated anywhere in $D_1\text{--}D_6$).
- **Scoring**:
  $$\text{Faithfulness} = \frac{2}{5} = \mathbf{0.40} \quad \text{(unfaithful 2/5 = 0.40)}$$

---

## 9. Latency Budgeting & Percentile Mathematics

### 9.1 Non-Additivity of Percentiles
- **Mathematical Law**: Percentiles are not additive (percentiles and quantiles are not subadditive):
  $$p95(A + B) \neq p95(A) + p95(B)$$
- **Pessimistic Summation Principle**:
  - Sum of stage p95s is a pessimistic estimate, not a guaranteed bound (summing stage p95 latencies is a pessimistic estimate, not a guaranteed mathematical upper bound).
  - A query experiencing p95 tail latency in retrieval does not necessarily experience p95 tail latency in token generation.
- **Production Direct Measurement**:
  - Real pipeline SLAs must measure end-to-end trace latency percentiles directly in distributed telemetry (e.g., OpenTelemetry), rather than summing stage-level quantiles.
