# 1. My SDE Projects

---

## Project 1: OmniOS — Unified Campus Intelligence Platform

### 1. Description
* **One-Line Pitch**: An AI-powered campus intelligence platform that unifies multiple university services behind a single conversational interface using Model Context Protocol (MCP) microservices, Server-Sent Events (SSE), an Express reverse-proxy gateway, Next.js, and Supabase.
* **Problem Statement**: University digital ecosystems are fragmented—students must navigate separate, disjointed web portals for academic records, library catalogs, cafeteria menus, and campus event schedules. OmniOS unifies these isolated services into a single context-aware conversational agent capable of autonomous tool execution.
* **Tech Stack**:
  * **Backend**: Node.js, Express.js, Model Context Protocol (MCP) SDK, Server-Sent Events (SSE).
  * **Frontend**: Next.js (App Router), React, Tailwind CSS, NextAuth.js.
  * **Database & Storage**: Supabase (PostgreSQL), indexed SQL queries for low-latency retrieval.
  * **AI & Orchestration**: Google Gemini API (reasoning & dynamic tool-calling engine).
  * **Monorepo Architecture**: NPM Workspaces.

---

### 2. What I Did
* **Microservices & Monorepo Architecture**:
  * Designed an NPM Workspaces monorepo housing 5 independent domain microservices: `Academics`, `Library`, `Cafeteria`, `Events`, and `Memory`.
  * Preserved strict domain isolation so each service maintains its own dependencies, schemas, and business logic without cross-service contamination.
* **Reverse-Proxy Gateway**:
  * Implemented a Node.js/Express reverse-proxy gateway that dynamically boots the 5 microservices on isolated ports.
  * Consolidated routing, handled Cross-Origin Resource Sharing (CORS), unified authentication headers, and reverse-proxied incoming traffic (`/academics`, `/library`, `/cafeteria`, `/events`, `/memory`) to appropriate backends.
* **MCP Integration over SSE**:
  * Built communication layers using the Model Context Protocol (MCP) over Server-Sent Events (SSE).
  * Allowed the AI agent to dynamically discover available tools, inspect input schemas, and stream execution responses back to the user interface in real time.
* **AI Tool Orchestration via Gemini**:
  * Integrated Google Gemini as the central reasoning layer. Configured Gemini to interpret unstructured natural-language student queries, select the appropriate MCP tools, invoke microservices via the gateway, and synthesize multi-service responses.
* **Frontend & Context Persistence**:
  * Developed a responsive interface using Next.js App Router and React.
  * Implemented session authentication using NextAuth.js.
  * Structured chat history and student profile memory persistence in Supabase PostgreSQL, adding composite indexes on student and session IDs for sub-10ms context lookups.

#### Architecture Flow
```
Next.js Client (App Router)
       │ (REST / SSE)
       ▼
Express API Gateway (Reverse Proxy / Auth / Routing)
       │ (MCP over SSE)
 ┌─────┼─────────────┬─────────────┬─────────────┐
 ▼     ▼             ▼             ▼             ▼
Academics  Library   Cafeteria     Events        Memory
(MCP Svc)  (MCP Svc) (MCP Svc)     (MCP Svc)     (MCP Svc)
       │       │             │             │             │
       └───────┴─────────────┼─────────────┴─────────────┘
                             ▼
                Supabase / PostgreSQL + Gemini
```

#### Key Engineering Decisions & Trade-offs
* **MCP vs Conventional REST**: Conventional REST requires hardcoded API wrappers and manual schema translation inside the LLM prompt. MCP provides standardized tool metadata, self-describing schemas, and standardized event streaming, making services dynamically discoverable and plug-and-play.
* **SSE vs WebSockets**: SSE was chosen because MCP tool output is predominantly unidirectional streaming (server streaming tool logs and final results to the client). SSE operates natively over standard HTTP, handles network reconnects out-of-the-box, and eliminates WebSocket connection state management.
* **Reverse-Proxy Gateway vs Direct Frontend Access**: Exposing 5 standalone ports to the frontend creates severe CORS complications, security surface vulnerabilities, and tight coupling. The reverse proxy provides a single entry point, centralized logging, and internal network isolation.

---

### 3. What Can Be Asked
1. **Why did you use Model Context Protocol (MCP) instead of standard REST endpoints?**
   * *Answer*: MCP standardizes how AI agents discover tools, inspect schemas, and access external data resources. With REST, adding a new campus service requires writing manual tool definitions, custom prompt templates, and bespoke error parsers. With MCP, any microservice implementing the protocol automatically registers its capabilities with Gemini through a uniform contract.
2. **How does Gemini decide which MCP service to call, and how does routing work?**
   * *Answer*: When the application starts, the gateway aggregates the tool manifests from all 5 MCP servers and injects them into Gemini’s system declaration. When a user asks "Are there any vegan lunch options today?", Gemini returns a structured function call for `cafeteria.get_menu(type="vegan")`. The Express gateway routes this payload to the Cafeteria MCP server over SSE, awaits the result, and returns it to Gemini to formulate the final conversational response.
3. **Why Server-Sent Events (SSE) instead of WebSockets?**
   * *Answer*: SSE is lightweight, operates over standard HTTP/1.1 or HTTP/2, supports native browser auto-reconnection, and easily traverses firewalls and HTTP proxies. Since the agent interaction pattern is predominantly request-response followed by streamed token outputs, SSE avoids the stateful connection pooling and resource overhead required by full-duplex WebSockets.
4. **What happens if one of the MCP microservices crashes?**
   * *Answer*: Because the services run in isolated processes, a crash in the `Cafeteria` service does not impact `Academics` or `Library`. The gateway incorporates timeout handlers and circuit breakers: if a service fails to respond within 3 seconds, the gateway catches the error and returns a graceful fallback message to Gemini (e.g., "Cafeteria service is currently unreachable"), allowing the model to answer the rest of the user query uninterrupted.
5. **How would you scale this architecture to support multi-campus deployments?**
   * *Answer*:
     1. *Containerization*: Dockerize each MCP service and deploy them as auto-scaling pods on Kubernetes (EKS/GKE).
     2. *Asynchronous Broker*: Introduce Redis / RabbitMQ for event distribution and tool job queuing during peak registration hours.
     3. *Database Multi-Tenancy*: Use Supabase PostgreSQL Row-Level Security (RLS) or schema-per-tenant isolation keyed on `campus_id`.
     4. *Edge Ingress*: Place Cloudflare or AWS API Gateway in front of the reverse proxy for global edge caching and rate limiting.
6. **How do you secure student data across the MCP services?**
   * *Answer*: NextAuth verifies student identity and issues signed JWTs. The Express gateway validates the JWT signature, extracts the student ID and role claims, and forwards them in internal request headers to the MCP servers. PostgreSQL Row-Level Security (RLS) ensures students can only query their own academic transcripts, fines, and memory logs.

#### 30-Second Interview Recall
* **Core Pitch**: Monorepo campus intelligence assistant coordinating 5 decoupled microservices via an Express reverse proxy using MCP over SSE and Gemini.
* **Top Talking Points**: Microservice boundaries, MCP tool discovery, SSE vs WebSockets, decoupled fault tolerance, Supabase indexing, multi-tenant scaling.

---
---

## Project 2: Capture Hub — Event & Media Management Platform

### 1. Description
* **One-Line Pitch**: A cloud-native event photography platform engineered for high-volume media ingestion, direct-to-S3 uploads via pre-signed URLs, AI-powered auto-tagging, AWS Rekognition facial search, real-time social interactions, and 4-tier RBAC.
* **Problem Statement**: High-resolution event photography generates gigabytes of media that overwhelm traditional web backends during batch uploads. Furthermore, attendees face a massive discovery barrier when searching through thousands of unindexed event photos to find pictures of themselves.
* **Tech Stack**:
  * **Frontend**: Next.js, React, Tailwind CSS.
  * **Backend**: Python, FastAPI, SQLAlchemy, WebSockets.
  * **Cloud Storage & Vision AI**: AWS S3 (pre-signed URLs), AWS Rekognition.
  * **Database**: PostgreSQL (Supabase) with foreign-key constraints and cascading deletes.
  * **AI & Processing**: Google Gemini (semantic captioning/tagging), Pillow (dynamic image watermarking).
  * **Deployment & Auth**: JWT, Vercel (Frontend), Render (FastAPI).

---

### 2. What I Did
* **Decoupled Architecture & Domain Routing**:
  * Built a modular FastAPI backend split into isolated domain routers: `auth`, `events`, `media`, `social`, `ai`, and `face`.
* **High-Throughput S3 Pre-signed URL Media Pipeline**:
  * Architected a direct-to-cloud upload pipeline using AWS S3 pre-signed URLs.
  * Clients request an upload token $\rightarrow$ FastAPI verifies event permissions and generates a cryptographic short-lived pre-signed PUT URL $\rightarrow$ client uploads image binaries directly to S3.
  * Prevented large media files from passing through the backend, eliminating memory bloat, socket starvation, and bandwidth bottlenecks.
* **Asynchronous Background Processing**:
  * Implemented non-blocking worker pipelines using FastAPI `BackgroundTasks` for compute-heavy workloads (Gemini image tagging, Rekognition face indexing, and Pillow thumbnail generation), keeping API response latencies under 50ms.
* **Facial Recognition Search Engine**:
  * Integrated AWS Rekognition Collections to index facial vector embeddings from uploaded event photos.
  * Built a zero-friction "Find My Photos" feature: users upload a single selfie, and the system queries the Rekognition collection to instantly locate all matching photos across thousands of event uploads.
* **Real-Time Social Feed & Granular RBAC**:
  * Implemented full-duplex WebSockets for instant like/comment updates and live event notifications with deep links to media.
  * Engineered a strict 4-level Role-Based Access Control (RBAC) system (`Viewer` $\rightarrow$ `Photographer` $\rightarrow$ `Admin` $\rightarrow$ `Superuser`), enforcing authorization at both API middleware and database query layers to prevent Insecure Direct Object Reference (IDOR) vulnerabilities.
* **Dynamic Media Protection**:
  * Added dynamic image protection interceptors using Pillow: downloads are dynamically processed to stamp photographer watermarks and event branding before delivering assets.

#### Architecture Flow
```
Client (Next.js)
  │─── 1. Request Pre-signed URL ───► FastAPI Backend
  │◄── 2. Return Signed S3 PUT URL ──│
  │
  │─── 3. Direct Binary Upload ──────► AWS S3 Bucket
  │
  │─── 4. Confirm Upload ───────────► FastAPI Backend
                                            │
                                            ├──► Background Tasks
                                            │      ├──► AWS Rekognition (Face Vectors)
                                            │      └──► Gemini API (Tags/Captions)
                                            │
                                            └──► Supabase PostgreSQL (Metadata)
```

#### Key Engineering Decisions & Trade-offs
* **Pre-signed URLs vs Server Proxying**: Streaming multi-megabyte camera RAWs and JPEGs through FastAPI instances rapidly exhausts server memory (OOM) and monopolizes worker threads. Direct-to-S3 uploads offload 100% of the network bandwidth and file IO to AWS infrastructure.
* **Cloud AI vs Local Model Hosting**: Used AWS Rekognition and Gemini Cloud APIs rather than self-hosting YOLO/FaceNet models to remain within memory and CPU limits on containerized cloud instances (Render) while ensuring sub-second inference.
* **PostgreSQL vs MongoDB**: Selected PostgreSQL because event media platforms require strict relational integrity: cascading deletes on deleted events/albums, foreign-key relationships across user permissions, and ACID guarantees on comments/likes.

---

### 3. What Can Be Asked
1. **How do S3 pre-signed URLs work end-to-end, and how do you prevent abuse?**
   * *Answer*: The client sends a metadata request (`filename`, `filesize`, `event_id`). The backend validates that the user possesses `Photographer` or `Admin` privileges for that event, then uses the AWS Boto3 SDK to sign a temporary PUT URL using IAM credentials. The URL enforces:
     1. Expiration time (strictly 10 minutes).
     2. Exact S3 key path (`events/{event_id}/{uuid}.jpg`).
     3. Strict content-type restriction (`image/jpeg`, `image/png`).
     4. Content-length bounds (e.g., max 25MB).
     After uploading to S3, the client calls a confirmation endpoint where the backend verifies the object exists in S3 via a `HeadObject` call before creating the database row.
2. **What are the limitations of FastAPI `BackgroundTasks`, and how would you upgrade it for production?**
   * *Answer*: `BackgroundTasks` run in-process on the same ASGI event loop worker. If the server instance restarts, crashes, or is redeployed mid-task, in-flight background tasks are lost forever. For production at scale, I would decouple workers using an external message queue: publish upload events to an AWS SQS queue or Redis, and process them with dedicated Celery/ARQ worker pools or AWS Lambda functions.
3. **How does your RBAC model prevent IDOR (Insecure Direct Object Reference) vulnerabilities?**
   * *Answer*: An attacker might attempt to delete another photographer's media by guessing the photo UUID (`DELETE /media/{id}`). We prevent this by never executing queries solely on resource IDs. Every database query filters by both resource ID and caller identity:
     ```python
     # Enforce ownership or administrative rights
     stmt = select(Media).where(
         Media.id == media_id,
         or_(
             Media.owner_id == current_user.id,
             Media.event.has(Event.admin_id == current_user.id)
         )
     )
     ```
4. **How would you scale this platform to support 100,000 photos uploaded simultaneously at a music festival?**
   * *Answer*:
     * *Ingestion*: S3 automatically scales to handle thousands of concurrent PUT requests.
     * *Processing*: Configure S3 `ObjectCreated` events to trigger AWS SNS/SQS, feeding an auto-scaling group of Celery workers or serverless AWS Lambda functions.
     * *Facial Indexing*: Batch Rekognition `IndexFaces` API calls (up to 15 faces per image) and apply exponential backoff with jitter to adhere to AWS TPS limits.
     * *Database*: Direct writes to PostgreSQL primary and route read queries (galleries, feeds) to read replicas.
5. **Why WebSockets over Server-Sent Events (SSE) in this project?**
   * *Answer*: In Capture Hub, social engagement requires bi-directional real-time communication: clients send interactions (likes, comments, typing indicators) and receive instant live broadcasts. WebSockets provide persistent full-duplex TCP connections, avoiding the HTTP request overhead of frequent POST calls.

#### 30-Second Interview Recall
* **Core Pitch**: Cloud-native event photography platform utilizing direct S3 pre-signed uploads, FastAPI background workers, AWS Rekognition facial discovery, and real-time WebSockets.
* **Top Talking Points**: Bandwidth optimization via pre-signed URLs, async worker architecture, cloud vision integration, IDOR defense, relational database integrity.

---
---

## Project 3: SBM Traders — Business Dashboard & Financial Application (Internship)

### 1. Description
* **One-Line Pitch**: A production business management dashboard and financial auditing application for managing lending workflows, installment schedules, CIBIL credit calculations, and automated discrepancy detection.
* **Problem Statement**: Financial tracking and credit reporting across distributed business clients suffered from data discrepancies, complex repayment edge cases (partial payments, overdue interest, split installments), and database latency during report generation.
* **Tech Stack**:
  * **Framework**: Python, Django, Django REST Framework.
  * **Database**: PostgreSQL / SQLite, Django ORM.
  * **Data Processing**: Pandas, OpenPyXL, CSV ETL pipelines.
  * **Testing & Quality**: PyTest, Django Test Runner, synthetic test dataset generators.

---

### 2. What I Did
* **Financial & CIBIL Calculation Engine**:
  * Implemented deterministic business logic for loan amortization, interest accrual, late fees, installment deductions, and net outstanding calculations according to credit bureau (CIBIL) standards.
* **Edge-Case Dataset Engineering & Rigorous Testing**:
  * Instead of testing only standard repayment paths, built dedicated automated test suites and synthetic datasets covering difficult financial edge cases:
    * Unpaid and delinquent installment schedules.
    * Partial payments split across principal vs interest tranches.
    * Advance payments and early settlement amortizations.
    * Leap year and irregular calendar boundary date adjustments.
* **ETL & Spreadsheets Data Ingestion**:
  * Developed ETL validation scripts using Pandas and OpenPyXL to ingest, sanitize, and validate legacy Excel/CSV client ledgers, detecting malformed balances before database insertion.
* **Query Optimization & Performance Tuning**:
  * Profiled slow database queries in financial reporting views: eliminated Django ORM $N+1$ query bottlenecks using `select_related` and `prefetch_related`, and added composite database indexes on customer IDs and transaction timestamps.
* **Regression Testing & Audit Protection**:
  * Maintained an integration test suite validating that bug fixes and accounting rule updates never altered outputs for historically audited financial periods.

#### Architecture Flow
```
External CSV / Excel Ledgers
       │ (Pandas / OpenPyXL ETL & Validation)
       ▼
Django Backend Processing Service
       ├── Deterministic Financial Calculation Engine (Decimals)
       ├── CIBIL / Repayment Schedule Rules
       └── Edge-Case & Regression Test Suite (Automated Fixtures)
       ▼
PostgreSQL Database (Optimized Indexes & Foreign Keys)
       ▼
Business Dashboard & Financial Audit Reports
```

#### Key Engineering Decisions & Trade-offs
* **Decimal Precision vs Floating-Point**: Used Python's `decimal.Decimal` and Django's `DecimalField` exclusively for all financial figures. Never used standard IEEE 754 `float` types, preventing binary rounding anomalies ($0.1 + 0.2 \neq 0.3$) that corrupt balances over time.
* **Deterministic Rules vs Machine Learning**: Enforced strict, deterministic procedural logic rather than probabilistic heuristics, as financial audits, tax obligations, and statutory credit reporting demand 100% mathematical reproducibility.
* **Test Data Engineering Focus**: Emphasized building synthetic boundary datasets rather than just writing unit tests, ensuring that edge-case business logic was battle-tested against extreme real-world scenarios.

---

### 3. What Can Be Asked
1. **How did you translate ambiguous financial rules into robust code?**
   * *Answer*: I worked directly with business stakeholders to document lending terms into formal mathematical truth tables. I decoupled the business logic into pure, stateless functions (e.g., `calculate_overdue_penalty(principal, days_late, rate)`), verified them against hand-calculated ledger sheets using unit tests, and then integrated them into the Django models and service layer.
2. **How do you prevent floating-point rounding errors in financial software?**
   * *Answer*: Standard `float` types in Python and SQL use binary floating-point representation, which cannot represent base-10 fractions like $0.1$ exactly. In financial software, every calculation must use Python’s `decimal.Decimal` with an explicitly configured rounding context (e.g., `ROUND_HALF_EVEN` / Banker's rounding) and store values in the database as `DECIMAL(12, 2)` or as integer cents.
3. **How did you diagnose and resolve slow database operations in Django?**
   * *Answer*: I used Django Debug Toolbar and SQL logging to analyze query execution counts. I identified classic $N+1$ query problems where rendering a ledger list made 1 query for customers and $N$ additional queries for each customer's transactions. I resolved this using `select_related` for one-to-one/foreign-key joins and `prefetch_related` for many-to-many/reverse relations, reducing total database queries from over 200 to 2. Added composite indexes on `(account_id, payment_date)`.
4. **What is the difference between unit, integration, and regression testing in this application?**
   * *Answer*:
     * *Unit Testing*: Testing individual arithmetic calculations (e.g., verifying interest calculated on a 15-day delinquent installment).
     * *Integration Testing*: Testing the full pipeline (e.g., uploading an Excel ledger, verifying records persist in PostgreSQL, and confirming customer balance sheets update accurately).
     * *Regression Testing*: Running historical test cases to guarantee that a new feature or calculation bug fix does not silently alter historical statements or CIBIL scores.
5. **How would you prevent race conditions when two payments hit the same account simultaneously?**
   * *Answer*: Wrap the payment allocation in an atomic database transaction (`transaction.atomic()`) and employ pessimistic row-level locking using Django's `select_for_update()`:
     ```python
     with transaction.atomic():
         account = Account.objects.select_for_update().get(id=account_id)
         account.balance += payment_amount
         account.save()
     ```
     This locks the specific account row until the transaction commits, preventing dirty reads or lost updates.

#### 30-Second Interview Recall
* **Core Pitch**: Full-stack enterprise financial dashboard; turned complex business lending requirements into deterministic, auditable CIBIL/repayment calculation pipelines with zero tolerance for floating-point errors.
* **Top Talking Points**: `Decimal` arithmetic precision, solving ORM $N+1$ latency, automated edge-case test datasets, regression testing, pessimistic locking.

---
---

# 2. My Data Science Projects

---

## Project 1: Support Integrity Auditor (SIA) — NLP & Weak Supervision Ticket Auditor

### 1. Description
* **One-Line Pitch**: An NLP and weak-supervision auditing system that automatically detects priority misclassifications in customer-support tickets without ground-truth labels, combining DistilBERT, heuristic pseudo-labeling, and a deterministic explainability engine.
* **Problem Statement**: Enterprise support systems suffer from frequent priority mismatches (e.g., critical outages mistakenly filed as "Low Priority" or routine questions logged as "Urgent"). This results in breached SLAs and delayed incident response. However, historical support databases lack ground-truth "is_mismatch" labels, and manual annotation of tens of thousands of tickets is prohibitively expensive.
* **Tech Stack**:
  * **Modeling & NLP**: Python, PyTorch, Hugging Face Transformers (`distilbert-base-uncased`), Scikit-learn.
  * **Label Generation**: Weak Supervision, Rule-based labeling functions, Weighted heuristic fusion.
  * **Explainability**: Deterministic Evidence Engine (keyword extraction, metadata pattern verification).
  * **Dashboard & Deployment**: Streamlit, Hugging Face Spaces.

---

### 2. What I Did
* **Weak Supervision & Pseudo-Label Generation**:
  * Formulated a weak-supervision pipeline to synthesize training labels from two independent, orthogonal signals:
    1. *NLP Severity Signal*: Pattern matching for domain-specific severity cues, system outages, and urgent terminology (`outage`, `lawsuit`, `production down`, `urgent`).
    2. *Resolution-Time Anomaly Signal*: Compared a ticket's actual resolution duration against the historical median resolution time for its category.
  * Merged both signals via a weighted voting function to generate reliable binary pseudo-labels (`is_mismatch`).
* **DistilBERT Fine-Tuning**:
  * Fine-tuned `distilbert-base-uncased` on concatenated ticket text (subject + body) and metadata embeddings.
  * Evaluated generalization on a strictly held-out test split to ensure the model was learning contextual semantics rather than simply memorizing heuristic rules.
* **Threshold Optimization for Mismatch Recall**:
  * Avoided the default 0.5 classification threshold; calibrated the decision boundary targeting **Macro F1**.
  * Optimized the threshold to **0.42**, purposefully boosting the minority class (mismatched tickets) because failing to detect an incorrectly deprioritized critical ticket is far more costly than flagging a ticket for manual review.
* **Deterministic Evidence Engine for Explainability**:
  * Paired the neural model with a deterministic evidence engine to prevent generative hallucinations:
    * The DistilBERT model outputs the probabilistic prediction.
    * The evidence engine traces the decision back to explicit keywords, metadata attributes, and SLA thresholds, generating a transparent, structured JSON audit explanation for support leads.
* **Production Deployment**:
  * Built an interactive Streamlit auditing dashboard deployed on Hugging Face Spaces, supporting both real-time single-ticket evaluation and high-throughput batch CSV auditing.

#### Architecture Flow
```
Raw Support Tickets (Text + Metadata)
       │
       ▼
Weak Supervision Labeling Engine
   ├── Signal 1: NLP Severity Lexicons (Outage, Legal, Escalation)
   └── Signal 2: Category Resolution-Time Delta (Median Deviation)
       │ (Weighted Heuristic Fusion)
       ▼
Pseudo-Labeled Training Set (Binary `is_mismatch` Target)
       │
       ▼
DistilBERT Fine-Tuning (`distilbert-base-uncased`)
       │
       ▼
Threshold Optimization (Calibrated on Macro F1 $\rightarrow$ Threshold = 0.42)
       │
       ├──► Probabilistic Prediction (is_mismatch: 88.95% Recall)
       └──► Deterministic Evidence Engine (Structured JSON Explanation)
       ▼
Streamlit Audit Dashboard (Hugging Face Spaces)
```

#### Experimental Results
| Metric | Result | Operational Significance |
| :--- | :--- | :--- |
| **Accuracy** | **87.08%** | High overall classification accuracy |
| **Macro F1** | **86.39%** | Robust balance across both imbalanced classes |
| **Recall (Consistent)** | **84.08%** | Accurately verifies correctly prioritized tickets |
| **Recall (Mismatch)** | **88.95%** | Maximizes detection of high-risk misprioritized tickets |
| **Adversarial Edge Tests** | **10 / 10** | 100% pass rate on deliberate edge cases & contradictory text |

#### Key Engineering Decisions & Trade-offs
* **Weak Supervision vs Manual Annotation**: Manual labeling would have cost weeks of engineering and domain-expert time. Weak supervision generated thousands of high-confidence training targets in minutes.
* **DistilBERT vs Traditional TF-IDF / XGBoost**: Traditional n-gram/TF-IDF models fail on contextual nuance and negation ("this is not an outage" vs "we have an outage"). DistilBERT captures bidirectional context while running 60% faster than full BERT.
* **Separation of Prediction and Explanation**: Instead of asking an LLM to generate natural-language explanations (which risk hallucinating unsupported reasons), used a deterministic rule engine to provide verifiable JSON evidence.

---

### 3. What Can Be Asked
1. **Why was weak supervision necessary, and what are its main risks?**
   * *Answer*: Support datasets contain raw tickets and assigned priority, but no historical ground truth on whether that priority was objectively accurate. Weak supervision synthesizes labels programmatically using domain rules. The risk is label noise: if heuristics contain biases or are strongly correlated, the model risks learning and reinforcing those heuristic shortcuts.
2. **Does using resolution time as a feature introduce target leakage?**
   * *Answer*: If resolution time were an input feature during inference, it would be severe target leakage because a ticket's resolution duration is unknown when it is opened. In this project, resolution time was used *strictly as a weak supervision signal during offline training to create pseudo-labels*. The downstream DistilBERT model itself only receives features available at ticket creation time (subject, body, category, user tier), completely eliminating inference-time data leakage.
3. **Why did you choose DistilBERT over standard BERT or TF-IDF + Logistic Regression?**
   * *Answer*: TF-IDF lacks semantic context, word order awareness, and struggles with negation and synonyms. Standard BERT-base has 110M parameters and requires higher inference latency. DistilBERT retains 97% of BERT's language understanding while having 40% fewer parameters (66M) and running 60% faster, making it optimal for real-time ticket ingestion pipelines.
4. **Why optimize the classification threshold to 0.42 instead of using the standard 0.5?**
   * *Answer*: In enterprise support, the cost matrix is asymmetric: a False Negative (missing an incorrectly deprioritized outage) causes SLA breaches and angry customers, whereas a False Positive (auditing a ticket that was actually fine) costs only a brief manual glance. Calibrating the threshold against Macro F1 shifted the decision boundary to 0.42, boosting Mismatch Recall to 88.95%.
5. **How does your deterministic evidence engine compare to SHAP or LIME?**
   * *Answer*: SHAP and LIME compute local feature importance through perturbation, which is computationally expensive for real-time APIs and can produce unstable explanations across minor perturbations. The deterministic evidence engine performs exact keyword and metadata rule validation, generating auditable, reproducible JSON proofs that human auditors can immediately trust.
6. **How would you extend this system if you obtained 5,000 manually verified ground-truth labels?**
   * *Answer*: Adopt semi-supervised learning or active learning. Train a baseline model on the 5,000 gold labels, evaluate confidence calibration, and use the model's predictive entropy to select the most uncertain unlabeled tickets for human review. Fine-tune using a multi-task loss combining cross-entropy on gold labels with consistency regularization on the remaining weak-supervised data.

#### 30-Second Interview Recall
* **Core Pitch**: NLP auditing system detecting support ticket priority mismatches without ground-truth labels using weak supervision (keywords + resolution time), fine-tuning DistilBERT to achieve 88.95% mismatch recall with a deterministic explainability engine.
* **Top Talking Points**: Weak supervision pipeline, avoiding target leakage, Macro F1 threshold tuning, transformer context vs TF-IDF, deterministic evidence vs hallucination.

---
---

# 3. My Paper

---

## Paper: Multiscale Diagnostics of Vision-Language Models

### 1. Description
* **Research Focus & Motivation**: A comprehensive empirical research study investigating how object scale (relative bounding-box area) impacts zero-shot recognition capability across Vision-Language Models (VLMs). The study evaluates six model architectures across three distinct visual domains to prove that standard aggregate benchmark scores conceal severe, systematic failure modes on small objects.
* **Core Research Question**: *How does the relative physical size of an object within an image affect zero-shot VLM classification, and how do architecture family, cross-modal attention mechanisms, and training paradigms govern this scale sensitivity?*
* **Models Evaluated (6 VLMs across 3 Architectural Paradigms)**:
  * *Contrastive Models*: CLIP ViT-B/32, OpenCLIP RN50, SigLIP.
  * *Generative / Autoregressive Vision-Language*: SmolVLM-256M.
  * *Cross-Modal Fusion / Attention Models*: BLIP-VQA Base, ViLT-B32.
* **Datasets (3 Visually Diverse Domains)**:
  * *PASCAL VOC*: 20-class multi-object detection benchmark.
  * *African Wildlife*: 4-class natural habitat domain (buffalo, elephant, rhinoceros, zebra).
  * *Vehicles-OpenImages*: Dense, bimodal vehicle scale distribution.

---

### 2. What I Did
* **Scale Normalization & Discretization Methodology**:
  * Defined normalized object scale as the ratio of bounding-box area to total image area:
    $$\text{Scale} = \frac{\text{Bounding Box Area}}{\text{Total Image Area}} \times 100\%$$
  * Discretized objects into five standardized logarithmic scale bins:
    * **Tiny**: $< 1\%$ of image area
    * **Small**: $1\% - 5\%$ of image area
    * **Medium**: $5\% - 15\%$ of image area
    * **Large**: $15\% - 40\%$ of image area
    * **Huge**: $\ge 40\%$ of image area
* **Context Isolation via Non-Target Surgical Masking**:
  * In standard photos, models frequently identify small objects indirectly through surrounding context (e.g., recognizing a tiny boat because it rests on water).
  * Implemented an automated surgical masking pipeline: all non-target objects and background context were masked/blackened prior to inference, forcing models to recognize objects strictly from their intrinsic visual features.
* **Zero-Shot Evaluation Protocol**:
  * Formulated standardized zero-shot classification prompt templates across all 6 models, calculating classification accuracy independently within each scale bin.
* **Key Findings & Empirical Results**:
  * **Severe Scale Sensitivity in Contrastive Models**: CLIP ViT-B/32 exhibited a dramatic 62.6 percentage-point accuracy collapse on PASCAL VOC:
    * *Huge objects*: **81.5%** accuracy.
    * *Tiny objects*: **18.9%** accuracy.
  * **Scale Invariance in Generative Models**: SmolVLM demonstrated remarkable scale stability:
    * Maintained **98.5% – 99.7%** accuracy across all scale bins on PASCAL VOC ($< 1.2\%$ variance across tiny to huge objects).
    * Exhibited similar scale stability across African Wildlife and Vehicles-OpenImages.
  * **Parameter Count Does Not Guarantee Scale Robustness**:
    * SmolVLM (256M parameters) drastically outperformed larger models like BLIP (385M parameters) in small-object robustness, showing that cross-modal token interaction design matters far more than parameter scale.
* **Theoretical Contribution**:
  * Demonstrated that contrastive vision encoders suffer from global visual representation bias caused by spatial pooling and patch token downsampling ($14 \times 14$ or $32 \times 32$ patches lose tiny object signatures). Real-world VLM benchmarking must mandate scale-stratified evaluation rather than relying solely on aggregate metrics.

#### Scale Evaluation Pipeline
```
Annotated Dataset (PASCAL VOC / Wildlife / Vehicles)
       │
       ▼
Scale Calculation: Scale = Area(BBox) / Area(Image)
       │
       ▼
Scale Bin Discretization (Tiny <1% | Small 1-5% | Med 5-15% | Large 15-40% | Huge ≥40%)
       │
       ▼
Non-Target Surgical Masking (Eliminates Background/Context Shortcuts)
       │
       ▼
Zero-Shot Model Inference (CLIP, OpenCLIP, SigLIP, SmolVLM, BLIP, ViLT)
       │
       ▼
Scale-Stratified Accuracy & Sensitivity Analysis
```

---

### 3. What Can Be Asked
1. **Why divide bounding-box area by total image area instead of using absolute pixel dimensions ($W \times H$)?**
   * *Answer*: Raw pixel counts vary drastically across camera sensors and datasets ($500 \times 375$ vs $4000 \times 3000$). Dividing by total image area provides a scale-invariant metric that reflects what fraction of the model's visual input and receptive field (patch tokens) is occupied by the target object.
2. **Why mask non-target objects, and doesn't masking introduce an artificial distribution shift?**
   * *Answer*: Masking was necessary to isolate intrinsic scale robustness from contextual shortcuts. If an image contains a tiny airplane in the sky, a model might predict "airplane" simply by detecting the blue sky. Masking removes this confounding variable. While masking introduces artificial boundary edges, the transformation was applied uniformly across all 6 models, ensuring an unbiased comparative baseline.
3. **Why do contrastive architectures like CLIP perform poorly on small objects?**
   * *Answer*: ViT patch tokenization decomposes images into fixed $32 \times 32$ or $16 \times 16$ patches. A tiny object occupying $< 1\%$ of the image might fall into a single patch or be split across patch boundaries. Furthermore, CLIP's InfoNCE contrastive objective aligns a single pooled global image embedding with text, training the visual backbone to prioritize dominant, high-area visual patterns while averaging out localized signals.
4. **Why did SmolVLM (256M) demonstrate near-perfect scale invariance while BLIP (385M) struggled, despite having fewer parameters?**
   * *Answer*: SmolVLM uses a multi-modal autoregressive architecture with dynamic high-resolution visual tiling and fine-grained cross-attention, allowing localized visual tokens to interact directly with text tokens without coarse global pooling. BLIP's contrastive pretraining phase retains global feature bias. This proves that cross-modal token interaction design is more critical to scale robustness than sheer parameter count.
5. **SmolVLM achieves ~99% accuracy across all scale bins. Is that result suspiciously high, and how would you audit it?**
   * *Answer*: That is a critical defense question. A near-perfect score across tiny objects warrants investigation into potential pretraining data contamination (PASCAL VOC images present in the model's web-scraped training corpus). To audit this:
     1. Evaluate on a freshly captured, unreleased dataset with guaranteed zero contamination.
     2. Inspect attention rollout / Grad-CAM attribution maps to verify that SmolVLM's cross-attention genuinely focuses on the tiny object rather than masking boundary artifacts.
     3. Progressively downsample the images to determine the physical resolution floor where the model's performance finally breaks down.
6. **What practical recommendations does your paper offer to machine learning engineers deploying VLMs?**
   * *Answer*: Never deploy VLMs in production (e.g., autonomous driving, surveillance, aerial imagery) based on aggregate benchmark accuracy alone. Teams must perform scale-stratified evaluations. If small-object recognition is critical, engineers should favor models with dense cross-modal attention or dynamic patch tiling (e.g., SmolVLM) over purely contrastive pooled encoders (e.g., standard CLIP).

#### 30-Second Research Pitch
* **Core Pitch**: Benchmarked 6 VLMs across 3 datasets to prove aggregate benchmarks conceal a 62.6 percentage-point accuracy collapse on small objects in CLIP-style models, while generative architectures like SmolVLM retain scale robustness due to cross-modal token interaction rather than raw parameter count.
* **Top Talking Points**: Scale normalization metric, contextual masking methodology, contrastive pooling vs cross-attention dynamics, critical audit of near-ceiling metrics, real-world deployment recommendations.