# Capture Hub / CIG Project — Interview Prep Dossier

**Repository:** https://github.com/divyansh070/cig_project_dev  
**Project:** Capture Hub — Event & Media Management Platform  
**Primary track:** SDE / Full-Stack / Backend

## 1. One-Line Summary

Capture Hub is a full-stack event photography/media platform for large media uploads, cloud storage, AI-powered tagging and face search, event-level RBAC, social interactions, real-time notifications, and protected media downloads.

The strongest interview framing is:

> I separated application compute, media storage, AI processing, authorization, and real-time communication so the API server would not become the bottleneck.

---

## 2. Problem

The system addresses several problems common to event photography:

- Thousands of high-resolution images/videos are expensive to route through an application server.
- Finding a particular person in a large gallery is difficult.
- Manual media tagging does not scale.
- Different users need different permissions within the same event.
- Photographers need protection against unrestricted raw-image downloads.
- Likes/comments should generate low-latency notifications.
- AI processing should not block normal HTTP requests.
- The deployment environment has limited backend memory/compute.

The repository describes a 512 MB deployment constraint and explicitly designs around it. [GitHub README](https://github.com/divyansh070/cig_project_dev)

---

## 3. Architecture

```text
                         Browser
                            |
                         HTTPS
                            v
                  +-------------------+
                  | Next.js / React   |
                  | Tailwind / UI     |
                  +---------+---------+
                            |
                       REST / JSON
                            v
                  +-------------------+
                  | FastAPI Backend   |
                  | Auth + RBAC       |
                  | Events / Media    |
                  | Social / AI / Face|
                  | WebSocket Manager |
                  +----+---------+----+
                       |         |
                  SQLAlchemy    Async Work
                       |         |
                       v         v
                +----------+  +----------------+
                | Postgres |  | Background     |
                | Supabase |  | Processing     |
                +----------+  +-------+--------+
                                     |
                          +----------+----------+
                          |          |          |
                          v          v          v
                         S3       Gemini    Rekognition
                       Storage    Vision       Faces
```

The repository's architecture document describes this same separation of frontend delivery, FastAPI compute, PostgreSQL persistence, S3 object storage, Gemini, Rekognition, WebSockets, and background workers.

[Architecture](https://github.com/divyansh070/cig_project_dev/blob/main/architecture.md)

---

# 4. The Most Important Design Decision: S3 + Pre-Signed URLs

Do **not** route large media through FastAPI.

### Bad

```text
Client -> FastAPI -> S3 -> FastAPI -> Client
```

The backend becomes a bandwidth bottleneck.

### Project approach

```text
Client -> FastAPI
             |
             | authorize + generate signed URL
             v
           S3
             |
             v
           Client
```

PostgreSQL stores metadata; S3 stores large binary objects.

The repository explicitly describes pre-signed URLs as a way to prevent the constrained Render instance from becoming a media-bandwidth bottleneck.

### Interview answer

> “I treated FastAPI as the control plane rather than the media data plane. It authenticates the request, checks authorization, and generates a temporary S3 pre-signed URL. The browser then transfers the large object directly with S3.”

### What is a pre-signed URL?

A temporary, scoped URL that grants permission to perform a particular storage operation without exposing permanent AWS credentials.

### Important security nuance

A pre-signed URL does **not** replace application authorization.

Correct sequence:

```text
Authenticate
    ↓
Authorize resource
    ↓
Generate short-lived signed URL
    ↓
Return URL
```

---

# 5. Upload Pipeline

Conceptually:

```text
User selects media
        ↓
FastAPI validates request
        ↓
Media registered / uploaded
        ↓
Metadata stored
        ↓
Background processing
        ├── Gemini semantic tags
        └── Rekognition face indexing
```

The key principle is:

> Uploading media should not synchronously wait for every expensive AI operation.

The README says S3 uploads and AI processing are dispatched to asynchronous background processing.

---

# 6. Why Asynchronous Processing?

A naive request might do:

```text
POST /upload
  ↓
save file
  ↓
Gemini
  ↓
Rekognition
  ↓
save everything
  ↓
response
```

This makes HTTP latency depend on external AI services.

The project instead separates the request path:

```text
HTTP request
    ↓
validate / register
    ↓
dispatch work
    ↓
return

Background:
    ↓
AI / indexing / enrichment
```

### Strong interview answer

> “The goal was to keep the synchronous request path short. AI inference and other expensive work are asynchronous so external service latency doesn't unnecessarily block the API.”

### Important limitation

FastAPI `BackgroundTasks` is not the same as a durable distributed queue.

If the system grows substantially, I'd use:

```text
FastAPI → SQS/RabbitMQ/Kafka → Workers
```

with retries, dead-letter queues and independently scalable workers.

---

# 7. AI Pipeline

## Semantic Tagging

Uploaded images can be processed by Gemini Vision.

Example:

```text
image
  ↓
Gemini
  ↓
["outdoor", "sunset", "wedding", "bride"]
  ↓
media metadata
```

This enables metadata-based discovery.

Videos are treated separately; the README says they bypass the image-oriented AI path and receive a default `video` tag.

## Why cloud inference?

Because the backend environment is memory constrained.

Instead of:

```text
Render → load large vision model → inference
```

the project uses:

```text
Render → API call → managed AI service
```

Benefits:

- no local GPU requirement
- lower application memory pressure
- simpler deployment
- managed inference

Trade-offs:

- API cost
- network latency
- vendor dependency
- external service availability
- data/privacy considerations

---

# 8. “Find Myself” Face Search

The platform allows a user to upload a selfie and find photographs containing them.

Conceptually:

```text
Selfie
  ↓
FastAPI
  ↓
AWS Rekognition
  ↓
face matching
  ↓
matching media
  ↓
authorization
  ↓
media URLs
```

The repository uses AWS Rekognition rather than training/deploying its own face-recognition model.

### Important edge case

The README notes that Rekognition's inference path expects JPEG/PNG, so WebP assets need filtering or preprocessing before indexing.

This is a useful interview example of handling an external API's input contract.

---

# 9. Database Design

Core tables:

```text
Users
Events
Event_Roles
Media
Likes
Comments
```

Relationship:

```text
Users
  |
  +---- Events
  |
  +---- Event_Roles ---- Events
  |
  +---- Media ---------- Events
  |
  +---- Likes ---------- Media
  |
  +---- Comments ------- Media
```

The schema document defines these relationships and uses SQLAlchemy with PostgreSQL.

[Database schema](https://github.com/divyansh070/cig_project_dev/blob/main/database_schema.md)

---

# 10. Why Event_Roles Matters

A global:

```text
User.role = Photographer
```

is insufficient.

The same user may be:

```text
Event A → Photographer
Event B → Viewer
Event C → no access
```

Therefore:

```text
Event_Roles
-----------
user_id
event_id
role
```

This makes permissions resource/event scoped.

### Strong interview statement

> “Authentication answers who the user is. Authorization answers whether that user can access this particular event or media object.”

---

# 11. RBAC

The repository describes:

### Viewer
- view authorized/public content
- like/comment

### Photographer
- upload media
- only to assigned events

### Admin
- event-level management/moderation

### Superuser
- platform-level developer/owner access

Do not describe this as a simple global role system. The interesting part is the event-scoped relationship.

---

# 12. IDOR — Know This Extremely Well

IDOR = Insecure Direct Object Reference.

Bad:

```http
GET /media/123
```

and the server only checks:

```text
user is logged in
```

An attacker can try:

```text
/media/124
/media/125
...
```

and potentially access another event's private media.

Correct authorization:

```text
authenticated
    AND
user has access to media.event_id
```

### Interview answer

> “I don't authorize only at the endpoint level. The authorization decision needs to be tied to the resource's event and the user's event role.”

This is one of the strongest security topics in the project.

---

# 13. JWT Authentication

Conceptually:

```text
Login
  ↓
verify credentials
  ↓
signed JWT
  ↓
client sends token
  ↓
server verifies token
  ↓
resource authorization
```

JWT makes authentication stateless, which simplifies horizontal scaling.

But JWT does **not** automatically provide authorization or complete security.

Production considerations:

- signing-key protection
- expiration
- appropriate claims
- secure transport
- token storage
- revocation strategy where necessary

---

# 14. Dynamic Watermarking

Normal media delivery:

```text
Client → S3
```

Protected image download:

```text
Download request
      ↓
authorize
      ↓
retrieve image
      ↓
Pillow adds watermark
      ↓
return transformed image
```

The watermark contains event/photographer information.

This protects the photographer's original asset from being the default raw download.

### Trade-off

Dynamic processing consumes CPU/memory, so it should be limited to protected download operations rather than every gallery view.

---

# 15. WebSocket Notifications

The platform supports likes/comments and real-time notifications.

```text
User A likes photo
       ↓
FastAPI
       ↓
DB update
       ↓
WebSocket
       ↓
User B browser
       ↓
toast notification
```

### Why WebSockets?

Polling:

```text
GET notifications
GET notifications
GET notifications
...
```

WebSocket:

```text
Client ← persistent connection → Server
```

The server can push an event immediately.

### WebSocket trade-offs

Pros:
- low latency
- persistent channel
- bidirectional communication

Cons:
- connection management
- reconnect handling
- horizontal scaling complexity

At large scale, a shared pub/sub layer such as Redis can help distribute events across WebSocket instances.

---

# 16. Deep Linking

Notifications use a media identifier such as:

```text
?photo=482
```

so clicking the notification can open the exact media rather than merely the event page.

This is a small feature but demonstrates product-level thinking:

> Real-time notifications should reduce navigation friction, not just display information.

---

# 17. Three Flows to Memorize

## Flow A — Upload

```text
Browser
 ↓
FastAPI
 ↓
authorization
 ↓
S3 / metadata
 ↓
background processing
 ├── Gemini
 └── Rekognition
```

## Flow B — Gallery

```text
Browser
 ↓
FastAPI
 ↓
authorization
 ↓
PostgreSQL metadata
 ↓
pre-signed URL
 ↓
Browser → S3
```

## Flow C — Find Myself

```text
Selfie
 ↓
FastAPI
 ↓
Rekognition
 ↓
face matches
 ↓
authorized media
 ↓
S3 URLs
```

If you can draw these three from memory, you can explain most of the project.

---

# 18. Scaling Question: “How Would You Scale This?”

Current architecture:

```text
Vercel
   ↓
FastAPI
   ↓
PostgreSQL
   ↓
S3
```

At larger scale:

```text
                 Load Balancer
                       |
          +------------+------------+
          |            |            |
       API #1       API #2       API #N
          |            |            |
          +------------+------------+
                       |
              +--------+--------+
              |                 |
           Postgres           Redis
              |
            Queue
              |
      +-------+-------+
      |       |       |
   Worker  Worker  Worker
      |       |       |
    Gemini Rekognition S3
```

Potential improvements:

- horizontal API scaling
- durable queue
- independent workers
- Redis caching/pub-sub
- CDN
- database indexes
- read replicas
- rate limiting
- observability

---

# 19. What Breaks First?

### Database

Potential pressure from media metadata, likes/comments and event queries.

Solutions:

- correct indexes
- connection pooling
- caching
- read replicas
- partitioning if needed

### Background processing

Millions of media objects can create a huge AI workload.

Solutions:

- durable queue
- worker pool
- retries
- dead-letter queue

### WebSockets

Many persistent connections require horizontal scaling and shared event distribution.

Solution:

```text
WebSocket servers + Redis/pub/sub
```

### Search

Simple PostgreSQL metadata search may become insufficient for huge collections.

Potential solution:

```text
PostgreSQL indexes → simple queries
OpenSearch/Elasticsearch → large-scale search
```

---

# 20. Database Indexes

Likely useful indexes include:

```text
Media.event_id
Media.uploader_id
Likes.media_id
Comments.media_id
Event_Roles.event_id
Event_Roles.user_id
Events.creator_id
```

If queries frequently do:

```text
WHERE event_id = ?
ORDER BY upload_date
```

a composite index could be appropriate:

```text
(event_id, upload_date)
```

### Interview principle

> Index based on actual query patterns, not simply every column.

---

# 21. Concurrency: Likes

Two requests may try to create the same like simultaneously.

A strong design can enforce:

```text
UNIQUE(media_id, user_id)
```

at the database layer.

This is safer than relying only on:

```python
if not already_liked:
    create_like()
```

because two concurrent requests can pass that check.

This is a great example of why database constraints matter.

---

# 22. Failure Handling

## Gemini unavailable

Core media should ideally remain usable:

```text
upload succeeds
AI enrichment fails
job retries later
```

AI enrichment should not necessarily determine whether the media itself exists.

## Rekognition unavailable

Similarly:

```text
media exists
face-indexing job pending/failed
retry later
```

## S3 unavailable

More serious because S3 is part of core media storage.

Need:

- retry transient failures
- clear upload state
- avoid claiming successful persistence before confirmation
- consistent DB/object-storage lifecycle

---

# 23. Security Checklist

Know these:

### Authentication
JWT

### Authorization
RBAC + event/resource checks

### Object storage
Pre-signed URLs

### Database
Foreign keys + constraints

### Asset protection
Dynamic watermarking

### Secrets
Environment variables

### Future hardening
- rate limiting
- file-size limits
- MIME validation
- malware scanning
- upload quotas
- short URL expiry
- audit logs

Do not claim future hardening features are already implemented unless you have verified them in the code.

---

# 24. Important Security Nuance: Signed URLs

If asked:

> “Why are signed URLs secure?”

Do not answer:

> “Because they prevent unauthorized access.”

Better:

> “They provide temporary scoped access to an object. The application still has to perform authorization before issuing the URL, and the URL should have a short appropriate expiry.”

---

# 25. Important Limitation: BackgroundTasks

If asked:

> “Why is this scalable?”

Do not say:

> “BackgroundTasks makes the system distributed.”

Better:

> “It removes expensive work from the request path. For higher workloads I would replace in-process background tasks with a durable queue and independent workers.”

This demonstrates that you understand the boundary of the current design.

---

# 26. Testing Strategy

## Unit tests

Test:

```text
role logic
permission checks
validation
watermarking
metadata parsing
```

## Integration tests

Test:

```text
API + database
authentication
event permissions
media lifecycle
```

## External services

Mock:

```text
S3
Gemini
Rekognition
```

Don't make every test depend on live cloud APIs.

## End-to-end

Example:

```text
login
→ create event
→ assign photographer
→ upload image
→ image becomes available
→ viewer sees it
→ unauthorized user is denied
```

---

# 27. Observability You Would Add

### API
- latency
- throughput
- error rate
- status codes

### Background jobs
- queue depth
- processing latency
- failure/retry count

### AI
- Gemini latency
- Rekognition latency
- failures
- usage/cost

### WebSockets
- active connections
- disconnects
- delivery failures

This is a strong answer to:

> “How do you know the production system is healthy?”

---

# 28. Trade-Offs

## S3 pre-signed URLs

**Pros**
- removes media bandwidth from API
- scalable
- cheap
- separates storage from compute

**Cons**
- URL lifecycle
- temporary access semantics
- authorization must happen first

## WebSockets

**Pros**
- real-time
- low latency

**Cons**
- persistent connection management
- harder horizontal scaling

## Managed AI

**Pros**
- no GPU infrastructure
- simpler deployment

**Cons**
- cost
- latency
- vendor dependency
- external availability/privacy considerations

## BackgroundTasks

**Pros**
- simple
- integrated with FastAPI

**Cons**
- not a durable distributed queue
- limited retry/failure semantics
- tied to application process

---

# 29. Strongest Engineering Stories

## Story 1 — S3

**Problem:** large media could overload API bandwidth.

**Decision:** pre-signed URLs and direct S3 transfer.

**Result:** application server remains focused on business logic.

## Story 2 — Async AI

**Problem:** AI APIs make uploads slow.

**Decision:** background processing.

**Result:** short synchronous request path.

## Story 3 — Event RBAC

**Problem:** global roles cannot express event-specific permissions.

**Decision:** `Event_Roles(user_id, event_id, role)`.

**Result:** resource-scoped authorization.

## Story 4 — Watermarking

**Problem:** raw photographer assets should not be unrestricted downloads.

**Decision:** transform protected image downloads dynamically.

**Result:** normal viewing and protected download paths are separated.

---

# 30. Likely Interview Questions

### Architecture
1. Why Next.js + FastAPI?
2. Why PostgreSQL?
3. Why S3 instead of DB blobs?
4. Why pre-signed URLs?
5. What is control plane vs data plane?
6. Why async processing?
7. What if Gemini fails?
8. How would you horizontally scale FastAPI?
9. What becomes the bottleneck at 1M users?
10. How would you handle millions of media objects?

### Backend
11. How does JWT work?
12. Authentication vs authorization?
13. How do you prevent IDOR?
14. Why SQLAlchemy?
15. ORM vs raw SQL?
16. What are foreign keys?
17. Why database constraints?
18. How do indexes work?
19. What is an N+1 query?

### Storage
20. What is a pre-signed URL?
21. What if a signed URL leaks?
22. How would you handle a 2 GB video?
23. How would you implement resumable uploads?

### Concurrency
24. What if two users like the same photo?
25. How do you prevent duplicate likes?
26. How do you handle simultaneous bulk uploads?
27. How would you make background jobs idempotent?

### WebSockets
28. Why WebSockets instead of polling?
29. How do you scale WebSockets?
30. What happens when a client disconnects?
31. How would you guarantee notification delivery?

### AI
32. Why Gemini?
33. Why Rekognition?
34. Why not host your own model?
35. What if Rekognition rejects an input format?
36. How do you retry AI jobs?

---

# 31. 30-Second Pitch

> “Capture Hub is an event media-management platform I built around the challenges of large-scale photo sharing. The frontend is Next.js/React and the backend is FastAPI with PostgreSQL. The key architecture decision was to keep large media out of the API path: metadata lives in PostgreSQL while images and videos live in S3 and are delivered through pre-signed URLs. Expensive AI work such as Gemini-based semantic tagging and AWS Rekognition face indexing is handled asynchronously. I also implemented event-scoped RBAC, JWT authentication, dynamic watermarking for protected downloads, and WebSocket notifications. The main engineering focus was keeping the system responsive and secure despite constrained backend compute.”

---

# 32. 2-Minute Pitch

> “The problem I wanted to solve was event photography management. A single event can contain thousands of large images, and a basic CRUD gallery creates problems around storage, discovery, permissions, and performance.
>
> I used Next.js and React for the frontend and FastAPI with PostgreSQL and SQLAlchemy for the backend. The most important architectural decision was separating application compute from media storage. Instead of sending large images and videos through FastAPI, the backend generates pre-signed S3 URLs so the client can transfer media directly with object storage. PostgreSQL stores the metadata and relationships.
>
> I also separated expensive processing from the synchronous HTTP path. Uploaded media can be processed asynchronously for Gemini-based semantic tagging and AWS Rekognition face indexing. This was particularly important because the deployment environment was memory constrained.
>
> For authorization, I didn't use only a global role. Permissions are scoped to events using an Event_Roles relationship, so the same user can be a photographer for one event and a viewer for another. JWT handles authentication while resource-level checks handle authorization.
>
> The platform also supports likes and comments with WebSocket notifications. For intellectual-property protection, image downloads can go through a dynamic watermarking pipeline, while videos use a separate delivery path.
>
> If I were scaling it further, I would introduce a durable job queue and independent workers, Redis/pub-sub for horizontally scaled WebSockets, stronger observability, CDN optimization, and database indexing/read replicas where needed.”

---

# 33. What You Must Be Able to Draw From Memory

Before the interview, practice drawing:

```text
1. Complete architecture
2. Upload flow
3. S3 pre-signed URL flow
4. AI background-processing flow
5. Face-search flow
6. Event_Roles schema
7. IDOR prevention
8. JWT authentication
9. WebSocket notification flow
10. Watermarked download flow
```

---

# 34. Claims to Phrase Carefully

The README uses phrases such as “enterprise-grade,” “highly scalable,” and “massive scalability.”

In an interview, don't present those as proof of production-scale validation.

Prefer:

> “The architecture was designed to scale by separating compute, storage, and asynchronous processing.”

rather than:

> “I proved the system scales to millions of users.”

Similarly, don't call the current system a fully distributed job-processing platform merely because it uses FastAPI BackgroundTasks.

---

# 35. Final Mental Model

Think of Capture Hub as five responsibilities:

```text
                 CAPTURE HUB
                     |
       +-------------+-------------+
       |             |             |
       v             v             v
   Metadata        Media           AI
 PostgreSQL          S3       Gemini/Rekognition
       |             |             |
       +-------------+-------------+
                     |
                 FastAPI
            Auth + Authorization
                     |
              +------+------+
              |             |
              v             v
          WebSockets     Next.js
          Real-time      Frontend
```

### Core sentence to remember

> **“Keep the API responsible for business logic and authorization, keep large binary data in object storage, and move expensive processing away from the synchronous request path.”**

---

# 36. One-Line Recall

> **Capture Hub is a media-heavy full-stack system where I separated metadata, media storage, AI processing, authorization, and real-time communication so the application server wouldn't become the bottleneck.**
