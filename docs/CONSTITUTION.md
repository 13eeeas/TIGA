# TIGA Constitution

Working charter for Fit Hub / TIGA Hunt.  
Budget figures are planning estimates, not vendor quotations.  
This document overrides older roadmap language where they conflict.

---

## 1. Objective

**Build an office-LAN company-knowledge search system that finds the right evidence first, then answers with citations.**

Staff on the LAN ask plain-English questions about project archives and get:

- relevant source files / pages / chunks
- a short synthesized answer grounded only in that evidence
- clickable citations back to the NAS originals

Success is measured by retrieval accuracy and daily usefulness — not by chatbot flash, agent demos, or indexing the whole 30–40 TB archive.

### North star (near term)

| Phase | Scope | Gate to continue |
|-------|--------|------------------|
| Spike | 1 project, end-to-end on LAN | Ingest + search + cited answer + enterprise API wired |
| POC | **3** projects (expand to 5–10 in Pilot), ≤ ~S$1,000 | 100-question benchmark strong; staff return to the tool |
| Pilot | 5–10 projects | Usage + IT in-principle security OK |
| Scale test | 50–200 projects, ≤ ~S$10,000 | Security signed off; usage justifies spend |
| Product | Only if earned | Proven demand + ongoing budget owner |

Full checkpoints and funding asks: [`MILESTONES.md`](MILESTONES.md).

**Rule:** do not scale corpus size, Atlas, Einstein, or multimodal CAD/BIM until retrieval quality and usage justify it.

---

## 2. Operating environment (hard constraint)

### Primary deployment: office LAN

- Server runs on firm hardware (e.g. workstation with GPU) on the **office LAN**.
- UI and API are reachable from office browsers / clients on that LAN only by default.
- Raw project files stay on existing approved NAS / storage. TIGA indexes **representations** (metadata, chunks, embeddings) — it does not copy the archive.
- Offline / air-gapped mode must remain possible: search + cited snippets work even if every external API is disabled.

### What “LAN-first” means

| Must work on LAN without internet | May optionally use internet if firm policy allows |
|-----------------------------------|--------------------------------------------------|
| File discovery, ingest, dedupe | Frontier LLM synthesis over retrieved excerpts |
| Hybrid keyword + vector search | Cheap API embeddings (if local embed is a bottleneck) |
| Local reranker | Metadata classification at volume |
| Citation validation + UI | Approved vendor APIs under the security options below |
| Permission-scoped retrieval (when ACL exists) | — |

If the internet or an API is down, the product degrades to **cited search results**, not a blank page.

---

## 3. Core architecture principle

**Search first. AI second.**

```
Company files (NAS)
  → Ingest + clean (parse, hash, dedupe, version)
  → Search index (metadata + BM25 + embeddings)
  → Rerank (top ~50 → top ~8–15)
  → Answer model (local and/or approved API)
  → Cited answer + source links
```

- Do **not** train a model to memorize the archive.
- Do **not** send whole files to a model by default.
- Search inside files; send only the best pages/sections; every factual claim must be traceable to a source.

### Minimal stack (POC)

| Layer | Lean choice | Purpose |
|-------|-------------|---------|
| UI | LAN web UI (current HTML / FastAPI) | Internal chat + search |
| Backend | Python + FastAPI | Ingest, search orchestration |
| Metadata + FTS | SQLite + FTS5 (Postgres later if needed) | Exact terms, drawing numbers |
| Vectors | Local vector store + embeddings | Semantic retrieval |
| Reranking | Local open-source cross-encoder | Precision over hybrid candidates |
| Answering | Local LLM **and/or** approved API | Synthesis over evidence pack only |

---

## 4. Resource restrictions

| Constraint | Implication |
|------------|-------------|
| Two-person build | Prefer boring, proven components over agents, graphs, and dashboards |
| POC budget ~S$1,000 | Keep context tight; spend on retrieval quality before premium models |
| Existing office hardware / NAS | Represent, don’t replicate; tiered indexing; night embed queues |
| Do not index all 40 TB in POC | Start with ~5 known projects staff can judge |
| No unapproved SaaS | External calls only under an option below, after firm approval |

**Indicative POC spend posture:** small LLM/API allowance, tiny embedding allowance, near-zero hosting if self-hosted, large unspent buffer. Do not burn budget because it exists.

If each query sends only ~8–15 relevant chunks, API cost stays modest. Dumping 100k+ tokens per query wastes money and hurts accuracy.

---

## 5. Data strategy (accuracy is a data problem)

Before more model spend, cut the fat:

1. **Exact duplicates** — hash files; process one copy.
2. **Near duplicates** — link almost-identical extracted text so copies cannot outvote the real source.
3. **Version awareness** — project, doc type, revision, date, status; prefer authoritative / latest by default.
4. **Junk exclusion** — caches, temps, autosaves, renders, texture libraries, backups.

Duplicates are an **accuracy risk**, not only a storage cost. Five copies of Rev B can drown one copy of Rev F.

---

## 6. Retrieval quality (what we optimize first)

Order of work:

1. Correct corpus and authoritative versions  
2. Metadata quality  
3. Hybrid keyword + semantic retrieval  
4. Local reranking (full chunk text, not tiny snippets)  
5. Evidence pack size (~8–15 chunks to the answer model)  
6. Citation quality  
7. Only then compare more expensive answer models  

### POC benchmark (decision gate)

Build ~100 questions with known answers and proving documents/pages.

| Metric | POC target |
|--------|------------|
| Correct source in top 5 | >90% |
| Correct final answer | >85% |
| Citation supports claim | >95% |
| Major hallucinations | <3% |
| Typical latency | <10 s (ideal ~5 s) |

Continue toward the ~S$10k stage only if the benchmark is strong, answers beat manual file hunting, users return, and the security path is acceptable to the firm.

---

## 7. Answer model posture (local vs API)

Local hardware is **required** for the search system. It is **not** required that the final reasoning model run only on that GPU.

| Task | Prefer | Why |
|------|--------|-----|
| Ingest, FTS, vectors, ACL filters | Local | Data stays in-building |
| Reranking | Local open-source | No per-query bill; no excerpt leave for this step |
| Routine / hard answers | Strong model (local *or* API) | Quality; choose by security option |
| Metadata classification | Cheap model (local or API) | High volume, low risk if scoped |

**Decided posture (2026):** Einstein uses a **firm-approved enterprise API** (evidence-pack only, zero retention). Local LLM on the LAN host is **fallback only** (cited snippets / degraded search). See [`MILESTONES.md`](MILESTONES.md) for phased gates and funding checkpoints.

**Default product behavior:** LAN search always works. Synthesis uses the approved enterprise provider once IT/legal sign off; until then, local fallback only.

---

## 8. External API options and data security

All options assume: **raw NAS files never leave the building as bulk upload.** Only an already permission-filtered evidence pack may leave — and only if the chosen option allows it.

### Option A — LAN only (no external LLM)

| | |
|--|--|
| **What leaves** | Nothing |
| **Answer model** | Local Ollama (or equivalent) on office hardware |
| **Security** | Strongest. Suitable for air-gap / strictest clients |
| **Tradeoff** | Weaker synthesis than frontier APIs; GPU/context limits |
| **When** | Default until firm approves another option; always the fallback |

### Option B — Evidence-pack API (recommended if API is allowed)

| | |
|--|--|
| **What leaves** | Only top ~8–15 retrieved chunk texts + minimal citation ids the user is allowed to see |
| **What stays** | Full files, NAS paths browse, index DB, embeddings, user ACL maps |
| **Vendor bar** | Enterprise contract, **zero retention / no training**, DPA, region if required |
| **Controls** | Feature flag; hard token cap; timeout; local fallback; per-env API keys; audit log of query + source ids (not full file bodies) |
| **Security rule** | Apply **user permissions before retrieval**. Never retrieve restricted rows and ask the model to hide them |
| **When** | After Option A retrieval quality is proven and IT/legal approve the vendor |

### Option C — Private endpoint in firm cloud / VPC

| | |
|--|--|
| **What leaves** | Evidence pack stays inside firm-controlled cloud (Azure OpenAI, Bedrock, Vertex, private gateway) |
| **Security** | Stronger than public multi-tenant API if networking, keys, and logging are firm-owned |
| **Tradeoff** | More IT setup; still not “files never leave” if cloud is off-prem |
| **When** | Firm already standardizes on a private AI gateway |

### Option D — On-prem inference appliance / private model server

| | |
|--|--|
| **What leaves** | Nothing off LAN if the appliance is in-building |
| **Security** | Same class as Option A, with potentially stronger models |
| **Tradeoff** | Capex, ops burden; still weaker than best frontier APIs unless continuously updated |
| **When** | Policy forbids any cloud text egress but local mistral is too weak |

### Option E — Embeddings-only / classification-only API

| | |
|--|--|
| **What leaves** | Chunk text for embedding or short metadata prompts — still minimize and prefer local embeds when GPU allows |
| **Security** | Lower blast radius than full Q&A if retention is zero; still needs approval |
| **When** | Indexing throughput is the bottleneck, not answer quality |

### Options explicitly out of scope for POC

- Uploading project folders to consumer ChatGPT / Claude projects  
- Fine-tuning a vendor model on the firm archive  
- Letting an agent freely browse the NAS via a cloud tool  
- Relying on the LLM for access control  

### Security checklist (any option that sends text out)

1. SSO / identity before query (POC may stub; required before broad rollout).  
2. ACL filter in the database layer **before** hybrid search returns candidates.  
3. Evidence pack only — no whole-file, no bulk path trees.  
4. Approved vendor + written retention/training terms.  
5. Feature flag + kill switch + local fallback.  
6. Audit: who asked, which source ids were used, model tier, latency, user feedback.  
7. Separate API keys per environment; never embed keys in the LAN UI.  
8. Rate limits and max tokens per request.  

---

## 9. Product naming (roadmap language)

| Name | Role | When |
|------|------|------|
| **TIGA Hunt** (Fit Hub search) | Ingest, index, retrieve, cited answer | Now — the product |
| **TIGA Atlas** | Project memory / cross-project graph | After Hunt retrieval gate |
| **TIGA Einstein** | Stronger expert reasoning over Hunt evidence | After Hunt retrieval gate; may be API-backed under §8 |

Do not build Atlas/Einstein features that compete with fixing retrieval.

---

## 10. Do not attempt in the POC

- Index all 30–40 TB  
- Fine-tune a model to memorize company knowledge  
- Agents, graph DBs, multimodal CAD/BIM understanding, or heavy dashboards before basic retrieval works  
- Send confidential files to unapproved SaaS  
- Scale data volume before the benchmark and security path clear  

---

## 11. One-sentence roadmap

**Curate a small authoritative LAN corpus → hybrid search → local rerank → send only the best evidence to a firm-approved answer model (local or API) → return cited answers → benchmark ruthlessly → scale only what proves useful.**
