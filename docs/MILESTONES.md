# TIGA Milestones & Funding Gates

Realistic checkpoints from cheap POC to full product.  
Binding principles: [`CONSTITUTION.md`](CONSTITUTION.md).

**Decided:** Einstein (answer synthesis) uses a **firm-approved enterprise AI API** (evidence-pack only, zero retention). Hunt (search) stays on the **office LAN host**. Local LLM remains fallback only.

**Final product goal:** One LAN-hosted service where staff ask plain-English questions, get **Hunt** retrieval + **Atlas** project memory + **Einstein** synthesis — all cited, permission-aware, and faster than hunting NAS folders.

---

## Gateways at a glance

| Gateway | What it is | Corpus | Budget cap | Unlocks |
|---------|------------|--------|------------|---------|
| **Prep** | Build setup, vendor shortlist | 0–1 project smoke test | ≤ S$100 | Permission to build POC |
| **Gateway 1 — POC product** | **Shipped LAN product that works super well** | **3–5 projects** | **≤ S$1,000** | **Pilot funding** |
| Gateway 2 — Pilot | Prove habit + security in principle | 5–10 projects | ≤ S$2,500 | Scale-test funding |
| Gateway 3 — Scale test | Firm-wide slice, ACL + SSO | 50–200 projects | ≤ S$10,000 | Production budget |
| Gateway 4 — Product | Earned expansion | Usage-driven | Ongoing opex | — |

**First real gate = Gateway 1.**  
Not a spike. Not a deck. A **POC product** on **3–5 indexed projects** that staff can use on the LAN and that **measurably works super well**.

---

## What “works super well” means (Gateway 1 bar)

Subjective “feels good” is not enough. Gateway 1 passes when **all** of these are true:

### Product shape
- One **LAN host** (your office machine is fine); staff open a **browser**
- **Hunt** — hybrid search, local rerank, validated citations
- **Atlas** — project cards + aliases + cross-project queries across the indexed set
- **Einstein** — enterprise API synthesis over an **8–15 chunk evidence pack** only
- Local fallback if API is down (cited results, not a blank screen)

### Quality (100-question benchmark on those 3–5 projects)
| Metric | Gateway 1 target |
|--------|------------------|
| Correct source in top 5 | **>90%** |
| Correct final answer | **>85%** |
| Citation supports claim | **>95%** |
| Major hallucinations | **<3%** |
| Typical latency | **<10 s** (ideal ~5 s) |

### Human proof
- ≥ **3 staff** use it unprompted and would choose it over NAS folder hunting on tested questions
- Sponsor can watch a **live LAN demo** and agree: “this is good enough to fund the next stage”

### Cost proof
- Total POC spend **≤ S$1,000** (mostly enterprise API credits; self-hosted index)
- Logged **median cost per query** so pilot opex is predictable

**If Gateway 1 fails:** stop or fix retrieval — do not ask for pilot money, more projects, SSO, or Atlas graph work.

---

## Architecture (fixed)

```
[NAS] → [LAN host: Hunt index + rerank + Atlas cards]
              ↓ evidence pack only (~8–15 chunks)
        [Enterprise API = Einstein synthesis]
              ↓
        [Cited answer in browser on LAN]
```

| Layer | Runs where |
|-------|------------|
| **Hunt** | LAN host — ingest, search, rerank, citations |
| **Atlas** | LAN host — project memory on indexed projects |
| **Einstein** | Enterprise API — synthesis; local = fallback only |

---

## Prep (not a gateway — internal setup)

**Goal:** Get ready to build Gateway 1 without burning budget.

- Constitution + milestones aligned
- Pick **3–5 completed projects** staff can judge (mix of typologies if possible)
- Shortlist one **enterprise API** path for IT/legal
- Optional: 1-project smoke test to verify ingest → search → API compose pipe

**Budget:** ≤ S$100 API smoke test · S$0 hosting  
**Not a funding moment.** Just don’t start Gateway 1 without a sponsor nod that ≤ S$1k POC is acceptable if the gate passes.

---

## Gateway 1 — POC product (3–5 projects)

**This is the first gateway.** Everything before it is setup; everything after it is earned.

**Duration:** ~4–6 weeks part-time  
**Host:** Office LAN machine (3070 + i9 is fine)  
**Corpus:** **3–5 representative completed projects** with known-good answers  

### Build checklist
- [ ] Index 3–5 projects: dedupe, junk skip, **version-aware** (prefer latest by default)
- [ ] Hybrid search + local reranker **on** (full chunk text, not tiny snippets)
- [ ] Evidence pack **8–15 chunks** → enterprise API
- [ ] LAN UI + API; kill switch + audit log (query, source ids, model)
- [ ] **Atlas:** project cards, aliases, stage for every indexed project
- [ ] **Atlas:** cross-project queries work on the indexed set (typology, waivers, scale, etc.)
- [ ] **100-question benchmark** built from real firm questions + proving docs/pages
- [ ] **POC results memo** (1–2 pages for funders)

### Hunt / Atlas / Einstein in Gateway 1
| Product | Gateway 1 scope |
|---------|-----------------|
| Hunt | The core — must feel sharp |
| Atlas | Thin but real — cards + cross-project on 3–5 only |
| Einstein | Enterprise API — quality bar for answers |

### Deliverable (what you show to unlock funding)
1. Live product on LAN over **3–5 projects**  
2. Benchmark report hitting the table above  
3. API cost actuals (total + per query)  
4. 3 staff quotes or a short screen recording  
5. One-page security note: excerpts only, enterprise terms  

**Budget cap:** ≤ **S$1,000** total  
Suggested: S$150–400 API · small embed/OCR contingency · rest buffer  

**Funding ask after Gateway 1 passes:** **Pilot tranche ~S$1,500–2,500**  
- Dedicated always-on LAN host (if needed)  
- Higher API monthly cap  
- Build time if internal capacity is tight  

---

## Gateway 2 — Pilot (5–10 projects, prove habit)

**Duration:** ~6–8 weeks  
**Corpus:** Grow from POC set to **5–10 projects**  

### Build
- [ ] Dedicated LAN host if office PC was the bottleneck
- [ ] Feedback loop (thumbs, wrong-source flag)
- [ ] Einstein: mid-tier default; top-tier only on low confidence
- [ ] Richer Atlas (authoritative doc hints, compare views)
- [ ] Basic project-level permissions (SSO can wait)
- [ ] Monthly usage + quality report

### Proof
| Metric | Target |
|--------|--------|
| Benchmark | >90% top-5 source on expanded set |
| Weekly active users | ≥ 5 staff |
| Repeat usage | Same users ≥ 3× in 4 weeks |
| Security | IT/legal **in principle** OK with evidence-pack flow |
| Incidents | Zero bulk egress |

**Budget cap:** ≤ S$2,500 cumulative  

**Funding ask after Gateway 2:** **Scale-test ~S$5,000–10,000** (SSO, ACL, 50+ projects, monitoring)

---

## Gateway 3 — Scale test (50–200 projects)

**Duration:** ~3–6 months  

### Build
- [ ] SSO + ACL before retrieval
- [ ] Automated metadata where manual tagging failed
- [ ] Full audit export; index economics dashboard
- [ ] Near-duplicate linking at scale

### Proof
- Benchmark holds on stratified sample  
- Formal security sign-off  
- Leadership names budget owner  
- Documented $/query and uptime  

**Budget cap:** ≤ S$10,000 cumulative  

**Funding ask after Gateway 3:** **Production line item** — hosting, API opex, maintenance

---

## Gateway 4 — Product (earned)

Expand corpus by **usage and retrieval quality**, not raw TB.  
Hunt stays core; Atlas = memory staff rely on; Einstein = enterprise API with firm tone.

Quarterly benchmark sample + cost review when vendor or policy changes.

---

## Explicit non-goals until Gateway 1 passes

- Indexing beyond **5 projects**  
- SSO, full ACL, Postgres migration  
- Knowledge graphs, agents, BIM/CAD multimodal  
- Training or fine-tuning on the archive  
- Consumer ChatGPT project uploads  

---

## Next 30 days → Gateway 1

| Week | Focus |
|------|--------|
| 1 | Lock **3–5 projects** + enterprise vendor paperwork |
| 2 | Index first 2; Hunt + rerank + API Einstein on LAN |
| 3 | Index remainder; Atlas cards + cross-project queries |
| 4 | 100-Q benchmark run + POC memo |

**Spend target until benchmark:** stay inside **S$1,000**.

**Backend checklist:** [`BACKEND_CHECKLIST.md`](BACKEND_CHECKLIST.md) — engineering tasks for Gateway 1.

---

## One-line pitch (Gateway 1)

> **We ship a LAN product on 3–5 projects. If it doesn’t score >90% on finding the right source and >85% on answers — for under S$1,000 — we stop. If it works super well, fund the pilot.**

That is Gateway 1. Everything else is earned.
