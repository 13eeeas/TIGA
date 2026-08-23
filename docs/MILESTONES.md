# TIGA Milestones & Funding Gates

Realistic checkpoints from cheap POC to full product.  
Binding principles: [`CONSTITUTION.md`](CONSTITUTION.md).

**Decided:** Einstein (answer synthesis) uses a **firm-approved enterprise AI API** (evidence-pack only, zero retention). Hunt (search) stays on the **office LAN host**. Local LLM remains fallback only.

**Final product goal:** One LAN-hosted service where staff ask plain-English questions, get **Hunt** retrieval + **Atlas** project memory + **Einstein** synthesis — all cited, permission-aware, and faster than hunting NAS folders — proven on a growing slice of the archive, not on a demo chatbot.

---

## How funding works (read this first)

Each phase has:

1. **Build goal** — what we ship  
2. **Budget cap** — planning ceiling (not a quote)  
3. **Proof required** — objective metrics, not vibes  
4. **Funding ask** — what to request *after* the gate passes  

**Rule:** the next tranche of money is earned by measured results, not by roadmap slides.

| Phase | Corpus | Budget cap (planning) | Funding ask after gate |
|-------|--------|------------------------|-------------------------|
| 0 Charter | — | S$0 | — (alignment only) |
| 1 Spike | 1 project | ≤ S$100 | — |
| 2 POC | 3 projects | ≤ S$500–1,000 | Pilot tranche |
| 3 Pilot | 5–10 projects | ≤ S$2,500 | Scale-test tranche |
| 4 Scale test | 50–200 projects | ≤ S$10,000 | Production tranche |
| 5 Product | Earned expansion | Firm-defined | Ongoing ops budget |

POC stays **cheap** by design: existing office machine (3070/i9), self-hosted index, tight evidence packs, small API allowance, no new hardware.

---

## Architecture (fixed for all phases)

```
[NAS] → [LAN host: Hunt index + rerank + Atlas cards]
              ↓ evidence pack only (~8–15 chunks)
        [Enterprise API = Einstein synthesis]
              ↓
        [Cited answer in browser on LAN]
```

| Layer | Runs where | Phase 1+ |
|-------|------------|----------|
| **Hunt** | LAN host | Ingest, hybrid search, rerank, citations |
| **Atlas** | LAN host (DB) | Project cards, aliases, cross-project filters |
| **Einstein** | Enterprise API | Answer synthesis; local = fallback snippets only |

---

## Phase 0 — Charter & vendor path

**Goal:** Align scope, security, and funding story before spend.

**Deliverables**
- Constitution + milestones (this doc)
- One-page pitch: problem → LAN Hunt → API Einstein → gates
- Shortlist **one** enterprise vendor path (e.g. Azure OpenAI enterprise, Anthropic enterprise, OpenAI enterprise — whichever IT/legal can approve)
- Draft data-flow diagram: what leaves the building (excerpts only)

**Budget:** S$0  
**Gate to Phase 1:** Sponsor agrees POC is worth ≤ S$1k and a named person can judge answers on 1 project.

**Funding ask:** None yet — approval to proceed in spare time.

---

## Phase 1 — Spike (1 project, prove the pipe)

**Duration:** ~1–2 weeks part-time  
**Host:** Your office machine on LAN (3070 + i9)  
**Corpus:** 1 completed project staff know well  

### Build
- [ ] Index 1 project (dedupe, junk skip, basic version fields)
- [ ] Hybrid search + local reranker **on**
- [ ] LAN UI reachable from office browser
- [ ] Enterprise API wired for Einstein (evidence pack only, kill switch)
- [ ] Local fallback: cited search if API off/down
- [ ] Audit log stub: query + source ids + model used

### Hunt / Atlas / Einstein at this phase
| Product | Scope |
|---------|--------|
| Hunt | End-to-end search + citations |
| Atlas | Minimal project card for that 1 code |
| Einstein | API synthesis over top evidence pack |

### Proof (checkpoint 1)
| Check | Target |
|-------|--------|
| End-to-end demo | 10 hand-picked questions answered with valid citations |
| Latency | < 15 s typical (spike tolerance) |
| Security | Written note: excerpts only; vendor enterprise terms cited |
| Cost | Log actual API spend for 10–20 queries |

**Budget cap:** ≤ S$100 API + S$0 hosting  
**Gate to Phase 2:** Sponsor watches live demo; ≥ 8/10 questions cite the right file/page in top 5.

**Funding ask after gate:** **POC tranche ≤ S$1,000** (API credits + optional small embed contingency; no hardware).

---

## Phase 2 — POC (3 projects, prove retrieval)

**Duration:** ~3–4 weeks part-time  
**Corpus:** 3 representative **completed** projects (staff can mark ground truth)  

### Build
- [ ] Index all 3 projects with version-aware ranking (prefer latest by default)
- [ ] Evidence pack → 8–15 chunks to enterprise API
- [ ] Rerank on **full chunk text**, not tiny snippets
- [ ] Atlas: project cards + aliases + stage for all 3
- [ ] Cross-project queries that work on 3 (e.g. typology, waivers list)
- [ ] Benchmark harness: 50 questions → grow to **100**
- [ ] One-page **POC results memo** for funders

### Proof (checkpoint 2 — main funding gate)

| Metric | POC target | Funder cares because |
|--------|------------|----------------------|
| Correct source in top 5 | **>85%** on 50 Q (then **>90%** on 100 Q) | Search actually works |
| Correct final answer | **>80%** (then **>85%**) | Not just pretty text |
| Citation supports claim | **>95%** | Auditable / low risk |
| Major hallucinations | **<5%** (then **<3%**) | Trust |
| Typical latency | **<10 s** | Daily usability |
| API cost per query | Logged median | Predictable opex |
| User signal | ≥ 3 staff try it twice unprompted | Real demand |

**Budget cap:** ≤ S$500–1,000 total (incl. Phase 1 spend)  
Suggested split: S$150–300 API · S$0–50 embed/OCR contingency · rest unspent buffer  

**Gate to Phase 3 (Pilot funding):**  
All of:
- 100-question benchmark at POC targets  
- POC memo + cost actuals  
- IT/legal **in principle** OK with enterprise evidence-pack flow  
- Sponsor statement: faster than manual file hunting on tested questions  

**Funding ask after gate:** **Pilot tranche ~S$1,500–2,500**  
- Dedicated always-on LAN host (or firm VM) — optional if office PC is bottleneck  
- Expanded API monthly cap  
- Part-time build capacity (if internal time isn’t enough)  

---

## Phase 3 — Pilot (5–10 projects, prove habit)

**Duration:** ~6–8 weeks  
**Corpus:** 5–10 projects; mix of typologies  

### Build
- [ ] Move host to dedicated LAN machine if needed
- [ ] Atlas: richer cards, cross-project compare, “authoritative doc” hints
- [ ] Einstein: mid-tier default, escalate to top-tier on low confidence only
- [ ] Feedback loop (thumbs + wrong-source flag)
- [ ] Basic project-level permissions (even if SSO comes later)
- [ ] Monthly usage + quality report auto-generated

### Proof (checkpoint 3)

| Metric | Pilot target |
|--------|----------------|
| Correct source in top 5 | >90% maintained on expanded set |
| Weekly active users | ≥ 5 staff |
| Repeat usage | Same users return ≥ 3× in 4 weeks |
| Time saved (sample) | Self-reported or timed: beat NAS hunt on ≥ 70% of tasks |
| Incidents | Zero bulk file egress; zero ACL bypass |

**Budget cap:** ≤ S$2,500 cumulative  
**Gate to Phase 4:** Usage + benchmark hold; security path **signed off**; sponsor wants 50+ projects.

**Funding ask after gate:** **Scale-test tranche ~S$5,000–10,000**  
- SSO integration  
- Postgres migration if SQLite limits hit  
- Higher API cap + monitoring  
- Optional private VPC endpoint (if firm requires)  

---

## Phase 4 — Scale test (50–200 projects)

**Duration:** ~3–6 months  
**Corpus:** 50–200 projects; automation where manual metadata failed  

### Build
- [ ] Automated metadata classification (cheap model, scoped)
- [ ] ACL before retrieval (SSO-linked)
- [ ] Full audit export for compliance
- [ ] Index economics dashboard (DB size, embed queue, cost/query)
- [ ] Near-duplicate linking at scale
- [ ] Atlas: cross-project patterns staff actually use (not graph science)

### Proof (checkpoint 4)

| Metric | Scale-test target |
|--------|-------------------|
| Benchmark | Holds on stratified sample across project types |
| Uptime | LAN service ≥ 99% during office hours |
| Cost | Documented $/query and $/project indexed |
| Security review | Formal sign-off complete |
| Demand | Leadership names owner + ongoing budget line |

**Budget cap:** ≤ S$10,000 cumulative planning envelope  
**Gate to Phase 5:** Proven ROI + signed security + budget owner.

**Funding ask after gate:** **Production / enterprise line item** — hosting, API opex, maintenance headcount.

---

## Phase 5 — Product (earned scale)

**Goal:** Firm-wide LAN knowledge service — not “index everything.”

- Expand corpus based on **usage and retrieval quality**, not raw TB count  
- Hunt: always the core  
- Atlas: project memory layer staff rely on  
- Einstein: enterprise API synthesis with firm tone + escalation rules  
- No agents / CAD multimodal / fine-tuning on archive unless a separate funded initiative passes its own gate  

**Ongoing gates:** quarterly benchmark sample + cost review + security re-check when vendor or policy changes.

---

## What to show funders at each ask

### POC tranche (after Phase 1)
- 2-minute LAN demo (1 project)
- Architecture one-pager (LAN Hunt + enterprise Einstein)
- Estimated POC cost ≤ S$1k
- Risk: “We stop if benchmark fails”

### Pilot tranche (after Phase 2)
- **100-question benchmark report** (the killer slide)
- Cost actuals vs estimate
- 3-user anecdotal wins (“found Rev F tender in 8s”)
- IT one-pager: excerpts only, enterprise DPA

### Scale tranche (after Phase 3)
- Usage graph + repeat users
- Security sign-off
- Index size vs archive size (prove represent-don’t-replicate)
- Opex model: $/month at current query volume

---

## Explicit non-goals until gates pass

| Until | Do not |
|-------|--------|
| Phase 2 gate | Index >3 projects, build graph DB, train models |
| Phase 3 gate | SSO, 50 projects, dashboards |
| Phase 4 gate | Full archive, BIM/CAD understanding, agents |
| Any phase | Upload folders to consumer ChatGPT / Claude projects |

---

## Suggested immediate next 30 days (you, 3070 host)

| Week | Focus |
|------|--------|
| 1 | Pick 1 project + enterprise vendor paperwork started |
| 2 | Phase 1 spike live on LAN + API Einstein wired |
| 3 | Add 2 more projects; turn on rerank + version defaults |
| 4 | 50-question benchmark draft; POC memo v0 |

**Spend so far target:** < S$100 API until checkpoint 1 passes.

---

## One-line pitch for sponsorship

> **For under S$1,000 we prove on 3 projects that staff can ask questions on the LAN and get cited, enterprise-grade answers from our own archive — measured on 100 real questions. If it fails, we stop. If it works, we fund a pilot.**

That is the funding story. Everything else is earned.
