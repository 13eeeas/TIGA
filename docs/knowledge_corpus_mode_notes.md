# Knowledge Corpus Mode Notes

## Implemented In This Pass

- Active config points to the Out of Office Seminars pilot corpus.
- Additive SQLite metadata columns for knowledge documents.
- Deterministic metadata inference for title, speaker, event, event type, date,
  file type, topic, keywords, language, duplicate group, parent document ID,
  and source relationship.
- Contextual chunk headers preserve document-level context during retrieval.
- Retrieval filters now understand topic, speaker, event name, event type, file
  type, and date range.
- Eval harness now reports hit@k, MRR, citation correctness, latency, and a
  groundedness proxy when fixture terms are supplied.
- First-pass wiki generation writes Markdown pages for topics, speakers, events,
  concepts, and an index page.

## Stubbed Or Still Uncertain

- Speaker/event extraction is heuristic. It should be reviewed after the first
  index and corrected with a small manual metadata override layer.
- OCR selection is not yet implemented as a queue. OCR remains opt-in.
- Duplicate detection uses normalized text/file signatures. It is not yet
  semantic near-duplicate clustering.
- Wiki pages are deterministic evidence maps. LLM synthesis can be added later
  once retrieval quality and citation correctness are stable.
- Hallucination/fabrication rate and wiki usefulness need a human or LLM-judge
  rubric with gold answers.
- The API/UI still expose some project-era labels for compatibility.

## Recommended Next Steps

1. Run `python tiga.py index` against the seminar corpus.
2. Inspect `python tiga.py status` and spot-check metadata in SQLite.
3. Replace placeholder eval entries with 30-50 real questions and expected
   source paths.
4. Run `python tiga.py eval` and review retrieval misses.
5. Run `python tiga.py wiki` and inspect `tiga_work/wiki/index.md`.
6. Only then tune reranking, chunk sizes, OCR, or model behavior.
