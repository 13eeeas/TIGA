# TIGA Hunt

TIGA Hunt is a local-first archive intelligence system. The current pilot is
focused on a seminar, talks, presentation, training, and exhibition corpus, not
project-document indexing.

Pilot corpus:

```text
F:\Shared\Non Project Image\02 Talks Presentation Training Exhibitions\000 Out of Office Seminars
```

The product goal is to let a user ask factual or synthesis questions about this
corpus and receive grounded answers with citations to retrieved evidence.

## Quick Start

```bat
setup.bat
run.bat
```

Then open:

```text
http://localhost:8501
```

Useful CLI commands:

```bat
python tiga.py init
python tiga.py discover
python tiga.py index
python tiga.py query "Which talks mentioned AI in architecture workflows?"
python tiga.py eval
python tiga.py wiki
python tiga.py status
python tiga.py health
python tiga.py serve
python tiga.py ui
```

## Knowledge Corpus Mode

Knowledge Corpus Mode indexes knowledge documents rather than project archives.
It assumes the most important entities are topics, speakers, events, training
material, transcripts, slide decks, and recurring concepts.

The active config profile is in:

```text
tiga_work/config.yaml
```

Key settings:

```yaml
corpus:
  mode: knowledge_corpus
  name: Out of Office Seminars
  metadata_first: true
  contextual_chunk_headers: true
```

Expected inputs are directories containing PDFs, PowerPoint decks, Word files,
plain text, Markdown, spreadsheets, images, and media exports. Text extraction
is preferred. OCR is opt-in and should be enabled only when needed.

Supported text extraction:

- PDF: page chunks
- PPTX: slide chunks
- DOCX: heading/section chunks
- TXT/MD: section chunks
- XLSX/XLS: sheet chunks when spreadsheet dependencies are installed

Metadata-only indexing:

- Images
- Video/audio exports
- CAD/BIM files
- Unsupported binary files

## Retrieval Philosophy

TIGA should not brute-force large raw contexts into an LLM. The intended path is:

1. Discover and classify files using filesystem metadata.
2. Extract text where practical.
3. Infer low-cost metadata such as title, event, speaker, topic, file type, date,
   duplicate group, and parent document relationship.
4. Retrieve with metadata filters first.
5. Search with BM25/FTS and dense embeddings.
6. Fuse ranks and optionally rerank a small candidate pool.
7. Send only a small, high-quality evidence pack to the LLM.
8. Answer with citations and explicit uncertainty.

The LLM is late in the pipeline. It synthesizes retrieved evidence; it is not
the primary search engine.

## Metadata Schema

Knowledge Corpus Mode adds these file-level fields:

- title
- speaker
- event name
- event type
- date
- source file path
- file type
- topic
- subtopic
- organization
- people mentioned
- keywords
- summary
- confidence
- extraction method
- language
- duplicate group
- parent document / slide deck / transcript relationship

Every chunk preserves source file path, chunk reference, and citation.
Contextual chunk headers add document-level metadata before chunk text so dense
retrieval has parent context.

## Output Artifacts

Runtime artifacts are written under `tiga_work/`:

```text
tiga_work/
  config.yaml
  db/tiga.db
  vectors/
  reports/
  wiki/
  fixtures/eval_queries.yaml
```

The wiki pipeline writes durable Markdown pages:

- `wiki/topics/*.md`
- `wiki/speakers/*.md`
- `wiki/events/*.md`
- `wiki/concepts/*.md`
- `wiki/index.md`

Pages include source citations and backlinks. Thin-evidence pages are marked as
provisional.

## Evaluation Flow

Start with:

```bat
python tiga.py eval
```

The eval fixture lives at:

```text
tiga_work/fixtures/eval_queries.yaml
```

The harness reports:

- retrieval hit@k
- MRR
- citation correctness
- latency p50/p95
- groundedness proxy when `answer_must_include` is supplied
- explicit placeholders for hallucination/fabrication rate and wiki usefulness

For a rigorous corpus eval, expand the fixture to 30-50 real questions after the
first full index. Use questions that test recurring themes, AI workflows,
design process, collaboration, fire safety, regulations, speaker attribution,
and comparisons between seminars.

## Current Limitations

- Speaker, event, topic, and date metadata are heuristic and may need manual
  correction for ambiguous filenames.
- OCR remains opt-in and is not yet part of a selective OCR queue.
- Near-duplicate detection is hash-based and practical, not semantic clustering.
- Wiki pages are deterministic evidence maps, not polished LLM essays.
- Hallucination rate and wiki usefulness require a human or LLM-judge rubric
  once a gold evaluation set exists.
- Some project-era commands remain for compatibility and are not central to
  Knowledge Corpus Mode.
