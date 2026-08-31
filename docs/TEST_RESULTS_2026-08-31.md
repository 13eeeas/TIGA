# Test results — 2026-08-31

## Build under test

- Branch: `master`
- Revision: `504c8bf` (`Harden Windows network-share indexing`)
- Runtime: Windows, Python 3.12.13, Ollama (`mistral`, `nomic-embed-text`)

## Automated tests

```text
171 passed, 1 skipped, 13 dependency deprecation warnings
```

The skip is expected. The warnings are from FastAPI's TestClient and LanceDB's
deprecated `table_names()` API; no test failures remain.

## NUS BIZ3 initial index

- Scan root: one configured project folder on the office Egnyte drive
- Files discovered: 6,232
- Files indexed: 5,453
- Files skipped by indexing policy: 779
- Searchable chunks: 17,610
- Health check: SQLite and Ollama healthy

## Retrieval smoke test

- Query: `NUS BIZ3`
- Top results returned: 5
- Latency: 7,246.5 ms
- Citation validation: 100% valid
- Evaluation exit code: 0

No expected-path fixture was supplied for this ad-hoc query, so top-5 recall is
not scored. It must not be interpreted as a retrieval failure.

## Observations

- The index completed locally using Ollama only; no project corpus was sent to
  an external LLM provider.
- Some source files emitted recoverable parser warnings (corrupt PowerPoint
  media, malformed PDF metadata, and unsupported Excel validation metadata).
- Long Windows/Egnyte UNC path handling and transient network-share scan
  handling were fixed in the tested revision.
