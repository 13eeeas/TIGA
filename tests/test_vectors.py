"""
tests/test_vectors.py — Vector store and embedding unit tests.

Uses mocks so Ollama and LanceDB are not required.
Run with: pytest tests/test_vectors.py -v
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_embedding(dim: int = 768) -> list[float]:
    return [0.1] * dim


def _mock_urlopen(embedding: list[float]):
    """Context manager mock that returns a fake Ollama /api/embeddings response."""
    body = json.dumps({"embedding": embedding}).encode()
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: mock_resp
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = body
    return patch("urllib.request.urlopen", return_value=mock_resp)


# ---------------------------------------------------------------------------
# embed_text
# ---------------------------------------------------------------------------

class TestEmbedText:
    def test_returns_vector(self) -> None:
        from config import cfg
        from core.vectors import embed_text

        with _mock_urlopen(_fake_embedding()):
            vec = embed_text("hospital brief", cfg)

        assert isinstance(vec, list)
        assert len(vec) == 768

    def test_raises_embed_error_on_network_failure(self) -> None:
        from config import cfg
        from core.vectors import EmbedError, embed_text

        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            with pytest.raises(EmbedError):
                embed_text("test", cfg)

    def test_raises_embed_error_on_bad_response(self) -> None:
        """Response without 'embedding' key must raise EmbedError."""
        from config import cfg
        from core.vectors import EmbedError, embed_text

        body = json.dumps({"error": "model not found"}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = body

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(EmbedError):
                embed_text("test", cfg)

    def test_uses_configured_model(self) -> None:
        """embed_text sends the embed_model from cfg in the request body."""
        from config import cfg
        from core.vectors import embed_text

        captured: list[bytes] = []

        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def read(self): return json.dumps({"embedding": _fake_embedding()}).encode()

        def fake_urlopen(req, timeout=None):
            captured.append(req.data)
            return _Resp()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            embed_text("hello", cfg)

        sent = json.loads(captured[0])
        assert sent["model"] == cfg.embed_model
        assert sent["prompt"] == "hello"


# ---------------------------------------------------------------------------
# embed_texts_batched
# ---------------------------------------------------------------------------

class TestEmbedTextsBatched:
    def test_returns_none_for_failed_items(self) -> None:
        from config import cfg
        from core.vectors import EmbedError, embed_texts_batched

        call_count = 0

        def flaky_embed(text, _cfg=None):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise EmbedError("gpu oom")
            return _fake_embedding()

        # Force fallback to per-item path by making the batch call fail
        with patch("core.vectors.embed_texts_batch", side_effect=EmbedError("batch fail")), \
             patch("core.vectors.embed_text", side_effect=flaky_embed):
            results = embed_texts_batched(["a", "b", "c", "d"], cfg)

        assert len(results) == 4
        assert results[0] is not None   # call 1 — ok
        assert results[1] is None       # call 2 — failed
        assert results[2] is not None   # call 3 — ok
        assert results[3] is None       # call 4 — failed

    def test_sleeps_between_batches(self) -> None:
        from config import cfg
        from core.vectors import embed_texts_batched

        sleep_calls: list[float] = []
        # batch_size = max(cfg.embed_batch_size, 64); use +1 to span two batches
        batch_size = max(cfg.embed_batch_size, 64)

        with patch("core.vectors.embed_texts_batch",
                   return_value=[_fake_embedding()] * batch_size), \
             patch("core.vectors.time.sleep", side_effect=sleep_calls.append):
            embed_texts_batched(["a"] * (batch_size + 1), cfg)

        assert len(sleep_calls) == 1
        assert sleep_calls[0] == pytest.approx(cfg.embed_batch_sleep_s)

    def test_no_sleep_for_single_batch(self) -> None:
        from config import cfg
        from core.vectors import embed_texts_batched

        sleep_calls: list[float] = []

        with patch("core.vectors.embed_text", return_value=_fake_embedding()), \
             patch("core.vectors.time.sleep", side_effect=sleep_calls.append):
            embed_texts_batched(["only one"], cfg)

        assert len(sleep_calls) == 0


# ---------------------------------------------------------------------------
# upsert_chunk
# ---------------------------------------------------------------------------

class TestUpsertChunk:
    def test_returns_true_on_success(self, tmp_path) -> None:
        from config import cfg
        from core.vectors import upsert_chunk

        with patch("core.vectors.embed_text", return_value=_fake_embedding()), \
             patch("core.vectors.get_db") as mock_get_db:
            mock_table = MagicMock()
            mock_get_db.return_value.table_names.return_value = []
            mock_get_db.return_value.create_table.return_value = mock_table

            result = upsert_chunk(
                "chunk-001",
                "hospital tender brief",
                {"file_id": "f1", "ref_value": "p1", "rel_path": "/a/b.pdf",
                 "project_id": "proj1", "typology": "healthcare", "ext": ".pdf",
                 "root_id": "/archive"},
                cfg,
            )

        assert result is True

    def test_returns_false_on_embed_error(self) -> None:
        from config import cfg
        from core.vectors import EmbedError, upsert_chunk

        with patch("core.vectors.embed_text", side_effect=EmbedError("timeout")):
            result = upsert_chunk("chunk-001", "text", {"file_id": "f1"}, cfg)

        assert result is False

    def test_returns_false_on_lancedb_error(self) -> None:
        from config import cfg
        from core.vectors import upsert_chunk

        with patch("core.vectors.embed_text", return_value=_fake_embedding()), \
             patch("core.vectors.get_db") as mock_get_db:
            mock_table = MagicMock()
            mock_table.merge_insert.return_value.when_matched_update_all.return_value \
                .when_not_matched_insert_all.return_value.execute.side_effect = \
                RuntimeError("disk full")
            mock_get_db.return_value.table_names.return_value = ["tiga_chunks"]
            mock_get_db.return_value.open_table.return_value = mock_table

            result = upsert_chunk("chunk-001", "text", {"file_id": "f1"}, cfg)

        assert result is False


# ---------------------------------------------------------------------------
# query_vector
# ---------------------------------------------------------------------------

class TestQueryVector:
    def test_returns_empty_on_embed_error(self) -> None:
        from config import cfg
        from core.vectors import EmbedError, query_vector

        with patch("core.vectors.embed_text", side_effect=EmbedError("down")):
            results = query_vector("hospital", top_k=5, cfg=cfg)

        assert results == []

    def test_returns_empty_when_table_empty(self) -> None:
        from config import cfg
        from core.vectors import query_vector

        with patch("core.vectors.embed_text", return_value=_fake_embedding()), \
             patch("core.vectors.get_db") as mock_get_db:
            mock_table = MagicMock()
            mock_table.count_rows.return_value = 0
            mock_get_db.return_value.table_names.return_value = ["tiga_chunks"]
            mock_get_db.return_value.open_table.return_value = mock_table

            results = query_vector("hospital", top_k=5, cfg=cfg)

        assert results == []

    def test_returns_correct_shape(self) -> None:
        from config import cfg
        from core.vectors import query_vector

        fake_rows = [
            {
                "chunk_id":   f"chunk-{i:03d}",
                "file_id":    "f1",
                "ref_value":  f"p{i}",
                "rel_path":   "/a/b.pdf",
                "project_id": "proj1",
                "typology":   "healthcare",
                "ext":        ".pdf",
                "root_id":    "/archive",
                "_distance":  0.1 * i,
            }
            for i in range(3)
        ]

        with patch("core.vectors.embed_text", return_value=_fake_embedding()), \
             patch("core.vectors.get_db") as mock_get_db:
            mock_table = MagicMock()
            mock_table.count_rows.return_value = 3
            mock_table.search.return_value.metric.return_value \
                .limit.return_value.to_list.return_value = fake_rows
            mock_get_db.return_value.table_names.return_value = ["tiga_chunks"]
            mock_get_db.return_value.open_table.return_value = mock_table

            results = query_vector("hospital", top_k=5, cfg=cfg)

        assert len(results) == 3
        for r in results:
            assert "chunk_id" in r
            assert "distance" in r
            assert "metadata" in r
            assert "file_id" in r["metadata"]

    def test_filter_applied(self) -> None:
        """filters dict gets translated to a .where() call."""
        from config import cfg
        from core.vectors import query_vector

        with patch("core.vectors.embed_text", return_value=_fake_embedding()), \
             patch("core.vectors.get_db") as mock_get_db:
            mock_search = MagicMock()
            mock_search.metric.return_value = mock_search
            mock_search.limit.return_value = mock_search
            mock_search.where.return_value = mock_search
            mock_search.to_list.return_value = []

            mock_table = MagicMock()
            mock_table.count_rows.return_value = 10
            mock_table.search.return_value = mock_search
            mock_get_db.return_value.table_names.return_value = ["tiga_chunks"]
            mock_get_db.return_value.open_table.return_value = mock_table

            query_vector("hospital", top_k=5, cfg=cfg, filters={"typology": "healthcare"})

        mock_search.where.assert_called_once()
        where_arg = mock_search.where.call_args[0][0]
        assert "typology" in where_arg
        assert "healthcare" in where_arg
