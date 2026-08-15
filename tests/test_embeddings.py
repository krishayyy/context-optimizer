import numpy as np

from context_optimizer import embeddings


def test_embed_many_returns_none_gracefully_if_unavailable(monkeypatch):
    monkeypatch.setattr(embeddings, "_get_model", lambda: None)
    assert embeddings.embed_many(["hello"]) is None


def test_embed_many_empty_list_returns_empty_array():
    # Doesn't require a real model: the empty-input branch returns before
    # touching _get_model.
    result = embeddings.embed_many([])
    assert result.shape == (0, embeddings.EMBED_DIM)


def test_embed_many_preserves_order_despite_internal_length_sorting(monkeypatch, tmp_path):
    """The core correctness property of the length-bucketing speed fix:
    embed_many sorts texts by length before calling the model (a real
    ~4.8x speedup on real data, see scorer.py's EMBED_TEXT_CAP comment),
    then must scatter results back to the CALLER's original order. A bug
    here would silently mismatch every embedding to the wrong chunk.
    """
    monkeypatch.setattr(embeddings, "CACHE_DIR", tmp_path)

    class FakeModel:
        def embed(self, texts):
            # Deterministic fake: each text's "embedding" just encodes its
            # own length, so we can verify the mapping without a real model.
            for t in texts:
                yield np.array([float(len(t))] * embeddings.EMBED_DIM)

    monkeypatch.setattr(embeddings, "_get_model", lambda: FakeModel())

    # Deliberately NOT sorted: long, short, medium -- exercises the
    # internal reordering.
    texts = ["x" * 50, "y" * 5, "z" * 20]
    result = embeddings.embed_many(texts)

    assert result[0][0] == 50.0
    assert result[1][0] == 5.0
    assert result[2][0] == 20.0


def test_embed_many_uses_cache_on_second_call(monkeypatch, tmp_path):
    monkeypatch.setattr(embeddings, "CACHE_DIR", tmp_path)
    call_count = {"n": 0}

    class FakeModel:
        def embed(self, texts):
            call_count["n"] += 1
            for t in texts:
                yield np.zeros(embeddings.EMBED_DIM)

    monkeypatch.setattr(embeddings, "_get_model", lambda: FakeModel())

    embeddings.embed_many(["same text"])
    embeddings.embed_many(["same text"])
    assert call_count["n"] == 1  # second call was a pure cache hit
