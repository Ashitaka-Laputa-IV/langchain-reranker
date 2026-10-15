"""Tests for `init_reranker` and the reranker classes."""

import pytest
from langchain_core.cross_encoders import BaseCrossEncoder
from langchain_core.documents import Document

from langchain_reranker.rerankers import (
    BaseReranker,
    CrossEncoderReranker,
    RerankResult,
    init_reranker,
)


class ToyReranker(BaseReranker):
    """Reranker that scores documents by their length."""

    model: str = "toy"

    def rerank(
        self,
        query: str,
        documents,
        top_n: int | None = None,
    ) -> list[RerankResult]:
        scored = sorted(
            zip(documents, [len(doc.page_content) for doc in documents]),
            key=lambda pair: -pair[1],
        )
        results = [
            RerankResult(index=index, relevance_score=float(score), document=doc)
            for index, (doc, score) in enumerate(scored)
        ]
        return results[:top_n] if top_n else results


@pytest.fixture
def documents() -> list[Document]:
    return [
        Document(page_content="banana"),
        Document(page_content="apple phone"),
    ]


def test_compress_documents_bridges_rerank(documents: list[Document]) -> None:
    out = ToyReranker(top_n=1).compress_documents(documents, query="apple")
    assert [doc.page_content for doc in out] == ["apple phone"]
    assert out[0].metadata["relevance_score"] == 11.0


def test_cross_encoder_reranker(documents: list[Document]) -> None:
    class FakeEncoder(BaseCrossEncoder):
        def score(self, text_pairs):
            return [float(index) for index, _ in enumerate(text_pairs)]

    results = CrossEncoderReranker(
        cross_encoder=FakeEncoder(),
        top_n=1,
    ).rerank("apple", documents)

    assert [result.index for result in results] == [1]
    assert results[0].relevance_score == 1.0


@pytest.mark.parametrize(
    "call",
    [
        lambda: init_reranker("unknown:model"),
        lambda: init_reranker("no-colon"),
        lambda: init_reranker("model", provider="unknown"),
        lambda: init_reranker(""),
    ],
)
def test_invalid_inputs_raise_value_error(call) -> None:
    with pytest.raises(ValueError):
        call()


def test_missing_package_raises_import_error(monkeypatch) -> None:
    from langchain_reranker.rerankers import base

    monkeypatch.setitem(
        base._BUILTIN_PROVIDERS,
        "fake",
        ("definitely_not_a_real_module", "Thing", base._call),
    )
    base._get_reranker_class_creator.cache_clear()
    try:
        with pytest.raises(
            ImportError,
            match="pip install definitely-not-a-real-module",
        ):
            init_reranker("fake:model")
    finally:
        base._get_reranker_class_creator.cache_clear()
