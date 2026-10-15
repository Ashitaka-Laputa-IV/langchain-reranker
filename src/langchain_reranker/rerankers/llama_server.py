"""Reranker backed by a llama.cpp `llama-server` rerank endpoint."""

from __future__ import annotations

from collections.abc import Sequence

import httpx
from langchain_core.documents import Document
from typing_extensions import override

from langchain_reranker.rerankers.base import BaseReranker, RerankResult


class LlamaServerReranker(BaseReranker):
    """Reranker that calls a `llama-server` rerank endpoint over HTTP.

    Works with any server exposing the Cohere-style rerank API
    (`{model, query, documents, top_n}` -> `{results: [{index, relevance_score}]}`),
    such as llama.cpp's `llama-server` or vLLM.

    Example:
        ```python
        from langchain_reranker.rerankers import init_reranker

        reranker = init_reranker(
            "llama_server:bge-reranker",
            base_url="http://127.0.0.1:9932",
        )
        reranker.compress_documents(documents, query="what is langchain?")
        ```
    """

    base_url: str
    """Base URL of the server, e.g. `http://127.0.0.1:9932`."""

    endpoint: str = "/rerank"
    """Path of the rerank endpoint."""

    api_key: str | None = None
    """API key sent as `Authorization: Bearer <api_key>` when set."""

    timeout: float = 30.0
    """Request timeout in seconds."""

    @override
    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Score each document against the query and sort by relevance."""
        limit = self.top_n if top_n is None else top_n
        payload: dict[str, object] = {
            "model": self.model,
            "query": query,
            "documents": [document.page_content for document in documents],
        }
        if limit is not None:
            payload["top_n"] = limit

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = httpx.post(
            f"{self.base_url.rstrip('/')}{self.endpoint}",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return [
            RerankResult(
                index=item["index"],
                relevance_score=float(item["relevance_score"]),
                document=documents[item["index"]],
            )
            for item in response.json()["results"]
        ]


__all__ = ["LlamaServerReranker"]
