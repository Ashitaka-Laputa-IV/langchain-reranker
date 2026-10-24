"""Reranker backed by a generic Cohere-style HTTP rerank endpoint."""

from __future__ import annotations

from collections.abc import Sequence

import httpx
from langchain_core.documents import Document
from typing_extensions import override

from langchain_reranker.rerankers.base import BaseReranker, RerankResult


class HttpReranker(BaseReranker):
    """Reranker that calls a Cohere-style HTTP rerank endpoint.

    The request body is `{model, query, documents, top_n}` and the response is
    expected to contain `{"results": [{"index": ..., "relevance_score": ...}]}`.

    Works with llama.cpp `llama-server`, vLLM, SiliconFlow and other compatible
    services; only `base_url` (and usually `api_key`) needs to change.

    Example:
        ```python
        from langchain_reranker.rerankers import init_reranker

        reranker = init_reranker(
            "custom:Qwen/Qwen3-Reranker-4B",
            base_url="https://api.siliconflow.cn/v1",
            api_key="sk-...",
        )
        reranker.compress_documents(documents, query="what is langchain?")
        ```
    """

    base_url: str
    """Base URL of the server, e.g. `https://api.siliconflow.cn/v1`."""

    endpoint: str = "/rerank"
    """Path of the rerank endpoint, appended to `base_url`."""

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


__all__ = ["HttpReranker"]
