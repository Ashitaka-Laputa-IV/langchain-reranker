"""Backwards compatible alias for `HttpReranker`."""

from langchain_reranker.rerankers.http_reranker import HttpReranker

LlamaServerReranker = HttpReranker

__all__ = ["LlamaServerReranker"]
