"""Factory function and base class for rerankers."""

from __future__ import annotations

import functools
import importlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

from langchain_core.callbacks import Callbacks
from langchain_core.cross_encoders import BaseCrossEncoder
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from pydantic import BaseModel, ConfigDict
from typing_extensions import override


class RerankResult(BaseModel):
    """A single reranked document together with its relevance score."""

    index: int
    """Index of the document in the input sequence."""

    relevance_score: float
    """Relevance score assigned by the reranker. Higher means more relevant."""

    document: Document
    """The reranked document."""


class BaseReranker(BaseDocumentCompressor, ABC):
    """Base class for rerankers.

    Subclasses implement [`rerank`][langchain_reranker.rerankers.BaseReranker.rerank].
    The inherited `compress_documents` method is implemented in terms of `rerank`, so a
    reranker can be used anywhere a
    [`BaseDocumentCompressor`][langchain_core.documents.compressor.BaseDocumentCompressor]
    is expected (for example inside a contextual compression retriever).
    """

    model: str = ""
    """Name of the reranking model."""

    top_n: int | None = None
    """Number of documents to return. If `None`, all documents are returned."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Rerank `documents` by relevance to `query`.

        Args:
            query: The query to rank the documents against.
            documents: The documents to rerank.
            top_n: Number of results to return. Defaults to `self.top_n`.

        Returns:
            The reranked results, ordered from most to least relevant.
        """

    @override
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """Rerank documents and attach the relevance score to their metadata."""
        _ = callbacks
        return [
            result.document.model_copy(
                update={
                    "metadata": {
                        **result.document.metadata,
                        "relevance_score": result.relevance_score,
                    }
                }
            )
            for result in self.rerank(query=query, documents=documents, top_n=self.top_n)
        ]


class CrossEncoderReranker(BaseReranker):
    """Reranker backed by a cross encoder.

    Example:
        ```python
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        from langchain_reranker.rerankers import CrossEncoderReranker

        cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        reranker = CrossEncoderReranker(cross_encoder=cross_encoder, top_n=3)
        reranker.rerank(query="...", documents=documents)
        ```
    """

    cross_encoder: BaseCrossEncoder
    """Cross encoder used to score query-document pairs."""

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    @override
    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Score each document against the query and sort by relevance."""
        scores = self.cross_encoder.score(
            [(query, document.page_content) for document in documents]
        )
        results = [
            RerankResult(
                index=index,
                relevance_score=float(score),
                document=document,
            )
            for index, (document, score) in enumerate(
                zip(documents, scores, strict=False)
            )
        ]
        results.sort(key=lambda result: result.relevance_score, reverse=True)
        limit = self.top_n if top_n is None else top_n
        if limit is not None:
            results = results[:limit]
        return results


def _call(
    cls: type[Any],
    model: str,
    **kwargs: Any,
) -> BaseDocumentCompressor:
    return cls(model=model, **kwargs)


def _call_cross_encoder(
    cls: type[Any],
    model: str,
    *,
    top_n: int | None = None,
    **kwargs: Any,
) -> BaseReranker:
    return CrossEncoderReranker(
        cross_encoder=cls(model_name=model, **kwargs),
        model=model,
        top_n=top_n,
    )


def _call_with_base_url(default_base_url: str) -> Callable[..., BaseDocumentCompressor]:
    def creator(
        cls: type[Any],
        model: str,
        **kwargs: Any,
    ) -> BaseDocumentCompressor:
        kwargs.setdefault("base_url", default_base_url)
        return cls(model=model, **kwargs)

    return creator


_call_siliconflow = _call_with_base_url("https://api.siliconflow.cn/v1")


_BUILTIN_PROVIDERS: dict[str, tuple[str, str, Callable[..., BaseDocumentCompressor]]] = {
    "bedrock": ("langchain_aws", "AmazonBedrockRerank", _call),
    "cohere": ("langchain_cohere", "CohereRerank", _call),
    "custom": (
        "langchain_reranker.rerankers.http_reranker",
        "HttpReranker",
        _call,
    ),
    "dashscope": (
        "langchain_community.document_compressors",
        "DashScopeRerank",
        _call,
    ),
    "flashrank": (
        "langchain_community.document_compressors",
        "FlashrankRerank",
        _call,
    ),
    "huggingface": (
        "langchain_huggingface",
        "HuggingFaceCrossEncoder",
        _call_cross_encoder,
    ),
    "infinity": (
        "langchain_community.document_compressors",
        "InfinityRerank",
        _call,
    ),
    "jina": ("langchain_community.document_compressors", "JinaRerank", _call),
    "llama_server": (
        "langchain_reranker.rerankers.http_reranker",
        "HttpReranker",
        _call,
    ),
    "local": (
        "langchain_community.cross_encoders",
        "HuggingFaceCrossEncoder",
        _call_cross_encoder,
    ),
    "openvino": (
        "langchain_community.document_compressors",
        "OpenVINOReranker",
        _call,
    ),
    "rankllm": (
        "langchain_community.document_compressors",
        "RankLLMRerank",
        _call,
    ),
    "siliconflow": (
        "langchain_reranker.rerankers.http_reranker",
        "HttpReranker",
        _call_siliconflow,
    ),
    "volcengine": (
        "langchain_community.document_compressors",
        "VolcengineRerank",
        _call,
    ),
    "voyageai": ("langchain_voyageai", "VoyageAIRerank", _call),
}
"""Registry mapping provider names to their import configuration.

Each entry maps a provider key to a tuple of:

- `module_path`: The Python module path containing the reranker class.
- `class_name`: The name of the reranker class to import.
- `creator_func`: A callable that instantiates the class with provided kwargs.

!!! note

    This dict is not exhaustive of all providers supported by LangChain, but is
    meant to cover the most popular ones and serve as a template for adding more
    providers in the future. If a provider is not in this dict, it can still be
    used with `init_reranker` as long as its integration package is installed, but
    the provider key will not be inferred from the model name and must be
    specified explicitly via the `provider` parameter.
"""


@functools.lru_cache(maxsize=len(_BUILTIN_PROVIDERS))
def _get_reranker_class_creator(
    provider: str,
) -> Callable[..., BaseDocumentCompressor]:
    """Return a factory that creates a reranker for the given provider.

    This function is cached to avoid repeated module imports.

    Args:
        provider: The name of the provider (e.g., `'cohere'`, `'jina'`). Must be a
            key in `_BUILTIN_PROVIDERS`.

    Returns:
        A callable that accepts model kwargs and returns a
            `BaseDocumentCompressor` instance for the specified provider.

    Raises:
        ValueError: If the provider is not in `_BUILTIN_PROVIDERS`.
        ImportError: If the provider's integration package is not installed.
    """
    if provider not in _BUILTIN_PROVIDERS:
        msg = (
            f"Provider '{provider}' is not supported.\n"
            f"Supported providers and their required packages:\n"
            f"{_get_provider_list()}"
        )
        raise ValueError(msg)

    module_name, class_name, creator_func = _BUILTIN_PROVIDERS[provider]
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        pkg = module_name.split(".", maxsplit=1)[0].replace("_", "-")
        msg = f"Could not import {pkg} python package. Please install it with `pip install {pkg}`"
        raise ImportError(msg) from e

    cls = getattr(module, class_name)
    return functools.partial(creator_func, cls=cls)


def _get_provider_list() -> str:
    """Get formatted list of providers and their packages."""
    return "\n".join(
        f"  - {p}: {pkg[0].split('.', maxsplit=1)[0].replace('_', '-')}"
        for p, pkg in _BUILTIN_PROVIDERS.items()
    )


def _parse_model_string(model_name: str) -> tuple[str, str]:
    """Parse a model string into provider and model name components.

    The model string should be in the format `'provider:model-name'`, where
    provider is one of the supported providers.

    Args:
        model_name: A model string in the format `'provider:model-name'`.

    Returns:
        A tuple of `(provider, model_name)`.

    Raises:
        ValueError: If the model string is not in the correct format or the
            provider is unsupported.
    """
    if ":" not in model_name:
        msg = (
            f"Invalid model format '{model_name}'.\n"
            f"Model name must be in format 'provider:model-name'\n"
            f"Example valid model strings:\n"
            f"  - cohere:rerank-v3.5\n"
            f"  - jina:jina-reranker-v2-base-multilingual\n"
            f"  - huggingface:BAAI/bge-reranker-v2-m3\n"
            f"Supported providers: {_BUILTIN_PROVIDERS.keys()}"
        )
        raise ValueError(msg)

    provider, model = model_name.split(":", 1)
    provider = provider.lower().strip()
    model = model.strip()

    if provider not in _BUILTIN_PROVIDERS:
        msg = (
            f"Provider '{provider}' is not supported.\n"
            f"Supported providers and their required packages:\n"
            f"{_get_provider_list()}"
        )
        raise ValueError(msg)
    if not model:
        msg = "Model name cannot be empty"
        raise ValueError(msg)
    return provider, model


def _infer_model_and_provider(
    model: str,
    *,
    provider: str | None = None,
) -> tuple[str, str]:
    if not model.strip():
        msg = "Model name cannot be empty"
        raise ValueError(msg)
    if provider is None and ":" in model:
        provider, model_name = _parse_model_string(model)
    else:
        model_name = model

    if not provider:
        msg = (
            "Must specify either:\n"
            "1. A model string in format 'provider:model-name'\n"
            "   Example: 'cohere:rerank-v3.5'\n"
            "2. Or explicitly set provider from: "
            f"{_BUILTIN_PROVIDERS.keys()}"
        )
        raise ValueError(msg)

    if provider not in _BUILTIN_PROVIDERS:
        msg = (
            f"Provider '{provider}' is not supported.\n"
            f"Supported providers and their required packages:\n"
            f"{_get_provider_list()}"
        )
        raise ValueError(msg)
    return provider, model_name


def init_reranker(
    model: str,
    *,
    provider: str | None = None,
    **kwargs: Any,
) -> BaseDocumentCompressor:
    """Initialize a reranker from a model name and optional provider.

    !!! note

        Requires the integration package for the chosen provider to be installed.

        See the `provider` parameter below for specific package names
        (e.g., `pip install langchain-cohere`).

    Args:
        model: The name of the model, e.g. `'cohere:rerank-v3.5'`.

            You can also specify model and provider in a single argument using
            the `'{provider}:{model}'` format, e.g. `'cohere:rerank-v3.5'`.
        provider: The provider if not specified as part of the model arg.

            Supported `provider` values and the corresponding integration package
            are:

            - `bedrock`      -> `langchain-aws`
            - `cohere`       -> `langchain-cohere`
            - `custom`       -> `langchain-reranker` (any Cohere-style HTTP endpoint)
            - `dashscope`    -> `langchain-community`
            - `flashrank`    -> `langchain-community`
            - `huggingface`  -> `langchain-huggingface`
            - `infinity`     -> `langchain-community`
            - `jina`         -> `langchain-community`
            - `llama_server` -> `langchain-reranker` (llama.cpp `llama-server`)
            - `local`        -> `langchain-community` (sentence-transformers)
            - `openvino`     -> `langchain-community`
            - `rankllm`      -> `langchain-community`
            - `siliconflow`  -> `langchain-reranker` (SiliconFlow)
            - `volcengine`   -> `langchain-community`
            - `voyageai`     -> `langchain-voyageai`

        **kwargs: Additional model-specific parameters passed to the reranker.

    Returns:
        A `BaseDocumentCompressor` that reranks documents for a query.

    Raises:
        ValueError: If the provider is not supported or cannot be determined.
        ImportError: If the required provider package is not installed.

    ???+ example

        ```python
        # pip install langchain-reranker langchain-cohere

        # Using a model string
        reranker = init_reranker("cohere:rerank-v3.5")
        reranker.compress_documents(documents, query="...")

        # Using an explicit provider
        reranker = init_reranker(model="jina-reranker-v2-base-multilingual", provider="jina")

        # With additional parameters
        reranker = init_reranker("cohere:rerank-v3.5", top_n=5)
        ```
    """
    if not model:
        providers = ", ".join(_BUILTIN_PROVIDERS.keys())
        msg = f"Must specify model name. Supported providers are: {providers}"
        raise ValueError(msg)

    provider, model_name = _infer_model_and_provider(model, provider=provider)
    return _get_reranker_class_creator(provider)(model=model_name, **kwargs)


__all__ = ["BaseReranker", "CrossEncoderReranker", "RerankResult", "init_reranker"]
