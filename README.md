# langchain-reranker

[![PyPI version](https://img.shields.io/pypi/v/langchain-reranker.svg)](https://pypi.org/project/langchain-reranker/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/langchain-reranker.svg)](https://pypi.org/project/langchain-reranker/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Add `init_reranker` to the LangChain ecosystem, following the same conventions as
`init_chat_model` and `init_embeddings`.

## Installation

Available on [PyPI](https://pypi.org/project/langchain-reranker/):

```bash
pip install langchain-reranker
```

Provider integration packages are optional and installed on demand, e.g.
`pip install langchain-cohere`.

## Usage

```python
from langchain_reranker.rerankers import init_reranker

reranker = init_reranker("cohere:rerank-v3.5")

compressed = reranker.compress_documents(documents, query="what is langchain?")
```

`init_reranker` returns a `BaseDocumentCompressor`, so it plugs directly into
existing retrieval pipelines such as a contextual compression retriever.

```python
reranker = init_reranker(model="jina-reranker-v2-base-multilingual", provider="jina")
```

## Custom rerankers

Implement `rerank` and inherit `compress_documents` for free.

```python
from langchain_reranker.rerankers import BaseReranker, RerankResult


class MyReranker(BaseReranker):
    model: str = "my-reranker"

    def rerank(self, query, documents, top_n=None):
        scores = my_api(query, [doc.page_content for doc in documents])
        ranked = sorted(zip(documents, scores), key=lambda pair: -pair[1])
        return [
            RerankResult(index=i, relevance_score=float(score), document=doc)
            for i, (doc, score) in enumerate(ranked)
        ]
```

## Local cross encoders

```python
reranker = init_reranker("huggingface:BAAI/bge-reranker-v2-m3")
```
