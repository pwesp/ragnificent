# ragnificent
How to go from retrieval to generation

## Installation

The vector database requires Python <= 3.12

```bash
conda create -n ragnificent python=3.12
conda activate ragnificent
pip install torch==2.8.0 torchvision==0.23.0
pip install -r requirements.txt
```

## Basics

Langchain Document: https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html#

Document loaders: https://python.langchain.com/api_reference/community/document_loaders.html


### Final test

```bash
Question: Who was the Vikings' starting quarterback in week 3?
```

```bash


LLM without context:
--------------------------------------------------
Answer: The Vikings started their season with a strong performance, and they had a great start to the game against the New Orleans Saints. The Vikings’ starting quarterback for that game was Kirk Cousins.

RAG system with context:
--------------------------------------------------
Answer: The Vikings started with Carson Wentz.
```