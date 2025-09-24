# RAG Systems: Hands-On Lecture
How to go from retrieval to generation

## Setup

The vector database requires Python <= 3.12

```bash
conda create -n ragnificent python=3.12
conda activate ragnificent
pip install torch==2.8.0 torchvision==0.23.0
pip install -r requirements.txt
```

## Part 1: LangChain Documents and How to Load Them

First, we need documents to work with. Let's explore different ways to load them.

### The Document Class

```python
from langchain_core.documents import Document

page_content = "Hello, world!"

docs = []
doc = Document(
    page_content=page_content,
    metadata={
        "source": "My custom document",
        "title": "Title of my custom document"
    }
)
docs.append(doc)

print("-"*50)
print("Metadata:")
for k, v in docs[0].metadata.items():
    print(f"{k}: {v}")
print("-"*50)
print("Page content (preview):")
print(docs[0].page_content if len(docs[0].page_content) < 500 else docs[0].page_content[:500] + "...")
```

### Convenience Loaders: PyPDFLoader

```python
from langchain_community.document_loaders import PyPDFLoader

pdf_path = "context/Weller et al. - 2025 - On the Theoretical Limitations of Embedding-Based Retrieval.pdf"

loader = PyPDFLoader(
    file_path=pdf_path,
    mode="single"
)
docs = loader.load()

print(f"Loaded PDF as {len(docs):d} document(s)")
print("-"*50)
print("Metadata:")
for k, v in docs[0].metadata.items():
    print(f"{k}: {v}")
print("-"*50)
print("Page content (preview):")
print(docs[0].page_content[:500] + "...")
```

### Convenience Loaders: WikipediaLoader

```python
from langchain_community.document_loaders import WikipediaLoader

loader = WikipediaLoader(
    query="What is the capital of France?",
)
docs = loader.load()

print(f"Loaded {len(docs):d} document(s)")
print("-"*50)
for doc in docs:
    print(doc.metadata)
```

### Convenience Loaders: WebBaseLoader

For demonstration purposes, the `WebBaseLoader` can provide timely context that is not included in any LLM's training data. For instance, the NFL scores and highlights from last week: https://www.cbssports.com/nfl/news/nfl-week-3-grades-scores-results-highlights-browns-packers-vikings-bengals/.

Note: Different websites require different parsing strategies.

```python
import bs4
from langchain_community.document_loaders import WebBaseLoader
import os
import re
os.environ['USER_AGENT'] = ("Demo")

loader = WebBaseLoader(
    web_paths=("https://www.cbssports.com/nfl/news/nfl-week-3-grades-scores-results-highlights-browns-packers-vikings-bengals/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer("article")
    ),
)

docs = loader.load()

# Clean and condense page content: strip, remove blank lines, trim lines, and join into a single line
docs[0].page_content = " ".join(
    line.strip() for line in docs[0].page_content.strip().splitlines() if line.strip()
)

# Replace sequences of two or more spaces with a single space
docs[0].page_content = re.sub(r'\s{2,}', ' ', docs[0].page_content)

print(f"Loaded Website as {len(docs):d} document(s)")
print("-"*50)
print("Metadata:")
for k, v in docs[0].metadata.items():
    print(f"{k}: {v}")
print("-"*50)
print("Page content (preview):")
print(docs[0].page_content if len(docs[0].page_content) < 10000 else docs[0].page_content[:10000] + "...")
```

## Part 2: Document Chunking

In RAG systems, it is common to split large documents into smaller chunks for effective retrieval.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Split document into chunks for vector storage
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
)
doc_chunks = text_splitter.split_documents(docs)

print(f"Document split into {len(doc_chunks):d} chunks.")

# Show first few chunks for verification
for index, chunk in enumerate(doc_chunks[:3]):
    print(f"\n\nChunk {index+1}")
    print("-"*50)
    print("-"*50)
    print("Metadata:")
    for k, v in chunk.metadata.items():
        print(f"{k}: {v}")
    print("-"*50)
    print("Page content:")
    print(chunk.page_content if len(chunk.page_content) < 500 else chunk.page_content[:500] + "...")
```

## Part 3: Embedding Documents

Now we'll embed our text chunks into vectors using a pre-trained embedding model.

### Setting up the Embedding Model and Vector Store

There are many pre-trained embedding models available on **HuggingFace**, here are some examples:
- intfloat/multilingual-e5-base
- Qwen/Qwen2.5-0.5B
- Qwen/Qwen3-Embedding-0.6B
- Qwen/Qwen3-Embedding-4B
- sentence-transformers/all-MiniLM-L6-v2

Choose an embedding model and set it in the following code block:
```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model_name = "<embedding_model_name>"

embedding_function = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)
vector_store = InMemoryVectorStore(embedding=embedding_function)
```

### Adding Documents to Vector Store

```python
# Add documents to vector store
document_chunk_ids = vector_store.add_documents(documents=doc_chunks)

print(f"Added {len(document_chunk_ids):d} documents to the vector store")
```

### Inspecting the Vector Store

```python
# Inspect vector store
n_chunks = 5
for index, (id, doc) in enumerate(vector_store.store.items()):
    if index < n_chunks:
        print("\n"+"-"*100)
        print(f"Chunk {index+1}\n")
        print(f"id: {id}")
        print(f"vector (length: {len(doc['vector'])}): {doc['vector']}")
        print(f"metadata: {doc['metadata']}")
        print(f"text:\n{doc['text'][:100]}")
    else:
        break
```

## Part 4: Testing Vector Similarity Search

Let's test our vector store with a sample query. This is a useful test to verify that the embedding model is working correctly and to compare the quality of the different embedding models.

```python
# Test vector store with a sample query
query = "Who was the Vikings' starting quarterback in week 3?"

top_k = 5
similar_document_chunks = vector_store.similarity_search_with_score(query, k=top_k)

print(f"List of {len(similar_document_chunks):d} most similar document chunks for query: '{query:s}'")
for i, (doc, score) in enumerate(similar_document_chunks):
    if i < top_k:
        print("\n" + "-"*50)
        print(f"Result {i+1} (Similarity Score: {score:.4f})")
        print(f"\tid: {doc.id}")
        print(f"\tmetadata: {doc.metadata}")
        print(f"\tpage content: {doc.page_content if len(doc.page_content) < 300 else doc.page_content[:300] + '...'}")
    else:
        break
```

## Part 5: Setting up the Language Model

Now we need an LLM to generate answers based on retrieved context.

### Creating the Prompt Template

This `ChatPromptTemplate` has two placeholders. The `{context}`, which will be replaced with the retrieved document chunks from the vector store, and the `{input}`, which will be replaced with the user's question.

```python
from langchain_core.prompts import ChatPromptTemplate

# Create prompt template for RAG system
chat_prompt = ChatPromptTemplate.from_messages([
    ("system",
    """
    You are an AI assistant that answers questions based on provided context documents. 
    """
    ),
    ("human",
    """
    Answer the question based on the context.

    CRITICAL RULES:
    - Answer concisely
    - Use information from the provided context
    - If the context doesn't contain enough information, state this clearly
    - Cite specific details from the context when possible

    CONTEXT:
    {context}

    QUESTION: {input}

    ANSWER:
    """
    )
])
```

### Loading the Language Model

Again, there are many pre-trained language models available on HuggingFace, here are some examples:
- google/gemma-3-1b-it
- google/gemma-3-4b-it

For the purpose of this demo, it can be interesting to try different models, even some lower-performance ones, to see the impact of the context provided through the RAG system.

Choose a language model and set it in the following code block:
```python
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "<model_name>"
print(f"Loading llm <{model_name:s}>")

# Load LLM
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create text generation pipeline
text_generation_pipeline = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=250,    # Limit answer length for concise RAG responses
    temperature=0.2,       # Low randomness, mostly deterministic
    top_p=0.95,            # Sample from top 95% probable tokens
    repetition_penalty=1.2 # Penalize repeated content to improve answer quality
)

# Wrap pipeline for LangChain
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
```

## Part 6: Building the RAG Chain

### The Document Combination Chain

Here we are using the `create_stuff_documents_chain` to combine the retrieved document chunks into a single context for the LLM.

It takes your docs and smooshes them all together into one big text. Then it takes your `ChatPromptTemplate` (the template that says "Hey AI, here's some context + my question...") and fills in the template with the smooshed-together documents. Finally, it sends everything to the LLM, which will generate a response.

```python
from langchain.chains.combine_documents import create_stuff_documents_chain

combine_docs_chain = create_stuff_documents_chain(llm, chat_prompt)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.7,
    }
)
```

### Testing the Document Chain

```python
secret_spider_man_docs = [
    Document(
    page_content="My first name is Peter.",
    metadata={"doc-nr.": "1"}
    ),
    Document(
    page_content="My last name is Parker.",
    metadata={"doc-nr.": "2"}
    )
]

response = combine_docs_chain.invoke({"context": secret_spider_man_docs, "input": "What is the person's full name?"})
print(f"Response:")
print("-"*50)
print(f"{response:s}")
```

### The Retrieval Chain (aka the RAG system!)

This combines retrieval with generation.

```python
from langchain.chains import create_retrieval_chain

retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

response = retrieval_chain.invoke({"input": "What is this document about?"})
```

### Examining the RAG Response

```python
print("ANSWER")
print("-"*50)
print(f"{response['answer']:s}")
print("\n\n")

print("CONTEXT")
print("-"*50)
for i, doc in enumerate(response['context']):
    print(f"Document {i+1}:")
    for key, value in doc.metadata.items():
        print(f"\t{key}: {value}")
    print(f"\tPage content: {doc.page_content if len(doc.page_content) < 300 else doc.page_content[:300] + '...'}")
    print("\n")
print("\n\n")
```

## Part 7: Basic LLM vs. RAG: The Final Comparison

Let's see the difference between an LLM with and without context. For demonstration purposes, we'll use the same question from earlier, to which the LLM can't provide an accurate answer.

```python
question = "Who was the Vikings' starting quarterback in week 3?"

print("LLM without context:")
print("-"*50)
llm_response = llm.invoke(question)
print(f"Answer: {llm_response}")
print("\n\n")

print("RAG system with context:")
print("-"*50)
rag_response = retrieval_chain.invoke({"input": question})
print(f"Answer: {rag_response['answer']}")

print("Context:")
for i, doc in enumerate(rag_response['context']):
    print(f"Document {i+1}:")
    for key, value in doc.metadata.items():
        print(f"\t{key}: {value}")
    print(f"\tPage content: {doc.page_content if len(doc.page_content) < 300 else doc.page_content[:300] + '...'}")
    print("\n")
print("\n\n")
```