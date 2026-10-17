# IBM Hybrid RAG for Technical Documentation

An end-to-end **Hybrid Retrieval-Augmented Generation (RAG)** project for answering technical questions about IBM products using semantic search, keyword-based retrieval, and locally hosted language models.

**Tech Stack:** Python · Google Colab · PostgreSQL · pgvector · HNSW · Qwen3 · BM25 · Reciprocal Rank Fusion (RRF) · Ollama · Llama 3.2

## Project Overview

This project focuses on building a Hybrid RAG system for technical IBM documentation, where specialized terminology, configuration parameters, and error codes are important for accurate information retrieval.

Instead of relying on traditional fixed-size chunking, the project implements a **hierarchical parent-child chunking strategy**. Large semantic chunks are first created using a language model, then divided and optimized into smaller overlapping chunks for retrieval.

The system combines **Dense Vector Search and BM25** using Reciprocal Rank Fusion (RRF). Once relevant retrieval chunks are identified, their original semantic parent chunks are recovered and passed to Llama 3.2 for context-grounded answer generation.

The complete pipeline was developed and tested using Google Colab, PostgreSQL, pgvector, and locally hosted models through Ollama.

## Implementation

### 1. Document Preparation and Semantic Chunking

The first notebook, `Chunk_process.ipynb`, processes IBM technical documentation from the [NVIDIA TechQA-RAG-Eval dataset](https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval).

The initial processing includes extracting documentation passages, cleaning technical text, and identifying duplicate documents using SHA-256 content hashes.

Instead of immediately splitting documents into fixed-length chunks, I used **Qwen3 14B (`qwen3:14b`) through Ollama** to identify meaningful semantic boundaries within each document.

The model was used to detect changes in topics and logically independent sections, helping preserve the context of technical explanations.

The identified boundaries were mapped back to the original document text to preserve the source content.

**Results:**
- 496 unique IBM technical documents
- 3,200 semantic parent chunks

This approach creates larger, semantically coherent chunks that retain the context needed for technical question answering.

### 2. Retrieval Chunking and Overlap Optimization

Although semantic parent chunks preserve meaningful context, some of them are too large for precise information retrieval.

To address this, I implemented a second chunking stage.

Semantic parent chunks shorter than or equal to **800 characters** were preserved as individual retrieval chunks. Larger parents were divided into smaller child chunks while prioritizing paragraph, line, and whitespace boundaries.

This stage initially produced **5,640 retrieval chunks**.

I then implemented an additional optimization process to improve the quality of small retrieval chunks.

Chunks shorter than **150 characters** were merged with neighboring text when possible, while maintaining the maximum chunk size of 800 characters.

In some cases, small text fragments were included in adjacent chunks to create **controlled overlap** and improve contextual continuity.

Unlike traditional sliding-window chunking, the overlap was introduced selectively during optimization rather than using a fixed overlap percentage.

**Final results:**
- 5,640 initial retrieval chunks
- 5,098 optimized retrieval chunks

Each retrieval chunk also retains the identifiers of its original semantic parent chunks through the `source_semantic_ids` field in PostgreSQL.

This relationship is essential for retrieving the original semantic context after a child chunk is selected.

### 3. Embedding Generation and Vector Indexing

The second notebook, `save_embedding.ipynb`, implements the embedding and vector storage pipeline.

I used **Qwen3 Embedding 0.6B (`qwen3-embedding:0.6b`)** through Ollama to generate vector representations for all optimized retrieval chunks.

Each chunk was converted into a **1,024-dimensional embedding**.

The embeddings were stored in PostgreSQL using the **pgvector** extension, together with their corresponding chunk identifiers and content hashes.

To support efficient approximate nearest-neighbor retrieval, I created an **HNSW index** using cosine distance.

**Results:**
- 5,098 stored chunk embeddings
- 1,024 embedding dimensions
- PostgreSQL with pgvector
- HNSW vector index

The completed database was also backed up so that the Hybrid RAG notebook could reuse the existing embeddings without regenerating them.

### 4. Hybrid Retrieval — Dense Search, BM25, and RRF

The third notebook, `Hybrid_RAG.ipynb`, implements the Hybrid Retrieval pipeline.

Two different search methods are used:

**Dense Vector Search**

The user's question is converted into an embedding using the same Qwen3 embedding model used for the document chunks.

PostgreSQL and pgvector are then used to retrieve semantically similar chunks based on cosine distance.

The HNSW index supports efficient vector similarity search.

**BM25 Lexical Search**

A separate BM25 retrieval system is built using the optimized retrieval chunk texts.

BM25 ranks documents using lexical information such as term frequency, document frequency, and document length.

This method is especially useful for technical terminology, product names, and exact error codes.

**Reciprocal Rank Fusion (RRF)**

Since Dense Search and BM25 produce scores on different scales, their raw scores are not directly combined.

Instead, I implemented **Reciprocal Rank Fusion (RRF)** with `k = 60` to merge the two ranked result lists.

RRF assigns higher combined scores to chunks that achieve strong rankings across the retrieval methods.

This creates a single Hybrid Retrieval ranking that benefits from both semantic similarity and lexical matching.

### 5. Semantic Parent Expansion

After retrieving the highest-ranked child chunks, the pipeline identifies their corresponding semantic parents.

The relationship between each retrieval chunk and its original semantic parent is stored in PostgreSQL through the `source_semantic_ids` field.

Using these identifiers, the system retrieves the larger semantic parent chunks from the database.

Repeated parent chunks are removed before preparing the context for answer generation.

This implements a **small-to-large retrieval strategy**:

- Smaller child chunks provide more focused retrieval.
- Larger semantic parent chunks preserve additional context.
- The expanded context is supplied to the language model for answer generation.

This approach combines retrieval precision with richer source information.

### 6. Answer Generation with Llama 3.2

For answer generation, I used **Llama 3.2 3B (`llama3.2:3b`)** through Ollama.

The language model receives the user question together with the semantic parent chunks selected by the Hybrid Retrieval pipeline.

Prompt engineering is used to encourage the model to:

- Generate answers based on the retrieved documentation.
- Avoid inventing unsupported technical commands or configuration steps.
- Reference the retrieved sources.
- Acknowledge when the available documentation does not provide sufficient information.

The end-to-end pipeline connects Hybrid Retrieval, semantic parent expansion, and LLM-based answer generation.

### 7. Retrieval Evaluation

I conducted a preliminary evaluation to compare the retrieval performance of **Dense Search, BM25, and Hybrid RRF**.

Using **Qwen3 8B (`qwen3:8b`)**, I generated eight synthetic technical questions from eight different IBM source documents.

Each question was associated with the semantic parent chunk from which it was generated.

The evaluation measured whether the retrieval methods could recover a child chunk linked to that known reference parent.

Three retrieval metrics were calculated:

- **Hit@5:** Whether the reference parent was represented within the top five retrieved chunks.
- **MRR@10:** The reciprocal rank of the first matching chunk within the top ten results.
- **Single-reference nDCG@10:** A simplified ranking measure based on the first matching reference parent.

**Evaluation Results**

| Metric | Dense | BM25 | Hybrid RRF |
|---|---:|---:|---:|
| Hit@5 | 0.750 | 0.875 | **0.875** |
| MRR@10 | 0.531 | 0.646 | **0.750** |
| Single-reference nDCG@10 | 0.615 | 0.704 | **0.783** |

In this experiment, Hybrid RRF achieved the highest MRR@10 and single-reference nDCG@10, while matching BM25 on Hit@5.

These results suggest that combining semantic and lexical retrieval can improve the ranking of relevant technical documentation.

**Evaluation Note:** This is a preliminary retrieval experiment using eight synthetic questions generated from the indexed documentation. The single-reference nDCG calculation is simplified and is not equivalent to standard nDCG with complete relevance judgments. The reported metrics evaluate retrieval rather than the factual correctness of generated answers.

## Dataset and System Summary

| Component | Details |
|---|---|
| Dataset | NVIDIA TechQA-RAG-Eval (IBM technical documentation) |
| Documents | 496 |
| Semantic parent chunks | 3,200 |
| Initial retrieval chunks | 5,640 |
| Optimized retrieval chunks | 5,098 |
| Stored embeddings | 5,098 |
| Embedding dimensions | 1,024 |
| Semantic chunking model | Qwen3 14B |
| Embedding model | Qwen3 Embedding 0.6B |
| Vector database | PostgreSQL + pgvector |
| Vector indexing | HNSW |
| Lexical retrieval | BM25 |
| Hybrid ranking | Reciprocal Rank Fusion |
| Answer generation | Llama 3.2 3B |
| Evaluation question generation | Qwen3 8B |

## Repository Structure

The repository contains three Jupyter notebooks:

| Notebook | Description |
|---|---|
| `Chunk_process.ipynb` | Document preprocessing, LLM-based semantic chunking, retrieval chunking, overlap optimization, and PostgreSQL storage |
| `save_embedding.ipynb` | Embedding generation, pgvector storage, HNSW indexing, and database backup |
| `Hybrid_RAG.ipynb` | Dense retrieval, BM25, RRF, semantic parent expansion, Llama answer generation, and retrieval evaluation |

## Running the Project

The project was developed using **Google Colab** with GPU support.

PostgreSQL, pgvector, and Ollama are configured through the notebook setup steps.

Run the notebooks in the following order:

**1. Chunk_process.ipynb**

Preprocess IBM documentation, generate semantic parent chunks using Qwen3 14B, create optimized retrieval chunks, and store the results in PostgreSQL.

**2. save_embedding.ipynb**

Restore the processed database, generate Qwen3 embeddings, create the HNSW vector index, and export the completed PostgreSQL backup.

**3. Hybrid_RAG.ipynb**

Restore the completed database, initialize the retrieval components, run Hybrid Search, expand semantic parents, generate answers using Llama 3.2, and evaluate retrieval performance.

The notebooks contain the necessary environment setup and installation steps.

**Note:** The database backups, processed documentation, and model files are not included in this repository. Google Drive paths must be adjusted for the target environment, and the database artifacts produced in earlier stages are needed for subsequent notebooks.

## Model and Hardware Considerations

This project was developed using **freely available, locally hosted models through Ollama**, rather than relying on paid commercial LLM APIs.

Response quality and inference performance depend on the selected model capabilities and available GPU resources.

Using locally hosted models provides a **cost-effective and self-hosted approach** to experimenting with Hybrid RAG systems.

## Data Attribution

This project uses IBM technical documentation from the [NVIDIA TechQA-RAG-Eval dataset](https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval).

Original documentation, processed database backups, and stored embeddings are not redistributed in this repository. Reuse and redistribution of source material remain subject to the applicable dataset and content terms.

This is an independent educational and portfolio project and is not affiliated with or endorsed by IBM.
