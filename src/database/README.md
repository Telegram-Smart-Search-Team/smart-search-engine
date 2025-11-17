# src/`Database`

## Description

Storage and retrieval backbone for Telegram Smart Search. This module hides the details of your underlying datastore and vector index while providing a clean API for the rest of the system.

In order to start the database, please run the chromadb server with: ```chroma run --host localhost --port 8000 --path ./chroma_server_data```. Provides a simple and documented API for the vector DB in order to potentially support any vector DB in future.

## What it talks to:
- **Input:** `src/dumper/` writes messages, embeddings, and metadata.
- **Output:** `src/retriever/` queries vectors and fetches message payloads.