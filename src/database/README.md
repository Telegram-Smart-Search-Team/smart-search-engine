# src/`Database`

## Description

Storage and retrieval backbone for Telegram Smart Search. This module hides the details of your underlying datastore and vector index while providing a clean API for the rest of the system.

## What it talks to
- **Input:** `src/dumper/` writes messages, embeddings, and metadata.
- **Output:** `src/retriever/` queries vectors and fetches message payloads.