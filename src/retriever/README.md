# src/`Retriever`

## Description

Turns a free-form user query into the most relevant messages. It embeds the query, runs vector search, and reranks the candidates for better final results.

## Responsibilities
- Receive search requests from `src/bot/`.
- Apply the same privacy masking to the query.
- Create a query embedding with the configured model.
- Run k-NN over the database vector index.
- Rerank and filter candidates (external LLM or lightweight reranker).
- Return message IDs + snippets + scores (optional).