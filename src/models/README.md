# src/`Models`

## Description

A single, programmable interface over multiple LLM providers (OpenAI, local HTTP/vLLM/Ollama/LM Studio, remote custom endpoints). You control **which models run** and **in what proportions**. Supports chat and embeddings, streaming, fallbacks, rate limits, and basic cost/usage accounting.

## Responsibilities
- Unifies providers behind one interface (`Chat`, `Embeddings`).
- Routes traffic across models with configurable weights/strategies.
- Falls back on failures, retries transient errors.
- Streams tokens or returns full text.
- Tracks usage (tokens, requests, latency) per provider.
- Works local or remote (localhost vLLM/Ollama/LM Studio, or cloud APIs).