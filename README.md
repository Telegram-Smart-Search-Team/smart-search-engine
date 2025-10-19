![logo](pictures/logo.png)

# Telegram Smart Search

Turn your Telegram history into a private, semantic, multimodal knowledge base. Text, images, voice notes, and files become searchable by meaning — with a local privacy layer that masks sensitive data before anything is embedded or stored.

## What this project does:
- **Ingests** all incoming Telegram messages (text, images, audio, files).
- **Normalizes** content (OCR for images, transcription for audio, images can also go straight to embeddings via CLIP).
- **Redacts** sensitive strings locally (passwords, tokens, secrets) before storage.
- **Embeds** messages and **stores** them with metadata in a vector-backed database.
- **Retrieves** the most relevant messages by semantic similarity and reranks them for a final answer.

## Licence

MIT