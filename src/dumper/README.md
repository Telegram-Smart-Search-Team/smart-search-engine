# src/`Dumper`

## Description

The ingestion worker. It converts raw Telegram updates into clean, privacy-safe, searchable records — then writes them to the database.

## Responsibilities
- Consume incoming messages from `src/client/`.
- Normalize content:
  - Text: direct.
  - Images: OCR to text, CLIP embeddings.
  - Audio/voice: transcription to text.
  - Files: lightweight extraction or metadata only.
- Run **Agent-Privacy** masking locally to scrub secrets.
- Generate embeddings via your chosen model provider(s).
- Persist payloads + embeddings to `src/database/`.