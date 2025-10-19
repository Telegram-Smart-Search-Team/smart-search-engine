# src/`Client`

## Description

Listens to Telegram updates and forwards new messages into the pipeline. This is the system’s gateway to your data.

## Responsibilities
- Connect to Telegram via Telethon.
- Subscribe to dialogs, channels, and groups per configuration.
- Normalize update events and hand off to `src/dumper/`.
- De-dupe and checkpoint progress to avoid reprocessing.