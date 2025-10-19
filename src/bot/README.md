# src/`Bot`

## Description

A Telegram bot that lets you query your own message corpus in natural language and receive precise, ranked results — in other words, a personal search interface.

## Responsibilities
- Accept user queries via commands or plain text.
- Send search requests to `src/retriever/`.
- Display ranked results with message previews, chat context, and timestamps.
- Provide basic controls: `/subscribe`, `/unsubscribe`, `/status`, `/start`, `/stop`, `/help`, etc.
- Enforce simple rate-limiting and access control.