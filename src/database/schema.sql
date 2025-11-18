-- Enable pgvector (idempotent)
CREATE EXTENSION IF NOT EXISTS vector;

-- Chat type enum (personal DM / group / channel)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'chat_type') THEN
        CREATE TYPE chat_type AS ENUM ('personal', 'group', 'channel');
    END IF;
END$$;

-- Chats table (optional but useful)
CREATE TABLE IF NOT EXISTS telegram_chats (
    id          BIGSERIAL PRIMARY KEY,
    peer_id     BIGINT NOT NULL UNIQUE,     -- Telegram's chat id
    chat_type   chat_type NOT NULL,
    title       TEXT,
    username    TEXT,
    is_tracked  BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS telegram_chats_is_tracked_idx
    ON telegram_chats (is_tracked);

-- Messages table
CREATE TABLE IF NOT EXISTS telegram_messages (
    id              BIGSERIAL PRIMARY KEY,

    peer_id         BIGINT NOT NULL,           -- which chat
    chat_type       chat_type NOT NULL,
    message_id      BIGINT NOT NULL,           -- Telegram message id in chat

    author_id       BIGINT,                    -- who sent it (may be NULL for some service msgs)
    is_outgoing     BOOLEAN NOT NULL DEFAULT FALSE,
    date            TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- message categorization
    msg_type        TEXT NOT NULL,             -- 'text', 'media', 'service', whatever you decide
    has_media       BOOLEAN NOT NULL DEFAULT FALSE,
    media_type      TEXT,                      -- 'photo','video','voice','document', etc.

    -- content
    content                 TEXT,              -- original message text (if exists)
    obfuscated_content      TEXT,              -- if you need to store masked version for privacy

    -- text retrieved from media (OCR/ASR/etc.)
    retrieved_text          TEXT,

    -- embeddings (all vector(768); change dim as needed)
    embedding               vector(768),       -- main text embedding
    retrieved_text_embedding vector(768),      -- embedding of retrieved_text
    media_embedding         vector(768),       -- embedding of visual/audio features, etc.

    -- misc
    reply_to_message_id BIGINT,               -- local Telegram msg id in same chat
    edited_at          TIMESTAMPTZ,
    deleted            BOOLEAN NOT NULL DEFAULT FALSE,

    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- uniqueness: per-chat message_id is unique
    CONSTRAINT telegram_messages_peer_msg_unique
        UNIQUE (peer_id, message_id)
);

CREATE INDEX IF NOT EXISTS telegram_messages_peer_idx
    ON telegram_messages (peer_id);

CREATE INDEX IF NOT EXISTS telegram_messages_date_idx
    ON telegram_messages (date);

-- HNSW index on main embedding for cosine similarity
CREATE INDEX IF NOT EXISTS telegram_messages_embedding_hnsw_idx
ON telegram_messages
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- Optional HNSW index on media_embedding if you do visual-only search
CREATE INDEX IF NOT EXISTS telegram_messages_media_embedding_hnsw_idx
ON telegram_messages
USING hnsw (media_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- Optional: track which chats each *app user* wants to include in search
CREATE TABLE IF NOT EXISTS tracked_chats (
    id          BIGSERIAL PRIMARY KEY,
    app_user_id BIGINT NOT NULL,          -- your own user id in your system, not Telegram
    peer_id     BIGINT NOT NULL,
    chat_type   chat_type NOT NULL,
    enabled     BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT tracked_chats_unique UNIQUE (app_user_id, peer_id, chat_type)
);
