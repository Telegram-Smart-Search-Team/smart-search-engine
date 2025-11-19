CREATE TABLE IF NOT EXISTS messages (
    id BIGSERIAL PRIMARY KEY,
    telegram_chat_id BIGINT NOT NULL,
    telegram_message_id BIGINT NOT NULL,
    timestamp INTEGER NOT NULL,
    topic_id BIGINT,
    is_forum BOOLEAN DEFAULT FALSE,
    reply_to_message_id BIGINT,
    payload JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE (telegram_chat_id, telegram_message_id)
);


CREATE TABLE IF NOT EXISTS graph_embeddings (
    id BIGSERIAL PRIMARY KEY,
    root_message_id BIGINT NOT NULL,
    telegram_chat_id BIGINT NOT NULL,
    graph_size INTEGER NOT NULL,
    nodes_ids BIGINT[] NOT NULL,
    edges JSONB NOT NULL,
    embedding vector(3072) NOT NULL,  -- text-embedding-3-large
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    FOREIGN KEY (root_message_id) REFERENCES messages(id)
);


-- messages
CREATE INDEX IF NOT EXISTS idx_messages_chat_timestamp ON messages (telegram_chat_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages (telegram_chat_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages (timestamp);

-- graph embeddings
CREATE INDEX IF NOT EXISTS idx_graph_embeddings_root ON graph_embeddings (root_message_id);
CREATE INDEX IF NOT EXISTS idx_graph_embeddings_chat ON graph_embeddings (telegram_chat_id);


CREATE OR REPLACE FUNCTION find_sequential_neighbors(
    p_telegram_chat_id BIGINT,
    p_message_id BIGINT,
    p_limit INTEGER
) RETURNS TABLE(
    telegram_message_id BIGINT,
    direction TEXT,
    distance INTEGER
) AS $$
DECLARE
    target_timestamp INTEGER;
    target_topic_id BIGINT;
BEGIN
    SELECT m.timestamp, m.topic_id
    INTO target_timestamp, target_topic_id
    FROM messages m
    WHERE m.telegram_chat_id = p_telegram_chat_id
      AND m.telegram_message_id = p_message_id;
    
    IF target_timestamp IS NULL THEN
        RETURN;
    END IF;
    
    RETURN QUERY
    SELECT 
        m.telegram_message_id,
        'previous' AS direction,
        (target_timestamp - m.timestamp) AS distance
    FROM messages m
    WHERE m.telegram_chat_id = p_telegram_chat_id
      AND m.timestamp < target_timestamp
      AND (m.topic_id = target_topic_id
           OR (m.topic_id IS NULL AND target_topic_id IS NULL))
    ORDER BY m.timestamp DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;