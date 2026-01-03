import re
import os
import sys
import json
import math
import logging
import pathlib
import typing as tp

from dotenv import load_dotenv

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=str(BASE_DIR / ".env"))

sys.path.append(str(BASE_DIR / "src"))

# telethon & related imports
from telethon import TelegramClient, events

# postgres & related imports
import asyncpg
from asyncpg.pool import Pool

# redis & related imports
import redis.asyncio as aioredis

# aiohttp & related imports
import asyncio
import aiohttp

# local imports
from utils.utils import get_embedding


# logging configuration
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "bot.log"

logger = logging.getLogger("client")
logger.setLevel(logging.DEBUG)

_file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_file_handler.setFormatter(_formatter)
logger.addHandler(_file_handler)
logger.propagate = False


CLIENT_APP_API_ID = int(os.getenv("CLIENT_APP_API_ID"))
CLIENT_APP_API_HASH = os.getenv("CLIENT_APP_API_HASH")

BOT_TOKEN = os.getenv("BOT_TOKEN")

# DB
POSTGRESQL_USER = os.getenv("POSTGRESQL_USER")
POSTGRESQL_PASSWORD = os.getenv("POSTGRESQL_PASSWORD")
POSTGRESQL_DATABASE_NAME = os.getenv("POSTGRESQL_DATABASE_NAME")
POSTGRESQL_HOST = os.getenv("POSTGRESQL_HOST")
POSTGRESQL_PORT = int(os.getenv("POSTGRESQL_PORT", "5432"))
POSTGRESQL_SSL = os.getenv("POSTGRESQL_SSL", "false").lower() == "true"

# redis
REDIS_URL = os.getenv("REDIS_URL")
REDIS_FORWARD_QUEUE_KEY = os.getenv(
    "REDIS_FORWARD_QUEUE_KEY",
    "smart_search:forward_queue",
)

# sampling params
TEMPERATURE = float(os.getenv("TEMPERATURE", "1.0"))
NUCLEUS_SAMPLING_P = float(os.getenv("NUCLEUS_SAMPLING_P", "0.9"))

# deepseek
DEEPSEEK_TOKEN = os.getenv("DEEPSEEK_TOKEN")
DEEPSEEK_API_URL = os.getenv(
    "DEEPSEEK_API_URL",
    "https://api.deepseek.com/v1/chat/completions",
)
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL")

MAX_CHARS_PER_MEDIA_IN_PROMPT = int(os.getenv("MAX_CHARS_PER_MEDIA_IN_PROMPT"))

# graph KNN
MAX_RAW_GRAPH_CANDIDATES = int(os.getenv("MAX_RAW_GRAPH_CANDIDATES", "64"))
MAX_CANDIDATES = int(os.getenv("MAX_CANDIDATES"))

# HyDE
HYDE_ENABLED = (os.getenv("HYDE_ENABLED") == "true")
HYDE_WEIGHT = float(os.getenv("HYDE_WEIGHT"))
HYDE_MAX_CHARS = int(os.getenv("HYDE_MAX_CHARS"))

db_pool: tp.Optional[Pool] = None
redis_client: tp.Optional[aioredis.Redis] = None


# ---------- elegant status UI ----------


def build_status(step: int) -> str:
    done = "✅"
    active = "⏳"
    pending = "▫️"

    if step <= 0:
        icons = (active, pending, pending)
    elif step == 1:
        icons = (done, active, pending)
    elif step == 2:
        icons = (done, done, active)
    else:
        icons = (done, done, done)

    return (
        "🔍 *Smart search status*\n\n"
        f"1. {icons[0]} Calculated text-embedding-v3-large embedding\n"
        f"2. {icons[1]} Formed candidate list\n"
        f"3. {icons[2]} Received an answer from {DEEPSEEK_MODEL}"
    )


async def update_status_message(status_msg, step: int) -> None:
    if not status_msg:
        return
    try:
        await status_msg.edit(build_status(step), parse_mode="markdown")
    except Exception:
        logger.exception("[WARN] Failed to edit status message", exc_info=True)


# ---------- DB helpers ----------


async def init_database() -> None:
    global db_pool

    if db_pool is not None:
        return

    db_pool = await asyncpg.create_pool(
        user=POSTGRESQL_USER,
        password=POSTGRESQL_PASSWORD,
        database=POSTGRESQL_DATABASE_NAME,
        host=POSTGRESQL_HOST,
        port=POSTGRESQL_PORT,
        ssl=POSTGRESQL_SSL,
    )


async def close_database() -> None:
    global db_pool

    if db_pool is not None:
        await db_pool.close()
        db_pool = None


async def get_message_by_telegram_id(chat_id: int, message_id: int) -> tp.Optional[dict[str, tp.Any]]:
    if db_pool is None:
        raise RuntimeError("Database pool is not initialized")

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT *
            FROM messages
            WHERE telegram_chat_id = $1
              AND telegram_message_id = $2
            """,
            chat_id,
            message_id,
        )
        return dict(row) if row else None


async def fetch_graph_candidates(
    query_embedding: list[float],
    limit: int = MAX_RAW_GRAPH_CANDIDATES,
) -> list[dict[str, tp.Any]]:
    if db_pool is None:
        raise RuntimeError("Database pool is not initialized")

    embedding_str = "[" + ",".join(str(float(x)) for x in query_embedding) + "]"

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                id,
                root_message_id,
                telegram_chat_id,
                graph_size,
                nodes_ids,
                edges,
                (embedding <-> $1::vector) AS distance
            FROM graph_embeddings
            ORDER BY embedding <-> $1::vector
            LIMIT $2
            """,
            embedding_str,
            limit,
        )

    return [dict(row) for row in rows]


# ---------- redis helpers ----------


async def get_redis() -> aioredis.Redis:
    global redis_client

    if redis_client is None:
        if not REDIS_URL:
            raise RuntimeError("REDIS_URL is not set")

        redis_client = aioredis.from_url(REDIS_URL)

    return redis_client


async def close_redis() -> None:
    global redis_client

    if redis_client is not None:
        await redis_client.aclose()
        redis_client = None


async def enqueue_forward_request(
    target_chat_id: int,
    messages_to_show: list[dict[str, int]],
) -> None:
    r = await get_redis()

    item = {
        "target_chat_id": target_chat_id,
        "messages_to_show": messages_to_show,
    }

    await r.rpush(
        REDIS_FORWARD_QUEUE_KEY,
        json.dumps(item, ensure_ascii=False),
    )


# ---------- graph prompt functions ----------


async def generate_graph_prompt(
    chat_id: int,
    message_ids: list[int],
    edges: list[list[tp.Any]],
) -> str:
    prompt_parts: list[str] = []

    # try:
    #     chat_title = await get_chat_title(chat_id) or f"Chat {chat_id}"
    # except Exception as e:
    #     chat_title = f"Chat {chat_id}"

    chat_title = f"Chat {chat_id}"

    for idx, message_id in enumerate(message_ids):
        message = await get_message_by_telegram_id(chat_id, message_id)
        if not message:
            continue

        payload = json.loads(message["payload"])
        prompt_parts_local: list[str] = []

        connected_edges: list[tuple[int, str]] = []
        for edge in edges:
            src_idx, dst_idx, conn_type = edge[0], edge[1], edge[2]

            if src_idx == idx:
                other_idx = dst_idx
            elif dst_idx == idx:
                other_idx = src_idx
            else:
                continue

            if 0 <= other_idx < len(message_ids):
                connected_edges.append((other_idx, conn_type))

        # header
        timestamp = payload.get("timestamp")
        if timestamp is not None:
            from datetime import datetime as _dt

            ts = _dt.fromtimestamp(timestamp)
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        else:
            ts_str = "unknown time"

        # print(chat_title)
        header = f'# --- Telegram Message {message_id} in Chat "{chat_title}" at {ts_str}'

        if connected_edges:
            edge_descs: list[str] = []
            for other_idx, conn_type in connected_edges:
                if other_idx < idx:
                    edge_descs.append(f"{conn_type} to message {message_ids[other_idx]}")
            if edge_descs:
                header += f" ({', '.join(edge_descs)})"

        prompt_parts_local.append(header)
        prompt_parts_local.append("")

        # attachments
        attachments = payload.get("attachments") or []
        if attachments:
            prompt_parts_local.append("## Attachments:")
            for attachment in attachments:
                filename = attachment.get("filename", "Unknown")
                desc = attachment.get("masked_description") or "(empty)"
                prompt_parts_local.append(
                    f'### Attachment "{filename}"\n```\n{desc[:MAX_CHARS_PER_MEDIA_IN_PROMPT]}\n```'
                )
                prompt_parts_local.append("")

        # media
        media_items = payload.get("media") or []
        if media_items:
            prompt_parts_local.append("## Media:")
            for media in media_items:
                filename = media.get("filename", "Unknown")

                if "masked_description" in media:
                    desc = media["masked_description"]
                else:
                    sound_desc = media.get("masked_sound_description")
                    vision_desc = media.get("masked_vision_description")

                    parts: list[str] = []
                    if sound_desc:
                        parts.append(f"[SOUND]\n{sound_desc}")
                    if vision_desc:
                        parts.append(f"[VIDEO]\n{vision_desc}")
                    desc = "\n\n".join(parts) if parts else "No description available"

                prompt_parts_local.append(f'### Media "{filename}"\n```\n{desc[:MAX_CHARS_PER_MEDIA_IN_PROMPT]}\n```')
                prompt_parts_local.append("")

        # message content
        text = payload.get("masked_text") or payload.get("original_text") or "(empty)"
        prompt_parts_local.append(f"## Message Content:\n```\n{text}\n```")

        prompt_parts.append("\n".join(prompt_parts_local).strip())

    return "\n\n".join(prompt_parts).strip()


# ---------- scoring + sampling ----------


def _l2_normalize(vec: list[float]) -> list[float]:
    s = 0.0
    for x in vec:
        s += x * x
    n = math.sqrt(s) or 1.0
    return [x / n for x in vec]


def _mix_embeddings(e_query: list[float], e_hyde: list[float], w: float) -> list[float]:
    # e = norm((1-w) e_query + w e_hyde)
    if len(e_query) != len(e_hyde):
        # fallback: can't mix, return query embedding
        return e_query
    a = 1.0 - float(w)
    b = float(w)
    mixed = [a * q + b * h for q, h in zip(e_query, e_hyde)]
    return _l2_normalize(mixed)


def _clean_hyde_text(text: str) -> str:
    # DeepSeek-style models sometimes wrap with <think>...</think> or add boilerplate.
    # HyDE doesn’t need to be “true”; it just needs to be a plausible, keyword-rich passage.
    t = text or ""
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL)
    t = t.strip()

    # remove surrounding quotes if model returns a quoted block
    if len(t) >= 2 and ((t[0] == t[-1] == '"') or (t[0] == t[-1] == "'")):
        t = t[1:-1].strip()

    # cap length
    if len(t) > HYDE_MAX_CHARS:
        t = t[:HYDE_MAX_CHARS].rsplit(" ", 1)[0].strip()

    return t


async def generate_hyde_document(raw_query: str) -> str:
    """
    HyDE: generate a hypothetical passage that *would* appear in the user's Telegram history
    and be useful to retrieve relevant messages. This is ONLY for embedding/retrieval.
    """
    hyde_system_prompt = (
        "You generate a short hypothetical document for search retrieval (HyDE).\n"
        "Given a user's search query, write a plausible excerpt that could exist in the user's "
        "Telegram chats and would directly help answer that query.\n\n"
        "Constraints:\n"
        "- Output ONLY the excerpt text (no JSON, no titles, no bullet labels).\n"
        "- 1 to 3 short paragraphs max.\n"
        "- Include concrete keywords, names of tools/libraries, commands, file names, variables, etc. if relevant.\n"
        "- Do not add any disclaimers.\n"
    )

    # We reuse your existing DeepSeek call; if you have a cheaper/faster model, swap it here.
    out = await call_deepseek_reasoner(hyde_system_prompt, raw_query)
    return _clean_hyde_text(out)


def softmax_scores(distances: list[float], temperature: float) -> list[float]:
    if not distances:
        return []

    scores = [-d for d in distances]
    t = max(temperature, 1e-6)
    scaled = [s / t for s in scores]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    s = sum(exps) or 1.0

    return [e / s for e in exps]


def nucleus_filter(
    graphs: list[dict[str, tp.Any]],
    probs: list[float],
    p: float,
    max_graphs: int,
) -> list[dict[str, tp.Any]]:
    if not graphs or not probs:
        return []

    p = max(min(p, 1.0), 0.0)

    indices = list(range(len(graphs)))
    indices.sort(key=lambda i: probs[i], reverse=True)

    selected: list[int] = []
    cumulative = 0.0

    for idx in indices:
        selected.append(idx)
        cumulative += probs[idx]

        if cumulative >= p:
            break

    chosen = [graphs[i] for i in selected]
    chosen.sort(key=lambda g: g["distance"])

    if max_graphs and len(chosen) > max_graphs:
        chosen = chosen[:max_graphs]

    return chosen


async def select_candidate_graphs(
    query_embedding: list[float],
) -> list[dict[str, tp.Any]]:
    graphs = await fetch_graph_candidates(
        query_embedding,
        limit=MAX_RAW_GRAPH_CANDIDATES,
    )

    if not graphs:
        return []

    distances = [float(g["distance"]) for g in graphs]
    probs = softmax_scores(distances, TEMPERATURE)
    return nucleus_filter(graphs, probs, NUCLEUS_SAMPLING_P, MAX_CANDIDATES)


# ---------- deepseek model ----------


class HTTPStatusError(Exception):
    def __init__(self, status: int, body: tp.Optional[tp.Any]):
        self.status = status
        self.body = body

        if isinstance(body, dict):
            msg = f"HTTP Status: {status}, `body` is of type {type(body)}:\n{json.dumps(self.body, indent=2, ensure_ascii=False)}"
        elif isinstance(body, str) and body != "":
            msg = f"HTTP Status: {status}, `body` is of type {type(body)}:\n{body}"
        elif isinstance(body, str) and body == "":
            msg = f"HTTP Status: {status}, `body` is of type {type(body)}, empty."
        else:
            msg = f"HTTP Status: {status}, `body` is of type {type(body)}."

        super().__init__(msg)


class InvalidJSONError(Exception):
    def __init__(self, status: tp.Optional[int], body_text: str) -> None:
        self.status = status
        self.body_text = body_text

        super().__init__(f"Invalid JSON (status={status}): {body_text[:3000]}.")


class EmptyBodyError(Exception):
    def __init__(self, status: tp.Optional[int]) -> None:
        self.status = status

        super().__init__(f"Empty response body (status={status}).")


async def deepseek_post_strict_safe_fixed_utf_8(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict[str, tp.Any],
    payload: dict[str, tp.Any],
    timeout: int = 300,
) -> dict[str, tp.Any]:
    buffer = bytearray()
    saw_left_brace = False

    async with session.post(url, json=payload, headers=headers, timeout=timeout) as response:
        status = response.status

        try:
            async for chunk in response.content.iter_chunked(8192):
                if not chunk.strip():
                    continue

                if not saw_left_brace:
                    i = chunk.find(b"{")

                    if i == -1:
                        continue

                    buffer.extend(chunk[i:])
                    saw_left_brace = True
                else:
                    buffer.extend(chunk)

                try:
                    text = buffer.decode("utf-8")  # strict
                except UnicodeDecodeError:
                    continue

                try:
                    response_data = json.loads(text)

                    if 200 <= status < 300:
                        return response_data

                    raise HTTPStatusError(status, response_data)

                except json.JSONDecodeError:
                    continue

        except (aiohttp.ClientPayloadError, aiohttp.http_exceptions.TransferEncodingError):
            # broken EOF
            pass

        try:
            text = buffer.decode("utf-8")

        except UnicodeDecodeError as e:
            tail = buffer[-16:].hex()
            raise InvalidJSONError(status, f"UTF-8 decode error at end: {e}, tail=0x{tail}.")

        text = text.strip()

        if not text:
            raise EmptyBodyError(status)

        try:
            response_data = json.loads(text)

            if 200 <= status < 300:
                return response_data

            raise HTTPStatusError(status, response_data)

        except json.JSONDecodeError:
            raise InvalidJSONError(status, text)


async def call_deepseek_reasoner(system_prompt: str, user_prompt: str) -> str:
    if not DEEPSEEK_TOKEN:
        raise RuntimeError("DEEPSEEK_TOKEN is not set")

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }

    logger.debug(f"System:\n{system_prompt}")
    logger.debug(f"User:\n{user_prompt}")

    async with aiohttp.ClientSession() as session:
        data = await deepseek_post_strict_safe_fixed_utf_8(
            session=session,
            url=DEEPSEEK_API_URL,
            headers=headers,
            payload=payload,
            timeout=300,
        )

    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Unexpected DeepSeek response format: {data!r}")


def extract_json_object(text: str) -> dict[str, tp.Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"```json(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError(f"Could not parse JSON from LLM output: {text!r}")


# ---------- user query handlers ----------


async def handle_query(event: events.NewMessage.Event) -> None:
    msg = event.message

    # 1. ignoring forwarded messages
    if getattr(msg, "fwd_from", None) is not None:
        return

    # 2. extracting text content
    query = (getattr(msg, "message", "") or "").strip()
    if not query or not query.startswith("/search"):
        return

    # Use the actual user intent for embeddings/HyDE
    raw_query = query[len("/search"):].strip()
    if not raw_query:
        await event.reply("Usage: /search <your query>")
        return

    target_chat_id = int(BOT_TOKEN.split(":")[0])

    logger.info(f"[INFO] Bot query in chat {target_chat_id}: {query!r}")

    status_msg = None
    try:
        status_msg = await event.reply(
            build_status(0),
            parse_mode="markdown",
        )
    except Exception:
        logger.exception("[WARN] Failed to send status message", exc_info=True)

    # 3. embedding query (+ HyDE)
    try:
        # 3.1 real query embedding
        query_embedding = await get_embedding(raw_query)

        # 3.2 HyDE embedding (optional)
        if HYDE_ENABLED:
            try:
                hyde_doc = await generate_hyde_document(raw_query)
                if hyde_doc:
                    hyde_embedding = await get_embedding(hyde_doc)
                    final_embedding = _mix_embeddings(query_embedding, hyde_embedding, HYDE_WEIGHT)
                    logger.debug(f"[DEBUG] HyDE doc (trimmed): {hyde_doc[:250]!r}")
                else:
                    final_embedding = query_embedding
            except Exception:
                logger.exception("[WARN] HyDE generation/embedding failed, fallback to query only", exc_info=True)
                final_embedding = query_embedding
        else:
            final_embedding = query_embedding

    except Exception:
        logger.exception("[ERROR] get_embedding failed", exc_info=True)
        await event.reply("Failed to embed your query. Try again later.")
        return

    await update_status_message(status_msg, 1)

    # 4. forming candidate graphs
    try:
        candidate_graphs = await select_candidate_graphs(final_embedding)
    except Exception:
        logger.exception("[ERROR] select_candidate_graphs failed", exc_info=True)
        await event.reply("Search is temporarily unavailable.")
        return

    await update_status_message(status_msg, 2)

    if not candidate_graphs:
        await event.reply("There are no indexed message graphs yet.")
        return

    # 5. building per-graph prompts
    graph_blocks: list[str] = []
    for g in candidate_graphs:
        chat_id = int(g["telegram_chat_id"])
        nodes_ids = list(g["nodes_ids"])
        edges_json = g["edges"]
        if isinstance(edges_json, str):
            edges = json.loads(edges_json)
        else:
            edges = edges_json

        graph_prompt = await generate_graph_prompt(chat_id, nodes_ids, edges)
        block = f">>> Chat {chat_id}\n{graph_prompt}"
        graph_blocks.append(block)

    combined_graphs_prompt = "\n\n".join(graph_blocks)

    # 6. system prompt
    system_prompt = (
        "You are an expert at searching through a user's Telegram message history.\n\n"
        "You will receive:\n"
        "- A user query (what they are trying to find in their history).\n"
        "- Several candidate groups of messages taken from the user's past conversations.\n\n"
        "The user query is:\n"
        f"{raw_query}\n\n"
        "Your task:\n"
        "1. Decide which individual messages (if any) can answer the user's query.\n"
        "2. Select at most 7 messages that best answer the query.\n"
        "3. Compose a clear answer to the user based only on information contained in "
        "the provided messages.\n\n"
        "Return a single valid JSON object with the exact structure:\n"
        "{\n"
        '  "messages_to_show": [\n'
        '    {"chat_id": <telegram_chat_id>, "msg_id": <telegram_message_id>}\n'
        "    // ordered in the exact order they should be shown to the user\n"
        "    // at most 7 entries\n"
        "  ],\n"
        '  "answer_to_user_query": "<short natural-language answer to the user query, '
        'or \\"Could not find anything relevant\\" if nothing is useful>",\n'
        '  "explanation": "<short explanation of why these messages were chosen>"\n'
        "}\n\n"
        "Rules:\n"
        "- Only use information present in the provided message groups.\n"
        '- If no messages clearly answer the query, set "messages_to_show" to an empty '
        'list and set "answer_to_user_query" to "Could not find anything relevant".\n'
        "- Do NOT include any text outside of the JSON object."
    )

    # 7. calling deepseek
    try:
        llm_output = await call_deepseek_reasoner(system_prompt, combined_graphs_prompt)
    except Exception:
        logger.exception("[ERROR] DeepSeek call failed", exc_info=True)
        await event.reply("Reasoning model failed. Try again later.")
        return

    await update_status_message(status_msg, 3)

    # 8. parsing response
    try:
        result = extract_json_object(llm_output)
    except Exception:
        logger.exception("[ERROR] Failed to parse DeepSeek JSON", exc_info=True)
        logger.debug(f"[DEBUG] Raw LLM output: {llm_output!r}")
        await event.reply("Got an invalid response from the reasoning model.")
        return

    messages_to_show_raw = result.get("messages_to_show") or []
    answer = (result.get("answer_to_user_query") or "").strip()
    explanation = result.get("explanation")

    # normalizing messages_to_show
    messages_normalized: list[dict[str, int]] = []
    if isinstance(messages_to_show_raw, list):
        for item in messages_to_show_raw:
            if not isinstance(item, dict):
                continue
            try:
                chat_id = int(item.get("chat_id"))
                msg_id = int(item.get("msg_id"))
            except Exception:
                continue
            messages_normalized.append({"chat_id": chat_id, "msg_id": msg_id})

    # 9. pushing forwarding batch into redis
    if messages_normalized:
        try:
            await enqueue_forward_request(target_chat_id, messages_normalized)
        except Exception:
            logger.exception("[ERROR] enqueue_forward_request failed", exc_info=True)

    # 10. providing bot's final answer
    if not answer:
        answer = "Could not find anything relevant"

    await event.reply(answer)

    if explanation:
        logger.info(f"[INFO] Explanation for query {query!r}: {explanation}")


# ---------- main ----------


async def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is not set")
    if not REDIS_URL:
        raise RuntimeError("REDIS_URL is not set")

    await init_database()

    bot_session_path = str(BASE_DIR / "bot_session.session")
    bot_client = TelegramClient(
        bot_session_path,
        CLIENT_APP_API_ID,
        CLIENT_APP_API_HASH,
    )

    @bot_client.on(events.NewMessage)
    async def _on_new_message(event: events.NewMessage.Event) -> None:  # type: ignore[override]
        await handle_query(event)

    await bot_client.start(bot_token=BOT_TOKEN)
    print("Smart Search bot is running.", flush=True)

    try:
        await bot_client.run_until_disconnected()
    finally:
        await close_database()
        await close_redis()
        await bot_client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
