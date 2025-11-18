import asyncio
import json
import os
import pathlib
from typing import Any, Dict

from dotenv import load_dotenv

from telethon import TelegramClient, events
from telethon.utils import get_peer_id

import redis.asyncio as aioredis

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=str(BASE_DIR / ".env"))

CLIENT_APP_API_ID = int(os.getenv("CLIENT_APP_API_ID"))
CLIENT_APP_API_HASH = os.getenv("CLIENT_APP_API_HASH")
SESSION_PATH = str(BASE_DIR / "base_session.session")

REDIS_URL = os.getenv("REDIS_URL")
CONFIG_KEY = "smartsearch:tracking-config"
CONFIG_CHANNEL = "smartsearch:config-updates"

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
BOT_USER_ID = None
if BOT_TOKEN:
    try:
        BOT_USER_ID = int(BOT_TOKEN.split(":", 1)[0])
    except (ValueError, IndexError):
        BOT_USER_ID = None

config: Dict[str, Any] = {"chats": {}}


def _peer_config_key_from_event(event: events.NewMessage.Event) -> str:
    """
    Map Telethon event to our config key.
    - For regular chats: peer:<peer_id>
    - For forum topics: topic:<peer_id>:<topic_id>
    """
    msg = event.message
    peer = msg.peer_id or msg.to_id
    base_peer_id = get_peer_id(peer)

    topic_id = None
    if getattr(msg, "reply_to", None) and getattr(msg.reply_to, "forum_topic_id", None):
        topic_id = msg.reply_to.forum_topic_id

    if topic_id is not None:
        return f"topic:{base_peer_id}:{topic_id}"
    return f"peer:{base_peer_id}"


def _detect_source_kind(event: events.NewMessage.Event) -> str:
    msg = event.message
    if msg.photo:
        return "image"
    if msg.video or (msg.document and getattr(msg.document, "mime_type", "").startswith("video/")):
        return "video"
    if msg.voice or msg.audio:
        return "audio"
    if msg.document:
        mt = getattr(msg.document, "mime_type", "") or ""
        if not mt.startswith("audio/") and not mt.startswith("video/"):
            return "file"
    return "text"


def should_track(event: events.NewMessage.Event) -> bool:
    msg = event.message
    peer = msg.peer_id or msg.to_id
    base_peer_id = get_peer_id(peer)

    # Never track our own bot (search bot) – prevents loops
    if BOT_USER_ID is not None and base_peer_id == BOT_USER_ID:
        return False

    key = _peer_config_key_from_event(event)
    cfg = config.get("chats", {}).get(key)
    if not cfg:
        # No entry => default is "track nothing"
        return False

    kind = _detect_source_kind(event)
    if kind == "text":
        return cfg.get("track_text", False)
    if kind == "image":
        return cfg.get("track_image", False)
    if kind == "video":
        return cfg.get("track_video", False)
    if kind == "audio":
        return cfg.get("track_audio", False)
    if kind == "file":
        return cfg.get("track_files", False)
    return False


async def load_config_from_redis(r: aioredis.Redis) -> None:
    global config
    raw = await r.get(CONFIG_KEY)
    if not raw:
        config = {"chats": {}}
        return
    try:
        config = json.loads(raw.decode("utf-8"))
    except Exception:
        config = {"chats": {}}


async def watch_config_updates(r: aioredis.Redis) -> None:
    pubsub = r.pubsub()
    await pubsub.subscribe(CONFIG_CHANNEL)
    async for message in pubsub.listen():
        if message.get("type") != "message":
            continue
        await load_config_from_redis(r)


async def main() -> None:
    r = aioredis.from_url(REDIS_URL)
    await load_config_from_redis(r)

    client = TelegramClient(SESSION_PATH, CLIENT_APP_API_ID, CLIENT_APP_API_HASH)

    @client.on(events.NewMessage)
    async def handle_new_message(event: events.NewMessage.Event):
        if not should_track(event):
            return
        # Here goes your logic that actually handles tracked messages
        print("TRACKED:", event.chat_id, event.id, _detect_source_kind(event))

    watcher_task = asyncio.create_task(watch_config_updates(r))

    async with client:
        await client.run_until_disconnected()

    watcher_task.cancel()
    try:
        await watcher_task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
