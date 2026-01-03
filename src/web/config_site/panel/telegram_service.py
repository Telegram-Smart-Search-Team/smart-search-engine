import asyncio
import datetime
import os
import pathlib
from typing import Any, Dict, List, Tuple

from django.conf import settings

from telethon import TelegramClient, types
from telethon.utils import get_peer_id
from telethon.tl.functions.messages import (
    GetForumTopicsRequest,
    GetCustomEmojiDocumentsRequest,
)

from .config_store import default_sources_config, save_config_and_publish


Entity = types.User | types.Chat | types.Channel


BASE_DIR = settings.BASE_DIR
MEDIA_ROOT = pathlib.Path(settings.MEDIA_ROOT)
MEDIA_ROOT.mkdir(parents=True, exist_ok=True)

TELEGRAM_MEDIA_DIR = MEDIA_ROOT / settings.TELEGRAM_MEDIA_SUBDIR
TELEGRAM_MEDIA_DIR.mkdir(parents=True, exist_ok=True)

SAVED_MESSAGES_ICON_PATH = MEDIA_ROOT / "saved_messages.png"

CLIENT_APP_API_ID = int(settings.DJANGO_CLIENT_APP_API_ID)
CLIENT_APP_API_HASH = settings.DJANGO_CLIENT_APP_API_HASH
SESSION_PATH = settings.DJANGO_SESSION_PATH

BOT_USER_ID = getattr(settings, "BOT_USER_ID", None)
CLIENT_USER_ID = getattr(settings, "CLIENT_USER_ID", None)

emoji_png_cache: Dict[int, str] = {}

CONCURRENT_CONVERSIONS = 4
conversion_semaphore = asyncio.Semaphore(CONCURRENT_CONVERSIONS)

AVATAR_SIZE = (64, 64)  # px


def _make_client() -> TelegramClient:
    return TelegramClient(SESSION_PATH, CLIENT_APP_API_ID, CLIENT_APP_API_HASH)


# ---------- type helpers ----------


def _chat_type_from_entity(entity: Entity) -> str:
    from telethon.tl.types import User, Chat, Channel

    if isinstance(entity, User):
        return "bot" if entity.bot else "personal"
    if isinstance(entity, Chat):
        return "group"
    if isinstance(entity, Channel):
        is_supergroup = bool(entity.megagroup)
        is_forum = bool(getattr(entity, "forum", False))
        if is_supergroup and is_forum:
            return "matryoshka_group"
        if is_supergroup:
            return "group"
        return "channel"
    return "unknown"


def _title_from_entity(entity: Entity) -> str:
    from telethon.tl.types import User, Chat, Channel

    if isinstance(entity, User):
        name = " ".join(x for x in [entity.first_name, entity.last_name] if x)
        return name or (entity.username or f"User {entity.id}")
    if isinstance(entity, (Chat, Channel)):
        return entity.title or (getattr(entity, "username", None) or f"Chat {entity.id}")
    return "Unknown"


def _count_tracked_sources(cfg: Dict[str, Any]) -> int:
    return (
        int(cfg.get("track_text", False))
        + int(cfg.get("track_image", False))
        + int(cfg.get("track_video", False))
        + int(cfg.get("track_audio", False))
        + int(cfg.get("track_files", False))
    )


def _peer_config_key(peer_id: int) -> str:
    return f"peer:{peer_id}"


def _topic_config_key(peer_id: int, topic_id: int) -> str:
    return f"topic:{peer_id}:{topic_id}"


def _media_url_from_path(path: str | None) -> str | None:
    if not path:
        return None
    p = pathlib.Path(path).resolve()
    try:
        rel = p.relative_to(MEDIA_ROOT)
    except ValueError:
        return None
    return settings.MEDIA_URL.rstrip("/") + "/" + str(rel).replace(os.sep, "/")


# ---------- avatar + emoji PNG helpers ----------


async def download_chat_photo(client: TelegramClient, entity: Entity) -> str | None:
    from PIL import Image  # local import

    peer_id = get_peer_id(entity)
    photo_obj = getattr(entity, "photo", None)
    if not photo_obj:
        return None

    # Skip video/short avatars completely
    has_video = bool(getattr(photo_obj, "has_video", False)) or bool(getattr(photo_obj, "video_sizes", None))
    if has_video:
        return None

    photo_id = getattr(photo_obj, "photo_id", None)
    if photo_id is None:
        return None

    # img_type is always "image" here, since Telegram exposes emoji avatars via photos too.
    filename = TELEGRAM_MEDIA_DIR / f"chat_{peer_id}_image_{photo_id}.jpg"

    # If we already have this exact photo_id file on disk, reuse it.
    if filename.exists():
        return str(filename)

    # Need to download
    try:
        path = await client.download_profile_photo(entity, file=str(filename))
        if not path:
            # download failed -> clean up and give up
            try:
                filename.unlink()
            except FileNotFoundError:
                pass
            return None

        # Resize to 40x40
        try:
            img = Image.open(filename)
            img = img.resize(AVATAR_SIZE, Image.LANCZOS)
            img.save(filename, format="JPEG", quality=85)
        except Exception:
            # If resize fails, keep original file.
            pass

        return str(filename)
    except Exception:
        # Any error -> no avatar (we don't re-download old stuff here)
        try:
            filename.unlink()
        except FileNotFoundError:
            pass
        return None


async def ensure_emoji_png(
    client: TelegramClient,
    emoji_id: int,
    doc: types.Document,
) -> str | None:
    """
    Convert custom emoji document to PNG in TELEGRAM_MEDIA_DIR.

    Supports:
      - image/webp
      - application/x-tgsticker (via lottie_convert.py)
      - video/webm (via ffmpeg)
      - any other format is saved as-is
    """
    if emoji_id in emoji_png_cache:
        return emoji_png_cache[emoji_id]

    base = TELEGRAM_MEDIA_DIR / f"emoji_{emoji_id}"
    png_path = base.with_suffix(".png")

    # fast path: already converted
    if png_path.exists():
        emoji_png_cache[emoji_id] = str(png_path)
        return str(png_path)

    raw_path_str = await client.download_media(doc, file=str(base))
    if raw_path_str is None:
        return None

    raw_path = pathlib.Path(raw_path_str)
    mime = doc.mime_type or ""

    async with conversion_semaphore:
        if mime == "image/webp":
            from PIL import Image

            def _convert_webp():
                img = Image.open(raw_path)
                img.save(png_path)

            await asyncio.to_thread(_convert_webp)
            emoji_png_cache[emoji_id] = str(png_path)
            return str(png_path)

        elif mime == "application/x-tgsticker":
            # .tgs → PNG (first frame) via lottie_convert.py
            proc = await asyncio.create_subprocess_exec(
                "lottie_convert.py",
                str(raw_path),
                str(png_path),
                "--frame",
                "0",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            if proc.returncode == 0 and png_path.exists():
                emoji_png_cache[emoji_id] = str(png_path)
                return str(png_path)
            emoji_png_cache[emoji_id] = str(raw_path)
            return str(raw_path)

        elif mime == "video/webm":
            # video emoji → PNG first frame via ffmpeg
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-y",
                "-i",
                str(raw_path),
                "-vframes",
                "1",
                str(png_path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            if proc.returncode == 0 and png_path.exists():
                emoji_png_cache[emoji_id] = str(png_path)
                return str(png_path)
            emoji_png_cache[emoji_id] = str(raw_path)
            return str(raw_path)

        else:
            emoji_png_cache[emoji_id] = str(raw_path)
            return str(raw_path)


async def fetch_emoji_docs_for_topics(
    client: TelegramClient,
    topics: List[types.ForumTopic],
) -> Dict[int, types.Document]:
    emoji_ids = {getattr(t, "icon_emoji_id", None) for t in topics}
    emoji_ids = {e for e in emoji_ids if e}

    if not emoji_ids:
        return {}

    docs = await client(GetCustomEmojiDocumentsRequest(document_id=list(emoji_ids)))

    mapping: Dict[int, types.Document] = {}
    for doc in docs:
        doc_id = getattr(doc, "id", None)
        if doc_id is not None:
            mapping[doc_id] = doc
    return mapping


async def get_all_forum_topics(
    client: TelegramClient,
    peer: types.Channel,
) -> List[types.ForumTopic]:
    topics: List[types.ForumTopic] = []

    offset_date = datetime.datetime(1970, 1, 1)
    offset_id = 0
    offset_topic = 0
    total = None

    while True:
        resp = await client(
            GetForumTopicsRequest(
                peer=peer,
                offset_date=offset_date,
                offset_id=offset_id,
                offset_topic=offset_topic,
                limit=100,
            )
        )

        if total is None:
            total = resp.count

        if not resp.topics:
            break

        topics.extend(resp.topics)

        if len(topics) >= total:
            break

        last = resp.topics[-1]
        offset_topic = last.id
        offset_id = last.top_message

        msg_dates = {m.id: m.date for m in resp.messages}
        offset_date = msg_dates.get(offset_id, offset_date)

    return topics


async def build_matryoshka_details(
    client: TelegramClient,
    entity: types.Channel,
    config: Dict[str, Any],
) -> Tuple[str | None, List[Dict[str, Any]]]:
    """
    Returns (group_photo_path, subchats_list)
    subchats_list: [{chat_key, title, icon_url, config, topic_id}, ...]
    """
    group_peer_id = get_peer_id(entity)
    group_photo_path = await download_chat_photo(client, entity)

    topics = await get_all_forum_topics(client, entity)
    if not topics:
        return group_photo_path, []

    emoji_docs = await fetch_emoji_docs_for_topics(client, topics)

    emoji_png_paths: Dict[int, str | None] = {}

    async def _convert_one(eid: int, doc: types.Document):
        try:
            path = await ensure_emoji_png(client, eid, doc)
            return eid, path
        except Exception:
            return eid, None

    tasks = [_convert_one(eid, doc) for eid, doc in emoji_docs.items()]
    if tasks:
        results = await asyncio.gather(*tasks)
        for eid, path in results:
            emoji_png_paths[eid] = path

    subchats: List[Dict[str, Any]] = []
    chats_cfg = config.get("chats", {})

    for t in topics:
        topic_id = t.id
        title = t.title or f"Topic {topic_id}"
        icon_emoji_id = getattr(t, "icon_emoji_id", None)

        icon_file = None
        if icon_emoji_id:
            icon_file = emoji_png_paths.get(icon_emoji_id)

        key = _topic_config_key(group_peer_id, topic_id)
        cfg = chats_cfg.get(key, default_sources_config())

        subchats.append(
            {
                "chat_key": key,
                "topic_id": topic_id,
                "title": title,
                "icon_url": _media_url_from_path(icon_file) if icon_file else None,
                "config": cfg,
                "parent_peer_id": group_peer_id,
            }
        )

    return group_photo_path, subchats


# ---------- bootstrap: chat list + ordering ----------


async def fetch_chat_entries(
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Return top-level entries (plain chats + matryoshka groups), and a flag
    indicating whether we modified the config by adding default entries
    for previously unseen chats.
    """
    client = _make_client()
    entries: List[Dict[str, Any]] = []
    chats_cfg = config.setdefault("chats", {})
    changed = False

    async with client:
        async for dialog in client.iter_dialogs():
            entity = dialog.entity
            peer_id = get_peer_id(entity)

            # Skip the bot used for queries (we never want to track it)
            if BOT_USER_ID is not None and peer_id == BOT_USER_ID:
                continue

            chat_type = _chat_type_from_entity(entity)
            entry_id = _peer_config_key(peer_id)
            title = _title_from_entity(entity)

            # "Saved Messages" special case
            if CLIENT_USER_ID is not None and peer_id == CLIENT_USER_ID:
                title = "Saved Messages"

            if entry_id in chats_cfg:
                cfg = chats_cfg[entry_id]
                is_new = False
            else:
                # Brand new chat: show as "new" once, but immediately
                # add default (all False) config and persist later.
                cfg = default_sources_config()
                chats_cfg[entry_id] = cfg
                is_new = True
                changed = True

            tracked_count = _count_tracked_sources(cfg)

            # For matryoshka groups, consider topics' configs as well.
            if chat_type == "matryoshka_group":
                max_child = 0
                prefix = f"topic:{peer_id}:"
                for key, val in chats_cfg.items():
                    if key.startswith(prefix):
                        c = _count_tracked_sources(val)
                        if c > max_child:
                            max_child = c
                if max_child > tracked_count:
                    tracked_count = max_child

            entries.append(
                {
                    "entry_id": entry_id,
                    "peer_id": peer_id,
                    "title": title,
                    "chat_type": chat_type,
                    "tracked_sources_count": tracked_count,
                    "is_new": is_new,
                }
            )

    # Sort: new chats first, then by tracked sources desc, then title
    entries.sort(
        key=lambda e: (
            0 if e["is_new"] else 1,
            -e["tracked_sources_count"],
            e["title"].lower(),
        )
    )
    return entries, changed


async def build_bootstrap_payload(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build bootstrap payload:
      - possibly extend config with "new" chats (default config all False)
      - return chat_order + meta for frontend.
    """
    entries, changed = await fetch_chat_entries(config)
    if changed:
        # Save config so "new" chats are only new once
        save_config_and_publish(config)

    chat_order = [e["entry_id"] for e in entries]

    meta = {
        e["entry_id"]: {
            "peer_id": e["peer_id"],
            "title": e["title"],
            "chat_type": e["chat_type"],
            "is_new": e["is_new"],
            "tracked_sources_count": e["tracked_sources_count"],
        }
        for e in entries
    }

    return {
        "chat_order": chat_order,
        "meta": meta,
    }


# ---------- batch fetch ----------


async def fetch_batch_details(entry_ids: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    client = _make_client()
    results: List[Dict[str, Any]] = []
    chats_cfg = config.get("chats", {})

    async with client:
        for entry_id in entry_ids:
            if not entry_id.startswith("peer:"):
                continue

            peer_id = int(entry_id.split(":", 1)[1])

            # Defensive: skip bot chat even if requested
            if BOT_USER_ID is not None and peer_id == BOT_USER_ID:
                continue

            entity = await client.get_entity(peer_id)
            chat_type = _chat_type_from_entity(entity)
            title = _title_from_entity(entity)

            if CLIENT_USER_ID is not None and peer_id == CLIENT_USER_ID:
                title = "Saved Messages"

            is_forum = chat_type == "matryoshka_group"
            cfg = chats_cfg.get(entry_id, default_sources_config())

            if is_forum and isinstance(entity, types.Channel):
                group_photo_path, subchats = await build_matryoshka_details(client, entity, config)
                avatar_url = _media_url_from_path(group_photo_path)
            else:
                # "saved messages" icon override
                if CLIENT_USER_ID is not None and peer_id == CLIENT_USER_ID and SAVED_MESSAGES_ICON_PATH.exists():
                    avatar_url = _media_url_from_path(str(SAVED_MESSAGES_ICON_PATH))
                else:
                    avatar_path = await download_chat_photo(client, entity)
                    avatar_url = _media_url_from_path(avatar_path)
                subchats = []

            results.append(
                {
                    "entry_id": entry_id,
                    "peer_id": peer_id,
                    "title": title,
                    "chat_type": chat_type,
                    "avatar_url": avatar_url,
                    "chat_key": entry_id,
                    "config": cfg,
                    "subchats": subchats,
                }
            )

    return results
