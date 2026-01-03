import os
import sys
import json
import shutil
import logging
import pathlib
import asyncio
import tempfile
import traceback
import typing as tp
import asyncio.subprocess as asp
from datetime import datetime
from dotenv import load_dotenv

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=str(BASE_DIR / ".env"))

sys.path.append(str(BASE_DIR / "src"))

# telethon & related imports
from telethon import TelegramClient, events
from telethon.utils import get_peer_id
from telethon.tl import types

# redis & related imports
import redis.asyncio as aioredis

# database imports
import asyncpg
from asyncpg.pool import Pool

# local imports
from utils.utils import *
from internal_llm.process import *


# logging configuration
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "client.log"

logger = logging.getLogger("client")
logger.setLevel(logging.DEBUG)

_file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_file_handler.setFormatter(_formatter)
logger.addHandler(_file_handler)
logger.propagate = False


# Environment variables
CLIENT_APP_API_ID = int(os.getenv("CLIENT_APP_API_ID"))
CLIENT_APP_API_HASH = os.getenv("CLIENT_APP_API_HASH")
SESSION_PATH = str(BASE_DIR / "base_session.session")

REDIS_URL = os.getenv("REDIS_URL")
REDIS_CONFIG_KEY = os.getenv("REDIS_CONFIG_KEY")
REDIS_CONFIG_CHANNEL = os.getenv("REDIS_CONFIG_CHANNEL")
REDIS_FORWARD_QUEUE_KEY = os.getenv(
    "REDIS_FORWARD_QUEUE_KEY",
    "smart_search:forward_queue",
)

MAX_FILE_DESCRIPTION_LENGTH_TO_MASK = int(os.getenv("MAX_FILE_DESCRIPTION_LENGTH_TO_MASK"))
MAX_VIDEO_FRAMES_TO_DESCRIBE = int(os.getenv("MAX_VIDEO_FRAMES_TO_DESCRIBE"))
MAX_FILE_SIZE_MB = float(os.getenv("MAX_FILE_SIZE_TO_DOWNLOAD", "50"))
MAX_FILE_SIZE_BYTES = int(MAX_FILE_SIZE_MB * 1024 * 1024)

BIG_FILE_DESCRIPTION = "[File is very big and could not be loaded for processing]"

MAX_CHARS_PER_MEDIA_IN_PROMPT = int(os.getenv("MAX_CHARS_PER_MEDIA_IN_PROMPT"))

# Database configuration
POSTGRESQL_USER = os.getenv("POSTGRESQL_USER")
POSTGRESQL_PASSWORD = os.getenv("POSTGRESQL_PASSWORD")
POSTGRESQL_DATABASE_NAME = os.getenv("POSTGRESQL_DATABASE_NAME")
POSTGRESQL_HOST = os.getenv("POSTGRESQL_HOST")
POSTGRESQL_PORT = int(os.getenv("POSTGRESQL_PORT", "5432"))
POSTGRESQL_SSL = os.getenv("POSTGRESQL_SSL", "false").lower() == "true"

# Graph configuration
GRAPH_SIZES = list(map(int, os.getenv("GRAPH_SIZES", "1,3,5,7").split(",")))
MAX_BFS_DEPTH = int(os.getenv("MAX_BFS_DEPTH", "10"))

config: dict[str, tp.Any] = {"chats": {}}
db_pool: tp.Optional[Pool] = None

_FORUM_CACHE: dict[int, bool] = {}

# ---------- helpers ----------


def is_too_big_file(msg) -> bool:
    """
    Return True if this message has media and its file size exceeds MAX_FILE_SIZE_BYTES.
    Works for photo/video/document/audio/voice using Telethon's msg.file.size.
    """
    f = getattr(msg, "file", None)
    size = getattr(f, "size", None)
    if size is None:
        return False
    return size > MAX_FILE_SIZE_BYTES


def _topic_id_from_reply_header(msg: types.Message | types.MessageService) -> int | None:
    rt = getattr(msg, "reply_to", None)
    if not rt:
        return None

    if getattr(rt, "forum_topic", False):
        top = getattr(rt, "reply_to_top_id", None)
        if top:
            return int(top)

        mid = getattr(rt, "reply_to_msg_id", None)
        if mid:
            return int(mid)

    return None


async def _is_forum_chat(event: events.NewMessage.Event, chat_id: int) -> bool:
    cached = _FORUM_CACHE.get(chat_id)
    if cached is not None:
        return cached

    try:
        chat = await event.get_chat()
    except Exception:
        chat = None

    is_forum = bool(getattr(chat, "forum", False)) if chat is not None else False
    _FORUM_CACHE[chat_id] = is_forum
    return is_forum


async def _peer_config_key_from_event(event: events.NewMessage.Event) -> str:
    msg = event.message

    chat_id = getattr(event, "chat_id", None)
    if chat_id is None:
        peer = getattr(msg, "peer_id", None) or getattr(msg, "to_id", None)
        chat_id = get_peer_id(peer) if peer is not None else 0

    topic_id = _topic_id_from_reply_header(msg)
    if topic_id is not None:
        return f"topic:{chat_id}:{topic_id}"

    if await _is_forum_chat(event, chat_id):
        return f"topic:{chat_id}:1"

    return f"peer:{chat_id}"


async def _get_cfg_for_event(event: events.NewMessage.Event) -> dict[str, tp.Any]:
    key = await _peer_config_key_from_event(event)
    return config.get("chats", {}).get(key, {}) or {}


def _classify_message(msg) -> dict[str, bool]:
    has_text = bool((getattr(msg, "message", None) or "").strip())

    doc = getattr(msg, "document", None)
    mt = (getattr(doc, "mime_type", "") or "") if doc else ""

    is_video_doc = bool(doc and mt.startswith("video/"))
    is_audio_doc = bool(doc and mt.startswith("audio/"))

    has_image = bool(getattr(msg, "photo", None))
    has_video = bool(getattr(msg, "video", None) or is_video_doc)
    has_audio = bool(getattr(msg, "voice", None) or getattr(msg, "audio", None) or is_audio_doc)
    has_file = bool(doc and not is_video_doc and not is_audio_doc)

    return {
        "has_text": has_text,
        "has_image": has_image,
        "has_video": has_video,
        "has_audio": has_audio,
        "has_file": has_file,
    }


def _extract_meta_from_message(msg) -> dict[str, tp.Any]:
    peer = msg.peer_id or msg.to_id
    base_peer_id = get_peer_id(peer)

    topic_id = None
    if getattr(msg, "reply_to", None) and getattr(msg.reply_to, "forum_topic_id", None):
        topic_id = msg.reply_to.forum_topic_id

    is_forum = topic_id is not None
    message_id = msg.id
    timestamp = int(msg.date.timestamp())

    # Reply target message id (in the same chat)
    reply_to: int | None = None
    if getattr(msg, "reply_to", None):
        # Normal replies
        reply_to = getattr(msg.reply_to, "reply_to_msg_id", None)
        # In forum threads, sometimes only reply_to_top_id is present
        if reply_to is None:
            reply_to = getattr(msg.reply_to, "reply_to_top_id", None)

    return {
        "peer_id": base_peer_id,
        "topic_id": topic_id,
        "is_forum": is_forum,
        "message_id": message_id,
        "timestamp": timestamp,
        "reply_to": reply_to,
    }


def _extract_original_filename(msg, fallback_path: pathlib.Path) -> str:
    """
    Prefer Telegram's original filename if present (msg.file.name),
    otherwise fall back to the downloaded path name.
    """
    try:
        file_obj = getattr(msg, "file", None)
        if file_obj is not None:
            name = getattr(file_obj, "name", None)
            if name:
                return str(name)
    except Exception:
        pass
    return fallback_path.name


async def _download_media_for_generic_description(msg) -> tuple[str | None, str | None]:
    """
    Download media/document into a temp directory and run file_to_text on it.

    If the file is larger than MAX_FILE_SIZE_BYTES, the file is NOT downloaded
    and we return (filename_from_telegram_or_None, BIG_FILE_DESCRIPTION).

    Returns:
        (filename, description)
    """
    # Gating: do not download very large files
    try:
        if is_too_big_file(msg):
            file_obj = getattr(msg, "file", None)
            filename = getattr(file_obj, "name", None) if file_obj else None
            return filename, BIG_FILE_DESCRIPTION
    except Exception:
        # If something weird happens while checking size, fall back to normal download
        pass

    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="tg-media-"))
    try:
        # Pass directory; Telethon will choose filename (respecting original name)
        downloaded = await msg.download_media(file=str(tmpdir))
        if not downloaded:
            return None, None

        p = pathlib.Path(downloaded)
        filename = _extract_original_filename(msg, p)

        try:
            description = await file_to_text(str(p))
        except Exception:
            description = None

        return filename, description
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


async def _get_video_duration_seconds(path: str) -> float | None:
    """
    Use ffprobe to get video duration in seconds.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
            stdout=asp.PIPE,
            stderr=asp.PIPE,
        )
        out, err = await proc.communicate()
        if proc.returncode != 0:
            return None
        text = out.decode("utf-8", errors="ignore").strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


async def _process_video_for_descriptions(video_path: str) -> tuple[str | None, str | None]:
    """
    Given a video file path, use ffmpeg to:
      - extract audio track to WAV and run transcribe_audio
      - downsample video to <= MAX_VIDEO_FRAMES_TO_DESCRIBE frames and run describe_video

    Returns:
      (sound_description, vision_description)
    """
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="tg-video-"))
    audio_path = tmpdir / "audio_track.wav"
    video_out_path = tmpdir / "video_track.mp4"

    sound_desc: str | None = None
    vision_desc: str | None = None

    try:
        # --- Extract audio track ---
        cmd_audio = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(audio_path),
        ]
        proc_a = await asyncio.create_subprocess_exec(
            *cmd_audio,
            stdout=asp.PIPE,
            stderr=asp.PIPE,
        )
        _, _ = await proc_a.communicate()

        if proc_a.returncode == 0 and audio_path.exists():
            try:
                segments = await transcribe_audio(str(audio_path))
                lines: list[str] = []

                for seg in segments:
                    start = seg.get("start")
                    end = seg.get("end")
                    content = seg.get("content", "")
                    if content is None:
                        continue
                    content = str(content)

                    if start is not None and end is not None:
                        try:
                            lines.append(f"[{float(start):.2f}-{float(end):.2f}] {content}")
                        except (TypeError, ValueError):
                            lines.append(content)
                    else:
                        lines.append(content)

                joined = "\n".join(lines).strip()
                sound_desc = joined or None
            except Exception:
                sound_desc = None

        # --- Downsample video frames ---
        duration = await _get_video_duration_seconds(video_path)
        fps: float | None = None
        if duration and duration > 0 and MAX_VIDEO_FRAMES_TO_DESCRIBE > 0:
            fps = MAX_VIDEO_FRAMES_TO_DESCRIBE / duration
        if fps is None or fps <= 0:
            fps = 1.0

        cmd_video = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-an",
            "-vf",
            f"fps=fps={fps}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            str(video_out_path),
        ]
        proc_v = await asyncio.create_subprocess_exec(
            *cmd_video,
            stdout=asp.PIPE,
            stderr=asp.PIPE,
        )
        _, _ = await proc_v.communicate()

        video_target_path = video_out_path
        if proc_v.returncode != 0 or not video_out_path.exists():
            # ffmpeg failed; fall back to original video
            video_target_path = pathlib.Path(video_path)

        try:
            vision_desc = await describe_video(str(video_target_path))
        except Exception:
            vision_desc = None

        return sound_desc, vision_desc
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


async def _download_media_for_video(
    msg,
) -> tuple[str | None, str | None, str | None]:
    """
    Download a video (including Telegram GIF / animated docs) into a tmp dir,
    keep the original filename, then run _process_video_for_descriptions.

    If the file is larger than MAX_FILE_SIZE_BYTES, the file is NOT downloaded
    and we return (filename_from_telegram_or_None, BIG_FILE_DESCRIPTION, BIG_FILE_DESCRIPTION).

    Returns:
        (filename, sound_description, vision_description)
    """
    # Gating: do not download very large files
    try:
        if is_too_big_file(msg):
            file_obj = getattr(msg, "file", None)
            filename = getattr(file_obj, "name", None) if file_obj else None
            return filename, BIG_FILE_DESCRIPTION, BIG_FILE_DESCRIPTION
    except Exception:
        # If size check fails, fall back to normal download
        pass

    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="tg-media-"))
    try:
        downloaded = await msg.download_media(file=str(tmpdir))
        if not downloaded:
            return None, None, None

        p = pathlib.Path(downloaded)
        filename = _extract_original_filename(msg, p)

        sound_desc, vision_desc = await _process_video_for_descriptions(str(p))
        return filename, sound_desc, vision_desc
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


async def build_message_payload(event: events.NewMessage.Event) -> dict[str, tp.Any]:
    """
    Build JSON for a single non-album message.
    """
    msg = event.message
    cfg = await _get_cfg_for_event(event)
    kinds = _classify_message(msg)
    meta = _extract_meta_from_message(msg)

    payload: dict[str, tp.Any] = {
        **meta,
        "original_text": None,
        "masked_text": None,
        "media": [],
        "attachments": [],
    }

    logger.debug(f"Config:{to_string_human(cfg)}\nKinds:{to_string_human(kinds)}\nMeta: {to_string_human(meta)}")

    # --- TEXT ---
    if cfg.get("track_text", False) and kinds["has_text"]:
        original_text = getattr(msg, "message", None) or ""
        payload["original_text"] = original_text or None
        if original_text:
            try:
                payload["masked_text"] = await mask_text(original_text)
            except Exception:
                payload["masked_text"] = original_text

    media_list: list[dict[str, tp.Any]] = []
    attachments_list: list[dict[str, tp.Any]] = []
    media_counter = 0
    attach_counter = 0

    # --- IMAGE ---
    if cfg.get("track_image", False) and kinds["has_image"]:
        filename, description = await _download_media_for_generic_description(msg)
        if filename or description:
            media_counter += 1

            if description == BIG_FILE_DESCRIPTION:
                masked_description = BIG_FILE_DESCRIPTION
            else:
                masked_description = description
                if description:
                    try:
                        masked_description = await mask_text(description)
                    except Exception:
                        pass

            media_list.append(
                {
                    "id": f"{meta['message_id']}:media:{media_counter}",
                    "filename": filename,
                    "description": description,
                    "masked_description": masked_description,
                }
            )

    # --- VIDEO (with audio split + frame downsample) ---
    if cfg.get("track_video", False) and kinds["has_video"]:
        filename, sound_desc, vision_desc = await _download_media_for_video(msg)
        if filename or sound_desc or vision_desc:
            media_counter += 1

            # sound
            if sound_desc == BIG_FILE_DESCRIPTION:
                masked_sound = BIG_FILE_DESCRIPTION
            else:
                masked_sound = sound_desc
                if sound_desc:
                    try:
                        masked_sound = await mask_text(sound_desc)
                    except Exception:
                        masked_sound = sound_desc

            # vision
            if vision_desc == BIG_FILE_DESCRIPTION:
                masked_vision = BIG_FILE_DESCRIPTION
            else:
                masked_vision = vision_desc
                if vision_desc:
                    try:
                        masked_vision = await mask_text(vision_desc)
                    except Exception:
                        masked_vision = vision_desc

            item: dict[str, tp.Any] = {
                "id": f"{meta['message_id']}:media:{media_counter}",
                "filename": filename,
                "sound_description": sound_desc,
                "masked_sound_description": masked_sound,
                "vision_description": vision_desc,
                "masked_vision_description": masked_vision,
            }

            # Optional combined fields for convenience / backward-compat.
            if sound_desc or vision_desc:
                parts = []
                if sound_desc:
                    parts.append("[SOUND]\n" + sound_desc)
                if vision_desc:
                    parts.append("[VIDEO]\n" + vision_desc)
                item["description"] = "\n\n".join(parts)

                parts_masked = []
                if masked_sound:
                    parts_masked.append("[SOUND]\n" + masked_sound)
                if masked_vision:
                    parts_masked.append("[VIDEO]\n" + masked_vision)
                item["masked_description"] = "\n\n".join(parts_masked)

            media_list.append(item)

    # --- AUDIO ---
    if cfg.get("track_audio", False) and kinds["has_audio"]:
        filename, description = await _download_media_for_generic_description(msg)
        if filename or description:
            media_counter += 1

            if description == BIG_FILE_DESCRIPTION:
                masked_description = BIG_FILE_DESCRIPTION
            else:
                masked_description = description
                if description:
                    try:
                        masked_description = await mask_text(description)
                    except Exception:
                        pass

            media_list.append(
                {
                    "id": f"{meta['message_id']}:media:{media_counter}",
                    "filename": filename,
                    "description": description,
                    "masked_description": masked_description,
                }
            )

    # --- FILES (non-audio / non-video docs) ---
    if cfg.get("track_files", False) and kinds["has_file"]:
        filename, description = await _download_media_for_generic_description(msg)
        if filename or description:
            attach_counter += 1

            if description == BIG_FILE_DESCRIPTION:
                masked_description = BIG_FILE_DESCRIPTION
            elif description and len(description) <= MAX_FILE_DESCRIPTION_LENGTH_TO_MASK:
                try:
                    masked_description = await mask_text(description)
                except Exception:
                    masked_description = description
            else:
                masked_description = description

            attachments_list.append(
                {
                    "id": f"{meta['message_id']}:file:{attach_counter}",
                    "filename": filename,
                    "description": description,
                    "masked_description": masked_description,
                }
            )

    payload["media"] = media_list
    payload["attachments"] = attachments_list
    return payload


async def build_album_payload(event: events.Album.Event) -> tp.Optional[dict[str, tp.Any]]:
    """
    Build a single JSON for a media album (multi-media post).

    We store:
      message_id = first message's real msg.id (so you can retrieve with get_messages).
    """
    msgs = list(event.messages)
    if not msgs:
        return None

    first = msgs[0]
    peer = first.peer_id or first.to_id
    base_peer_id = get_peer_id(peer)

    topic_id = None
    if getattr(first, "reply_to", None) and getattr(first.reply_to, "forum_topic_id", None):
        topic_id = first.reply_to.forum_topic_id

    if topic_id is not None:
        key = f"topic:{base_peer_id}:{topic_id}"
    else:
        key = f"peer:{base_peer_id}"

    cfg = config.get("chats", {}).get(key) or {}
    if not cfg:
        # no config => track nothing
        return None

    is_forum = topic_id is not None
    main_msg_id = first.id
    timestamp = int(first.date.timestamp())

    payload: dict[str, tp.Any] = {
        "peer_id": base_peer_id,
        "topic_id": topic_id,
        "is_forum": is_forum,
        "message_id": main_msg_id,
        "timestamp": timestamp,
        "original_text": None,
        "masked_text": None,
        "media": [],
        "attachments": [],
    }

    # Caption text for album
    original_text = (event.text or "").strip()
    if original_text and cfg.get("track_text", False):
        payload["original_text"] = original_text
        try:
            payload["masked_text"] = await mask_text(original_text)
        except Exception:
            payload["masked_text"] = original_text

    media_tasks: list[tp.Awaitable[tp.Optional[dict[str, tp.Any]]]] = []
    attach_tasks: list[tp.Awaitable[tp.Optional[dict[str, tp.Any]]]] = []
    media_index = 0
    attach_index = 0

    async def process_image_item(idx: int, msg):
        filename, description = await _download_media_for_generic_description(msg)
        if not (filename or description):
            return None

        if description == BIG_FILE_DESCRIPTION:
            masked_description = BIG_FILE_DESCRIPTION
        else:
            masked_description = description
            if description:
                try:
                    masked_description = await mask_text(description)
                except Exception:
                    pass

        return {
            "id": f"{main_msg_id}:media:{idx}",
            "filename": filename,
            "description": description,
            "masked_description": masked_description,
        }

    async def process_video_item(idx: int, msg):
        filename, sound_desc, vision_desc = await _download_media_for_video(msg)
        if not (filename or sound_desc or vision_desc):
            return None

        # sound
        if sound_desc == BIG_FILE_DESCRIPTION:
            masked_sound = BIG_FILE_DESCRIPTION
        else:
            masked_sound = sound_desc
            if sound_desc:
                try:
                    masked_sound = await mask_text(sound_desc)
                except Exception:
                    masked_sound = sound_desc

        # vision
        if vision_desc == BIG_FILE_DESCRIPTION:
            masked_vision = BIG_FILE_DESCRIPTION
        else:
            masked_vision = vision_desc
            if vision_desc:
                try:
                    masked_vision = await mask_text(vision_desc)
                except Exception:
                    masked_vision = vision_desc

        item: dict[str, tp.Any] = {
            "id": f"{main_msg_id}:media:{idx}",
            "filename": filename,
            "sound_description": sound_desc,
            "masked_sound_description": masked_sound,
            "vision_description": vision_desc,
            "masked_vision_description": masked_vision,
        }

        if sound_desc or vision_desc:
            parts = []
            if sound_desc:
                parts.append("[SOUND]\n" + sound_desc)
            if vision_desc:
                parts.append("[VIDEO]\n" + vision_desc)
            item["description"] = "\n\n".join(parts)

            parts_masked = []
            if masked_sound:
                parts_masked.append("[SOUND]\n" + masked_sound)
            if masked_vision:
                parts_masked.append("[VIDEO]\n" + masked_vision)
            item["masked_description"] = "\n\n".join(parts_masked)

        return item

    async def process_audio_item(idx: int, msg):
        filename, description = await _download_media_for_generic_description(msg)
        if not (filename or description):
            return None

        if description == BIG_FILE_DESCRIPTION:
            masked_description = BIG_FILE_DESCRIPTION
        else:
            masked_description = description
            if description:
                try:
                    masked_description = await mask_text(description)
                except Exception:
                    pass

        return {
            "id": f"{main_msg_id}:media:{idx}",
            "filename": filename,
            "description": description,
            "masked_description": masked_description,
        }

    async def process_attach_item(idx: int, msg):
        filename, description = await _download_media_for_generic_description(msg)
        if not (filename or description):
            return None

        if description == BIG_FILE_DESCRIPTION:
            masked_description = BIG_FILE_DESCRIPTION
        elif description and len(description) <= MAX_FILE_DESCRIPTION_LENGTH_TO_MASK:
            try:
                masked_description = await mask_text(description)
            except Exception:
                masked_description = description
        else:
            masked_description = description

        return {
            "id": f"{main_msg_id}:file:{idx}",
            "filename": filename,
            "description": description,
            "masked_description": masked_description,
        }

    # classify each message once, then schedule the correct task
    for msg in msgs:
        kinds = _classify_message(msg)

        if cfg.get("track_image", False) and kinds["has_image"]:
            media_index += 1
            media_tasks.append(process_image_item(media_index, msg))
            continue

        if cfg.get("track_video", False) and kinds["has_video"]:
            media_index += 1
            media_tasks.append(process_video_item(media_index, msg))
            continue

        if cfg.get("track_audio", False) and kinds["has_audio"]:
            media_index += 1
            media_tasks.append(process_audio_item(media_index, msg))
            continue

        if cfg.get("track_files", False) and kinds["has_file"]:
            attach_index += 1
            attach_tasks.append(process_attach_item(attach_index, msg))
            continue

    media_results: list[tp.Optional[dict[str, tp.Any]]] = []
    attach_results: list[tp.Optional[dict[str, tp.Any]]] = []

    if media_tasks:
        media_results = await asyncio.gather(*media_tasks, return_exceptions=False)

    if attach_tasks:
        attach_results = await asyncio.gather(*attach_tasks, return_exceptions=False)

    payload["media"] = [res for res in media_results if res is not None]
    payload["attachments"] = [res for res in attach_results if res is not None]

    if not payload["original_text"] and not payload["media"] and not payload["attachments"]:
        # nothing tracked for this album
        return None

    return payload


async def should_track(event: events.NewMessage.Event) -> bool:
    """
    True if at least one actually present source in this message is enabled in config.
    (Used only for non-album messages.)
    """
    cfg = await _get_cfg_for_event(event)
    if not cfg:
        return False

    msg = event.message
    kinds = _classify_message(msg)

    if cfg.get("track_text", False) and kinds["has_text"]:
        return True
    if cfg.get("track_image", False) and kinds["has_image"]:
        return True
    if cfg.get("track_video", False) and kinds["has_video"]:
        return True
    if cfg.get("track_audio", False) and kinds["has_audio"]:
        return True
    if cfg.get("track_files", False) and kinds["has_file"]:
        return True

    return False


async def load_config_from_redis(r: aioredis.Redis) -> None:
    global config
    raw = await r.get(REDIS_CONFIG_KEY)
    if not raw:
        config = {"chats": {}}
        return
    try:
        config = json.loads(raw.decode("utf-8"))
    except Exception:
        config = {"chats": {}}


async def watch_config_updates(r: aioredis.Redis) -> None:
    pubsub = r.pubsub()
    await pubsub.subscribe(REDIS_CONFIG_CHANNEL)
    async for message in pubsub.listen():
        if message.get("type") != "message":
            continue
        await load_config_from_redis(r)


# ---------- retrieval & forwarding helpers ----------


async def fetch_message_by_id(
    client: TelegramClient,
    peer_id: int,
    message_id: int,
):
    """
    Fetch a single Telegram message by peer_id + message_id.
    Works for both normal and album messages, as long as message_id
    is a real msg.id in that chat (we store it that way).
    """
    try:
        entity = await client.get_entity(peer_id)
        msg = await client.get_messages(entity, ids=message_id)
        return msg
    except Exception:
        logger.exception(f"[ERROR] fetch_message_by_id peer={peer_id} msg={message_id}", exc_info=True)
        return None


async def forward_message_by_id(
    client: TelegramClient,
    from_peer_id: int,
    message_id: int,
    to_peer_id: int,
):
    """
    Forward a single message from one chat to another by ids.
    If the message is part of an album, only that element is forwarded.
    """
    try:
        from_entity = await client.get_entity(from_peer_id)
        to_entity = await client.get_entity(to_peer_id)

        msg = await client.get_messages(from_entity, ids=message_id)
        if not msg:
            logger.warning("[WARN] forward_message_by_id: source message not found")
            return

        await client.forward_messages(
            to_entity,
            msg,
            from_peer=from_entity,
        )
    except Exception as e:
        logger.exception(
            f"[ERROR] forward_message_by_id from={from_peer_id} msg={message_id} to={to_peer_id}: {e!r}", exc_info=True
        )


# ---------- database functions ----------


async def init_database():
    """Initialize database connection pool"""
    global db_pool
    db_pool = await asyncpg.create_pool(
        user=POSTGRESQL_USER,
        password=POSTGRESQL_PASSWORD,
        database=POSTGRESQL_DATABASE_NAME,
        host=POSTGRESQL_HOST,
        port=POSTGRESQL_PORT,
        ssl=POSTGRESQL_SSL,
    )


async def store_message(payload: dict[str, tp.Any]) -> int:
    """Store message in database and return database ID"""
    async with db_pool.acquire() as conn:
        # Extract reply_to information if present
        message_id = await conn.fetchval(
            """
            INSERT INTO messages 
            (telegram_chat_id, telegram_message_id, timestamp, topic_id, is_forum, reply_to_message_id, payload)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (telegram_chat_id, telegram_message_id) 
            DO UPDATE SET payload = EXCLUDED.payload
            RETURNING id
        """,
            payload["peer_id"],
            payload["message_id"],
            payload["timestamp"],
            payload.get("topic_id"),
            payload.get("is_forum", False),
            payload.get("reply_to", None),
            json.dumps(payload),
        )
        return message_id


async def get_message_by_telegram_id(chat_id: int, message_id: int) -> tp.Optional[dict]:
    """Get message from database by Telegram IDs"""
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM messages WHERE telegram_chat_id = $1 AND telegram_message_id = $2", chat_id, message_id
        )
        return dict(row) if row else None


async def get_sequential_neighbors(chat_id: int, message_id: int, limit: int = 5) -> list[dict]:
    """Get sequential neighbors of a message"""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM find_sequential_neighbors($1, $2, $3)", chat_id, message_id, limit)
        return [dict(row) for row in rows]


async def store_graph_embedding(
    root_message_id: int,
    chat_id: int,
    graph_size: int,
    nodes_ids: list[int],
    edges: list[list[tp.Any]],
    embedding: list[float],
) -> None:
    """Store graph embedding in database"""

    # Convert Python list -> JSON for JSONB column
    edges_json = json.dumps(edges, ensure_ascii=False)

    # Convert Python list[float] -> pgvector text format: [0.1,0.2,...]
    # (unless you register a proper pgvector codec)
    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO graph_embeddings 
            (root_message_id, telegram_chat_id, graph_size, nodes_ids, edges, embedding)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            root_message_id,
            chat_id,
            graph_size,
            nodes_ids,  # BIGINT[] -> list[int] is fine
            edges_json,  # JSONB
            embedding_str,  # vector(3072) via implicit cast from text
        )


# ---------- graph functions ----------


async def build_message_graph(
    root_chat_id: int, root_message_id: int, target_size: int
) -> tp.Tuple[list[int], list[list[tp.Any]]]:
    visited: set[tuple[int, int]] = set()
    nodes: list[tuple[int, int]] = []  # list of (chat_id, message_id)
    edges: list[list[tp.Any]] = []  # list of [node_idx1, node_idx2, connection_type]

    # Queue: (chat_id, message_id, depth, node_index_in_nodes_array)
    queue: asyncio.Queue[tuple[int, int, int, int]] = asyncio.Queue()
    await queue.put((root_chat_id, root_message_id, 0, 0))
    visited.add((root_chat_id, root_message_id))

    total_nodes = 1

    while not queue.empty():
        chat_id, message_id, depth, node_index = await queue.get()

        # Keep the BFS logical node regardless of DB presence;
        # we'll filter missing ones later.
        nodes.append((chat_id, message_id))

        if depth >= MAX_BFS_DEPTH:
            continue

        # Get message from database to find replies / sequential neighbors
        message = await get_message_by_telegram_id(chat_id, message_id)
        if not message:
            # No DB row -> we do not expand neighbors from this node,
            # but edges from other nodes may still point to it.
            continue

        payload = json.loads(message["payload"])

        # Follow reply edges first
        if total_nodes < target_size and "reply_to" in payload and payload["reply_to"]:
            reply_to_id = payload["reply_to"]
            reply_key = (chat_id, reply_to_id)

            if reply_key not in visited:
                visited.add(reply_key)
                await queue.put((chat_id, reply_to_id, depth + 1, total_nodes))
                edges.append([node_index, total_nodes, "reply"])
                total_nodes += 1

        # Then follow sequential edges (previous neighbours)
        if total_nodes < target_size:
            neighbors = await get_sequential_neighbors(chat_id, message_id, 1)

            for neighbor in neighbors:
                neighbor_msg_id = neighbor["telegram_message_id"]
                neighbor_key = (chat_id, neighbor_msg_id)

                if neighbor_key not in visited:
                    visited.add(neighbor_key)
                    await queue.put((chat_id, neighbor_msg_id, depth + 1, total_nodes))
                    edges.append([node_index, total_nodes, "sequential"])
                    total_nodes += 1

    # ---------- Post-processing: filter nodes that actually exist in DB ----------

    # Build list of (timestamp, msg_id, chat_id) only for messages that exist
    node_messages: list[tuple[int, int, int]] = []
    msg_id_to_timestamp: dict[int, int] = {}

    for chat_id, msg_id in nodes:
        msg = await get_message_by_telegram_id(chat_id, msg_id)
        if not msg:
            continue
        ts = int(msg["timestamp"])
        node_messages.append((ts, msg_id, chat_id))
        msg_id_to_timestamp[msg_id] = ts

    # If nothing valid was found
    if not node_messages:
        return [], []

    # Sort by timestamp
    node_messages.sort(key=lambda x: x[0])
    sorted_ids: list[int] = [msg_id for _, msg_id, _ in node_messages]

    # Map msg_id -> index in sorted_ids
    msg_id_to_sorted_idx: dict[int, int] = {msg_id: idx for idx, msg_id in enumerate(sorted_ids)}

    # ---------- Remap edges to sorted indices, drop edges to missing nodes ----------

    sorted_edges: list[list[tp.Any]] = []
    for from_idx_old, to_idx_old, conn_type in edges:
        # Safeguard against any out-of-range indices
        if not (0 <= from_idx_old < len(nodes) and 0 <= to_idx_old < len(nodes)):
            continue

        from_msg_id = nodes[from_idx_old][1]
        to_msg_id = nodes[to_idx_old][1]

        # Skip if either endpoint doesn't have a valid DB message
        if from_msg_id not in msg_id_to_sorted_idx or to_msg_id not in msg_id_to_sorted_idx:
            continue

        from_idx_new = msg_id_to_sorted_idx[from_msg_id]
        to_idx_new = msg_id_to_sorted_idx[to_msg_id]

        sorted_edges.append([from_idx_new, to_idx_new, conn_type])

    return sorted_ids, sorted_edges


async def generate_graph_prompt(
    client: TelegramClient,
    chat_id: int,
    message_ids: list[int],
    edges: list[list[tp.Any]],
) -> str:
    """Generate prompt text for a graph of Telegram messages."""
    prompt_parts: list[str] = []

    # Chat title
    chat_title = await get_chat_title(client, chat_id) or f"Chat {chat_id}"

    for idx, message_id in enumerate(message_ids):
        message = await get_message_by_telegram_id(chat_id, message_id)
        if not message:
            continue

        payload = json.loads(message["payload"])
        prompt_parts_local: list[str] = []

        # Collect edges connected to this node
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

        # Header
        timestamp = datetime.fromtimestamp(payload["timestamp"])
        header = (
            f'# --- Telegram Message {message_id} in Chat "{chat_title}" at {timestamp.strftime("%Y-%m-%d %H:%M:%S")}'
        )

        if connected_edges:
            edge_descs: list[str] = []
            for other_idx, conn_type in connected_edges:
                # Only describe edges to already-seen messages to avoid duplication
                if other_idx < idx:
                    edge_descs.append(f"{conn_type} to message {message_ids[other_idx]}")

            if edge_descs:
                header += f" ({', '.join(edge_descs)})"

        prompt_parts_local.append(header)
        prompt_parts_local.append("")

        # Attachments
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

        # Media
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
                
                if desc is None:
                    desc = "(empty)"

                prompt_parts_local.append(f'### Media "{filename}"\n```\n{desc[:MAX_CHARS_PER_MEDIA_IN_PROMPT]}\n```')
                prompt_parts_local.append("")

        # Message content
        text = payload.get("masked_text") or payload.get("original_text") or "(empty)"
        prompt_parts_local.append(f"## Message Content:\n```\n{text}\n```")

        prompt_parts.append("\n".join(prompt_parts_local).strip())

    return "\n\n".join(prompt_parts).strip()


async def get_chat_title(client: TelegramClient, chat_id: int) -> str:
    me = await client.get_me()

    # Special case: Saved Messages
    if chat_id == me.id:
        return "Saved Messages"

    entity = await client.get_entity(chat_id)

    # Groups/channels/supergroups
    if getattr(entity, "title", None):
        return entity.title

    # Private chats (users)
    parts = []
    if getattr(entity, "first_name", None):
        parts.append(entity.first_name)
    if getattr(entity, "last_name", None):
        parts.append(entity.last_name)
    if parts:
        return " ".join(parts)

    if getattr(entity, "username", None):
        return entity.username

    # Fallback
    return str(entity.id)


async def process_message_graphs(client: TelegramClient, db_message_id: int, chat_id: int, message_id: int):
    """Process graphs for different sizes and store embeddings"""
    previous_graph_size, actual_size_already_was = None, False

    for graph_size in GRAPH_SIZES:
        try:
            if actual_size_already_was:
                break

            # Build graph
            nodes_ids, edges = await build_message_graph(chat_id, message_id, graph_size)

            # If we couldn't build a graph of requested size, skip
            if len(nodes_ids) < graph_size:
                if len(nodes_ids) == 1 and graph_size > 1:
                    break  # Skip if we only have the root but want larger graphs

                # Use the actual size we got
                actual_size = len(nodes_ids)
                actual_size_already_was = True

                if previous_graph_size == actual_size:
                    break
            else:
                actual_size = graph_size

            # Generate prompt and get embedding
            prompt = await generate_graph_prompt(client, chat_id, nodes_ids, edges)
            logger.debug(f"=================\n{prompt}\n====================\n\n")
            embedding = await get_embedding(prompt)

            # Store graph embedding
            await store_graph_embedding(db_message_id, chat_id, actual_size, nodes_ids, edges, embedding)

            logger.info(f"Stored graph embedding for message {message_id}, size {actual_size}")

            previous_graph_size = graph_size

        except Exception:
            logger.exception(f"Error processing graph size {graph_size} for message {message_id}", exc_info=True)
            traceback.print_exc()


# ---------- message handlers ----------


async def watch_forward_queue(r: aioredis.Redis, client: TelegramClient) -> None:
    while True:
        try:
            result = await r.blpop(REDIS_FORWARD_QUEUE_KEY)
            if not result or len(result) < 2:
                continue

            _, raw = result
            try:
                item = json.loads(raw.decode("utf-8"))
            except Exception:
                logger.exception("[ERROR] watch_forward_queue: invalid JSON", exc_info=True)
                continue

            target_chat_id = item.get("target_chat_id")
            if target_chat_id is None:
                continue

            try:
                target_chat_id = int(target_chat_id)
            except (TypeError, ValueError):
                continue

            msgs = item.get("messages_to_show") or item.get("messages") or []
            if not isinstance(msgs, list):
                continue

            for m in msgs:
                try:
                    src_chat_id = int(m.get("chat_id"))
                    src_msg_id = int(m.get("msg_id"))
                except Exception:
                    continue

                await forward_message_by_id(
                    client,
                    from_peer_id=src_chat_id,
                    message_id=src_msg_id,
                    to_peer_id=target_chat_id,
                )

        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("[ERROR] watch_forward_queue loop", exc_info=True)
            await asyncio.sleep(1.0)


async def handle_processed_message(client: TelegramClient, payload: dict[str, tp.Any]):
    """Handle a processed message - store in DB and build graphs"""
    try:
        # Store message in database
        db_message_id = await store_message(payload)

        # Process graphs in background
        asyncio.create_task(process_message_graphs(client, db_message_id, payload["peer_id"], payload["message_id"]))

    except Exception:
        logger.exception("Error storing message or processing graphs", exc_info=True)


# ---------- main event handlers ----------


async def main() -> None:
    await init_database()

    r = aioredis.from_url(REDIS_URL)
    r_forward = aioredis.from_url(REDIS_URL)

    await load_config_from_redis(r)

    client = TelegramClient(SESSION_PATH, CLIENT_APP_API_ID, CLIENT_APP_API_HASH)

    @client.on(events.NewMessage)
    async def handle_new_message(event: events.NewMessage.Event):
        if getattr(event.message, "grouped_id", None):
            return

        if not await should_track(event):
            return

        try:
            payload = await build_message_payload(event)
        except Exception:
            logger.exception("[ERROR] building payload for NewMessage", exc_info=True)
            return

        logger.debug(json.dumps(payload, ensure_ascii=False))

        # storing and processing graphs
        await handle_processed_message(client, payload)

    @client.on(events.Album)
    async def handle_album(event: events.Album.Event):
        try:
            payload = await build_album_payload(event)
        except Exception:
            logger.exception("[ERROR] building payload for Album", exc_info=True)
            return

        if not payload:
            return

        logger.debug(json.dumps(payload, indent=2, ensure_ascii=False))

        # storing and processing graphs
        await handle_processed_message(client, payload)

    watcher_task = asyncio.create_task(watch_config_updates(r))
    forward_task = asyncio.create_task(watch_forward_queue(r_forward, client))

    async with client:
        await client.run_until_disconnected()

    watcher_task.cancel()
    forward_task.cancel()
    try:
        await watcher_task
    except asyncio.CancelledError:
        pass
    try:
        await forward_task
    except asyncio.CancelledError:
        pass

    await r.aclose()
    await r_forward.aclose()
    await db_pool.close()


if __name__ == "__main__":
    asyncio.run(main())
