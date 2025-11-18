import asyncio
import os
import pathlib
import datetime

from dotenv import load_dotenv

from telethon import TelegramClient, types
from telethon.utils import get_peer_id
from telethon.tl.functions.messages import (
    GetForumTopicsRequest,
    GetCustomEmojiDocumentsRequest,
)

# ---------- config / env ----------

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=str(BASE_DIR / ".env"))

CLIENT_APP_API_ID = int(os.getenv("CLIENT_APP_API_ID"))
CLIENT_APP_API_HASH = os.getenv("CLIENT_APP_API_HASH")

SESSION_PATH = str(BASE_DIR / "base_session.session")

OUTPUT_DIR = BASE_DIR / "forum_debug"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

client = TelegramClient(SESSION_PATH, CLIENT_APP_API_ID, CLIENT_APP_API_HASH)

# emoji_id -> png_path (in-memory cache for this process)
emoji_png_cache: dict[int, str] = {}

# limit heavy conversions running in parallel
CONCURRENT_CONVERSIONS = 4
conversion_semaphore = asyncio.Semaphore(CONCURRENT_CONVERSIONS)


# ---------- helpers ----------


async def get_all_forum_topics(peer: types.Channel):
    """
    Fetch all forum topics for a given forum-enabled channel using
    messages.GetForumTopicsRequest with proper pagination.
    """
    topics: list[types.ForumTopic] = []

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


async def download_group_photo(channel: types.Channel) -> str | None:
    """
    Download group's profile photo (if any) and return local path.
    """
    file_path = OUTPUT_DIR / f"group_{get_peer_id(channel)}.jpg"
    try:
        path = await client.download_profile_photo(channel, file=str(file_path))
        return path
    except Exception as e:
        print(f"    [warn] failed to download group photo: {type(e).__name__}: {e}")
        return None


async def ensure_emoji_png(emoji_id: int, doc: types.Document) -> str | None:
    """
    For a given custom emoji id + document, ensure we have a PNG on disk.

    - Uses in-memory cache so we never convert the same emoji twice in one run.
    - Uses OUTPUT_DIR / f"emoji_{emoji_id}.png" as canonical PNG path.
    """
    if emoji_id in emoji_png_cache:
        return emoji_png_cache[emoji_id]

    base = OUTPUT_DIR / f"emoji_{emoji_id}"
    png_path = base.with_suffix(".png")

    # fast path: already converted before this run
    if png_path.exists():
        emoji_png_cache[emoji_id] = str(png_path)
        return str(png_path)

    # 1) download raw file once
    raw_path_str = await client.download_media(doc, file=str(base))
    if raw_path_str is None:
        return None

    raw_path = pathlib.Path(raw_path_str)
    mime = doc.mime_type or ""

    # 2) heavy I/O / CPU / subprocess under semaphore
    async with conversion_semaphore:
        if mime == "image/webp":
            # static sticker → PNG (Pillow in a thread)
            from PIL import Image

            def _convert_webp_to_png():
                img = Image.open(raw_path)
                img.save(png_path)

            await asyncio.to_thread(_convert_webp_to_png)
            emoji_png_cache[emoji_id] = str(png_path)
            return str(png_path)

        elif mime == "application/x-tgsticker":
            # .tgs (Lottie) → PNG (first frame) via lottie_convert.py
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

            # fallback: keep raw file path
            emoji_png_cache[emoji_id] = str(raw_path)
            return str(raw_path)

        elif mime == "video/webm":
            # video emoji → PNG using ffmpeg first frame
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
            # unknown / already PNG/JPEG/etc.
            emoji_png_cache[emoji_id] = str(raw_path)
            return str(raw_path)


async def fetch_emoji_docs_for_topics(topics: list[types.ForumTopic]):
    """
    For given topics, collect all distinct icon_emoji_id values and fetch
    all their documents in a single GetCustomEmojiDocumentsRequest.

    Returns: dict[emoji_id -> types.Document]

    NOTE: ForumTopic.icon_emoji_id is the same as Document.id returned
    by GetCustomEmojiDocumentsRequest, so we just map doc.id -> doc.
    """
    emoji_ids = {getattr(t, "icon_emoji_id", None) for t in topics}
    emoji_ids = {e for e in emoji_ids if e}

    if not emoji_ids:
        return {}

    docs = await client(GetCustomEmojiDocumentsRequest(document_id=list(emoji_ids)))

    # map returned Document.id back to the emoji_id
    emoji_id_to_doc: dict[int, types.Document] = {}
    for doc in docs:
        doc_id = getattr(doc, "id", None)
        if doc_id is not None:
            emoji_id_to_doc[doc_id] = doc

    return emoji_id_to_doc


# ---------- main listing logic ----------


async def list_matryoshka_groups():
    """
    Iterate dialogs, find forum-enabled supergroups (matryoshka groups),
    print group info + topics + cached PNG icons.
    """
    async for dialog in client.iter_dialogs():
        entity = dialog.entity

        if not isinstance(entity, types.Channel):
            continue

        is_supergroup = bool(entity.megagroup)
        is_forum = bool(getattr(entity, "forum", False))

        if not (is_supergroup and is_forum):
            continue

        group_peer_id = get_peer_id(entity)  # -100...
        group_title = entity.title or ""

        group_photo_path = await download_group_photo(entity)

        print("=== MATRYOSHKA GROUP ===")
        print(f"id: {group_peer_id}")
        print(f"title: {group_title!r}")
        print(f"photo_file: {group_photo_path}")
        print("subgroups:")

        topics = await get_all_forum_topics(entity)

        if not topics:
            print("  <no topics found>")
            print()
            continue

        # 1) fetch all emoji docs for this group in one RPC
        emoji_docs = await fetch_emoji_docs_for_topics(topics)

        # 2) convert each distinct emoji to PNG (parallel, limited)
        conversion_tasks = []
        for emoji_id, doc in emoji_docs.items():

            async def _convert(eid=emoji_id, d=doc):
                try:
                    path = await ensure_emoji_png(eid, d)
                    return eid, path
                except Exception as exc:
                    print(f"      [warn] conversion failed for emoji {eid}: " f"{type(exc).__name__}: {exc}")
                    return eid, None

            conversion_tasks.append(_convert())

        emoji_png_paths: dict[int, str | None] = {}
        if conversion_tasks:
            results = await asyncio.gather(*conversion_tasks)
            for eid, path in results:
                emoji_png_paths[eid] = path

        # 3) print topics using cached PNG paths
        for t in topics:
            topic_id = t.id
            topic_title = t.title or ""
            icon_emoji_id = getattr(t, "icon_emoji_id", None)

            icon_file = None
            if icon_emoji_id:
                icon_file = emoji_png_paths.get(icon_emoji_id)

            print("  - topic:")
            print(f"      id: {topic_id}")
            print(f"      title: {topic_title!r}")
            print(f"      icon_emoji_id: {icon_emoji_id}")
            print(f"      icon_file: {icon_file}")
        print()


async def main():
    async with client:
        await list_matryoshka_groups()


if __name__ == "__main__":
    asyncio.run(main())
