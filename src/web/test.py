import asyncio
import os
import pathlib
import datetime
import subprocess

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


async def get_all_forum_topics(peer: types.Channel):
    """
    Fetch all forum topics for a given forum-enabled channel using
    messages.GetForumTopicsRequest with proper pagination.
    """
    topics = []

    # Telegram "date" type -> datetime; start from epoch
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
                # q can be omitted (None) to list all topics
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

        # msg.id -> msg.date mapping to update offset_date
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


async def download_topic_icon_png(icon_emoji_id: int, group_peer_id: int, topic_id: int) -> str | None:
    """
    Download topic icon and convert it to PNG.

    Supported:
      - image/webp         => PNG
      - application/x-tgsticker (.tgs) => PNG via lottie
      - video/webm         => PNG (first frame) via ffmpeg

    Returns path to PNG file (or raw file path if conversion not possible),
    or None on total failure.
    """
    if not icon_emoji_id:
        return None

    try:
        docs = await client(GetCustomEmojiDocumentsRequest(document_id=[icon_emoji_id]))
        if not docs:
            return None

        doc: types.Document = docs[0]
        mime = doc.mime_type or ""
        base = OUTPUT_DIR / f"topic_{group_peer_id}_{topic_id}"

        # 1) Download raw file
        raw_path = await client.download_media(doc, file=str(base))
        if raw_path is None:
            return None

        raw_path = pathlib.Path(raw_path)

        # 2) Decide how to convert
        png_path = raw_path.with_suffix(".png")

        if mime == "image/webp":
            # static sticker → PNG
            from PIL import Image

            img = Image.open(raw_path)
            img.save(png_path)
            return str(png_path)

        elif mime == "application/x-tgsticker":
            # .tgs (Lottie) → PNG (first frame) via python-lottie CLI
            # lottie_convert.py AnimatedSticker.tgs output_file.png --frame 0
            subprocess.run(
                ["lottie_convert.py", str(raw_path), str(png_path), "--frame", "0"],
                check=True,
            )
            return str(png_path)

        elif mime == "video/webm":
            # video emoji → PNG using ffmpeg first frame
            # ffmpeg -y -i in.webm -vframes 1 out.png
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(raw_path),
                    "-vframes",
                    "1",
                    str(png_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return str(png_path)

        else:
            # Unknown / already PNG/JPEG/etc., just return what we have
            return str(raw_path)

    except Exception as e:
        print(f"      [warn] failed to download/convert topic icon for topic {topic_id}: " f"{type(e).__name__}: {e}")
        return None


async def list_matryoshka_groups():
    """
    Iterate dialogs, find forum-enabled supergroups (matryoshka groups),
    print group info + topics.
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

        for t in topics:
            topic_id = t.id
            topic_title = t.title or ""
            icon_emoji_id = getattr(t, "icon_emoji_id", None)

            icon_file = None
            if icon_emoji_id:
                icon_file = await download_topic_icon_png(icon_emoji_id, group_peer_id, topic_id)

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
