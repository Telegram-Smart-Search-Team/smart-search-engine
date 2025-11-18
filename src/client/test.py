import os
import pathlib
from dotenv import load_dotenv

BASE_DIR = pathlib.Path(__file__).parents[2]
load_dotenv(dotenv_path=str(BASE_DIR / ".env"))

# telethon & related imports
from telethon import TelegramClient


CLIENT_APP_API_ID = os.getenv("CLIENT_APP_API_ID")
CLIENT_APP_API_HASH = os.getenv("CLIENT_APP_API_HASH")
SESSION_PATH = str(BASE_DIR / "base_session.session")


async def print_last_messages_per_dialog():
    async for dialog in client.iter_dialogs():
        entity = dialog.entity

        # Optional: filter if you want only specific types
        # if not (dialog.is_user or dialog.is_group or dialog.is_channel):
        #     continue

        print("=" * 60)
        print(f"Dialog: {dialog.name!r} (id={dialog.id})")
        # For bots: dialog.is_user and getattr(entity, 'bot', False) is True

        # get_messages returns newest first
        messages = await client.get_messages(entity, limit=N)

        # Reverse to print from oldest -> newest if you want chronology
        for msg in reversed(messages):
            # Some messages have no text (media-only)
            text = msg.message or "<no text>"
            print(f"[{msg.date}] {msg.id}: {text}")


async def fetch_single_message():
    # chat_identifier can be:
    # - @username
    # - phone number
    # - numeric ID
    # - dialog.entity from iter_dialogs()
    # - InputPeer / User / Chat / Channel object
    chat_identifier = -1001124038902

    message_id = 65924  # the message id inside that chat

    entity = await client.get_entity(chat_identifier)
    msg = await client.get_messages(entity, ids=message_id)

    # msg is a telethon.tl.custom.message.Message or None
    if msg:
        print(f"[{msg.date}] {msg.sender_id}: {msg.text!r}")
    else:
        print("Message not found (deleted or wrong id).")


if __name__ == "__main__":
    client = TelegramClient(SESSION_PATH, CLIENT_APP_API_ID, CLIENT_APP_API_HASH)

    N = 5  # how many last messages per dialog

    with client:
        client.loop.run_until_complete(print_last_messages_per_dialog())
        # client.loop.run_until_complete(fetch_single_message())
