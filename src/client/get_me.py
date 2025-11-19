import os
import pathlib
from dotenv import load_dotenv

BASE_DIR = pathlib.Path(__file__).parents[2]
load_dotenv(dotenv_path=str(BASE_DIR / ".env"))

# telethon & related imports
from telethon.sync import TelegramClient


CLIENT_APP_API_ID = os.getenv("CLIENT_APP_API_ID")
CLIENT_APP_API_HASH = os.getenv("CLIENT_APP_API_HASH")
SESSION_PATH = str(BASE_DIR / "base_session.session")


with TelegramClient(SESSION_PATH, CLIENT_APP_API_ID, CLIENT_APP_API_HASH) as tclient:
    print(tclient.get_me().id)
