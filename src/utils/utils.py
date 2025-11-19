import os
import sys
import pathlib
import aiohttp
import mimetypes
from dotenv import load_dotenv

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=str(BASE_DIR / ".env"))

sys.path.append(str(BASE_DIR / "src"))

# OpenAI configuration
OPENAI_API_KEY = os.getenv("CHATGPT_TOKEN")
OPENAI_EMBEDDING_URL = "https://api.openai.com/v1/embeddings"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
MAX_OPENAI_EMBEDDING_INPUT = int(os.getenv("MAX_OPENAI_EMBEDDING_INPUT"))


def guess_mimetype(path: str, default: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or default


async def get_embedding(text: str) -> list[float]:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    data = {"model": OPENAI_EMBEDDING_MODEL, "input": text[:MAX_OPENAI_EMBEDDING_INPUT]}

    async with aiohttp.ClientSession() as session:
        async with session.post(OPENAI_EMBEDDING_URL, headers=headers, json=data) as response:
            if response.status == 200:
                result = await response.json()
                return result["data"][0]["embedding"]
            else:
                error_text = await response.text()
                raise Exception(f"OpenAI API error: {error_text}")


if __name__ == "__main__":
    pass
