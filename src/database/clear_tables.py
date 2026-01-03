import os
import asyncio
import pathlib

import asyncpg
from dotenv import load_dotenv

BASE_DIR = pathlib.Path(__file__).parents[2]
load_dotenv(dotenv_path=str(BASE_DIR / ".env"))


async def clear_tables() -> None:
    conn = await asyncpg.connect(
        user=os.environ["POSTGRESQL_USER"],
        password=os.environ["POSTGRESQL_PASSWORD"],
        database=os.environ["POSTGRESQL_DATABASE_NAME"],
        host=os.environ.get("POSTGRESQL_HOST", "localhost"),
        port=int(os.environ.get("POSTGRESQL_PORT", "5432")),
    )

    try:
        await conn.execute("TRUNCATE TABLE graph_embeddings, messages RESTART IDENTITY;")
        print("✅ Cleared tables: messages, graph_embeddings (RESTART IDENTITY).")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(clear_tables())
