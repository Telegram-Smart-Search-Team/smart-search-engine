import os
import pathlib
import asyncio
import asyncpg
from dotenv import load_dotenv


BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=str(BASE_DIR / ".env"))


async def init_db():
    conn = await asyncpg.connect(
        user=os.getenv("POSTGRESQL_USER"),
        password=os.getenv("POSTGRESQL_PASSWORD"),
        database=os.getenv("POSTGRESQL_DATABASE_NAME"),
        host=os.getenv("POSTGRESQL_HOST"),
        port=os.getenv("POSTGRESQL_PORT"),
    )

    with open(str(BASE_DIR / "src/database/schema.sql"), "r") as f:
        await conn.execute(f.read())

    await conn.close()


asyncio.run(init_db())
