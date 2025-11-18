import aiohttp
import asyncio

API_URL = "http://localhost:8080/v1/qwen/text"


async def main():
    async with aiohttp.ClientSession() as session:
        payload = {
            "system": "You are a sanitizer. Mask all passwords, tokens and secrets.",
            "prompt": "My email is foo@bar.com and my password is 1234abcd.",
            "max_new_tokens": 128,
            "temperature": 0.0,
        }
        async with session.post(API_URL, json=payload) as resp:
            data = await resp.json()
            print(data)


asyncio.run(main())
