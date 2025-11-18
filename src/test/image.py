import aiohttp
import asyncio

API_URL = "http://localhost:8080/v1/qwen/image"


async def main():
    async with aiohttp.ClientSession() as session:
        with open(
            "/workspace/userspace/telegram-smart-search/smart-search-engine/src/test/tg_image_196380812.png", "rb"
        ) as f:
            form = aiohttp.FormData()
            form.add_field("image_file", f, filename="image.png", content_type="image/png")
            form.add_field("prompt", "Describe this image in Russian.")
            form.add_field("system", "You are a careful visual describer.")

            async with session.post(API_URL, data=form) as resp:
                data = await resp.json()
                print(data)


asyncio.run(main())
