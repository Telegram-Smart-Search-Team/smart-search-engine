import aiohttp
import asyncio


API_URL = "http://localhost:8080/v1/qwen/video"


async def main():
    async with aiohttp.ClientSession() as session:
        with open("/workspace/userspace/telegram-smart-search/smart-search-engine/src/test/tenor.gif", "rb") as f:
            form = aiohttp.FormData()
            form.add_field("video_file", f, filename="clip.gif", content_type="image/gif")
            form.add_field("system", "You are an expert video describer for Telegram media attachments.")
            form.add_field(
                "prompt",
                "Describe what happens in this video. Split the clip into sensible segments. Describe people, objects, locations, actions, emotions, camera movement, on-screen text and logos. Try to identify and name some well known people, brands or animated characters.",
            )

            async with session.post(API_URL, data=form) as resp:
                data = await resp.json()
                print(data)


asyncio.run(main())
