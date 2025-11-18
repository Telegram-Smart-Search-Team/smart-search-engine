import aiohttp
import asyncio


API_URL = "http://localhost:8080/v1/audio/transcribe"


async def main():
    async with aiohttp.ClientSession() as session:
        with open(
            "/workspace/userspace/telegram-smart-search/smart-search-engine/src/test/LearningEnglishConversations-20251111-TheEnglishWeSpeakNoTwoWaysAboutIt.mp3",
            "rb",
        ) as f:
            form = aiohttp.FormData()
            form.add_field("audio_file", f, filename="call.mp3", content_type="audio/mpeg")

            async with session.post(API_URL, data=form) as resp:
                data = await resp.json()
                for seg in data["segments"]:
                    print(f'{seg["start"]:.2f}-{seg["end"]:.2f}: {seg["content"]}')


asyncio.run(main())
