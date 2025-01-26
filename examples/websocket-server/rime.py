from typing import AsyncGenerator
import aiohttp
from aiohttp import ClientTimeout

from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.ai_services import TTSService


class RimeTTSService(TTSService):
    def __init__(
        self,
        *,
        voice: str = "aura-helios-en",
        sample_rate: int = 22050,
        encoding: str = "linear16",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._settings = {
            "sample_rate": sample_rate,
            "encoding": encoding,
        }
        self.set_voice(voice)

    def can_generate_metrics(self) -> bool:
        return True

    def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        async def generator():
            logger.debug(f"Generating TTS: [{text}]")

            url = "http://{}:{}/inference_{}".format("192.168.50.81", "50000", "cross_lingual")
             
            headers = {
            }
            form_data = aiohttp.FormData()
            form_data.add_field('tts_text', text)
            try:
                await self.start_ttfb_metrics()

                async with aiohttp.ClientSession(
                    timeout=ClientTimeout(total=3000)
                ) as session:
                    async with session.post(
                        url, data=form_data, headers=headers
                    ) as response:
                        if response.status != 200:
                            raise ValueError(f"Rime API error: {response.status}")

                        await self.start_tts_usage_metrics(text)
                        yield TTSStartedFrame()

                        await self.stop_ttfb_metrics()

                        chunk_size = 8192  # Use a fixed buffer size
                        async for chunk in response.content.iter_any():
                            if chunk:
                                frame = TTSAudioRawFrame(
                                    audio=chunk,
                                    sample_rate=self._settings["sample_rate"],
                                    num_channels=1,
                                )
                                yield frame

                yield TTSStoppedFrame()

            except Exception as e:
                logger.exception(f"{self} exception: {e}")
                yield ErrorFrame(f"Error getting audio: {str(e)}")

        return (frame async for frame in generator())
