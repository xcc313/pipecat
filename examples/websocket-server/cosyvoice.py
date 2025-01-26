import requests
import base64
import io
import array
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)

 
from pipecat.services.ai_services import TTSService


ValidVoice = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
VALID_VOICES: Dict[str, ValidVoice] = {
    "alloy": "alloy",
    "echo": "echo",
    "fable": "fable",
    "onyx": "onyx",
    "nova": "nova",
    "shimmer": "shimmer",
}

class OpenAITTSService(TTSService):
    """OpenAI Text-to-Speech service that generates audio from text.

    This service uses the OpenAI TTS API to generate PCM-encoded audio at 24kHz.
    When using with DailyTransport, configure the sample rate in DailyParams
    as shown below:

    DailyParams(
        audio_out_enabled=True,
        audio_out_sample_rate=24_000,
    )

    Args:
        api_key: OpenAI API key. Defaults to None.
        voice: Voice ID to use. Defaults to "alloy".
        model: TTS model to use ("tts-1" or "tts-1-hd"). Defaults to "tts-1".
        sample_rate: Output audio sample rate in Hz. Defaults to 24000.
        **kwargs: Additional keyword arguments passed to TTSService.

    The service returns PCM-encoded audio at the specified sample rate.
    """

    def __init__(
        self,
        *,
        voice: str = "alloy",
        model: Literal["tts-1", "tts-1-hd"] = "tts-1",
        sample_rate: int = 22050,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._settings = {
            "sample_rate": sample_rate,
        }
        self.set_model_name(model)
        self.set_voice(voice)

    def can_generate_metrics(self) -> bool:
        return True

    async def set_model(self, model: str):
        logger.info(f"Switching TTS model to: [{model}]")
        self.set_model_name(model)

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        await self.start_ttfb_metrics()

        url = "http://{}:{}/inference_{}".format("192.168.50.81", "50000", "cross_lingual")
        payload = {
            'tts_text': text,
        }
        response = requests.request("POST", url, data=payload, stream=True)

        await self.start_tts_usage_metrics(text)

        yield TTSStartedFrame()
        for chunk in response.iter_content(chunk_size=8192):
            if len(chunk) > 0:
                await self.stop_ttfb_metrics()
                frame = TTSAudioRawFrame(chunk, self._settings["sample_rate"], 1)
                yield frame
        yield TTSStoppedFrame()


