"""ElevenLabs text-to-speech backend.

Uses the official ElevenLabs Text-to-Speech REST endpoint
(``POST /v1/text-to-speech/{voice_id}``).

The API key is read **only** from the ``ELEVENLABS_API_KEY`` environment
variable (or passed explicitly in code); it is never stored in config files.

The default model is ``eleven_v3`` because it is the ElevenLabs model that
supports Bengali as well as English (``eleven_multilingual_v2`` does not cover
Bengali). Override with ``ELEVENLABS_MODEL`` if you prefer another model, e.g.
``eleven_flash_v2_5`` for lower latency (English/Hindi etc., no Bengali).
"""

from __future__ import annotations

import os
from typing import List

import httpx

from openjarvis.core.registry import TTSRegistry
from openjarvis.speech.tts import TTSBackend, TTSResult

_ELEVENLABS_API_BASE = "https://api.elevenlabs.io"
_API_KEY_ENV = "ELEVENLABS_API_KEY"
_MODEL_ENV = "ELEVENLABS_MODEL"
DEFAULT_MODEL = "eleven_v3"

# OpenJarvis output_format -> (ElevenLabs output_format, result format, sample rate)
_FORMATS = {
    "mp3": ("mp3_44100_128", "mp3", 44100),
    "pcm": ("pcm_24000", "pcm", 24000),
}

# ElevenLabs accepts voice_settings.speed in this range.
_MIN_SPEED, _MAX_SPEED = 0.7, 1.2


def _elevenlabs_synthesize(
    api_key: str,
    text: str,
    voice_id: str,
    model: str = DEFAULT_MODEL,
    output_format: str = "mp3_44100_128",
    speed: float = 1.0,
) -> bytes:
    """Call the ElevenLabs TTS API and return raw audio bytes."""
    body: dict = {"text": text, "model_id": model}
    if speed != 1.0:
        body["voice_settings"] = {"speed": max(_MIN_SPEED, min(_MAX_SPEED, speed))}

    resp = httpx.post(
        f"{_ELEVENLABS_API_BASE}/v1/text-to-speech/{voice_id}",
        params={"output_format": output_format},
        headers={"xi-api-key": api_key},
        json=body,
        timeout=120.0,
    )
    if resp.status_code >= 400:
        # Surface ElevenLabs' own error text (e.g. paid-plan-required for
        # library voices) without ever echoing request headers / the key.
        detail = resp.text[:300].replace(api_key, "***") if api_key else resp.text[:300]
        raise RuntimeError(f"ElevenLabs TTS failed ({resp.status_code}): {detail}")
    return resp.content


@TTSRegistry.register("elevenlabs")
class ElevenLabsTTSBackend(TTSBackend):
    """ElevenLabs TTS backend (multilingual, incl. Bengali via ``eleven_v3``)."""

    backend_id = "elevenlabs"

    def __init__(self, *, api_key: str = "", model: str = "") -> None:
        self._api_key = api_key or os.environ.get(_API_KEY_ENV, "")
        self._model = model or os.environ.get(_MODEL_ENV, "") or DEFAULT_MODEL

    def synthesize(
        self,
        text: str,
        *,
        voice_id: str = "",
        speed: float = 1.0,
        output_format: str = "mp3",
    ) -> TTSResult:
        if not self._api_key:
            raise RuntimeError(f"{_API_KEY_ENV} not set")
        if not voice_id:
            raise RuntimeError(
                "ElevenLabs needs a voice ID: set `voice_id` under [speech] "
                "in ~/.openjarvis/config.toml"
            )

        api_format, result_format, sample_rate = _FORMATS.get(
            output_format, _FORMATS["mp3"]
        )
        audio = _elevenlabs_synthesize(
            self._api_key,
            text,
            voice_id=voice_id,
            model=self._model,
            output_format=api_format,
            speed=speed,
        )
        return TTSResult(
            audio=audio,
            format=result_format,
            voice_id=voice_id,
            sample_rate=sample_rate,
            metadata={"backend": "elevenlabs", "model": self._model},
        )

    def available_voices(self) -> List[str]:
        if not self._api_key:
            return []
        resp = httpx.get(
            f"{_ELEVENLABS_API_BASE}/v1/voices",
            headers={"xi-api-key": self._api_key},
            timeout=30.0,
        )
        resp.raise_for_status()
        return [v["voice_id"] for v in resp.json().get("voices", [])]

    def health(self) -> bool:
        return bool(self._api_key)
