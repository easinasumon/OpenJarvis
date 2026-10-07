"""Local Bengali text-to-speech using Meta MMS-TTS.

The default model is facebook/mms-tts-ben (VITS). It runs locally after
the first model download and does not require an API key.
"""

from __future__ import annotations

import io
import threading
from typing import Any, List

from openjarvis.core.registry import TTSRegistry
from openjarvis.speech.tts import TTSBackend, TTSResult

_DEFAULT_MODEL_ID = "facebook/mms-tts-ben"


@TTSRegistry.register("mms_bengali")
class MMSBengaliTTSBackend(TTSBackend):
    """Offline Bengali TTS backed by Meta MMS VITS."""

    backend_id = "mms_bengali"

    def __init__(
        self,
        *,
        model_id: str = _DEFAULT_MODEL_ID,
        device: str = "auto",
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._lock = threading.RLock()

    def _resolved_device(self) -> str:
        if self._device != "auto":
            return self._device
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_model(self) -> tuple[Any, Any]:
        with self._lock:
            if self._model is not None and self._tokenizer is not None:
                return self._tokenizer, self._model

            try:
                import torch
                from transformers import AutoTokenizer, VitsModel
            except ImportError as exc:
                raise RuntimeError(
                    "Bengali TTS dependencies are missing. "
                    "Run: uv sync --extra voice"
                ) from exc

            device = self._resolved_device()
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
            self._model = VitsModel.from_pretrained(self._model_id)
            self._model.to(device).eval()
            return self._tokenizer, self._model

    def synthesize(
        self,
        text: str,
        *,
        voice_id: str = _DEFAULT_MODEL_ID,
        speed: float = 1.0,
        output_format: str = "wav",
    ) -> TTSResult:
        import numpy as np
        import soundfile as sf
        import torch

        if not text.strip():
            return TTSResult(
                audio=b"",
                format=output_format,
                voice_id=voice_id or self._model_id,
                metadata={"backend": self.backend_id, "model": self._model_id},
            )

        tokenizer, model = self._ensure_model()
        device = self._resolved_device()
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        speaking_rate = max(0.7, min(float(speed), 1.5))

        with self._lock, torch.inference_mode():
            outputs = model(**inputs, speaking_rate=speaking_rate)

        waveform = outputs.waveform[0].detach().cpu().float().numpy()
        waveform = np.asarray(waveform, dtype=np.float32)
        sample_rate = int(model.config.sampling_rate)

        buf = io.BytesIO()
        sf.write(buf, waveform, sample_rate, format="WAV")
        audio = buf.getvalue()

        return TTSResult(
            audio=audio,
            format="wav",
            voice_id=voice_id or self._model_id,
            sample_rate=sample_rate,
            duration_seconds=(len(waveform) / sample_rate if sample_rate else 0.0),
            metadata={"backend": self.backend_id, "model": self._model_id},
        )

    def available_voices(self) -> List[str]:
        return [self._model_id]

    def health(self) -> bool:
        # Keep discovery cheap: do not download/load the model here.
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False
        return True


__all__ = ["MMSBengaliTTSBackend"]
