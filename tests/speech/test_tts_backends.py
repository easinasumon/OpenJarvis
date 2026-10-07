"""Tests for TTS backend infrastructure."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from openjarvis.core.registry import TTSRegistry
from openjarvis.speech.tts import TTSResult

# ---------------------------------------------------------------------------
# TTSResult tests
# ---------------------------------------------------------------------------


def test_tts_result_dataclass():
    result = TTSResult(
        audio=b"fake-audio-bytes",
        format="mp3",
        duration_seconds=3.5,
        voice_id="jarvis-v1",
    )
    assert result.audio == b"fake-audio-bytes"
    assert result.format == "mp3"
    assert result.duration_seconds == 3.5


def test_tts_result_save(tmp_path):
    result = TTSResult(audio=b"fake-mp3-data", format="mp3")
    out = result.save(tmp_path / "test.mp3")
    assert out.exists()
    assert out.read_bytes() == b"fake-mp3-data"


# ---------------------------------------------------------------------------
# Cartesia backend tests
# ---------------------------------------------------------------------------


def test_cartesia_registered():
    from openjarvis.speech.cartesia_tts import CartesiaTTSBackend

    TTSRegistry.register_value("cartesia", CartesiaTTSBackend)
    assert TTSRegistry.contains("cartesia")


def test_cartesia_synthesize():
    from openjarvis.speech.cartesia_tts import CartesiaTTSBackend

    backend = CartesiaTTSBackend(api_key="fake-key")

    with patch(
        "openjarvis.speech.cartesia_tts._cartesia_synthesize",
        return_value=b"fake-audio-mp3-bytes",
    ):
        result = backend.synthesize("Hello world", voice_id="test-voice")

    assert result.audio == b"fake-audio-mp3-bytes"
    assert result.format == "mp3"
    assert result.voice_id == "test-voice"


# ---------------------------------------------------------------------------
# ElevenLabs backend tests (no real network / paid API requests)
# ---------------------------------------------------------------------------

_USER_VOICE = "QAmlwgbPtjxpk7u98Qs9"


def _fake_response(status=200, content=b"fake-mp3", text=""):
    from unittest.mock import MagicMock

    resp = MagicMock()
    resp.status_code = status
    resp.content = content
    resp.text = text
    return resp


def test_elevenlabs_registered():
    from openjarvis.speech.elevenlabs_tts import ElevenLabsTTSBackend

    TTSRegistry.register_value("elevenlabs", ElevenLabsTTSBackend)
    assert TTSRegistry.contains("elevenlabs")
    assert TTSRegistry.get("elevenlabs") is ElevenLabsTTSBackend


def test_elevenlabs_in_discovery_order():
    from openjarvis.speech._tts_discovery import TTS_BACKEND_ORDER

    assert "elevenlabs" in TTS_BACKEND_ORDER
    # existing backends must remain, in their original relative order
    assert [b for b in TTS_BACKEND_ORDER if b != "elevenlabs"] == [
        "kokoro",
        "openai_tts",
        "cartesia",
    ]


def test_elevenlabs_reads_key_only_from_env(monkeypatch):
    from openjarvis.speech.elevenlabs_tts import ElevenLabsTTSBackend

    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    assert ElevenLabsTTSBackend().health() is False

    monkeypatch.setenv("ELEVENLABS_API_KEY", "env-key")
    backend = ElevenLabsTTSBackend()
    assert backend.health() is True
    assert backend._api_key == "env-key"


def test_elevenlabs_default_model_supports_bengali(monkeypatch):
    from openjarvis.speech.elevenlabs_tts import ElevenLabsTTSBackend

    monkeypatch.delenv("ELEVENLABS_MODEL", raising=False)
    # eleven_multilingual_v2 does NOT cover Bengali; eleven_v3 does.
    assert ElevenLabsTTSBackend(api_key="k")._model == "eleven_v3"


def test_elevenlabs_synthesize_request_shape():
    from openjarvis.speech.elevenlabs_tts import ElevenLabsTTSBackend

    backend = ElevenLabsTTSBackend(api_key="fake-key")
    with patch(
        "openjarvis.speech.elevenlabs_tts.httpx.post",
        return_value=_fake_response(content=b"fake-mp3-bytes"),
    ) as post:
        result = backend.synthesize("আসসালামু আলাইকুম, hello", voice_id=_USER_VOICE)

    assert result.audio == b"fake-mp3-bytes"
    assert result.format == "mp3"
    assert result.voice_id == _USER_VOICE
    assert result.metadata["backend"] == "elevenlabs"

    args, kwargs = post.call_args
    # speech.voice_id is passed through to the ElevenLabs URL
    assert args[0] == f"https://api.elevenlabs.io/v1/text-to-speech/{_USER_VOICE}"
    assert kwargs["headers"] == {"xi-api-key": "fake-key"}
    assert kwargs["params"] == {"output_format": "mp3_44100_128"}
    assert kwargs["json"]["model_id"] == "eleven_v3"
    assert kwargs["json"]["text"] == "আসসালামু আলাইকুম, hello"
    assert "voice_settings" not in kwargs["json"]  # speed 1.0 -> not sent


def test_elevenlabs_speed_is_clamped():
    from openjarvis.speech.elevenlabs_tts import ElevenLabsTTSBackend

    backend = ElevenLabsTTSBackend(api_key="fake-key")
    with patch(
        "openjarvis.speech.elevenlabs_tts.httpx.post",
        return_value=_fake_response(),
    ) as post:
        backend.synthesize("hi", voice_id=_USER_VOICE, speed=3.0)
    assert post.call_args.kwargs["json"]["voice_settings"] == {"speed": 1.2}


def test_elevenlabs_requires_api_key_and_voice(monkeypatch):
    from openjarvis.speech.elevenlabs_tts import ElevenLabsTTSBackend

    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with patch("openjarvis.speech.elevenlabs_tts.httpx.post") as post:
        with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
            ElevenLabsTTSBackend().synthesize("hi", voice_id=_USER_VOICE)
        with pytest.raises(RuntimeError, match="voice"):
            ElevenLabsTTSBackend(api_key="k").synthesize("hi")
        post.assert_not_called()


def test_elevenlabs_http_error_never_leaks_key():
    from openjarvis.speech.elevenlabs_tts import ElevenLabsTTSBackend

    backend = ElevenLabsTTSBackend(api_key="super-secret-key")
    bad = _fake_response(status=402, text="paid_plan_required super-secret-key")
    with patch("openjarvis.speech.elevenlabs_tts.httpx.post", return_value=bad):
        with pytest.raises(RuntimeError) as exc:
            backend.synthesize("hi", voice_id=_USER_VOICE)
    assert "402" in str(exc.value)
    assert "paid_plan_required" in str(exc.value)
    assert "super-secret-key" not in str(exc.value)


# ---------------------------------------------------------------------------
# Kokoro backend tests
# ---------------------------------------------------------------------------


def test_kokoro_registered():
    from openjarvis.speech.kokoro_tts import KokoroTTSBackend

    TTSRegistry.register_value("kokoro", KokoroTTSBackend)
    assert TTSRegistry.contains("kokoro")


def test_kokoro_health_false_without_package():
    from openjarvis.speech.kokoro_tts import KokoroTTSBackend

    # The assertion only holds when kokoro is genuinely absent. With the
    # optional `voice` extra installed, health() legitimately returns True
    # (and warming the pipeline would hit the HF Hub), so skip instead.
    try:
        import kokoro  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("kokoro installed (openjarvis[voice]); health() is True")

    backend = KokoroTTSBackend()
    # Without kokoro installed, health returns False
    assert backend.health() is False


def test_kokoro_lang_for_voice_mandarin():
    """Mandarin voice IDs (zf_*, zm_*) map to lang_code='z'."""
    from openjarvis.speech.kokoro_tts import KokoroTTSBackend

    assert KokoroTTSBackend._lang_for_voice("zf_xiaoxiao") == "z"
    assert KokoroTTSBackend._lang_for_voice("zm_yunxi") == "z"


def test_kokoro_lang_for_voice_english_and_other_languages():
    """Voice prefix detection covers English variants and other languages."""
    from openjarvis.speech.kokoro_tts import KokoroTTSBackend

    assert KokoroTTSBackend._lang_for_voice("af_heart") == "a"
    assert KokoroTTSBackend._lang_for_voice("am_adam") == "a"
    assert KokoroTTSBackend._lang_for_voice("bf_emma") == "b"
    assert KokoroTTSBackend._lang_for_voice("jf_alpha") == "j"
    # Kokoro 0.9.x has no Korean KPipeline language code. Do not route a
    # catalog-style Korean voice ID to an unsupported ``lang_code='k'``.
    assert KokoroTTSBackend._lang_for_voice("kf_example") == "a"


def test_kokoro_lang_for_voice_unknown_falls_back_to_english():
    """Unknown prefixes fall back to American English (Kokoro's most-stocked
    language) rather than crashing — keeps backward-compat for any
    user-supplied voice ID we haven't catalogued."""
    from openjarvis.speech.kokoro_tts import KokoroTTSBackend

    assert KokoroTTSBackend._lang_for_voice("") == "a"
    assert KokoroTTSBackend._lang_for_voice("xx_unknown") == "a"


def test_kokoro_available_voices_includes_mandarin_and_english():
    """Voice catalog must list at least one Mandarin voice and preserve
    the original English voices for backward compatibility."""
    from openjarvis.speech.kokoro_tts import KokoroTTSBackend

    voices = KokoroTTSBackend().available_voices()
    # New Mandarin coverage (the focus of this change)
    for v in ("zf_xiaoxiao", "zf_xiaoyi", "zm_yunxi", "zm_yunjian"):
        assert v in voices, f"Mandarin voice {v} missing from catalog"
    # Backward compat — voices the previous implementation listed
    for v in ("af_heart", "af_bella", "am_adam", "am_michael"):
        assert v in voices, f"Existing voice {v} dropped from catalog"


def test_kokoro_missing_chinese_deps_gives_actionable_error(monkeypatch):
    """When Mandarin G2P deps are missing, the user must see an install hint
    pointing at ``misaki[zh]`` rather than a raw ``ordered_set`` ImportError."""
    import sys
    import types

    import pytest

    from openjarvis.speech.kokoro_tts import KokoroTTSBackend

    class FakeKPipeline:
        def __init__(self, lang_code, model):
            # Mimic what happens inside Kokoro when misaki.zh can't import.
            raise ImportError("No module named 'ordered_set'")

    class FakeKModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

    fake_kokoro = types.SimpleNamespace(KModel=FakeKModel, KPipeline=FakeKPipeline)
    monkeypatch.setitem(sys.modules, "kokoro", fake_kokoro)

    backend = KokoroTTSBackend()
    with pytest.raises(RuntimeError) as exc_info:
        backend._ensure_pipeline("z")
    msg = str(exc_info.value)
    assert "misaki[zh]" in msg
    assert "lang_code='z'" in msg or 'lang_code="z"' in msg


def test_kokoro_pipeline_cached_per_language(monkeypatch):
    """Each lang_code instantiates one pipeline; same lang reuses the cache."""
    import sys
    import types

    from openjarvis.speech.kokoro_tts import KokoroTTSBackend

    init_calls = []

    class FakeKModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def to(self, _device):
            return self

        def eval(self):
            return self

    class FakeKPipeline:
        def __init__(self, lang_code, model):
            init_calls.append(lang_code)
            self.model = model

    fake_kokoro = types.SimpleNamespace(KModel=FakeKModel, KPipeline=FakeKPipeline)
    monkeypatch.setitem(sys.modules, "kokoro", fake_kokoro)

    backend = KokoroTTSBackend()
    backend._ensure_pipeline("a")
    backend._ensure_pipeline("z")
    backend._ensure_pipeline("a")  # cache hit — must NOT re-instantiate
    backend._ensure_pipeline("z")  # cache hit

    assert init_calls == ["a", "z"]


def test_kokoro_pipeline_cache_is_thread_safe(monkeypatch):
    """Concurrent requests for one language construct exactly one pipeline."""
    import sys
    import threading
    import time
    import types

    from openjarvis.speech.kokoro_tts import KokoroTTSBackend

    init_calls = []

    class FakeKModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

    class FakeKPipeline:
        def __init__(self, lang_code, model):
            init_calls.append(lang_code)
            time.sleep(0.02)

    monkeypatch.setitem(
        sys.modules,
        "kokoro",
        types.SimpleNamespace(KModel=FakeKModel, KPipeline=FakeKPipeline),
    )
    backend = KokoroTTSBackend()
    pipelines = []

    def load_pipeline():
        pipelines.append(backend._ensure_pipeline("z"))

    threads = [threading.Thread(target=load_pipeline) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()

    assert init_calls == ["z"]
    assert len({id(pipeline) for pipeline in pipelines}) == 1


def test_kokoro_pipeline_cache_evicts_lru_and_cleans_up(monkeypatch):
    """The cache is bounded and closes evicted and explicitly released entries."""
    import sys
    import types

    from openjarvis.speech.kokoro_tts import KokoroTTSBackend

    instances = {}

    class FakeKModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

    class FakeKPipeline:
        def __init__(self, lang_code, model):
            self.closed = False
            instances[lang_code] = self

        def close(self):
            self.closed = True

    monkeypatch.setitem(
        sys.modules,
        "kokoro",
        types.SimpleNamespace(KModel=FakeKModel, KPipeline=FakeKPipeline),
    )
    backend = KokoroTTSBackend(max_cached_pipelines=2)

    backend._ensure_pipeline("a")
    backend._ensure_pipeline("z")
    backend._ensure_pipeline("a")  # make American English most recently used
    backend._ensure_pipeline("j")

    assert list(backend._pipelines) == ["a", "j"]
    assert instances["z"].closed is True
    assert instances["a"].closed is False
    assert instances["j"].closed is False

    backend.close()
    assert not backend._pipelines
    assert instances["a"].closed is True
    assert instances["j"].closed is True


def test_kokoro_pipeline_cache_size_must_be_positive():
    import pytest

    from openjarvis.speech.kokoro_tts import KokoroTTSBackend

    with pytest.raises(ValueError, match="at least 1"):
        KokoroTTSBackend(max_cached_pipelines=0)


def test_kokoro_synthesize_routes_voice_to_correct_language(monkeypatch):
    """synthesize() with a Mandarin voice initializes a Mandarin pipeline
    and tags the result metadata with the resolved lang_code."""
    import sys
    import types

    import numpy as np

    from openjarvis.speech.kokoro_tts import KokoroTTSBackend

    init_args = []
    call_args = []

    class FakeKModel:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

    class FakeKPipeline:
        def __init__(self, lang_code, model):
            init_args.append(lang_code)
            self.model = model

        def __call__(self, text, voice, speed):
            call_args.append({"text": text, "voice": voice, "speed": speed})
            yield (None, None, np.zeros(2400, dtype=np.float32))

    fake_kokoro = types.SimpleNamespace(KModel=FakeKModel, KPipeline=FakeKPipeline)
    monkeypatch.setitem(sys.modules, "kokoro", fake_kokoro)

    # ``soundfile`` is a kokoro-runtime dep; stub it so we can test the
    # routing logic without requiring it in the OpenJarvis test env.
    def _fake_sf_write(buf, _samples, _sr, format=None):  # noqa: A002
        buf.write(b"FAKE_AUDIO_BYTES")

    fake_soundfile = types.SimpleNamespace(write=_fake_sf_write)
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)

    backend = KokoroTTSBackend()
    result = backend.synthesize("你好世界", voice_id="zf_xiaoxiao")

    assert init_args == ["z"]
    assert call_args[0]["voice"] == "zf_xiaoxiao"
    assert call_args[0]["text"] == "你好世界"
    assert result.voice_id == "zf_xiaoxiao"
    assert result.metadata["lang_code"] == "z"
    assert result.metadata["backend"] == "kokoro"
    assert result.format == "wav"
    assert len(result.audio) > 0


def test_kokoro_reuses_model_and_honors_path_and_device(monkeypatch):
    """All language pipelines share the configured model instance."""
    import sys
    import types

    from openjarvis.speech.kokoro_tts import KokoroTTSBackend

    model_inits = []
    model_devices = []
    pipeline_models = []

    class FakeKModel:
        def __init__(self, **kwargs):
            model_inits.append(kwargs)

        def to(self, device):
            model_devices.append(device)
            return self

        def eval(self):
            return self

    class FakeKPipeline:
        def __init__(self, lang_code, model):
            pipeline_models.append((lang_code, model))

    monkeypatch.setitem(
        sys.modules,
        "kokoro",
        types.SimpleNamespace(KModel=FakeKModel, KPipeline=FakeKPipeline),
    )

    backend = KokoroTTSBackend(model_path="/models/kokoro.pth", device="cpu")
    backend._ensure_pipeline("a")
    backend._ensure_pipeline("z")

    assert model_inits == [{"model": "/models/kokoro.pth"}]
    assert model_devices == ["cpu"]
    assert pipeline_models[0][1] is pipeline_models[1][1]


# ---------------------------------------------------------------------------
# OpenAI TTS backend tests
# ---------------------------------------------------------------------------


def test_openai_tts_registered():
    from openjarvis.speech.openai_tts import OpenAITTSBackend

    TTSRegistry.register_value("openai_tts", OpenAITTSBackend)
    assert TTSRegistry.contains("openai_tts")


def test_openai_tts_synthesize():
    from openjarvis.speech.openai_tts import OpenAITTSBackend

    backend = OpenAITTSBackend(api_key="fake-key")

    with patch(
        "openjarvis.speech.openai_tts._openai_tts_request",
        return_value=b"fake-openai-audio",
    ):
        result = backend.synthesize("Hello", voice_id="nova")

    assert result.audio == b"fake-openai-audio"
    assert result.voice_id == "nova"
