# ElevenLabs Text-to-Speech

OpenJarvis can speak with any [ElevenLabs](https://elevenlabs.io) voice using the
official Text-to-Speech API. ElevenLabs is a **cloud** backend: the text OpenJarvis
speaks is sent to ElevenLabs. Kokoro (local), OpenAI TTS and Cartesia keep working
unchanged.

## 1. Configuration

Add this to `~/.openjarvis/config.toml`:

```toml
[speech]
tts_backend = "elevenlabs"
voice_id = "QAmlwgbPtjxpk7u98Qs9"
```

`speech.voice_id` is passed straight to ElevenLabs as the voice to use. Replace the
example ID with your own voice; you can copy it from the voice's page in the
ElevenLabs Voice Library or from your account.

A ready-to-copy file lives at `configs/openjarvis/examples/elevenlabs-tts.toml`.

## 2. API key

The key is read **only** from the `ELEVENLABS_API_KEY` environment variable. It is
never read from, or written to, `config.toml`, so it cannot end up in git.

=== "Windows (PowerShell)"

    ```powershell
    setx ELEVENLABS_API_KEY "your-key-here"
    ```

    Close and reopen the terminal afterwards. `setx` stores the variable in your
    user environment, so it is available in every new terminal.

=== "macOS / Linux"

    ```bash
    export ELEVENLABS_API_KEY="your-key-here"   # add to ~/.bashrc or ~/.zshrc
    ```

Create a key at <https://elevenlabs.io/app/developers/api-keys>.

## 3. Use it

```bash
jarvis chat --voice
```

The `text_to_speech` tool also accepts `backend = "elevenlabs"`; when no voice is
given it uses `speech.voice_id`.

## Models, Bengali and English

The default model is **`eleven_v3`**. It is the ElevenLabs model that supports
Bengali (74 languages including Bengali and English). `eleven_multilingual_v2`
supports 29 languages and does **not** include Bengali.

To use another model, set `ELEVENLABS_MODEL`, for example `eleven_flash_v2_5` for
lower latency (no Bengali).

## Notes

- `voice_speed` is sent to ElevenLabs only when it is not `1.0`, and is clamped to
  the 0.7-1.2 range ElevenLabs accepts. Some models may ignore it.
- Voice IDs are not portable between backends. If ElevenLabs is unavailable and
  OpenJarvis falls back to another backend, that backend uses its own default voice.
- Some Voice Library voices require a paid ElevenLabs plan through the API. In that
  case ElevenLabs returns an error such as `paid_plan_required`, which OpenJarvis
  shows (without your key).
- ElevenLabs is reported as a cloud TTS backend by `jarvis scan --data-boundaries`
  (the data-boundary audit), and `ELEVENLABS_API_KEY` is reported as present or
  absent only, never its value.
