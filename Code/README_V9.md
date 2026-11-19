# KHMERDUB AI STUDIO — V9 Development Package

This package is based on the supplied V8 source and contains a first engineering hardening pass. It is **not yet a claim of production readiness**.

## Changes in this V9 development pass

- Video duration probing now fails with an actionable error instead of silently inventing a 10-second duration.
- Speaker assignment no longer guesses speaker identity or gender from zero-crossing rate. When a Hugging Face token is configured, the pipeline attempts pyannote speaker diarization; without it, lines are explicitly assigned to one neutral `Speaker 01` rather than fabricated multi-speaker labels.
- Speaker reference clips are extracted per diarized speaker when a diarization model is available.
- VoxCPM2 errors are no longer silently hidden by Edge TTS. Edge TTS fallback is disabled by default and must be explicitly enabled in settings.
- Audio synchronization uses bounded tempo adjustment and reports when a generated line cannot fit safely instead of silently leaving it out of sync.
- Video export now runs in a background QThread, keeping the UI event loop responsive during export.
- Line regeneration is blocked while another processing/export task is running.
- The application prevents closing while worker threads are active, avoiding database teardown during background work.
- SQLite access is protected by a re-entrant lock and uses a busy timeout; line updates validate allowed fields.
- App paths no longer depend on the current working directory, and the hard-coded developer-specific VoxCPM2 paths were removed. Existing V8 settings are migrated from the legacy settings folder.
- Added regression tests for honest speaker fallback, duration probing, and unsafe synchronization.

## Important remaining work

- The AI worker still runs in a QThread inside the application process. Full process isolation, hard cancellation of active model inference/FFmpeg, and worker restart supervision are not implemented in this pass.
- Pyannote diarization requires an authorized Hugging Face token and access to the selected model. It is not an offline bundled model.
- VoxCPM2 quality, Khmer pronunciation, reference voice similarity, and GPU memory behavior require real Windows/GPU media testing.
- The current voice reference is a segment crop, not a verified clean isolated speaker recording. Voice cloning quality is not guaranteed.
- Synchronization intentionally raises an error if a line exceeds the configured safe stretch ratio; it does not yet provide a waveform timeline editor or semantic rephrasing loop.
- No 30-minute real-media test, full UI automation test, crash-injection test, or final Windows EXE runtime test has been performed here.
- Existing video preview flicker behavior has not been verified or fully replaced.

## Run tests

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
```

## Run from source

```bash
python main.py
```

## Configuration notes

- Configure a Gemini API key or a running Ollama endpoint for translation.
- Configure a Hugging Face token with permission to access `pyannote/speaker-diarization-3.1` for multi-speaker diarization.
- VoxCPM2 weights are downloaded or loaded separately; they are not included in the source ZIP or Windows build artifact.
- Whisper model weights are downloaded on first use.
- Edge TTS fallback is off by default. Set `allow_edge_tts_fallback` to `true` in the settings file only if an Internet-based fallback is intentionally accepted.
