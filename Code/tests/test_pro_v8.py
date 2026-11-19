import os
import tempfile
import numpy as np
from studio.core.config import SettingsManager
from studio.core.project_db import ProjectStore
from studio.pipeline.media_audio import (
    write_wav, read_wav, detect_speakers_and_extract_voice_refs,
    export_srt_file
)

def test_sqlite_checkpoint_resume():
    with tempfile.TemporaryDirectory() as tmp:
        db = ProjectStore(tmp)
        db.set_meta("video_path", "sample.mp4")
        db.replace_all_lines([
            {"idx": 1, "start_sec": 0.5, "end_sec": 3.0, "speaker_id": "Speaker 01", "source_text": "Hello", "khmer_text": "សួស្តី", "stage_done": 2}
        ])
        db.close()

        # Simulate reopening after crash/restart
        db2 = ProjectStore(tmp)
        assert db2.get_meta("video_path") == "sample.mp4"
        lines = db2.get_lines()
        assert len(lines) == 1 and lines[0]["khmer_text"] == "សួស្តី"
        db2.close()

def test_speaker_detection_and_voice_ref_clipping():
    with tempfile.TemporaryDirectory() as tmp:
        sr = 48000
        t = np.linspace(0, 8.0, sr * 8, endpoint=False)
        # Different acoustic frequencies must not be misrepresented as real identities.
        sig = np.zeros_like(t, dtype=np.float32)
        sig[:sr * 4] = 0.3 * np.sin(2 * np.pi * 120 * t[:sr * 4])
        sig[sr * 4:] = 0.3 * np.sin(2 * np.pi * 340 * t[sr * 4:])
        wav_p = os.path.join(tmp, "master_48k.wav")
        write_wav(wav_p, sig, sr)

        lines = [
            {"idx": 1, "start_sec": 0.2, "end_sec": 3.8, "speaker_id": "", "source_text": "Line 1"},
            {"idx": 2, "start_sec": 4.2, "end_sec": 7.8, "speaker_id": "", "source_text": "Line 2"},
        ]
        updated, refs = detect_speakers_and_extract_voice_refs(wav_p, lines, os.path.join(tmp, "refs"))
        assert updated[0]["speaker_id"] == "Speaker 01"
        assert updated[1]["speaker_id"] == "Speaker 01"
        assert len(refs) == 1
        for r_path in refs.values():
            assert os.path.exists(r_path)

def test_srt_export():
    with tempfile.TemporaryDirectory() as tmp:
        out_srt = os.path.join(tmp, "out.srt")
        export_srt_file(out_srt, [{"start_sec": 1.0, "end_sec": 3.0, "source_text": "Hi", "khmer_text": "សួស្តី"}])
        with open(out_srt, "r", encoding="utf-8") as f:
            content = f.read()
        assert "សួស្តី" in content and "00:00:01,000 --> 00:00:03,000" in content


def test_speaker_diarization_requires_real_model_for_multiple_speakers():
    with tempfile.TemporaryDirectory() as tmp:
        sr = 48000
        wav_p = os.path.join(tmp, "master.wav")
        write_wav(wav_p, np.zeros(sr * 4, dtype=np.float32), sr)
        lines = [{"idx": 1, "start_sec": 0.1, "end_sec": 3.5, "speaker_id": "", "source_text": "Test"}]
        updated, refs = detect_speakers_and_extract_voice_refs(wav_p, lines, os.path.join(tmp, "refs"), {})
        assert updated[0]["speaker_id"] == "Speaker 01"
        assert len(refs) == 1


def test_sync_refuses_to_silently_leave_unfit_speech():
    from studio.pipeline.media_audio import auto_sync_segment_wav
    import pytest
    with tempfile.TemporaryDirectory() as tmp:
        wav_p = os.path.join(tmp, "long.wav")
        write_wav(wav_p, np.zeros(48000 * 5, dtype=np.float32), 48000)
        with pytest.raises(RuntimeError, match="exceeds"):
            auto_sync_segment_wav(wav_p, 1.0, max_stretch=1.8)

def test_duration_probe_does_not_fabricate_ten_seconds():
    from studio.pipeline.media_audio import probe_video_duration
    import pytest
    with pytest.raises(RuntimeError, match="real duration"):
        probe_video_duration(os.path.join(tempfile.gettempdir(), "definitely_missing_khmerdub_media.mp4"))
