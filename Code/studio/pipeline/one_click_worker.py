import os
from PySide6.QtCore import QThread, Signal
from studio.pipeline.media_audio import (
    probe_video_duration, extract_audio_proxies,
    detect_speakers_and_extract_voice_refs, auto_sync_segment_wav,
    render_final_dubbed_video
)
from studio.pipeline.ai_engines import PersistentVoxCPM2Singleton, StoryTranslator

class OneClickDubbingWorker(QThread):
    stage_changed = Signal(int, str)          # percent, stage_label
    line_updated = Signal(int)                # line idx updated in SQLite
    pipeline_completed = Signal(str, str)     # final_mp4_path, final_srt_path
    pipeline_failed = Signal(str)             # friendly_error_message

    def __init__(self, video_path, store, settings, single_line_idx=None):
        super().__init__()
        self.video_path = video_path
        self.store = store
        self.settings = settings
        self.single_line_idx = single_line_idx
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        try:
            if self.single_line_idx is not None:
                self._regenerate_single_line(self.single_line_idx)
                return

            work_dir = self.store.project_dir
            dub_dir = os.path.join(work_dir, "dub_clips")
            refs_dir = os.path.join(work_dir, "voice_refs")
            os.makedirs(dub_dir, exist_ok=True)

            # STAGE 1: Audio Extraction (Audio-First Proxy)
            self.stage_changed.emit(5, "Stage 1/6: Extracting lightweight audio proxies...")
            proxy_16k, clean_48k = extract_audio_proxies(self.video_path, work_dir)
            total_dur = probe_video_duration(self.video_path)

            # STAGE 2: Speech Recognition (Whisper CUDA) — Skipped if already saved in SQLite (Resume Support!)
            existing_lines = self.store.get_lines()
            if not existing_lines:
                self.stage_changed.emit(15, "Stage 2/6: Recognizing speech & timestamps (Whisper AI)...")
                from faster_whisper import WhisperModel
                m_name = self.settings.get("whisper_model", "medium")
                try:
                    whisper = WhisperModel(m_name, device="cuda", compute_type="float16")
                except Exception:
                    whisper = WhisperModel("base", device="cpu", compute_type="int8")

                segs, _ = whisper.transcribe(proxy_16k, vad_filter=True, beam_size=5)
                rows = []
                for i, s in enumerate(segs, start=1):
                    if self._stop_requested:
                        return
                    rows.append({
                        "idx": i,
                        "start_sec": round(s.start, 3),
                        "end_sec": round(s.end, 3),
                        "speaker_id": "Speaker 01 (Male/Lead)",
                        "source_text": s.text.strip(),
                        "khmer_text": "",
                        "dub_wav_path": "",
                        "stage_done": 1
                    })
                if not rows:
                    raise RuntimeError("No speech dialogue was detected in this video.")

                # STAGE 3: Speaker Detection & Voice-Matching Reference Extraction
                self.stage_changed.emit(32, "Stage 3/6: Detecting speakers & extracting voice-matching clips...")
                rows, spk_refs = detect_speakers_and_extract_voice_refs(clean_48k, rows, refs_dir, self.settings)
                for spk_id, ref_p in spk_refs.items():
                    self.store.upsert_speaker(spk_id, ref_clip_path=ref_p)
                self.store.replace_all_lines(rows)
                self.line_updated.emit(-1)

            # STAGE 4: Context-Aware Khmer Translation (Chunked + Resumable)
            lines = self.store.get_lines()
            untranslated = [ln for ln in lines if not ln["khmer_text"].strip()]
            if untranslated:
                self.stage_changed.emit(45, f"Stage 4/6: Translating {len(untranslated)} lines into natural Khmer...")
                chunk_size = 8
                for c_start in range(0, len(untranslated), chunk_size):
                    if self._stop_requested:
                        return
                    chunk = untranslated[c_start:c_start + chunk_size]
                    kh_map = StoryTranslator.translate_chunk(chunk, self.settings)
                    for ln in chunk:
                        kh_txt = kh_map.get(ln["idx"], ln["source_text"])
                        self.store.update_line(ln["idx"], khmer_text=kh_txt, stage_done=2)
                        self.line_updated.emit(ln["idx"])
                    pct = 45 + int(((c_start + len(chunk)) / len(untranslated)) * 18)
                    self.stage_changed.emit(pct, f"Stage 4/6: Translated {min(len(untranslated), c_start + chunk_size)}/{len(untranslated)} lines...")

            # STAGE 5: Khmer Voice Generation (VoxCPM2 Loaded ONCE in VRAM) + Auto Sync
            lines = self.store.get_lines()
            speakers = self.store.get_speakers()
            undubbed = [ln for ln in lines if not ln["dub_wav_path"] or not os.path.exists(ln["dub_wav_path"])]
            if undubbed:
                self.stage_changed.emit(65, "Stage 5/6: Generating Khmer voices (VoxCPM2) & Auto-Syncing...")
                for idx_i, ln in enumerate(undubbed):
                    if self._stop_requested:
                        return
                    spk_info = speakers.get(ln["speaker_id"], {})
                    ref_wav = spk_info.get("ref_clip_path", "")
                    is_fem = "Female" in ln["speaker_id"] or "02" in ln["speaker_id"]
                    out_wav = os.path.join(dub_dir, f"line_{ln['idx']:04d}.wav")

                    PersistentVoxCPM2Singleton.synthesize_line(
                        ln["khmer_text"] or ln["source_text"],
                        out_wav,
                        self.settings,
                        ref_wav=ref_wav,
                        is_female=is_fem
                    )
                    target_dur = max(0.4, float(ln["end_sec"]) - float(ln["start_sec"]))
                    auto_sync_segment_wav(out_wav, target_dur, max_stretch=float(self.settings.get("max_time_stretch", 1.25)))

                    self.store.update_line(ln["idx"], dub_wav_path=out_wav, stage_done=3)
                    self.line_updated.emit(ln["idx"])
                    pct = 65 + int(((idx_i + 1) / len(undubbed)) * 25)
                    self.stage_changed.emit(pct, f"Stage 5/6: Dubbed & synced {idx_i + 1}/{len(undubbed)} lines...")

            # STAGE 6: Final Video Rendering
            self.stage_changed.emit(92, "Stage 6/6: Rendering final dubbed MP4 video & subtitles...")
            base_name = os.path.splitext(os.path.basename(self.video_path))[0]
            out_mp4 = os.path.join(work_dir, f"{base_name}_KHMER_DUBBED.mp4")
            final_lines = self.store.get_lines()
            final_mp4, final_srt = render_final_dubbed_video(
                self.video_path,
                final_lines,
                total_dur,
                out_mp4,
                bg_vol=float(self.settings.get("bg_volume", 0.18)),
                burn_subs=bool(self.settings.get("burn_subtitles", True))
            )
            self.stage_changed.emit(100, "Completed! Video is ready to play and export.")
            self.pipeline_completed.emit(final_mp4, final_srt)

        except Exception as e:
            self.pipeline_failed.emit(self._humanize_error(e))

    def _regenerate_single_line(self, line_idx):
        ln = next((x for x in self.store.get_lines() if x["idx"] == line_idx), None)
        if not ln:
            return
        self.stage_changed.emit(50, f"Re-generating line #{line_idx}...")
        speakers = self.store.get_speakers()
        ref_wav = speakers.get(ln["speaker_id"], {}).get("ref_clip_path", "")
        is_fem = "Female" in ln["speaker_id"] or "02" in ln["speaker_id"]
        dub_dir = os.path.join(self.store.project_dir, "dub_clips")
        out_wav = os.path.join(dub_dir, f"line_{ln['idx']:04d}.wav")

        PersistentVoxCPM2Singleton.synthesize_line(
            ln["khmer_text"] or ln["source_text"], out_wav, self.settings, ref_wav=ref_wav, is_female=is_fem
        )
        target_dur = max(0.4, float(ln["end_sec"]) - float(ln["start_sec"]))
        auto_sync_segment_wav(out_wav, target_dur, max_stretch=float(self.settings.get("max_time_stretch", 1.25)))
        self.store.update_line(line_idx, dub_wav_path=out_wav, stage_done=3)
        self.line_updated.emit(line_idx)
        self.stage_changed.emit(100, f"Line #{line_idx} updated!")

    @staticmethod
    def _humanize_error(err):
        msg = str(err)
        if "cublas64_12.dll" in msg or "cudnn" in msg:
            return (
                "GPU CUDA 12 Library Missing (cublas64_12.dll).\n"
                "Please ensure NVIDIA CUDA DLLs are inside the application folder or switch Whisper to CPU in Advanced Settings."
            )
        if "Gemini API Key" in msg or "401" in msg or "403" in msg:
            return "Translation API Key is missing or invalid. Please open 'Advanced Settings' and enter a valid Gemini API Key (or switch to Local Ollama)."
        if "ffmpeg" in msg.lower():
            return "FFmpeg audio/video engine encountered an error. Please check that the video file is not corrupted or open in another player."
        return f"Processing Notice: {msg}"
