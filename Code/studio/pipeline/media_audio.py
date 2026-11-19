import os
import wave
import json
import shutil
import subprocess
import numpy as np
from studio.core.config import APP_ROOT

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

def find_tool(name):
    exe = f"{name}.exe" if os.name == "nt" else name
    local = os.path.join(APP_ROOT, "bin", exe)
    if os.path.exists(local):
        return local
    found = shutil.which(name)
    return found if found else name

def read_wav(path):
    if HAS_SF:
        return sf.read(path, dtype="float32")
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        arr = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        if ch > 1:
            arr = arr.reshape(-1, ch)
        return arr, sr

def write_wav(path, data, sr):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if HAS_SF:
        sf.write(path, data, sr)
        return
    arr = np.clip(data, -1.0, 1.0)
    pcm = (arr * 32767.0).astype(np.int16)
    ch = 1 if pcm.ndim == 1 else pcm.shape[1]
    with wave.open(path, "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

def probe_video_duration(video_path):
    """Return the actual media duration; never substitute a fabricated duration."""
    ffprobe = find_tool("ffprobe")
    cmd = [ffprobe, "-v", "error", "-print_format", "json", "-show_format", video_path]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE,
                                      creationflags=CREATE_NO_WINDOW)
        duration = float(json.loads(out)["format"]["duration"])
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Media duration is missing or invalid.")
        return duration
    except Exception as exc:
        raise RuntimeError(f"Unable to read the real duration of this media file: {video_path}") from exc

def extract_audio_proxies(video_path, work_dir):
    """Extracts lightweight 16kHz mono proxy for instant Whisper & 48kHz stereo for clean character voice-matching clips."""
    os.makedirs(work_dir, exist_ok=True)
    ffmpeg = find_tool("ffmpeg")
    proxy_16k = os.path.join(work_dir, "audio_proxy_16k.wav")
    clean_48k = os.path.join(work_dir, "audio_master_48k.wav")

    subprocess.run([ffmpeg, "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", proxy_16k],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
    subprocess.run([ffmpeg, "-y", "-i", video_path, "-vn", "-ac", "2", "-ar", "48000", clean_48k],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
    return proxy_16k, clean_48k

def detect_speakers_and_extract_voice_refs(clean_wav_48k, lines, refs_dir, settings=None):
    """Use pyannote diarization when configured; otherwise mark one honest unknown speaker.

    No pitch/ZCR-to-gender guessing is performed. Real multi-speaker diarization requires
    pyannote.audio and a Hugging Face token with access to the selected model.
    """
    settings = settings or {}
    os.makedirs(refs_dir, exist_ok=True)
    if not os.path.exists(clean_wav_48k) or not lines:
        return lines, {}

    diarization = None
    token = str(settings.get("hf_token", "")).strip()
    model_id = str(settings.get("diarization_model", "pyannote/speaker-diarization-3.1")).strip()
    if token:
        try:
            from pyannote.audio import Pipeline
            pipeline = Pipeline.from_pretrained(model_id, use_auth_token=token)
            diarization = pipeline(clean_wav_48k)
        except Exception as exc:
            raise RuntimeError(
                "Speaker diarization could not be initialized. Check pyannote.audio, "
                "the Hugging Face token, and model access. No guessed speaker labels were applied."
            ) from exc

    assignments = {}
    if diarization is not None:
        for turn, _, label in diarization.itertracks(yield_label=True):
            assignments.setdefault(str(label), []).append((float(turn.start), float(turn.end)))

    for ln in lines:
        best_label, best_overlap = None, 0.0
        for label, turns in assignments.items():
            overlap = sum(max(0.0, min(float(ln["end_sec"]), end) - max(float(ln["start_sec"]), start))
                          for start, end in turns)
            if overlap > best_overlap:
                best_label, best_overlap = label, overlap
        ln["speaker_id"] = f"Speaker {best_label}" if best_label and best_overlap > 0 else "Speaker 01"

    data, sr = read_wav(clean_wav_48k)
    best_clip_for_spk = {}
    for ln in lines:
        dur = float(ln["end_sec"]) - float(ln["start_sec"])
        if 2.5 <= dur <= 12.0:
            prev = best_clip_for_spk.get(ln["speaker_id"], (0.0, None))[0]
            if dur > prev:
                best_clip_for_spk[ln["speaker_id"]] = (dur, ln)

    speaker_refs = {}
    for index, (spk, (_, ln)) in enumerate(best_clip_for_spk.items(), start=1):
        safe_name = f"speaker_{index:02d}_ref.wav"
        ref_out = os.path.join(refs_dir, safe_name)
        s_idx = max(0, int(float(ln["start_sec"]) * sr))
        e_idx = min(len(data), int(float(ln["end_sec"]) * sr))
        if e_idx > s_idx:
            write_wav(ref_out, data[s_idx:e_idx], sr)
            speaker_refs[spk] = ref_out
    return lines, speaker_refs

def auto_sync_segment_wav(wav_path, target_duration_sec, max_stretch=1.8):
    """Fit speech into its subtitle slot with bounded tempo change; fail if impossible."""
    if not os.path.exists(wav_path):
        raise FileNotFoundError(wav_path)
    if target_duration_sec <= 0.35:
        raise ValueError("Dialogue slot is too short for safe synchronization.")
    data, sr = read_wav(wav_path)
    actual_dur = len(data) / float(sr)
    if actual_dur <= target_duration_sec:
        return actual_dur
    ratio = actual_dur / float(target_duration_sec)
    if ratio > float(max_stretch):
        raise RuntimeError(
            f"Generated speech is {actual_dur:.2f}s but its dialogue slot is "
            f"{target_duration_sec:.2f}s. Required speed-up {ratio:.2f}x exceeds "
            f"the configured safe limit {float(max_stretch):.2f}x. Shorten the translation or regenerate."
        )
    factors = []
    remaining = ratio
    while remaining > 2.0:
        factors.append("atempo=2.0")
        remaining /= 2.0
    factors.append(f"atempo={remaining:.5f}")
    ffmpeg = find_tool("ffmpeg")
    tmp = wav_path + ".sync.wav"
    try:
        subprocess.run([ffmpeg, "-y", "-i", wav_path, "-filter:a", ",".join(factors), tmp],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                       creationflags=CREATE_NO_WINDOW)
        os.replace(tmp, wav_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return len(read_wav(wav_path)[0]) / float(sr)

def render_final_dubbed_video(video_path, lines, total_dur_sec, out_video_path, bg_vol=0.18, burn_subs=True):
    """Assembles sample-accurate Khmer dub track, ducks original audio, and exports final MP4/MKV."""
    work_dir = os.path.dirname(os.path.abspath(out_video_path))
    os.makedirs(work_dir, exist_ok=True)
    sr = 48000
    total_samples = max(int(total_dur_sec * sr), sr * 2)
    master_dub = np.zeros((total_samples, 2), dtype=np.float32)

    ffmpeg = find_tool("ffmpeg")
    for ln in lines:
        wpath = ln.get("dub_wav_path", "")
        if not wpath or not os.path.exists(wpath):
            continue
        data, w_sr = read_wav(wpath)
        if w_sr != sr:
            rs = wpath + ".48k.wav"
            subprocess.run([ffmpeg, "-y", "-i", wpath, "-ar", str(sr), "-ac", "2", rs],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
            data, _ = read_wav(rs)
            os.remove(rs)
        if data.ndim == 1:
            data = np.stack([data, data], axis=1)
        st = int(float(ln["start_sec"]) * sr)
        en = min(total_samples, st + len(data))
        if en > st:
            master_dub[st:en] += data[:en - st]

    peak = float(np.max(np.abs(master_dub)))
    if peak > 0.98:
        master_dub = master_dub * (0.95 / peak)

    master_wav = os.path.join(work_dir, "_master_khmer_dub.wav")
    write_wav(master_wav, master_dub, sr)

    srt_path = os.path.join(work_dir, "_final_khmer.srt")
    export_srt_file(srt_path, lines, use_khmer=True)

    cmd = [
        ffmpeg, "-y", "-i", video_path, "-i", master_wav,
        "-filter_complex", f"[0:a]volume={bg_vol}[bg];[1:a]volume=1.0[dub];[bg][dub]amix=inputs=2:duration=first[aout]",
        "-map", "0:v", "-map", "[aout]"
    ]
    if burn_subs and os.path.exists(srt_path):
        safe_srt = srt_path.replace("\\", "/").replace(":", "\\:")
        cmd.extend(["-vf", f"subtitles='{safe_srt}':force_style='FontName=Khmer OS Siemreap,FontSize=20'"])
        cmd.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "19"])
    else:
        cmd.extend(["-c:v", "copy"])

    cmd.extend(["-c:a", "aac", "-b:a", "192k", out_video_path])
    subprocess.run(cmd, check=True, creationflags=CREATE_NO_WINDOW)
    return out_video_path, srt_path

def sec_to_tc(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def export_srt_file(filepath, lines, use_khmer=True):
    blocks = []
    for i, ln in enumerate(lines, start=1):
        txt = (ln.get("khmer_text") if use_khmer else ln.get("source_text")) or ln.get("source_text", "")
        blocks.append(f"{i}\n{sec_to_tc(ln['start_sec'])} --> {sec_to_tc(ln['end_sec'])}\n{txt}\n")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks))
