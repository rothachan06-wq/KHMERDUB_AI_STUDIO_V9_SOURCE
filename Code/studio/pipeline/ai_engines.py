import os
import gc
import asyncio
import requests
from studio.pipeline.media_audio import write_wav

try:
    import edge_tts
    HAS_EDGE = True
except ImportError:
    HAS_EDGE = False

class PersistentVoxCPM2Singleton:
    """
    Loads VoxCPM2 ONCE into RTX 3060 Ti VRAM and reuses it across all 100+ dialogue lines
    of a 7-15 minute movie instead of reloading per line.
    """
    _model = None
    _loaded_path = None

    @classmethod
    def get_or_load(cls, settings):
        local_dir = settings.get("voxcpm2_local_dir", "").strip()
        repo_id = settings.get("voxcpm2_repo", "ravuthz/VoxCPM2-khmer").strip()
        source = local_dir if (local_dir and os.path.exists(local_dir)) else repo_id

        if cls._model is not None and cls._loaded_path == source:
            return cls._model

        cls.unload()
        import torch
        from voxcpm import VoxCPM
        torch.backends.cudnn.benchmark = True
        cls._model = VoxCPM.from_pretrained(source, load_denoiser=False)
        cls._loaded_path = source
        return cls._model

    @classmethod
    def unload(cls):
        cls._model = None
        cls._loaded_path = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @classmethod
    def synthesize_line(cls, text, out_wav, settings, ref_wav="", is_female=False):
        os.makedirs(os.path.dirname(os.path.abspath(out_wav)), exist_ok=True)
        sr = int(settings.get("voxcpm2_sample_rate", 24000))
        steps = int(settings.get("voxcpm2_steps", 12))

        try:
            model = cls.get_or_load(settings)
            prompt_w = ref_wav if (ref_wav and os.path.exists(ref_wav)) else None
            wav = model.generate(text=text, prompt_wav_path=prompt_w, inference_timesteps=steps)
            write_wav(out_wav, wav, sr)
            return out_wav, "VoxCPM2 GPU"
        except Exception as exc:
            if not settings.get("allow_edge_tts_fallback", False) or not HAS_EDGE:
                raise RuntimeError(
                    "VoxCPM2 synthesis failed. Edge TTS fallback is disabled by default; "
                    "check the VoxCPM2 installation/model and retry."
                ) from exc
            voice = "km-KH-SreymomNeural" if is_female else "km-KH-PisethNeural"
            async def _run():
                comm = edge_tts.Communicate(text, voice)
                await comm.save(out_wav)
            asyncio.run(_run())
            return out_wav, "Edge TTS fallback (explicitly enabled)"

class StoryTranslator:
    @staticmethod
    def translate_chunk(lines, settings):
        provider = settings.get("translation_provider", "Gemini (Context-Aware Story Mode)")
        if "Gemini" in provider:
            return StoryTranslator._translate_gemini(lines, settings)
        if "Ollama" in provider:
            return StoryTranslator._translate_ollama(lines, settings)
        return StoryTranslator._translate_hf(lines, settings)

    @staticmethod
    def _translate_gemini(lines, settings):
        api_key = settings.get("gemini_api_key", "").strip()
        model = settings.get("gemini_model", "gemini-3.8-flash").strip()
        if not api_key:
            raise ValueError("Please enter your Gemini API Key in Advanced Settings (or switch to Local Ollama / HuggingFace).")

        script_block = "\n".join([f"[{ln['idx']}] ({ln['speaker_id']}): {ln['source_text']}" for ln in lines])
        sys_prompt = (
            "You are an expert Khmer movie dubbing scriptwriter. Translate the dialogue into natural, emotional spoken Cambodian Khmer (អក្សរខ្មែរ).\n"
            "Rules:\n"
            "1. Output ONLY pure Cambodian Khmer script. NEVER output Thai characters.\n"
            "2. Preserve character relationships, emotion, and concise spoken timing.\n"
            "3. Return strictly formatted lines: [index] translated_khmer_text"
        )
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": sys_prompt}]},
            "contents": [{"parts": [{"text": script_block}]}],
            "generationConfig": {"temperature": 0.25}
        }
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        mapping = {}
        for row in raw.splitlines():
            row = row.strip()
            if row.startswith("[") and "]" in row:
                id_s, kh = row[1:].split("]", 1)
                if id_s.strip().isdigit():
                    mapping[int(id_s.strip())] = kh.strip()
        return {ln["idx"]: mapping.get(ln["idx"], ln["source_text"]) for ln in lines}

    @staticmethod
    def _translate_ollama(lines, settings):
        base = settings.get("ollama_url", "http://localhost:11434/v1").rstrip("/")
        model = settings.get("ollama_model", "qwen2.5")
        script_block = "\n".join([f"[{ln['idx']}] {ln['source_text']}" for ln in lines])
        r = requests.post(f"{base}/chat/completions", json={
            "model": model,
            "messages": [
                {"role": "system", "content": "Translate movie subtitles into natural spoken Khmer (អក្សរខ្មែរ). Return strictly: [index] khmer_text"},
                {"role": "user", "content": script_block}
            ],
            "temperature": 0.25
        }, timeout=90)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]
        mapping = {}
        for row in raw.splitlines():
            row = row.strip()
            if row.startswith("[") and "]" in row:
                id_s, kh = row[1:].split("]", 1)
                if id_s.strip().isdigit():
                    mapping[int(id_s.strip())] = kh.strip()
        return {ln["idx"]: mapping.get(ln["idx"], ln["source_text"]) for ln in lines}

    @staticmethod
    def _translate_hf(lines, settings):
        token = settings.get("hf_token", "").strip()
        model = settings.get("hf_model", "facebook/nllb-200-distilled-600M").strip()
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        out = {}
        for ln in lines:
            r = requests.post(
                f"https://api-inference.huggingface.co/models/{model}",
                headers=headers,
                json={"inputs": ln["source_text"], "parameters": {"src_lang": "eng_Latn", "tgt_lang": "khm_Khmr"}},
                timeout=45
            )
            if r.status_code != 200:
                raise RuntimeError(f"HuggingFace Inference Notice ({r.status_code}): Check HF Token or use Gemini / Ollama in Advanced Settings.")
            data = r.json()
            out[ln["idx"]] = data[0]["translation_text"].strip() if isinstance(data, list) and data else ln["source_text"]
        return out
