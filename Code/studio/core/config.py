import json
import os
import sys

def get_app_root():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

APP_ROOT = get_app_root()
if os.name == "nt":
    USER_DATA_DIR = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "KHMERDUB AI STUDIO")
else:
    USER_DATA_DIR = os.path.join(os.environ.get("XDG_CONFIG_HOME", os.path.join(os.path.expanduser("~"), ".config")), "khmerdub-ai-studio")
os.makedirs(USER_DATA_DIR, exist_ok=True)
CONFIG_PATH = os.path.join(USER_DATA_DIR, "settings.json")
LEGACY_CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".khmerdub_pro_v8", "settings.json")

DEFAULT_SETTINGS = {
    "translation_provider": "Gemini (Context-Aware Story Mode)",
    "gemini_api_key": "",
    "gemini_model": "gemini-3.8-flash",
    "hf_token": "",
    "hf_model": "facebook/nllb-200-distilled-600M",
    "diarization_model": "pyannote/speaker-diarization-3.1",
    "ollama_url": "http://localhost:11434/v1",
    "ollama_model": "qwen2.5",
    "whisper_model": "medium",
    "voxcpm2_repo": "ravuthz/VoxCPM2-khmer",
    "voxcpm2_local_dir": "",
    "voxcpm2_external_python": "",
    "allow_edge_tts_fallback": False,
    "voxcpm2_sample_rate": 24000,
    "voxcpm2_steps": 12,
    "bg_volume": 0.18,
    "burn_subtitles": True,
    "max_time_stretch": 1.8,
    "last_project_dir": ""
}

class SettingsManager:
    def __init__(self, path=CONFIG_PATH):
        self.path = path
        self.data = DEFAULT_SETTINGS.copy()
        self.load()

    def load(self):
        source = self.path if os.path.exists(self.path) else LEGACY_CONFIG_PATH
        if os.path.exists(source):
            try:
                with open(source, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        self.data.update(loaded)
            except Exception:
                pass

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value
        self.save()
