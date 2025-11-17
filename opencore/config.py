"""Central configuration and path constants for the OpenCore stack."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = os.getenv("TEACHABLE_MODEL_PATH", "./models/teachable_model")
DEFAULT_CLASS_NAMES: List[str] = ["class_1", "class_2", "class_3"]
GPT_MODEL = os.getenv("OPENAI_GPT_MODEL", "gpt-4.1-mini")
SIMPLE_UI_PATH = BASE_DIR / "static" / "index.html"
CONFIG_UI_PATH = BASE_DIR / "static" / "config.html"
COMPLETIONS_UI_PATH = BASE_DIR / "static" / "completions.html"
SHARE_UI_PATH = BASE_DIR / "static" / "share.html"
TM_MODELS_DIR = BASE_DIR / "TM-models"
TM_MODELS_DIR.mkdir(parents=True, exist_ok=True)
TM_REGISTRY_PATH = TM_MODELS_DIR / "registry.json"
TFJS_REQUIRED_FILES = {"metadata.json", "model.json", "weights.bin"}
TFJS_REQUIRED_FILES_LOWER = {name.lower() for name in TFJS_REQUIRED_FILES}
KERAS_LABEL_FILE = "labels.txt"
KERAS_SUFFIXES = {".keras", ".h5", ".hdf5"}
NETWORK_CONFIG_FILENAME = "network-config.json"
SETTINGS_PATH = BASE_DIR / "app-settings.json"
DOCS_DIR = BASE_DIR / "doc"
SHARE_STORE_DIR = BASE_DIR / "share-store"
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOAD_REGISTRY_PATH = UPLOADS_DIR / "registry.json"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
SHARE_STORE_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_TYPE_ALIASES: Dict[str, str] = {
    "trichome": "trichome",
    "trichomen": "trichome",
    "trichomen analyse": "trichome",
    "trichome analysis": "trichome",
    "trichomes": "trichome",
    "health": "health",
    "healthcare": "health",
    "gesundheit": "health",
}

NETWORK_DEFAULT_CONFIG = {
    "enabled": False,
    "hostname": "ottcolab.local",
    "port": 8000,
    "ip": None,
    "url": None,
}
STREAM_CAPTURE_INTERVAL_DEFAULT = 5.0
STREAM_BATCH_INTERVAL_DEFAULT = 30.0
STREAM_BUFFER_MAX = 24
DEFAULT_LLM_CONFIG = {
    "provider": "openai",
    "apiBase": "",
    "model": "",
    "apiKey": "",
    "vision": "yes",
    "systemPrompt": "",
}
LLM_ALLOWED_PROVIDERS = {"openai", "ollama", "lmstudio"}
OPENAI_BASE_URL = "https://api.openai.com/v1"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
LLM_HTTP_TIMEOUT = int(os.getenv("OTTC_LLM_TIMEOUT", "90"))
ANALYZER_SYSTEM_PROMPT = (
    "You analyze cannabis images for ottcouture.eu. Combine Teachable Machine "
    "classifications with the user prompt, describe risks, and keep language clean and non-medical."
)
LOG_LEVEL = os.getenv("OTTC_LOG_LEVEL", "INFO").upper()
UPSTREAM_REPO_URL = os.getenv("OTTC_REPO_URL", "https://github.com/ottco-dev/ottcops.git")
UPSTREAM_REPO_BRANCH = os.getenv("OTTC_REPO_BRANCH", "main")

MQTT_DEFAULT_CONFIG: Dict[str, object] = {
    "broker": "",
    "port": 1883,
    "username": "",
    "password": "",
    "use_tls": False,
    "sensors": [],
}
MQTT_SENSOR_KINDS = {"co2", "ppfd", "lux", "humidity", "temperature", "ec", "ph"}

# Default FAQ blocks for the prompt builder. These can be overridden via the config UI.
DEFAULT_FAQ_BLOCKS = [
    {
        "id": "faq_full_health",
        "title": "Full plant health",
        "blurb": "Overall vigor, leaf color, deficiencies, stress.",
        "prompt": "Provide a concise health assessment: vigor, leaf color, nutrient deficiency signs, stress markers, and any remedial actions.",
    },
    {
        "id": "faq_environment",
        "title": "Environment & sensors",
        "blurb": "Blend sensor data with visible cues.",
        "prompt": "Use the provided environment readings inside the prompt (humidity, temperature, CO2/PPFD, EC/PH) to contextualize what you see in the image. Flag mismatches between visuals and sensor values.",
    },
    {
        "id": "faq_pests",
        "title": "Pest sweep",
        "blurb": "Check for pests and typical damage patterns.",
        "prompt": "Scan for pests or pest damage. Name likely culprits and list next steps to contain them.",
    },
    {
        "id": "faq_trichomes",
        "title": "Trichome timing",
        "blurb": "Harvest window via trichome mix.",
        "prompt": "Describe trichome mix (clear/milky/amber) and estimate a harvest window.",
    },
    {
        "id": "faq_cleanliness",
        "title": "Cleanliness check",
        "blurb": "Mold risk, bud handling, wash suggestions.",
        "prompt": "Check for mold/rot risk, handling artifacts, and recommend any bud wash or drying adjustments.",
    },
    {
        "id": "faq_vpd",
        "title": "VPD & climate",
        "blurb": "Target humidity/temperature and VPD alignment.",
        "prompt": "Review vapor pressure deficit alignment using the provided humidity and temperature. Flag if VPD is out of range for the current growth stage and suggest humidity or temperature tweaks to get back on track.",
    },
    {
        "id": "faq_ripeness",
        "title": "Ripeness check",
        "blurb": "Flower maturity and fade cues.",
        "prompt": "Assess flower ripeness and fade cues beyond trichomes: pistil color, leaf fade patterns, bud density, and any late-stage stress. Provide a harvest readiness note.",
    },
    {
        "id": "faq_training",
        "title": "Canopy & training",
        "blurb": "Node spacing, airflow, light spread.",
        "prompt": "Evaluate canopy shape, node spacing, airflow lanes, and light distribution. Recommend low-stress training, defoliation, or trellising actions to optimize coverage without stressing the plant.",
    },
    {
        "id": "faq_feeding",
        "title": "Feeding & EC/PH",
        "blurb": "Nutrient strength and pH context.",
        "prompt": "Use the EC and pH readings (if provided) together with visible leaf cues to decide if feeding strength is appropriate. Flag over/underfeeding signs and suggest small EC or pH adjustments.",
    },
]
