import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

MAX_CONCURRENCY = 2
LAMA_SERVER = os.getenv("LAMA_SERVER", "http://127.0.0.1:8090")
LAMA_CONNECT_TIMEOUT = float(os.getenv("LAMA_CONNECT_TIMEOUT", "5"))
LAMA_TIMEOUT = float(os.getenv("LAMA_TIMEOUT", "120"))
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
EDITOR_SLOTS = 8
PREVIEW_HEIGHT = 520
EDITOR_HEIGHT = PREVIEW_HEIGHT + 100
EDITOR_CANVAS_SIZE = (PREVIEW_HEIGHT, PREVIEW_HEIGHT)
REMBG_MODEL_PATH = os.getenv("REMBG_MODEL_PATH", "").strip()
MODELS_DIR = os.path.abspath("./models")

OUT_DIR = os.path.abspath("./_outputs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.environ.setdefault("U2NET_HOME", MODELS_DIR)

COMPRESS_FORMAT_AUTO = "自动（保持原格式）"
COMPRESS_FORMAT_CHOICES = [COMPRESS_FORMAT_AUTO, "WEBP", "JPG", "PNG"]
EDITOR_ZOOM_CHOICES = [1, 2, 3]

# Ensure localhost bypasses any proxy to avoid Gradio url_ok failure.
def _ensure_no_proxy():
    for key in ("NO_PROXY", "no_proxy"):
        val = os.environ.get(key, "")
        parts = [p.strip() for p in val.split(",") if p.strip()]
        for host in ("127.0.0.1", "localhost"):
            if host not in parts:
                parts.append(host)
        os.environ[key] = ",".join(parts)


_ensure_no_proxy()

UI_CSS = """
.mask-editor canvas {
  cursor: crosshair;
}
#zoom-panel {
  position: fixed;
  top: 4vh;
  left: 4vw;
  width: 92vw;
  height: 92vh;
  z-index: 1000;
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  box-shadow: 0 18px 48px rgba(0, 0, 0, 0.22);
  padding: 16px;
  overflow: auto;
}
#zoom-panel .gradio-container {
  max-width: 100%;
}
"""
