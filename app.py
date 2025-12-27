import os
import io
import zipfile
import uuid
import time
import functools
from typing import List, Tuple, Optional, Any, Dict

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import httpx
import gradio as gr
from PIL import Image, ImageOps
from rembg import remove as rembg_remove
from rembg.session_factory import new_session
from rembg.sessions import sessions_names, sessions_class

MAX_CONCURRENCY = 2
LAMA_SERVER = os.getenv("LAMA_SERVER", "http://127.0.0.1:8090")
LAMA_CONNECT_TIMEOUT = float(os.getenv("LAMA_CONNECT_TIMEOUT", "5"))
LAMA_TIMEOUT = float(os.getenv("LAMA_TIMEOUT", "120"))
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
EDITOR_HEIGHT = 520
REMBG_MODEL_PATH = os.getenv("REMBG_MODEL_PATH", "").strip()
MODELS_DIR = os.path.abspath("./models")

OUT_DIR = os.path.abspath("./_outputs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.environ.setdefault("U2NET_HOME", MODELS_DIR)

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
.mask-editor .image-container {
  justify-content: flex-start;
  align-items: flex-start;
}
.mask-editor .tools-wrap {
  margin-top: var(--spacing-lg);
}
"""

REMBG_MODEL_AUTO = "auto (REMBG_MODEL_PATH / u2net)"
REMBG_MODEL_CUSTOM = "custom (REMBG_MODEL_PATH)"
REMBG_EXCLUDE_MODELS = {"u2net_custom", "u2net_cloth_seg", "sam"}
REMBG_MODELS = sorted([m for m in sessions_names if m not in REMBG_EXCLUDE_MODELS])
REMBG_MODEL_CHOICES = [REMBG_MODEL_AUTO]
if REMBG_MODEL_PATH:
    REMBG_MODEL_CHOICES.append(REMBG_MODEL_CUSTOM)
REMBG_MODEL_CHOICES.extend(REMBG_MODELS)
REMBG_MODEL_DEFAULT = REMBG_MODEL_AUTO
COMPRESS_FORMAT_AUTO = "自动（保持原格式）"
COMPRESS_FORMAT_CHOICES = [COMPRESS_FORMAT_AUTO, "WEBP", "JPG", "PNG"]

_REMBG_SESSIONS: Dict[str, Any] = {}
_REMBG_SESSION_ERRS: Dict[str, str] = {}
_REMBG_MODEL_CLASSES = {cls.name(): cls for cls in sessions_class}


def _rembg_session_key(model_name: str, model_path: Optional[str]) -> str:
    if model_name == "u2net_custom":
        return f"{model_name}:{model_path}"
    return model_name


def _resolve_rembg_choice(model_choice: Optional[str]) -> Tuple[str, Optional[str]]:
    if not model_choice or model_choice == REMBG_MODEL_AUTO:
        if REMBG_MODEL_PATH:
            return "u2net_custom", REMBG_MODEL_PATH
        return "u2net", None
    if model_choice == REMBG_MODEL_CUSTOM:
        if not REMBG_MODEL_PATH:
            raise RuntimeError("REMBG_MODEL_PATH is not set.")
        return "u2net_custom", REMBG_MODEL_PATH
    return model_choice, None


def _get_rembg_session(model_choice: Optional[str]):
    model_name, model_path = _resolve_rembg_choice(model_choice)
    key = _rembg_session_key(model_name, model_path)
    if key in _REMBG_SESSION_ERRS:
        raise RuntimeError(_REMBG_SESSION_ERRS[key])
    if key in _REMBG_SESSIONS:
        return _REMBG_SESSIONS[key]
    if model_name != "u2net_custom" and model_name not in _REMBG_MODEL_CLASSES:
        raise RuntimeError(f"Unknown model: {model_name}")
    try:
        if model_name == "u2net_custom":
            if not model_path:
                raise RuntimeError("REMBG_MODEL_PATH is not set.")
            if not os.path.isfile(model_path):
                raise RuntimeError(f"REMBG_MODEL_PATH not found: {model_path}")
            session = new_session("u2net_custom", model_path=model_path)
        else:
            session = new_session(model_name)
        _REMBG_SESSIONS[key] = session
        return session
    except Exception as e:
        _REMBG_SESSION_ERRS[key] = str(e)
        raise


def _model_local_path(model_name: str) -> str:
    return os.path.join(MODELS_DIR, f"{model_name}.onnx")


def _list_rembg_models_status() -> Tuple[List[str], List[str]]:
    downloaded = []
    missing = []
    for name in REMBG_MODELS:
        if os.path.isfile(_model_local_path(name)):
            downloaded.append(name)
        else:
            missing.append(name)
    return downloaded, missing


def _format_model_status(model_choice: Optional[str], header: Optional[str] = None) -> str:
    lines = []
    if header:
        lines.append(header)
    lines.append(f"模型目录: {MODELS_DIR}")
    if REMBG_MODEL_PATH:
        exists = "已存在" if os.path.isfile(REMBG_MODEL_PATH) else "不存在"
        lines.append(f"自定义模型: {REMBG_MODEL_PATH} ({exists})")

    choice = model_choice or REMBG_MODEL_DEFAULT
    try:
        model_name, model_path = _resolve_rembg_choice(choice)
        if model_name == "u2net_custom":
            status = "已存在" if model_path and os.path.isfile(model_path) else "不存在"
            lines.append(f"当前选择: {choice} -> u2net_custom ({status})")
            if model_path:
                lines.append(f"路径: {model_path}")
        else:
            local_path = _model_local_path(model_name)
            status = "已下载" if os.path.isfile(local_path) else "未下载"
            lines.append(f"当前选择: {model_name} ({status})")
            lines.append(f"路径: {local_path}")
    except Exception as e:
        lines.append(f"当前选择: {choice} (错误: {e})")

    downloaded, missing = _list_rembg_models_status()
    lines.append(f"已下载({len(downloaded)}): {', '.join(downloaded) if downloaded else '无'}")
    lines.append(f"未下载({len(missing)}): {', '.join(missing) if missing else '无'}")
    return "\n".join(lines)


def _download_rembg_model(model_choice: Optional[str]) -> str:
    result = None
    try:
        model_name, model_path = _resolve_rembg_choice(model_choice)
    except Exception as e:
        result = f"模型选择失败: {e}"
        return _format_model_status(model_choice, header=f"下载结果: {result}")

    if model_name == "u2net_custom":
        if not model_path:
            result = "REMBG_MODEL_PATH 未设置"
            return _format_model_status(model_choice, header=f"下载结果: {result}")
        if not os.path.isfile(model_path):
            result = f"自定义模型文件不存在: {model_path}"
            return _format_model_status(model_choice, header=f"下载结果: {result}")
        result = f"自定义模型路径: {model_path}"
        return _format_model_status(model_choice, header=f"下载结果: {result}")

    session_class = _REMBG_MODEL_CLASSES.get(model_name)
    if session_class is None:
        result = f"未知模型: {model_name}"
        return _format_model_status(model_choice, header=f"下载结果: {result}")

    local_path = _model_local_path(model_name)
    if os.path.isfile(local_path):
        result = f"模型已存在: {local_path}"
        return _format_model_status(model_choice, header=f"下载结果: {result}")

    try:
        path = session_class.download_models()
        result = f"下载成功: {path}"
    except Exception as e:
        result = f"下载失败: {e}"
    return _format_model_status(model_choice, header=f"下载结果: {result}")


# ----------------------------
# Files / paths normalize (关键修复点)
# ----------------------------
def _file_to_path(f: Any) -> Optional[str]:
    """
    gr.Files 在 gradio 4.x 可能返回：
    - str path
    - dict {"path": "..."} / {"name": "...", "data": ...}
    - FileData 对象（有 .path 属性）
    - None
    我们统一转为真实路径 str
    """
    if f is None:
        return None
    if isinstance(f, str):
        return f
    if isinstance(f, dict):
        # gradio 常见：{"path": "..."}
        if "path" in f and f["path"]:
            return f["path"]
        # 有些情况是 {"name": "..."}，但 name 不是本地路径
        return None
    # FileData / NamedString 等
    p = getattr(f, "path", None)
    if p:
        return p
    # 最后兜底：尝试转字符串，但必须是看起来像路径
    try:
        s = str(f)
        if os.path.exists(s):
            return s
    except Exception:
        pass
    return None


def _normalize_files(files: Any) -> List[str]:
    if not files:
        return []
    # 有些时候 files 是单个
    if not isinstance(files, list):
        files = [files]
    out = []
    for f in files:
        p = _file_to_path(f)
        if p and os.path.exists(p):
            out.append(p)
    return out


# ---------- Utils ----------
def _to_pil(x: Any) -> Image.Image:
    if x is None:
        raise ValueError("Empty image")
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, (str, os.PathLike)):
        return Image.open(x)
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return Image.fromarray(x)
    except Exception:
        pass
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(x))
    if isinstance(x, dict) and "image" in x:
        return _to_pil(x["image"])
    raise TypeError(f"Unsupported type: {type(x)}")


def _save_image_bytes(img: Image.Image, fmt: str, quality: int = 92, bg_color=(255, 255, 255)) -> bytes:
    fmt = (fmt or "PNG").upper()
    buf = io.BytesIO()

    if fmt in ("JPG", "JPEG"):
        if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
            bg = Image.new("RGB", img.size, bg_color)
            rgba = img.convert("RGBA")
            bg.paste(rgba, mask=rgba.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=int(quality), optimize=True)
    elif fmt == "WEBP":
        img.save(buf, format="WEBP", quality=int(quality), method=6)
    else:
        if img.mode not in ("RGBA", "RGB", "LA", "L"):
            img = img.convert("RGBA")
        img.save(buf, format="PNG", optimize=True)

    return buf.getvalue()


def _resolve_compress_format(choice: str, img_format: Optional[str], path: str) -> str:
    if choice and choice != COMPRESS_FORMAT_AUTO:
        return choice
    fmt = (img_format or "").upper()
    if fmt == "JPEG":
        return "JPG"
    if fmt in ("JPG", "PNG", "WEBP"):
        return fmt
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        return "JPG"
    if ext == ".png":
        return "PNG"
    if ext == ".webp":
        return "WEBP"
    return "PNG"


def _zip_bytes(files: List[Tuple[str, bytes]]) -> Optional[str]:
    if not files:
        return None
    zip_path = os.path.join(OUT_DIR, f"batch_{uuid.uuid4().hex}.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, b in files:
            z.writestr(name, b)
    return zip_path


def _write_preview(name: str, b: bytes) -> str:
    p = os.path.join(OUT_DIR, f"{uuid.uuid4().hex}_{name}")
    with open(p, "wb") as f:
        f.write(b)
    return p


def _pick_color(name: str, default=(255, 255, 255)):
    if isinstance(name, (tuple, list)) and len(name) == 3:
        try:
            return (int(name[0]), int(name[1]), int(name[2]))
        except Exception:
            return default

    s = (name or "").strip().lower()
    if s in ("black",):
        return (0, 0, 0)
    if s in ("gray", "grey"):
        return (240, 240, 240)
    if s in ("white",):
        return (255, 255, 255)
    if s.startswith("#"):
        hexv = s[1:]
        if len(hexv) == 3:
            hexv = "".join([c * 2 for c in hexv])
        if len(hexv) == 6:
            try:
                return (int(hexv[0:2], 16), int(hexv[2:4], 16), int(hexv[4:6], 16))
            except Exception:
                return default
    if s.startswith("rgb(") and s.endswith(")"):
        try:
            parts = [p.strip() for p in s[4:-1].split(",")]
            if len(parts) >= 3:
                return (int(float(parts[0])), int(float(parts[1])), int(float(parts[2])))
        except Exception:
            return default
    return default


def _apply_background(img: Image.Image, bg_color) -> Image.Image:
    bg = Image.new("RGB", img.size, bg_color)
    if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
        rgba = img.convert("RGBA")
        bg.paste(rgba, mask=rgba.split()[-1])
    else:
        bg.paste(img.convert("RGB"))
    return bg


def _pad_to_target(img: Image.Image, w: int, h: int, pad_color) -> Image.Image:
    if img.size == (w, h):
        return img
    canvas = Image.new("RGBA", (w, h), pad_color + (255,))
    x = (w - img.size[0]) // 2
    y = (h - img.size[1]) // 2
    if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
        rgba = img.convert("RGBA")
        canvas.paste(rgba, (x, y), rgba.split()[-1])
    else:
        canvas.paste(img.convert("RGB"), (x, y))
    return canvas


# ---------- Remove BG ----------
def batch_remove_bg(
    input_files: Any,
    out_format: str,
    quality: int,
    jpg_bg: str,
    fill_bg: bool,
    fill_color: str,
    model_choice: str,
):
    input_paths = _normalize_files(input_files)
    if not input_paths:
        return [], None, "No input files."

    try:
        session = _get_rembg_session(model_choice)
    except Exception as e:
        return [], None, f"rembg session failed: {e}"

    jpg_color = _pick_color(jpg_bg, (255, 255, 255))
    fill_color = _pick_color(fill_color, jpg_color)
    outputs_gallery = []
    outputs_zip_items = []
    logs = []

    for p in input_paths:
        base = os.path.splitext(os.path.basename(p))[0]
        try:
            raw = open(p, "rb").read()
            cut = rembg_remove(raw, session=session)
            pil = _to_pil(cut).convert("RGBA")
            if fill_bg:
                pil = _apply_background(pil, fill_color)
            out_bytes = _save_image_bytes(
                pil,
                out_format,
                quality=int(quality),
                bg_color=fill_color if fill_bg else jpg_color,
            )
        except Exception as e:
            logs.append(f"[{base}] remove-bg/export failed: {e}")
            continue

        ext = out_format.lower().replace("jpeg", "jpg")
        name = f"{base}_nobg.{ext}"
        outputs_zip_items.append((name, out_bytes))
        outputs_gallery.append(_write_preview(name, out_bytes))

    zip_path = _zip_bytes(outputs_zip_items)
    return outputs_gallery, zip_path, ("\n".join(logs) if logs else "OK")


# ---------- LAMA Inpaint ----------
def _lama_form_defaults(image_size: Tuple[int, int]) -> dict:
    # Lama-cleaner /inpaint expects a full set of form keys (server.py uses form["..."]).
    width, height = image_size
    width = int(width) if width else 0
    height = int(height) if height else 0
    return {
        "ldmSteps": "20",
        "ldmSampler": "plms",
        "zitsWireframe": "false",
        "hdStrategy": "Crop",
        "hdStrategyCropMargin": "196",
        "hdStrategyCropTrigerSize": "800",
        "hdStrategyResizeLimit": "2048",
        "prompt": "",
        "negativePrompt": "",
        "useCroper": "false",
        "croperX": "0",
        "croperY": "0",
        "croperHeight": str(height),
        "croperWidth": str(width),
        "sdScale": "1.0",
        "sdMaskBlur": "0",
        "sdStrength": "0.75",
        "sdSteps": "50",
        "sdGuidanceScale": "7.5",
        "sdSampler": "uni_pc",
        "sdSeed": "-1",
        "sdMatchHistograms": "false",
        "cv2Flag": "INPAINT_NS",
        "cv2Radius": "4",
        "paintByExampleSteps": "50",
        "paintByExampleGuidanceScale": "7.5",
        "paintByExampleMaskBlur": "0",
        "paintByExampleSeed": "-1",
        "paintByExampleMatchHistograms": "false",
        "p2pSteps": "50",
        "p2pImageGuidanceScale": "1.5",
        "p2pGuidanceScale": "7.5",
        "controlnet_conditioning_scale": "0.4",
        "controlnet_method": "control_v11p_sd15_canny",
        "name": "lama",
        "upscale": "false",
        "clicks": "[]",
        "filename": "image.png",
        "clean_img": "false",
    }

async def _lama_inpaint(image_bytes: bytes, mask_bytes: bytes) -> bytes:
    files = {
        "image": ("image.png", image_bytes, "image/png"),
        "mask": ("mask.png", mask_bytes, "image/png"),
    }
    img = _to_pil(image_bytes)
    data = _lama_form_defaults(img.size)

    timeout = httpx.Timeout(LAMA_TIMEOUT, connect=LAMA_CONNECT_TIMEOUT)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{LAMA_SERVER}/inpaint", files=files, data=data)
        if r.status_code != 200:
            raise RuntimeError(f"/inpaint failed {r.status_code}: {r.text[:800]}")
        return r.content


def _extract_editor_mask(editor_value) -> Optional[bytes]:
    if editor_value is None:
        return None

    img = None
    if isinstance(editor_value, dict):
        layers = editor_value.get("layers")
        if layers and isinstance(layers, list) and len(layers) > 0:
            try:
                img = _to_pil(layers[0]).convert("L")
            except Exception:
                try:
                    img = _to_pil(layers[0].get("image")).convert("L")
                except Exception:
                    img = None

        if img is None and editor_value.get("mask") is not None:
            try:
                img = _to_pil(editor_value["mask"]).convert("L")
            except Exception:
                img = None

    if img is None:
        try:
            img = _to_pil(editor_value).convert("L")
        except Exception:
            return None

    # 二值化（白=擦除）
    img = img.point(lambda x: 255 if x > 10 else 0)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def batch_inpaint_ui(input_files: Any, *args):
    input_paths = _normalize_files(input_files)
    if not input_paths:
        return [], None, "No input files."

    out_format = args[-2]
    quality = int(args[-1])
    editor_values = list(args[:-2])

    outputs_gallery = []
    outputs_zip_items = []
    logs = []

    import asyncio

    async def _run_one(idx: int, path: str, sem: asyncio.Semaphore):
        base = os.path.splitext(os.path.basename(path))[0]

        try:
            img_bytes = open(path, "rb").read()
        except Exception as e:
            logs.append(f"[{base}] read failed: {e}")
            return

        mask_bytes = _extract_editor_mask(editor_values[idx])
        if mask_bytes is None:
            logs.append(f"[{base}] No mask -> skip")
            return

        async with sem:
            try:
                out_png = await _lama_inpaint(img_bytes, mask_bytes)
            except Exception as e:
                logs.append(f"[{base}] inpaint failed: {e}")
                return

        try:
            pil = _to_pil(out_png).convert("RGBA")
            out_bytes = _save_image_bytes(pil, out_format, quality=quality)
        except Exception as e:
            logs.append(f"[{base}] export failed: {e}")
            return

        ext = out_format.lower().replace("jpeg", "jpg")
        name = f"{base}_clean.{ext}"
        outputs_zip_items.append((name, out_bytes))
        outputs_gallery.append(_write_preview(name, out_bytes))

    async def _run_all():
        sem = asyncio.Semaphore(MAX_CONCURRENCY)
        max_n = min(len(input_paths), len(editor_values))
        tasks = [asyncio.create_task(_run_one(i, input_paths[i], sem)) for i in range(max_n)]
        await asyncio.gather(*tasks)

    asyncio.run(_run_all())

    zip_path = _zip_bytes(outputs_zip_items)
    return outputs_gallery, zip_path, ("\n".join(logs) if logs else "OK")


# ---------- Single Inpaint Helpers ----------
def _editor_file_path(editor_value: Any) -> Optional[str]:
    if isinstance(editor_value, dict):
        for key in ("background", "composite", "image"):
            val = editor_value.get(key)
            p = _file_to_path(val)
            if p and os.path.exists(p):
                return p
    return None


def _extract_editor_image_bytes(editor_value: Any) -> Optional[bytes]:
    if editor_value is None:
        return None
    if isinstance(editor_value, dict):
        for key in ("background", "composite", "image"):
            val = editor_value.get(key)
            if val is None:
                continue
            p = _file_to_path(val)
            if p and os.path.exists(p):
                try:
                    return open(p, "rb").read()
                except Exception:
                    pass
            try:
                pil = _to_pil(val).convert("RGBA")
                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                return buf.getvalue()
            except Exception:
                pass
    try:
        pil = _to_pil(editor_value).convert("RGBA")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


# ---------- Single Inpaint ----------
def inpaint_single_ui(file_list: Any, editor_value: Any, out_format: str, quality: int, index: int = 0):
    def _no_change(msg: str):
        return gr.update(), gr.update(), gr.update(), msg

    file_path = None
    if file_list and isinstance(file_list, list) and index < len(file_list):
        file_path = _file_to_path(file_list[index])
    elif file_list and not isinstance(file_list, list):
        file_path = _file_to_path(file_list)

    if not file_path:
        file_path = _editor_file_path(editor_value)

    img_bytes = None
    base = None
    if file_path and os.path.exists(file_path):
        base = os.path.splitext(os.path.basename(file_path))[0]
        try:
            img_bytes = open(file_path, "rb").read()
        except Exception as e:
            return _no_change(f"[{base}] read failed: {e}")
    else:
        img_bytes = _extract_editor_image_bytes(editor_value)
        if not img_bytes:
            return _no_change("No input file.")
        base = f"editor_{index+1}"

    mask_bytes = _extract_editor_mask(editor_value)
    if mask_bytes is None:
        return _no_change(f"[{base}] No mask -> skip")

    import asyncio

    async def _run():
        return await _lama_inpaint(img_bytes, mask_bytes)

    try:
        out_png = asyncio.run(_run())
    except Exception as e:
        return _no_change(f"[{base}] inpaint failed: {e}")

    try:
        pil = _to_pil(out_png).convert("RGBA")
        out_bytes = _save_image_bytes(pil, out_format, quality=int(quality))
    except Exception as e:
        return _no_change(f"[{base}] export failed: {e}")

    ext = out_format.lower().replace("jpeg", "jpg")
    name = f"{base}_clean.{ext}"
    out_path = _write_preview(name, out_bytes)
    zip_path = _zip_bytes([(name, out_bytes)])
    return out_path, [out_path], zip_path, "OK"

# ---------- Resize ----------
def _resize_one(
    img: Image.Image,
    w: int,
    h: int,
    mode: str,
    pad_color=(255, 255, 255),
    force_exact: bool = False,
) -> Image.Image:
    img = img.convert("RGBA")

    if mode == "Fit":
        im = img.copy()
        im.thumbnail((w, h), Image.LANCZOS)
        if force_exact:
            return _pad_to_target(im, w, h, pad_color)
        return im

    if mode == "Crop":
        return ImageOps.fit(img, (w, h), method=Image.LANCZOS, centering=(0.5, 0.5))

    if mode == "Pad":
        im2 = ImageOps.contain(img, (w, h), method=Image.LANCZOS)
        canvas = Image.new("RGBA", (w, h), pad_color + (255,))
        x = (w - im2.size[0]) // 2
        y = (h - im2.size[1]) // 2
        canvas.paste(im2, (x, y), im2)
        return canvas

    return img


def batch_resize(
    input_files: Any,
    target_w: int,
    target_h: int,
    mode: str,
    out_format: str,
    quality: int,
    pad_color: str,
    force_exact: bool,
):
    input_paths = _normalize_files(input_files)
    if not input_paths:
        return [], None, "No input files."

    c = _pick_color(pad_color, (255, 255, 255))
    outputs_gallery = []
    outputs_zip_items = []
    logs = []

    for p in input_paths:
        base = os.path.splitext(os.path.basename(p))[0]
        try:
            img = Image.open(p)
            out_img = _resize_one(
                img,
                int(target_w),
                int(target_h),
                mode,
                pad_color=c,
                force_exact=force_exact,
            )
            out_bytes = _save_image_bytes(out_img, out_format, quality=int(quality), bg_color=c)
        except Exception as e:
            logs.append(f"[{base}] resize/export failed: {e}")
            continue

        ext = out_format.lower().replace("jpeg", "jpg")
        name = f"{base}_{int(target_w)}x{int(target_h)}_{mode.lower()}.{ext}"
        outputs_zip_items.append((name, out_bytes))
        outputs_gallery.append(_write_preview(name, out_bytes))

    zip_path = _zip_bytes(outputs_zip_items)
    return outputs_gallery, zip_path, ("\n".join(logs) if logs else "OK")


def batch_compress(
    input_files: Any,
    out_format: str,
    quality: int,
    jpg_bg: str,
):
    input_paths = _normalize_files(input_files)
    if not input_paths:
        return [], None, "No input files."

    bg = _pick_color(jpg_bg, (255, 255, 255))
    outputs_gallery = []
    outputs_zip_items = []
    logs = []

    for p in input_paths:
        base = os.path.splitext(os.path.basename(p))[0]
        try:
            with Image.open(p) as img:
                fmt = _resolve_compress_format(out_format, img.format, p)
                out_bytes = _save_image_bytes(img, fmt, quality=int(quality), bg_color=bg)
        except Exception as e:
            logs.append(f"[{base}] compress/export failed: {e}")
            continue

        ext = fmt.lower().replace("jpeg", "jpg")
        name = f"{base}_compressed.{ext}"
        outputs_zip_items.append((name, out_bytes))
        outputs_gallery.append(_write_preview(name, out_bytes))

    zip_path = _zip_bytes(outputs_zip_items)
    return outputs_gallery, zip_path, ("\n".join(logs) if logs else "OK")


# ---------- UI ----------
with gr.Blocks(
    title="Free Batch Image Tool (Remove BG / Remove Logo / Resize / Compress)",
    css=UI_CSS,
) as demo:
    gr.Markdown(
        "##  免费可视化批量图片处理：扣白底 / 去 Logo / 改尺寸 / 压缩\n"
        f"- inpaint 引擎：lama-cleaner server: {LAMA_SERVER}\n"
        "- 扣白底：rembg\n"
        "- 改尺寸：Pillow\n"
        "- 压缩：Pillow\n"
    )

    with gr.Tab("Batch Remove Background（扣白底）"):
        files_bg = gr.Files(label="拖拽上传多张图片", file_types=["image"])
        out_fmt_bg = gr.Dropdown(["PNG", "WEBP", "JPG"], value="PNG", label="输出格式")
        quality_bg = gr.Slider(50, 100, value=92, step=1, label="质量（JPG/WEBP 有效）")
        jpg_bg = gr.Dropdown(["white", "gray", "black"], value="white", label="JPG 背景色（JPG 输出用）")
        with gr.Row():
            rembg_model = gr.Dropdown(REMBG_MODEL_CHOICES, value=REMBG_MODEL_DEFAULT, label="扣白底模型")
            btn_model_dl = gr.Button("下载模型")
        model_status = gr.Textbox(label="模型状态", value=_format_model_status(REMBG_MODEL_DEFAULT), interactive=False)
        with gr.Row():
            fill_bg = gr.Checkbox(label="填充背景色（输出不透明）", value=False)
            fill_color = gr.ColorPicker(label="填充颜色", value="#FFFFFF")

        btn_bg = gr.Button("开始批量扣白底")
        gallery_bg = gr.Gallery(label="结果预览", columns=4, height=360)
        zip_bg = gr.File(label="下载结果 ZIP")
        log_bg = gr.Textbox(label="日志", lines=6, value="等待点击", interactive=False)

        btn_model_dl.click(
            fn=_download_rembg_model,
            inputs=[rembg_model],
            outputs=[model_status],
        )
        rembg_model.change(
            fn=_format_model_status,
            inputs=[rembg_model],
            outputs=[model_status],
        )
        btn_bg.click(
            fn=batch_remove_bg,
            inputs=[files_bg, out_fmt_bg, quality_bg, jpg_bg, fill_bg, fill_color, rembg_model],
            outputs=[gallery_bg, zip_bg, log_bg],
            concurrency_limit=MAX_CONCURRENCY,
        )

    with gr.Tab("Batch Remove Logo（去 Logo / 去杂物）"):
        gr.Markdown(
            "### 使用说明\n"
            "1) 上传多张图片\n"
            "2) 对每张图用画笔涂抹要移除区域（白色=擦除）\n"
            "3) 点击批量处理\n"
            "\n"
            "（最多 8 张，超过请分批）"
        )

        files_lp = gr.Files(label="拖拽上传多张图片", file_types=["image"])

        with gr.Row():
            out_fmt_lp = gr.Dropdown(["PNG", "WEBP", "JPG"], value="PNG", label="输出格式")
            quality_lp = gr.Slider(50, 100, value=92, step=1, label="质量（JPG/WEBP 有效）")
            btn_lp = gr.Button("开始批量去 Logo（inpaint）")
        log_lp = gr.Textbox(label="日志", lines=6, value="等待点击", interactive=False)

        gr.Markdown("#### Mask 编辑器（最多 8 张）")
        editors = []
        previews = []
        single_btns = []
        for i in range(8):
            with gr.Row():
                img_prev = gr.Image(label=f"第 {i+1} 张预览", interactive=False, height=EDITOR_HEIGHT)
                ed = gr.ImageEditor(
                    label=f"第 {i+1} 张：涂抹要移除区域（白色）",
                    brush=gr.Brush(colors=["#FFFFFF"]),
                    height=EDITOR_HEIGHT,
                    elem_classes=["mask-editor"]
                )
                btn_one = gr.Button(f"只处理第 {i+1} 张", scale=0, min_width=160)
            previews.append(img_prev)
            editors.append(ed)
            single_btns.append(btn_one)

        def _load_previews(files):
            #  关键：把 Files 先变成 path 列表
            paths = _normalize_files(files)
            outs = []
            for i in range(8):
                outs.append(paths[i] if i < len(paths) else None)

            # 对 editor 的初始值：优先给 “背景图”
            # 兼容大多数 gradio 4.x：直接 path
            return outs + outs

        files_lp.change(fn=_load_previews, inputs=[files_lp], outputs=previews + editors)
        gallery_lp = gr.Gallery(label="结果预览", columns=4, height=360)
        zip_lp = gr.File(label="下载结果 ZIP")

        btn_lp.click(fn=lambda: "处理中...",
                     outputs=[log_lp],
                     queue=False).then(
                         fn=batch_inpaint_ui,
                         inputs=[files_lp] + editors + [out_fmt_lp, quality_lp],
                         outputs=[gallery_lp, zip_lp, log_lp],
                         concurrency_limit=MAX_CONCURRENCY)

        for i, btn_one in enumerate(single_btns):
            btn_one.click(fn=lambda: "处理中...",
                          outputs=[log_lp],
                          queue=False).then(
                              fn=functools.partial(inpaint_single_ui, index=i),
                              inputs=[files_lp, editors[i], out_fmt_lp, quality_lp],
                              outputs=[previews[i], gallery_lp, zip_lp, log_lp],
                              concurrency_limit=MAX_CONCURRENCY)

    with gr.Tab("Batch Resize（批量改尺寸）"):
        files_rs = gr.Files(label="拖拽上传多张图片", file_types=["image"])
        with gr.Row():
            w = gr.Number(value=1200, label="目标宽度(px)", precision=0)
            h = gr.Number(value=1200, label="目标高度(px)", precision=0)
        mode = gr.Radio(["Fit", "Crop", "Pad"], value="Fit", label="模式：Fit(不裁切)/Crop(裁切)/Pad(补边)")
        pad_color = gr.Dropdown(["white", "gray", "black"], value="white", label="Pad 补边颜色（Pad模式）")
        force_exact = gr.Checkbox(value=True, label="输出固定尺寸（自动补边）")
        out_fmt_rs = gr.Dropdown(["PNG", "WEBP", "JPG"], value="PNG", label="输出格式")
        quality_rs = gr.Slider(50, 100, value=92, step=1, label="质量（JPG/WEBP 有效）")

        btn_rs = gr.Button("开始批量改尺寸")
        gallery_rs = gr.Gallery(label="结果预览", columns=4, height=360)
        zip_rs = gr.File(label="下载结果 ZIP")
        log_rs = gr.Textbox(label="日志", lines=6, value="等待点击", interactive=False)

        btn_rs.click(fn=batch_resize,
                     inputs=[files_rs, w, h, mode, out_fmt_rs, quality_rs, pad_color, force_exact],
                     outputs=[gallery_rs, zip_rs, log_rs],
                     concurrency_limit=MAX_CONCURRENCY)

    with gr.Tab("Batch Compress（批量压缩）"):
        files_cp = gr.Files(label="拖拽上传多张图片", file_types=["image"])
        with gr.Row():
            out_fmt_cp = gr.Dropdown(COMPRESS_FORMAT_CHOICES, value=COMPRESS_FORMAT_AUTO, label="输出格式")
            quality_cp = gr.Slider(50, 100, value=82, step=1, label="质量（JPG/WEBP 有效）")
            jpg_bg_cp = gr.Dropdown(["white", "gray", "black"], value="white", label="JPG 背景色（JPG 输出用）")

        btn_cp = gr.Button("开始批量压缩")
        gallery_cp = gr.Gallery(label="结果预览", columns=4, height=360)
        zip_cp = gr.File(label="下载结果 ZIP")
        log_cp = gr.Textbox(label="日志", lines=6, value="等待点击", interactive=False)

        btn_cp.click(fn=batch_compress,
                     inputs=[files_cp, out_fmt_cp, quality_cp, jpg_bg_cp],
                     outputs=[gallery_cp, zip_cp, log_cp],
                     concurrency_limit=MAX_CONCURRENCY)

#  建议先不要 queue，等按钮确认都正常后再开 queue（避免 WS 被拦误判）
demo.launch(
    server_name=GRADIO_SERVER_NAME,
    server_port=GRADIO_SERVER_PORT,
    show_error=True,
    debug=True,
)
