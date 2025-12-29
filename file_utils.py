import os
import io
import zipfile
import uuid
import hashlib
from typing import List, Tuple, Optional, Any

from PIL import Image

from config import OUT_DIR


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


def _editor_zoom_path(src_path: str, zoom: int) -> str:
    try:
        mtime = int(os.path.getmtime(src_path))
    except Exception:
        mtime = 0
    key = hashlib.md5(src_path.encode("utf-8")).hexdigest()[:12]
    return os.path.join(OUT_DIR, f"editor_zoom_{key}_{zoom}x_{mtime}.png")


def _editor_fit_path(src_path: str, max_w: int, max_h: int) -> str:
    try:
        mtime = int(os.path.getmtime(src_path))
    except Exception:
        mtime = 0
    key = hashlib.md5(src_path.encode("utf-8")).hexdigest()[:12]
    return os.path.join(OUT_DIR, f"editor_fit_{key}_{max_w}x{max_h}_{mtime}.png")


def _make_editor_image(src_path: str, max_size: Tuple[int, int]) -> str:
    try:
        max_w, max_h = int(max_size[0]), int(max_size[1])
    except Exception:
        return src_path
    if max_w <= 0 or max_h <= 0:
        return src_path
    out_path = _editor_fit_path(src_path, max_w, max_h)
    if os.path.isfile(out_path):
        return out_path
    try:
        with Image.open(src_path) as img:
            if img.width <= max_w and img.height <= max_h:
                return src_path
            img.thumbnail((max_w, max_h), Image.LANCZOS)
            img.save(out_path, format="PNG", optimize=True)
        return out_path
    except Exception:
        return src_path


def _make_zoom_image(src_path: str, zoom: int) -> str:
    if not zoom or zoom <= 1:
        return src_path
    out_path = _editor_zoom_path(src_path, zoom)
    if os.path.isfile(out_path):
        return out_path
    try:
        with Image.open(src_path) as img:
            w, h = img.size
            new_size = (max(1, int(w * zoom)), max(1, int(h * zoom)))
            zoomed = img.resize(new_size, Image.LANCZOS)
            zoomed.save(out_path, format="PNG", optimize=True)
        return out_path
    except Exception:
        return src_path
