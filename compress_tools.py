import os
from typing import Any, Optional

from PIL import Image

from config import COMPRESS_FORMAT_AUTO
from file_utils import (
    _normalize_files,
    _save_image_bytes,
    _pick_color,
    _write_preview,
    _zip_bytes,
)


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
