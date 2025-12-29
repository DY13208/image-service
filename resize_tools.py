import os
from typing import Any

from PIL import Image, ImageOps

from file_utils import (
    _normalize_files,
    _save_image_bytes,
    _pick_color,
    _write_preview,
    _zip_bytes,
    _to_pil,
    _pad_to_target,
)


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
        try:
            outputs_gallery.append(_to_pil(out_bytes).copy())
        except Exception:
            outputs_gallery.append(_write_preview(name, out_bytes))

    zip_path = _zip_bytes(outputs_zip_items)
    return outputs_gallery, zip_path, ("\n".join(logs) if logs else "OK")
