import os
import io
from typing import Tuple, Optional, Any

import httpx
import gradio as gr
from PIL import Image

from config import MAX_CONCURRENCY, LAMA_SERVER, LAMA_CONNECT_TIMEOUT, LAMA_TIMEOUT, EDITOR_SLOTS
from file_utils import (
    _normalize_files,
    _file_to_path,
    _to_pil,
    _save_image_bytes,
    _write_preview,
    _zip_bytes,
    _make_zoom_image,
)


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


def _extract_editor_mask(editor_value, target_size: Optional[Tuple[int, int]] = None) -> Optional[bytes]:
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
    if target_size and img.size != target_size:
        img = img.resize(target_size, Image.NEAREST)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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


def _get_mask_override(mask_overrides: Optional[dict], idx: int) -> Optional[bytes]:
    if not mask_overrides:
        return None
    if idx in mask_overrides:
        return mask_overrides[idx]
    key = str(idx)
    if key in mask_overrides:
        return mask_overrides[key]
    return None


def _open_zoom_editor(files, zoom, index: int):
    paths = _normalize_files(files)
    if index >= len(paths):
        return gr.update(visible=False), "### 放大编辑", None, {}, "未找到对应图片"
    path = paths[index]
    zoom = int(zoom) if zoom else 1
    zoom_path = _make_zoom_image(path, zoom)
    size = None
    try:
        with Image.open(path) as img:
            size = [img.size[0], img.size[1]]
    except Exception:
        size = None
    state = {"index": int(index), "path": path, "zoom": int(zoom), "size": size}
    title = f"### 放大编辑第 {index+1} 张（{zoom}x）"
    return gr.update(visible=True), title, zoom_path, state, f"进入放大编辑：第 {index+1} 张"


def _save_zoom_mask(zoom_editor_value, zoom_state: dict, mask_overrides: Optional[dict]):
    if not zoom_state or zoom_state.get("index") is None:
        return mask_overrides or {}, gr.update(visible=False), "未选择要放大的图片"
    idx = int(zoom_state.get("index", 0))
    size = zoom_state.get("size")
    target_size = None
    if size and len(size) == 2:
        target_size = (int(size[0]), int(size[1]))
    mask_bytes = _extract_editor_mask(zoom_editor_value, target_size=target_size)
    if mask_bytes is None:
        return mask_overrides or {}, gr.update(visible=True), "没有检测到蒙版"
    new_overrides = dict(mask_overrides or {})
    new_overrides[idx] = mask_bytes
    return new_overrides, gr.update(visible=False), f"已保存第 {idx+1} 张放大蒙版"


def _close_zoom_editor():
    return gr.update(visible=False)


def _reset_mask_overrides():
    return {}


def _format_exc(e: Exception) -> str:
    try:
        msg = str(e).strip()
    except Exception:
        msg = ""
    if msg:
        return f"{type(e).__name__}: {msg}"
    return f"{type(e).__name__}: {repr(e)}"


def batch_inpaint_ui(input_files: Any, *args):
    input_paths = _normalize_files(input_files)
    if not input_paths:
        return [], None, "No input files."

    editor_values = list(args[:EDITOR_SLOTS])
    out_format = args[EDITOR_SLOTS]
    quality = int(args[EDITOR_SLOTS + 1])
    mask_overrides = args[EDITOR_SLOTS + 2] if len(args) > (EDITOR_SLOTS + 2) else None

    outputs_gallery = []
    outputs_zip_items = []
    logs = []

    import asyncio

    async def _run_one(idx: int, path: str, sem: asyncio.Semaphore):
        base = os.path.splitext(os.path.basename(path))[0]

        try:
            img_bytes = open(path, "rb").read()
            img_size = _to_pil(img_bytes).size
        except Exception as e:
            logs.append(f"[{base}] read failed: {e}")
            return

        mask_bytes = _get_mask_override(mask_overrides, idx)
        if mask_bytes is None:
            mask_bytes = _extract_editor_mask(editor_values[idx], target_size=img_size)
        if mask_bytes is None:
            logs.append(f"[{base}] No mask -> skip")
            return

        async with sem:
            try:
                out_png = await _lama_inpaint(img_bytes, mask_bytes)
            except Exception as e:
                logs.append(f"[{base}] inpaint failed: {_format_exc(e)}")
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


def inpaint_single_ui(
    file_list: Any,
    editor_value: Any,
    out_format: str,
    quality: int,
    mask_overrides: Optional[dict],
    index: int = 0,
):
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
    img_size = None
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

    try:
        img_size = _to_pil(img_bytes).size
    except Exception:
        img_size = None

    mask_bytes = _get_mask_override(mask_overrides, index)
    if mask_bytes is None:
        mask_bytes = _extract_editor_mask(editor_value, target_size=img_size)
    if mask_bytes is None:
        return _no_change(f"[{base}] No mask -> skip")

    import asyncio

    async def _run():
        return await _lama_inpaint(img_bytes, mask_bytes)

    try:
        out_png = asyncio.run(_run())
    except Exception as e:
        return _no_change(f"[{base}] inpaint failed: {_format_exc(e)}")

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
