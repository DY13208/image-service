import os
import io
import zipfile
import uuid
import time
import functools
from typing import List, Tuple, Optional, Any

import httpx
import gradio as gr
from PIL import Image, ImageOps
from rembg import remove as rembg_remove
from rembg.session_factory import new_session

MAX_CONCURRENCY = 2
LAMA_SERVER = os.getenv("LAMA_SERVER", "http://127.0.0.1:8090")
LAMA_CONNECT_TIMEOUT = float(os.getenv("LAMA_CONNECT_TIMEOUT", "5"))
LAMA_TIMEOUT = float(os.getenv("LAMA_TIMEOUT", "120"))
EDITOR_HEIGHT = 520
REMBG_MODEL_PATH = os.getenv("REMBG_MODEL_PATH", "").strip()

OUT_DIR = os.path.abspath("./_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

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

_REMBG_SESSION = None
_REMBG_SESSION_ERR = None


def _get_rembg_session():
    global _REMBG_SESSION, _REMBG_SESSION_ERR
    if _REMBG_SESSION_ERR:
        raise RuntimeError(_REMBG_SESSION_ERR)
    if _REMBG_SESSION is not None:
        return _REMBG_SESSION
    if REMBG_MODEL_PATH:
        if not os.path.isfile(REMBG_MODEL_PATH):
            _REMBG_SESSION_ERR = f"REMBG_MODEL_PATH not found: {REMBG_MODEL_PATH}"
            raise RuntimeError(_REMBG_SESSION_ERR)
        _REMBG_SESSION = new_session("u2net_custom", model_path=REMBG_MODEL_PATH)
    return _REMBG_SESSION


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
    name = (name or "").lower()
    if name == "black":
        return (0, 0, 0)
    if name == "gray":
        return (240, 240, 240)
    return default


# ---------- Remove BG ----------
def batch_remove_bg(input_files: Any, out_format: str, quality: int, jpg_bg: str):
    input_paths = _normalize_files(input_files)
    if not input_paths:
        return [], None, "No input files."

    bg_color = _pick_color(jpg_bg, (255, 255, 255))
    outputs_gallery = []
    outputs_zip_items = []
    logs = []

    for p in input_paths:
        base = os.path.splitext(os.path.basename(p))[0]
        try:
            raw = open(p, "rb").read()
            session = None
            try:
                session = _get_rembg_session()
            except Exception as e:
                logs.append(f"[{base}] rembg session failed: {e}")
                continue
            cut = rembg_remove(raw, session=session) if session else rembg_remove(raw)
            pil = _to_pil(cut).convert("RGBA")
            out_bytes = _save_image_bytes(pil, out_format, quality=int(quality), bg_color=bg_color)
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
def _resize_one(img: Image.Image, w: int, h: int, mode: str, pad_color=(255, 255, 255)) -> Image.Image:
    img = img.convert("RGBA")

    if mode == "Fit":
        im = img.copy()
        im.thumbnail((w, h), Image.LANCZOS)
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


def batch_resize(input_files: Any, target_w: int, target_h: int, mode: str, out_format: str, quality: int, pad_color: str):
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
            out_img = _resize_one(img, int(target_w), int(target_h), mode, pad_color=c)
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


# ---------- UI ----------
with gr.Blocks(
    title="Free Batch Image Tool (Remove BG / Remove Logo / Resize)",
    css=UI_CSS,
) as demo:
    gr.Markdown(
        "##  免费可视化批量图片处理：扣白底 / 去 Logo / 改尺寸\n"
        f"- inpaint 引擎：lama-cleaner server: {LAMA_SERVER}\n"
        "- 扣白底：rembg\n"
        "- 改尺寸：Pillow\n"
    )

    # 这个 ping 用来证明按钮事件真的会触发后端
    with gr.Row():
        ping_btn = gr.Button("前端点击测试（必应答）")
        ping_status = gr.Textbox(label="连接状态", value="未连接", interactive=False)

    def _ping_backend():
        msg = f"Backend OK {time.strftime('%H:%M:%S')}"
        print(f"[PING] {msg}", flush=True)
        return msg

    ping_btn.click(fn=_ping_backend, outputs=[ping_status], queue=False)

    with gr.Tab("Batch Remove Background（扣白底）"):
        files_bg = gr.Files(label="拖拽上传多张图片", file_types=["image"])
        out_fmt_bg = gr.Dropdown(["PNG", "WEBP", "JPG"], value="PNG", label="输出格式")
        quality_bg = gr.Slider(50, 100, value=92, step=1, label="质量（JPG/WEBP 有效）")
        jpg_bg = gr.Dropdown(["white", "gray", "black"], value="white", label="JPG 背景色（PNG透明不受影响）")

        btn_bg = gr.Button("开始批量扣白底")
        gallery_bg = gr.Gallery(label="结果预览", columns=4, height=360)
        zip_bg = gr.File(label="下载结果 ZIP")
        log_bg = gr.Textbox(label="日志", lines=6, value="等待点击", interactive=False)

        btn_bg.click(fn=batch_remove_bg,
                     inputs=[files_bg, out_fmt_bg, quality_bg, jpg_bg],
                     outputs=[gallery_bg, zip_bg, log_bg],
                     concurrency_limit=MAX_CONCURRENCY)

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
        out_fmt_rs = gr.Dropdown(["PNG", "WEBP", "JPG"], value="PNG", label="输出格式")
        quality_rs = gr.Slider(50, 100, value=92, step=1, label="质量（JPG/WEBP 有效）")

        btn_rs = gr.Button("开始批量改尺寸")
        gallery_rs = gr.Gallery(label="结果预览", columns=4, height=360)
        zip_rs = gr.File(label="下载结果 ZIP")
        log_rs = gr.Textbox(label="日志", lines=6, value="等待点击", interactive=False)

        btn_rs.click(fn=batch_resize,
                     inputs=[files_rs, w, h, mode, out_fmt_rs, quality_rs, pad_color],
                     outputs=[gallery_rs, zip_rs, log_rs],
                     concurrency_limit=MAX_CONCURRENCY)

#  建议先不要 queue，等按钮确认都正常后再开 queue（避免 WS 被拦误判）
demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    show_error=True,
    debug=True,
    analytics_enabled=False,
)
