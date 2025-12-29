import os
from typing import List, Tuple, Optional, Any, Dict

from rembg import remove as rembg_remove
from rembg.session_factory import new_session
from rembg.sessions import sessions_names, sessions_class

from config import MODELS_DIR, REMBG_MODEL_PATH
from file_utils import (
    _normalize_files,
    _to_pil,
    _save_image_bytes,
    _zip_bytes,
    _write_preview,
    _pick_color,
    _apply_background,
)

REMBG_MODEL_AUTO = "auto (REMBG_MODEL_PATH / u2net)"
REMBG_MODEL_CUSTOM = "custom (REMBG_MODEL_PATH)"
REMBG_EXCLUDE_MODELS = {"u2net_custom", "u2net_cloth_seg", "sam"}
REMBG_MODELS = sorted([m for m in sessions_names if m not in REMBG_EXCLUDE_MODELS])
REMBG_MODEL_CHOICES = [REMBG_MODEL_AUTO]
if REMBG_MODEL_PATH:
    REMBG_MODEL_CHOICES.append(REMBG_MODEL_CUSTOM)
REMBG_MODEL_CHOICES.extend(REMBG_MODELS)
REMBG_MODEL_DEFAULT = REMBG_MODEL_AUTO

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
