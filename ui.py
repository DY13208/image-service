import functools

import gradio as gr

from config import (
    UI_CSS,
    MAX_CONCURRENCY,
    LAMA_SERVER,
    PREVIEW_HEIGHT,
    EDITOR_HEIGHT,
    EDITOR_CANVAS_SIZE,
    EDITOR_ZOOM_CHOICES,
    COMPRESS_FORMAT_CHOICES,
    COMPRESS_FORMAT_AUTO,
    EDITOR_SLOTS,
)
from file_utils import _normalize_files, _make_zoom_image, _make_editor_image
from rembg_tools import (
    REMBG_MODEL_CHOICES,
    REMBG_MODEL_DEFAULT,
    _format_model_status,
    _download_rembg_model,
    batch_remove_bg,
)
from inpaint_tools import (
    batch_inpaint_ui,
    inpaint_single_ui,
    _open_zoom_editor,
    _save_zoom_mask,
    _close_zoom_editor,
    _reset_mask_overrides,
)
from resize_tools import batch_resize
from compress_tools import batch_compress


def build_demo():
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
                zoom_lp = gr.Dropdown(EDITOR_ZOOM_CHOICES, value=1, label="放大编辑倍数")
                btn_lp = gr.Button("开始批量去 Logo（inpaint）")
            log_lp = gr.Textbox(label="日志", lines=6, value="等待点击", interactive=False)
            mask_overrides = gr.State({})
            zoom_state = gr.State({})

            gr.Markdown("#### Mask 编辑器（最多 8 张）")
            editors = []
            previews = []
            single_btns = []
            zoom_btns = []
            for i in range(EDITOR_SLOTS):
                with gr.Row():
                    img_prev = gr.Image(label=f"第 {i+1} 张预览", interactive=False, height=PREVIEW_HEIGHT)
                    ed = gr.ImageEditor(
                        label=f"第 {i+1} 张：涂抹要移除区域（白色）",
                        brush=gr.Brush(colors=["#FFFFFF"]),
                        height=EDITOR_HEIGHT,
                        canvas_size=EDITOR_CANVAS_SIZE,
                        elem_classes=["mask-editor"]
                    )
                with gr.Column(scale=0, min_width=140):
                    btn_one = gr.Button(f"只处理第 {i+1} 张")
                    btn_zoom = gr.Button("放大编辑", variant="secondary", size="sm")
                previews.append(img_prev)
                editors.append(ed)
                single_btns.append(btn_one)
                zoom_btns.append(btn_zoom)

            def _load_previews(files, zoom):
                #  关键：把 Files 先变成 path 列表
                paths = _normalize_files(files)
                outs = []
                editor_outs = []
                for i in range(EDITOR_SLOTS):
                    p = paths[i] if i < len(paths) else None
                    outs.append(p)
                    editor_outs.append(_make_editor_image(p, EDITOR_CANVAS_SIZE) if p else None)

                # 对 editor 的初始值：优先给 “背景图”
                # 兼容大多数 gradio 4.x：直接 path
                return outs + editor_outs

            files_lp.change(fn=_load_previews, inputs=[files_lp, zoom_lp], outputs=previews + editors)
            zoom_lp.change(fn=_load_previews, inputs=[files_lp, zoom_lp], outputs=previews + editors)
            files_lp.change(fn=_reset_mask_overrides, outputs=[mask_overrides], queue=False)

            with gr.Column(visible=False, elem_id="zoom-panel") as zoom_panel:
                zoom_title = gr.Markdown("### 放大编辑")
                zoom_editor = gr.ImageEditor(
                    label="放大涂抹（白色=擦除）",
                    brush=gr.Brush(colors=["#FFFFFF"]),
                    height=EDITOR_HEIGHT,
                    elem_classes=["mask-editor"]
                )
                with gr.Row():
                    zoom_save = gr.Button("保存并关闭")
                    zoom_close = gr.Button("关闭")

            gallery_lp = gr.Gallery(label="结果预览", columns=4, height=360)
            zip_lp = gr.File(label="下载结果 ZIP")

            btn_lp.click(fn=lambda: "处理中...",
                         outputs=[log_lp],
                         queue=False).then(
                             fn=batch_inpaint_ui,
                             inputs=[files_lp] + editors + [out_fmt_lp, quality_lp, mask_overrides],
                             outputs=[gallery_lp, zip_lp, log_lp],
                             concurrency_limit=MAX_CONCURRENCY)

            for i, btn_one in enumerate(single_btns):
                btn_one.click(fn=lambda: "处理中...",
                              outputs=[log_lp],
                              queue=False).then(
                                  fn=functools.partial(inpaint_single_ui, index=i),
                                  inputs=[files_lp, editors[i], out_fmt_lp, quality_lp, mask_overrides],
                                  outputs=[previews[i], gallery_lp, zip_lp, log_lp],
                                  concurrency_limit=MAX_CONCURRENCY)

            for i, btn_zoom in enumerate(zoom_btns):
                btn_zoom.click(
                    fn=functools.partial(_open_zoom_editor, index=i),
                    inputs=[files_lp, zoom_lp],
                    outputs=[zoom_panel, zoom_title, zoom_editor, zoom_state, log_lp],
                )

            zoom_save.click(
                fn=_save_zoom_mask,
                inputs=[zoom_editor, zoom_state, mask_overrides],
                outputs=[mask_overrides, zoom_panel, log_lp],
            )
            zoom_close.click(fn=_close_zoom_editor, outputs=[zoom_panel], queue=False)

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

    return demo
