# Image Service / 图片服务

Batch image tools with a Gradio UI. Provides remove background, remove logo/inpaint,
resize, and a pipeline that combines steps. The UI runs on port 7860 and the
lama-cleaner server runs on port 8090.

基于 Gradio 的批量图片工具。包含扣白底、去 Logo/去杂物（inpaint）、改尺寸，以及
可组合的流水线处理。UI 端口 7860，lama-cleaner 端口 8090。

## Features / 功能
- Batch remove background using rembg (U2NET model) / 使用 rembg 批量扣白底（U2NET 模型）
- Batch remove logo or objects via lama-cleaner inpaint / 使用 lama-cleaner 批量去 Logo/去杂物
- Batch resize (fit/crop/pad) with Pillow / 使用 Pillow 批量改尺寸（Fit/Crop/Pad）
- Pipeline: remove background -> inpaint -> resize / 流水线：扣白底 -> 去 Logo -> 改尺寸
- ZIP output for batch results / 批量结果打包 ZIP 下载

## Requirements / 环境要求
- Windows PowerShell (scripts are .ps1) / Windows PowerShell（脚本为 .ps1）
- Two Python virtual environments / 两个 Python 虚拟环境:
  - `.venv-lama` for lama-cleaner / `.venv-lama` 用于 lama-cleaner
  - `.venv-ui` for the Gradio UI and rembg / `.venv-ui` 用于 Gradio UI 和 rembg

## Quick Start / 快速启动
Use the provided scripts / 使用脚本启动:
```powershell
cd D:\image-service
powershell -ExecutionPolicy Bypass -File .\start-all.ps1
```

Stop services / 停止服务:
```powershell
cd D:\image-service
powershell -ExecutionPolicy Bypass -File .\stop-all.ps1
```

Open the UI in the browser / 浏览器打开:
```
http://127.0.0.1:7860
```

## Manual Start / 手动启动
Start lama-cleaner in one terminal / 在一个终端启动 lama-cleaner:
```powershell
cd D:\image-service
. .\.venv-lama\Scripts\Activate.ps1
lama-cleaner --host 0.0.0.0 --port 8090 --device=cuda --model=lama
```

Start Gradio UI in another terminal / 另一个终端启动 Gradio UI:
```powershell
cd D:\image-service
. .\.venv-ui\Scripts\Activate.ps1
$env:LAMA_SERVER = "http://127.0.0.1:8090"
python app.py
```

## Packaging to EXE / 打包成 EXE
This project depends on two virtual environments and large ML models, so a true
single-file EXE would be very large. The recommended approach is to build a
launcher EXE that starts lama-cleaner and the Gradio UI.

本项目依赖两个虚拟环境和较大的模型，真正的单文件 EXE 体积会很大。
推荐打包一个“启动器 EXE”，用于启动 lama-cleaner 与 Gradio UI。

Install PyInstaller (once) / 安装 PyInstaller（一次性）:
```powershell
.\.venv-ui\Scripts\python.exe -m pip install pyinstaller
```

Build the launcher / 构建启动器:
```powershell
cd D:\image-service
.\tools\build_exe.ps1
```

Run the launcher / 运行启动器:
```
dist\image-service.exe
```

Notes / 说明:
- If the EXE stays in `dist\`, it will automatically use the project root (`dist\..`) to find `.venv-ui` and `.venv-lama`.
- If you move the EXE elsewhere, set `IMAGE_SERVICE_ROOT` to the project root.

注意:
- 如果 EXE 在 `dist\` 目录，会自动向上查找项目根目录中的 `.venv-ui` 与 `.venv-lama`。
- 如果 EXE 被移动到其他位置，请设置 `IMAGE_SERVICE_ROOT` 指向项目根目录。

Options / 可选参数:
```
dist\image-service.exe --no-browser
dist\image-service.exe --lama-device cpu
dist\image-service.exe --lama-port 8090 --ui-port 7860
```

## Usage Notes / 使用说明
- Remove Background tab uses rembg and needs the U2NET model. / 扣白底使用 rembg，需要 U2NET 模型。
- Remove Logo tab requires a mask drawn in the ImageEditor (white = remove). / 去 Logo 需要在编辑器里涂抹蒙版（白色为擦除）。
- Pipeline tab lets you combine steps. If you do not want a step, turn it off. / 流水线可组合步骤，不需要的步骤可以关闭。

## Environment Variables / 环境变量
- `LAMA_SERVER` (default `http://127.0.0.1:8090`): lama-cleaner base URL / lama-cleaner 服务地址
- `REMBG_MODEL_PATH`: local path to `u2net.onnx` to avoid downloads / 指定本地 `u2net.onnx`，避免下载
- `U2NET_HOME`: directory where rembg caches model files / rembg 模型缓存目录
- `GRADIO_SERVER_NAME` / `GRADIO_SERVER_PORT`: override UI host/port / 覆盖 UI 主机与端口
- `GRADIO_ANALYTICS_ENABLED=False`: disable Gradio telemetry / 关闭 Gradio 统计

## Output / 输出目录
Outputs are written to `_outputs` / 输出写入 `_outputs`:
- Preview images used by the Gallery / 结果预览图
- ZIP files for batch downloads / 批量下载 ZIP

## Troubleshooting / 常见问题
### Buttons do nothing / 按钮无反应
- Open DevTools Console. If you see `Failed to construct 'URL': Invalid URL`,
  a bad `<a href>` is breaking Gradio initialization. Check any Markdown links
  and make sure `href` is a valid URL.
- 打开浏览器控制台。如果看到 `Failed to construct 'URL': Invalid URL`，
  通常是 Markdown 里的链接 `href` 不合法，导致 Gradio 初始化失败。

### Remove background fails with GitHub download error / 扣白底下载模型失败
rembg tries to download `u2net.onnx` from GitHub. Fix options / rembg 会从 GitHub 下载模型，可用以下方式解决:
1) Set a local model path / 指定本地模型路径:
   ```powershell
   $env:REMBG_MODEL_PATH = "D:\image-service\models\u2net.onnx"
   python app.py
   ```
2) Place the model in the default cache / 放到默认缓存目录:
   ```
   C:\Users\Administrator\.u2net\u2net.onnx
   ```
3) Disable or fix proxy settings if downloads are blocked / 代理导致下载失败时可清理代理:
   ```powershell
   Remove-Item Env:HTTP_PROXY,Env:HTTPS_PROXY,Env:ALL_PROXY -ErrorAction SilentlyContinue
   ```
4) Set LAMA timeout / 设置 LAMA 超时:
   ```powershell
   $env:LAMA_CONNECT_TIMEOUT=3
   $env:LAMA_TIMEOUT=60
   python app.py
   ```

### lama-cleaner not reachable / lama-cleaner 无法访问
- Ensure port 8090 is open and the server is running. / 确认 8090 端口可用且服务已启动。
- Check `LAMA_SERVER` matches the actual host/port. / 检查 `LAMA_SERVER` 是否正确。

### Port already in use / 端口被占用
- Stop the existing process or change the port in `start-all.ps1` and `app.py`. / 停止占用进程，或在 `start-all.ps1` 和 `app.py` 里改端口。

## Project Layout / 项目结构
- `app.py` - Gradio UI and processing logic / Gradio UI 与处理逻辑
- `start-all.ps1` / `stop-all.ps1` - start and stop services / 启动与停止脚本
- `_outputs` - generated previews and ZIPs / 结果预览与 ZIP
