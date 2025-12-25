# Image Service

Batch image tools with a Gradio UI. Provides remove background, remove logo/inpaint,
resize, and a pipeline that combines steps. The UI runs on port 7860 and the
lama-cleaner server runs on port 8090.

## Features
- Batch remove background using rembg (U2NET model)
- Batch remove logo or objects via lama-cleaner inpaint
- Batch resize (fit/crop/pad) with Pillow
- Pipeline: remove background -> inpaint -> resize
- ZIP output for batch results

## Requirements
- Windows PowerShell (scripts are .ps1)
- Two Python virtual environments:
  - `.venv-lama` for lama-cleaner
  - `.venv-ui` for the Gradio UI and rembg

## Quick Start
Use the provided scripts:
```powershell
cd D:\image-service
powershell -ExecutionPolicy Bypass -File .\start-all.ps1
```

Stop services:
```powershell
cd D:\image-service
powershell -ExecutionPolicy Bypass -File .\stop-all.ps1
```

Open the UI in the browser:
```
http://127.0.0.1:7860
```

## Manual Start
Start lama-cleaner in one terminal:
```powershell
cd D:\image-service
. .\.venv-lama\Scripts\Activate.ps1
lama-cleaner --host 0.0.0.0 --port 8090 --device=cuda --model=lama
```

Start Gradio UI in another terminal:
```powershell
cd D:\image-service
. .\.venv-ui\Scripts\Activate.ps1
$env:LAMA_SERVER = "http://127.0.0.1:8090"
python app.py
```

## Usage Notes
- Remove Background tab uses rembg and needs the U2NET model.
- Remove Logo tab requires a mask drawn in the ImageEditor (white = remove).
- Pipeline tab lets you combine steps. If you do not want a step, turn it off.

## Environment Variables
- `LAMA_SERVER` (default `http://127.0.0.1:8090`): lama-cleaner base URL
- `REMBG_MODEL_PATH`: local path to `u2net.onnx` to avoid downloads
- `U2NET_HOME`: directory where rembg caches model files
- `GRADIO_SERVER_NAME` / `GRADIO_SERVER_PORT`: override UI host/port

## Output
Outputs are written to `_outputs`:
- Preview images used by the Gallery
- ZIP files for batch downloads

## Troubleshooting
### Buttons do nothing
- Open DevTools Console. If you see `Failed to construct 'URL': Invalid URL`,
  a bad `<a href>` is breaking Gradio initialization. Check any Markdown links
  and make sure `href` is a valid URL.

### Remove background fails with GitHub download error
rembg tries to download `u2net.onnx` from GitHub. Fix options:
1) Set a local model path:
   ```powershell
   $env:REMBG_MODEL_PATH = "D:\image-service\models\u2net.onnx"
   python app.py
   ```
2) Place the model in the default cache:
   ```
   C:\Users\Administrator\.u2net\u2net.onnx
   ```
3) Disable or fix proxy settings if downloads are blocked:
   ```powershell
   Remove-Item Env:HTTP_PROXY,Env:HTTPS_PROXY,Env:ALL_PROXY -ErrorAction SilentlyContinue
   ```
4) Set lama time out
   $env:LAMA_CONNECT_TIMEOUT=3
   $env:LAMA_TIMEOUT=60
   python app.py

### lama-cleaner not reachable
- Ensure port 8090 is open and the server is running.
- Check `LAMA_SERVER` matches the actual host/port.

### Port already in use
- Stop the existing process or change the port in `start-all.ps1` and `app.py`.

## Project Layout
- `app.py` - Gradio UI and processing logic
- `start-all.ps1` / `stop-all.ps1` - start and stop services
- `_outputs` - generated previews and ZIPs
