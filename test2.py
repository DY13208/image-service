import os, io
import httpx
from PIL import Image, ImageDraw

LAMA = "http://127.0.0.1:8090"
img_path = os.path.join(os.path.dirname(__file__), "test.png")

# 读图
img = Image.open(img_path).convert("RGBA")

# 生成 mask：黑底 + 中间白块（白=要擦除）
mask = Image.new("L", img.size, 0)
draw = ImageDraw.Draw(mask)
w, h = img.size
draw.rectangle([int(w*0.35), int(h*0.35), int(w*0.65), int(h*0.65)], fill=255)

# PNG bytes
img_buf = io.BytesIO(); img.save(img_buf, format="PNG"); img_bytes = img_buf.getvalue()
mask_buf = io.BytesIO(); mask.save(mask_buf, format="PNG"); mask_bytes = mask_buf.getvalue()

files = {
    "image": ("image.png", img_bytes, "image/png"),
    "mask": ("mask.png", mask_bytes, "image/png"),
}

# ✅ 按你提取的 42 个 key 全部补齐
data = {
    # --- inpaint 核心 ---
    "hdStrategy": "Crop",
    "hdStrategyCropMargin": "196",
    "hdStrategyCropTrigerSize": "800",
    "hdStrategyResizeLimit": "2048",

    "useCroper": "false",
    "croperX": "0",
    "croperY": "0",
    "croperHeight": str(h),
    "croperWidth": str(w),

    # 文本提示（lama 模型不一定用，但必须给）
    "prompt": "",
    "negativePrompt": "",

    # --- SD 相关（即使不用也给默认） ---
    "sdMaskBlur": "8",
    "sdStrength": "0.75",
    "sdSteps": "20",
    "sdGuidanceScale": "7.5",
    "sdSampler": "euler",
    "sdSeed": "-1",
    "sdMatchHistograms": "false",
    "sdScale": "1",

    # --- LDM 相关 ---
    "ldmSteps": "20",
    "ldmSampler": "plms",

    # --- ZITS / CV2 ---
    "zitsWireframe": "false",
    "cv2Radius": "5",
    "cv2Flag": "false",

    # --- Paint by Example ---
    "paintByExampleSteps": "30",
    "paintByExampleGuidanceScale": "7.5",
    "paintByExampleSeed": "-1",
    "paintByExampleMaskBlur": "8",
    "paintByExampleMatchHistograms": "false",
    # 不提供示例图就不要传文件，但 key 仍给空（避免缺键）
    "paintByExampleImage": "",

    # --- Pix2Pix / ControlNet ---
    "p2pSteps": "20",
    "p2pImageGuidanceScale": "1.5",
    "p2pGuidanceScale": "7.5",
    "controlnet_conditioning_scale": "0.5",
    "controlnet_method": "control_v11p_sd15_inpaint",

    # --- 下面这些 key 可能是别的接口也共用，先给默认避免缺键 ---
    "name": "lama",
    "upscale": "false",
    "clicks": "[]",

    "filename": "image.png",
    "clean_img": "false",
}

r = httpx.post(f"{LAMA}/inpaint", files=files, data=data, timeout=300)

print("status:", r.status_code)
print("content-type:", r.headers.get("content-type"))
if r.status_code != 200:
    # 这次不应该再是“缺字段 BadRequest”，如果还失败会更具体
    print("error head:", r.text[:1500])
else:
    out_path = os.path.join(os.path.dirname(__file__), "out.png")
    with open(out_path, "wb") as f:
        f.write(r.content)
    print("saved:", out_path)