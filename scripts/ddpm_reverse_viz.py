"""
DDPM Denoising (Reverse) Process — 精确可视化
=============================================
使用 Hugging Face diffusers 的预训练 DDPM 模型展示真实去噪过程。

论文: Ho et al. 2020 "Denoising Diffusion Probabilistic Models"
反向过程: p_θ(x_{t-1} | x_t) = N(μ_θ(x_t, t), σ_t² I)

展示从 x_T ~ N(0,I) 逐步去噪为清晰图像的过程。
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import DDPMPipeline
import os, json

# ── 1. 组装 6 帧横向可视化（与正向图一致布局）──────────────────
def make_reverse_visualization(frames, labels, output_path, title="DDPM Reverse Denoising Process"):
    """将 6 帧去噪过程拼接为横向大图"""
    n = len(frames)
    # 所有帧统一 resize 到 256×256
    final_size = 256
    resized = []
    for f in frames:
        if isinstance(f, torch.Tensor):
            f = f.detach().cpu().numpy()
        if isinstance(f, np.ndarray):
            f = Image.fromarray(f)
        f = f.resize((final_size, final_size), Image.NEAREST)
        resized.append(f)

    h, w = final_size, final_size
    pad = 20
    label_h = 40
    title_h = 50

    total_w = n * w + (n - 1) * pad + 2 * pad
    total_h = title_h + h + label_h + pad

    canvas = Image.new('RGB', (total_w, total_h), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("C:\\Windows\\Fonts\\segoeui.ttf", 22)
        font_label = ImageFont.truetype("C:\\Windows\\Fonts\\segoeui.ttf", 16)
    except:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), title, font=font_title)
    tw = bbox[2] - bbox[0]
    draw.text(((total_w - tw) // 2, 12), title, fill=(50, 50, 50), font=font_title)

    for i, (frame, label) in enumerate(zip(resized, labels)):
        x_offset = pad + i * (w + pad)
        y_offset = title_h
        canvas.paste(frame, (x_offset, y_offset))

        draw.rectangle(
            [x_offset - 1, y_offset - 1, x_offset + w, y_offset + h],
            outline=(180, 180, 180), width=1
        )

        bbox = draw.textbbox((0, 0), label, font=font_label)
        lw = bbox[2] - bbox[0]
        draw.text(
            (x_offset + (w - lw) // 2, title_h + h + 8),
            label, fill=(80, 80, 80), font=font_label
        )

        if i < n - 1:
            arrow_x = x_offset + w + 4
            arrow_y = title_h + h // 2
            draw.polygon(
                [(arrow_x, arrow_y - 6), (arrow_x + 10, arrow_y),
                 (arrow_x, arrow_y + 6)],
                fill=(160, 160, 160)
            )

    canvas.save(output_path, quality=95)
    return output_path

# ── 2. 手动控制 DDPM 反向去噪（保存中间帧）─────────────────────
def demo_ddpm_denoising(pipe, seed=42, num_show=6, num_steps=50):
    """
    手动执行 DDPM 反向去噪，捕获中间帧。
    使用 DDIM 风格的等间隔采样，每 num_steps/num_show 步保存一帧。
    """
    # 设置随机种子
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # 获取调度器参数
    scheduler = pipe.scheduler
    num_train_timesteps = scheduler.config.num_train_timesteps

    # 设置推理步数
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps

    # 从纯噪声开始
    batch_size = 1
    in_channels = pipe.unet.config.in_channels
    image_size = pipe.unet.config.sample_size
    image = torch.randn(batch_size, in_channels, image_size, image_size,
                        generator=generator)

    frames = [image.clone()]  # t = T (纯噪声)
    save_indices = set()

    # 计算保存间隔
    total_steps = len(timesteps)
    save_step = max(1, total_steps // (num_show - 1))
    for j in range(1, num_show):
        idx = min(j * save_step, total_steps - 1)
        save_indices.add(idx)
    # 确保最后一帧总是保存
    save_indices.add(total_steps - 1)

    # 逐步去噪
    for i, t in enumerate(timesteps):
        with torch.no_grad():
            model_output = pipe.unet(image, t).sample
            image = scheduler.step(model_output, t, image).prev_sample

        if i in save_indices:
            frames.append(image.clone())

    # 将 tensor 转换为 PIL 图像
    pil_frames = []
    for f in frames:
        # DDPM 输出在 [-1, 1] 范围
        f_np = f.cpu().squeeze().detach().numpy()
        f_scaled = (f_np * 0.5 + 0.5)  # [-1,1] -> [0,1]
        f_scaled = np.clip(f_scaled, 0, 1) * 255
        f_scaled = f_scaled.astype(np.uint8)
        if f_scaled.shape[0] == 3:
            f_scaled = np.transpose(f_scaled, (1, 2, 0))  # CHW -> HWC
        elif len(f_scaled.shape) == 2:
            f_scaled = np.stack([f_scaled]*3, axis=-1)
        pil_frames.append(Image.fromarray(f_scaled))

    return pil_frames

# ── 3. 主流程 ───────────────────────────────────────────────────
def main():
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("Loading DDPM model (google/ddpm-cifar10-32)...")
    pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
    pipe.to("cpu")
    print("Model loaded.\n")

    print("Running denoising (100 inference steps, seed=42)...")
    frames = demo_ddpm_denoising(pipe, seed=42, num_show=12, num_steps=100)

    # 标签 — 12 帧，更密的中间步
    labels = [
        "x_T (纯噪声)",
        "t ≈ 909",
        "t ≈ 818",
        "t ≈ 727",
        "t ≈ 636",
        "t ≈ 545",
        "t ≈ 454",
        "t ≈ 363",
        "t ≈ 272",
        "t ≈ 181",
        "t ≈ 90",
        "x₀ (去噪完成)"
    ]

    output_dir = r"C:\Users\12926\WorkBuddy\2026-06-27-18-36-45\ai-learning-site\images"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ddpm-reverse-denoising.png")

    make_reverse_visualization(frames, labels, output_path)

    meta = {
        "paper": "Denoising Diffusion Probabilistic Models (Ho et al., NeurIPS 2020)",
        "model": "google/ddpm-cifar10-32",
        "inference_steps": 100,
        "frames": 12,
        "seed": 42,
        "training_steps": 1000,
        "note": "Real DDPM reverse denoising process. x_T ~ N(0,I), iterative denoising via p_θ(x_{t-1}|x_t). 12-frame dense visualization."
    }
    meta_path = os.path.join(output_dir, "ddpm-reverse-denoising-meta.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Denoising image saved: {output_path}")
    print(f"Metadata saved: {meta_path}")

if __name__ == "__main__":
    main()
