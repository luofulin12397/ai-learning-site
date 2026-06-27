"""
DDPM Forward Diffusion — 精确数学可视化
=========================================
基于 Ho et al. 2020 "Denoising Diffusion Probabilistic Models"

公式: x_t = √ᾱₜ · x₀ + √(1-ᾱₜ) · ε
其中 ᾱₜ = ∏(1-βᵢ), βₜ 为噪声调度 (cosine schedule from Improved DDPM)

生成 6 帧 (t=0, 200, 400, 600, 800, 1000)
展示从清晰原图到纯噪声的马尔可夫正向扩散过程。每帧使用相同的 ε 随机种子。
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io, os, sys, json

# ── 1. 创建测试原图 x₀ ──────────────────────────────────────────
def make_test_image(w=256, h=256):
    """创建一张有清晰结构的测试图：彩色几何体 + 网格"""
    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # 浅灰网格背景
    for i in range(0, w, 32):
        draw.line([(i, 0), (i, h)], fill=(240, 240, 240), width=1)
    for j in range(0, h, 32):
        draw.line([(0, j), (w, j)], fill=(240, 240, 240), width=1)

    # 左上：红色方块
    draw.rectangle([(16, 16), (96, 96)], fill=(220, 60, 60))

    # 右上：蓝色圆形
    draw.ellipse([(136, 16), (240, 120)], fill=(60, 100, 220))

    # 中：绿色菱形
    pts = [(128, 72), (176, 128), (128, 184), (80, 128)]
    draw.polygon(pts, fill=(60, 180, 90))

    # 左下：黄色三角形
    pts2 = [(16, 176), (96, 240), (16, 240)]
    draw.polygon(pts2, fill=(220, 200, 40))

    # 右下：紫色半透明
    draw.rectangle([(136, 168), (240, 240)], fill=(150, 50, 200))

    # 文字
    draw.text((100, 90), "DDPM", fill=(255, 255, 255))

    return img

# ── 2. DDPM 噪声调度 (Cosine Schedule, Improved DDPM) ──────────
def cosine_beta_schedule(T=1000, s=0.008):
    """Cosine noise schedule from Nichol & Dhariwal 2021"""
    t = np.linspace(0, T, T + 1)
    α_bar_t = np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    α_bar_t = α_bar_t / α_bar_t[0]  # normalize so ᾱ₀=1
    β_t = 1 - α_bar_t[1:] / α_bar_t[:-1]
    β_t = np.clip(β_t, 0, 0.999)
    return β_t

# ── 3. 正向扩散: x_t = √ᾱₜ · x₀ + √(1-ᾱₜ) · ε ──────────────────
def forward_diffuse(x0, betas, steps):
    """对给定时间步计算 DDPM 正向扩散结果"""
    T = len(betas)
    alpha = 1 - betas
    alpha_bar = np.cumprod(alpha)

    # 固定噪声（使用相同种子保证可复现）
    rng = np.random.RandomState(42)
    eps = rng.randn(*x0.shape)

    results = []
    for t in steps:
        if t == 0:
            xt = x0.copy()
        else:
            a_bar_t = alpha_bar[t - 1]  # 0-indexed
            xt = np.sqrt(a_bar_t) * x0 + np.sqrt(1 - a_bar_t) * eps
            xt = np.clip(xt, 0, 1)  # keep in [0,1]
        results.append(xt)

    return results

# ── 4. 组装 6 帧横向可视化 ──────────────────────────────────────
def make_visualization(frames, labels, output_path, title="DDPM Forward Diffusion Process"):
    """将 6 帧拼接为横向大图，添加标签"""
    n = len(frames)
    h, w = frames[0].shape[:2]
    pad = 20  # 帧间距
    label_h = 40  # 标签高度
    title_h = 50  # 标题高度

    total_w = n * w + (n - 1) * pad + 2 * pad
    total_h = title_h + h + label_h + pad

    canvas = Image.new('RGB', (total_w, total_h), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)

    # 标题
    try:
        font_title = ImageFont.truetype("C:\\Windows\\Fonts\\segoeui.ttf", 22)
        font_label = ImageFont.truetype("C:\\Windows\\Fonts\\segoeui.ttf", 16)
    except:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), title, font=font_title)
    tw = bbox[2] - bbox[0]
    draw.text(((total_w - tw) // 2, 12), title, fill=(50, 50, 50), font=font_title)

    for i, (frame, label) in enumerate(zip(frames, labels)):
        x_offset = pad + i * (w + pad)
        y_offset = title_h

        # 粘贴帧
        frame_uint8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        frame_img = Image.fromarray(frame_uint8)
        canvas.paste(frame_img, (x_offset, y_offset))

        # 帧边框
        draw.rectangle(
            [x_offset - 1, y_offset - 1, x_offset + w, y_offset + h],
            outline=(180, 180, 180), width=1
        )

        # 标签
        bbox = draw.textbbox((0, 0), label, font=font_label)
        lw = bbox[2] - bbox[0]
        draw.text(
            (x_offset + (w - lw) // 2, title_h + h + 8),
            label, fill=(80, 80, 80), font=font_label
        )

        # 箭头（最后一帧不加）
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

# ── 5. 主流程 ───────────────────────────────────────────────────
def main():
    # 创建原图
    x0_img = make_test_image(256, 256)
    x0 = np.array(x0_img, dtype=np.float32) / 255.0

    # DDPM Cosine 噪声调度 (T=1000)
    T = 1000
    betas = cosine_beta_schedule(T)

    # 选择 6 个时间步
    steps = [0, 200, 400, 600, 800, 1000]
    labels = [
        "t = 0 (x₀)",
        "t = 200",
        "t = 400",
        "t = 600",
        "t = 800",
        "t = 1000 (纯噪声)"
    ]

    # 计算正向扩散
    frames = forward_diffuse(x0, betas, steps)

    # 输出
    output_dir = r"C:\Users\12926\WorkBuddy\2026-06-27-18-36-45\ai-learning-site\images"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ddpm-forward-diffusion.png")

    make_visualization(frames, labels, output_path)

    # 输出元信息供引用
    meta = {
        "paper": "Denoising Diffusion Probabilistic Models (Ho et al., NeurIPS 2020)",
        "formula": "x_t = √ᾱₜ · x₀ + √(1-ᾱₜ) · ε",
        "noise_schedule": "Cosine schedule (Improved DDPM, Nichol & Dhariwal 2021)",
        "T": T,
        "steps": steps,
        "image_size": "256×256",
        "note": "Forward process only. Reverse (denoising) requires trained ε_θ network."
    }
    meta_path = os.path.join(output_dir, "ddpm-forward-diffusion-meta.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ 图片已保存: {output_path}")
    print(f"✅ 元信息已保存: {meta_path}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
