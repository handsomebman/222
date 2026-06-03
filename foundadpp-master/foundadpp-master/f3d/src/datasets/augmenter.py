import math
import os
import random
import glob

import PIL
import numpy as np
import cv2
import albumentations as A
from PIL import Image

augmenter = A.Compose([
            A.GaussNoise(per_channel=True, p=0.2),
            A.SaltAndPepper(amount=(0.02, 0.1), p=0.2),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, elementwise=True, p=0.2),
            A.ShotNoise(scale_range=(0.1, 0.5), p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=0.3),
            A.MotionBlur(blur_limit=(5, 15), p=0.3),
            A.MedianBlur(blur_limit=(3, 7), p=0.3),
            A.GlassBlur(sigma=0.7, max_delta=2, iterations=2, p=0.2),
            A.Defocus(radius=(3, 10), alias_blur=(0.1, 0.5), p=0.2),
            A.ZoomBlur(max_factor=(1.05, 1.3), step_factor=(0.01, 0.03), p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.3),
            A.ChannelShuffle(p=0.1),
            A.Affine(
                scale=(0.8, 1.2),           # 缩放范围
                translate_percent=(-0.1, 0.1), # 平移范围（图像尺寸的百分比）
                rotate=(-30, 30),            # 旋转角度范围
                shear=(-10, 10),              # 剪切变换范围
                interpolation=cv2.INTER_LINEAR,
                # 关键参数：使用反射填充而非默认的黑边填充
                border_mode=cv2.BORDER_REFLECT,  # 反射模式避免黑边
                p=0.5
            ),
            A.ElasticTransform(
                alpha=50,                     # 位移幅度
                sigma=5,                      # 平滑程度
                interpolation=cv2.INTER_LINEAR,
                # 关键参数：使用反射填充
                border_mode=cv2.BORDER_REFLECT,
                p=0.3
            ),
            A.GridDistortion(
                num_steps=5,                   # 网格步数
                distort_limit=0.3,              # 扭曲幅度
                interpolation=cv2.INTER_LINEAR,
                # 关键参数：使用反射填充
                border_mode=cv2.BORDER_REFLECT,
                p=0.2
            ),
            A.OpticalDistortion(
                distort_limit=0.3,              # 畸变程度
                interpolation=cv2.INTER_LINEAR,
                # 关键参数：使用反射填充
                border_mode=cv2.BORDER_REFLECT,
                p=0.2
            ),
        ])

def lerp_np(x,y,w):
    fin_out = (y-x)*w + x
    return fin_out

def rand_perlin_2d_np(shape, res=(10, 10), fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tt = np.repeat(np.repeat(gradients,d[0],axis=0),d[1],axis=1)

    tile_grads = lambda slice1, slice2: np.repeat(np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]],d[0],axis=0),d[1],axis=1)
    dot = lambda grad, shift: (
                np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])


class PerlinDTDOverlay:
    """基于 Perlin 噪声掩码与 DTD 纹理的图像增强类。

    对输入图像：
    1. 生成同尺寸的 Perlin 噪声；
    2. 通过随机阈值分割得到二值掩码；
    3. 从 DTD 数据集中随机采样一张纹理图，经 ``augmenter`` 增强；
    4. 将增强后的纹理通过掩码为 1 的区域叠加到原图上；
    5. 返回合成图像与掩码。
    """

    def __init__(
        self,
        dtd_dir="assets/dtd/images",
        threshold_range=(0.2, 0.5),
        octaves=4,
        persistence=0.5,
        lacunarity=2.0,
    ):
        self.dtd_dir = dtd_dir
        self.threshold_range = threshold_range
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity

        self.dtd_images = glob.glob(os.path.join(dtd_dir, "*", "*.jpg"))
        if not self.dtd_images:
            raise ValueError(f"在 {dtd_dir} 中未找到 DTD 图像，请检查路径。")

    def __call__(self, image, return_info=False):
        """执行增强。

        Parameters
        ----------
        image : np.ndarray | PIL.Image.Image
            输入图像。若为 ``np.ndarray``，期望形状为 (H, W, C)；
            若为 ``PIL.Image.Image``，会自动转换为 numpy 数组。
        return_info : bool
            若为 ``True``，额外返回 Perlin 噪声图和增强后的 DTD 纹理图，
            便于调试与可视化。

        Returns
        -------
        tuple
            默认返回 ``(output, mask)``。
            当 ``return_info=True`` 时返回 ``(output, mask, noise, dtd_aug)``。
        """
        # 统一转换为 numpy 数组
        if isinstance(image, Image.Image):
            img = np.array(image)
        elif isinstance(image, np.ndarray):
            img = image.copy()
        else:
            raise TypeError(f"不支持的输入类型: {type(image)}，请传入 np.ndarray 或 PIL.Image。")

        if img.ndim == 2:
            img = img[:, :, None]

        h, w = img.shape[:2]

        # 1. 生成 Perlin 噪声并阈值分割得到掩码
        noise = rand_perlin_2d_np((h, w), (10, 10))
        threshold = random.uniform(*self.threshold_range)
        mask = (noise > threshold).astype(np.float32)  # (H, W), values in {0, 1}

        # 2. 随机采样 DTD 纹理图
        dtd_path = random.choice(self.dtd_images)
        dtd_img = Image.open(dtd_path).convert("RGB")
        dtd_img = np.array(dtd_img)

        # 3. 使用 augmenter 进行图像增强
        dtd_aug = augmenter(image=dtd_img)["image"]

        # 4. resize 到与输入图像相同尺寸
        dtd_aug = cv2.resize(dtd_aug, (w, h))
        if dtd_aug.ndim == 2:
            dtd_aug = dtd_aug[:, :, None]

        # 5. 通道对齐
        in_ch = img.shape[2]
        dtd_ch = dtd_aug.shape[2]
        if in_ch != dtd_ch:
            if in_ch == 1 and dtd_ch == 3:
                dtd_aug = cv2.cvtColor(dtd_aug, cv2.COLOR_RGB2GRAY)[:, :, None]
            elif in_ch == 3 and dtd_ch == 1:
                dtd_aug = np.repeat(dtd_aug, 3, axis=2)
            else:
                raise ValueError(f"通道数不匹配: 输入 {in_ch}, DTD {dtd_ch}")

        # 6. 按掩码叠加：mask=1 的区域使用 DTD，mask=0 保留原图
        mask_expanded = mask[:, :, None] if mask.ndim == 2 else mask
        output = img.astype(np.float32) * (1.0 - mask_expanded) + dtd_aug.astype(np.float32) * mask_expanded

        # 保持原数据类型
        if img.dtype == np.uint8:
            output = np.clip(output, 0, 255).astype(np.uint8)
        else:
            output = output.astype(img.dtype)

        if return_info:
            return output, mask, noise, dtd_aug
        return PIL.Image.fromarray(output).convert("RGB"),  mask




if __name__ == "__main__":
    """简易可视化测试：使用 MVTec-3D bagel 训练集 good 图像验证 PerlinDTDOverlay。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 固定随机种子，保证可复现
    random.seed(42)
    np.random.seed(42)

    # 输入图像路径（项目根目录下的相对路径）
    test_img_path = "assets/mvtec_3d/bagel/train/good/rgb/000.png"
    save_path = "outputs/perlin_dtd_overlay_test.png"

    # 读取图像
    input_img = Image.open(test_img_path).convert("RGB")
    input_np = np.array(input_img)

    # 实例化增强器并执行（带回调试信息）
    overlay = PerlinDTDOverlay()
    output, mask, noise, dtd_aug = overlay(input_np, return_info=True)

    # 绘图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    axes[0].imshow(input_np)
    axes[0].set_title("Input Image (bagel/train/good/rgb/000.png)")
    axes[0].axis("off")

    axes[1].imshow(noise, cmap="gray")
    axes[1].set_title("Perlin Noise")
    axes[1].axis("off")

    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title(f"Binary Mask (threshold={overlay.threshold_range})")
    axes[2].axis("off")

    axes[3].imshow(dtd_aug.astype(np.uint8))
    axes[3].set_title("Augmented DTD Texture")
    axes[3].axis("off")

    axes[4].imshow(output)
    axes[4].set_title("Overlay Result")
    axes[4].axis("off")

    # 掩码叠加效果示意（半透明红色覆盖）
    overlay_vis = output.copy()
    red_mask = np.zeros_like(overlay_vis)
    red_mask[:, :, 0] = (mask * 255).astype(np.uint8)
    overlay_vis = cv2.addWeighted(overlay_vis, 0.7, red_mask, 0.3, 0)
    axes[5].imshow(overlay_vis)
    axes[5].set_title("Result with Mask Overlay (red)")
    axes[5].axis("off")

    plt.suptitle("PerlinDTDOverlay Visualization Test", fontsize=16)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"可视化结果已保存至: {save_path}")
