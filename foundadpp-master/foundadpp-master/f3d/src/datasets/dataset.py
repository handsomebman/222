import glob
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, distributed
from torchvision import transforms
from torchvision.transforms import functional as TF
import PIL

from src.utils.mvtec3d_util import read_tiff_organized_pc, resize_organized_pc, organized_pc_to_depth_map, replace_depth_in_organized_pc
from src.utils.general_utils import SquarePad
from src.datasets.augmenter import PerlinDTDOverlay

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomRotate90or270:
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            angle = random.choice([90, 270])
            return TF.rotate(img, angle)
        return img


def build_base_transform(resize: int = 518):
    return [
        SquarePad(),
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]


def build_train_transform_staged(
    resize=518,
    use_hflip=False,
    use_vflip=False,
    use_rotate90=False,
    use_color_jitter=False,
    use_gray=False,
    use_blur=False,
    p_orient=0.3,
    p_appear=0.3,
):
    ops = []

    orient_candidates = []
    if use_hflip:
        orient_candidates.append(transforms.RandomHorizontalFlip(p=1.0))
    if use_vflip:
        orient_candidates.append(transforms.RandomVerticalFlip(p=1.0))
    if use_rotate90:
        orient_candidates.append(RandomRotate90or270(p=1.0))

    if orient_candidates:
        ops.append(
            transforms.RandomApply(
                [transforms.RandomChoice(orient_candidates)], p=p_orient
            )
        )

    appear_candidates = []
    if use_color_jitter:
        appear_candidates.append(
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.05)
        )
    if use_gray:
        appear_candidates.append(transforms.RandomGrayscale(p=1.0))
    if use_blur:
        ksz = 23 if resize >= 384 else 11
        appear_candidates.append(
            transforms.GaussianBlur(kernel_size=ksz, sigma=(0.1, 2.0))
        )

    if appear_candidates:
        ops.append(
            transforms.RandomApply(
                [transforms.RandomChoice(appear_candidates)], p=p_appear
            )
        )

    return transforms.Compose(ops)


# ---------------------------------------------------------------------------
#  Training dataset – auto‑detects 3D (MVTec‑3D) or 2D (MVTec/VisA) layout
# ---------------------------------------------------------------------------

_TRAIN_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_TRAIN_TIFF_EXTS = {".tiff"}

class TrainDataset(Dataset):
    """Train dataset that supports both
    * 3D layout: ``<root>/<class>/train/good/rgb/*.png``
    * 2D layout: ``<root>/train/<class>/*.png`` (legacy ImageFolder style)
    """

    def __init__(self, root: str, resize=518, **kwargs):
        self.root = Path(root)
        self.resize = resize
        self.transform = build_train_transform_staged(
            self.resize,
            use_hflip=kwargs.get("use_hflip", False),
            use_vflip=kwargs.get("use_vflip", False),
            use_rotate90=kwargs.get("use_rotate90", False),
            use_color_jitter=kwargs.get("use_color_jitter", False),
            use_gray=kwargs.get("use_gray", False),
            use_blur=kwargs.get("use_blur", False),
        )
        self.samples = []   # [(img_path, class_idx), ...]
        self.classes: list[str] = []
        self.class_to_idx: dict[str, int] = {}

        if not self._load_3d():
            self._load_2d()

        print(f"Totally {len(self.samples)} samples will be trained.")

    # -- 3D layout ----------------------------------------------------------
    def _load_3d(self) -> bool:
        found = False
        for class_dir in sorted(filter(Path.is_dir, self.root.iterdir())):
            if class_dir.name.startswith("."):
                continue
            rgb_dir = class_dir / "train" / "good" / "rgb"
            xyz_dir = class_dir / "train" / "good" / "xyz"
            if not rgb_dir.is_dir():
                continue
            imgs = sorted(
                p for p in rgb_dir.iterdir()
                if p.is_file() and p.suffix.lower() in _TRAIN_IMG_EXTS
            )
            tiffs = sorted(
                p for p in xyz_dir.iterdir()
                if p.is_file() and p.suffix.lower() in _TRAIN_TIFF_EXTS
            )
            if not imgs:
                continue
            idx = self.class_to_idx.setdefault(class_dir.name, len(self.classes))
            if idx == len(self.classes):
                self.classes.append(class_dir.name)
            for img, tiff in zip(imgs, tiffs):
                self.samples.append(([str(img), str(tiff)], idx))
            found = True
        return found

    # -- 2D layout ----------------------------------------------------------
    def _load_2d(self) -> bool:
        train_dir = self.root / "train"
        if not train_dir.is_dir():
            raise FileNotFoundError(
                f"Neither 3D layout (<root>/<class>/train/good/rgb/) nor "
                f"2D layout (<root>/train/<class>/) found under {self.root}"
            )
        found = False
        for class_dir in sorted(filter(Path.is_dir, train_dir.iterdir())):
            idx = self.class_to_idx.setdefault(class_dir.name, len(self.classes))
            if idx == len(self.classes):
                self.classes.append(class_dir.name)
            for p in sorted(class_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in _TRAIN_IMG_EXTS:
                    self.samples.append((str(p), idx))
                    found = True
        return found

    def __getitem__(self, index):
        path_train, target = self.samples[index]
        img_path, tiff_path = path_train

        rgb = PIL.Image.open(img_path).convert("RGB")
        rgb = self.transform(rgb)

        organized_pc = read_tiff_organized_pc(tiff_path)
        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        zzz = resize_organized_pc(depth_map_3channel, target_height=self.resize, target_width=self.resize)
        xyz = resize_organized_pc(organized_pc, target_height=self.resize, target_width=self.resize)
        xyz = xyz.clone().detach().float()

        return {
            "rgb": rgb,
            "xyz": xyz,
            "zzz": zzz,
            "target": target,
            "path_train": path_train,
        }

    def __len__(self):
        return len(self.samples)


class AugmentedTrainDataset(Dataset):
    """Train dataset that supports both
    * 3D layout: ``<root>/<class>/train/good/rgb/*.png``
    * 2D layout: ``<root>/train/<class>/*.png`` (legacy ImageFolder style)
    """

    def __init__(self, root: str, resize=518, **kwargs):
        self.root = Path(root)
        self.resize = resize
        self.base_transform = transforms.Compose(build_base_transform(resize))
        self.transform = build_train_transform_staged(
            self.resize,
            use_hflip=kwargs.get("use_hflip", False),
            use_vflip=kwargs.get("use_vflip", False),
            use_rotate90=kwargs.get("use_rotate90", False),
            use_color_jitter=kwargs.get("use_color_jitter", False),
            use_gray=kwargs.get("use_gray", False),
            use_blur=kwargs.get("use_blur", False),
        )

        self.augmenter = PerlinDTDOverlay()

        self.samples = []   # [(img_path, class_idx), ...]
        self.classes: list[str] = []
        self.class_to_idx: dict[str, int] = {}

        if not self._load_3d():
            self._load_2d()

        print(f"Totally {len(self.samples)} samples will be trained.")

    # -- 3D layout ----------------------------------------------------------
    def _load_3d(self) -> bool:
        found = False
        for class_dir in sorted(filter(Path.is_dir, self.root.iterdir())):
            if class_dir.name.startswith("."):
                continue
            rgb_dir = class_dir / "train" / "good" / "rgb"
            xyz_dir = class_dir / "train" / "good" / "xyz"
            if not rgb_dir.is_dir():
                continue
            imgs = sorted(
                p for p in rgb_dir.iterdir()
                if p.is_file() and p.suffix.lower() in _TRAIN_IMG_EXTS
            )
            tiffs = sorted(
                p for p in xyz_dir.iterdir()
                if p.is_file() and p.suffix.lower() in _TRAIN_TIFF_EXTS
            )
            if not imgs:
                continue
            idx = self.class_to_idx.setdefault(class_dir.name, len(self.classes))
            if idx == len(self.classes):
                self.classes.append(class_dir.name)
            for img, tiff in zip(imgs, tiffs):
                self.samples.append(([str(img), str(tiff)], idx))
            found = True
        return found

    # -- 2D layout ----------------------------------------------------------
    def _load_2d(self) -> bool:
        train_dir = self.root / "train"
        if not train_dir.is_dir():
            raise FileNotFoundError(
                f"Neither 3D layout (<root>/<class>/train/good/rgb/) nor "
                f"2D layout (<root>/train/<class>/) found under {self.root}"
            )
        found = False
        for class_dir in sorted(filter(Path.is_dir, train_dir.iterdir())):
            idx = self.class_to_idx.setdefault(class_dir.name, len(self.classes))
            if idx == len(self.classes):
                self.classes.append(class_dir.name)
            for p in sorted(class_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in _TRAIN_IMG_EXTS:
                    self.samples.append((str(p), idx))
                    found = True
        return found

    def __getitem__(self, index):
        path_train, target = self.samples[index]
        img_path, tiff_path = path_train

        rgb = PIL.Image.open(img_path).convert("RGB")
        rgb = self.transform(rgb)
        aug_rgb, _ = self.augmenter(rgb)
        rgb = self.base_transform(rgb)
        aug_rgb = self.base_transform(aug_rgb)

        organized_pc = read_tiff_organized_pc(tiff_path)
        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        depth_map_3channel = ((depth_map_3channel - np.min(depth_map_3channel))/(np.max(depth_map_3channel) - np.min(depth_map_3channel))*255.0).astype(np.uint8)

        depth_map_3channel = PIL.Image.fromarray(depth_map_3channel).convert("RGB")
        depth_map_3channel = self.transform(depth_map_3channel)
        aug_depth_map_3channel, _ = self.augmenter(depth_map_3channel)
        zzz = self.base_transform(depth_map_3channel)
        aug_zzz = self.base_transform(aug_depth_map_3channel)

        xyz = resize_organized_pc(organized_pc, target_height=self.resize, target_width=self.resize)
        xyz = xyz.clone().detach().float()

        return {
            "rgb": rgb,
            "xyz": xyz,
            "zzz": zzz,
            "aug_rgb": aug_rgb,
            "aug_zzz": aug_zzz,
            "target": target,
            "path_train": path_train,
        }

    def __len__(self):
        return len(self.samples)


# ---------------------------------------------------------------------------
#  Test dataset – auto‑detects 3D (MVTec‑3D) or 2D (MVTec/VisA) layout
# ---------------------------------------------------------------------------

class TestDataset(Dataset):
    def __init__(
        self,
        source,
        classname,
        resize=518,
        datasetname="mvtec",
        **kwargs,
    ):
        super().__init__()
        self.source = Path(source)
        self.classname = classname
        self.datasetname = datasetname
        self.resize = resize
        self.transform_img = transforms.Compose(build_base_transform(resize))
        self.transform_mask = transforms.Compose(
            [
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
            ]
        )
        self.imagesize = (3, resize, resize)
        self.data_to_iterate = self._get_image_data()

    def __getitem__(self, idx):
        classname, anomaly, image_path, tiff_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)
        
        organized_pc = read_tiff_organized_pc(tiff_path)
        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        depth_map_3channel = ((depth_map_3channel - np.min(depth_map_3channel))/(np.max(depth_map_3channel) - np.min(depth_map_3channel))*255.0).astype(np.uint8)

        depth_map_3channel = PIL.Image.fromarray(depth_map_3channel).convert("RGB")
        resized_depth_map_3channel = self.transform_img(depth_map_3channel)
        
        resized_organized_pc = resize_organized_pc(organized_pc, target_height=self.resize, target_width=self.resize)
        resized_organized_pc = resized_organized_pc.clone().detach().float()

        if mask_path is not None:
            mask = PIL.Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "rgb": image,
            "zzz": resized_depth_map_3channel,
            "xyz": resized_organized_pc,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly not in ("good", "ok")),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def _get_image_data(self):
        data_to_iterate = []
        classpath = self.source / self.classname / "test"
        if not classpath.is_dir():
            return data_to_iterate

        # Heuristic: if the first anomaly folder contains an ``rgb`` subdir,
        # we treat the whole class as 3D (MVTec‑3D) layout.
        is_3d = False
        for anomaly_dir in filter(Path.is_dir, classpath.iterdir()):
            if (anomaly_dir / "rgb").is_dir():
                is_3d = True
                break

        for anomaly_dir in sorted(filter(Path.is_dir, classpath.iterdir())):
            anomaly = anomaly_dir.name

            if is_3d:
                # 3D layout: test/<anomaly>/rgb/  +  test/<anomaly>/gt/
                img_dir = anomaly_dir / "rgb"
                tiff_dir = anomaly_dir / "xyz"
                mask_dir = anomaly_dir / "gt"
                if not img_dir.is_dir():
                    continue
                img_files = sorted(str(p) for p in img_dir.iterdir() if p.is_file())
                tiff_files = sorted(str(p) for p in tiff_dir.iterdir() if p.is_file())
                for img_path, tiff_path in zip(img_files, tiff_files):
                    if anomaly != "good":
                        mask_path = (
                            str(mask_dir / Path(img_path).name)
                            if mask_dir.is_dir()
                            else None
                        )
                    else:
                        mask_path = None
                    data_to_iterate.append([self.classname, anomaly, img_path, tiff_path, mask_path])
            else:
                # 2D layout: test/<anomaly>/img.png  +  ground_truth/<anomaly>/mask.png
                img_files = sorted(
                    str(p) for p in anomaly_dir.iterdir() if p.is_file()
                )
                maskpath = self.source / self.classname / "ground_truth" / anomaly

                for i, img_path in enumerate(img_files):
                    if self.datasetname == "mvtec":
                        if anomaly != "good":
                            mask_files = sorted(
                                str(p) for p in maskpath.iterdir() if p.is_file()
                            )
                            mask_path = mask_files[i] if i < len(mask_files) else None
                        else:
                            mask_path = None
                    elif self.datasetname == "visa":
                        if anomaly != "ok":
                            mask_files = sorted(
                                str(p) for p in maskpath.iterdir() if p.is_file()
                            )
                            mask_path = mask_files[i] if i < len(mask_files) else None
                        else:
                            mask_path = None
                    else:
                        mask_path = None

                    data_to_iterate.append(
                        [self.classname, anomaly, img_path, mask_path]
                    )

        return data_to_iterate


class TrainMultiModalDataset(Dataset):
    def __init__(self, root: str, resize=518, **kwargs):
        self.root = Path(root)
        self.resize = resize
        self.transform = build_train_transform_staged(
            self.resize,
            use_hflip=kwargs.get("use_hflip", False),
            use_vflip=kwargs.get("use_vflip", False),
            use_rotate90=kwargs.get("use_rotate90", False),
            use_color_jitter=kwargs.get("use_color_jitter", False),
            use_gray=kwargs.get("use_gray", False),
            use_blur=kwargs.get("use_blur", False),
        )
        self.samples = []   # [(img_path, class_idx), ...]
        self.classes: list[str] = []
        self.class_to_idx: dict[str, int] = {}

        if not self._load_3d():
            raise FileNotFoundError(
                f"No (<root>/<class>/train/good/rgb/) under {self.root} "
            )

        print(f"Totally {len(self.samples)} samples will be trained.")

    def _load_3d(self) -> bool:
        found = False
        for class_dir in sorted(filter(Path.is_dir, self.root.iterdir())):
            if class_dir.name.startswith("."):
                continue
            rgb_dir = class_dir / "train" / "good" / "rgb"
            xyz_dir = class_dir / "train" / "good" / "xyz"
            if not rgb_dir.is_dir():
                continue
            imgs = sorted(
                p for p in rgb_dir.iterdir()
                if p.is_file() and p.suffix.lower() in _TRAIN_IMG_EXTS
            )
            tiffs = sorted(
                p for p in xyz_dir.iterdir()
                if p.is_file() and p.suffix.lower() in _TRAIN_TIFF_EXTS
            )
            if not imgs:
                continue
            idx = self.class_to_idx.setdefault(class_dir.name, len(self.classes))
            if idx == len(self.classes):
                self.classes.append(class_dir.name)
            for img, tiff in zip(imgs, tiffs):
                self.samples.append(([str(img), str(tiff)], idx))
            found = True
        return found

    def __getitem__(self, index):
        path_train, target = self.samples[index]
        img_path, tiff_path = path_train

        rgb = PIL.Image.open(img_path).convert("RGB")
        rgb = self.transform(rgb)

        organized_pc = read_tiff_organized_pc(tiff_path)
        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        zzz = resize_organized_pc(depth_map_3channel, target_height=self.resize, target_width=self.resize)
        xyz = resize_organized_pc(organized_pc, target_height=self.resize, target_width=self.resize)
        xyz = xyz.clone().detach().float()

        return {
            "rgb": rgb,
            "xyz": xyz,
            "zzz": zzz,
            "target": target,
            "path_train": path_train,
        }

    def __len__(self):
        return len(self.samples)


# ---------------------------------------------------------------------------
#  DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(
    mode: str,
    root: str,
    batch_size: int,
    pin_mem: bool = True,
    **kwargs,
):
    """Return (dataset, dataloader, sampler).

    Parameters
    ----------
    mode : str
        "train" | "test"
    root : str
        For train: dataset root that either follows 3D layout
        (``<class>/train/good/rgb/``) or 2D layout (``train/<class>/``).
        For test: dataset root that contains class folders.
    **kwargs : dict
        Extra arguments forwarded to the respective dataset constructor.
    """

    if mode == "train":
        dataset = AugmentedTrainDataset(root=root, **kwargs)
        sampler = distributed.DistributedSampler(dataset)
        drop_last = True
    elif mode == "test":
        dataset = TestDataset(source=root, **kwargs)
        sampler = None
        drop_last = False
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and mode == "test"),
        pin_memory=pin_mem,
        drop_last=drop_last,
    )

    return dataset, dataloader, sampler
