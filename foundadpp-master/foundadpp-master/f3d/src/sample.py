from __future__ import annotations
import sys
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf
import hydra


def load_config_file(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # noqa
        except ModuleNotFoundError:
            sys.exit("PyYAML is required for YAML configs. Install via `pip install pyyaml`.")
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}
    else:
        return json.loads(path.read_text())


def _find_paired_samples(
    category_dir: Path,
    train_subpaths: Tuple[str, ...],
    rgb_exts: Tuple[str, ...],
    xyz_ext: str,
) -> Tuple[List[Tuple[Path, Path]], str] | Tuple[None, None]:
    """Find paired RGB–XYZ samples under candidate sub‑paths.

    Returns
    -------
    paired : list of (rgb_path, xyz_path)
    chosen_subpath : the first ``train_subpaths`` entry that yielded samples.
    """
    for sub in train_subpaths:
        candidate = category_dir / sub
        if not candidate.is_dir():
            continue

        rgb_dir = candidate / "rgb"
        xyz_dir = candidate / "xyz"
        if not rgb_dir.is_dir() or not xyz_dir.is_dir():
            continue

        rgb_files = [
            p for p in rgb_dir.iterdir()
            if p.is_file() and p.suffix.lower() in rgb_exts
        ]
        if not rgb_files:
            continue

        paired = []
        for rgb_path in rgb_files:
            xyz_path = xyz_dir / (rgb_path.stem + xyz_ext)
            if xyz_path.is_file():
                paired.append((rgb_path, xyz_path))

        if paired:
            return paired, sub

    return None, None


def _copy_directory_flat(src_dir: Path, dst_dir: Path, allowed_exts: Tuple[str, ...] | None = None) -> int:
    """Copy all files from *src_dir* to *dst_dir*.  Return number of files copied."""
    if not src_dir.is_dir():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for p in src_dir.iterdir():
        if not p.is_file():
            continue
        if allowed_exts is not None and p.suffix.lower() not in allowed_exts:
            continue
        shutil.copy2(p, dst_dir / p.name)
        count += 1
    return count


def sample_images(
    source_root: Path,
    target_root: Path,
    num_samples: int,
    train_subpaths: Tuple[str, ...],
    rgb_exts: Tuple[str, ...] = (".png",),
    xyz_ext: str = ".tiff",
    rename_images: bool = True,
    copy_test_set: bool = False,
    copy_validation_set: bool = False,
    copy_metadata: bool = True,
) -> None:
    """Sample few‑shot paired RGB–XYZ training data from an MVTec‑3D‑like dataset.

    Expected source layout::

        <category>/
            train/good/rgb/*.png
            train/good/xyz/*.tiff
            test/<anomaly>/rgb/*.png
            test/<anomaly>/xyz/*.tiff
            test/<anomaly>/gt/*.png
            validation/good/rgb/*.png
            validation/good/xyz/*.tiff
            calibration/camera_parameters.json
            class_ids.json
            readme.txt
            license.txt

    The *train* split is randomly sub‑sampled; *test*, *validation* and metadata
    are copied verbatim when the corresponding flags are enabled.
    """
    rgb_exts = tuple(ext.lower() for ext in rgb_exts)
    xyz_ext = xyz_ext.lower()

    target_root.mkdir(parents=True, exist_ok=True)

    for category_dir in filter(Path.is_dir, source_root.iterdir()):
        cat_name = category_dir.name
        # Skip hidden directories and stray symlinks at the root level.
        if cat_name.startswith("."):
            continue

        # ------------------------------------------------------------------
        # 1. Locate paired training samples
        # ------------------------------------------------------------------
        paired, chosen_subpath = _find_paired_samples(
            category_dir, train_subpaths, rgb_exts, xyz_ext
        )
        if not paired:
            print(f"[skip] {cat_name}: no paired RGB–XYZ found in {list(train_subpaths)}")
            continue

        # ------------------------------------------------------------------
        # 2. Random sub‑sample
        # ------------------------------------------------------------------
        k = min(num_samples, len(paired))
        random.shuffle(paired)
        selected = paired[:k]

        # ------------------------------------------------------------------
        # 3. Write sampled train split
        # ------------------------------------------------------------------
        train_rgb_dir = target_root / cat_name / chosen_subpath / "rgb"
        train_xyz_dir = target_root / cat_name / chosen_subpath / "xyz"
        train_rgb_dir.mkdir(parents=True, exist_ok=True)
        train_xyz_dir.mkdir(parents=True, exist_ok=True)

        # Support incremental sampling (resume from existing files).
        existing_rgb = [
            p for p in train_rgb_dir.iterdir()
            if p.is_file() and p.suffix.lower() in rgb_exts
        ]
        start_idx = len(existing_rgb)

        for i, (rgb_src, xyz_src) in enumerate(selected):
            if rename_images:
                stem = f"{start_idx + i:03d}"
                rgb_name = stem + rgb_src.suffix.lower()
                xyz_name = stem + xyz_src.suffix.lower()
            else:
                rgb_name = rgb_src.name
                xyz_name = xyz_src.name

            shutil.copy2(rgb_src, train_rgb_dir / rgb_name)
            shutil.copy2(xyz_src, train_xyz_dir / xyz_name)

        print(
            f"[✓] {cat_name}: sampled {k}/{len(paired)} paired images "
            f"from '{chosen_subpath}'"
        )

        # ------------------------------------------------------------------
        # 4. Validation set (verbatim copy)
        # ------------------------------------------------------------------
        if copy_validation_set:
            val_root = category_dir / "validation" / "good"
            n_rgb = _copy_directory_flat(
                val_root / "rgb", target_root / cat_name / "validation" / "good" / "rgb", rgb_exts
            )
            n_xyz = _copy_directory_flat(
                val_root / "xyz", target_root / cat_name / "validation" / "good" / "xyz", (xyz_ext,)
            )
            if n_rgb or n_xyz:
                print(f"    validation/good copied (rgb={n_rgb}, xyz={n_xyz})")

        # ------------------------------------------------------------------
        # 5. Test set (all anomalies + good, with rgb/xyz/gt)
        # ------------------------------------------------------------------
        if copy_test_set:
            test_src = category_dir / "test"
            if test_src.is_dir():
                copied_anomalies = 0
                for anomaly_dir in sorted(filter(Path.is_dir, test_src.iterdir())):
                    anomaly_name = anomaly_dir.name
                    dst_base = target_root / cat_name / "test" / anomaly_name

                    _copy_directory_flat(anomaly_dir / "rgb", dst_base / "rgb")
                    _copy_directory_flat(anomaly_dir / "xyz", dst_base / "xyz", (xyz_ext,))
                    _copy_directory_flat(anomaly_dir / "gt", dst_base / "gt")
                    copied_anomalies += 1

                if copied_anomalies:
                    print(f"    test set copied ({copied_anomalies} anomaly types)")

        # ------------------------------------------------------------------
        # 6. Metadata (calibration, class ids, readme, license)
        # ------------------------------------------------------------------
        if copy_metadata:
            for rel_path in (
                "calibration/camera_parameters.json",
                "class_ids.json",
                "readme.txt",
                "license.txt",
            ):
                src = category_dir / rel_path
                if src.is_file():
                    dst = target_root / cat_name / rel_path
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)


# ------------------------------------------------------------------------------
# Configuration & CLI entry‑point
# ------------------------------------------------------------------------------

@dataclass
class SamplerCfg:
    source: str = ""               # Required
    target: str = ""               # Required

    num_samples: int = 1
    train_subpaths: List[str] = field(default_factory=lambda: ["train/good"])
    rgb_exts: List[str] = field(default_factory=lambda: [".png"])
    xyz_ext: str = ".tiff"
    rename_images: bool = True

    # Whether to copy splits that are NOT sampled (test/val) and metadata.
    copy_test_set: bool = True
    copy_validation_set: bool = True
    copy_metadata: bool = True

    seed: int | None = None
    user_config: str = ""


def _finalize_cfg(cfg: DictConfig) -> Dict[str, Any]:
    if cfg.user_config:
        ext = load_config_file(Path(cfg.user_config))
        cfg = OmegaConf.merge(OmegaConf.structured(SamplerCfg), cfg, ext)

    source = cfg.get("source")
    target = cfg.get("target")
    if not source or not target:
        sys.exit(
            "Both 'source' and 'target' must be specified. "
            "Provide via CLI (source=..., target=...) or user_config=..."
        )

    return OmegaConf.to_container(cfg, resolve=True)


@hydra.main(version_base="1.3.2", config_path="/home/index/project/p260402/f3d/configs", config_name="sample_few_shot")
def main(cfg: DictConfig) -> None:
    cfgd = _finalize_cfg(cfg)
    random.seed(int(cfgd["seed"]))

    sample_images(
        source_root=Path(cfgd["source"]),
        target_root=Path(cfgd["target"]),
        num_samples=int(cfgd["num_samples"]),
        train_subpaths=tuple(cfgd["train_subpaths"]),
        rgb_exts=tuple(cfgd["rgb_exts"]),
        xyz_ext=str(cfgd["xyz_ext"]),
        rename_images=bool(cfgd["rename_images"]),
        copy_test_set=bool(cfgd.get("copy_test_set", True)),
        copy_validation_set=bool(cfgd.get("copy_validation_set", True)),
        copy_metadata=bool(cfgd.get("copy_metadata", True)),
    )


if __name__ == "__main__":
    main()
