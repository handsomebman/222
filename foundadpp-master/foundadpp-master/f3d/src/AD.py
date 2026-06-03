import os
import logging
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
from matplotlib import cm, pyplot as plt
from PIL import Image
from sklearn.linear_model import SGDOneClassSVM

from src.datasets.dataset import build_dataloader
from src.utils.metrics import (
    calculate_pro,
    compute_imagewise_retrieval_metrics,
    compute_pixelwise_retrieval_metrics,
)
from src.helper import save_segmentation_grid
from src.utils.logging import CSVLogger
from src.foundad import VisionModule          

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("evaluator")


def _build_model(meta: Dict[str, Any]) -> VisionModule:
    return VisionModule(
        model_name=meta["model"],
        pred_depth=meta["pred_depth"],
        pred_emb_dim=meta["pred_emb_dim"],
        if_pe=meta.get("if_pred_pe", True),
        feat_normed=meta.get("feat_normed", False),
    )


@torch.inference_mode()
def _evaluate_single_ckpt(ckpt: Path, args: Dict[str, Any]) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = _build_model(args["meta"])
    state = torch.load(ckpt, map_location="cpu")
    model.predictor.load_state_dict(state["predictor"])
    if model.projector is not None:
        model.projector.load_state_dict(state["projector"])
    model.to(device)
    model.eval()

    crop = args["meta"]["crop_size"]
    n_layer = args["meta"].get("n_layer", 3)

    # error = cfg["meta"].get("loss_mode", "l2")

    dataset_name = args["data"].get("dataset", "mvtec")
    if dataset_name == 'mvtec':
        classnames = args["data"]["mvtec_classnames"] 
        K = args["testing"]["K_top_mvtec"]
    elif dataset_name == 'visa':
        classnames = args["data"]["visa_classnames"]
        K = args["testing"]["K_top_visa"]
    elif dataset_name == 'mvtec_3d':
        classnames = args["data"]["mvtec_3d_classnames"] 
        K = args["testing"]["K_top_mvtec"]
    else:
        raise NotImplementedError
    assert dataset_name in args["data"]["test_root"] # check if eval on the same dataset the ckpt trained on

    
    logger.info(f"Evaluating {ckpt.name} on {dataset_name}")
    
    os.makedirs(Path(args["logging"]["folder"]), exist_ok=True)
    csv_path = Path(args["logging"]["folder"]) / f"{args['logging']['write_tag']}_eval.csv"
    csv_logger = CSVLogger(
        csv_path,
        ("%s", "checkpoint"), ("%s", "class"),
        ("%.8f", "inst_auroc"), ("%.8f", "inst_aupr"),
        ("%.8f", "pix_auroc"),  ("%.8f", "pro_auc"),
    )

    inst_auc, inst_aupr, pix_auc, pro_auc = [], [], [], []

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    for cls in classnames:
        _, loader, _ = build_dataloader(
            mode="test",
            root=args["data"]["test_root"],
            batch_size=1,
            classname=cls,
            resize=crop,
            datasetname=dataset_name,
        )

        print(f"Evaluating {cls}...")

        patch_scores, labels = [], []
        pix_buf, img_buf, mask_buf, name_buf = [], [], [], []

        for batch in tqdm(loader):
            if args["modal"] == 'rgb':
                sample = batch["rgb"].to(device, non_blocking=True)
            elif  args["modal"] == 'xyz':
                sample = batch["xyz"].to(device, non_blocking=True)
            elif  args["modal"] == 'zzz':
                sample = batch["zzz"].to(device, non_blocking=True)
                
            mask = batch["mask"].to(device, non_blocking=True)
            paths = batch["image_path"]
            labels.extend(batch["is_anomaly"])
            name_buf.extend(batch["image_name"])
            enc = model.target_features(sample, paths, n_layer=n_layer)
            pred = model.predict(enc)

            l = F.mse_loss(enc, pred, reduction="none").mean(dim=2)

            topk = torch.topk(l, K, dim=1).values.mean(dim=1)
            patch_scores.extend(topk.cpu())
            h = w = int(math.sqrt(l.size(1)))
            pix = F.interpolate(l.view(-1,1,h,w), size=sample.shape[2:], mode="bilinear", align_corners=False)
            pix_buf.append(pix.squeeze(1).cpu()); img_buf.append(sample.cpu()); mask_buf.append(mask.cpu())

        p_np = torch.tensor(patch_scores).numpy()
        p_np = (p_np - p_np.min()) / (p_np.max() - p_np.min() + 1e-8) # normed

        pix_all = torch.cat(pix_buf)
        gmin, gmax = pix_all.min(), pix_all.max()
        pix_norm = ((pix_all - gmin) / (gmax - gmin + 1e-8)).numpy()
        mask_np  = torch.cat(mask_buf).squeeze(1).numpy()

        inst = compute_imagewise_retrieval_metrics(p_np, np.array(labels))
        pix  = compute_pixelwise_retrieval_metrics(pix_norm, mask_np)
        pro  = calculate_pro(mask_np, pix_norm,
                             max_steps=args["testing"]["max_steps"], expect_fpr=args["testing"]["expect_fpr"])

        logger.info("%s | AUROC_i %.4f | AUPR_i %.4f | AUROC_p %.4f | PRO-AUC %.4f",
                    cls, inst["auroc"], inst["aupr"], pix["auroc"], pro)
        csv_logger.log(ckpt.name, cls, inst["auroc"], inst["aupr"], pix["auroc"], pro)

        inst_auc.append(inst["auroc"]); inst_aupr.append(inst["aupr"])
        pix_auc.append(pix["auroc"]);   pro_auc.append(pro)

        # Generate visualizations
        if args["testing"].get("segmentation_vis", False):
            std_cpu, mean_cpu = std.cpu(), mean.cpu()
            imgs_un = (torch.cat(img_buf) * std_cpu + mean_cpu).permute(0,2,3,1).numpy()
            out_dir = Path(args["logging"]["folder"]) / "segmentation" / cls
            save_segmentation_grid(out_dir, name_buf, imgs_un, mask_np, pix_norm)

    logger.info("Mean | AUROC_i %.4f | AUPR_i %.4f | AUROC_p %.4f | PRO-AUC %.4f",
                np.mean(inst_auc), np.mean(inst_aupr), np.mean(pix_auc), np.mean(pro_auc))
    csv_logger.log(ckpt.name, "Mean", np.mean(inst_auc), np.mean(inst_aupr),
                   np.mean(pix_auc), np.mean(pro_auc))


@torch.inference_mode()
def _evaluate_single_ckpt_mem(ckpt: Path, cfg: Dict[str, Any]) -> None:
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = _build_model(cfg["meta"])
    state = torch.load(ckpt, map_location="cpu")
    model.predictor.load_state_dict(state["predictor"])
    if model.projector is not None:
        model.projector.load_state_dict(state["projector"])
    model.to(device)
    model.eval()

    crop = cfg["meta"]["crop_size"]
    n_layer = cfg["meta"].get("n_layer", 3)
    patch_size = cfg["meta"].get("patch_size", 16)
    knn_num = cfg["memory_bank"].get("knn_num", 7)

    H = W = int(crop / patch_size)

    dataset_name = cfg["data"].get("dataset", "mvtec")
    if dataset_name == 'mvtec':
        classnames = cfg["data"]["mvtec_classnames"] 
        topk_num = cfg["testing"]["K_top_mvtec"]
    elif dataset_name == 'visa':
        classnames = cfg["data"]["visa_classnames"]
        topk_num = cfg["testing"]["K_top_visa"]
    elif dataset_name == 'mvtec_3d':
        classnames = cfg["data"]["mvtec_3d_classnames"] 
        topk_num = cfg["testing"]["K_top_mvtec"]
    else:
        raise NotImplementedError
    assert dataset_name in cfg["data"]["test_root"] # check if eval on the same dataset the ckpt trained on

    
    logger.info(f"Evaluating {ckpt.name} on {dataset_name}")
    
    os.makedirs(Path(cfg["logging"]["folder"]), exist_ok=True)
    csv_path = Path(cfg["logging"]["folder"]) / f"{cfg['logging']['write_tag']}_eval.csv"
    csv_logger = CSVLogger(
        csv_path,
        ("%s", "checkpoint"), ("%s", "class"),
        ("%.8f", "inst_auroc"), ("%.8f", "inst_aupr"),
        ("%.8f", "pix_auroc"),  ("%.8f", "pro_auc"),
    )

    inst_auc, inst_aupr, pix_auc, pro_auc = [], [], [], []

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    # gaussian_blur = GaussianBlur(kernel_size=5, sigma=1.0)

    for cls in classnames:
        _, loader, _ = build_dataloader(
            mode="test",
            root=cfg["data"]["test_root"],
            batch_size=1,
            classname=cls,
            resize=crop,
            datasetname=dataset_name,
        )

        print(f"Evaluating {cls}...")

        patch_scores, labels = [], []
        pix_buf, img_buf, mask_buf, name_buf = [], [], [], []
        enc_lib, pred_lib = [], []

        for batch in tqdm(loader, desc='Ciallo～ (∠・ω < )⌒★'):
            img = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            paths = batch["image_path"]; labels.extend(batch["is_anomaly"]); name_buf.extend(batch["image_name"])

            enc = model.target_features(img, paths, n_layer=n_layer)
            pred = model.predict(enc)

            enc_lib.extend(enc)
            
            img_buf.extend(img.cpu())
            mask_buf.extend(mask.cpu())
            for p in pred.unbind(dim=0):
                pred_lib.extend(p)
        
        pred_lib = torch.stack(pred_lib)
        pred_lib_mean = torch.mean(pred_lib)
        pred_lib_std = torch.std(pred_lib)
        pred_lib_norm = (pred_lib - pred_lib_mean) / (pred_lib_std + 1e-8)


        for enc, pred, img in tqdm(zip(enc_lib, pred_lib,img_buf)):
            enc_norm = (enc - pred_lib_mean) / (pred_lib_std + 1e-8)

            dist = torch.cdist(enc_norm, pred_lib_norm)
            knn_dist = torch.topk(dist, knn_num, largest=False, dim=1).values.mean(dim=1)
            s_reweighted = torch.topk(knn_dist, topk_num).values.mean(dim=0)

            patch_scores.append(s_reweighted.cpu())

            s_map = knn_dist.view(1, 1, H, W)
            s_map = F.interpolate(s_map, size=img.shape[1:], mode="bilinear", align_corners=False)
            # s_map = gaussian_blur(s_map)
            pix_buf.append(s_map.squeeze(1).cpu())

        p_np = torch.tensor(patch_scores).numpy()
        p_np = (p_np - p_np.min()) / (p_np.max() - p_np.min() + 1e-8) # normed

        pix_all = torch.cat(pix_buf)
        gmin, gmax = pix_all.min(), pix_all.max()
        pix_norm = ((pix_all - gmin) / (gmax - gmin + 1e-8)).numpy()
        mask_np  = torch.cat(mask_buf).squeeze(1).numpy()

        inst = compute_imagewise_retrieval_metrics(p_np, np.array(labels))
        pix  = compute_pixelwise_retrieval_metrics(pix_norm, mask_np)
        pro  = calculate_pro(mask_np, pix_norm,
                             max_steps=cfg["testing"]["max_steps"], expect_fpr=cfg["testing"]["expect_fpr"])

        logger.info("%s | AUROC_i %.4f | AUPR_i %.4f | AUROC_p %.4f | PRO-AUC %.4f",
                    cls, inst["auroc"], inst["aupr"], pix["auroc"], pro)
        csv_logger.log(ckpt.name, cls, inst["auroc"], inst["aupr"], pix["auroc"], pro)

        inst_auc.append(inst["auroc"]); inst_aupr.append(inst["aupr"])
        pix_auc.append(pix["auroc"]);   pro_auc.append(pro)

        # Generate visualizations
        if cfg["testing"].get("segmentation_vis", False):
            std_cpu, mean_cpu = std.cpu(), mean.cpu()
            imgs_un = (torch.stack(img_buf) * std_cpu + mean_cpu).permute(0,2,3,1).numpy()
            out_dir = Path(cfg["logging"]["folder"]) / "segmentation" / cls
            save_segmentation_grid(out_dir, name_buf, imgs_un, mask_np, pix_norm)

    logger.info("Mean | AUROC_i %.4f | AUPR_i %.4f | AUROC_p %.4f | PRO-AUC %.4f",
                np.mean(inst_auc), np.mean(inst_aupr), np.mean(pix_auc), np.mean(pro_auc))
    csv_logger.log(ckpt.name, "Mean", np.mean(inst_auc), np.mean(inst_aupr),
                   np.mean(pix_auc), np.mean(pro_auc))


@torch.inference_mode()
def _evaluate_dual_ckpt_mem(ckpt: Path, cfg: Dict[str, Any]) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt_rgb = os.path.join(ckpt.parent, f'{ckpt.stem}_rgb{ckpt.suffix}' )
    ckpt_zzz = os.path.join(ckpt.parent, f'{ckpt.stem}_zzz{ckpt.suffix}' )

    rgbmodel = _build_model(cfg["meta"])
    rgbstate = torch.load(ckpt_rgb, map_location="cpu")
    rgbmodel.predictor.load_state_dict(rgbstate["predictor"])
    if rgbmodel.projector is not None:
        rgbmodel.projector.load_state_dict(rgbstate["projector"])
    rgbmodel.to(device)
    rgbmodel.eval()

    zzzmodel = _build_model(cfg["meta"])
    zzzstate = torch.load(ckpt_zzz, map_location="cpu")
    zzzmodel.predictor.load_state_dict(zzzstate["predictor"])
    if zzzmodel.projector is not None:
        zzzmodel.projector.load_state_dict(zzzstate["projector"])
    zzzmodel.to(device)
    zzzmodel.eval()

    crop = cfg["meta"]["crop_size"]
    n_layer = cfg["meta"].get("n_layer", 3)
    patch_size = cfg["meta"].get("patch_size", 16)
    knn_num = cfg["memory_bank"].get("knn_num", 7)

    H = W = int(crop / patch_size)

    dataset_name = cfg["data"].get("dataset", "mvtec")
    if dataset_name == 'mvtec':
        classnames = cfg["data"]["mvtec_classnames"] 
        topk_num = cfg["testing"]["K_top_mvtec"]
    elif dataset_name == 'visa':
        classnames = cfg["data"]["visa_classnames"]
        topk_num = cfg["testing"]["K_top_visa"]
    elif dataset_name == 'mvtec_3d':
        classnames = cfg["data"]["mvtec_3d_classnames"] 
        topk_num = cfg["testing"]["K_top_mvtec"]
    else:
        raise NotImplementedError
    assert dataset_name in cfg["data"]["test_root"] # check if eval on the same dataset the ckpt trained on

    
    logger.info(f"Evaluating {ckpt.name} on {dataset_name}")
    
    os.makedirs(Path(cfg["logging"]["folder"]), exist_ok=True)
    csv_path = Path(cfg["logging"]["folder"]) / f"{cfg['logging']['write_tag']}_eval.csv"
    csv_logger = CSVLogger(
        csv_path,
        ("%s", "checkpoint"), ("%s", "class"),
        ("%.8f", "inst_auroc"), ("%.8f", "inst_aupr"),
        ("%.8f", "pix_auroc"),  ("%.8f", "pro_auc"),
    )

    inst_auc, inst_aupr, pix_auc, pro_auc = [], [], [], []

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    for cls in classnames:
        _, loader, _ = build_dataloader(
            mode="test",
            root=cfg["data"]["test_root"],
            batch_size=1,
            classname=cls,
            resize=crop,
            datasetname=dataset_name,
        )

        print(f"Evaluating {cls}...")

        patch_scores, labels = [], []
        pix_buf, img_buf, mask_buf, name_buf = [], [], [], []
        rgb_enc_lib, rgb_pred_lib = [], []
        zzz_enc_lib, zzz_pred_lib = [], []

        for batch in tqdm(loader, desc='Ciallo～ (∠・ω < )⌒★'):
            rgb = batch["rgb"].to(device, non_blocking=True)
            zzz = batch['zzz'].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            paths = batch["image_path"]; labels.extend(batch["is_anomaly"]); name_buf.extend(batch["image_name"])

            rgb_enc = rgbmodel.target_features(rgb, paths, n_layer=n_layer)
            zzz_enc = zzzmodel.target_features(zzz, paths, n_layer=n_layer)

            rgb_pred = rgbmodel.predict(rgb_enc)
            zzz_pred = zzzmodel.predict(zzz_enc)

            rgb_enc_lib.extend(rgb_enc)
            zzz_enc_lib.extend(zzz_enc)
            
            img_buf.extend(rgb.cpu())
            mask_buf.extend(mask.cpu())
            for rgb_p, zzz_p in zip(rgb_pred.unbind(dim=0), zzz_pred.unbind(dim=0)):
                rgb_pred_lib.extend(rgb_p)
                zzz_pred_lib.extend(zzz_p)
        
        rgb_pred_lib = torch.stack(rgb_pred_lib)
        zzz_pred_lib = torch.stack(zzz_pred_lib)
        rgb_pred_lib_mean = torch.mean(rgb_pred_lib)
        zzz_pred_lib_mean = torch.mean(zzz_pred_lib)
        rgb_pred_lib_std = torch.std(rgb_pred_lib)
        zzz_pred_lib_std = torch.std(zzz_pred_lib)
        
        rgb_pred_lib_norm = (rgb_pred_lib - rgb_pred_lib_mean) / (rgb_pred_lib_std + 1e-8)
        zzz_pred_lib_norm = (zzz_pred_lib - zzz_pred_lib_mean) / (zzz_pred_lib_std + 1e-8)

        for rgb_enc, rgb_pred, zzz_enc, zzz_pred, rgb in tqdm(zip(rgb_enc_lib, rgb_pred_lib, zzz_enc_lib, zzz_pred_lib, img_buf)):
            rgb_enc_norm = (rgb_enc - rgb_pred_lib_mean) / (rgb_pred_lib_std + 1e-8)
            zzz_enc_norm = (zzz_enc - zzz_pred_lib_mean) / (zzz_pred_lib_std + 1e-8)

            rgb_dist = torch.cdist(rgb_enc_norm, rgb_pred_lib_norm)
            zzz_dist = torch.cdist(zzz_enc_norm, zzz_pred_lib_norm)
            
            rgb_knn_dist = torch.topk(rgb_dist, knn_num, largest=False, dim=1).values.mean(dim=1)
            zzz_knn_dist = torch.topk(zzz_dist, knn_num, largest=False, dim=1).values.mean(dim=1)

            fus_knn_dist = torch.stack((rgb_knn_dist, zzz_knn_dist)).amax(dim=0)

            knn_dist = rgb_knn_dist

            s_reweighted = torch.topk(knn_dist, topk_num).values.mean(dim=0)

            patch_scores.append(s_reweighted.cpu())

            s_map = knn_dist.view(1, 1, H, W)
            s_map = F.interpolate(s_map, size=rgb.shape[1:], mode="bilinear", align_corners=False)

            pix_buf.append(s_map.squeeze(1).cpu())

        p_np = torch.tensor(patch_scores).numpy()
        p_np = (p_np - p_np.min()) / (p_np.max() - p_np.min() + 1e-8) # normed

        pix_all = torch.cat(pix_buf)
        gmin, gmax = pix_all.min(), pix_all.max()
        pix_norm = ((pix_all - gmin) / (gmax - gmin + 1e-8)).numpy()
        mask_np  = torch.cat(mask_buf).squeeze(1).numpy()

        inst = compute_imagewise_retrieval_metrics(p_np, np.array(labels))
        pix  = compute_pixelwise_retrieval_metrics(pix_norm, mask_np)
        pro  = calculate_pro(mask_np, pix_norm,
                             max_steps=cfg["testing"]["max_steps"], expect_fpr=cfg["testing"]["expect_fpr"])

        logger.info("%s | AUROC_i %.4f | AUPR_i %.4f | AUROC_p %.4f | PRO-AUC %.4f",
                    cls, inst["auroc"], inst["aupr"], pix["auroc"], pro)
        csv_logger.log(ckpt.name, cls, inst["auroc"], inst["aupr"], pix["auroc"], pro)

        inst_auc.append(inst["auroc"]); inst_aupr.append(inst["aupr"])
        pix_auc.append(pix["auroc"]);   pro_auc.append(pro)

        # Generate visualizations
        if cfg["testing"].get("segmentation_vis", False):
            std_cpu, mean_cpu = std.cpu(), mean.cpu()
            imgs_un = (torch.stack(img_buf) * std_cpu + mean_cpu).permute(0,2,3,1).numpy()
            out_dir = Path(cfg["logging"]["folder"]) / "segmentation" / cls
            save_segmentation_grid(out_dir, name_buf, imgs_un, mask_np, pix_norm)

    logger.info("Mean | AUROC_i %.4f | AUPR_i %.4f | AUROC_p %.4f | PRO-AUC %.4f",
                np.mean(inst_auc), np.mean(inst_aupr), np.mean(pix_auc), np.mean(pro_auc))
    csv_logger.log(ckpt.name, "Mean", np.mean(inst_auc), np.mean(inst_aupr),
                   np.mean(pix_auc), np.mean(pro_auc))


@torch.inference_mode()
def _demo(ckpt: Path, cfg: Dict[str, Any]) -> None:
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = _build_model(cfg["meta"])
    state = torch.load(ckpt, map_location="cpu")
    model.predictor.load_state_dict(state["predictor"])
    if model.projector is not None:
        model.projector.load_state_dict(state["projector"])
    model.to(device)
    model.eval()

    crop = cfg["meta"]["crop_size"]
    n_layer = cfg["meta"].get("n_layer", 3)
    out_root = Path(cfg["logging"]["folder"]) / "heatmaps"
    out_root.mkdir(parents=True, exist_ok=True)

    dataset_name = cfg["data"].get("dataset", "mvtec")
    assert dataset_name in cfg["data"]["test_root"] # check if eval on the same dataset the ckpt trained on
    
    test_root = Path(cfg["data"]["test_root"])
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIF", "*.TIFF", "*.WEBP")
    img_paths: List[Path] = []
    for ext in exts:
        img_paths += list(test_root.rglob(ext))
    img_paths = sorted(set(img_paths))
    if not img_paths:
        raise FileNotFoundError(f"No images found under: {test_root}")
    print(f"[INFO] Found {len(img_paths)} images under {test_root}")
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    def _load_and_preprocess(path: Path):
        pil = Image.open(path).convert("RGB")

        W0, H0 = pil.size

        pil_resized = pil.resize((crop, crop), Image.BILINEAR)

        img = torch.from_numpy(np.array(pil_resized)).float() / 255.0   # [H,W,3], 0~1
        img = img.permute(2, 0, 1).unsqueeze(0).to(device)              # [1,3,H,W]
        img = (img - mean) / std

        return pil, (W0, H0), img
    
    def _to_numpy_image(t_img: torch.Tensor):
        # t_img: [1,3,H,W]
        x = (t_img * std + mean).clamp(0, 1)
        x = x[0].permute(1, 2, 0).detach().cpu().numpy()  # [H,W,3]
        return (x * 255.0).astype(np.uint8)
    
    def _save_overlay_heatmap(rgb_uint8: np.ndarray, heat: np.ndarray, save_path: Path, alpha: float = 0.5):
        """
        rgb_uint8: [H,W,3] 0~255
        heat:      [H,W]   0~1
        """
        import cv2
        H, W = heat.shape

        heat_255 = (heat * 255.0).clip(0, 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_255, cv2.COLORMAP_JET)      # BGR
        rgb_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)            # RGB->BGR
        overlay = cv2.addWeighted(heat_color, alpha, rgb_bgr, 1 - alpha, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        Image.fromarray(overlay_rgb).save(save_path)

    for i, path in enumerate(img_paths, 1):
        pil_orig, (W0, H0), img = _load_and_preprocess(path)

        enc = model.target_features(img, [str(path)], n_layer=n_layer)  # [1, P, D]
        pred = model.predict(enc)                                       # [1, P, D]

        l = F.mse_loss(enc, pred, reduction="none").mean(dim=2)         # [1, P]

        h = w = int(math.sqrt(l.size(1)))
        pix = F.interpolate(l.view(1, 1, h, w), size=img.shape[2:], mode="bilinear", align_corners=False)  # [1,1,H,W]
        pix = pix.squeeze(0).squeeze(0)  # [H,W]

        pmin, pmax = pix.min(), pix.max()
        pix_norm = (pix - pmin) / (pmax - pmin + 1e-8)                  # [H,W], 0~1

        img_uint8 = _to_numpy_image(img)                                 # [H,W,3] @ crop

        rel = path.relative_to(test_root)
        save_dir = (out_root / rel.parent)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{path.stem}_heatmap.png"

        _save_overlay_heatmap(img_uint8, pix_norm.detach().cpu().numpy(), save_path)
        print(f"[{i}/{len(img_paths)}] Saved: {save_path}")


def main(args: Dict[str, Any]) -> None:
    ckpt = Path(args["ckpt_path"])
    print(f"loading {ckpt}...")
    if args['memory_bank']['use'] == True:
        if args['modal'] == 'rgb+zzz':
            _evaluate_dual_ckpt_mem(ckpt, args)
        else:
            _evaluate_single_ckpt_mem(ckpt, args)
    else:
        _evaluate_single_ckpt(ckpt, args)
    logger.info("Finished. Metrics appended to CSV.")

if __name__ == "__main__":
    main()