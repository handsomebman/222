from __future__ import annotations

import os, sys, random, logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch import autocast

from src.utils.logging import CSVLogger, grad_logger, AverageMeter
from src.datasets.dataset import build_dataloader
from src.foundad import VisionModule
from src.helper import init_opt


_GLOBAL_SEED = 0
random.seed(42); np.random.seed(0); torch.manual_seed(0)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class ModalTrainer:
    """单模态训练器，支持 rgb 或 zzz。"""

    def __init__(self, args: Dict[str, Any], modal: str):
        self.args = args
        self.modal = modal
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)

        # ---------- model ----------
        mcfg = args["meta"]
        self.model = VisionModule(
            mcfg["model"], mcfg["pred_depth"], mcfg["pred_emb_dim"],
            if_pe=mcfg.get("if_pred_pe", True), feat_normed=mcfg.get("feat_normed", False),
        )
        self.n_layer = args["meta"].get("n_layer", 3)
        self.model.predictor.requires_grad_(True)
        if self.model.projector:
            self.model.projector.requires_grad_(True)
        self.loss_mode = args["meta"].get("loss_mode", "l2")
        logger.info(f"Loss mode {self.loss_mode}")

        # ---------- data ----------
        dcfg = args["data"]
        assert dcfg["dataset"] in dcfg["data_name"]
        _, self.loader, self.sampler = build_dataloader(
            mode="train",
            root=dcfg["train_root"],
            batch_size=dcfg["batch_size"],
            pin_mem=dcfg["pin_mem"],
            resize=mcfg["crop_size"],
            use_hflip=dcfg.get("use_hflip", False),
            use_vflip=dcfg.get("use_vflip", False),
            use_rotate90=dcfg.get("use_rotate90", False),
            use_color_jitter=dcfg.get("use_color_jitter", False),
            use_gray=dcfg.get("use_gray", False),
            use_blur=dcfg.get("use_blur", False),
        )
        self.batch_size = dcfg["batch_size"]

        # ---------- optimization ----------
        ocfg = args["optimization"]
        self.optimizer, self.scheduler, self.scaler = init_opt(
            predictor=self.model.predictor,
            wd=float(ocfg["weight_decay"]),
            lr=ocfg["lr"],
            lr_config=ocfg.get("lr_config", "const"),
            max_epoch=ocfg["epochs"],
            min_lr=ocfg.get("min_lr", 1e-6),
            warmup_epoch=ocfg.get("warmup_epoch", 5),
            step_size=ocfg.get("step_size", 300),
            gamma=ocfg.get("gamma", 0.1),
        )
        self.epochs = ocfg["epochs"]
        self.use_bf16 = mcfg["use_bfloat16"]

        # ---------- logging ----------
        lcfg: Dict[str, Any] = args.get("logging", {})
        log_dir = Path(lcfg.get("folder", "logs"))
        self.ckpt_dir = log_dir
        self.tag = lcfg.get("write_tag", "train")
        self.csv_logger = CSVLogger(
            str(self.ckpt_dir / f"{self.tag}.csv"),
            ("%d", "epoch"),
            ("%d", "itr"),
            ("%.5f", "loss"),
            ("%d", "time (ms)"),
        )

    def _loss_fn(self, h, p) -> torch.Tensor:
        if self.loss_mode == 'l2':
            return F.mse_loss(h.flatten(0, 1), p.flatten(0, 1), reduction="mean")
        elif self.loss_mode == 'smooth_l1':
            return F.smooth_l1_loss(h.flatten(0, 1), p.flatten(0, 1), reduction="mean")
        else:
            raise NotImplementedError(f"Loss mode {self.loss_mode} not implemented")

    def _save_ckpt(self, ep, step=None):
        name = f"{self.tag}-step{step}.pth.tar" if step else f"{self.tag}-ep{ep}.pth.tar"
        torch.save({
            "predictor": self.model.predictor.state_dict(),
            "projector": self.model.projector.state_dict() if self.model.projector else None,
            "epoch": ep,
            "lr": self.optimizer.param_groups[0]["lr"],
        }, self.ckpt_dir / name)

    def train(self):
        mp.set_start_method("spawn", force=True)
        gstep = 0
        for ep in tqdm(range(self.epochs), desc='Training'):
            logger.info("Epoch %d", ep + 1)
            self.sampler.set_epoch(ep)

            loss_m = AverageMeter()

            for itr, sample in enumerate(self.loader):
                x = sample[self.modal].to(self.device, non_blocking=True)
                ctx = sample[f'aug_{self.modal}'].to(self.device, non_blocking=True)
                paths = sample['path_train']

                with autocast(device_type='cuda:0', dtype=torch.bfloat16, enabled=self.use_bf16):
                    h = self.model.target_features(x, paths, n_layer=self.n_layer)
                    _, p = self.model.context_features(ctx, paths, n_layer=self.n_layer)

                loss = self._loss_fn(h, p)

                if self.use_bf16:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                grad_stats = grad_logger(self.model.predictor.named_parameters())
                self.optimizer.zero_grad()

                loss_m.update(loss.item())
                gstep += 1
                if gstep % 100 == 0:
                    self._save_ckpt(ep, gstep)
                self.csv_logger.log(ep + 1, itr, loss.item())
                if itr % 100 == 0:
                    logger.info(
                        "[E %d I %d] loss %.6f (avg %.6f) mem %.2fMB",
                        ep + 1, itr, loss.item(), loss_m.avg,
                        torch.cuda.max_memory_allocated() / 1024 ** 2,
                    )
                    if grad_stats:
                        logger.info(
                            "    grad: [%.2e %.2e] (%.2e %.2e]",
                            grad_stats.first_layer, grad_stats.last_layer,
                            grad_stats.min, grad_stats.max,
                        )
            logger.info(
                "Epoch %d complete. Avg loss %.6f, lr %.6f",
                ep + 1,
                loss_m.avg,
                self.optimizer.param_groups[0]['lr'],
            )
            if self.scheduler is not None:
                self.scheduler.step()


class VDTrainer:
    """双模态 (rgb + zzz) 训练器。"""

    def __init__(self, args: Dict[str, Any]):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # ---------- model ----------
        mcfg = args["meta"]
        self.n_layer = args["meta"].get("n_layer", 3)

        self.rgbmodel = VisionModule(
            mcfg["model"], mcfg["pred_depth"], mcfg["pred_emb_dim"],
            if_pe=mcfg.get("if_pred_pe", True), feat_normed=mcfg.get("feat_normed", False),
        )
        self.rgbmodel.predictor.requires_grad_(True)
        if self.rgbmodel.projector:
            self.rgbmodel.projector.requires_grad_(True)

        self.zzzmodel = VisionModule(
            mcfg["model"], mcfg["pred_depth"], mcfg["pred_emb_dim"],
            if_pe=mcfg.get("if_pred_pe", True), feat_normed=mcfg.get("feat_normed", False),
        )
        self.zzzmodel.predictor.requires_grad_(True)
        if self.zzzmodel.projector:
            self.zzzmodel.projector.requires_grad_(True)

        self.loss_mode = args["meta"].get("loss_mode", "l2")
        logger.info(f"Loss mode {self.loss_mode}")

        # ---------- data ----------
        dcfg = args["data"]
        assert dcfg["dataset"] in dcfg["data_name"]
        _, self.loader, self.sampler = build_dataloader(
            mode="train",
            root=dcfg["train_root"],
            batch_size=dcfg["batch_size"],
            pin_mem=dcfg["pin_mem"],
            resize=mcfg["crop_size"],
            use_hflip=dcfg.get("use_hflip", False),
            use_vflip=dcfg.get("use_vflip", False),
            use_rotate90=dcfg.get("use_rotate90", False),
            use_color_jitter=dcfg.get("use_color_jitter", False),
            use_gray=dcfg.get("use_gray", False),
            use_blur=dcfg.get("use_blur", False),
        )

        self.batch_size = dcfg["batch_size"]

        # ---------- optimization ----------
        ocfg = args["optimization"]

        self.optimizer_rgb, self.scheduler_rgb, self.scaler_rgb = init_opt(
            predictor=self.rgbmodel.predictor,
            wd=float(ocfg["weight_decay"]),
            lr=ocfg["lr"],
            lr_config=ocfg.get("lr_config", "const"),
            max_epoch=ocfg["epochs"],
            min_lr=ocfg.get("min_lr", 1e-6),
            warmup_epoch=ocfg.get("warmup_epoch", 5),
            step_size=ocfg.get("step_size", 300),
            gamma=ocfg.get("gamma", 0.1),
        )

        self.optimizer_zzz, self.scheduler_zzz, self.scaler_zzz = init_opt(
            predictor=self.zzzmodel.predictor,
            wd=float(ocfg["weight_decay"]),
            lr=ocfg["lr"],
            lr_config=ocfg.get("lr_config", "const"),
            max_epoch=ocfg["epochs"],
            min_lr=ocfg.get("min_lr", 1e-6),
            warmup_epoch=ocfg.get("warmup_epoch", 5),
            step_size=ocfg.get("step_size", 300),
            gamma=ocfg.get("gamma", 0.1),
        )

        self.epochs = ocfg["epochs"]
        self.use_bf16 = mcfg["use_bfloat16"]

        # ---------- logging ----------
        lcfg: Dict[str, Any] = args.get("logging", {})
        log_dir = Path(lcfg.get("folder", "logs"))
        self.ckpt_dir = log_dir
        self.tag = lcfg.get("write_tag", "train")
        self.csv_logger = CSVLogger(
            str(self.ckpt_dir / f"{self.tag}.csv"),
            ("%d", "epoch"),
            ("%d", "itr"),
            ("%.5f", "rgb_loss"),
            ("%.5f", "zzz_loss"),
        )

    def _loss_fn(self, h, p) -> torch.Tensor:
        if self.loss_mode == 'l2':
            return F.mse_loss(h.flatten(0, 1), p.flatten(0, 1), reduction="mean")
        elif self.loss_mode == 'smooth_l1':
            return F.smooth_l1_loss(h.flatten(0, 1), p.flatten(0, 1), reduction="mean")
        else:
            raise NotImplementedError(f"Loss mode {self.loss_mode} not implemented")

    def _save_ckpt(self, ep, step=None):
        name = f"{self.tag}-step{step}_rgb.pth" if step else f"{self.tag}-ep{ep}_rgb.pth"
        torch.save({
            "predictor": self.rgbmodel.predictor.state_dict(),
            "projector": self.rgbmodel.projector.state_dict() if self.rgbmodel.projector else None,
            "epoch": ep,
            "lr": self.optimizer_rgb.param_groups[0]["lr"],
        }, self.ckpt_dir / name)

        name = f"{self.tag}-step{step}_zzz.pth" if step else f"{self.tag}-ep{ep}_zzz.pth"
        torch.save({
            "predictor": self.zzzmodel.predictor.state_dict(),
            "projector": self.zzzmodel.projector.state_dict() if self.zzzmodel.projector else None,
            "epoch": ep,
            "lr": self.optimizer_zzz.param_groups[0]["lr"],
        }, self.ckpt_dir / name)

    def train(self):
        mp.set_start_method("spawn", force=True)
        gstep = 0
        for ep in tqdm(range(self.epochs), desc='Training'):
            logger.info("Epoch %d", ep + 1)
            self.sampler.set_epoch(ep)

            loss_m_rgb = AverageMeter()
            loss_m_zzz = AverageMeter()

            for itr, sample in enumerate(self.loader):
                rgb = sample['rgb'].to(self.device, non_blocking=True)
                zzz = sample['zzz'].to(self.device, non_blocking=True)
                paths = sample['path_train']

                ctx_rgb = sample['aug_rgb'].to(self.device, non_blocking=True)
                ctx_zzz = sample['aug_zzz'].to(self.device, non_blocking=True)

                with autocast(device_type='cuda:0', dtype=torch.bfloat16, enabled=self.use_bf16):
                    h_rgb = self.rgbmodel.target_features(rgb, paths, n_layer=self.n_layer)
                    _, p_rgb = self.rgbmodel.context_features(ctx_rgb, paths, n_layer=self.n_layer)
                    h_zzz = self.zzzmodel.target_features(zzz, paths, n_layer=self.n_layer)
                    _, p_zzz = self.zzzmodel.context_features(ctx_zzz, paths, n_layer=self.n_layer)

                loss_rgb = self._loss_fn(h_rgb, p_rgb)
                loss_zzz = self._loss_fn(h_zzz, p_zzz)

                self.optimizer_rgb.zero_grad()
                self.optimizer_zzz.zero_grad()

                if self.use_bf16:
                    self.scaler_rgb.scale(loss_rgb).backward()
                    self.scaler_rgb.step(self.optimizer_rgb)
                    self.scaler_rgb.update()
                else:
                    loss_rgb.backward()
                    self.optimizer_rgb.step()
                loss_m_rgb.update(loss_rgb.item())

                if self.use_bf16:
                    self.scaler_zzz.scale(loss_zzz).backward()
                    self.scaler_zzz.step(self.optimizer_zzz)
                    self.scaler_zzz.update()
                else:
                    loss_zzz.backward()
                    self.optimizer_zzz.step()
                loss_m_zzz.update(loss_zzz.item())

                gstep += 1
                if gstep % 100 == 0:
                    self._save_ckpt(ep, gstep)
                self.csv_logger.log(ep + 1, itr, loss_rgb.item(), loss_zzz.item())

            logger.info(
                "Epoch %d complete. Avg loss %.6f %.6f, lr %.6f lr %.6f",
                ep + 1,
                loss_m_rgb.avg,
                loss_m_zzz.avg,
                self.optimizer_rgb.param_groups[0]['lr'],
                self.optimizer_zzz.param_groups[0]['lr'],
            )
            if self.scheduler_rgb is not None:
                self.scheduler_rgb.step()
            if self.scheduler_zzz is not None:
                self.scheduler_zzz.step()


def main(args: Dict[str, Any]) -> None:
    if args is None:
        cfg_path = Path(__file__).with_name("params.yaml")
        if not cfg_path.exists():
            raise FileNotFoundError("No args provided and default parameter file does not exist")
        with open(cfg_path) as f:
            args = yaml.safe_load(f)
    modal = args.get('modal', 'rgb')
    if modal in ('rgb', 'zzz'):
        ModalTrainer(args, modal).train()
    elif modal == 'rgb+zzz':
        VDTrainer(args).train()


if __name__ == "__main__":
    main()
