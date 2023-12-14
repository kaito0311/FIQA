import argparse
import logging
import os
from threading import local
import time

import torch
import mlflow
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

import losses
from models.head import Head_Cls
from config import config as cfg
from models.imintv5 import ONNX_IMINT
from utils.utils import cosine_lr, get_lr
from dataset import MXFaceDataset, FaceDataset, FaceDatasetImage
from backbones.iresnet_imintv5 import iresnet160
from backbones.iresnet import iresnet100, iresnet50
from models.vit import VisionTransformer
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint


torch.backends.cudnn.benchmark = True
torch.set_num_threads(4)


class VIT_FIQA(torch.nn.Module):
    def __init__(self, pretrained_vit=None, pretrained_head=None, freeze_backbone=True) -> None:
        super().__init__()
        self.backbone_vit = VisionTransformer(
            input_size=224, patch_size=32, in_channels=3, dim=768, embedding_size=512,
            depth=12, num_heads=12, drop_path_rate=0.1, using_checkpoint=False)
        self.head = Head_Cls(512, 1)
        self.sigmoid = torch.sigmoid

        if pretrained_head is not None:
            print("[INFO] Loading pretrained : ", pretrained_head)
            self.head.load_state_dict(torch.load(pretrained_head))

        if pretrained_vit is not None:
            print("[INFO] Loading pretrained : ", pretrained_vit)
            self.backbone_vit.load_state_dict(torch.load(pretrained_vit))

        if freeze_backbone:
            self.backbone_vit.eval()
            for p in self.backbone_vit.parameters():
                p.requires_grad = False

    def forward(self, x):
        features = self.backbone_vit(x)
        output = self.head(features)
        return self.sigmoid(output)


def train():
    if not os.path.exists(cfg.output):
        os.makedirs(cfg.output)
    else:
        time.sleep(2)

    log_root = logging.getLogger()
    init_logging(log_root, 0, cfg.output)

    trainset = FaceDatasetImage(
        root_dir="/home1/webface_260M/unzip_folder/WebFace260M",
        path_diction_name=cfg.dict_name_features,
        path_list_name=cfg.path_list_name,
        path_list_id=cfg.path_list_id,
        is_save_listed=True
    )
    dataloader = DataLoader(trainset, cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, drop_last=True)

    model_fiqa = VIT_FIQA(pretrained_vit="pretrained/FP16-ViT-B-32.pt",
                          pretrained_head=None, freeze_backbone=False)

    imint_backbone = ONNX_IMINT(cfg.path_backbone_imint)
    
    if cfg.resume_vit is not None:
        model_fiqa.load_state_dict(torch.load(cfg.resume_vit))

    model_fiqa.to(cfg.device)
    model_fiqa.train()

    # Loss functions
    criterion = CrossEntropyLoss()
    criterion_qs = torch.nn.L1Loss()
    rank_loss_func = torch.nn.BCELoss()

    FR_loss = losses.CR_FIQA_LOSS_COSINE(
        path_mean_feature=cfg.path_mean_feature,
        path_list_mean_cosine=cfg.path_list_mean_cosine,
        path_list_std_cosine=cfg.path_list_std_cosine,
        device=cfg.device,
        s=64.0,
        m=0.5,
    )

    start_epoch = 0
    total_step = int(len(trainset) / cfg.batch_size * cfg.num_epoch)

    opt_head = torch.optim.AdamW(
        model_fiqa.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler_header = cosine_lr(opt_head, cfg.lr, 0, total_step)

    rank = 0
    world_size = 1
    # callback_verification = CallBackVerification(cfg.eval_step, rank, cfg.val_targets, cfg.rec)
    callback_logging = CallBackLogging(
        50, rank, total_step, cfg.batch_size, world_size, writer=None)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)

    loss = AverageMeter()
    global_step = cfg.global_step

    for epoch in range(start_epoch, cfg.num_epoch):
        for index, (images_numpy, images_tensor, labels) in enumerate(dataloader):
            global_step += 1
            lr = scheduler_header(global_step)

            images_tensor = images_tensor.to(cfg.device)
            labels = labels.to(cfg.device)

            qs = model_fiqa(images_tensor)
            
            features_imint = imint_backbone(images_numpy)
            features_imint = torch.Tensor(features_imint.astype(np.float32)).to(cfg.device)
            _, _, ccs, nnccs = FR_loss(features_imint, labels)

            def prev_sub(q):
                prev = torch.empty_like(q.reshape(1, -1))
                prev[0][0:-1] = q.reshape(1, -1).clone()[0][1:]
                prev[0][-1] = q.reshape(1, -1).clone()[0][0]
                return prev

            ccs_sub1 = prev_sub(ccs)  # shape (1, bs)
            y_truth = torch.where(
                (ccs_sub1[0] - ccs.reshape(1, -1)[0]) < 0, 0.0, 1.0)

            qs_sub = prev_sub(qs)
            y_pred = qs_sub[0] - qs.reshape(1, -1)[0]
            y_pred = torch.exp(y_pred)
            y_pred = y_pred / (y_pred + 1)
            bce_loss = rank_loss_func(y_pred, y_truth)

            loss_qs = criterion_qs(qs, torch.sigmoid(ccs))

            loss_v = 4 * bce_loss + 10 * loss_qs
            loss_v.backward()

            clip_grad_norm_(model_fiqa.parameters(), max_norm=5, norm_type=2)
            
            opt_head.step()

            opt_head.zero_grad()

            loss.update(loss_v.item(), 1)

            callback_logging(global_step, loss, epoch, 0,
                             loss_qs, get_lr(opt_head))
            if global_step % 10 == 0:
                print("ref_score: ", torch.sigmoid(ccs)[:10])
                # print("loss KL: ", loss_kl)
                print("qs score: ", (qs)[:10])
                print("bce loss: ", bce_loss)

            # callback_verification(global_step, backbone)
            if global_step % 100 == 0:
                callback_checkpoint(global_step, None, model_fiqa)

        callback_checkpoint(global_step, None, model_fiqa)

train()