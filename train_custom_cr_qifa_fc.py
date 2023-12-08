import argparse
import logging
import os
from threading import local
import time

import torch
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss

import losses
from config import config as cfg
from dataset import MXFaceDataset, FaceDataset
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter, init_logging

from models.imintv5 import ONNX_IMINT
from backbones.iresnet import iresnet100, iresnet50
from backbones.iresnet_imintv5 import iresnet160

torch.backends.cudnn.benchmark = True
torch.set_num_threads(4)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
        return lr
    return _lr_adjuster


class Head_Cls(torch.nn.Module):
    def __init__(self, in_features=512, out_features=1) -> None:
        super().__init__()
        self.middle = torch.nn.Linear(in_features, 128)
        self.leaky = torch.nn.LeakyReLU(negative_slope=0.1)
        self.dropout = torch.nn.Dropout(p=0.4)
        self.qs = torch.nn.Linear(128, out_features)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.middle(x)
        x = self.leaky(x)
        x = self.dropout(x)
        return self.qs(x)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def main():
    if not os.path.exists(cfg.output):
        os.makedirs(cfg.output)
    else:
        time.sleep(2)

    log_root = logging.getLogger()

    init_logging(log_root, 0, cfg.output)

    trainset = FaceDataset(
        "feature_dir", "/home2/tanminh/FIQA/dict_name_features.npy",
        path_dict_mean_distance="/home2/tanminh/FIQA/data/diction_mean_cluster_thresh_5e-1.npy")
    dataloader = DataLoader(trainset, cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, drop_last=True)

    # backbone = iresnet160(False)
    # backbone.load_state_dict(torch.load("/home1/data/tanminh/Face_Recognize_Quaility_Assessment/r160_imintv4_statedict.pth"))

    backbone = ONNX_IMINT(
        "/home2/tanminh/FIQA/pretrained/stacking_avg_r160+ada-unnorm-stacking-ada-1.6.onnx")
    head = Head_Cls(1024, 1)

    head.cuda()

    if cfg.resume:
        print("[INFO] Resume. Loading last chk...")
        head.load_state_dict(torch.load(cfg.resume_head))

    head.train()

    FR_loss = losses.CR_FIQA_LOSS_ONTOP(
        path_mean="/home2/tanminh/FIQA/data/mean_cluster.npy",
        path_std="/home2/tanminh/FIQA/data/std_cluster.npy",
        device="cuda",
    )

    # opt_head = torch.optim.SGD(
    # params=[{'params': head.parameters()}],
    # lr=cfg.lr,
    # momentum=0.9, weight_decay=cfg.weight_decay)

    criterion = CrossEntropyLoss()

    def smooth_l1_loss(x, y, scale_loss=None, beta=0.5):
        sub = torch.abs(x - y)

        score = torch.where(sub < beta, 0.5 * sub **
                            2 / beta, sub - 0.5 * beta)

        if scale_loss is not None:
            score = score * scale_loss

        return torch.mean(score)

    criterion_qs = torch.nn.L1Loss()
    rank_loss_func = torch.nn.BCELoss() 

    start_epoch = 0
    total_step = int(len(trainset) / cfg.batch_size * cfg.num_epoch)

    opt_head = torch.optim.AdamW(
        head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler_header = cosine_lr(opt_head, cfg.lr, 0, total_step)

    rank = 0
    world_size = 1
    # callback_verification = CallBackVerification(cfg.eval_step, rank, cfg.val_targets, cfg.rec)
    callback_logging = CallBackLogging(
        50, rank, total_step, cfg.batch_size, world_size, writer=None)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)
    alpha = 10.0  # 10.0
    loss = AverageMeter()
    global_step = cfg.global_step

    for epoch in range(start_epoch, cfg.num_epoch):
        for index, (list_id, list_name_image, embedding, label, mean_distance) in enumerate(dataloader):

            global_step += 1
            lr = scheduler_header(global_step)
            # print("learning rate: ", lr, get_lr(opt_head))

            list_id = list(list_id)
            list_name_image = list(list_name_image)

            embedding = embedding.to("cuda")
            label = label.cuda()
            mean_distance = torch.unsqueeze(mean_distance, 1).cuda()
            # print("index: ", index)
            features = embedding
            qs = head(features)
            thetas, index_nq, ccs, nnccs = FR_loss(features, label)
            # print(ccs.shape)
            # print(mean_distance.shape)
            ccs = ccs

            # if len(index_nq) > 0:

            #     root_dir = "/home1/webface_260M/unzip_folder/WebFace260M"
            #     save_dir = "low_quality_image"
            #     file = open("note_score.txt", "a")

            #     list_low_id = [list_id[i] for i in index_nq]
            #     list_low_image = [list_name_image[i] for i in index_nq]
            #     os.makedirs(save_dir, exist_ok= True)
            #     for id, name in zip(list_low_id, list_low_image):
            #         path_image = os.path.join(root_dir, id, name)
            #         cmd = f"cp {path_image} {os.path.join(save_dir, str(id) + '-' + str(name))}"
            #         os.system(cmd)
            # print(mean_distance)
            # print(mean_distance.size())
            # exit()

            ''''''
            # copy_ccs = ccs.detach().cpu().numpy()
            # copy_nnccs = nnccs.detach().cpu().numpy()

            # root_dir = "/home1/webface_260M/unzip_folder/WebFace260M"
            # save_dir = "test_images_shuffle"
            # file = open("note_score_shuffle.txt", "a")
            # os.makedirs(save_dir, exist_ok= True)
            # count_idx = 0
            # for id, name in zip(list_id, list_name_image):
            #     path_image = os.path.join(root_dir, id, name)
            #     # cmd = f"cp {path_image} {os.path.join(save_dir, str(id) + '-' + str(name))}"
            #     # os.system(cmd)

            #     file.write(
            #         str(path_image) + " " + str(copy_ccs[count_idx][0]) + " " + str(copy_nnccs[count_idx][0]) + "\n"
            #     )
            #     count_idx += 1

            # exit()
            ''''''
            def prev_sub(q):
                prev = torch.empty_like(q.reshape(1, -1))
                prev[0][0:-1] = q.reshape(1, -1).clone()[0][1:]
                prev[0][-1] = q.reshape(1, -1)[0][0] 
                return prev 
        

            ccs_sub1 = prev_sub(ccs) # shape (1, bs)
            y_truth = torch.where((ccs_sub1[0] - ccs.reshape(1, 128)[0]) < 0, 0.0, 1.0)

            qs_sub = prev_sub(qs) 
            y_pred = qs_sub[0] - qs.reshape(1, 128)[0]
            y_pred = torch.exp(y_pred) 
            y_pred = y_pred / (y_pred + 1)
            bce_loss = rank_loss_func(y_pred, y_truth)

            


            # ref_score = ((ccs) / (nnccs + 1 + 1e-9))
            ref_score = (ccs)
            qs = (qs)
            # loss_qs = smooth_l1_loss(ref_score, qs ,scale_loss(ref_score), beta=0.5)
            # loss_qs = smooth_l1_loss(ref_score, qs , None, beta=0.5)
            loss_qs = criterion_qs(qs, ref_score)
            # print(qs.reshape(1, -1).size())
            # softmax_qs = F.log_softmax(qs.reshape(1, -1), dim=1)
            # softmax_ref_score = F.softmax(ref_score.reshape(1, -1), dim=1)

            # loss_kl = F.kl_div(softmax_qs, softmax_ref_score, log_target=True)
            # loss_kl = rank_loss(ref_score, qs)
            # loss_kl = compute_emd(softmax_qs, softmax_ref_score)

            # exit()

            loss_v = 10 * loss_qs + 4 * bce_loss
            loss_v.backward()
            if global_step % 10 == 0:
                print("ref_score: ", ref_score[:10])
                print("qs score: ", qs[:10])
                # print("loss KL: ", loss_kl)
                print("bce loss: ", bce_loss)


            clip_grad_norm_(head.parameters(), max_norm=5, norm_type=2)

            opt_head.step()

            opt_head.zero_grad()

            loss.update(loss_v.item(), 1)

            callback_logging(global_step, loss, epoch, 0,
                             loss_qs, get_lr(opt_head))
            # callback_verification(global_step, backbone)
            if global_step % 1000 == 0:
                callback_checkpoint(global_step, backbone, head)

        callback_checkpoint(global_step, backbone, head)


main()
