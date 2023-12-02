import argparse
import logging
import os
from threading import local
import time

import torch
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

class Head_Cls(torch.nn.Module):
    def __init__(self, in_features= 512, out_features=1) -> None:
        super().__init__() 
        self.qs = torch.nn.Linear(in_features, out_features)
        
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0, 0.1) 
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
    def forward(self, x): 
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

    trainset = FaceDataset("feature_dir", "/home2/tanminh/FIQA/dict_name_features.npy")
    dataloader = DataLoader(trainset, cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, drop_last=True)

    # backbone = iresnet160(False)
    # backbone.load_state_dict(torch.load("/home1/data/tanminh/Face_Recognize_Quaility_Assessment/r160_imintv4_statedict.pth"))

    backbone = ONNX_IMINT("/home2/tanminh/FIQA/pretrained/stacking_avg_r160+ada-unnorm-stacking-ada-1.6.onnx")
    head = Head_Cls(1024, 1)
 
    head.cuda() 

    if cfg.resume: 
        print("[INFO] Resume. Loading last chk...")
        head.load_state_dict(torch.load(cfg.resume_head))

    
    head.train()  

    FR_loss= losses.CR_FIQA_LOSS_ONTOP(
        device= "cuda",
    )

    # opt_head = torch.optim.SGD(
    # params=[{'params': head.parameters()}],
    # lr=cfg.lr,
    # momentum=0.9, weight_decay=cfg.weight_decay)

    opt_head = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_head, lr_lambda=cfg.lr_func)

    criterion = CrossEntropyLoss()
    criterion_qs= torch.nn.SmoothL1Loss(beta=0.5)

    start_epoch = 0
    total_step = int(len(trainset) / cfg.batch_size * cfg.num_epoch)

    rank = 0
    world_size=1
    # callback_verification = CallBackVerification(cfg.eval_step, rank, cfg.val_targets, cfg.rec)
    callback_logging = CallBackLogging(50, rank, total_step, cfg.batch_size, world_size, writer=None)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)
    alpha=10.0  #10.0
    loss = AverageMeter()
    global_step = cfg.global_step


    for epoch in range(start_epoch, cfg.num_epoch):
        for index, (embedding, label) in enumerate(dataloader):

            global_step += 1
            embedding = embedding.to("cuda")
            label = label.cuda()
            # print("index: ", index)
            features= embedding
            qs = head(features)
            thetas, index_nq, ccs,nnccs = FR_loss(features, label)

            ref_score = ((ccs)/ (nnccs + 1 + 1e-9))
            qs = (qs)
            loss_qs=criterion_qs(ref_score,qs)
            loss_v = 100 * loss_qs
            loss_v.backward()
            clip_grad_norm_(head.parameters(), max_norm=5, norm_type=2)

            opt_head.step()

            opt_head.zero_grad()

            loss.update(loss_v.item(), 1)
            
            callback_logging(global_step, loss, epoch, 0,loss_qs, get_lr(opt_head))
            # callback_verification(global_step, backbone)
            if global_step % 1000 == 0: 
                callback_checkpoint(global_step, backbone, head)

        scheduler_backbone.step()
        
        callback_checkpoint(global_step, backbone, head)

main() 
