from easydict import EasyDict as edict

config = edict()
config.dataset = "adam" # training dataset
config.embedding_size = 1024 # embedding size of evaluation
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128 # batch size per GPU
config.lr = 1e-3
config.output = "output/sub_mean_dis" # train evaluation output folder
config.s=64.0
config.m=0.50
config.beta=0.5
config.num_workers = 4
config.device= "cuda"

config.resume = False
config.resume_head = "/home2/tanminh/FIQA/output/pure_L1_add_noise_ngaongo_them_kldiv/271000header.pth"
config.resume_backbone = "/home1/data/tanminh/FIQA/output/R50_CRFIQA/18000backbone.pth"
config.global_step=0 # step to resume


# type of network to train [ iresnet100 | iresnet50 ]
config.network = "iresnet160"




if config.dataset == "emoreIresNet":
    config.rec = "/home1/data/tanminh/faces_emore"
    config.num_classes = 85742
    config.num_image = 5822653
    config.num_epoch =  18
    config.warmup_epoch = -1
    config.val_targets = ["cplfw"]
    # config.val_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
    config.eval_step=5686
    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in [8, 14,20,25] if m - 1 <= epoch])  # [m for m in [8, 14,20,25] if m - 1 <= epoch])
    config.lr_func = lr_step_func

elif config.dataset == "webface":
    config.rec = "data/faces_webface_112x112"
    config.num_classes = 10572
    config.num_image = 501195
    config.num_epoch = 34   #  [22, 30, 35] [22, 30, 40]
    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
    config.eval_step= 958 #33350

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [20, 28, 32] if m - 1 <= epoch])
    config.lr_func = lr_step_func
elif config.dataset == "adam":
    config.rec = "/home1/data/tanminh/faces_emore"
    config.num_classes = 4998
    config.num_epoch =  50
    config.warmup_epoch = -1
    config.val_targets = ["cplfw"]
    # config.val_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
    config.eval_step=5686
    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [20, 28, 32] if m - 1 <= epoch])
    config.lr_func = lr_step_func