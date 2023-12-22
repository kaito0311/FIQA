from easydict import EasyDict as edict


class config:
    dataset = "adam"  # training dataset
    embedding_size = 1024  # embedding size of evaluation
    momentum = 0.9
    weight_decay = 5e-4
    batch_size = 32  # batch size per GPU
    lr = 0.000046
    output = "output/l2r_vit_flip_from_scratch_10k_id"  # train evaluation output folder
    s = 64.0
    m = 0.50
    beta = 0.5
    num_workers = 4
    device = "cuda"

    resume = False
    resume_vit = "/home/data2/tanminh/FIQA/FIQA/output/l2r_vit_flip_from_scratch_10k_id/58000header.pth"
    resume_head = None
    resume_backbone = None
    global_step = 58000  # step to resume

    # type of network to train [ iresnet100 | iresnet50 ]
    network = "iresnet160"

    start_epoch = 0
    num_epoch = 15
    warmup_epoch = -1

    # dataset
    # feature_dir = "feature_dir"
    dict_name_features = "data/10k_id/dict_name_features.npy"
    path_dict_mean_distance = None
    # path_mean_feature = "data/2k_id/no_flip/mean_feature.npy"
    # path_list_mean_cosine = "data/2k_id/no_flip/ls_mean_cosine_similar.npy"
    # path_list_std_cosine = "data/2k_id/no_flip/ls_std_cosine_similar.npy"
    path_mean_feature = "data/10k_id/both_flip/mean_feature.npy"
    path_list_mean_cosine = "data/10k_id/both_flip/ls_mean_cosine_similar.npy"
    path_list_std_cosine = "data/10k_id/both_flip/ls_std_cosine_similar.npy"
    path_list_name = "data/10k_id/list_name.npy"
    path_list_id = "data/10k_id/list_id.npy"
    path_backbone_imint = "/home/data2/tanminh/Evaluate_FIQA_EVR/pretrained/stacking_avg_r160+ada-unnorm-stacking-ada-1.6.onnx"
    num_feature_out = 1024