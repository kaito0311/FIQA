from easydict import EasyDict as edict


class config:
    dataset = "adam"  # training dataset
    embedding_size = 1024  # embedding size of evaluation
    momentum = 0.9
    weight_decay = 5e-4
    batch_size = 128  # batch size per GPU
    lr = 1e-3
    output = "output/learning_to_rank_norm_cosine_truth"  # train evaluation output folder
    s = 64.0
    m = 0.50
    beta = 0.5
    num_workers = 4
    device = "cuda"

    resume = False
    resume_head = "/home2/tanminh/FIQA/output/pure_L1_add_noise_ngaongo_them_kldiv/271000header.pth"
    resume_backbone = "/home1/data/tanminh/FIQA/output/R50_CRFIQA/18000backbone.pth"
    global_step = 0  # step to resume

    # type of network to train [ iresnet100 | iresnet50 ]
    network = "iresnet160"

    num_epoch = 18
    warmup_epoch = -1

    # dataset
    feature_dir = "feature_dir"
    dict_name_features = "/home2/tanminh/FIQA/data/100k_id/dict_name_features.npy"
    path_dict_mean_distance = "/home2/tanminh/FIQA/data/100k_id/diction_mean_cluster_thresh_5e-1.npy"
    path_mean_feature = "/home2/tanminh/FIQA/data/100k_id/mean_cluster.npy"
    path_list_mean_cosine = "/home2/tanminh/FIQA/data/100k_id/list_mean_similar.npy"
    path_list_std_cosine = "/home2/tanminh/FIQA/data/100k_id/list_std_similar.npy"
    path_list_name = "/home2/tanminh/FIQA/data/100k_id/list_name.npy"
    path_list_id = "/home2/tanminh/FIQA/data/100k_id/list_id.npy"
    path_backbone_imint = "/home2/tanminh/FIQA/pretrained/stacking_avg_r160+ada-unnorm-stacking-ada-1.6.onnx"
    num_feature_out = 1024