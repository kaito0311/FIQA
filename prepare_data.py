import os

import cv2
import torch
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


from backbones.iresnet_imintv5 import iresnet160
from models.imintv5 import ONNX_IMINT

torch.set_num_threads(5)


class FaceDataset(Dataset):
    def __init__(self, num_ids, root_dir, feature_dirs, device="cuda") -> None:
        super().__init__()
        self.root_dir = root_dir
        self.feature_dirs = feature_dirs
        self.num_ids = num_ids
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        self.list_path = []
        self.list_id = []

        self.device = device

    def prepare_data(self):
        if self.num_ids is None:
            list_unique_id = os.listdir(self.root_dir)
        else:
            assert len(os.listdir(self.root_dir)) > self.num_ids
            list_unique_id = os.listdir(self.root_dir)[:self.num_ids]
        print("[INFO] Number of id: ", len(list_unique_id))
        self.list_unique_id = list_unique_id
        os.makedirs(self.feature_dirs, exist_ok=True)

        self.list_path = []
        self.list_id = []
        print("[INFO] Process list name")
        for name_id in tqdm(list_unique_id):
            self.list_path = self.list_path + \
                glob.glob(os.path.join(self.root_dir, name_id) + "/*.jpg")
            self.list_id = self.list_id + [name_id] * len(os.listdir(os.path.join(self.root_dir, name_id))
                                                          )
        print("Done!!!")
        assert len(self.list_id) == len(self.list_path)

    def __getitem__(self, index):
        image = Image.open(self.list_path[index])
        image_tensor = self.transform(image)
        id = self.list_id[index]
        name_image = os.path.basename(self.list_path[index])
        return id, name_image, image_tensor

    def __len__(self):
        return len(self.list_id)

    # def convert_to_tensor(self, path_image):
    #     image = Image.open(path_image)
    #     tensor = self.transform(image)
    #     # tensor = torch.unsqueeze(tensor, 0)
    #     # tensor = tensor.to(self.device)
    #     return tensor

    # def __getitem__(self, index_id):
    #     path_id = os.path.join(self.root_dir, self.list_unique_id[index_id])
    #     list_images = []
    #     for name_image in os.listdir(path_id):
    #         list_images.append(self.convert_to_tensor(os.path.join(path_id, name_image)))

    #     list_images = torch.vstack(list_images)

    #     return os.listdir(path_id), list_images


if __name__ == "__main__":

    features_dir = "feature_dir"
    os.makedirs(features_dir, exist_ok=True)
    dataset = FaceDataset(
        num_ids=100000, root_dir="/home1/webface_260M/unzip_folder/WebFace260M", feature_dirs=features_dir)
    dataset.prepare_data()

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    # model= iresnet160(False)
    # model.load_state_dict(torch.load("pretrained/r160_imintv4_statedict.pth"))
    # model.eval()
    # model.to("cuda")

    model = ONNX_IMINT(
        "/home2/tanminh/FIQA/pretrained/stacking_avg_r160+ada-unnorm-stacking-ada-1.6.onnx")
    # model.to("cuda")

    dict_name_features = dict()

    curr_id = None
    ls_features = []

    for _, (list_id, list_name, list_images) in tqdm(enumerate(dataloader)):
        features = model(list_images.to("cuda"))
        list_id = list(list_id)
        list_name = list(list_name)
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu()

        for id, name, feature in zip(list_id, list_name, features):

            if id not in dict_name_features.keys():
                dict_name_features[id] = [name]
            else:
                dict_name_features[id].append(name)

            if curr_id is None:
                curr_id = id
            if curr_id == id:
                ls_features.append(feature)
            else:
                if len(dict_name_features[curr_id]) > 3:
                    np.save(os.path.join(features_dir, curr_id), ls_features)
                else:
                    del dict_name_features[curr_id]
                ls_features = [feature]
                curr_id = id

    if len(dict_name_features[curr_id]) > 3:
        np.save(os.path.join(features_dir, curr_id), ls_features)
    else:
        del dict_name_features[curr_id]

    print("num identity: ", len(dict_name_features.keys()))
    print("num identity: ", (len(os.listdir(features_dir))))
    assert len(os.listdir(features_dir)) == len(dict_name_features.keys())
    np.save("dict_name_features", dict_name_features)
