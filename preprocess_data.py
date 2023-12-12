import os
import time
import numbers

import torch
import mxnet as mx
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from backbones.iresnet_imintv5 import iresnet160

torch.set_num_threads(5)


class Head_Cls(torch.nn.Module):
    def __init__(self, in_features=512, out_features=1) -> None:
        super().__init__()
        self.qs = torch.nn.Linear(in_features, out_features)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(
            path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label, index

    def __len__(self):
        return len(self.imgidx)

if __name__ == "__main__":
    dataset = FaceDataset(
        "feature_dir", "/home2/tanminh/FIQA/dict_name_features.npy")
    batch_size = 4

    dataloader = DataLoader(dataset, batch_size, shuffle=False,
                            num_workers=4, drop_last=False)

    backbone = iresnet160(False)
    head = Head_Cls(512, 1)

    sum_array = torch.zeros(size=(85742, 512))
    sum_norm_array = torch.zeros(size=(85742, 512))
    count_array = torch.zeros(size=(85742,))

    backbone.load_state_dict(torch.load(
        "/home1/data/tanminh/Face_Recognize_Quaility_Assessment/r160_imintv4_statedict.pth"))
    backbone.eval()

    backbone.to('cuda')
    # sum_array = sum_array.cuda()
    # count_array = count_array.cuda()
    count = 0
    for _, (img, label, index) in enumerate(dataloader):
        count += batch_size
        features = backbone(img.to("cuda"))
        print("infered.")
        label = label.detach().cpu()
        features = features.detach().cpu()
        norm_features = features / (np.linalg.norm(features, axis=0))
        for unit_label, unit_feature, unit_norm_feature in zip(label, features, norm_features):
            sum_array[unit_label] += unit_feature
            sum_norm_array[unit_label] += unit_norm_feature
            # print(sum_array.shape)
            count_array[unit_label] += 1

        features = features.numpy()
        np.save("features_sample", features)
        label = label.numpy()
        np.save("label", label)
        exit()

        del features
#     if count % 1000 == 0:
#         print(count)
#         np.save("mean.npy", (sum_array / count_array.reshape(-1, 1)).detach().cpu().numpy())
#         np.save("sum.npy",sum_array.detach().numpy() )
#         np.save("count.npy",count_array.detach().numpy() )
#         np.save("sum_norm.npy", sum_norm_array.detach().numpy())
#     pass

# np.save("mean.npy", (sum_array / count_array.reshape(-1, 1)).detach().cpu().numpy())
# np.save("sum.npy",sum_array.detach().numpy() )
# np.save("count.npy",count_array.detach().numpy() )
# np.save("sum_norm.npy", sum_norm_array.detach().numpy())
