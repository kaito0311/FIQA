import numbers
import os
import queue as Queue
import threading

import mxnet as mx
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from degrade_image.degrade import auto_degrade


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank,
                                                 non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
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
        sample_tensor = self.transform(sample)
        return sample, sample_tensor, label

    def __len__(self):
        return len(self.imgidx)


class FaceDataset(Dataset):
    def __init__(self, feature_dir, path_dict, path_dict_mean_distance=None, save_listed=True, path_list_name="list_name.npy", path_list_id="list_id.npy") -> None:
        super().__init__()
        self.feature_dir = feature_dir
        self.dict = np.load(path_dict, allow_pickle=True).item()
        self.list_name = []
        self.list_id = []
        self.list_name_uni_id = list(self.dict.keys())
        self.path_list_name = path_list_name
        self.path_list_id = path_list_id
        self.save_listed = save_listed
        self.path_dict_mean_distance = path_dict_mean_distance
        self.dict_mean_distance = None
        self.prepare()

    def prepare(self):

        if self.path_dict_mean_distance is not None:
            self.dict_mean_distance = np.load(
                self.path_dict_mean_distance, allow_pickle=True).item()

        if os.path.isfile(self.path_list_id):
            self.list_name = np.load(self.path_list_name)
            self.list_id = np.load(self.path_list_id)
            assert len(self.list_name) == len(self.list_id)
            if len(set(self.list_id)) == len(self.list_name_uni_id):
                return

        for key in tqdm(self.dict.keys()):
            self.list_name = self.list_name.extend(self.dict[key])
            self.list_id = self.list_id.extend([key] * len(self.dict[key]))
        print(len(self.list_name))
        print(len(self.list_id))
        if self.save_listed:
            np.save(self.path_list_name, self.list_name)
            np.save(self.path_list_id, self.list_id)
        assert len(self.list_name) == len(self.list_id)

    def __getitem__(self, index):
        name_image = self.list_name[index]
        name_id = self.list_id[index]
        list_file: list = self.dict[name_id]

        idx = list_file.index(name_image)

        embedding = np.load(os.path.join(
            self.feature_dir, str(name_id) + ".npy"))[idx]

        if self.dict_mean_distance is not None:
            mean_dis = self.dict_mean_distance[name_id]
            mean_dis_tensor = torch.tensor(np.float32(mean_dis))
        else:
            mean_dis_tensor = 0.0

        if np.random.rand() < 0.5:
            embedding = embedding + \
                np.random.normal(loc=0, scale=0.2, size=embedding.shape)

        embedding_tensor = torch.Tensor(embedding.astype(np.float32))
        label_tensor = torch.tensor(self.list_name_uni_id.index(
            self.list_id[index]), dtype=torch.long)
        return name_id, name_image, embedding_tensor, label_tensor, mean_dis_tensor

    def __len__(self):
        return len(self.list_name)


class FaceDatasetImage(Dataset):
    def __init__(self, root_dir: str, path_diction_name: str, path_list_name: str = None, path_list_id: str = None, is_save_listed=True, prob_augment: float = 0.5) -> None:
        super().__init__()
        self.diction_name = np.load(
            path_diction_name, allow_pickle=True).item()
        self.list_name = []
        self.list_id = []
        self.list_name_uni_id = list(self.diction_name.keys())
        self.root_dir = root_dir

        self.path_list_name = path_list_name
        self.path_list_id = path_list_id
        self.is_save_listed = is_save_listed
        self.prob_augment = prob_augment

        self.transforms_resize_224 =  transforms.Compose([
            transforms.Resize((224, 224)),

        ])
        self.transforms_resize_112 =  transforms.Compose([
            transforms.Resize((112, 112)),
        ])

        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])


        self.prepare()

    def prepare(self):
        if os.path.isfile(self.path_list_id):
            self.list_name = np.load(self.path_list_name)
            self.list_id = np.load(self.path_list_id)

            assert len(self.list_name) == len(self.list_id)
            if len(set(self.list_id)) == len(self.list_name_uni_id):
                return

        for key in tqdm(self.diction_name.keys()):
            self.list_name = self.list_name.extend(self.diction_name[key])
            self.list_id = self.list_id.extend[key] * \
                len(self.diction_name[key])

        print("[INFO] Total images: ", len(self.list_name))
        print("[INFO] Total ids: ", len(self.list_id))

        if self.is_save_listed and type(self.path_list_id) is str:
            np.save(self.path_list_id, self.list_id)
            np.save(self.path_list_name, self.list_name)

        assert len(self.list_name) == len(self.list_id)

    def __getitem__(self, index):
        name_image = self.list_name[index]
        name_id = self.list_id[index]

        label_tensor = torch.tensor(self.list_name_uni_id.index(
            self.list_id[index]), dtype=torch.long)

        image = Image.open(os.path.join(self.root_dir, name_id, name_image))
        image = np.array(image)
        if np.random.rand() < self.prob_augment:
            image_degrade = auto_degrade(image)
        else:
            image_degrade = image 
        image_tensor_224 = self.transforms(self.transforms_resize_224(Image.fromarray(image_degrade)))
        image_tensor_112 = self.transforms(self.transforms_resize_112(Image.fromarray(image_degrade)))

        return image_tensor_112, image_tensor_224, label_tensor

    def __len__(self):
        return len(self.list_name)
