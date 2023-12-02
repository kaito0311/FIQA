import os 
import time 

import cv2
import torch 
import numpy as np 
# import mxnet as mx 
# from mxnet import ndarray as nd
import pickle 

from backbones.iresnet_imintv5 import iresnet160









diction = np.load("dict_name_features.npy", allow_pickle= True).item()
mean = np.load("./data/mean.npy", allow_pickle= True)

name_id = list(diction.keys())[0]
feature  = np.load("./feature_dir/" + str(name_id) + ".npy")


cos_theta = feature @ mean.T


pass 









exit() 



class FaceDataset():
    def __init__(self, feature_dir, path_dict) -> None:
        super().__init__()
        self.feature_dir = feature_dir 
        self.dict = np.load(path_dict, allow_pickle= True).item() 
        self.list_name = [] 
        self.list_id = []
    
    def prepare(self):
        for key in self.dict.keys():
            self.list_name = self.list_name + self.dict[key] 
            self.list_id = self.list_id + [key] * len(self.dict[key])
        print(len(self.list_name))
        print(len(self.list_id))
        assert len(self.list_name) == len(self.list_id) 
    
    def __getitem__(self, index):
        name_image = self.list_name[index] 
        list_file:list = self.dict[self.list_id[index]]
        idx = list_file.index(name_image)

        embedding = np.load(os.path.join(self.feature_dir, str(self.list_id[index]) + ".npy"))[idx]

        return embedding 
    def __len__(self):
        return len(self.list_name)
 

dataset = FaceDataset("feature_dir", "/home2/tanminh/FIQA/dict_name_features.npy")

dataset.prepare() 

embed = dataset[10]
print(embed.shape)


# import losses 

# embeddings = torch.rand(4, 512).to("cpu")
# label = torch.randint(0, 85742, (4,)).to("cpu")

# fr = losses.CR_FIQA_LOSS_ONTOP(device="cpu")

# thetas, std, ccs,nnccs = fr(embeddings, label) 

# print(ccs)
# print(nnccs)



# dict = np.load("/home2/tanminh/FIQA/dict_name_features.npy", allow_pickle= True)

# dict = dict.item() 

# print(dict.keys())
# print(dict["2_3_0110870"])

# features= np.load("/home2/tanminh/FIQA/feature_dir/0_3_0091257.npy")
# print(features)
# print(features.shape)





exit() 

for idx_source in range(1,144):
    if idx_source in [7,11]:
        continue
    try:
        ls_image = np.load(f"low_quality_images/{idx_source}_low_quality.npy")
    except: 
        continue
    path = "/home1/data/tanminh/FIQA/images"
    ls_embed = np.load(f"low_quality_images/{idx_source}_low_quality_features.npy")
    ls_index_neq = np.load(f"/home1/data/tanminh/FIQA/low_quality_images/{idx_source}_index_low_quality.npy")
    ls_label = np.load(f"/home1/data/tanminh/FIQA/low_quality_images/{idx_source}_label_low_quality.npy")
    mean_ = np.load("/home1/data/tanminh/CR-FIQA/mean.npy")
    mean_ = mean_ / np.linalg.norm(mean_, axis= 1).reshape(-1, 1)
    print("index source: ", idx_source)
    os.makedirs(path, exist_ok= True)
    for index, image in enumerate(ls_image): 
        feature = ls_embed[ls_index_neq[index]]
        feature = feature / np.linalg.norm(feature, axis=0).reshape(-1, 1) 
        cos_theta = feature @ mean_.T
        print(np.max(cos_theta))
        print(np.argmax(cos_theta))
        print(cos_theta[0][ls_label[ls_index_neq[index]]])
        print(ls_label[ls_index_neq[index]])
        cv2.imwrite(
            os.path.join(path, f"{idx_source}_" + str(index) + ".jpg"), 
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )




exit()


from torchvision import transforms 

transform = transforms.Compose(
    [transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])


root_dir = "/home1/data/tanminh/faces_emore"
path_imgrec = os.path.join(root_dir, 'train.rec')
path_imgidx = os.path.join(root_dir, 'train.idx')
imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

s = imgrec.read_idx(8 * 128 + 68)
header, img = mx.recordio.unpack(s)
sample = mx.image.imdecode(img).asnumpy() 


tensor_lowquality = np.load("low_quality.npy")
image = sample 
image = transform(np.array(image)).detach().numpy()
print(image.shape)

tensor_image = tensor_lowquality[2]
print(np.sum(tensor_image - image))
print(tensor_image.shape)
exit()




original = tensor_lowquality * 0.5 + 0.5 
original = tensor_lowquality * 255. 
original = np.clip(original, 0, 255)
original = np.array(original, dtype=np.uint8) 
original = np.transpose(original, (0,2,3,1))

path_dir = "low_quality_images"
os.makedirs(path_dir, exist_ok= True)

for idx, img in enumerate(original):
    cv2.imwrite(os.path.join(path_dir, str(idx) + ".jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


backbone = iresnet160(False)
backbone.load_state_dict(torch.load("/home1/data/tanminh/Face_Recognize_Quaility_Assessment/r160_imintv4_statedict.pth"))
backbone.to("cuda")
backbone.eval() 


count = np.load("/home1/data/tanminh/CR-FIQA/count.npy")
sum_ = np.load("/home1/data/tanminh/CR-FIQA/sum.npy")
mean_ = np.load("/home1/data/tanminh/CR-FIQA/mean.npy")
# print(np.sum(count))
# print(count[:10])
# print(sum_.shape)
# print(sum_[0][:10])
# print(mean_.shape) 
# print(mean_[0][:10])


features = np.load("/home1/data/tanminh/FIQA/low_quality_features.npy")
labels = np.load("/home1/data/tanminh/FIQA/label_low_quality.npy")
features = features / np.linalg.norm(features, axis= 1).reshape(-1, 1)
mean_ = mean_ / np.linalg.norm(mean_, axis= 1).reshape(-1, 1)
index_low = np.load("/home1/data/tanminh/FIQA/index_low_quality.npy")
cos_theta = features @ mean_.T

# print(features.shape)


root_dir = "/home1/data/tanminh/faces_emore"
path_imgrec = os.path.join(root_dir, 'train.rec')
path_imgidx = os.path.join(root_dir, 'train.idx')

imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
import torch 

for index in index_low: 
    s = imgrec.read_idx(8 * 128 + index+1)
    header, img = mx.recordio.unpack(s)
    sample = mx.image.imdecode(img).asnumpy() 
    print(header.label)
    path_dir= "original_images/"
    tensor = ((sample / 255.0) - 0.5) / 0.5  
    tensor = np.transpose(tensor, (2, 0, 1)) 
    tensor = torch.from_numpy(tensor.astype(np.float32)) 
    tensor = torch.unsqueeze(tensor, 0) 
    feature = backbone(tensor.to("cuda")).detach().cpu().numpy() 
    cos_theta2 = feature @ mean_.T
    os.makedirs(path_dir,exist_ok= True) 
    cv2.imwrite(os.path.join(path_dir, str(index) + ".jpg"), cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))
print()
# print("he")
# # with open(calfw_bin, 'rb') as file:
# #     bins, issame_list = pickle.load(file, encoding="bytes")
# # file.close() 
# mx.recordio.unpack(imgrec.read_idx(1))
# calfw_bin = "/home1/data/tanminh/faces_emore/cplfw.bin"
# print(len(bins))
# print(len(issame_list))
# print(issame_list[:10])
# idx = 16
# img = mx.image.imdecode(bins[idx])
# img = img.asnumpy() 
# cv2.imwrite(
#     "test.jpg", 
#     img
# )

# img = mx.image.imdecode(bins[idx+1])
# img = img.asnumpy() 
# cv2.imwrite(
#     "test_1.jpg", 
#     img
# )

# import os 

# cmd = "df -h /home1/data/tanminh/CR-FIQA/"
# (os.system(cmd))