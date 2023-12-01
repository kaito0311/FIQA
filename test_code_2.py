import os 
import numpy as np 
import cv2
import mxnet as mx 
from mxnet import ndarray as nd

root_dir = "/home1/data/tanminh/faces_emore"
path_imgrec = os.path.join(root_dir, 'train.rec')
path_imgidx = os.path.join(root_dir, 'train.idx')
imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

path_dir = "source_images/"
os.makedirs(path_dir, exist_ok= True) 
count = np.load("/home1/data/tanminh/CR-FIQA/count.npy")

id = 15700

start_idx = np.sum(count[:id])
for i in range(int(count[id])):
    s = imgrec.read_idx(start_idx + i+1)
    header, img = mx.recordio.unpack(s)
    sample = mx.image.imdecode(img).asnumpy() 
    print(header.label)
    cv2.imwrite(
        os.path.join(path_dir, f"{id}_" + str(i) + ".jpg"), 
        cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
    )

