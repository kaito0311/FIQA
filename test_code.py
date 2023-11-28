import os 
import time 

import cv2
import numpy as np 
import mxnet as mx 
from mxnet import ndarray as nd
import pickle 

# root_dir = "/home1/data/tanminh/faces_emore"
# path_imgrec = os.path.join(root_dir, 'train.rec')
# path_imgidx = os.path.join(root_dir, 'train.idx')

# imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

# s = imgrec.read_idx(0)
# header, _ = mx.recordio.unpack(s)
# if header.flag > 0:
#     header0 = (int(header.label[0]), int(header.label[1]))
#     imgidx = np.array(range(1, int(header.label[0])))
# else:
#     imgidx = np.array(list(imgrec.keys))

# calfw_bin = "/home1/data/tanminh/faces_emore/cplfw.bin"

# with open(calfw_bin, 'rb') as file:
#     bins, issame_list = pickle.load(file, encoding="bytes")
# file.close() 

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

import os 

cmd = "df -h /home1/data/tanminh/CR-FIQA/"
(os.system(cmd))