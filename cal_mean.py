import os 

import cv2 
import numpy as np 


feature_dir = '/home2/tanminh/FIQA/feature_dir'
diction = np.load("dict_name_features.npy", allow_pickle= True).item()
ls_mean = [] 

for name_id in diction.keys():
    data = np.load(os.path.join(feature_dir, name_id+".npy")).reshape(-1, 1024)
    
    ls_mean.append(np.mean(data, axis=0).reshape(1, -1)) 

ls_mean = np.concatenate(ls_mean, axis=0)
print(ls_mean.shape)

np.save("data/mean.npy", ls_mean)


