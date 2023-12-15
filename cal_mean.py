import os

import cv2
import numpy as np
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering, KMeans
from collections import Counter

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from scipy.cluster.hierarchy import linkage, fcluster


def clustering_data(X, threshold=0.7):
    '''
    Return list vector that being the most common cluster
    '''
    from numba import njit
    # @njit

    def cosine_distance_matrix(X):
        norm = np.sum((X**2), axis=1).reshape(-1, 1) ** 0.5
        X_norm = X / norm
        return 1 - np.dot(X_norm, X_norm.T)

    dist_matrix = cosine_distance_matrix(X)

    # Sử dụng phương pháp 'linkage' để tính toán các liên kết giữa các cụm
    Z = linkage(dist_matrix, method='average')

    # Sử dụng hàm 'fcluster' để tạo cụm dựa trên ngưỡng
    labels = fcluster(Z, t=threshold, criterion='distance')

    # Đếm số lượng phần tử trong mỗi cụm
    counter = Counter(labels)

    # Tìm chỉ số của cụm có số lượng phần tử lớn nhất
    max_cluster_index = counter.most_common(1)[0][0]

    max_cluster_indices = [index for index, label in enumerate(
        labels) if label == max_cluster_index]
    
    dis_avg_matrix = ((dist_matrix[max_cluster_indices]).T)[max_cluster_indices]


    return X[max_cluster_indices].reshape(-1, 1024), np.mean(dis_avg_matrix)


feature_dir = 'feature_dir'
feature_flip_dir = "feature_flip_dir"
diction = np.load("dict_name_features.npy", allow_pickle=True).item()
ls_mean = []

diction_mean_cluster = dict() 

for name_id in tqdm(diction.keys()):
    data = np.load(os.path.join(feature_dir, name_id+".npy")).reshape(-1, 1024)
    if len(data) > 2:
        data, mean_distance_cluster = clustering_data(data, threshold=5.0)
        diction_mean_cluster[name_id] = mean_distance_cluster 
    ls_mean.append(np.mean(data, axis=0).reshape(1, -1))

ls_mean = np.concatenate(ls_mean, axis=0)
print(ls_mean.shape)

np.save("data/diction_mean_cluster_thresh_5e0.npy", diction_mean_cluster)
np.save("data/mean_cluster.npy", ls_mean)
