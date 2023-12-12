import os

import cv2
import numpy as np
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering, KMeans
from collections import Counter

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from scipy.cluster.hierarchy import linkage, fcluster


def get_upper_triangular_list(matrix):
    # Lấy chỉ số của các phần tử ở tam giác trên, không bao gồm đường chéo
    indices = np.triu_indices(n=matrix.shape[0], k=1)

    # Sử dụng các chỉ số này để trích xuất các giá trị từ ma trận
    upper_triangular_values = matrix[indices]

    # Chuyển đổi mảng numpy thành danh sách
    upper_triangular_list = upper_triangular_values

    return upper_triangular_list


def clustering_data(X, threshold=0.7, return_infor_prob=True):
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

    if return_infor_prob:
        dis_avg_matrix = ((dist_matrix[max_cluster_indices]).T)[
            max_cluster_indices]

        distance_unique = get_upper_triangular_list(dis_avg_matrix)
        mean_similar = np.mean(1 - distance_unique)
        std_similar = np.std(1 - distance_unique)

        return X[max_cluster_indices].reshape(-1, 1024), mean_similar, std_similar

    else:
        return X[max_cluster_indices].reshape(-1, 1024), None, None


feature_dir = '/home2/tanminh/FIQA/feature_dir'
diction = np.load("dict_name_features.npy", allow_pickle=True).item()
ls_mean_feature = []
ls_std_feature = []

ls_mean_cosine = [] 
ls_std_cosine = []

diction_mean_cluster = dict()
diction_std_cluster = dict()

for name_id in tqdm(diction.keys()):
    data = np.load(os.path.join(feature_dir, name_id+".npy")).reshape(-1, 1024)

    if len(data) > 2:
        data, mean_consine_distance, std_cosine_distance = clustering_data(data, threshold=2)
        diction_mean_cluster[name_id] = mean_consine_distance 
        diction_std_cluster[name_id] = std_cosine_distance
    

#     ls_mean_feature.append(np.mean(data, axis=0).reshape(1, -1))
#     ls_std_feature.append(np.std(data / np.linalg.norm(data,
#                   axis=1).reshape(-1, 1), axis=0).reshape(1, -1))


# ls_mean_feature = np.concatenate(ls_mean_feature, axis=0)
# ls_std_feature = np.concatenate(ls_std_feature, axis=0)
# print(ls_mean_feature.shape)

np.save("data/mean_cosine_distance_cluster.npy", diction_mean_cluster)
np.save("./data/std_cosine_distance_cluster.npy", diction_std_cluster)

# np.save("data/mean_cluster.npy", ls_mean_feature)
# np.save("data/std_cluster_norm.npy", ls_std_feature)
