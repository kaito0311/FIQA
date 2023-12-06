import os

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from collections import Counter

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from scipy.cluster.hierarchy import linkage, fcluster

# Giả sử X là dữ liệu của bạn

# X = X / np.linalg.norm(X, axis=1).reshape(-1, 1)
# diction_name = np.load("/home2/tanminh/FIQA/dict_name_features.npy", allow_pickle=True).item()
# # Tính toán ma trận khoảng cách cosine
# dist_matrix = cosine_distances(X)
# print(dist_matrix[0])
# # Sử dụng phương pháp 'linkage' để tính toán các liên kết giữa các cụm
# Z = linkage(dist_matrix, method='average')

# # Sử dụng hàm 'fcluster' để tạo cụm dựa trên ngưỡng
# labels = fcluster(Z, t=0.7, criterion='distance')

# print(np.max(dist_matrix))
# # Bây giờ, 'labels' chứa nhãn cụm cho mỗi điểm dữ liệu trong 'X'
# print(labels)
# for idx in set(labels):
#     print("set of ", idx)
#     for i, label in enumerate(labels):
#         if label == idx:
#             print(i)
#             print(list(diction_name["2_2_0079041"])[i])

# print(np.sqrt(np.sum((X[0] - X[1])**2)))
# print(dist_matrix[0][3])


def clustering_data(X, threshold=0.7):
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

    return max_cluster_indices


X = np.load(os.path.join("feature_dir", "0_3_0100996.npy"))
X = X / np.linalg.norm(X, axis=1).reshape(-1, 1)
indexes = clustering_data(X)

X_clus = X[indexes]
mean = np.mean(X_clus, axis=0)

print(np.sqrt(np.sum((mean-X[6])**2)))
print(np.dot(mean, X[6].T))