import os 

import cv2 
import numpy as np 



import os

import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm 

def plot_images_with_scores(images, scores, num_cols=10, name_save="yeah.jpg"):
    # Số lượng ảnh
    num_images = len(images)

    # Tính số dòng cần thiết
    num_rows = num_images // num_cols + (num_images % num_cols > 0)

    # Tạo figure và axes
    fig, axs = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))

    # Duyệt qua từng ảnh và score tương ứng
    for i, (image, score) in tqdm(enumerate(zip(images, scores))):
        # Tính toán chỉ số dòng và cột tương ứng
        row = i // num_cols
        col = i % num_cols

        # Hiển thị ảnh
        axs[row, col].imshow(image)
        axs[row, col].axis('off')  # Tắt các trục

        # Hiển thị score dưới ảnh
        axs[row, col].set_title(f'Score: {score}')

    # Xóa các axes không sử dụng
    for i in range(num_images, num_rows * num_cols):
        fig.delaxes(axs.flatten()[i])

    # Hiển thị figure
    plt.show()

    plt.savefig(name_save)

if __name__ == "__main__":
    path_list_score = "note_score_shuffle.txt"

    file = open(path_list_score, "r")
    data = file.readlines()
    file.close()
    list_images = []
    list_scores = []
    for line in tqdm(data[:5000]):
        score_neg = 0
        name_images, score_pos= line.split(" ")
        score_pos = float(score_pos)
        score_neg = float(score_neg)
        score_pos = score_pos / (score_neg + 1 + 1e-9)
        image = cv2.imread(os.path.join("sample_training", name_images))
        list_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        list_scores.append(score_pos)

    # arg = np.argsort(list_scores)
    # list_images = [list_images[i] for i in arg]
    # list_scores = [list_scores[i] for i in arg]

    for start in range(0, len(list_images), 500):
        plot_images_with_scores(list_images[start:min(
            start + 500, len(list_images))], list_scores[start:min(start + 500, len(list_images))], name_save= f"sub_mean_ori_{start}.jpg")
