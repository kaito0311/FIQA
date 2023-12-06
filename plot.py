import os 

import cv2 
import numpy as np 


path_txt = "/home2/tanminh/FIQA/note_score_shuffle.txt"

data = None 
with open(path_txt, 'r') as file:
    data = file.readlines() 
file.close()

break_line = 10
list_image = [] 
list_temp = []

for line in data: 
    path, ccs, nncs =  line.split(" ")

    if len(list_temp) > 0 and len(list_temp) % break_line == 0: 
        list_image.append(np.concatenate(list_temp, axis=1))
        list_temp = [] 
    score = float(ccs) / (float(nncs) + 1 + 1e-9)
    # if  score > 0.5 and score < 0.6: 
    if  score > 0.6: 
        list_temp.append(cv2.resize((cv2.imread(path)), (64, 64))) 

if len(list_image) == 0: 
    exit() 
if len(list_image) == 1: 
    cv2.imwrite("6e-1.jpg", list_image[0])
    exit()
cv2.imwrite("6e-1.jpg", np.concatenate(list_image, axis=0))


