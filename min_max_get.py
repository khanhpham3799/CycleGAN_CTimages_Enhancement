import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torch.nn as nn

train_img_link = "/media/khanhpham/새 볼륨1/AAPM_data/train/quarter_dose"
train_mask_link = "/media/khanhpham/새 볼륨1/AAPM_data/train/full_dose"
test_img_link = "/media/khanhpham/새 볼륨1/AAPM_data/test/quarter_dose"
test_mask_link = "/media/khanhpham/새 볼륨1/AAPM_data/test/full_dose"

model_path = "/media/khanhpham/새 볼륨1/AAPM_data/min_max_get.pth"

def find_min_max(dir_link):
    list_img = sorted(os.listdir(dir_link))
    max_value = 0
    min_value = 0
    for img in list_img:
        img_link = os.path.join(dir_link,img)
        temp_img = np.load(str(img_link)).astype(np.float64)
        max_value = max(max_value,np.max(temp_img))
        min_value = min(min_value,np.min(temp_img))   
    max_value = 1000*(max_value-0.194)/0.194
    min_value = 1000*(min_value-0.194)/0.194
    print(max_value,min_value)
    return max_value, min_value

max_train_img, min_train_img = find_min_max(train_img_link)
max_train_mask, min_train_mask = find_min_max(train_mask_link)
max_test_img, min_test_img = find_min_max(test_img_link)
max_test_mask, min_test_mask = find_min_max(test_mask_link)

max_range = max(max_train_img,max_train_mask,max_test_img,max_test_mask)
min_range = min(min_train_img,min_train_mask,min_test_img,min_test_mask)

print("max:",max_range,"min:",min_range)
torch.save({
    "max": max_range,
    "min": min_range,
    }, model_path
    )
print("Finish saving!")
'''
a = np.random.randint(5, size=(2, 3))
b = np.random.randint(8, size=(2, 3))
c = np.vstack((a,b))
print(c)
c = 1000*(c-0.0194)/0.0194
print(c)
print(np.max(c))
'''