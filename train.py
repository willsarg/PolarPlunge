import torch
from transformers import AutoImageProcessor, EfficientNetModel
import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import warnings
import pathlib
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

import torch.optim as optim


warnings.filterwarnings('ignore')


images_patt = "C:\\Users\\Carbonite\\Dropbox\\~Temp\\PolarPlunge\\WAID\\WAID\\images\\train"
annotation_patt = "C:\\Users\\Carbonite\\Dropbox\\~Temp\\PolarPlunge\\WAID\\WAID\\labels\\train"

class PolarImageDataset(Dataset):
    def __init__(self):
        self.indices = {}
        self.filename = {}
        self.labels = {}
        
        for idx, file in tqdm(zip(range(0, min(100, len(os.listdir(images_patt)))), os.listdir(images_patt)[:100])):
            f = open(os.path.join(images_patt, file), "r")
            self.indices[file.split('.')[0]] = idx
            self.filename[idx] = os.path.join(images_patt, file)
            
            # bboxes[file.split('.')[0]] = []
            # for i in parts[1:]:
            #     bboxes[file.split('.')[0]].append(float(i.strip()))
            f_label = open(os.path.join(annotation_patt, file[:-4]+'.txt'), "r")
            self.labels[idx] = int(f_label.readlines()[0].split(' ')[0])
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # img = Image.open(self.filename[idx])
        print(self.filename[idx])
        img = Image.open(self.filename[idx])
        transform = transforms.ToTensor()
        img_tensor = transform(img)
        # image = read_image(path=str(self.filename[idx]))
        label = self.labels[idx]
        return img_tensor, label

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

image_processor = AutoImageProcessor.from_pretrained("google/efficientnet-b0")
model = EfficientNetModel.from_pretrained("google/efficientnet-b0")
prediction_layer = nn.Linear(1280, 6)
softmax = nn.Softmax()

print("Loading data...")
data = PolarImageDataset()

# print("Processing data...")
# inputs = image_processor(data[0][0], return_tensors="pt")


# Define the loss function - For classification problem
loss_function = nn.CrossEntropyLoss()

# Define your optimizer with weight decay
optimizer = optim.Adam(prediction_layer.parameters(), lr=0.001, weight_decay=0.0001)


train_dataloader = DataLoader(data, batch_size=10, shuffle=True)

for batch in train_dataloader:
    optimizer.zero_grad()
    inputs = image_processor(batch[0], return_tensors="pt")
    answer = softmax(prediction_layer(model(**inputs).pooler_output))
    
    # y_onehot = torch.zeros(10, 6)
    # y_onehot.scatter_(0, batch[1], 1)
    y_onehot = nn.functional.one_hot(batch[1], num_classes=6).double()
    
    loss = loss_function(answer, y_onehot)
    loss.backward()
    optimizer.step()

