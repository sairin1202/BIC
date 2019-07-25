import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import skimage.io as io
import glob
import numpy as np
import random
import torch

class BatchData(Dataset):
    def __init__(self, images, labels, input_transform=None):
        self.images = images
        self.labels = labels
        self.input_transform = input_transform

    def __getitem__(self, index):
        # print("1",self.images[index].shape)
        image = self.images[index]
        image = Image.fromarray(np.uint8(image))
        # print("3",image.size)
        label = self.labels[index]
        if self.input_transform is not None:
            image = self.input_transform(image)
        # print("4",image.size())
        label = torch.LongTensor([label])
        return image, label

    def __len__(self):
        return len(self.images)
