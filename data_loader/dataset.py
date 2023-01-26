import os
import random
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """Custom dataset for image processing"""

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.image_names = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        input_image = Image.open(os.path.join(self.data_dir, image_name)).convert('RGB')
        height = input_image.size[1]
        width = input_image.size[0]
        new_height = height - height % 4 + 4 if height % 4 != 0 else height
        new_w = width - width % 4 + 4 if width % 4 != 0 else width
        input_image = transforms.Resize([new_height, new_w], Image.BICUBIC)(input_image)

        if self.transform:
            input_image = self.transform(input_image)

        return {'input_image': input_image, 'image_name': image_name}
