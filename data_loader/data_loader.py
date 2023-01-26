from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader

from . import dataset
from base.base_data_loader import BaseDataLoader


class CustomDataLoader(BaseDataLoader):
    """
    Custom data loader for image deblurring
    """

    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers):
        transform = transforms.Compose([
            transforms.ToTensor(),  # convert to tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize
        ])
        self.dataset = dataset.CustomDataset(data_dir, transform=transform)

        super(CustomDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
