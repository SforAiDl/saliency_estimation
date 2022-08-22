import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class MIT1003(Dataset):
    """
    Dataset class for loading MIT1003 dataset
    """

    def __init__(self, data_frame, root_dir, map_dir, transform=None) -> None:
        """
        init function
        Args:
        """
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.transform = transform
        self.map_dir = map_dir

    def __len__(self):
        """
        Return the length of the dataset
        """
        return len(self.data_frame)

    def __getitem__(self, idx):
        """
        Return an item
        Args:
            index (int): index of the item
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name)

        map_name = os.path.join(self.map_dir, self.data_frame.iloc[idx, 1])
        map_image = Image.open(map_name)

        if self.transform:
            image = self.transform(image)
            map_image = self.transform(map_image)
        return image, map_image


transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])


def build_mit1003(data_frame, root_dir, map_dir, transform, val_ratio, num_workers, batch_size):
    """
    Return training and validation dataloader
        Args:
            #TODO
    """
    # Dataset
    dataset = MIT1003(data_frame, root_dir, map_dir, transform)
    # Number of samples
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    # Splitting the dataset
    train, val = random_split(dataset, [train_size, val_size])
    # Dataloaders
    train_dataloader = DataLoader(train, num_workers=num_workers, batch_size=batch_size)
    val_dataloader = DataLoader(val, num_workers=num_workers, batch_size=batch_size)
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    # TODO: testing the dataloader
    pass
