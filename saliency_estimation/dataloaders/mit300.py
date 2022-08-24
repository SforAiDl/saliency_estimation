import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MIT300(Dataset):
    """
    Dataset class for loading MIT300 dataset
    """

    def __init__(self, csv_path, root_dir, transform=None) -> None:
        """
        init function
        Args:
        """
        self.csv_path = csv_path
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """
        Return the length of the dataset
        """
        return len(self.csv_path)

    def __getitem__(self, idx):
        """
        Return an item
        Args:
            index (int): index of the item
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.csv_path.iloc[idx])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)
        return image


transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])


def build_mit300(csv_path, root_dir, transform, num_workers, batch_size):
    """
    Return training and validation dataloader
        Args:
            #TODO
    """
    # Dataset
    dataset = MIT300(csv_path, root_dir, transform)
    # Dataloaders
    test_dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size)
    return test_dataloader


if __name__ == "__main__":
    # TODO: testing the dataloader
    pass
