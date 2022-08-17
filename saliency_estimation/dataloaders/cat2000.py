"""Functions for loading 3D MRI data"""
from glob import glob
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import pandas as pd

class CAT2000(Dataset):
    """
    Dataset class for loading CAT2000 dataset
    """
    def __init__(self,
     csv_path,
     data_path,
     labels
    ) -> None:
        """
        init function
        Args:
        """
        self.data_path = data_path

    def __len__(self):
        """
        Return the length of the dataset
        """
        # TODO
        return None

    def __getitem__(self, idx: int):
        """
        Return an item
        Args:
            index (int): index of the item
        """
        # TODO
        idx_tensor = None
        idx_label = None
        return idx_tensor, idx_label


def build_cat2000(
    csv_path, data_path, labels, \
    val_ratio, num_workers, batch_size
):
    """
    # TODO
    """
    # Dataset
    dataset = CAT2000(csv_path, data_path, labels)
    # Number of samples
    val_size = int(len(dataset)*val_ratio)
    train_size = len(dataset) - val_size
    # Splitting the dataset
    train, val = random_split(dataset, [train_size, val_size])
    # Dataloaders
    train_dataloader = DataLoader(train, num_workers=num_workers, batch_size=batch_size)
    val_dataloader = DataLoader(val, num_workers=num_workers, batch_size=batch_size)
    return train_dataloader, val_dataloader

if __name__ == '__main__':
    # TODO: testing the dataloader
    pass
