"""Utilities for manipulating paths"""
import os


def create_folders(pth):
    """
    Create nested folders
    Args:
        pth(str): Path to be created
    """
    folders = pth.split("/")
    cur_pth = os.getcwd()
    for fldr in folders:
        if fldr != ".." and fldr not in os.listdir(cur_pth):
            os.mkdir(cur_pth + "/" + fldr)
        cur_pth = cur_pth + "/" + fldr
