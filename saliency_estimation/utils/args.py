"""Utilities for passing arguments using argparse"""
import argparse

def str2bool(text):
    """
    Function to convert string to boolean
    """
    if isinstance(text, bool):
        return text
    if text.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif text.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2list(text):
    """
    Function to convert string to list of strings
    items should be comma separated
    """
    if not isinstance(text, list):
        text = text.replace(' ', '')
        text = text.split(',')
    return text

def str2none(text):
    """
    Function to convert a string to NoneType
    """
    if text.lower() in ('none', 'n'):
        return None
    else:
        return text
