# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import os
import numpy as np
import torch
from torch import save, load
import torchvision.transforms as T
import torchvision.transforms.functional as F

def load_mnist(input_filepath):
    """
    Loads the train and test MNIST data from the data/raw folder
    """
    # alternative to move to the preset directory
    # os.chdir("../../data/raw")
    # datadir = os.getcwd()
    # print(f"Data director: {datadir}")

    print(f"\nLoading data from: {input_filepath}")
    data = np.load(input_filepath + "/train_1.npz")
    train_images = data["images"]
    print(f"Shape of training images: {train_images.shape}")
    train_labels = data["labels"]
    _ = data["allow_pickle"]

    data = np.load(input_filepath + "/test.npz")
    test_images = data["images"]
    print(f"Shape of test images: {test_images.shape}")
    test_labels = data["labels"]
    _ = data["allow_pickle"]

    return [train_images, train_labels, test_images, test_labels]

def normalize(image, m2=0, s2=1):
    """
    Function to normalize a given image with a specific mean and std. deviation
    """
    # current mean and std dev
    m1 = np.mean(image)
    s1 = np.std(image)

    image_norm = m2 + ((image - m1) * (s2/s1))

    return image_norm

def preprocess(data):
    """
    Convert numpy data (images) into pytorch tensors and normalizes them with a mean of 0 and std. deviation of 1.
    Source: https://stats.stackexchange.com/questions/46429/transform-data-to-desired-mean-and-standard-deviation
    """

    print("\nPreprocessing data: Converting to tensor and normalizing with μ=0 and σ=1")

    # transforms = T.Compose([ 
    #                   T.ToTensor(),
    #                   T.Normalize(
    #                     mean=[0.485, 0.456, 0.406],
    #                     std=[0.229, 0.224, 0.225])
    #             ])
    
    train_images, train_labels, test_images, test_labels = data

    # normalize features
    print("Preprocessing features")
    out_images = []
    for x in [train_images, test_images]:
        # current mean and std dev
        m1 = np.mean(x, axis=(1,2)).reshape(-1, 1)
        s1 = np.std(x, axis=(1,2)).reshape(-1, 1)
        # print(f"Shape of μ, σ vector (should be 5000x1): {np.shape(m1), np.shape(s1)}")
        # desired mean and std dev
        m2 = 0
        s2 = 1

        normalizer = lambda xi, mi, si: m2 + ((xi - mi) * (s2/si))
        x_norm = np.array([normalizer(x_i, m_i, s_i) for (x_i, m_i, s_i) in zip(x, m1, s1)])
        m2 = np.mean(x_norm, axis=(1,2)).reshape(-1, 1)
        s2 = np.std(x_norm, axis=(1,2)).reshape(-1, 1)
        # print(f"Shape of μ, σ vector (should be 5000x1): {np.shape(m2), np.shape(s2)}")

        out_images.append(x)

    # normalize labels
    print("Preprocessing labels")
    out_labels = []
    for x in [train_labels, test_labels]:
        # current mean and std dev
        m1 = np.mean(x).reshape(-1, 1)
        s1 = np.std(x).reshape(-1, 1)
        # print(f"Shape of μ, σ vector (should be 1x1): {np.shape(m1), np.shape(s1)}")
        # desired mean and std dev
        m2 = 0
        s2 = 1

        x_norm = m2 + ((x - m1) * (s2/s1))
        x_norm = x.reshape(-1, 1)
        m2 = np.mean(x_norm).reshape(-1, 1)
        s2 = np.std(x_norm).reshape(-1, 1)
        # print(f"Shape of μ, σ vector (should be 1x1): {np.shape(m2), np.shape(s2)}")

        out_labels.append(x)

    return [out_images[0], out_labels[0], out_images[1], out_labels[1]]


def save_data(data, output_filepath):
    """
    Saves normalized pytorch tensor to the specified filepath.
    """

    print(f"\nSaving data at: {output_filepath}")
    train_images, train_labels, test_images, test_labels = data
    torch.save(train_images, output_filepath + "/train_images.pt")
    torch.save(train_labels, output_filepath + "/train_labels.pt")
    torch.save(test_images, output_filepath + "/test_images.pt")
    torch.save(test_labels, output_filepath + "/test_labels.pt")


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    data = load_mnist(input_filepath)
    data = preprocess(data)
    save_data(data, output_filepath)

    return 



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
