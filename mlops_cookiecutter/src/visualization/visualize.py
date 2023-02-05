import argparse
from pathlib import Path
import os
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from sklearn.manifold import TSNE

def visualize_metrics(e, version,
                    train_steps, test_steps, 
                    train_losses, train_accuracies, 
                    test_losses, test_accuracies, 
                    model_dir="models", vis_dir='reports/figures/metrics'):
    """
    Visualizes and saves the figure of the loss and accuracy over time of the trained model.
    """

    if version == "latest":
        vis_version_dir = os.path.join(vis_dir, version)
    else:
        vis_version_dir = os.path.join(vis_dir, 'v{}'.format(version))

    fig, (ax1, ax2) = plt.subplots(2)

    ax1.plot(train_steps, train_losses[0:len(train_steps)], label="Train")
    ax1.plot(test_steps, test_losses[0:len(test_steps)], "-*", label="Validation")
    ax1.set_ylabel("NLL Loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(train_steps, train_accuracies[0:len(train_steps)], label="Train")
    ax2.plot(test_steps, test_accuracies[0:len(test_steps)], "-*", label="Validation")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()

    fig.supxlabel("step")

    fig.suptitle(f"Model: v{version} , Epoch: {e} , Step: {train_steps[-1]}")

    if not os.path.exists(vis_version_dir):
        os.makedirs(vis_version_dir)
    figdir = vis_version_dir + "/metrics.png"
    if version == "last":
        print(f"Saving figure in: {figdir}")
    else:
        print(f"Saving LATEST figure in: {figdir}")
    plt.savefig(figdir)
    plt.close()

def visualize_features(features,
                    model_dir='models',
                    vis_dir='reports/figures/features'):
    """
    Visualizes and saves the figure of the loss and accuracy over time of the trained model.
    """

    version = "latest"
    vis_version_dir = os.path.join(vis_dir, 'latest')

    fig, ax = plt.subplots()

    ax.scatter(features[:,0], features[:,1], s=1)
    ax.set_xlabel("Dimension #1")
    ax.set_ylabel("Dimension #2")
    ax.set_title("2D features from training data")

    if not os.path.exists(vis_version_dir):
        os.makedirs(vis_version_dir)
    figdir = vis_version_dir + "/features.png"
    print(f"Saving LATEST figure in: {figdir}")
    plt.savefig(figdir)
    plt.close()

def extract_features(model, dataloader):
    """
    Create predictions based on the latest version of the trained model.
    """

    feature_extractor = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)

    features_2d_list = []
    model.eval()
    for images, labels in dataloader:
        images = images.resize_(images.size()[0], 784)
        labels = labels.flatten()

        features, _ = model.forward(images)
        features = features.detach().numpy()

        features_2d = feature_extractor.fit_transform(features)
        features_2d_list.extend(features_2d.tolist())

    return np.array(features_2d_list)

def main(model_fpath, data_fpath):
    # load model
    model = MyModel()
    model = load_model(model, model_fpath)

    # load data
    data = load_data(data_fpath)
    data = preprocess(data)
    dataloader = data2dataloader(data)

    # extract features from the trainset
    print("Extracting features...")
    features = extract_features(model, dataloader)

    # visualize features
    print("Visualizing 2d features using TSNE...")
    visualize_features(features)

    return

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Make visualization on the trained data with the trained model.')
    parser.add_argument('model_fpath', nargs='?', type=str, default="models", help='the path to the trained model <model.pth>')
    parser.add_argument('data_fpath', nargs='?', type=str, default="data/raw/train_1.npz", help='the path to the .npz data')

    args = parser.parse_args()
    model_fpath = args.model_fpath
    data_fpath = args.data_fpath

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # append sibling dir to system path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    import_dir = os.path.join(current_dir, '..', 'models')
    sys.path.append(import_dir)

    # load functions from "models" sibling directory
    from utils.model_utils import load_model, get_latest_version
    from model import MyModel
    from predict_model import data2dataloader, load_model, load_data, preprocess

    main(model_fpath, data_fpath)
