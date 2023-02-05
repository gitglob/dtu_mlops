from numpy import arange
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import os

class MnistDataset(Dataset):
    """
    Dataloader for the MNIST dataset based on numpy arrays.
    """
    def __init__(self, images, labels=None, inference=False):
        self.inference = inference
        self.images = torch.from_numpy(images).float()
        if not inference:
            self.labels = torch.from_numpy(labels).long()
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if not self.inference:
            return self.images[idx], self.labels[idx]
        else:
            return self.images[idx]


def get_latest_version(model_dir="models"):
    """
    Gets the latest version of the model.
    If no previous version exists, it returns -1.
    """
    version = -1
    for folder in os.listdir(model_dir):
        if folder.startswith('v'):
            curr_version = int(folder[1:])
            version = max(version, curr_version)

    return version

def save_latest_model(model, model_dir='models'):
    """
    Saves the latest model in the "latest" fubfolder in the "models" folder
    """
    version = get_latest_version()
    if version > -1:
        latest_model_path = os.path.join(model_dir, 'latest')
        if not os.path.exists(latest_model_path):
            os.makedirs(latest_model_path)
        print(f"Saving LATEST model in directory: {latest_model_path}/model.pth")
        torch.save(model.state_dict(), latest_model_path + "/model.pth")

def load_tensors(datadir="data/processed"):
    """
    Loads the saved tensors in data/processed
    """

    # alternative to move to the preset directory
    print(f"Loading data from: {datadir}")

    train_images = torch.load(datadir + '/train_images.pt')
    train_labels = torch.load(datadir + '/train_labels.pt')
    test_images = torch.load(datadir + '/test_images.pt')
    test_labels = torch.load(datadir + '/test_labels.pt')

    # Create the custom dataset object
    train_dataset = MnistDataset(train_images, train_labels)
    test_dataset = MnistDataset(test_images, test_labels)

    # Create the DataLoader
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    return trainloader, testloader

def save_model(model, version, model_dir='models'):
    """
    Saves the model in a subfolder in the "models" folder that is named after the version of the model (i.e. v1, v2, v3 etc...)
    """
    version_dir = os.path.join(model_dir, 'v{}'.format(version))
    if not os.path.exists(version_dir):
        os.makedirs(version_dir)
    model_path = os.path.join(version_dir, 'model.pth')
    print(f"Saving model in directory: {model_path}")
    torch.save(model.state_dict(), model_path)
    

def load_model(model, model_dir='models'):
    """
    Loads the last saved version of the model, or if there is none, does nothing.
    """
    print(f"Loading model from directory: {model_dir}")

    version = get_latest_version(model_dir)
    latest_dir = model_dir + "/latest"
    if os.path.exists(latest_dir):
        print(f"Loading the latest model. This is likely version: v{version}")
        model_path = os.path.join(model_dir, 'latest', 'model.pth')
        model.load_state_dict(torch.load(model_path))
    else:
        print("Loading the newest version, the <latest> subfolder does not exist!")
        if version > -1:
            model_path = os.path.join(model_dir, 'v{}'.format(version), 'model.pth')
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path))
            else:
                print("Warning: model version v{} does not exist. Unable to load model.".format(version))
        else:
            print("Warning: no previous version of the model exists. Creating an empty (v0) model...".format(version))
    
    return model

