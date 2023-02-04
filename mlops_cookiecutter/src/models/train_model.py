import click
import os
from typing import Callable, Tuple, Union, Optional, List
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from model import MyModel

class MnistDataset(Dataset):
    """
    Dataloader for the MNIST dataset based on numpy arrays.
    """
    def __init__(self, images, labels):
        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).long()
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def validation(
    model: nn.Module, 
    testloader: torch.utils.data.DataLoader, 
    criterion: Union[Callable, nn.Module]
) -> Tuple[float, float]:
    accuracy = 0
    test_loss = 0
    for images, labels in testloader:

        images = images.resize_(images.size()[0], 784)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy 
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean().item()

    return test_loss, accuracy


def train(
    model: nn.Module, 
    trainloader: torch.utils.data.DataLoader, 
    testloader: torch.utils.data.DataLoader, 
    criterion: Union[Callable, nn.Module], 
    optimizer: Optional[torch.optim.Optimizer] = None, 
    epochs: int = 5, 
    print_every: int = 40,
) -> None:

    # check which version of the model we are running
    model_relative_dir = "../../models"
    version = get_latest_version(model_relative_dir) + 1

    # train the network
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    steps = 0
    running_loss = 0
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for images, labels in trainloader:
            steps += 1
            
            # Flatten images into a 784 long vector
            images.resize_(images.size()[0], 784)
            
            optimizer.zero_grad()
            
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()
                
                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion)
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
                
                running_loss = 0
                
                # save the model every logging period
                save_model(model, version)

                # Make sure dropout and grads are on for training
                model.train()


def load_tensors():
    """
    Loads the saved tensors in data/processed
    """

    # alternative to move to the preset directory
    os.chdir("../../data/processed")
    datadir = os.getcwd()
    print(f"Loading data from: {datadir}")

    train_images = torch.load('train_images.pt')
    train_labels = torch.load('train_labels.pt')
    test_images = torch.load('test_images.pt')
    test_labels = torch.load('test_labels.pt')

    # Create the custom dataset object
    train_dataset = MnistDataset(train_images, train_labels)
    test_dataset = MnistDataset(test_images, test_labels)

    # Create the DataLoader
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    return trainloader, testloader

def save_model(model, version, model_dir='../../models'):
    """
    Saves the model in a subfolder in the "models" folder that is named after the version of the model (i.e. v1, v2, v3 etc...)
    """
    version_dir = os.path.join(model_dir, 'v{}'.format(version))
    if not os.path.exists(version_dir):
        os.makedirs(version_dir)
    model_path = os.path.join(version_dir, 'model.pth')
    print(f"Saving model in directory: {model_path}")
    torch.save(model.state_dict(), model_path)
    

def load_model(model, model_dir='../../models'):
    """
    Loads the last saved version of the model, or if there is none, does nothing.
    """
    print(f"Loading model from directory: {model_dir}")
    version = get_latest_version(model_dir)
    if version > -1:
        model_path = os.path.join(model_dir, 'v{}'.format(version), 'model.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
        else:
            print("Warning: model version v{} does not exist. Unable to load model.".format(version))
    else:
        print("Warning: no previous version of the model exists. Creating an empty (v0) model...".format(version))
    
    return model

def get_latest_version(model_dir):
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

def main():
    # load model
    model = MyModel()
    model = load_model(model)

    # load data
    trainloader, testloader = load_tensors()

    # define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # define number of epochs and log frequency
    epochs = 10
    print_every = 500

    # train model
    train(model, trainloader, testloader, criterion, optimizer, epochs, print_every)



if __name__ == '__main__':
    main()