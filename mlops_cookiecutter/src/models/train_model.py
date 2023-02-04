import click
import os
from typing import Callable, Tuple, Union, Optional, List
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from utils.model_utils import get_latest_version, visualize_metrics
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
    model_relative_dir = "models"
    version = get_latest_version(model_relative_dir) + 1

    # train the network
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    step = 0
    train_steps = []
    test_steps = []
    running_loss = 0
    running_correct = 0
    running_tot = 0
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    try:
        for e in range(epochs):
            # Model in training mode, dropout is on
            model.train()
            for images, labels in trainloader:
                train_steps.append(step)
                
                # Flatten images into a 784 long vector
                images.resize_(images.size()[0], 784)
                
                optimizer.zero_grad()
                
                output = model.forward(images)
                ps = torch.exp(output)
                _, predicted = torch.max(ps, dim=1)

                # calculate loss
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                
                # calculate running loss
                running_loss += loss.item()
                
                # calculate running correct correct and total predictions
                running_correct += (predicted == labels).sum().item()
                running_tot += len(labels)

                # log current training loss and accuracy
                train_losses.append(loss.item())
                train_accuracies.append((predicted == labels).sum().item() / len(labels))


                if step % print_every == 0:
                    test_steps.append(step)
                    # Model in inference mode, dropout is off
                    model.eval()
                    
                    # Turn off gradients for validation, will speed up inference
                    with torch.no_grad():
                        test_loss, accuracy = validation(model, testloader, criterion)
                    
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                        "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                        "Training Accuracy: {:.3f}.. ".format(running_correct/running_tot),
                        "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                        "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

                    # log current loss and accuracy
                    test_losses.append(test_loss/print_every)
                    test_accuracies.append(accuracy/len(testloader))
                                
                    # save the model every logging period
                    save_model(model, version)

                    # visualize and save the loss and accuracy thus far
                    visualize_metrics(e, print_every, 
                                    train_steps, test_steps, 
                                    train_losses, train_accuracies, 
                                    test_losses, test_accuracies)

                    # set running training loss and number of correct predictions to 0
                    running_loss = 0
                    running_correct = 0
                    running_tot = 0

                    # Make sure dropout and grads are on for training
                    model.train()

                # increase the step        
                step += 1
    # if the training ends or is interrupted, we want the model and the last visualizations to be saved in the "latest" folder
    except KeyboardInterrupt:
        save_latest_model(model)
        visualize_metrics(e, print_every, 
                        train_steps, test_steps, 
                        train_losses, train_accuracies, 
                        test_losses, test_accuracies,
                        last = True)

    save_latest_model(model)
    visualize_metrics(e, print_every, 
                    train_steps, test_steps, 
                    train_losses, train_accuracies, 
                    test_losses, test_accuracies,
                    last = True)

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
    if version > -1:
        model_path = os.path.join(model_dir, 'v{}'.format(version), 'model.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
        else:
            print("Warning: model version v{} does not exist. Unable to load model.".format(version))
    else:
        print("Warning: no previous version of the model exists. Creating an empty (v0) model...".format(version))
    
    return model

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