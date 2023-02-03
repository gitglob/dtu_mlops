import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

import torch
import click

from data.data import mnist
from models.model import MyModel


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")

    # TODO: Implement training loop here
    model = MyModel()
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    # 5k, 1k datasets
    train_set, train_labels, _, _ = mnist()
    print(f"Length of dataset: {len(train_labels)}")
    batch_size = 50
    train_batch_set = np.split(train_set, int(len(train_set)/batch_size))
    train_batch_labels = np.split(train_labels, int(len(train_labels)/batch_size))
    num_epochs = 20
    print_every = 100
    steps = 0
    running_loss = 0
    running_correct = 0
    loss_list = []
    acc_list = []
    # Model in training mode, dropout is on
    model.train()
    for e in range(num_epochs):
        for i in range(500): #len(train_set)):
            image = train_set[i]
            label = train_labels[i]
            image = torch.from_numpy(image).float()
            label = torch.tensor(label)
            steps += 1
            
            # Flatten image into a 784 long vector
            image.resize(1, 784)
            
            optimizer.zero_grad()
            
            output = model.forward(image)
            # Model's output is log-softmax, take exponential to get the probabilities
            probs = torch.exp(output)
            _, predicted = torch.max(probs, dim=0)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_correct += (predicted == label).sum().item()

            if steps % print_every == 0:                
                print("Epoch: {}/{}.. ".format(e+1, num_epochs),
                      "Running Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Running Accuracy: {:.3f}..".format(running_correct/print_every))
                
                loss_list.append(running_loss/print_every)
                acc_list.append(running_correct/print_every)

                running_loss = 0
                running_correct = 0
                
    accuracy = acc_list[-1]
    loss = loss_list[-1]
    print(f"Final running loss: {loss}, \nAccuracy: {accuracy}")

    # save model
    torch.save(model.state_dict(), 'checkpoint.pth')

    # print model loss over time
    plt.plot(loss_list, label="loss")
    plt.title("Train: Loss vs step")
    plt.legend()
    plt.show()



@click.command()
@click.argument("model_checkpoint", default="checkpoint.pth")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyModel()
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)
    criterion = torch.nn.NLLLoss()

    _, _, test_set, test_labels = mnist()
    batch_size = 50
    test_batch_set = np.split(test_set, int(len(test_set)/batch_size))
    test_batch_labels = np.split(test_labels, int(len(test_labels)/batch_size))

    test_acc = 0
    test_loss = 0 
                
    # Turn off gradients for validation, will speed up inference
    model.eval()
    with torch.no_grad():
        running_correct = 0
        for i in range(len(test_set)):
            image = test_set[i]
            label = test_labels[i]
            image = torch.from_numpy(image).float()
            label = torch.tensor(label)

            image.resize(1, 784)

            output = model.forward(image)
            probs = torch.exp(output)
            _, predicted = torch.max(probs, dim=0)
            loss = criterion(output, label)

            test_loss += loss.item()

            ## Calculating the accuracy
            running_correct += (predicted == label).sum().item()

        accuracy = running_correct / len(test_labels)
        print("Test accuracy: ", accuracy)


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()

  