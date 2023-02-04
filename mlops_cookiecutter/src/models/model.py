from torch import nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        # self.pshape(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # self.pshape(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # self.pshape(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        # self.pshape(x)
        x = F.log_softmax(self.fc4(x), dim=0)
        # self.pshape(x)
        
        return x

    def pshape(self, x):
        print("Shape: ", x.shape)
