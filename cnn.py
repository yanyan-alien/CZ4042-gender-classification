import numpy as np

import tqdm
import random
import pandas as pd
# import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from collections import OrderedDict

from dataprocessing import DataLoaderWrapper

data=DataLoaderWrapper(batch_size=32)
celebA_train,celebA_val,celebA_test=data.initialize_celebA_dataloaders()
adience=data.initialize_adience_dataloaders()


class Levi_Hassner(nn.Module):
    def __init__(self,output=2,deformable=False) -> None:
        super().__init__()
        self.deformable=deformable

        self.layers=nn.Sequential(OrderedDict([
            # first convolutional layer
            ('conv1',nn.Conv2d(3, 96, 7, padding='valid', stride=4)),  # No padding
            ('relu1',nn.ReLU()),
            ('maxpool1',nn.MaxPool2d(3, stride=2)),  # Max pooling over a (3, 3) window with 2 pixel stride)
            ('lrn1',nn.LocalResponseNorm(size=5, k=2, alpha=10**(-4), beta=0.75)),

            # second convolutional layer
            ('conv2',nn.Conv2d(96, 256, 5, padding='same')), # Same padding
            ('relu2',nn.ReLU()),
            ('maxpool2',nn.MaxPool2d(3, stride=2)),  # Max pooling over a (3, 3) window with 2 pixel stride)
            ('lrn2',nn.LocalResponseNorm(size=5, k=2, alpha=10**(-4), beta=0.75)),

            # third convolutional layer
            ('conv3',nn.Conv2d(256, 384, 3, padding='same')),  # Same padding
            ('relu3',nn.ReLU()),
            ('maxpool3',nn.MaxPool2d(3, stride=2)),  # Max pooling over a (3, 3) window with 2 pixel stride)
            ('flatten',nn.Flatten()),

            ('fc1',nn.Linear(384*6*6, 512)), # input 384 * 6 * 6 = 13824, output 512
            ('relu4',nn.ReLU()),
            ('dropout1',nn.Dropout(0.5)),

            ('fc2',nn.Linear(512,512)),
            ('relu5',nn.ReLU()),
            ('dropout2',nn.Dropout(0.5)),
            
            ('fc3',nn.Linear(512,output)), # output = number of classes 
        ]))
        self.prob=nn.Softmax(dim=1) # new stuff to check if its causes harm

    def forward(self,x):
        x=self.layers(x)
        prob=self.prob(x)
        return prob
    

model=Levi_Hassner()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 100

for epoch in range(EPOCHS):
    running_loss=0.0
    for i,data in enumerate(celebA_train):
        inputs, labels = data['images'],data['male']
        labels=torch.argmax(labels,dim=1)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print(running_loss)
        if i==2:break
        # if i % 2000 == 1999:    # print every 2000 mini-batches
        #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        #     running_loss = 0.0

torch.save(model.state_dict(), "celebA_cnn")