import deeplake
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models

ds_adience=deeplake.load('hub://activeloop/adience')

ds_celebA_train = deeplake.load("hub://activeloop/celeb-a-train")
ds_celebA_val = deeplake.load("hub://activeloop/celeb-a-val")
ds_celebA_test = deeplake.load("hub://activeloop/celeb-a-test")

tform={
    'train': transforms.Compose([
    transforms.ToPILImage(), # Must convert to PIL image for subsequent operations to run
    transforms.Resize((227,227)),
    transforms.RandomRotation(20), # Image augmentation
    transforms.RandomHorizontalFlip(p=0.5), # Image augmentation
    transforms.ToTensor(), # Must convert to pytorch tensor for subsequent operations to run
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]),
    'test': transforms.Compose([
    transforms.ToPILImage(), # Must convert to PIL image for subsequent operations to run
    transforms.Resize((227,227)),
    transforms.ToTensor(), # Must convert to pytorch tensor for subsequent operations to run
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
}

batch_size=32

celebA_train_dataloader = ds_celebA_train.pytorch(batch_size=batch_size, num_workers=0, transform={'images':tform['train'],'male':None,'young':None}, shuffle=True)
celebA_val_dataloader = ds_celebA_val.pytorch(batch_size=batch_size, num_workers=0, transform={'images':tform['test'],'male':None,'young':None}, shuffle=True)
celebA_test_dataloader = ds_celebA_test.pytorch(batch_size=batch_size, num_workers=0, transform={'images':tform['test'],'male':None,'young':None}, shuffle=True)

#PyTorch Dataloader
adience_dataloader=ds_adience.pytorch(batch_size=32, num_workers=0, transform={'images':tform['test'], 'genders':None, 'ages':None}, shuffle=True)