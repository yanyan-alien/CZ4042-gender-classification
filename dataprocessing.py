import deeplake
from torchvision import transforms

class DataLoaderWrapper:
    def __init__(self,batch_size=32, image_size=227) -> None:
        self.batch_size=batch_size
        self.tform={
                        'train': transforms.Compose([
                        transforms.ToPILImage(), # Must convert to PIL image for subsequent operations to run
                        transforms.Resize((image_size,image_size)),
                        transforms.RandomRotation(20), # Image augmentation
                        transforms.RandomHorizontalFlip(p=0.5), # Image augmentation
                        transforms.ToTensor(), # Must convert to pytorch tensor for subsequent operations to run
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                        ]),
                        'test': transforms.Compose([
                        transforms.ToPILImage(), # Must convert to PIL image for subsequent operations to run
                        transforms.Resize((image_size,image_size)),
                        transforms.ToTensor(), # Must convert to pytorch tensor for subsequent operations to run
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                        ])
                   }
        
        self.ds_adience_train = deeplake.load("hub://activeloop/adience-train", reset=True)
        self.ds_adience_val = deeplake.load("hub://activeloop/adience-val", reset=True)
        self.ds_adience_test = deeplake.load("hub://activeloop/adience-test", reset=True)

        self.ds_celebA_train = deeplake.load("hub://activeloop/celeb-a-train", reset=True)
        self.ds_celebA_val = deeplake.load("hub://activeloop/celeb-a-val", reset=True)
        self.ds_celebA_test = deeplake.load("hub://activeloop/celeb-a-test", reset=True)

    def initialize_celebA_dataloaders(self):
        celebA_train_dataloader = self.ds_celebA_train.pytorch(batch_size=self.batch_size, num_workers=0, transform={'images':self.tform['train'],'male':None,'young':None}, shuffle=True)
        celebA_val_dataloader = self.ds_celebA_val.pytorch(batch_size=self.batch_size, num_workers=0, transform={'images':self.tform['test'],'male':None,'young':None}, shuffle=True)
        celebA_test_dataloader = self.ds_celebA_test.pytorch(batch_size=self.batch_size, num_workers=0, transform={'images':self.tform['test'],'male':None,'young':None}, shuffle=True)
        return celebA_train_dataloader,celebA_val_dataloader,celebA_test_dataloader

    def initialize_adience_dataloaders(self):
        adience_train_dataloader = self.ds_adience.pytorch(batch_size=self.batch_size, num_workers=0, transform={'images':self.tform['train'], 'genders':None, 'ages':None}, shuffle=True)
        adience_val_dataloader = self.ds_adience.pytorch(batch_size=self.batch_size, num_workers=0, transform={'images':self.tform['test'], 'genders':None, 'ages':None}, shuffle=True)
        adience_test_dataloader = self.ds_adience.pytorch(batch_size=self.batch_size, num_workers=0, transform={'images':self.tform['test'], 'genders':None, 'ages':None}, shuffle=True)
        return adience_train_dataloader, adience_val_dataloader, adience_test_dataloader