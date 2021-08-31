import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
from torch import Tensor, int32

class Dataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, path_data, trainsize, augmentations):
        self.trainsize = trainsize
        self.augmentations = augmentations
        print(self.augmentations)
        # fix file train
        self.data_raw = open(path_data,"r")
        self.data_raw=self.data_raw.readlines()
        #print(self.data_raw)
        self.images=[p.split('\t')[0] for p in self.data_raw]
        self.labels=[int(p.split('\t')[1].split('\n')[0]) for p in self.data_raw]
        #print(self.labels)
        #self.labels=[1 for p in self.data_raw]
        self.labels=torch.LongTensor(self.labels)
        #print(self.labels)
        #self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
            

    def __getitem__(self, index):
        
        image = self.rgb_loader(self.images[index])
        #gt = self.binary_loader(self.gts[index])
        label=self.labels[index]
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)
            
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        # if self.gt_transform is not None:
        #     gt = self.gt_transform(gt)
        return (image, label)

    def filter_files(self):
        assert len(self.images) == len(self.labels)
        images = []
        labels = []
        for img_path, label in zip(self.images, self.labels):
            images.append(img_path)
            labels.append(label)
        self.images = images
        self.labels = labels

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


    def resize(self, img,label):
        return (img.resize((self.trainsize, self.trainsize), Image.BILINEAR),label)
    
    def __len__(self):
        return self.size


def get_loader(path_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True, augmentation=False):

    dataset = Dataset(path_root, trainsize, augmentation)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


