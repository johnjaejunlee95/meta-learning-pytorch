from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import os
import numpy as np
import pickle

ROOT_PATH = '/data01/jjlee_hdd/data/tieredimagenet/'


class TieredImageNet(Dataset):
    def __init__(self, split):
        
        self.ds_name = ROOT_PATH 
        self.split = split

        pkl_path = os.path.join(self.ds_name, "{}_labels.pkl".format(split))
        images_path = os.path.join(self.ds_name, "{}_images.pkl".format(split))
        
        labels = load_data(pkl_path)
        data_label = labels['labels']      
        label_set = set(data_label)
        label_dict = dict(zip(label_set, range(len(label_set))))
        self.labels = [label_dict[x] for x in data_label]
        
        with open(images_path, 'rb') as f:
            self.samples = pickle.load(f)
        
        
        mean, std = [0.4721, 0.4533, 0.4099], [0.2771, 0.2677, 0.2844]
        normalize_transform = transforms.Normalize(mean=mean,  std=std)

        self.transform = None
    
        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            normalize_transform,
        ])

    def __getitem__(self, index):
        image, label = self.samples[index], self.labels[index]
        image = Image.fromarray(image)
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)

    def get_label_num(self):
        return len(set(self.labels))

def load_data(file):
    try:
        with open(file, 'rb') as fo:

            data = pickle.load(fo)

        return data
    except Exception as e:
        print(e)
        with open(file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        return data