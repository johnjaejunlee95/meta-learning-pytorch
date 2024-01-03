import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import os.path as osp
from PIL import Image
import os

ROOT_PATH = '/data01/jjlee_hdd/data/miniimagenet/'


class MiniImageNet(Dataset):

    def __init__(self, setname):
        csv_path = osp.join(ROOT_PATH + "splits/",  setname + '.csv')
        images_path = os.path.join(ROOT_PATH, "images")
        
        lines = [x.strip() for x in open(csv_path).readlines()[1:]]

        self.samples = []
        self.labels = []
        label_dict = {}
        label_index = 0
        self.label_index_2_name = []
        for e in lines:
            image_name, label_name = e.split(",")
            if label_name not in label_dict:
                label_dict[label_name] = label_index
                label_index += 1
                self.label_index_2_name.append(label_name)

            self.samples.append(os.path.join(images_path, image_name))
            self.labels.append(label_dict[label_name])

        # self.data = data
        # self.label = label

        self.transform = transforms.Compose([
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.labels)
    
    def get_label_num(self):
        return len(set(self.labels))
    
    def __getitem__(self, index):

        image_path, label = self.samples[index], self.labels[index]
        image = self.transform(Image.open(image_path).convert('RGB'))
        return image, label
