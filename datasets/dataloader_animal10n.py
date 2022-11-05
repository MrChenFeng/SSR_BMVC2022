from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


class animal_dataset(Dataset):
    def __init__(self, root, transform=None, mode='train'):
        train_path = os.listdir(os.path.abspath(root) + '/training')
        test_path = os.listdir(os.path.abspath(root) + '/testing')
        # print(train_path)
        print('Please be patient for image loading!')
        if mode == 'train':
            dir_path = os.path.abspath(root) + '/training'
            self.targets = [int(i.split('_')[0]) for i in train_path]
            self.data = [np.asarray(Image.open(dir_path + '/' + i)) for i in train_path]
        else:
            dir_path = os.path.abspath(root) + '/testing'
            self.targets = [int(i.split('_')[0]) for i in test_path]
            self.data = [np.asarray(Image.open(dir_path + '/' + i)) for i in test_path]
        print('Loading finished!')

        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index

    def update_labels(self, new_label):
        self.targets = new_label.cpu()

    def __len__(self):
        return len(self.targets)
