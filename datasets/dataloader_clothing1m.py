from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch


class clothing_dataset(Dataset):
    def __init__(self, root_dir, transform, dataset_mode, subset=None, num_samples=0, paths=[], labels=None, num_class=14):

        self.root = root_dir
        self.transform = transform
        self.mode = dataset_mode
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}

        with open('%s/noisy_label_kv.txt' % self.root, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = '%s/' % self.root + entry[0]  # [7:]
                self.train_labels[img_path] = int(entry[1])
        with open('%s/clean_label_kv.txt' % self.root, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = '%s/' % self.root + entry[0]  # [7:]
                self.test_labels[img_path] = int(entry[1])

        # started random selected evaluate samples
        if self.mode == 'eval': # select part of samples for subsequent training [clothing1m only]
            train_imgs = []
            with open('%s/noisy_train_key_list.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/' % self.root + l  # [7:]
                    train_imgs.append(img_path)
            # select same samples always
            random.shuffle(train_imgs)

            class_num = torch.zeros(num_class)
            self.train_imgs = []
            for impath in train_imgs:
                label = self.train_labels[impath]
                if class_num[label] < (num_samples / 14) and len(self.train_imgs) < num_samples:
                    self.train_imgs.append(impath)
                    class_num[label] += 1
            random.shuffle(self.train_imgs)

        elif self.mode == "unlabeled":
            self.train_imgs = paths

        elif self.mode == "train":
            train_imgs = paths
            self.train_imgs = [train_imgs[i] for i in subset]
            # self.train_imgs = paths[subset]
            self.semi_labels = labels[subset].cpu()
            class_num = [torch.sum(self.semi_labels == i) for i in range(14)]

        elif self.mode == 'test':
            self.test_imgs = []
            with open('%s/clean_test_key_list.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/' % self.root + l  # [7:]
                    self.test_imgs.append(img_path)


    def __getitem__(self, index):
        if self.mode == 'train':
            img_path = self.train_imgs[index]
            # target = self.train_labels[img_path]
            target = self.semi_labels[index]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            # return img, target, img_path, index
            return img, target, index
        elif self.mode == 'unlabeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]

            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            # return img, target, img_path, index
            return img, target, index
        elif self.mode == 'eval':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target, img_path
        elif self.mode == 'test':
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target, img_path

    def __len__(self):
        if self.mode == 'test':
            return len(self.test_imgs)
        else:
            return len(self.train_imgs)
