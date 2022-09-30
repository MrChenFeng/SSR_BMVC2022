from torch.utils.data.sampler import *
import torch

class ClassBalancedSampler(Sampler[int]):

    def __init__(self, labels, num_classes, num_samples=None, num_fold=1):
        # self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_fold = num_fold
        self.labels = torch.as_tensor(labels, dtype=torch.int)
        self.classes = torch.arange(num_classes)
        # self.num_classes = torch.as_tensor([torch.sum(self.labels == torch.as_tensor(i)) for i in self.classes], dtype=torch.int)
        self.num_classes = torch.as_tensor([torch.sum(self.labels == i) for i in self.classes], dtype=torch.int)
        # print(self.num_classes.max(), self.num_classes)
        if num_samples is not None and num_fold is None:
            self.num_fold = torch.floor(torch.tensor(num_samples / len(labels)))
        self.max_num = self.num_classes.max() * self.num_fold
        ids = []
        # print(self.max_num)
        for i, cid in enumerate(self.classes):
            if self.num_classes[i] == 0:
                continue
            else:
                fold_i = torch.ceil(self.max_num / self.num_classes[i]).to(torch.int)
            # print(fold_i)
            tmp_i = torch.where(self.labels == cid)[0].repeat(fold_i)  # [:self.max_num]
            # print(tmp_i)
            # extra = torch.fmod(self.max_num, self.num_classes[i]).to(torch.int)
            # full = self.max_num - extra
            rand = torch.randperm(self.num_classes.max())
            # print(extra, full)
            tmp_i[-self.num_classes.max():] = tmp_i[-self.num_classes.max():][rand]
            ids.append(tmp_i[:self.max_num])
        self.ids = torch.cat(ids)

    def __iter__(self):
        rand = torch.randperm(len(self.ids))
        ids = self.ids[rand]
        # rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement)
        return iter(ids.tolist())

    def __len__(self):
        return len(self.ids)
