from __future__ import print_function

import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageFilter


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def D(p, z, version='simplified'):  # negative cosine similarity
    if version == 'original':
        z = z.detach()  # stop gradient
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return -(p * z).sum(dim=1).mean()

    elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    # print(sim_weight.shape, sim_labels.shape)
    sim_weight = torch.ones_like(sim_weight)

    sim_weight = sim_weight / sim_weight.sum(dim=-1, keepdim=True)

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    # print(one_hot_label.shape)
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    # print(pred_scores.shape)
    pred_labels = pred_scores.argmax(dim=-1)
    return pred_scores, pred_labels

def weighted_knn(cur_feature, feature, label, num_classes, knn_k=100, chunks=10, norm='global'):
    # distributed fast KNN and sample selection with three different modes
    num = len(cur_feature)
    num_class = torch.tensor([torch.sum(label == i).item() for i in range(num_classes)]).to(
        feature.device) + 1e-10
    pi = num_class / num_class.sum()
    split = torch.tensor(np.linspace(0, num, chunks + 1, dtype=int), dtype=torch.long).to(feature.device)
    score = torch.tensor([]).to(feature.device)
    pred = torch.tensor([], dtype=torch.long).to(feature.device)
    feature = torch.nn.functional.normalize(feature, dim=1)
    with torch.no_grad():
        for i in range(chunks):
            torch.cuda.empty_cache()
            part_feature = cur_feature[split[i]: split[i + 1]]

            part_score, part_pred = knn_predict(part_feature, feature.T, label, num_classes, knn_k)
            score = torch.cat([score, part_score], dim=0)
            pred = torch.cat([pred, part_pred], dim=0)

        # balanced vote
        if norm == 'global':
            # global normalization
            score = score / pi
        else:  # no normalization
            pass
        score = score/score.sum(1, keepdim=True)

    return score  # , pred

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_config(args, path):
    dict = vars(args)
    if not os.path.isdir(path):
        os.mkdir(os.path.abspath(path))
    with open(path + '/params.csv', 'w') as f:
        for key in dict.keys():
            f.write("%s\t%s\n" % (key, dict[key]))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class KCropsTransform:
    """Take K random crops of one image as the query and key."""

    def __init__(self, base_transform, K=2):
        self.base_transform = base_transform
        self.K = K

    def __call__(self, x):
        res = [self.base_transform(x) for i in range(self.K)]
        return res


class MixTransform:
    def __init__(self, strong_transform, weak_transform, K=2):
        self.strong_transform = strong_transform
        self.weak_transform = weak_transform
        self.K = K

    def __call__(self, x):
        res = [self.weak_transform(x) for i in range(self.K)] + [self.strong_transform(x) for i in range(self.K)]
        return res


class DMixTransform:
    def __init__(self, transforms, nums):
        self.transforms = transforms
        self.nums = nums

    def __call__(self, x):
        res = []
        for i, trans in enumerate(self.transforms):
            res += [trans(x) for _ in range(self.nums[i])]
        return res


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
