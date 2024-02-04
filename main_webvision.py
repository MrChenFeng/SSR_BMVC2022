'''
Update 2022.11.5
To ensure a direct minimal reproduction applicable, I removed the codes on imagenet evaluation. [Marked with *]
To evaluate on iamgenet, please modify the code yourself
'''
import argparse

import torchvision.transforms as transforms
import wandb
from torch.optim import SGD
from torch.utils.data import Subset
from tqdm import tqdm

from datasets.dataloader_webvision import miniwebvision_dataset, imagenet_dataset
from models.inceptionresnetv2 import InceptionResNetV2
from utils import *

parser = argparse.ArgumentParser('Train with Webvision dataset')
parser.add_argument('--dataset_path', default='~/WebVision', help=f'dataset path')
#*************************************************************************************************************#
# parser.add_argument('--imagenet_path', default='~/ImageNet', help=f'dataset_path for imagenet evaluation')
#*************************************************************************************************************#

# model settings
parser.add_argument('--lambda_fc', default=1, type=float, metavar='N', help='weight of unlabeled data (default: 1)')
parser.add_argument('--theta_s', default=1.0, type=float, help='Initial threshold for voted correct samples (default: 1.0)')
parser.add_argument('--theta_r', default=0.95, type=float, help='threshold for relabel samples (default: 0.8)')
parser.add_argument('--k', default=200, type=int, help='neighbors for soft-voting (default: 200)')

# train settings
parser.add_argument('--epochs', default=150, type=int, metavar='N', help='number of total epochs to run (default: 300)')
parser.add_argument('--batch_size', default=32, type=int, help='mini-batch size (default: 32)')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate (default: 0.1)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver (default: 0.9)')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--seed', default=3047, type=int, help='seed for initializing training. (default: 3047)')
parser.add_argument('--parallel', default=0, action='store_true', help='Multi-GPU training (default: False)')
parser.add_argument('--gpuid', default='0', type=str, help='Selected GPU (default: "0")')
parser.add_argument('--entity', type=str, help='Wandb user entity')
parser.add_argument('--run_path', type=str, help='run path containing all results')


def train(labeled_trainloader, modified_label, all_trainloader, encoder, classifier, proj_head, pred_head, optimizer, epoch, args):
    encoder.train()
    classifier.train()
    proj_head.train()
    pred_head.train()
    xlosses = AverageMeter('xloss')
    ulosses = AverageMeter('uloss')
    labeled_train_iter = iter(labeled_trainloader)
    all_bar = tqdm(all_trainloader)
    for batch_idx, ([inputs_u1, inputs_u2], _, _) in enumerate(all_bar):
        try:
            # [inputs_x1, inputs_x2], labels_x, index = labeled_train_iter.next()
            [inputs_x1, inputs_x2], labels_x, index = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            # [inputs_x1, inputs_x2], labels_x, index = labeled_train_iter.next()
            [inputs_x1, inputs_x2], labels_x, index = next(labeled_train_iter)

        # cross-entropy training with mixup
        batch_size = inputs_x1.size(0)
        inputs_x1, inputs_x2 = inputs_x1.cuda(), inputs_x2.cuda()
        labels_x = modified_label[index]
        targets_x = torch.zeros(batch_size, args.num_classes, device=inputs_x1.device).scatter_(1, labels_x.view(-1, 1), 1)
        l = np.random.beta(0.5, 0.5)
        l = max(l, 1 - l)
        all_inputs_x = torch.cat([inputs_x1, inputs_x2], dim=0)
        all_targets_x = torch.cat([targets_x, targets_x], dim=0)
        idx = torch.randperm(all_inputs_x.size()[0])
        input_a, input_b = all_inputs_x, all_inputs_x[idx]
        target_a, target_b = all_targets_x, all_targets_x[idx]
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        logits = classifier(encoder(mixed_input))
        Lce = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))

        # optional feature-consistency
        inputs_u1, inputs_u2 = inputs_u1.cuda(), inputs_u2.cuda()

        feats_u1 = encoder(inputs_u1)
        feats_u2 = encoder(inputs_u2)
        f, h = proj_head, pred_head

        z1, z2 = f(feats_u1), f(feats_u2)
        p1, p2 = h(z1), h(z2)
        Lfc = D(p2, z1)
        loss = Lce + args.lambda_fc * Lfc
        xlosses.update(Lce.item())
        ulosses.update(Lfc.item())
        all_bar.set_description(
            f'Train epoch {epoch} LR:{optimizer.param_groups[0]["lr"]} Labeled loss: {xlosses.avg:.4f} Unlabeled loss: {ulosses.avg:.4f}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logger.log({'ce loss': xlosses.avg, 'fc loss': ulosses.avg})


def test(testloader, encoder, classifier, epoch):
    encoder.eval()
    classifier.eval()
    # accuracy = AverageMeter('accuracy')
    accs1 = AverageMeter('accs1')
    accs5 = AverageMeter('accs5')
    data_bar = tqdm(testloader)
    with torch.no_grad():
        for i, (data, label, _) in enumerate(data_bar):
            data, label = data.cuda(), label.cuda()
            feat = encoder(data)
            res = classifier(feat)
            acc1, acc5 = accuracy(res, label, [1, 5])
            accs1.update(acc1.item(), data.size(0))
            accs5.update(acc5.item(), data.size(0))
            data_bar.set_description(f'Test epoch {epoch}: Accuracy1@{accs1.avg:.4f} Accuracy5@{accs5.avg:.4f}')
    return accs1.avg, accs5.avg


def evaluate(dataloader, encoder, classifier, args, noisy_label):
    encoder.eval()
    classifier.eval()
    feature_bank = []
    prediction = []

    with torch.no_grad():
        # generate feature bank
        for (data, target, index) in tqdm(dataloader, desc='Feature extracting'):
            data = data.cuda()
            feature = encoder(data)
            feature_bank.append(feature)
            res = classifier(feature)
            prediction.append(res)
        feature_bank = F.normalize(torch.cat(feature_bank, dim=0), dim=1)

        ################################### sample relabelling ###################################
        prediction_cls = torch.softmax(torch.cat(prediction, dim=0), dim=1)
        his_score, his_label = prediction_cls.max(1)

        print(f'Prediction track: mean: {his_score.mean()} max: {his_score.max()} min: {his_score.min()}')
        conf_id = torch.where(his_score > args.theta_r)[0]
        modified_label = torch.clone(noisy_label).detach()
        modified_label[conf_id] = his_label[conf_id]

        ################################### sample selection ###################################
        prediction_knn = weighted_knn(feature_bank, feature_bank, modified_label, args.num_classes, args.k, 10)  # temperature in weighted KNN
        vote_y = torch.gather(prediction_knn, 1, modified_label.view(-1, 1)).squeeze()
        vote_max = prediction_knn.max(dim=1)[0]
        right_score = vote_y / vote_max
        clean_id = torch.where(right_score >= args.theta_s)[0]
        noisy_id = torch.where(right_score < args.theta_s)[0]

    return clean_id, noisy_id, modified_label


def main():
    args = parser.parse_args()
    seed_everything(args.seed)
    if args.run_path is None:
        args.run_path = f'Dataset(webvision_Model({args.theta_r}_{args.theta_s})'
    global logger
    logger = wandb.init(project='webvision', entity=args.entity, name=args.run_path)
    logger.config.update(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    # generate noisy dataset with our transformation
    if not os.path.isdir(f'webvision'):
        os.mkdir(f'webvision')
    if not os.path.isdir(f'webvision/{args.run_path}'):
        os.mkdir(f'webvision/{args.run_path}')
    args.num_classes = 50

    ################################ Model initialization ###########################################
    encoder = InceptionResNetV2(args.num_classes)
    dim = encoder.last_linear.in_features
    encoder.last_linear = torch.nn.Identity()
    classifier = torch.nn.Linear(dim, args.num_classes)
    proj_head = torch.nn.Sequential(torch.nn.Linear(dim, 256),
                                    torch.nn.BatchNorm1d(256),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(256, 256))
    pred_head = torch.nn.Sequential(torch.nn.Linear(256, 256),
                                    torch.nn.BatchNorm1d(256),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(256, 256))
    encoder.cuda()
    classifier.cuda()
    proj_head.cuda()
    pred_head.cuda()
    if args.parallel:
        encoder = torch.nn.DataParallel(encoder).cuda()
        classifier = torch.nn.DataParallel(classifier).cuda()
        proj_head = torch.nn.DataParallel(proj_head).cuda()
        pred_head = torch.nn.DataParallel(pred_head).cuda()

    ############################# Dataset initialization ##############################################
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    weak_transform = transforms.Compose([
        transforms.Resize(320),
        transforms.RandomResizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    none_transform = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        normalize])  # no augmentation
    strong_transform = transforms.Compose([transforms.Resize(320),
                                           transforms.RandomResizedCrop(299),
                                           transforms.RandomHorizontalFlip(),
                                           ImageNetPolicy(),
                                           transforms.ToTensor(),
                                           normalize])
    test_transform = none_transform

    eval_data = miniwebvision_dataset(root_dir=args.dataset_path, transform=weak_transform, dataset_mode='train',num_class=args.num_classes)
    test_data = miniwebvision_dataset(root_dir=args.dataset_path, transform=test_transform, dataset_mode='test', num_class=args.num_classes)
    train_data = miniwebvision_dataset(root_dir=args.dataset_path, transform=KCropsTransform(strong_transform, 2),
                                       dataset_mode='train', num_class=args.num_classes)

    all_data = miniwebvision_dataset(root_dir=args.dataset_path, transform=MixTransform(strong_transform, weak_transform, 1),
                                     dataset_mode='train', num_class=args.num_classes)

    noisy_label = torch.tensor(eval_data.train_labels).cuda()

    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=args.batch_size * 10, shuffle=False, pin_memory=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    all_loader = torch.utils.data.DataLoader(all_data, batch_size=args.batch_size, num_workers=4, shuffle=True)

    # *************************************************************************************************************#
    # to evaluate on imagenet_datset
    # test_data2 = imagenet_dataset(transform=test_transform)
    # test_loader2 = torch.utils.data.DataLoader(test_data2, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    # *************************************************************************************************************#

    #################################### Training initialization #######################################
    optimizer = SGD([{'params': encoder.parameters()}, {'params': classifier.parameters()}, {'params': proj_head.parameters()},
                     {'params': pred_head.parameters()}], lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)

    acc_logs = open(f'webvision/{args.run_path}/acc.txt', 'w')
    stat_logs = open(f'webvision/{args.run_path}/stat.txt', 'w')
    save_config(args, f'{args.run_path}')
    # best_acc1img = 0
    # best_acc5img = 0
    best_acc1web = 0
    best_acc5web = 0
    print('Train args: \n', args)

    ################################ Training loop ###########################################
    for i in range(args.epochs):
        clean_id, noisy_id, modified_label = evaluate(eval_loader, encoder, classifier, args, noisy_label)

        print(f'Epoch [{i}/{args.epochs}]: clean samples_1: {len(clean_id)}, noisy samples_1: {len(noisy_id)}')

        labeled_data = Subset(train_data, clean_id.cpu())
        sampler = ClassBalancedSampler(labels=modified_label[clean_id], num_classes=args.num_classes)
        labeled_loader = torch.utils.data.DataLoader(labeled_data, batch_size=args.batch_size, sampler=sampler, num_workers=4)

        xloss, uloss = train(labeled_loader, modified_label, all_loader, encoder, classifier, proj_head, pred_head, optimizer, i, args)

        stat_logs.write(f'Epoch [{i}/{args.epochs}]: clean samples_1: {len(clean_id)}, noisy samples_1: {len(noisy_id)} \n')
        stat_logs.flush()

        cur_acc1web, cur_acc5web = test(test_loader, encoder, classifier, i)
        scheduler.step()

        # *************************************************************************************************************#
        # cur_acc1img, cur_acc5img = test(test_loader2, encoder, classifier, i)
        # logger.log({'xloss': xloss, 'uloss': uloss, 'imgnet_acc1': cur_acc1img, 'imgnet_acc5': cur_acc5img, 'web_acc1': cur_acc1web, 'web_acc5': cur_acc5web})
        # if cur_acc5img > best_acc5img:
        #     best_acc5img = cur_acc5img
        # if cur_acc1img > best_acc1img:
        #     best_acc1img = cur_acc1img
        # *************************************************************************************************************#

        logger.log({'xloss': xloss, 'uloss': uloss, 'web_acc1': cur_acc1web, 'web_acc5': cur_acc5web})


        if cur_acc5web > best_acc5web:
            best_acc5web = cur_acc5web
        if cur_acc1web > best_acc1web:
            best_acc1web = cur_acc1web
            save_checkpoint({
                'cur_epoch': i,
                'classifier': classifier.state_dict(),
                'encoder': encoder.state_dict(),
                'proj_head': proj_head.state_dict(),
                'pred_head': pred_head.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=f'webvision/{args.run_path}/best_acc.pth.tar')
        acc_logs.write(
            f'Epoch [{i}/{args.epochs}]: web Best accuracy@1#:{best_acc1web}! Current accuracy@1#:{cur_acc1web}! Best accuracy@5#:{best_acc5web}! Current accuracy@5#:{cur_acc5web}\n')
        # acc_logs.write(
        #     f'Epoch [{i}/{args.epochs}]: img Best accuracy@1#:{best_acc1img}! Current accuracy@1#:{cur_acc1img}! Best accuracy@5#:{best_acc5img}! Current accuracy@5#:{cur_acc5img}\n')
        acc_logs.flush()
    save_checkpoint({
        'cur_epoch': i,
        'classifier': classifier.state_dict(),
        'encoder': encoder.state_dict(),
        'proj_head': proj_head.state_dict(),
        'pred_head': pred_head.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, filename=f'webvision/{args.run_path}/last.pth.tar')


if __name__ == '__main__':
    main()
