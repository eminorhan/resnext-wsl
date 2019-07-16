"""White-box attacks against models"""
import os
import time
import argparse
import numpy as np
from collections import OrderedDict
import torch
import torch.utils.data
import torchvision
import torchvision.models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from foolbox.models import PyTorchModel
from foolbox.attacks import RandomStartProjectedGradientDescentAttack
from foolbox.criteria import TargetClass
from foolbox.distances import Linfinity


parser = argparse.ArgumentParser(description='Run white-box attacks against models')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--model-name', type=str, default='resnext101_32x16d_wsl',
                    choices=['resnext101_32x8d', 'resnext101_32x8d_wsl', 'resnext101_32x16d_wsl',
                             'resnext101_32x32d_wsl', 'resnext101_32x48d_wsl'], help='evaluated model')
parser.add_argument('--workers', default=4, type=int, help='no of data loading workers')
parser.add_argument('--batch-size', default=2, type=int, help='mini-batch size')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--print-freq', default=250, type=int, help='print frequency')
parser.add_argument('--epsilon', default=0.06, type=float, help='perturbation size')
parser.add_argument('--pgd-steps', default=10, type=int, help='number of PGD steps')


def load_model(model_name):
    "Loads one of the pretrained models."

    if model_name in ['resnext101_32x8d_wsl', 'resnext101_32x16d_wsl', 'resnext101_32x32d_wsl',
                      'resnext101_32x48d_wsl']:
        model = torch.hub.load('facebookresearch/WSL-Images', model_name)
    elif model_name == 'resnext101_32x8d':
        model = torchvision.models.resnext101_32x8d(pretrained=True)
    else:
        raise ValueError('Model not available.')

    model = torch.nn.DataParallel(model).cuda()
    print('Loaded model:', model_name)

    return model


def validate(val_loader, model, epsilon, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, top1], prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    preprocessing = (mean, std)
    fmodel = PyTorchModel(model, bounds=(0, 1), num_classes=1000, preprocessing=preprocessing)

    clean_labels = np.zeros(len(val_loader))
    target_labels = np.zeros(len(val_loader))
    clean_pred_labels = np.zeros(len(val_loader))
    adv_pred_labels = np.zeros(len(val_loader))

    end = time.time()

    # Batch processing is not supported in in foolbox 1.8, so we feed images one by one. Note that we are using a batch
    # size of 2, which means we consider every other image (due to computational costs)
    for i, (images, target) in enumerate(val_loader):

        image = images.cpu().numpy()[0]
        clean_label = target.cpu().numpy()[0]

        target_label = np.random.choice(np.setdiff1d(np.arange(1000), clean_label))
        attack = RandomStartProjectedGradientDescentAttack(model=fmodel, criterion=TargetClass(target_label),
                                                           distance=Linfinity)
        adversarial = attack(image, clean_label, binary_search=False, epsilon=epsilon, stepsize=2./255,
                             iterations=args.pgd_steps, random_start=True)

        if np.any(adversarial==None):
            # Non-adversarial
            adversarial = image
            target_label = clean_label

        adv_pred_labels[i] = np.argmax(fmodel.predictions(adversarial))
        clean_labels[i] = clean_label
        target_labels[i] = target_label
        clean_pred_labels[i] = np.argmax(fmodel.predictions(image))

        print('Iter, Clean, Clean_pred, Adv, Adv_pred: ', i, clean_label, clean_pred_labels[i], target_label,
              adv_pred_labels[i])

        # measure accuracy and update average
        acc1 = 100. * np.mean(clean_label==adv_pred_labels[i])
        top1.update(acc1, 1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    print('* Acc@1 {top1.avg:.3f} '.format(top1=top1))

    return top1.avg


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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == "__main__":

    args = parser.parse_args()

    model = load_model(args.model_name)

    valdir = os.path.join(args.data, 'val')

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # run white-box attacks on validation set
    print('Running white-box attacks with epsilon:', args.epsilon, 'Number of PGD steps:', args.pgd_steps)

    acc1 = validate(val_loader, model, args.epsilon, args)
    print('Epsilon:', args.epsilon, 'Adv. accuracy:', acc1)

    np.save('whitebox_' + str(args.pgd_steps) + '_' + str(args.epsilon) + '_' + args.model_name + '.npy', acc1)