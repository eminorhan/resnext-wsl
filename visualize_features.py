"""Feature visualizations"""
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
from foolbox.attacks import ProjectedGradientDescentAttack
from foolbox.criteria import TargetClass, TargetClassProbability
from foolbox.distances import Linfinity
from foolbox.distances import Distance
import matplotlib.pylab as plt


parser = argparse.ArgumentParser(description='Feature visualizations')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--model-name', type=str, default='resnext101_32x16d_wsl',
                    choices=['resnext101_32x8d', 'resnext101_32x8d_wsl', 'resnext101_32x16d_wsl',
                             'resnext101_32x32d_wsl', 'resnext101_32x48d_wsl'], help='evaluated model')
parser.add_argument('--workers', default=4, type=int, help='no of data loading workers')
parser.add_argument('--batch-size', default=1, type=int, help='mini-batch size')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--epsilon', default=4., type=float, help='perturbation size')


class Squeeze(torch.nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, input):
        output = torch.squeeze(torch.squeeze(input, dim=-1), dim=-1)
        return output


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
    x = list(model.module.children())[:-1]  # remove the last (softmax) layer
    x.append(Squeeze())
    new_model = torch.nn.Sequential(*x)

    print('Loaded model:', model_name)

    return new_model


def validate(val_loader, model, epsilon, args):
    # switch to evaluate mode
    model.eval()

    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    preprocessing = (mean, std)
    fmodel = PyTorchModel(model, bounds=(0, 1), num_classes=2048, preprocessing=preprocessing)

    np.random.seed(1)  # fix seed for reproducibility across models, images

    clean_label = 0  # dummy label
    target_labels = np.random.choice(np.setdiff1d(np.arange(2048), clean_label), 6)
    print(target_labels)

    imgs = []
    advs = []

    # Batch processing is experimental in foolbox, so we feed images one by one.
    for i, (images, target) in enumerate(val_loader):

        if i == 2:
            image = np.float32(np.random.rand(3, 224, 224))
            imgs.append(image)
            print(image)
        else:
            image = images.cpu().numpy()[0]
            imgs.append(image)
            print(image)

        for j in range(len(target_labels)):
            target_label = target_labels[j]
            attack = ProjectedGradientDescentAttack(model=fmodel,
                                                    criterion=TargetClassProbability(target_label, 1.-1e-6),
                                                    distance=Linfinity)
            adversarial = attack(image, clean_label, binary_search=False, epsilon=epsilon, stepsize=1./255,
                                 iterations=500, random_start=False, return_early=False)

            adv_pred_label = np.argmax(fmodel.predictions(adversarial))
            clean_pred_label = np.argmax(fmodel.predictions(image))
            print('Iter, Clean_pred, Adv, Adv_pred: ', i, clean_pred_label, target_label, adv_pred_label)

            advs.append(adversarial)

        if i == 2:
            return imgs, advs


if __name__ == "__main__":

    args = parser.parse_args()

    model = load_model(args.model_name)

    valdir = os.path.join(args.data, 'val')

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print('Visualizing features')

    imgs, advs = validate(val_loader, model, args.epsilon, args)

    np.savez('featvis_' + args.model_name + '.npz', imgs=imgs, advs=advs)