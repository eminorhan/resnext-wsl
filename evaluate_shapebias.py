"""Evaluate models on shape-texture cue conflict stimuli"""
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
from PIL import Image


parser = argparse.ArgumentParser(description='Evaluate models on shape-texture cue-conflict stimuli')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--model-name', type=str, default='resnext101_32x16d_wsl',
                    choices=['resnext101_32x8d', 'resnext101_32x8d_wsl', 'resnext101_32x16d_wsl',
                             'resnext101_32x32d_wsl', 'resnext101_32x48d_wsl'], help='evaluated model')
parser.add_argument('--workers', default=4, type=int, help='no of data loading workers')
parser.add_argument('--batch-size', default=64, type=int, help='mini-batch size')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')


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


def validate(model, preprocessing, args):
    from utils import rel_inds

    preds = []

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():

        file_list = os.listdir(args.data)
        file_list.sort()

        for file_name in file_list:

            img = Image.open(os.path.join(args.data, file_name))
            proc_img = preprocessing(img)
            proc_img = proc_img.unsqueeze(0)

            if args.gpu is not None:
                proc_img = proc_img.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(proc_img)
            output = output.cpu().numpy().squeeze()

            new_output = -np.inf * np.ones(1000)
            new_output[rel_inds] = output[rel_inds]

            pred = np.argmax(new_output)

            preds.append(pred)

        preds = np.array(preds)

    return preds


def accuracies(labels, preds):
    from utils import conversion_table

    shape_matches = np.zeros(len(preds))
    texture_matches = np.zeros(len(preds))

    for i in range(len(preds)):
        pred = preds[i]
        texture = labels['textures'][i]
        shape = labels['shapes'][i]

        if pred in conversion_table[texture]:
            texture_matches[i] = 1

        if pred in conversion_table[shape]:
            shape_matches[i] = 1

    correct = shape_matches + texture_matches
    frac_correct = np.mean(correct)
    frac_shape = np.sum(shape_matches) / np.sum(correct)
    frac_texture = np.sum(texture_matches) / np.sum(correct)

    return frac_correct, frac_shape, frac_texture


if __name__ == "__main__":

    args = parser.parse_args()

    model = load_model(args.model_name)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([transforms.ToTensor(), normalize])

    # evaluate on validation set
    preds = validate(model, preprocessing, args)
    labels = np.load('cueconflict_labels.npz')
    frac_correct, frac_shape, frac_texture = accuracies(labels, preds)

    print('Correct:', frac_correct, 'Shape:', frac_shape, 'Texture:', frac_texture)

    np.savez('shapebias_' + args.model_name + '.npz', frac_correct=frac_correct, frac_shape=frac_shape,
             frac_texture=frac_texture)