"""Evaluate models on ImageNet-P"""
import os
import time
import argparse
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import torch
import torch.utils.data
import torchvision
import torchvision.models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cv2
from torchvision.datasets.folder import DatasetFolder
from PIL import Image
from scipy.stats import rankdata


parser = argparse.ArgumentParser(description='Evaluate model on ImageNet-P')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--model-name', type=str, default='resnext101_32x16d_wsl',
                    choices=['resnext101_32x8d', 'resnext101_32x8d_wsl', 'resnext101_32x16d_wsl',
                             'resnext101_32x32d_wsl', 'resnext101_32x48d_wsl'], help='evaluated model')
parser.add_argument('--distortion-name', type=str, default='resnext101_32x16d_wsl',
                    choices=['gaussian_noise', 'shot_noise', 'motion_blur', 'zoom_blur', 'brightness', 'translate',
                             'rotate', 'tilt', 'scale', 'snow'], help='distortion name')
parser.add_argument('--workers', default=5, type=int, help='no of data loading workers')
parser.add_argument('--batch-size', default=1, type=int, help='mini-batch size')
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


# /////////////// Video Loader ///////////////
class VideoFolder(DatasetFolder):

    def __init__(self, root, transform=None, target_transform=None, loader=None):
        super(VideoFolder, self).__init__(
            root, loader, '.mp4', transform=transform, target_transform=target_transform)

        self.vids = self.samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]

        # cap = VideoCapture(path)
        cap = cv2.VideoCapture(path)

        frames = []

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret: break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self.transform(Image.fromarray(frame)).unsqueeze(0))

        cap.release()

        return torch.cat(frames, 0), target


# /////////////// Stability Measurements ///////////////
def dist(sigma, mode='top5'):
    identity = np.asarray(range(1, 1001))
    cum_sum_top5 = np.cumsum(np.asarray([0] + [1] * 5 + [0] * (999 - 5)))
    recip = 1. / identity

    if mode == 'top5':
        return np.sum(np.abs(cum_sum_top5[:5] - cum_sum_top5[sigma-1][:5]))
    elif mode == 'zipf':
        return np.sum(np.abs(recip - recip[sigma-1])*recip)


def ranking_dist(ranks, perturbation, mode='top5'):
    noise_perturbation = True if 'noise' in perturbation else False
    result = 0
    step_size = 1

    for vid_ranks in ranks:
        result_for_vid = []

        for i in range(step_size):
            perm1 = vid_ranks[i]
            perm1_inv = np.argsort(perm1)

            for rank in vid_ranks[i::step_size][1:]:
                perm2 = rank
                result_for_vid.append(dist(perm2[perm1_inv], mode))
                if not noise_perturbation:
                    perm1 = perm2
                    perm1_inv = np.argsort(perm1)

        result += np.mean(result_for_vid) / len(ranks)

    return result


def flip_prob(predictions, perturbation):
    noise_perturbation = True if 'noise' in perturbation else False
    result = 0
    step_size = 1

    for vid_preds in predictions:
        result_for_vid = []

        for i in range(step_size):
            prev_pred = vid_preds[i]

            for pred in vid_preds[i::step_size][1:]:
                result_for_vid.append(int(prev_pred != pred))
                if not noise_perturbation: prev_pred = pred

        result += np.mean(result_for_vid) / len(predictions)

    return result


if __name__ == "__main__":

    args = parser.parse_args()

    model = load_model(args.model_name)
    model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    valdir = args.data + args.distortion_name
    val_loader = torch.utils.data.DataLoader(
        VideoFolder(root=valdir, transform=transforms.Compose([transforms.ToTensor(), normalize])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    predictions, ranks = [], []
    with torch.no_grad():

        for data, target in val_loader:
            num_vids = data.size(0)
            data = data.view(-1, 3, 224, 224).cuda()

            output = model(data)

            for vid in output.view(num_vids, -1, 1000):
                predictions.append(vid.argmax(1).to('cpu').numpy())
                ranks.append([np.uint16(rankdata(-frame, method='ordinal')) for frame in vid.to('cpu').numpy()])

    ranks = np.asarray(ranks)

    fr = flip_prob(predictions, args.distortion_name)
    t5d = ranking_dist(ranks, args.distortion_name, mode='top5')

    print('Computing Metrics\n')
    print('Flipping Prob\t{:.5f}'.format(fr))
    print('Top5 Distance\t{:.5f}'.format(t5d))

    np.savez('imagenetp_' + args.model_name + '_' + args.distortion_name + '.npz', fr=fr, t5d=t5d)