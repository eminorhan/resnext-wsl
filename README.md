# Robustness properties of Facebook's ResNeXt WSL models
The code here can be used to reproduce the results reported in the following paper:

Orhan AE (2019) [Robustness properties of Facebook's ResNeXt WSL models.](https://arxiv.org/abs/1907.07640) arXiv:1907.07640.

All simulation results reported in the paper are provided in the [`results`](https://github.com/eminorhan/resnext-wsl/tree/master/results) folder. 

## Requirements
* torch == 1.1.0
* torchvision == 0.3.0
* foolbox == 1.8.0
* ImageNet validation data in its standard directory structure.
* [ImageNet-C and ImageNet-P](https://github.com/hendrycks/robustness) data in their standard directory structure.
* [ImageNet-A](https://github.com/hendrycks/natural-adv-examples) data in its standard directory structure.

## Replication
In total, there are eight experiments reported in the paper. They can be reproduced as follows:

1. To evaluate the ImageNet validation accuracy of the models, run [`evaluate_validation.py`](https://github.com/eminorhan/resnext-wsl/blob/master/evaluate_validation.py), e.g.:
```
python3 evaluate_validation.py /IMAGENET/DIR/ --model-name 'resnext101_32x16d_wsl'
```
Here and below, `model-name` should be one of `'resnext101_32x8d'`, `'resnext101_32x8d_wsl'`, `'resnext101_32x16d_wsl'`, `'resnext101_32x32d_wsl'`, `'resnext101_32x48d_wsl'`. `/IMAGENET/DIR/` is the top-level ImageNet directory (it should contain a `val` directory containing the validation images).


2. To evaluate the models on ImageNet-A, run [`evaluate_imageneta.py`](https://github.com/eminorhan/resnext-wsl/blob/master/evaluate_imageneta.py), e.g.:
```
python3 evaluate_imageneta.py /IMAGENETA/DIR/ --model-name 'resnext101_32x16d_wsl'
```
where `/IMAGENETA/DIR/` is the top-level ImageNet-A directory.


3. To evaluate the models on ImageNet-C, run [`evaluate_imagenetc.py`](https://github.com/eminorhan/resnext-wsl/blob/master/evaluate_imagenetc.py), e.g.:
```
python3 evaluate_imagenetc.py /IMAGENETC/DIR/ --model-name 'resnext101_32x16d_wsl'
```
where `/IMAGENETC/DIR/` is the top-level ImageNet-C directory.


4. To evaluate the models on ImageNet-P, run [`evaluate_imagenetp.py`](https://github.com/eminorhan/resnext-wsl/blob/master/evaluate_imagenetp.py), e.g.:
```
python3 evaluate_imagenetc.py /IMAGENETP/DIR/ --model-name 'resnext101_32x16d_wsl' --distortion-name 'gaussian_noise'
```
where `/IMAGENETP/DIR/` is the top-level ImageNet-P directory, and `distortion-name` should be one of `'gaussian_noise'`, `'shot_noise'`, `'motion_blur'`, `'zoom_blur'`, `'brightness'`, `'translate'`, `'rotate'`, `'tilt'`, `'scale'`, `'snow'`.


5. To run black-box adversarial attacks on the models, run [`evaluate_blackbox.py`](https://github.com/eminorhan/resnext-wsl/blob/master/evaluate_blackbox.py), e.g.:
```
python3 evaluate_blackbox.py /IMAGENET/DIR/ --model-name 'resnext101_32x16d_wsl' --epsilon 0.06
```
where `epsilon` is the normalized perturbation size.


6. To run white-box adversarial attacks on the models, run [`evaluate_whitebox.py`](https://github.com/eminorhan/resnext-wsl/blob/master/evaluate_whitebox.py), e.g.:
```
python3 evaluate_whitebox.py /IMAGENET/DIR/ --model-name 'resnext101_32x16d_wsl' --epsilon 0.06 --pgd-steps 10
```
where `epsilon` is the normalized perturbation size and `pgd-steps` is the number of PGD steps.


7. To evaluate the shape biases of the models, run [`evaluate_shapebias.py`](https://github.com/eminorhan/resnext-wsl/blob/master/evaluate_shapebias.py), e.g.:
```
python3 evaluate_shapebias.py /CUECONFLICT/DIR/ --model-name 'resnext101_32x16d_wsl'
```
where `/CUECONFLICT/DIR/` is the directory containing the shape-texture cue-conflict images. We provide these images in the [`cueconflict_images`](https://github.com/eminorhan/resnext-wsl/tree/master/cueconflict_images) folder. They are copied from Robert Geirhos's [texture-vs-shape](https://github.com/rgeirhos/texture-vs-shape) repository (see [here](https://github.com/rgeirhos/texture-vs-shape/tree/master/stimuli/style-transfer-preprocessed-512)), but with the non-conflicting images (images with the same shape and texture category) removed.


8. To visualize the learned features of the models, run [`visualize_features.py`](https://github.com/eminorhan/resnext-wsl/blob/master/visualize_features.py), e.g.:
```
python3 visualize_features.py /IMAGENET/DIR/ --model-name 'resnext101_32x16d_wsl'
```

## Acknowledgments
The code here utilizes code and stimuli from the [texture-vs-shape](https://github.com/rgeirhos/texture-vs-shape) repository by Robert Geirhos, the [robustness](https://github.com/hendrycks/robustness) and [natural adversarial examples](https://github.com/hendrycks/natural-adv-examples) repositories by Dan Hendrycks, and the [ImageNet example](https://github.com/pytorch/examples/tree/master/imagenet) from PyTorch. We are also grateful to the authors of [Mahajan et al. (2018)](https://arxiv.org/abs/1805.00932) for making their pre-trained ResNeXt WSL models publicly available.
