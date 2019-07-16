# Robustness properties of Facebook's ResNeXt WSL models
The code here can be used to reproduce the results reported in the following paper:

[Robustness properties of Facebook's ResNeXt WSL models](https://arxiv.org/abs/1907.????).

## Requirements
* torch >= 1.1.0
* torchvision >= 0.3.0
* foolbox >= 1.8.0
* ImageNet validation data in its standard directory structure.
* [ImageNet-C and ImageNet-P](https://github.com/hendrycks/robustness) data in their standard directory structure.

## Replication
There are in total seven experiments reported in the paper. They can be reproduced as follows:

1. To evaluate the ImageNet validation accuracy of the models, run [`evaluate_validation.py`](https://github.com/eminorhan/resnext-wsl/blob/master/evaluate_validation.py), e.g.:
```
python3 evaluate_validation.py /IMAGENET/DIR/ --model-name 'resnext101_32x16d_wsl'
```

2. To evaluate the models on ImageNet-C, run [`evaluate_imagenetc.py`](https://github.com/eminorhan/resnext-wsl/blob/master/evaluate_imagenetc.py), e.g.:
```
python3 evaluate_imagenetc.py /IMAGENETC/DIR/ --model-name 'resnext101_32x16d_wsl'
```

3. To evaluate the models on ImageNet-P, run [`evaluate_imagenetp.py`](https://github.com/eminorhan/resnext-wsl/blob/master/evaluate_imagenetp.py), e.g.:
```
python3 evaluate_imagenetc.py /IMAGENETP/DIR/ --model-name 'resnext101_32x16d_wsl' --distortion-name 'gaussian_noise'
```

4. To run black-box adversarial attacks on the models, run [`evaluate_blackbox.py`](https://github.com/eminorhan/resnext-wsl/blob/master/evaluate_blackbox.py), e.g.:
```
python3 evaluate_blackbox.py /IMAGENET/DIR/ --model-name 'resnext101_32x16d_wsl' --epsilon 0.06
```

5. To run white-box adversarial attacks on the models, run [`evaluate_whitebox.py`](https://github.com/eminorhan/resnext-wsl/blob/master/evaluate_whitebox.py), e.g.:
```
python3 evaluate_whitebox.py /IMAGENET/DIR/ --model-name 'resnext101_32x16d_wsl' --epsilon 0.06 --pgd-steps 10
```

6. To evaluate the shape biases of the models, run [`evaluate_shapebias.py`](https://github.com/eminorhan/resnext-wsl/blob/master/evaluate_shapebias.py), e.g.:
```
python3 evaluate_shapebias.py /SHAPE/BIAS/DIR/ --model-name 'resnext101_32x16d_wsl'
```

7. To visualize the learned features of the models, run [`visualize_features.py`](https://github.com/eminorhan/resnext-wsl/blob/master/visualize_features.py), e.g.:
```
python3 visualize_features.py /IMAGENET/DIR/ --model-name 'resnext101_32x16d_wsl'
```

## Acknowledgments
The code here utilizes code and stimuli from the [texture-vs-shape](https://github.com/rgeirhos/texture-vs-shape) repository by 
Robert Geirhos, the [robustness](https://github.com/hendrycks/robustness) repository by Dan Hendrycks, and the 
[ImageNet example](https://github.com/pytorch/examples/tree/master/imagenet) from PyTorch. We are also grateful to the authors of
[Mahajan et al. (2018)](https://arxiv.org/abs/1805.00932) for making their pre-trained ResNeXt WSL models publicly available.


