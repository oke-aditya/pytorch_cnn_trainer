# PyTorch CNN Trainer

![Check Formatting](https://github.com/oke-aditya/pytorch_cnn_trainer/workflows/Check%20Formatting/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Train CNN for your task.

A simple engine to train your CNN. It aims to support most features that you would ever use to train CNN.
## Why This repo ?

It is very annoying to write training loop and training code for CNN training. Also to support all the training stuff it will take massive time.
Usually we don't need distributed training and it is very uncomfortable to use argparse and get the job done.

This simplifies the training. It provide you a powerful `engine.py` which can do lot of training functionalities. 
Also a `dataset.py` to load dataset in common scenarios.

Note: - 
Pytorch Trainer is not a distributed training script.

It will work good for single GPU machine for Google Colab / Kaggle.

But for distributed Training you can use the PyTorch Lightning Trainer. 

It will train on multiple GPUs just the way lightning supports.

# To Do: -


- [x] Support PyTorch image models (timm).
- [x] Quantization Aware training example.
- [x] Early stopping with patience.
- [x] Support torchvision models transfer learning.
- [x] Add Keras Like fit method.
- [ ] Add history like object as in Keras for plotting.
- [ ] Other metrics such as precision, recall, f1, etc.
- [x] Support torchvision quantized models transfer learning.
- [ ] Mixed precision training using PyTorch 1.6
- [ ] PyTorch Lightning Trainer with all these features.
- [x] Add docstring, create package, etc.
- [x] Minimal tests and examples.

Hope this repo helps guys to train models using transfer learning. 

If you like it do give it a * and tell people about it.
