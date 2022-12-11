import os
import glob
import argparse

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split


from models import Initiator
from datasets import P2FDataset

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--train_datadir', type=str, default='data/labels.csv')
parser.add_argument('--checkpoints_dir', type=str, default='checkpoints')

parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--precision', type=int, default=32)
parser.add_argument('--name', type=str, default='v1')
parser.add_argument('--num_workers', type=int, default=6)

parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--adam_beta', type=float, default=0)

parser.add_argument('--recon_weight', type=float, default=1)

args = parser.parse_args()

model = Initiator.load_from_checkpoint(
    'checkpoints\\epoch=459-step=57500.ckpt', args=args)
print()
