import os
import glob
import argparse

import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from torchvision.utils import save_image

from models import Imitator
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

model = Imitator.load_from_checkpoint('checkpoints/epoch=999-step=187000.ckpt', args=args)
model.generator.eval()
model.generator.cuda()

def convert_param(base_parameter):
    face_base_parameter = F.one_hot(torch.tensor(base_parameter[2]), 25)
    hair_parameter = F.one_hot(torch.tensor(base_parameter[3]), 30)

    # parameter = torch.tensor(temp[3:].astype(np.float32))
    face_attribute_parameter = torch.tensor(base_parameter[4:].astype(np.float32))
    input_parameter = torch.cat([face_base_parameter, hair_parameter, face_attribute_parameter])
    return input_parameter.cuda()

#FACE_BASE,HAIR_BASE,EYE_EYESIZE_W,EYE_EYESIZE_H,EYE_EYEPOS_W,EYE_EYEPOS_H,EYE_EYESHAPE_ANGLE,EYEBROW_EYEBROWSHAPE_H,EYEBROW_EYEBROWSHAPE_ANGLE,CHEEKBONE_CHEEKBONESHAPE_THRUST,CHEEKBONE_CHEEKBONESHAPE_H,JAW_JAWSIZE_SIZE,JAW_JAWSIZE_H,JAW_JAWSHAPE_THRUST,JAW_JAWSHAPE_ANGLE,NOSE_NOSESHAPE_H,NOSE_NOSESHAPE_W,NOSE_NOSESHAPE_THRUST,MOUTH_MOUTHSHAPE_H
# 21,5,0.211055276,0.266331658,0.48241206,0.040201005,0.793969849,0.27638191,0.206030151,0.793969849,0.939698492,0.010050251,0.497487437,0.251256281,0.150753769,0.728643216,0.989949749,0.944723618,0.16080402

base_parameter = [0,"data/images/00000.jpg",21,5,0.211055276,0.266331658,0.48241206,0.040201005,0.793969849,0.27638191,0.206030151,0.793969849,0.939698492,0.010050251,0.497487437,0.251256281,0.150753769,0.728643216,0.989949749,0.944723618,0.16080402]


# 얼굴 베이스 체크
with torch.no_grad():
    for i in range(25):
        base_parameter = np.array([0,"data/images/00000.jpg",i,5,0.211055276,0.266331658,0.48241206,0.040201005,0.793969849,0.27638191,0.206030151,0.793969849,0.939698492,0.010050251,0.497487437,0.251256281,0.150753769,0.728643216,0.989949749,0.944723618,0.16080402], dtype=object)
        input_parameter = convert_param(base_parameter)
        tensor_image_render = model(input_parameter)
        tensor_image_render = (tensor_image_render + 1) * 127.5
        np_image_render = tensor_image_render[0].cpu().permute(1, 2, 0).numpy()
        np_image_render = cv2.cvtColor(np_image_render, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"result/face_type_{i:02d}.jpg", np_image_render)
        #save_image(tensor_image_render, f"result/face_type_{i:02d}.jpg")
        
    # 헤어스타일 체크
    for i in range(30):
        base_parameter = np.array([0,"data/images/00000.jpg",0,i,0.211055276,0.266331658,0.48241206,0.040201005,0.793969849,0.27638191,0.206030151,0.793969849,0.939698492,0.010050251,0.497487437,0.251256281,0.150753769,0.728643216,0.989949749,0.944723618,0.16080402], dtype=object)
        input_parameter = convert_param(base_parameter)
        tensor_image_render = model(input_parameter)
        tensor_image_render = (tensor_image_render + 1) * 127.5
        np_image_render = tensor_image_render[0].cpu().permute(1, 2, 0).numpy()
        np_image_render = cv2.cvtColor(np_image_render, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"result/hair_type_{i:02d}.jpg", np_image_render)

    for i in np.arange(0, 1, 0.1):
        base_parameter = np.array([0,"data/images/00000.jpg",0,0,i,0.266331658,0.48241206,0.040201005,0.793969849,0.27638191,0.206030151,0.793969849,0.939698492,0.010050251,0.497487437,0.251256281,0.150753769,0.728643216,0.989949749,0.944723618,0.16080402], dtype=object)
        input_parameter = convert_param(base_parameter)
        tensor_image_render = model(input_parameter)
        tensor_image_render = (tensor_image_render + 1) * 127.5
        np_image_render = tensor_image_render[0].cpu().permute(1, 2, 0).numpy()
        np_image_render = cv2.cvtColor(np_image_render, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"result/param_eye_w_size_{i:.2f}.jpg", np_image_render)

    for i in np.arange(0, 1, 0.1):
        base_parameter = np.array([0,"data/images/00000.jpg",0,0,0,i,0.48241206,0.040201005,0.793969849,0.27638191,0.206030151,0.793969849,0.939698492,0.010050251,0.497487437,0.251256281,0.150753769,0.728643216,0.989949749,0.944723618,0.16080402], dtype=object)
        input_parameter = convert_param(base_parameter)
        tensor_image_render = model(input_parameter)
        tensor_image_render = (tensor_image_render + 1) * 127.5
        np_image_render = tensor_image_render[0].cpu().permute(1, 2, 0).numpy()
        np_image_render = cv2.cvtColor(np_image_render, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"result/param_eye_h_size_{i:.2f}.jpg", np_image_render)

# 파라미터 체크
