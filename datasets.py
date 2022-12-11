import os
import glob
import random

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


class P2FDataset(torch.utils.data.Dataset):
    def __init__(self, df, paramter_transform=None, face_image_transform=None):
        self.df = df.reset_index(drop=True)
        self.paramter_transform = paramter_transform
        self.face_image_transform = face_image_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, target_idx):
        temp = self.df.loc[target_idx].to_numpy()
        face_base_parameter = F.one_hot(torch.tensor(temp[2]), 25)
        hair_parameter = F.one_hot(torch.tensor(temp[3]), 30)

        # parameter = torch.tensor(temp[3:].astype(np.float32))
        face_attribute_parameter = torch.tensor(temp[4:].astype(np.float32))
        parameter = torch.cat([face_base_parameter, hair_parameter, face_attribute_parameter])
        face_image = Image.open(temp[1])

        if self.paramter_transform:
            parameter = self.paramter_transform(parameter)
        if self.face_image_transform:
            face_image = self.face_image_transform(face_image)

        return parameter, face_image


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('data\labels.csv')
    # temp = df.loc[0].to_numpy()
    # print()

    ds = P2FDataset(df)
    ds[0]
    print()