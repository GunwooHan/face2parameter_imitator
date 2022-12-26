import argparse
import os

import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split

import wandb
from datasets import P2FDataset
from models import Imitator

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
parser.add_argument('--gan_weight', type=float, default=0.1)

args = parser.parse_args()

if __name__ == '__main__':
    pl.seed_everything(args.seed)
    wandb_logger = WandbLogger(project=f'Face2Parameter_imitator', name=f'{args.name}')
    wandb_logger.log_hyperparams(args)

    face_image_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
    )

    df = pd.read_csv(args.train_datadir)
    train_df, valid_df = train_test_split(df, test_size=0.2)

    train_ds = P2FDataset(train_df, paramter_transform=None, face_image_transform=face_image_transform)
    valid_ds = P2FDataset(valid_df, paramter_transform=None, face_image_transform=face_image_transform)

    train_dataloader = torch.utils.data.DataLoader(train_ds,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.num_workers,
                                                   shuffle=True,
                                                   drop_last=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_ds,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.num_workers,
                                                   #    shuffle=True,
                                                   drop_last=True)

    model = Imitator(args)

    os.makedirs(os.path.join(args.checkpoints_dir, args.name), exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoints_dir, args.name),
        verbose=True,
        every_n_epochs=1
    )

    trainer = pl.Trainer(accelerator="gpu",
                        devices=args.gpus,
                        precision=args.precision,
                        max_epochs=args.epochs,
                        strategy='ddp',
                        #  limit_train_batches=1,
                        #  log_every_n_steps=1,
                        logger=wandb_logger,
                        callbacks=[checkpoint_callback]
                        )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    wandb.finish()
