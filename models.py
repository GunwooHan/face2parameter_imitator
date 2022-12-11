import os

import cv2
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler


class UpsampleBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation="relu"):
        super(UpsampleBN, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)

    def forward(self, tensor):
        x = self.upsample(tensor)
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class TransConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation="relu"):
        super(TransConvBN, self).__init__()
        self.trans_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)

    def forward(self, tensor):
        x = self.trans_conv(tensor)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ConvBlockBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation="relu"):
        super(ConvBlockBN, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)

    def forward(self, tensor):
        x = self.conv(tensor)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Generator(nn.Module):
    def __init__(self, parameter_size=17):
        super(Generator, self).__init__()
        self.fc = nn.Linear(parameter_size, 512 * 4 * 4)
        self.dec1 = UpsampleBN(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.dec2 = UpsampleBN(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.dec3 = UpsampleBN(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.dec4 = UpsampleBN(
            in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.dec5 = UpsampleBN(
            in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.dec6 = UpsampleBN(
            in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dec7 = UpsampleBN(
            in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=32, out_channels=3,
                      kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, tensor):
        x = self.fc(tensor)
        x = torch.reshape(x, (-1, 512, 4, 4))
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        x = self.dec6(x)
        x = self.dec7(x)

        x = self.conv1(x)
        return x

class Blur(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, tensor):
        return


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False) -> None:
        super().__init__()
        self.conv1 = 

    def forward(self, tensor):
        return 




class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, tensor):
        return 

class Imitator(pl.LightningModule):
    def __init__(self, args):
        super(Imitator, self).__init__()
        self.args = args
        self.recon_loss = nn.L1Loss()
        self.generator = Generator(parameter_size=72)

    def forward(self, parameter):
        return self.model(parameter)

    def configure_optimizers(self):
        opt_g = optim.SGD(self.generator.parameters(),
                          lr=self.args.learning_rate, momentum=0.9)
        scheduler_g = lr_scheduler.StepLR(
            opt_g, step_size=50, gamma=0.9, verbose=True)

        return {"optimizer": opt_g, "lr_scheduler": {"scheduler": scheduler_g, "interval": "epoch"}}

    def training_step(self, train_batch, batch_idx):
        parameter, face_image = train_batch

        generated_face_image = self.generator(parameter)
        recon_loss = self.recon_loss(face_image, generated_face_image)
        total_loss = recon_loss * self.args.recon_weight
        self.log('train/recon_loss', recon_loss, on_step=True, on_epoch=True)
        return {"loss": total_loss}

    def validation_step(self, val_batch, batch_idx):
        parameter, face_image = val_batch
        generated_face_image = self.generator(parameter)
        recon_loss = self.recon_loss(face_image, generated_face_image)

        if batch_idx == 0:
            sample_count = 4 if parameter.size(0) > 4 else parameter.size(0)
            parameter = parameter[:sample_count]
            face_image = face_image[:sample_count]

            result_image = self.generator(parameter)

            result_grid = torchvision.utils.make_grid(torch.cat(
                [face_image, result_image]), nrow=sample_count).permute(1, 2, 0).cpu().numpy()
            result_grid = cv2.cvtColor(result_grid, cv2.COLOR_BGR2RGB)
            self.logger.log_image(key='sample_images', images=[
                result_grid], caption=[self.current_epoch + 1])

            if self.current_epoch % 1 == 0:
                torch.save(self.generator, os.path.join(
                    'checkpoints', self.args.name, f'{self.current_epoch:03d}_G.pt'))

        total_loss = recon_loss * self.args.recon_weight

        self.log('val/recon_loss', recon_loss, on_step=True, on_epoch=True)
        return {"loss": total_loss}


if __name__ == '__main__':
    model = Generator(parameter_size=72)

    tensor = torch.randn(1, 72)
    output = model(tensor)
    print(output.shape)
