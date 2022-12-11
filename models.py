import os

import kornia
import cv2
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
import torch.nn.functional as F

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


class GaussianBlur(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size=kernel_size

    def forward(self, tensor):
        x = kornia.filters.gaussian_blur2d(tensor, kernel_size=self.kernel_size, sigma=(1, 1))
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)),
            GaussianBlur(kernel_size=(3, 3)),
            nn.LeakyReLU(0.2)
        )

        self.skip = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)),
        ) 

    def forward(self, tensor):
        x = self.conv1(tensor)
        x = self.conv2(x)

        skip = self.skip(tensor)
        x = (x + skip) / (2 ** 0.5)
        return x



class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            ResBlock(64, 128),
            ResBlock(128, 256),
            ResBlock(256, 512),
            ResBlock(512, 512),
            ResBlock(512, 512),
            ResBlock(512, 512),
            ResBlock(512, 512),
            nn.Flatten(start_dim=1),
            nn.Linear(512 * 4 * 4, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
        )


    def forward(self, tensor):
        x = self.model(tensor)
        return x

class Imitator(pl.LightningModule):
    def __init__(self, args):
        super(Imitator, self).__init__()
        self.args = args
        self.recon_loss = nn.L1Loss()
        self.generator = Generator(parameter_size=72)
        self.discriminator = Discriminator()

    def forward(self, parameter):
        return self.model(parameter)

    def configure_optimizers(self):
        opt_g = optim.SGD(self.generator.parameters(), lr=self.args.learning_rate, momentum=0.9)
        opt_d = optim.SGD(self.discriminator.parameters(), lr=self.args.learning_rate, momentum=0.9)

        scheduler_g = lr_scheduler.StepLR(opt_g, step_size=50, gamma=0.9, verbose=True)
        scheduler_d = lr_scheduler.StepLR(opt_d, step_size=50, gamma=0.9, verbose=True)

        return [{"optimizer": opt_g, "lr_scheduler": {"scheduler": scheduler_g, "interval": "epoch"}},
        {"optimizer": opt_d, "lr_scheduler": {"scheduler": scheduler_d, "interval": "epoch"}},]

    def training_step(self, train_batch, batch_idx, optimizer_idx):
        parameter, face_image = train_batch

        if optimizer_idx == 0:
            generated_face_image = self.generator(parameter)
            recon_loss = self.recon_loss(face_image, generated_face_image)

            disc_out_fake = self.discriminator(generated_face_image)
            gen_loss = - disc_out_fake.mean()

            total_loss = recon_loss * self.args.recon_weight + gen_loss * self.args.gan_weight
            self.log('train/recon_loss', recon_loss, on_step=True, on_epoch=True)
            self.log('train/gen_loss', gen_loss, on_step=True, on_epoch=True)
            return {"loss": total_loss}

        if optimizer_idx==1:
            generated_face_image = self.generator(parameter)
            recon_loss = self.recon_loss(face_image, generated_face_image)

            disc_out_real = self.discriminator(face_image)
            disc_out_fake = self.discriminator(generated_face_image)

            disc_loss_real = torch.relu(1.0 - disc_out_real).mean()
            disc_loss_fake = torch.relu(1.0 + disc_out_fake).mean()
            disc_loss = (disc_loss_real + disc_loss_fake)/2 * self.args.gan_weight

            self.log('train/disc_loss', disc_loss, on_step=True, on_epoch=True)

            if batch_idx % 500 == 0:
                sample_count = 4 if parameter.size(0) > 4 else parameter.size(0)
                generated_face_image = generated_face_image[:sample_count]
                face_image = face_image[:sample_count]
                result_grid = torchvision.utils.make_grid(torch.cat([face_image, generated_face_image]), nrow=sample_count).permute(1, 2, 0).cpu().numpy()
                self.logger.log_image(key='sample_images', images=[result_grid], caption=[self.current_epoch + 1])

            return {"loss": disc_loss}

    def validation_step(self, val_batch, batch_idx):
        parameter, face_image = val_batch
        generated_face_image = self.generator(parameter)
        recon_loss = self.recon_loss(face_image, generated_face_image)

        if batch_idx == 0:
            sample_count = 4 if parameter.size(0) > 4 else parameter.size(0)
            generated_face_image = generated_face_image[:sample_count]
            face_image = face_image[:sample_count]
            result_grid = torchvision.utils.make_grid(torch.cat([face_image, generated_face_image]), nrow=sample_count).permute(1, 2, 0).cpu().numpy()
            self.logger.log_image(key='sample_images', images=[result_grid], caption=[self.current_epoch + 1])

        total_loss = recon_loss * self.args.recon_weight

        self.log('val/recon_loss', recon_loss, on_step=True, on_epoch=True)
        return {"loss": total_loss}


if __name__ == '__main__':
    # model = Generator(parameter_size=72)

    # tensor = torch.randn(1, 72)
    # output = model(tensor)
    # print(output.shape)

    model = Discriminator()
    inputs = torch.randn(2, 3, 512, 512)
    outputs = model(inputs)
    print(outputs.shape)