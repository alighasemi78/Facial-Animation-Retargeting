# Facial Animation Retargeting

## Importing the Required Libraries
import argparse
import torch
from utilities import create_mesh, create_image
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_images_dir", type=str, default="pipeline_images")
    parser.add_argument("--images_dir", type=str, default="images")
    parser.add_argument(
        "--blendshape_weights_dir", type=str, default="blendshape_weights"
    )
    parser.add_argument("--models_dir", type=str, default="models")
    return parser.parse_args()


## Creating a Custom Dataset
class MyDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        return image


## Defining the Architectures
### ReenactNet
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()

        # Queries, keys, and values
        query = (
            self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        )  # B x (H*W) x C'
        key = self.key_conv(x).view(batch_size, -1, H * W)  # B x C' x (H*W)
        value = self.value_conv(x).view(batch_size, -1, H * W)  # B x C x (H*W)

        # Attention map (softmax applied to key-query dot product)
        attention = F.softmax(torch.bmm(query, key), dim=-1)  # B x (H*W) x (H*W)

        # Weighted value map
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x (H*W)
        out = out.view(batch_size, C, H, W)  # B x C x H x W

        # Scale output
        out = self.gamma * out + x
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            SelfAttention(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(512),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 512),
            nn.Linear(512, 8192),
            nn.Unflatten(1, (512, 4, 4)),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            SelfAttention(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # print("!!!")
        return self.decoder(x)


class ReenactNet(nn.Module):
    def __init__(self):
        super(ReenactNet, self).__init__()

        self.E = Encoder()
        self.D_s = Decoder()
        self.D_t = Decoder()

    def forward(self, I_s, I_t=None):
        if I_t is None:
            z = self.E(I_s)
            I_t_bar = self.D_t(z)

            return I_t_bar

        z_s = self.E(I_s)
        z_t = self.E(I_t)

        I_s_bar = self.D_s(z_s)
        I_t_bar = self.D_t(z_t)

        return I_s_bar, I_t_bar

    def loss(self, I_s, I_t, I_s_bar, I_t_bar):
        l1_loss_s = torch.abs(I_s - I_s_bar).mean()
        l1_loss_t = torch.abs(I_t - I_t_bar).mean()

        return l1_loss_s + l1_loss_t


### BPNet
class BPNet(nn.Module):
    def __init__(self, lambda_w, lambda_r):
        super(BPNet, self).__init__()

        self.lambda_w = lambda_w
        self.lambda_r = lambda_r
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 12),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)

    def loss(self, w_t, I_t, w_t_bar, I_t_hat):
        loss_w = torch.abs(w_t - w_t_bar).mean()
        loss_r = torch.abs(I_t - I_t_hat).mean()

        return self.lambda_w * loss_w + self.lambda_r * loss_r


## Performing the Method
def run_pipeline(models_dir, saved_images_dir):
    reenactment_path = os.path.join(models_dir, "reenactment_model.pt")
    reenactNet = torch.load(reenactment_path)
    bpnet_path = os.path.join(models_dir, "bpnet_model.pt")
    bpNet = torch.load(bpnet_path)

    reenactNet.eval()
    bpNet.eval()

    with torch.no_grad():
        for I_s in dataloader:
            I_s = I_s.to(device)

            I_t_bar = reenactNet(I_s)
            w_t_bar = bpNet(I_t_bar)

            new_verts = (b0 + B @ w_t_bar.T).T.view(w_t_bar.shape[0], -1, 3)

            I_t_hat = []
            for verts in new_verts:
                mesh = create_mesh(verts, base_faces, device)

                image = create_image(device, mesh)
                I_t_hat.append(image)
            I_t_hat = torch.stack(I_t_hat)

            I_s = I_s.permute(0, 2, 3, 1)
            I_t_bar = I_t_bar.permute(0, 2, 3, 1)
            I_t_hat = I_t_hat.permute(0, 2, 3, 1)

            for i in range(I_s.shape[0]):
                images = [
                    I_s[i].cpu().numpy(),
                    I_t_bar[i].cpu().numpy(),
                    I_t_hat[i].cpu().numpy(),
                ]

                fig, axes = plt.subplots(1, 3, figsize=(5, 15))
                for _, (image, ax) in enumerate(zip(images, axes.flatten())):
                    ax.imshow(image)
                    ax.axis("off")

                plt.tight_layout()

                save_path = os.path.join(saved_images_dir, f"test_result_{i}.png")
                plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
                plt.close(fig)
            break


if __name__ == "__main__":
    ## Initial Settings
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        print("WARNING: CPU only, this will be slow!")

    args = get_args()

    saved_images_dir = args.saved_images_dir
    if not os.path.exists(saved_images_dir):
        os.makedirs(saved_images_dir)

    ## Loading the Images
    images_dir = args.images_dir
    source_images_dir = os.path.join(images_dir, "source")
    images_names = os.listdir(source_images_dir)

    images = []
    for image in tqdm(images_names):
        image_path = os.path.join(source_images_dir, image)
        images.append(torch.load(image_path))

    images = torch.stack(images)

    print("Images Shape:")
    print(images.shape)

    ## Loading the Target Base Faces
    blendshape_weights_dir = args.blendshape_weights_dir
    blendshape_weights_path = os.path.join(blendshape_weights_dir, "target.pt")
    blendshape_weights = torch.load(blendshape_weights_path)

    b0 = blendshape_weights["b0"].to(device)
    B = blendshape_weights["B"].to(device)
    base_faces = blendshape_weights["base_faces"].to(device)

    dataset = MyDataset(images)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    models_dir = args.models_dir
    run_pipeline(models_dir, saved_images_dir)
