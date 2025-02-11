# Training Facial Reenactment

## Importing the Required Libraries
import argparse
import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn as nn
import torch.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_images_dir", type=str, default="reenactment_images")
    parser.add_argument("--saved_models_dir", type=str, default="models")
    parser.add_argument("--images_dir", type=str, default="images")
    return parser.parse_args()


## Creating a Custom Dataset
class MyDataset(Dataset):
    def __init__(self, source_images, target_images):
        self.source_images = source_images
        self.target_images = target_images

    def __len__(self):
        return len(self.source_images)

    def __getitem__(self, idx):
        source_image = self.source_images[idx]
        target_image = self.target_images[idx]

        return source_image, target_image


## Defining the Architecture of the Autoencoder
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


## Training the Model
def train(train_dataloader, saved_models_dir, device):
    model = ReenactNet()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 16

    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0

        for batch in tqdm(train_dataloader, leave=False):
            I_s = batch[0].to(device)
            I_t = batch[1].to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            I_s_bar, I_t_bar = model(I_s, I_t)

            # Compute the reconstruction loss
            loss = model.loss(I_s, I_t, I_s_bar, I_t_bar)
            # print(loss)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate batch loss
            epoch_loss += loss.item()

        # Average loss for the epoch
        epoch_loss /= len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    saved_model_path = os.path.join(saved_models_dir, "reenactment_model.pt")
    torch.save(model, saved_model_path)

    return saved_model_path


## Testing the Trained Model
def test(test_dataloader, saved_model_path, saved_images_dir, device):
    model = torch.load(saved_model_path)
    model.eval()

    with torch.no_grad():
        for batch in test_dataloader:
            I_s = batch[0].to(device)

            # Forward pass
            I_t_bar = model(I_s)

            I_s = I_s.permute(0, 2, 3, 1)
            I_t_bar = I_t_bar.permute(0, 2, 3, 1)

            for i in range(I_s.shape[0]):
                images = [I_s[i].cpu().numpy(), I_t_bar[i].cpu().numpy()]

                fig, axes = plt.subplots(1, 2, figsize=(5, 10))
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

    saved_models_dir = args.saved_models_dir
    if not os.path.exists(saved_models_dir):
        os.makedirs(saved_models_dir)

    ## Loading the Images
    images_dir = args.images_dir
    source_images_dir = os.path.join(images_dir, "source")
    target_images_dir = os.path.join(images_dir, "target")
    source_image_names = os.listdir(source_images_dir)
    target_image_names = os.listdir(target_images_dir)

    source_images = []
    for image in tqdm(source_image_names):
        image_path = os.path.join(source_images_dir, image)
        source_images.append(torch.load(image_path))

    target_images = []
    for image in tqdm(target_image_names):
        image_path = os.path.join(target_images_dir, image)
        target_images.append(torch.load(image_path))

    source_images = torch.stack(source_images)
    target_images = torch.stack(target_images)

    print("Shape of source images and target images:")
    print(source_images.shape)
    print(target_images.shape)

    dataset = MyDataset(source_images, target_images)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    for batch in train_dataloader:
        I_s = batch[0].permute(0, 2, 3, 1)
        I_t = batch[1].permute(0, 2, 3, 1)

        images = [I_s[0].cpu().numpy(), I_t[0].cpu().numpy()]

        fig, axes = plt.subplots(1, 2, figsize=(5, 10))
        for i, (image, ax) in enumerate(zip(images, axes.flatten())):
            ax.imshow(image)
            ax.axis("off")

        plt.tight_layout()
        save_path = os.path.join(saved_images_dir, f"batch_sample.png")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        break

    saved_model_path = train(train_dataloader, saved_models_dir, device)
    test(test_dataloader, saved_model_path, saved_images_dir, device)
