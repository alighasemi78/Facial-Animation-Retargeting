# Training Blenshape Prediction

## Importing the Required Libraries
import argparse
import torch
from utilities import create_mesh, create_image
import os
from tqdm import tqdm
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_images_dir", type=str, default="bpnet_images")
    parser.add_argument("--saved_models_dir", type=str, default="models")
    parser.add_argument("--images_dir", type=str, default="images")
    parser.add_argument(
        "--blendshape_weights_dir", type=str, default="blendshape_weights"
    )
    return parser.parse_args()


## Creating a Custom Dataset
class MyDataset(Dataset):
    def __init__(self, images, weights):
        self.images = images
        self.w = weights

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        w = self.w[idx]

        return image, w


## Defining the Architecture of the BPNet
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


## Training the Model
def train(train_dataloader, saved_models_dir, device):
    model = BPNet(1, 1)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    num_epochs = 8

    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0

        for batch in tqdm(train_dataloader, leave=False):
            I_t = batch[0].to(device)
            w_t = batch[1].to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            w_t_bar = model(I_t)

            new_verts = (b0 + B @ w_t_bar.T).T.view(w_t_bar.shape[0], -1, 3)

            I_t_hat = []
            for verts in new_verts:
                mesh = create_mesh(verts, base_faces, device)

                image = create_image(device, mesh, True)
                I_t_hat.append(image)
            I_t_hat = torch.stack(I_t_hat)

            # Compute the reconstruction loss
            loss = model.loss(w_t, I_t, w_t_bar, I_t_hat)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate batch loss
            epoch_loss += loss.item()

        # Average loss for the epoch
        epoch_loss /= len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    saved_model_path = os.path.join(saved_models_dir, "bpnet_model.pt")
    torch.save(model, saved_model_path)

    return saved_model_path


## Testing the Trained Model
def test(test_dataloader, saved_model_path, saved_images_dir, device):
    model = torch.load(saved_model_path)
    model.eval()

    with torch.no_grad():
        for batch in test_dataloader:
            I_t = batch[0].to(device)

            # Forward pass
            w_t_bar = model(I_t)

            new_verts = (b0 + B @ w_t_bar.T).T.view(w_t_bar.shape[0], -1, 3)

            I_t_hat = []
            for verts in new_verts:
                mesh = create_mesh(verts, base_faces, device)

                image = create_image(device, mesh)
                I_t_hat.append(image)
            I_t_hat = torch.stack(I_t_hat)

            I_t = I_t.permute(0, 2, 3, 1)
            I_t_hat = I_t_hat.permute(0, 2, 3, 1)

            for i in range(I_t.shape[0]):
                images = [I_t[i].cpu().numpy(), I_t_hat[i].cpu().numpy()]

                fig, axes = plt.subplots(1, 2, figsize=(5, 10))
                for _, (image, ax) in enumerate(zip(images, axes.flatten())):
                    ax.imshow(image)
                    ax.axis("off")

                plt.tight_layout()

                save_path = os.path.join(saved_images_dir, f"test_result_{i}.png")
                plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
                plt.close(fig)

            break


def get_metrics(test_dataloader, saved_model_path, device):
    model = torch.load(saved_model_path)
    model.eval()

    total_mae_weights = 0
    total_mae_images = 0
    num_samples = 0

    with torch.no_grad():
        for batch in test_dataloader:
            I_t = batch[0].to(device)  # Ground truth image
            w_t = batch[1].to(device)  # Ground truth weights

            # Forward pass
            w_t_bar = model(I_t)

            # Compute weight difference
            mae_weights = F.l1_loss(
                w_t_bar, w_t
            ).item()  # Mean Absolute Error (L1 Loss)

            # Generate new vertices from predicted weights
            new_verts = (b0 + B @ w_t_bar.T).T.view(w_t_bar.shape[0], -1, 3)

            # Reconstruct images
            I_t_hat = []
            for verts in new_verts:
                mesh = create_mesh(verts, base_faces, device)
                image = create_image(device, mesh)
                I_t_hat.append(image)
            I_t_hat = torch.stack(I_t_hat)

            # Compute image difference
            mae_images = torch.abs(I_t - I_t_hat).mean().item()

            # Accumulate results
            total_mae_weights += mae_weights
            total_mae_images += mae_images
            num_samples += 1

    avg_mae_weights = total_mae_weights / num_samples
    avg_mae_images = total_mae_images / num_samples

    print(f"Test Set Evaluation Results:")
    print(f"Weight Prediction - Mean Absolute Error (MAE): {avg_mae_weights:.4f}")
    print(f"Image Reconstruction - Mean Absolute Error (MAE): {avg_mae_images:.4f}")


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
    target_images_dir = os.path.join(images_dir, "target")
    images_names = os.listdir(target_images_dir)
    images_names.sort(key=lambda x: int(x.split(".")[0]))

    images = []
    for image in tqdm(images_names):
        image_path = os.path.join(target_images_dir, image)
        images.append(torch.load(image_path))

    images = torch.stack(images)

    print("Images Shape:")
    print(images.shape)

    ## Loading the Blendshape Weights
    blendshape_weights_dir = args.blendshape_weights_dir
    blendshape_weights_path = os.path.join(blendshape_weights_dir, "target.pt")
    blendshape_weights = torch.load(blendshape_weights_path)

    b0 = blendshape_weights["b0"].to(device)
    B = blendshape_weights["B"].to(device)
    base_faces = blendshape_weights["base_faces"].to(device)
    w = blendshape_weights["w"].to(device)

    dataset = MyDataset(images, w)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=5, shuffle=False)

    for batch in train_dataloader:
        I_t = batch[0]
        w_t = batch[1]

        new_verts = (b0 + B @ w_t.T).T.view(w_t.shape[0], -1, 3)
        I_t_from_w = []
        for verts in new_verts:
            mesh = create_mesh(verts, base_faces, device)

            image = create_image(device, mesh)
            I_t_from_w.append(image)
        I_t_from_w = torch.stack(I_t_from_w)

        I_t = I_t.permute(0, 2, 3, 1)
        I_t_from_w = I_t_from_w.permute(0, 2, 3, 1)

        images = [I_t[0].cpu().numpy(), I_t_from_w[0].cpu().numpy()]

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
    get_metrics(test_dataloader, saved_model_path, device)
