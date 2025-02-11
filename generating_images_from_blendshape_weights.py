# Generating Images from Blendshape Weights

## Importing the Required Libraries
import argparse
import torch
import os
import shutil
from tqdm import tqdm
from utilities import create_mesh, create_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_images_dir", type=str, default="images")
    parser.add_argument(
        "--blendshape_weights_dir", type=str, default="blendshape_weights"
    )
    return parser.parse_args()


def save_images(name, saved_images_dir, blendshape_weights_dir, device):
    ## Loading the Blendshape Weights
    blendshape_weights_path = os.path.join(blendshape_weights_dir, f"{name}.pt")
    blendshape_weights = torch.load(blendshape_weights_path)

    ## Generating the Images for Source and Target Models
    b0 = blendshape_weights["b0"]
    B = blendshape_weights["B"]
    base_faces = blendshape_weights["base_faces"]
    w = blendshape_weights["w"]

    num_meshes = w.shape[0]

    new_verts = (b0 + B @ w.T).T.view(num_meshes, -1, 3)

    images_dir = os.path.join(saved_images_dir, name)
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)

    for i, verts in tqdm(enumerate(new_verts), total=len(new_verts)):
        image_path = os.path.join(images_dir, f"{i}.pt")

        if not os.path.exists(image_path):
            mesh = create_mesh(verts, base_faces, device)

            image = create_image(device, mesh)
            torch.save(image, image_path)


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
        os.mkdir(saved_images_dir)

    blendshape_weights_dir = args.blendshape_weights_dir

    save_images("source", saved_images_dir, blendshape_weights_dir, device)
    save_images("target", saved_images_dir, blendshape_weights_dir, device)
