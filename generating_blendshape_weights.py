# Generating Blendshape Weights

## Importing the Required Libraries
import argparse
import torch
import os
from utilities import create_mesh, create_image
from tqdm import tqdm
from pytorch3d.io import load_ply
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="COMA_data")
    parser.add_argument(
        "--saved_images_dir", type=str, default="blendshape_weights_images"
    )
    parser.add_argument("--saved_models_dir", type=str, default="blendshape_weights")
    return parser.parse_args()


## Creating the Blendshapes
### From the notebook "Exploring CoMA Data", we understand that in order to find the max expression,
### we need to find the frame with the most displacement in the vertices.
def get_blendshapes(face_dir):
    neutral_verts = None
    neutral_faces = None
    max_expression_verts = []

    expressions = os.listdir(face_dir)
    expressions.sort()

    for expression in tqdm(expressions):
        expression_dir = os.path.join(face_dir, expression)
        frames = os.listdir(expression_dir)
        frames.sort()

        first_frame_path = os.path.join(expression_dir, frames[0])
        first_verts, faces = load_ply(first_frame_path)
        center = first_verts.mean(0)
        first_verts -= center
        scale = max(first_verts.abs().max(0)[0])
        first_verts /= scale

        max_displacement = 0.0
        max_verts = None

        for frame in tqdm(frames, leave=False):
            frame_path = os.path.join(expression_dir, frame)
            verts, _ = load_ply(frame_path)
            center = verts.mean(0)
            verts -= center
            scale = max(verts.abs().max(0)[0])
            verts /= scale

            displacement = torch.norm(verts - first_verts, dim=1).mean().item()
            if displacement > max_displacement:
                max_displacement = displacement
                max_verts = verts

        if neutral_verts is None:
            neutral_verts = first_verts.view(-1, 1)
            neutral_faces = faces

        max_expression_verts.append(max_verts.view(-1, 1))

    max_expression_verts = torch.cat(max_expression_verts, dim=1)
    return {
        "base_faces": neutral_faces,
        "base_blendshape": neutral_verts,
        "delta_blendshapes": max_expression_verts - neutral_verts,
    }


## Generating Blendshape Weights
def get_blendshape_weight(blendshape, device, saved_images_dir, image_name):
    num_meshes = 10_000

    b0 = blendshape["base_blendshape"]
    B = blendshape["delta_blendshapes"]

    num_blendshapes = B.shape[1]

    w = torch.zeros(num_meshes, num_blendshapes)

    # For each mesh, randomly select 1-3 blendshapes to emphasize
    for i in tqdm(range(num_meshes)):
        num_active = torch.randint(1, 4, (1,)).item()  # Choose 1 to 3 blendshapes
        active_indices = torch.randint(
            0, num_blendshapes, (num_active,)
        )  # Indices of active blendshapes
        w[i, active_indices] = (
            torch.rand(num_active) * 0.5 + 0.5
        )  # Values in [0.5, 1.0]

    # Normalize weights so each row sums to 1
    w /= w.sum(dim=1, keepdim=True)

    new_verts = (b0 + B @ w.T).T.view(num_meshes, -1, 3)

    base_faces = blendshape["base_faces"]

    meshes = [create_mesh(verts, base_faces, device) for verts in new_verts[:4]]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i, (mesh, ax) in enumerate(zip(meshes, axes.flatten())):
        image = create_image(device, mesh)
        image = image.permute(1, 2, 0).cpu().numpy()

        ax.imshow(image)
        ax.axis("off")

    plt.tight_layout()

    save_path = os.path.join(saved_images_dir, f"{image_name}.png")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return {
        "b0": b0,
        "B": B,
        "base_faces": blendshape["base_faces"],
        "w": w,
    }


def save_weights(name, face, data_path, device, saved_images_dir, saved_models_dir):
    face_dir = os.path.join(data_path, face)

    blendshapes = get_blendshapes(face_dir)

    blendshape_weights = get_blendshape_weight(
        blendshapes, device, saved_images_dir, name
    )

    blendshape_weights_path = os.path.join(saved_models_dir, f"{name}.pt")
    torch.save(blendshape_weights, blendshape_weights_path)


if __name__ == "__main__":
    ## Initial Settings
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        print("WARNING: CPU only, this will be slow!")

    args = get_args()

    data_path = args.data_dir

    saved_images_dir = args.saved_images_dir
    if not os.path.exists(saved_images_dir):
        os.mkdir(saved_images_dir)

    saved_models_dir = args.saved_models_dir
    if not os.path.exists(saved_models_dir):
        os.mkdir(saved_models_dir)

    source = 0
    target = 1

    faces = os.listdir(data_path)
    faces.sort()

    source_face = faces[source]
    target_face = faces[target]

    save_weights(
        "source", source_face, data_path, device, saved_images_dir, saved_models_dir
    )
    save_weights(
        "target", target_face, data_path, device, saved_images_dir, saved_models_dir
    )
