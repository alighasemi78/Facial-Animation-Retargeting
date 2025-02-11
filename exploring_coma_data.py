# Exploring CoMA Data

## Importing the Required Libraries
import argparse
import torch
import cv2
import os
from pytorch3d.io import load_ply
from tqdm import tqdm
from utilities import create_mesh, create_image
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="COMA_data")
    parser.add_argument("--saved_videos_dir", type=str, default="videos")
    return parser.parse_args()


## Visualizing the Animation of Some Faces
def expression_to_video(
    frames, expression_dir, output_video, device, image_size=128, fps=10
):
    # Create OpenCV video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (image_size, image_size))

    for frame in tqdm(frames, leave=False):
        frame_path = os.path.join(expression_dir, frame)

        verts, faces = load_ply(frame_path)
        center = verts.mean(0)
        verts -= center
        scale = max(verts.abs().max(0)[0])
        verts /= scale

        mesh = create_mesh(verts, faces, device)

        rendered_image = create_image(device, mesh)
        rendered_image = rendered_image.permute(1, 2, 0).cpu().numpy()

        rendered_image = (rendered_image * 255).astype(
            np.uint8
        )  # Convert to 8-bit image

        # Write the frame to the video
        video_writer.write(rendered_image)

    # Release video writer
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ## Initial Settings
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        print("WARNING: CPU only, this will be slow!")

    args = get_args()

    saved_videos_dir = args.saved_videos_dir
    if not os.path.exists(saved_videos_dir):
        os.makedirs(saved_videos_dir)

    data_dir = args.data_dir

    faces = os.listdir(data_dir)
    faces.sort()

    face_indices = [0, 1]

    faces = [faces[i] for i in face_indices]
    for face in tqdm(faces):
        face_dir = os.path.join(data_dir, face)
        if os.path.isdir(face_dir):
            expressions = os.listdir(face_dir)
            expressions.sort()
            for expression in tqdm(expressions, leave=False):
                expression_dir = os.path.join(face_dir, expression)
                frames = os.listdir(expression_dir)
                frames.sort()

                output_video_path = os.path.join(
                    saved_videos_dir, f"{face}_{expression}.mp4"
                )
                expression_to_video(frames, expression_dir, output_video_path, device)
