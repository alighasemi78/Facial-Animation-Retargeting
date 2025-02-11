# Utility Functions for the Project

## Importing the Required Libraries
import torch
from pytorch3d.renderer import (
    TexturesVertex,
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    PointLights,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
)
from pytorch3d.structures import Meshes
import cv2


## Defining the Functions
def create_mesh(verts, faces, device):
    # Initialize each vertex to be white in color
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    # Create a Meshes object
    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures,
    )

    return mesh


def create_renderer(device):
    # Initialize a camera
    R, T = look_at_view_transform(5, 0, 0)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Define the settings for rasterization and shading
    raster_settings = RasterizationSettings(
        image_size=1024,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Place a point light in front of the object
    lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])

    # Create a Phong renderer by composing a rasterizer and a shader
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
        ),
    )

    return renderer


def create_image(device, mesh, requires_grad=False):
    renderer = create_renderer(device)
    images = renderer(mesh)
    image = images[0, ..., :3]

    height, width, _ = image.shape

    crop_size = 200
    x_start = (width - crop_size) // 2
    y_start = (height - crop_size) // 2
    x_end = x_start + crop_size
    y_end = y_start + crop_size

    image = image[y_start:y_end, x_start:x_end, :]
    if requires_grad:
        image = image.detach().cpu().numpy()
    else:
        image = image.cpu().numpy()
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)

    image = torch.from_numpy(image)
    image = image.to(device)
    image = image.permute(2, 0, 1)
    image = (image - 0.5) * 2

    return image
