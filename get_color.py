import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
import matplotlib.pyplot as plt
from pytorch3d.io import save_obj
def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()
# Sample vertices and faces (replace with your actual mesh data)
import trimesh
mm = trimesh.load("")

# Compute a correspondence metric for demonstration (e.g., distance to some point)
# Here, we're just using the y-coordinate as an example.
correspondence_metric = mm.vertices[:, 1]

# Map metric to color using matplotlib's 'plasma' colormap
cmap = plt.get_cmap("jet")
normed_metric = (correspondence_metric - correspondence_metric.min()) / (correspondence_metric.max() - correspondence_metric.min())
normed_metric_cpu = normed_metric  # Move to CPU if the tensor is on GPU
colors = torch.tensor([cmap(value)[:3] for value in normed_metric_cpu])
import numpy as np
np.save("vis_c.npy", colors)
# Create texture
# vertex_textures = TexturesVertex(verts_features=colors.unsqueeze(0))
# breakpoint()
# Create the mesh
# mesh = Meshes(verts=[verts], faces=[faces], textures=vertex_textures)

save_obj_mesh_with_color("", mm.vertices, mm.faces, colors)