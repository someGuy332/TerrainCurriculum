from terrain_utils import generate_heightfield,convert_to_trimesh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D 
import noise

terrains = []
for i in range(16):
    terrain = generate_heightfield(4, 18, 0.05)
    # terrain = terrain[:, :]
    terrains.append(terrain)


plt.figure(figsize=(20, 20))
norm = colors.Normalize(vmin=-4, vmax=4)
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(terrains[i], cmap='terrain', norm=norm)
    # plt.colorbar()
plt.tight_layout(h_pad=10.0)
plt.savefig("terrain.png")
