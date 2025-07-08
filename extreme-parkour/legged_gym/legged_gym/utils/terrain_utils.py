import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
from mpl_toolkits.mplot3d import Axes3D 

random.seed(time.time())
np.random.seed(int(time.time()) + np.random.randint(0, 10000))

def plot_terrain_3d(noise_grid, xy_spacing=0.1):
    height, width = noise_grid.shape
    # Create a grid of x and y coordinates (in meters)
    X, Y = np.meshgrid(np.arange(width) * xy_spacing, 
                       np.arange(height) * xy_spacing)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    norm = colors.Normalize(vmin=-2, vmax=2)
    surf = ax.plot_surface(X, Y, noise_grid, cmap='terrain', linewidth=0, antialiased=True, norm=norm)
    
    
    # Set the axes limits to reflect physical scale
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(noise_grid.min(), noise_grid.max())
    
    # Set the box aspect to reflect true dimensions.
    # This makes the x, y, and z scales match the physical measurements.
    x_range = X.max() - X.min()
    y_range = Y.max() - Y.min()
    z_range = noise_grid.max() - noise_grid.min()
    ax.set_box_aspect((x_range, y_range, z_range))
    
    # Add colorbar and labels
    fig.colorbar(surf, shrink=0.5, aspect=10)
    ax.set_title("3D Terrain (Physical Scale)")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    # ax.set_zlabel("Height (meters)")
    
    plt.show()

class Vector2: 
    def __init__(self, x, y): 
        self.x = x 
        self.y = y

    def dot(self, other):
        return self.x * other.x + self.y * other.y
    
def shuffle(array_to_shuffle): 
    for e in range(len(array_to_shuffle)-1, 0, -1): 
        index = random.randint(0, e-1) 
        array_to_shuffle[e], array_to_shuffle[index] = array_to_shuffle[index], array_to_shuffle[e]

def make_permutation(): 
    permutation = list(range(256)) 
    shuffle(permutation) 
    permutation.extend(permutation) # Repeat the array return permutation

    return permutation


def get_constant_vector(v): # v is the value from the permutation table 
    h = v & 3 
    if h == 0: 
        return Vector2(1.0, 1.0) 
    elif h == 1: 
        return Vector2(-1.0, 1.0) 
    elif h == 2: 
        return Vector2(-1.0, -1.0) 
    else: 
        return Vector2(1.0, -1.0)

def fade(t): 
    return ((6 * t - 15) * t + 10) * t * t * t

def lerp(t, a1, a2): 
    return a1 + t * (a2 - a1)

def noise2d(x, y, permutation): 
    X = math.floor(x) & 255 
    Y = math.floor(y) & 255

    xf = x - math.floor(x)
    yf = y - math.floor(y)

    top_right = Vector2(xf - 1.0, yf - 1.0)
    top_left = Vector2(xf, yf - 1.0)
    bottom_right = Vector2(xf - 1.0, yf)
    bottom_left = Vector2(xf, yf)

    # Select a value from the permutation array for each of the 4 corners
    value_top_right = permutation[permutation[X + 1] + Y + 1]
    value_top_left = permutation[permutation[X] + Y + 1]
    value_bottom_right = permutation[permutation[X + 1] + Y]
    value_bottom_left = permutation[permutation[X] + Y]

    dot_top_right = top_right.dot(get_constant_vector(value_top_right))
    dot_top_left = top_left.dot(get_constant_vector(value_top_left))
    dot_bottom_right = bottom_right.dot(get_constant_vector(value_bottom_right))
    dot_bottom_left = bottom_left.dot(get_constant_vector(value_bottom_left))

    u = fade(xf)
    v = fade(yf)

    return lerp(u,
                lerp(v, dot_bottom_left, dot_top_left),
                lerp(v, dot_bottom_right, dot_top_right))


def perlin2d(width:int, length:int, octaves=6, lacunarity=1.5, persistence=0.5, xy_scale = 0.05):
    hmap = np.zeros((width, length))

    base_scale = 20 / xy_scale

    for octave in range(octaves):
        perm = make_permutation()
        xy_scale = base_scale / (lacunarity ** octave)
        z_scale = persistence ** octave
        for i in range(width):
            for j in range(length):
                hmap[i,j] += noise2d(i / xy_scale, j / xy_scale, perm) * z_scale

    return hmap

def generate_dynamic_terrain(width:int, length:int, max_z_scale = 2.5, xy_scale = 0.05):
    hmap = perlin2d(width, 
                    length, 
                    persistence=np.random.uniform(0.6, 0.95), 
                    lacunarity=np.random.uniform(1.2, 1.5), 
                    xy_scale = xy_scale)

    hmap *= np.random.uniform(0.5, max_z_scale)
    return hmap

def generate_continental_terrain(width:int, length:int, max_z_scale = 2.5, xy_scale = 0.05):
    hmap = perlin2d(width, length, octaves=1, xy_scale = xy_scale)

    hmap *= np.random.uniform(0.0, max_z_scale)
    hmap += np.random.randn(hmap.shape[0], hmap.shape[1]) * np.random.uniform(0.0, 0.015)

    return hmap

def discretize_terrain(terrain:np.array, height):
    # height_discrete = terrain // height
    width, length = terrain.shape

    i_range = range(width)
    j_range = range(length)
    if np.random.uniform(0.0, 1.0) < 0.5:
        i_range = range(width -1, -1, -1)
    if np.random.uniform(0.0, 1.0) < 0.5:
        j_range = range(length -1, -1, -1)

    for i in i_range:
        for j in j_range:
            min_x = max(i-1, 0)
            max_x = min(i+1, width - 1)
            min_y = max(j-1, 0)
            max_y = min(j+1, length - 1)

            terrain[i, j] = np.median(terrain[min_x:max_x, min_y:max_y])

    terrain = (terrain // height) * height
    terrain += np.random.randn(width, length) * np.random.uniform(0.0, 0.015)
    return terrain

def generate_pyramid_stairs(width, length, xy_scale=0.1, step_width=0.25, step_height=0.2, platform_size=1.):
    """
    Generate stairs

    Parameters:
        terrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float): the step_height [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """

    height_field = np.zeros((length, width))
    # switch parameters to discrete units
    step_width = int(step_width / xy_scale)
    step_height = step_height
    platform_size = int(platform_size / xy_scale)
    
    h = 0
    start_x = 0
    stop_x = length
    start_y = 0
    stop_y = width

    while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
        start_x += step_width
        stop_x -= step_width
        start_y += step_width
        stop_y -= step_width
        h += step_height
        height_field[start_x:stop_x, start_y:stop_y] = h

    # print(height_field)
    return height_field


def generate_heightfield(width, length, xy_scale, seed=None):
    if seed is not None:
        np.random.seed(int(time.time()) + seed)
    length = int(length / xy_scale)
    width = int(width / xy_scale)
    # Generate Perlin noise terrain
    if np.random.uniform(0.0, 1.0) < 0.3:
        # gemerate terrain with stairs
        terrain = generate_continental_terrain(width, length, max_z_scale=2.0, xy_scale=xy_scale)

        num_stairs = 0

        remaining_length = length
        num_stairs = 0
        
        while remaining_length > 0:
            min_x = width * num_stairs
            max_x = width * num_stairs + min(width, remaining_length)
            min_y = 0
            max_y = width

            size_x = np.random.randint((max_x - min_x) // 2, max_x - min_x)
            size_y = np.random.randint((max_y - min_y) // 2, max_y - min_y)

            start_x = np.random.randint(min_x, max_x - size_x + 1)
            start_y = np.random.randint(min_y, max_y - size_y + 1)

            terrain[start_y : start_y + size_y, start_x : start_x + size_x] += generate_pyramid_stairs(size_x, size_y, 
                                                                                                       xy_scale=xy_scale, 
                                                                                                       step_width=np.random.uniform(0.25,0.5), 
                                                                                                       step_height=np.random.uniform(-0.4,0.4)
                                                                                                       )

            num_stairs += 1
            remaining_length -= width

    elif np.random.uniform(0.0, 1.0) < 0.7:
        # gemerate terrain with obstacles
        terrain = generate_continental_terrain(width, length, xy_scale=xy_scale)
        terrain += generate_dynamic_terrain(width, length, xy_scale=xy_scale)
    
    else:
        terrain = generate_continental_terrain(width, length, xy_scale=xy_scale)
        terrain += generate_dynamic_terrain(width, length, xy_scale=xy_scale)
        terrain = discretize_terrain(terrain, np.random.uniform(0.15, 0.3))
    # plot_terrain_3d(terrain, xy_spacing=xy_scale)
    return terrain

def convert_to_trimesh(height_field_raw, horizontal_scale, slope_threshold=1.5):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:

        slope_threshold *= horizontal_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows-1, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows-1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols-1] += (hf[:, 1:num_cols] - hf[:, :num_cols-1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols-1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows-1, :num_cols-1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows-1, :num_cols-1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows-1, :num_cols-1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners*(move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners*(move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten()
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3

    return vertices, triangles
