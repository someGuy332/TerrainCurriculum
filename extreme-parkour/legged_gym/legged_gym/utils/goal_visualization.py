import numpy as np

import os
import cv2

def save_image_with_goals(
    image,
    goals,
    idx,
    save_dir,
    goal_radius=0.05,
    goal_color=(255, 0, 0),
    goal_thickness=2,
):
    """
    Save an image with goals drawn on it.

    Args:
        image (np.ndarray): The image to draw on.
        goals (np.ndarray): The goals to draw.
        goal_radius (float): The radius of the goals.
        goal_color (tuple): The color of the goals.
        goal_thickness (int): The thickness of the goal lines.
    """

    for goal in goals:
        cv2.circle(image, tuple(goal[:2].astype(int)), int(goal_radius), goal_color, goal_thickness)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = os.path.join(save_dir, f"image{idx}.png")
    cv2.imwrite(filename, image)
    

def save_image(image, save_dir):
    """
    Save an image to a directory.

    Args:
        image (np.ndarray): The image to save.
        save_dir (str): The directory to save the image in.
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = os.path.join(save_dir, "image.png")
    cv2.imwrite(filename, image)