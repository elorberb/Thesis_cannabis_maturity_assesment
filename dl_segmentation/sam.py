import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import sys

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)


def show_anns(anns):
    # If there are no annotations, exit the function early
    if len(anns) == 0:
        return

    # Sort the annotations by area in descending order
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    # Get the current axis (plot) on which to draw
    ax = plt.gca()
    # Disable autoscaling of the axis
    ax.set_autoscale_on(False)

    # Create a transparent image of the same size as the first (largest) annotation
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0

    # Loop over each annotation
    for ann in sorted_anns:
        # Get the mask for this annotation
        m = ann['segmentation']

        # Generate a random color for this mask (RGB + alpha, where alpha is 0.35)
        color_mask = np.concatenate([np.random.random(3), [0.35]])

        # Apply the color to the mask on the image
        img[m] = color_mask

    # Display the image with the colored masks
    ax.imshow(img)


def segments_patches_SAM(images_names):
    segmentation_dict = {}

    for image, name in images_names:
        mask = mask_generator.generate(image)
        num_segments = len(mask)

        # Initialize an array of False with the same shape as the segment masks
        instance_bitmap = np.zeros_like(mask[0]['segmentation'], dtype=bool)

        # create a single instance bitmap
        for seg in mask:
            instance_bitmap = np.logical_or(instance_bitmap, seg['segmentation'])

        segmentation_dict[name] = {
            'mask': mask,
            'num_segments': num_segments,
            "instance_bitmap": instance_bitmap,
        }

    return segmentation_dict
