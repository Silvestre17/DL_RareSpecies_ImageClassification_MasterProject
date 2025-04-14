# ==================================================================
# Functions DL Project 
#
# This file contains utility auxiliary functions for the DL project.
#
# Group: 37
# Members:
#   - André Silvestre, 20240502
#   - Diogo Duarte, 20240525
#   - Filipa Pereira, 20240509
#   - Maria Cruz, 20230760
#   - Umeima Adam Mahomed, 20240543
# ==================================================================

# System Libraries  
import os
import random
from pathlib import Path
import importlib
import utilities_InnovativeApproaches
import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
from typing import Union, List
    
# Data Manipulation & Visualization Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.image import imread
from PIL import Image

# Setting seaborn style
sns.set_theme(style="white")


# ------------------------------------------------------------------------
# Function to reload utilities.py in Jupyter Notebook
def reload_utilities():
    """
    Reload the utilities module in Jupyter Notebook.
    
    This function is useful for reloading the utilities module after making changes to it.
    """

    importlib.reload(utilities_InnovativeApproaches)
    print(f"Utilities module reloaded successfully ({datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')})")
    
# ------------------------------------------------------------------------
# Function to plot images in a row with titles
def plot_images_from_directory(image_paths: Union[str, List[Union[str, Path]]],
                               titles: Union[str, List[str]],
                               num_rows: int = 1):
    """
    Plots images in a grid layout with titles.

    Args:
        image_paths (Union[str, List[Union[str, Path]]]): List of image file paths 
                                                          (or Path objects) to display.
        titles (Union[str, List[str]]): List of titles corresponding to each image.
        num_rows (int): Number of rows to arrange the images. Default is 1.

    Returns:
        None: Displays the images in a grid layout.
    """
    # Ensure image_paths and titles are lists
    if isinstance(image_paths, (str, Path)):
        image_paths = [image_paths]
    if isinstance(titles, str):
        titles = [titles]

    # Convert string paths to Path objects for robustness
    image_paths = [Path(p) for p in image_paths]

    # Check if the number of images matches the number of titles
    if len(image_paths) != len(titles):
        raise ValueError("The number of image paths and titles must be the same.")

    # Calculate the number of columns needed
    num_images = len(image_paths)
    num_cols = (num_images + num_rows - 1) // num_rows

    # Create a figure with the specified number of rows and columns
    # Adjust figsize dynamically based on columns and rows
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4.5))

    # Ensure 'axes' is always a flat NumPy array, even if only one subplot is created
    # np.ravel handles single Axes object, 1D array, and 2D array correctly
    axes = np.ravel(axes)

    # Iterate through the images and plot them
    for i, (image_path, title) in enumerate(zip(image_paths, titles)):
        try:
            # Load the image using PIL
            image = Image.open(image_path)

            # Display the image on the i-th Axes object
            axes[i].imshow(image)
            # Set the title for the i-th Axes object
            axes[i].set_title(title, fontsize=10, fontweight='bold')
            # Remove axis for cleaner visualization
            axes[i].axis('off')
            
        except FileNotFoundError:
            print(f"Warning: Image file not found at {image_path}. Skipping.")
            axes[i].text(0.5, 0.5, 'Image Not Found', horizontalalignment='center', verticalalignment='center')
            axes[i].set_title(title, fontsize=10, fontweight='bold', color='red')
            axes[i].axis('off')
        
        except Exception as e:
            print(f"Warning: Could not load or plot image {image_path}. Error: {e}")
            axes[i].text(0.5, 0.5, 'Error Loading', horizontalalignment='center', verticalalignment='center')
            axes[i].set_title(title, fontsize=10, fontweight='bold', color='red')
            axes[i].axis('off')


    # Hide any unused subplots at the end
    for j in range(num_images, len(axes)):
        axes[j].axis('off')

    # Adjust layout to prevent titles/labels overlapping
    plt.tight_layout()
    plt.show()
    
# ------------------------------------------------------------------------
# 
# Source: https://medium.com/@kerry.halupka/getting-started-with-openais-clip-a3b8f5277867
#         
# Function to visualize images and their probabilities
# Updated to fix the TypeError and ensure proper layout of image and graph
# ------------------------------------------------------------------------
def visualize_images_and_probs(images: Image.Image, probs, 
                               classes: List[str], title: str = "Image Probabilities"):
    """
    Visualizes an image alongside its corresponding probabilities in a horizontal layout.

    Args:
        images (Image.Image): A PIL Image object to display.
        probs (torch.Tensor): A 1D tensor of probabilities for each class.
        classes (List[str]): A list of class labels corresponding to the probabilities.
        title (str): Title for the plot. Default is "Image Probabilities".

    Returns:
        None: Displays the image and its probabilities side by side.
    """
    # Ensure probabilities are a 1D NumPy array
    prob_values = probs.detach().numpy().flatten()

        # Check if the length of probs and classes match
    if len(prob_values) != len(classes):
        raise ValueError(f"Length mismatch: {len(prob_values)} probabilities and {len(classes)} classes.")

    # Determine the color for each bar based on the class name
    color_palette_dict = {'animal': '#22c1c3','not animal': '#090979',}
    bar_colors = [color_palette_dict.get(cls, '#808080') for cls in classes] # Default to gray if class not in dict

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # --- Plot Image ---
    axes[0].imshow(images)
    axes[0].axis('off')
    axes[0].set_title("Image", fontsize=14, fontweight='bold')

    # --- Plot Probabilities Bar Chart ---
    # Create horizontal bars, assigning colors individually
    y_pos = np.arange(len(classes)) # Positions for the bars
    axes[1].barh(y_pos, prob_values, color=bar_colors, alpha=0.8) # Use the 'color' argument

    # Set y-axis ticks and labels
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(classes, fontsize=10)
    axes[1].invert_yaxis()  # Labels read top-to-bottom

    # Set x-axis label and limits
    axes[1].set_xlabel("Probability", fontsize=12)
    axes[1].set_xlim(0, 1.0)

    # Set title for the bar chart
    axes[1].set_title(title, fontsize=14, fontweight='bold')

    # Add probability values as text labels on the bars
    for index, value in enumerate(prob_values):
        # Position text slightly to the right of the bar end
        axes[1].text(x=value + 0.01,                # x position
                     y=index,                       # y position    
                     s=f"{value*100:.1f}%",         # Text to display
                     va='center', fontsize=10,      # Vertical alignment 
                     color=color_palette_dict.get(classes[index], '#808080'))  # Adjust text color based on the class name


    axes[1].grid(axis='x', linestyle='--', alpha=0.7)

    # Remove spines for a cleaner look using Seaborn's despine
    sns.despine(ax=axes[1], left=True, bottom=True, top=True, right=True)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()