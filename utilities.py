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
import importlib
import utilities
import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    
# Data Manipulation & Visualization Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.image import imread
from PIL import Image

# Setting seaborn style
sns.set_theme(style="white")


# Machine Learning Libraries
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# TensorFlow Libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory

# ------------------------------------------------------------------------
# Function to reload utilities.py in Jupyter Notebook
def reload_utilities():
    """
    Reload the utilities module in Jupyter Notebook.
    
    This function is useful for reloading the utilities module after making changes to it.
    """

    importlib.reload(utilities)
    print(f"Utilities module reloaded successfully ({datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')})")
    
# ------------------------------------------------------------------------
# Function to load images from a directory
def load_images_from_directory(# Directory paths 
                               train_dir: str,
                               val_dir: str,
                               test_dir: str,
                               
                               # Optional parameters for image loading
                               labels = 'inferred',
                               label_mode = 'categorical',
                               class_names = None,
                               color_mode = 'rgb',
                               batch_size: int = 32,
                               image_size: tuple = (256, 256),
                               seed: int = 2025,
                               interpolation: str = 'bilinear',
                               crop_to_aspect_ratio: bool = False,
                               pad_to_aspect_ratio: bool = False) -> tf.data.Dataset:
    """
    Load images from a specified directory using TensorFlow's image_dataset_from_directory.
    This function creates a data generator for loading images, which can be used for training, validation, or testing.
    The images are resized to the specified size and can be loaded in either RGB or grayscale color mode.
    The function also includes data augmentation options, such as rotation, zoom, and horizontal flipping.

    Args:
        train_dir (str): Path to the training directory.
        val_dir (str): Path to the validation directory.
        test_dir (str): Path to the test directory.
        
        # Optional parameters for image loading 
        # Source: https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
        labels (str): Type of labels to generate. Default is 'inferred'.
        label_mode (str): Type of labels to generate. Default is 'categorical'.
        class_names (list): List of class names. Default is None.
        color_mode (str): Color mode to read images. Default is 'rgb'.
        batch_size (int): Size of the batches of data. Default is 32.
        image_size (tuple): Size of the images to read (height, width). Default is (256, 256) == default size for tf.keras.preprocessing.image_dataset_from_directory
        seed (int): Random seed for shuffling and transformations. Default is 2025.
        interpolation (str): Interpolation method to resample the image. Default is 'bilinear'.
        crop_to_aspect_ratio (bool): Whether to crop the image to the aspect ratio. Default is False.
        pad_to_aspect_ratio (bool): Whether to pad the image to the aspect ratio. Default is False.
        
    Returns:
        train_datagen (tf.data.Dataset): Data generator for training data.
        val_datagen (tf.data.Dataset): Data generator for validation data.
        test_datagen (tf.data.Dataset): Data generator for test data.
    Raises:
        FileNotFoundError: If the specified directories do not exist.
        
    ----
    Assumptions:
        - The directory structure is assumed to be organized like this:
        data/RareSpecies_Split/
                ├── train
                │   ├── class1
                |   |     ├── image1.1.jpg        
                |   |     ├── image1.2.jpg        
                |   |     └── image1.3.jpg        
                │   ├── class2
                |   |     ├── image2.1.jpg
                ├── val
                ├── test
    """
    # Check if the directories exist
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory '{train_dir}' does not exist.")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation directory '{val_dir}' does not exist.")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory '{test_dir}' does not exist.")
    
    # Data generators with built-in rescaling (no augmentation yet)
    # Source: https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
    #         https://stackoverflow.com/questions/59228816/what-do-the-tensorflow-datasets-functions-cache-and-prefetch-do

    # Training data generator
    train_datagen = image_dataset_from_directory(
        train_dir,                                      # Path to the directory
        labels=labels,                                  # Type of labels to generate (inferred = from the directory structure)
        label_mode=label_mode,                          # Type of labels to generate (categorical = 'float32' tensor of shape (batch_size, num_classes), representing a one-hot encoding of the class index.)
        class_names=class_names,                        # List of class names (if None, the class names are inferred from the directory structure)
        color_mode=color_mode,                          # Color mode to read images
        batch_size=batch_size,                          # Size of the batches of data
        image_size=image_size,                          # Size of the images to read (256x256)
        shuffle=True,                                   # Whether to shuffle the data - True for training data 
        seed=seed,                                      # Random seed for shuffling and transformations
        interpolation=interpolation,                    # Interpolation method to resample the image
        crop_to_aspect_ratio=crop_to_aspect_ratio,      # Whether to crop the image to the aspect ratio
        pad_to_aspect_ratio=pad_to_aspect_ratio         # Whether to pad the image to the aspect ratio
    )

    # Validation data generator
    val_datagen = image_dataset_from_directory(val_dir, labels=labels, label_mode=label_mode, image_size=image_size, batch_size=batch_size,
                                               class_names=class_names, color_mode=color_mode, interpolation=interpolation, 
                                               seed=seed, shuffle=False,   # No shuffling for validation data
                                               crop_to_aspect_ratio=crop_to_aspect_ratio, pad_to_aspect_ratio=pad_to_aspect_ratio)

    # Test data generator
    test_datagen = image_dataset_from_directory(test_dir, labels=labels, label_mode=label_mode, image_size=image_size, batch_size=batch_size,
                                                class_names=class_names, color_mode=color_mode, interpolation=interpolation,
                                                seed=seed, shuffle=False,  # No shuffling for test data
                                                crop_to_aspect_ratio=crop_to_aspect_ratio, pad_to_aspect_ratio=pad_to_aspect_ratio)
    
    
    # --- Optimize Data Pipelines with Prefetch ---
    # Apply prefetch to all datasets for performance optimization
    # Source: https://www.tensorflow.org/guide/data_performance#prefetching
    train_datagen = train_datagen.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_datagen = val_datagen.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_datagen = test_datagen.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    
    # Return the data generators
    return train_datagen, val_datagen, test_datagen

# ------------------------------------------------------------------------
# Function to show an image from a given path
def show_image(image_path: str) -> None:
    """
    Displays an image using Matplotlib.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        None. Displays the image.
    """
    image = Image.open(image_path)      # Open the image
    plt.figure(figsize=(10, 10))        # Set the figure size   
    plt.imshow(image)                   # Display the image
    plt.axis('off')                     # Remove axis for cleaner visualization
    plt.show()                          # Show the image
    


# ------------------------------------------------------------------------
# Auxiliary Function to display multiple dataframes side by side
# Source: https://python.plainenglish.io/displaying-multiple-dataframes-side-by-side-in-jupyter-lab-notebook-9a4649a4940
from IPython.display import display_html
from itertools import chain,cycle
def display_side_by_side(*args, super_title: str, titles=cycle([''])):
    """
    :param args: Variable number of DataFrame objects to be displayed side by side.
    :param super_title: The main title to be displayed at the top of the combined view.
    :param titles: An iterable containing titles for each DataFrame to be displayed. Defaults to an infinite cycle of empty strings.
    
    :return: None. The function generates and displays HTML content side by side for given DataFrames.
    """
    html_str = ''
    html_str += f'<h1 style="text-align: left; margin-bottom: -15px;">{super_title}</h1><br>'
    html_str += '<div style="display: flex;">'
    for df, title in zip(args, chain(titles, cycle(['</br>']))):
        html_str += f'<div style="margin-right: 20px;"><h3 style="text-align: center;color:#555555;">{title}</h3>'
        html_str += df.to_html().replace('table', 'table style="display:inline; margin-right: 20px;"')
        html_str += '</div>'
    html_str += '</div>'
    display_html(html_str, raw=True)

# ------------------------------------------------------------------------
# Function to create a DataFrame for model evaluation metrics
def create_evaluation_dataframe(model_name: str, variation: str, 
                                train_metrics: dict, val_metrics: dict, test_metrics: dict,
                                train_time: float, round_decimals: int = 4) -> pd.DataFrame:
    """
    Create a MultiIndex DataFrame for model evaluation metrics with row and column levels.

    Args:
        model_name(str) : Name of the model (e.g., "Baseline Model")
        variation(str) : Variation or experiment label (e.g., "W/ SMOTE", "Tuned").
        train_metrics(dict) : Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1_score', 'auc'.
        val_metrics(dict) : Same structure as train_metrics, for the validation set.
        test_metrics(dict) : Same structure as train_metrics, for the test set.
        train_time(float) : Time of execution in seconds.
        round_decimals(int, optional) : Number of decimal places to round the metrics (default is 4).

    Returns:
        df(pd.DataFrame) : A styled MultiIndex DataFrame with metrics across Train/Validation/Test sets.
    """
    
    # Define column MultiIndex: (Set, Metric)
    column_index = pd.MultiIndex.from_product(
        [["Train", "Validation", "Test"],
         ["Accuracy", "Precision", "Recall", "F1 Score", "AUROC"]],
        names=["Set", "Metric"]
    )
    
    # Collect metrics in the correct order
    metrics_data = [
        train_metrics['accuracy'], train_metrics['precision'], train_metrics['recall'], train_metrics['f1_score'], train_metrics['auc'],
        val_metrics['accuracy'], val_metrics['precision'], val_metrics['recall'], val_metrics['f1_score'], val_metrics['auc'],
        test_metrics['accuracy'], test_metrics['precision'], test_metrics['recall'], test_metrics['f1_score'], test_metrics['auc'],
    ]
    
    # Round values and create the DataFrame
    data = [np.round(metrics_data, round_decimals)]
    df = pd.DataFrame(data, columns=column_index)
    
    # Add time of execution column
    df.insert(0, ("", "Time of Execution"), round(train_time, 2))                                    # Round time to 2 decimal places
    
    # Set row MultiIndex: (Model, Variation)
    df.index = pd.MultiIndex.from_tuples([(model_name, variation)], names=["Model", "Variation"])
    
    # Clean column index names for better readability
    df.columns.names = ["", ""]
    
    return df



# ------------------------------------------------------------------------
# Function to plot metrics
def plot_metrics(history: tf.keras.callbacks.History, file_path=None, model_name=None):
    """Plots training and validation metrics.
    Args:
        history (History): Keras History object containing training metrics.
        file_path (str): Path to save the plot. If None, the plot is displayed only.
        model_name (str): Name of the model for the title.
    Returns:
        Show the plot.
    """
    # Create a figure with 5 subplots (1 row, 5 columns)
    fig, ax = plt.subplots(1, 5, figsize=(20, 4))
    
    # Plot each metric (loss, accuracy, precision, recall, f1_score)
    metrics = [('loss', 'Loss'), ('accuracy', 'Accuracy'), ('precision', 'Precision'),  ('recall', 'Recall'), ('f1_score', 'F1 Score')]
    
    # Iterate through the metrics and plot them
    for i, (metric, title) in enumerate(metrics):
        ax[i].plot(history.history[metric], label='Train', color='#22c1c3')
        ax[i].plot(history.history[f'val_{metric}'], label='Val', color='#090979')
        ax[i].set_title(title, fontsize=14, fontweight='bold')
        ax[i].set_xlabel('Epoch', fontsize=12)
        ax[i].set_ylabel(title, fontsize=12)
        ax[i].legend()
        ax[i].set_yticklabels([f'{int(t*100)}%' if metric != 'loss' else t for t in ax[i].get_yticks()])
        
        # Remove the top and right spines
        sns.despine(top=True, right=True)
        
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if a file path is provided
    if file_path:
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
        
    # Add a title if model_name is provided
    if model_name:
        plt.suptitle(f'Training and Validation Metrics\n {model_name}', fontsize=16, fontweight='bold')
        
    # Plot the metrics
    plt.show()
    
    
# ------------------------------------------------------------------------
# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title=None, cmap=plt.cm.Blues, file_path=None):
    """
    This function plots a confusion matrix using Matplotlib and Seaborn.
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        title (str): Title for the plot. Default is None.
        cmap: Colormap to use. Default is plt.cm.Blues.
        file_path (str): Path to save the plot. If None, the plot is displayed only.

    Returns:
        None: Displays the confusion matrix plot.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Transform the confusion matrix into a DataFrame
    cm_df = pd.DataFrame(cm)
    
    # Plot the confusion matrix using Seaborn heatmap
    fig = plt.figure(figsize=(30, 30))
    
    # Show only the lower triangle of the matrix and values more than 0
    mask_annot = cm_df.values > 0
    annot = np.where(mask_annot, cm_df.values, np.full(cm_df.shape,""))

    # Create a heatmap
    sns.heatmap(cm_df, annot=annot, fmt='s', cmap=cmap, cbar=False, annot_kws={"size": 6})   
    
    # Set plot title and labels
    plt.title(title or 'Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    
    # Set tick labels
    plt.xticks(ticks=np.arange(len(np.unique(y_true))), labels=np.unique(y_true), rotation=90, fontsize=8)
    plt.yticks(ticks=np.arange(len(np.unique(y_true))), labels=np.unique(y_true), rotation=0, fontsize=8)
    plt.gca().set_aspect('equal')  # Set aspect ratio to be equal
    
    if file_path:
        # Save the figure if a file path is provided
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {file_path}")

    # Show the plot
    plt.show()

# ------------------------------------------------------------------------
# Function to plot 5 right predictions and 5 wrong predictions (image pred, sample image of the true class, predicted class)
def plot_predictions(model, test_data, num_images=5, file_path=None):
    """
    Plot 5 right predictions and 5 wrong predictions from the test data.
    
    Args:
        model (tf.keras.Model): Trained model for making predictions.
        test_data (tf.data.Dataset): Test dataset containing images and labels.
        num_images (int): Number of images to plot. Default is 5.
        file_path (str): Path to save the plot. If None, the plot is displayed only.
        
    Returns:
        None: Displays or saves the plot.
    """
    # Get the true labels and predicted labels
    y_true = np.concatenate([y.numpy() for _, y in test_data], axis=0)
    y_pred = np.concatenate([model.predict(x) for x, _ in test_data], axis=0)
    
    # Get the predicted class indices
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get the true class indices
    y_true_classes = np.argmax(y_true, axis=1)
    
    # Get the indices of right and wrong predictions
    right_indices = np.where(y_pred_classes == y_true_classes)[0]
    wrong_indices = np.where(y_pred_classes != y_true_classes)[0]
    
    # Select random indices for right and wrong predictions
    right_indices = np.random.choice(right_indices, num_images, replace=False)
    wrong_indices = np.random.choice(wrong_indices, num_images, replace=False)
    
    # Create a figure with 2 subplots (1 row, 2 columns)
    fig, ax = plt.subplots(2, num_images, figsize=(20, 10))
    
    # Plot right predictions
    for i in range(num_images):
        ax[0, i].imshow(test_data[right_indices[i]][0][0].numpy().astype(np.uint8))
        ax[0, i].set_title(f'True: {y_true_classes[right_indices[i]]}, Pred: {y_pred_classes[right_indices[i]]}')
        ax[0, i].axis('off')
    
    # Plot wrong predictions
    for i in range(num_images):
        ax[1, i].imshow(test_data[wrong_indices[i]][0][0].numpy().astype(np.uint8))
        ax[1, i].set_title(f'True: {y_true_classes[wrong_indices[i]]}, Pred: {y_pred_classes[wrong_indices[i]]}')
        ax[1, i].axis('off')
        
    # Set the title for the entire figure
    fig.suptitle('Right and Wrong Predictions', fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if a file path is provided
    if file_path:
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Predictions plot saved to {file_path}")
        
    # Show the plot
    plt.show()

    
# ------------------------------------------------------------------------
# Function to clear all variables in the current namespace
def clear_all_variables():
    """
    Clear all variables in the current namespace.
    
    This function is useful for resetting the environment and freeing up memory.
    """
    # Clear all variables in the current namespace
    for name in dir():
        if not name.startswith('_'):
            del globals()[name]
    print(f"All variables cleared successfully ({datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')})")
    
    # Clear the console
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Console cleared successfully")
    print(f"Console cleared successfully ({datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')})")