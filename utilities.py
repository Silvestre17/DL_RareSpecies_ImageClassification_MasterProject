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
import utilities
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

# Machine Learning Libraries
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# TensorFlow Libraries
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image             # Use alias to avoid conflict
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
    
    # # --- Optimize Data Pipelines with Prefetch ---
    # # Apply prefetch to all datasets for performance optimization
    # # Source: https://www.tensorflow.org/guide/data_performance#prefetching
    # train_datagen = train_datagen.prefetch(buffer_size=tf.data.AUTOTUNE)
    # val_datagen = val_datagen.prefetch(buffer_size=tf.data.AUTOTUNE)
    # test_datagen = test_datagen.prefetch(buffer_size=tf.data.AUTOTUNE)
    
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
                                train_time: Union[float, str] = None, round_decimals: int = 4, csv_save_path: str = None) -> pd.DataFrame:
    """
    Create a MultiIndex DataFrame for model evaluation metrics with row and column levels.

    Args:
        model_name(str) : Name of the model (e.g., "Baseline Model")
        variation(str) : Variation or experiment label (e.g., "W/ SMOTE", "Tuned").
        train_metrics(dict) : Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1_score', 'auc'.
        val_metrics(dict) : Same structure as train_metrics, for the validation set.
        test_metrics(dict) : Same structure as train_metrics, for the test set.
        train_time(Union[float, str]) : Time of execution in seconds. If None, it will be '' in the DataFrame.
        round_decimals(int, optional) : Number of decimal places to round the metrics (default is 4).
        csv_save_path(str, optional) : Path to save the DataFrame as a CSV file (default is None, no saving).

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
    if train_time is None:
        # If train_time is None, set it to an empty string
        train_time = ''
    else:
        # Round time to 2 decimal places
        train_time = round(train_time, 2)
    # Insert time of execution at the beginning of the DataFrame
    df.insert(0, ("", "Time of Execution"), train_time)
    
    # Set row MultiIndex: (Model, Variation)
    df.index = pd.MultiIndex.from_tuples([(model_name, variation)], names=["Model", "Variation"])
    
    # Clean column index names for better readability
    df.columns.names = ["", ""]
    
    # Save the DataFrame to a CSV file
    if csv_save_path:
        df.to_csv(csv_save_path, index=True)
    
    return df



# ------------------------------------------------------------------------
# Function to plot metrics
def plot_metrics(history: Union[tf.keras.callbacks.History, pd.DataFrame], file_path=None, model_name=None):
    """Plots training and validation metrics.
    Args:
        history (Union[History, pd.DataFrame]): Keras History object or DataFrame containing training metrics.
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
        if type(history) == tf.keras.callbacks.History:
            ax[i].plot(history.history[metric], label='Train', color='#22c1c3')
            ax[i].plot(history.history[f'val_{metric}'], label='Val', color='#090979')
        else:
            # Assuming history is a DataFrame
            ax[i].plot(history[metric], label='Train', color='#22c1c3')
            ax[i].plot(history[f'val_{metric}'], label='Val', color='#090979')
        ax[i].set_title(title, fontsize=14, fontweight='bold')
        ax[i].set_xlabel('Epoch', fontsize=10, fontweight='bold')
        ax[i].set_ylabel(title, fontsize=10, fontweight='bold')
        ax[i].legend()
        ax[i].set_yticklabels([f'{int(t*100)}%' if metric != 'loss' else round(t,2) for t in ax[i].get_yticks()])
        
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
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = None, 
                          cmap: plt.cm = plt.cm.Blues, file_path: str = None):
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
# Function to plot 5 right predictions and 5 wrong predictions 
# (image pred, sample image of the true class, predicted class)
def plot_predictions(model: tf.keras.Model, test_data: tf.data.Dataset, class_names: list, 
                     train_dir: str, num_images: int = 5, file_path: str = None):
    """
    Plot 5 right predictions and 5 wrong predictions from the test data.
    
    Args:
        model (tf.keras.Model): Trained model for making predictions.
        test_data (tf.data.Dataset): Test dataset containing images and labels.
        class_names (list): List of class names corresponding to the labels.
        train_dir (str): Directory where training images are stored.
        num_images (int): Number of images to plot. Default is 5.
        file_path (str): Path to save the plot. If None, the plot is displayed only.
        
    Returns:
        None: Displays or saves the plot.
    """
    # --- 1. Data Collection and Prediction ---
    images_list = []
    labels_list = []

    # Iterate through the dataset to collect all samples
    for images, labels in test_data:
        images_list.append(images.numpy()) # Convert tensor to numpy array
        labels_list.append(labels.numpy())  

    # Check if data was loaded
    if not images_list or not labels_list:
        print("Error: No data collected from test_data.")
        return

    # Concatenate numpy arrays
    try:
        images_all = np.concatenate(images_list, axis=0)
        labels_all = np.concatenate(labels_list, axis=0)
    except ValueError as e:
        print(f"Error concatenating data: {e}")
        print("Please ensure all batches have compatible shapes.")
        return
    
    # Get true class indices (assuming one-hot encoding)
    if labels_all.ndim > 1 and labels_all.shape[1] > 1:
        y_true_indices = np.argmax(labels_all, axis=1)

    # Handle integer labels directly
    elif labels_all.ndim == 1:
         y_true_indices = labels_all
    else:
        print("Error: Unexpected label format.")
        return

    # Predict on the dataset
    y_pred_probs = model.predict(test_data, verbose=1)
    if y_pred_probs.shape[0] != y_true_indices.shape[0]:
         print(f"Warning: Number of predictions ({y_pred_probs.shape[0]}) does not match number of labels ({y_true_indices.shape[0]}). Check dataset integrity.")
         # Fallback to predicting on the gathered numpy array if counts mismatch
         if images_all.shape[0] == y_true_indices.shape[0]:
              print("Retrying prediction on gathered numpy array...")
              y_pred_probs = model.predict(images_all, batch_size=test_data.batch_size, verbose=1) # Use original batch size
         else:
              print("Error: Cannot align predictions with labels.")
              return

    y_pred_indices = np.argmax(y_pred_probs, axis=1)

    # Ensure number of predictions matches labels
    if y_pred_indices.shape[0] != y_true_indices.shape[0]:
         print("Error: Mismatch between number of predictions and true labels after prediction.")
         return

    # --- 2. Identify Correct/Incorrect Predictions ---
    correct_indices = np.where(y_pred_indices == y_true_indices)[0]
    incorrect_indices = np.where(y_pred_indices != y_true_indices)[0]

    print(f"Found \033[1m{len(correct_indices)} correct\033[0m and \033[1m{len(incorrect_indices)} incorrect\033[0m predictions.\n")

    # --- Print Top 5 Most/Least Accurate Classes ---
    # Count correct predictions per class
    correct_per_class = np.zeros(len(class_names), dtype=int)
    total_per_class = np.zeros(len(class_names), dtype=int)
    for true_idx, pred_idx in zip(y_true_indices, y_pred_indices):
        total_per_class[true_idx] += 1
        if true_idx == pred_idx:
            correct_per_class[true_idx] += 1
    # Avoid division by zero
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.errstate.html
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_per_class = np.where(total_per_class > 0, correct_per_class / total_per_class, 0)
    # Get top 5 most and least accurate classes
    top5_idx = np.argsort(-acc_per_class)[:5]
    bottom5_idx = np.argsort(acc_per_class)[:5]
    print("\033[1mTop 5 classes with most correct predictions:\033[0m")
    for idx in top5_idx:
        print(f"  \033[1m{class_names[idx]}:\033[0m {correct_per_class[idx]}/{total_per_class[idx]} ({acc_per_class[idx]*100:.1f}%)")
    print("\033[1mTop 5 classes with least correct predictions:\033[0m")
    for idx in bottom5_idx:
        print(f"  \033[1m{class_names[idx]}:\033[0m {correct_per_class[idx]}/{total_per_class[idx]} ({acc_per_class[idx]*100:.1f}%)")

    # Check if enough samples are available
    if len(correct_indices) < num_images or len(incorrect_indices) < num_images:
        print(f"Warning: Not enough correct ({len(correct_indices)}) or incorrect ({len(incorrect_indices)}) examples to sample {num_images}. Adjusting num_images.")
        num_images = min(len(correct_indices), len(incorrect_indices), num_images)
        if num_images == 0:
            print("Error: No examples to plot.")
            return

    # Randomly sample indices
    correct_sample_indices = np.random.choice(correct_indices, num_images, replace=False)
    incorrect_sample_indices = np.random.choice(incorrect_indices, num_images, replace=False)

    # --- 3. Plotting ---
    # Create figure: 4 rows (Correct, Incorrect, True Example, Pred Example) x num_images columns
    fig, ax = plt.subplots(4, num_images, figsize=(num_images * 4, 16))
    fig.suptitle('Model Predictions Analysis\n', fontsize=18, fontweight='bold')          

    # Plot Correct Predictions (Row 0)
    ax[0, 0].set_ylabel('Correct\nPredictions', fontsize=12, fontweight='bold')
    for i, idx in enumerate(correct_sample_indices):
        true_class_name = class_names[y_true_indices[idx]]
        pred_class_name = class_names[y_pred_indices[idx]] # Should be same as true
        img_display = images_all[idx]
        if img_display.max() <= 1.0:
            img_display = (img_display * 255)
        ax[0, i].imshow(img_display.astype(np.uint8))
        ax[0, i].set_title(f"Pred: {pred_class_name}\n(True: {true_class_name})", fontsize=10, color='green')
        ax[0, i].axis('off')

    # Plot Incorrect Predictions (Row 1), True Class Examples (Row 2), Pred Class Examples (Row 3)
    ax[1, 0].set_ylabel('Incorrect\nPredictions', fontsize=12, fontweight='bold')
    ax[2, 0].set_ylabel('Example of\nTrue Class', fontsize=12, fontweight='bold')
    ax[3, 0].set_ylabel('Example of\nPred Class', fontsize=12, fontweight='bold')
    train_dir_path = Path(train_dir)

    for i, idx in enumerate(incorrect_sample_indices):
        true_class_idx = y_true_indices[idx]
        pred_class_idx = y_pred_indices[idx]
        true_class_name = class_names[true_class_idx]
        pred_class_name = class_names[pred_class_idx]

        # Row 1: Plot the incorrectly predicted image
        img_display = images_all[idx]
        if img_display.max() <= 1.0:
            img_display = (img_display * 255)
        ax[1, i].imshow(img_display.astype(np.uint8))
        ax[1, i].set_title(f"Pred: {pred_class_name}\n(True: {true_class_name})", fontsize=10, color='red')
        ax[1, i].axis('off')

        # Row 2: Example of True Class
        true_class_dir = train_dir_path / true_class_name
        example_img = None
        if true_class_dir.is_dir():
            try:
                possible_images = list(true_class_dir.glob('*[.jpg][.jpeg][.png]'))
                if possible_images:
                    example_img_path = random.choice(possible_images)
                    img = keras_image.load_img(example_img_path, target_size=(images_all.shape[1], images_all.shape[2]))
                    example_img = keras_image.img_to_array(img)
            except Exception as e:
                print(f"Error loading example image from {true_class_dir}: {e}")
        if example_img is not None:
            ax[2, i].imshow(example_img.astype(np.uint8))
            ax[2, i].set_title(f"Example of True:\n{true_class_name}", fontsize=10, color='blue')
        else:
            ax[2, i].text(0.5, 0.5, 'No Example Found', horizontalalignment='center', verticalalignment='center')
            ax[2, i].set_title(f"Example of True:\n{true_class_name}", fontsize=10, color='gray')
        ax[2, i].axis('off')

        # Row 3: Example of Predicted Class
        pred_class_dir = train_dir_path / pred_class_name
        pred_example_img = None
        if pred_class_dir.is_dir():
            try:
                possible_pred_images = list(pred_class_dir.glob('*[.jpg][.jpeg][.png]'))
                if possible_pred_images:
                    pred_example_img_path = random.choice(possible_pred_images)
                    pred_img = keras_image.load_img(pred_example_img_path, target_size=(images_all.shape[1], images_all.shape[2]))
                    pred_example_img = keras_image.img_to_array(pred_img)
            except Exception as e:
                print(f"Error loading example image from {pred_class_dir}: {e}")
        if pred_example_img is not None:
            ax[3, i].imshow(pred_example_img.astype(np.uint8))
            ax[3, i].set_title(f"Example of Pred:\n{pred_class_name}", fontsize=10, color='orange')
        else:
            ax[3, i].text(0.5, 0.5, 'No Example Found', horizontalalignment='center', verticalalignment='center')
            ax[3, i].set_title(f"Example of Pred:\n{pred_class_name}", fontsize=10, color='gray')
        ax[3, i].axis('off')

    plt.tight_layout()
    if file_path:
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"\nPredictions plot saved to {file_path}\n")
    plt.show()

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