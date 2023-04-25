import os
import shutil
import random
import yaml

def create_train_test_val_folders(source_dir, dest_dir, nc, names, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    
    """
        Splits the images and annotations in a source directory into train, validation and test sets, and saves them in a
        destination directory. Also creates a YAML file in the destination directory containing information about the data
        split and the number and names of the classes.

    Parameters:
        source_dir (str): Path to the directory containing the images and annotations.
        dest_dir (str): Path to the destination directory where the train, validation and test sets will be saved.
        nc (int): Number of classes in the dataset.
        names (list): List of names of the classes in the dataset.
        train_ratio (float): Proportion of data to include in the training set (default: 0.7).
        val_ratio (float): Proportion of data to include in the validation set (default: 0.15).
        test_ratio (float): Proportion of data to include in the test set (default: 0.15).

    Arguments:
        None
    """

    
    # Create train/val/test directories
    os.makedirs(os.path.join(dest_dir, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "train", "annotations"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "valid", "images"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "valid", "annotations"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "test", "images"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "test", "annotations"), exist_ok=True)

    # Get a list of all image and annotation files
    img_files = [f for f in os.listdir(os.path.join(source_dir, "images")) if f.endswith(".jpg")]
    ann_files = [f for f in os.listdir(os.path.join(source_dir, "annotations")) if f.endswith(".txt")]

    # Randomly shuffle the list
    random.shuffle(img_files)

    # Split the data into train/val/test sets
    num_imgs = len(img_files)
    train_end_idx = int(num_imgs * train_ratio)
    val_end_idx = int(num_imgs * (train_ratio + val_ratio))

    train_imgs = img_files[:train_end_idx]
    val_imgs = img_files[train_end_idx:val_end_idx]
    test_imgs = img_files[val_end_idx:]

    # Move the images and annotations to the appropriate directories
    for img in train_imgs:
        shutil.copy(os.path.join(source_dir, "images", img), os.path.join(dest_dir, "train", "images"))
        shutil.copy(os.path.join(source_dir, "annotations", img.replace(".jpg", ".txt")), os.path.join(dest_dir, "train", "annotations"))

    for img in val_imgs:
        shutil.copy(os.path.join(source_dir, "images", img), os.path.join(dest_dir, "valid", "images"))
        shutil.copy(os.path.join(source_dir, "annotations", img.replace(".jpg", ".txt")), os.path.join(dest_dir, "valid", "annotations"))

    for img in test_imgs:
        shutil.copy(os.path.join(source_dir, "images", img), os.path.join(dest_dir, "test", "images"))
        shutil.copy(os.path.join(source_dir, "annotations", img.replace(".jpg", ".txt")), os.path.join(dest_dir, "test", "annotations"))

    # Write the yaml file
    data = {"train": "train/images", "val": "valid/images", "test": "test/images", "nc": nc, "names": names}
    with open(os.path.join(dest_dir, "data.yaml"), "w") as f:
        yaml.dump(data, f)
