import os
import glob
import shutil
from pathlib import Path

"""
This script unpacks all the image files from the folders
inside the base folders there are subfolders with with subfolders
the second subfolders contain the images
move the images from the second subfolders to the base folder.

This is the structure I got the raw images in.
"""


new_folder = "20250716" # change this to the folder you want to unpack
root = Path(__file__).resolve().parent.parent
base_folders = [root / "data" / "images" / new_folder / folder for folder in os.listdir(root / "data" / "images" / new_folder)]

for base_folder in base_folders:
    subfolders = os.listdir(base_folder)
    subfolders = [os.path.join(base_folder, subfolder) for subfolder in subfolders if os.path.isdir(os.path.join(base_folder, subfolder))]
        
    for subfolder in subfolders:
        image_files = glob.glob(os.path.join(subfolder, "*.*"))
        for image_file in image_files:
            # move the image file to the base folder
            shutil.move(image_file, base_folder)
        # remove the empty subfolder
        os.rmdir(subfolder)