import os
import glob
import shutil

# unpack all the image files from the folders
base_folders = [os.path.join("D:/intezet/gabor/data/images/20250716", folder) for folder in os.listdir("D:/intezet/gabor/data/images/20250716")]

# inside the base folders there are subfolders with with subfolders
# the second subfolders contain the images
# move the images from the second subfolders to the base folder
for base_folder in base_folders:
    # get all the subfolders in the base folder
    subfolders = os.listdir(base_folder)
    subfolders = [os.path.join(base_folder, subfolder) for subfolder in subfolders if os.path.isdir(os.path.join(base_folder, subfolder))]
        
    for subfolder in subfolders:
        # get all the image files in the subusubfolder
        image_files = glob.glob(os.path.join(subfolder, "*.*"))
        for image_file in image_files:
            # move the image file to the base folder
            shutil.move(image_file, base_folder)
        # remove the empty subusubfolder
        os.rmdir(subfolder)