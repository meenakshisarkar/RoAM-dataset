import os
import shutil

# Define the root directory
root_dir = "../../../catkin_ws/RoAM_dataset/processed/train"

# Traverse through the date and location folders
for date_folder in os.listdir(root_dir):
    date_path = os.path.join(root_dir, date_folder)
    if os.path.isdir(date_path):
        for location_folder in os.listdir(date_path):
            location_path = os.path.join(date_path, location_folder)
            if os.path.isdir(location_path):
                for zed_folder in os.listdir(location_path):
                    zed_path = os.path.join(location_path, zed_folder)
                    if os.path.isdir(zed_path):
                        left_path = os.path.join(zed_path, "left")
                        if os.path.isdir(left_path):
                            left_corrected_path = os.path.join(zed_path, "left_corrected")
                            os.makedirs(left_corrected_path, exist_ok=True)

                            # Get a list of image files in the left folder
                            image_files = [f for f in os.listdir(left_path) if os.path.isfile(os.path.join(left_path, f))]
                            
                            # Sort the image files to rename them in order
                            image_files.sort()
                            
                            # Rename the files to start from left00000001.png and save them in left_corrected
                            for idx, image_file in enumerate(image_files):
                                # Get the file extension
                                file_ext = os.path.splitext(image_file)[1]
                                
                                # Generate the new file name with sequential index starting from 0001
                                new_file_name = f"left{idx + 1:08d}{file_ext}"
                                
                                # Get the full path for the old and new file names
                                old_file_path = os.path.join(left_path, image_file)
                                new_file_path = os.path.join(left_corrected_path, new_file_name)
                                
                                # Copy the file to the new location with the new name
                                shutil.copy2(old_file_path, new_file_path)

print("Image files have been renamed and saved in the left_corrected folders successfully.")