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
                        left_corrected_path = os.path.join(zed_path, "left_corrected")

                        # Remove the original left folder if it exists
                        if os.path.isdir(left_path):
                            shutil.rmtree(left_path)
                        
                        # Rename left_corrected to left if left_corrected exists
                        if os.path.isdir(left_corrected_path):
                            os.rename(left_corrected_path, left_path)

print("Folders have been renamed successfully.")
