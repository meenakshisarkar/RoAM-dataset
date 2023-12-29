import os
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8x.pt")

def count_people_in_images(images_path, model):
    # Get a list of image files in the directory
    image_files = sorted([f for f in os.listdir(images_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    people_per_frame = []
    # Process each image and count the number of people
    for filename in image_files:
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(images_path, filename)
            results = model(image_path)
            count = 0
            if results[0].boxes == None:
                people_per_frame.append(0)
                continue

            for res in results[0].boxes:
                if res.cls == 0:
                    count += 1
            people_per_frame.append(count)

    return people_per_frame

def plot_histogram(data):
    # Calculate the running average with a window size of 5
    plt.hist(data, bins=10, color='blue', edgecolor='black', alpha=0.7, label='Original Data')
    plt.title('Histogram of People Count per Frame with Running Average')
    plt.xlabel('Number of People')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def plot_people_count_over_frames(data):
    # Plot the running average of the number of people per frame
    # Plot the number of people detected over frames
    plt.figure(figsize=(12, 6))
    plt.plot(data, color='red')
    plt.title('Number of People Detected Over Frames')
    plt.xlabel('Frame Index')
    plt.ylabel('Number of People')
    plt.show()

def main():
    # Path to the directory containing images
    images_path = "path/to/folder/containing/images"

    # Count the number of people in each image
    people_counts = count_people_in_images(images_path, model)

    # Plot the results
    plot_histogram(people_counts)
    plot_people_count_over_frames(people_counts)

    # Find the index of frames where 4 people were detected

if __name__ == "__main__":
    main()
