import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Function to convert a JPEG image to a NumPy array and save as .npy
def convert_jpg_to_npy(jpg_path, npy_path):
    # Open the image using PIL
    image = Image.open(jpg_path)
    # Convert to grayscale (optional, depending on your needs)
    image = image.convert("L")  # Convert to grayscale
    # Convert to NumPy array
    image_array = np.array(image)
    # Save the array as .npy
    np.save(npy_path, image_array)

# Convert JPEG files to NumPy arrays and save as .npy
convert_jpg_to_npy("/Users/handeeryilmaz/Downloads/HW1/images/hist1.jpg", "/Users/handeeryilmaz/Downloads/HW1/images/hist1.npy")
convert_jpg_to_npy("/Users/handeeryilmaz/Downloads/HW1/images/hist2.jpg", "/Users/handeeryilmaz/Downloads/HW1/images/hist2.npy")

# Load the .npy files
image_matrix1 = np.load("/Users/handeeryilmaz/Downloads/HW1/images/hist1.npy")
image_matrix2 = np.load("/Users/handeeryilmaz/Downloads/HW1/images/hist2.npy")

# Define the histogram function
def histogram(image_matrix):
    hist = np.zeros(256, dtype=int)
    for pixel in image_matrix.flatten():
        hist[pixel] += 1
    return hist

# Generate histograms for both images
hist1 = histogram(image_matrix1)
hist2 = histogram(image_matrix2)

# Plot the histograms
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(hist1, color='black')
plt.title("Histogram of Image 1")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.plot(hist2, color='black')
plt.title("Histogram of Image 2")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.show()