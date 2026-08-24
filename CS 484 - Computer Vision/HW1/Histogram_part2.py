import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Function to convert a PNG image to a NumPy array and save as .npy
def convert_png_to_npy(png_path, npy_path):
    # Open the image using PIL
    image = Image.open(png_path)
    # Convert to grayscale
    image = image.convert("L")  # Convert to grayscale
    # Convert to NumPy array
    image_array = np.array(image)
    # Save the array as .npy
    np.save(npy_path, image_array)

# Convert PNG file to NumPy array and save as .npy
convert_png_to_npy("/Users/handeeryilmaz/Downloads/HW1/images/contrastive_strecth.png", "/Users/handeeryilmaz/Downloads/HW1/images/contrastive_stretch.npy")

# Load the grayscale image matrix from the .npy file
image_path = "/Users/handeeryilmaz/Downloads/HW1/images/contrastive_stretch.npy"
image_matrix = np.load(image_path, allow_pickle=False)

# Automatically determine minimum and maximum intensity values
a, b = image_matrix.min(), image_matrix.max()

# Define the contrast stretching function
def contrast_stretching(image_matrix, c, d):
    # Apply contrast stretching formula
    stretched_image = ((image_matrix - a) / (b - a) * (d - c) + c).clip(0, 255).astype(np.uint8)
    return stretched_image

# Apply contrast stretching for different target ranges
stretched_0_255 = contrast_stretching(image_matrix, 0, 255)
stretched_128_255 = contrast_stretching(image_matrix, 128, 255)
stretched_0_128 = contrast_stretching(image_matrix, 0, 128)

# Display intensity ranges to verify correctness
print("Original Image Intensity Range:", image_matrix.min(), image_matrix.max())
print("Stretched [0, 255] Intensity Range:", stretched_0_255.min(), stretched_0_255.max())
print("Stretched [128, 255] Intensity Range:", stretched_128_255.min(), stretched_128_255.max())
print("Stretched [0, 128] Intensity Range:", stretched_0_128.min(), stretched_0_128.max())

# Display the images
fig, ax = plt.subplots(2, 2, figsize=(12, 8))

ax[0, 0].imshow(image_matrix, cmap="gray")
ax[0, 0].set_title("Original Image (Using True a and b)")

ax[0, 1].imshow(stretched_0_255, cmap="gray")
ax[0, 1].set_title("Contrast Stretched [0, 255]")

ax[1, 0].imshow(stretched_128_255, cmap="gray")
ax[1, 0].set_title("Contrast Stretched [128, 255]")

ax[1, 1].imshow(stretched_0_128, cmap="gray")
ax[1, 1].set_title("Contrast Stretched [0, 128]")

plt.tight_layout()
plt.show()