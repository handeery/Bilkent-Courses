import numpy as np
import matplotlib.pyplot as plt

# Load the grayscale images from the .png and .jpg files
image_path1 = "/Users/handeeryilmaz/Downloads/HW1/images/otsu_1.png"
image_path2 = "/Users/handeeryilmaz/Downloads/HW1/images/otsu_2.jpg"

# Load images using plt.imread and convert them to grayscale if needed
# For JPG files, imread loads directly as grayscale; for PNG, we'll ensure it's 2D by averaging if it has RGB channels
image1 = plt.imread(image_path1)
if image1.ndim == 3:  # Convert to grayscale if it is RGB
    image1 = np.mean(image1[:, :, :3], axis=2).astype(np.uint8)

image2 = plt.imread(image_path2)
if image2.ndim == 3:  # Convert to grayscale if it is RGB
    image2 = np.mean(image2[:, :, :3], axis=2).astype(np.uint8)

# Define Otsu's thresholding function
def otsu_threshold(image):
    # Calculate histogram
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    total_pixels = image.size
    
    # Initialize variables
    sum_total = np.dot(np.arange(256), hist)  # Weighted sum of pixel intensities
    sum_background = 0
    weight_background = 0
    max_variance = 0
    threshold = 0
    
    # Loop through all possible threshold values
    for t in range(256):
        weight_background += hist[t]
        if weight_background == 0:
            continue
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break

        sum_background += t * hist[t]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        # Calculate between-class variance
        variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
        # Check if new maximum variance is found
        if variance_between > max_variance:
            max_variance = variance_between
            threshold = t

    # Apply threshold to create a binary image
    binary_image = (image >= threshold).astype(np.uint8) * 255
    return binary_image, threshold

# Apply Otsu's thresholding
binary_image1, threshold1 = otsu_threshold(image1)
binary_image2, threshold2 = otsu_threshold(image2)

print(f"Optimal threshold for image 1: {threshold1}")
print(f"Optimal threshold for image 2: {threshold2}")

# Display original and binary images
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(image1, cmap="gray")
axes[0, 0].set_title("Original Image 1")

axes[0, 1].imshow(binary_image1, cmap="gray")
axes[0, 1].set_title(f"Otsu's Thresholded Image 1 (Threshold: {threshold1})")

axes[1, 0].imshow(image2, cmap="gray")
axes[1, 0].set_title("Original Image 2")

axes[1, 1].imshow(binary_image2, cmap="gray")
axes[1, 1].set_title(f"Otsu's Thresholded Image 2 (Threshold: {threshold2})")

plt.tight_layout()
plt.show()