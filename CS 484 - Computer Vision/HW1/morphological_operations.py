import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the uploaded image
image_path = "/Users/handeeryilmaz/Downloads/HW1/images/morphological_operations.png"
src_image = Image.open(image_path)

# Convert the image to grayscale
gray_image = np.array(src_image.convert("L"))

# Convert to binary image (0s and 1s)
binary_image = (gray_image > 128).astype(int)

# Define the structuring element: 11x11 square (increased size)
structuring_element = np.ones((11, 11), dtype=int)

# Define the dilation function
def dilation(image, structuring_element):
    padded_image = np.pad(image, pad_width=5, mode='constant', constant_values=0)
    dilated_image = np.zeros(image.shape, dtype=int)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + structuring_element.shape[0], j:j + structuring_element.shape[1]]
            dilated_image[i, j] = 1 if np.any(region * structuring_element) else 0

    return dilated_image

# Define the erosion function
def erosion(image, structuring_element):
    padded_image = np.pad(image, pad_width=5, mode='constant', constant_values=0)
    eroded_image = np.zeros(image.shape, dtype=int)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + structuring_element.shape[0], j:j + structuring_element.shape[1]]
            eroded_image[i, j] = 1 if np.all(region * structuring_element) else 0

    return eroded_image

# Step 1: Initial Opening (Erosion followed by Dilation) with increased iterations
eroded_image = erosion(binary_image, structuring_element)
opened_image = dilation(eroded_image, structuring_element)

# Step 2: Additional Closing (Dilation followed by Erosion) with increased iterations
dilated_image = dilation(opened_image, structuring_element)
closed_image = erosion(dilated_image, structuring_element)

# Step 3: Final Opening with more iterations
for _ in range(2):  # Repeat the opening process
    eroded_again = erosion(closed_image, structuring_element)  
    closed_image = dilation(eroded_again, structuring_element)

# Display the original and processed images
plt.figure(figsize=(16, 6))

plt.subplot(1, 4, 1)
plt.imshow(binary_image, cmap="gray")
plt.title("Original Binary Image")
plt.axis('off')  # Hide axis

plt.subplot(1, 4, 2)
plt.imshow(opened_image, cmap="gray")
plt.title("After Initial Opening")
plt.axis('off')  # Hide axis

plt.subplot(1, 4, 3)
plt.imshow(closed_image, cmap="gray")
plt.title("After Closing (Dilation + Erosion)")
plt.axis('off')  # Hide axis

plt.subplot(1, 4, 4)
plt.imshow(closed_image, cmap="gray")
plt.title("Final Image (Aggressive Opening)")
plt.axis('off')  # Hide axis

plt.show()