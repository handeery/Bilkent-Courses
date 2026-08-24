import numpy as np
import matplotlib.pyplot as plt

# Convolution function
def convolve_2d(image, kernel):
    # Image and kernel dimensions
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Output image after convolution
    output = np.zeros_like(image)
    
    # Calculate padding for each side
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    # Pad the image with zeros on all sides
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    
    # Perform the convolution
    for i in range(image_height):
        for j in range(image_width):
            # Extract the region of interest
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            # Apply the filter (element-wise multiplication and summing)
            output[i, j] = np.sum(region * kernel)
    
    return output

# Sobel and Prewitt kernels for edge detection
sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

# Load and prepare the grayscale image
image_path = "/Users/handeeryilmaz/Downloads/HW1/images/convolution_spatial_domain.jpg"
image = plt.imread(image_path)
if image.ndim == 3:  # Convert to grayscale if the image is RGB
    image = np.mean(image[:, :, :3], axis=2)

# Apply Sobel filters
sobel_x_result = convolve_2d(image, sobel_x)
sobel_y_result = convolve_2d(image, sobel_y)
sobel_result = np.hypot(sobel_x_result, sobel_y_result)  # Combined magnitude of Sobel

# Apply Prewitt filters
prewitt_x_result = convolve_2d(image, prewitt_x)
prewitt_y_result = convolve_2d(image, prewitt_y)
prewitt_result = np.hypot(prewitt_x_result, prewitt_y_result)  # Combined magnitude of Prewitt

# Display the results
fig, ax = plt.subplots(2, 3, figsize=(15, 10))

ax[0, 0].imshow(image, cmap="gray")
ax[0, 0].set_title("Original Image")

ax[0, 1].imshow(sobel_result, cmap="gray")
ax[0, 1].set_title("Sobel Edge Detection")

ax[0, 2].imshow(prewitt_result, cmap="gray")
ax[0, 2].set_title("Prewitt Edge Detection")

ax[1, 0].imshow(sobel_x_result, cmap="gray")
ax[1, 0].set_title("Sobel - Horizontal Edges")

ax[1, 1].imshow(sobel_y_result, cmap="gray")
ax[1, 1].set_title("Sobel - Vertical Edges")

ax[1, 2].imshow(prewitt_x_result, cmap="gray")
ax[1, 2].set_title("Prewitt - Horizontal Edges")

plt.tight_layout()
plt.show()