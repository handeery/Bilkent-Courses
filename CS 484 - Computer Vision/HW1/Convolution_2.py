import numpy as np
import matplotlib.pyplot as plt

# Load the grayscale images
image_path = "/Users/handeeryilmaz/Downloads/HW1/images/convolution_freq_domain.jpg"

image = plt.imread(image_path)
if image.ndim == 3:  # Convert to grayscale if the image is RGB
    image = np.mean(image[:, :, :3], axis=2)

# Step 1: Transform the image to the frequency domain using FFT
freq_domain = np.fft.fft2(image)
freq_domain_shifted = np.fft.fftshift(freq_domain)  # Shift the zero frequency to the center

# Step 2: Design a Gaussian low-pass filter
def gaussian_low_pass_filter(shape, cutoff):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - center_col)**2 + (y - center_row)**2)
    filter_mask = np.exp(-(distance**2) / (2 * (cutoff**2)))
    return filter_mask

cutoff_frequency = 30  # Adjust this cutoff value for more or less blurring
gaussian_filter = gaussian_low_pass_filter(image.shape, cutoff_frequency)

# Step 3: Apply the Gaussian filter in the frequency domain
filtered_freq_domain = freq_domain_shifted * gaussian_filter

# Step 4: Transform the result back to the spatial domain
filtered_freq_domain_shifted = np.fft.ifftshift(filtered_freq_domain)  # Inverse shift
filtered_image = np.fft.ifft2(filtered_freq_domain_shifted)
filtered_image = np.abs(filtered_image)  # Take the magnitude for the real part of the result

# Display the original and filtered images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(image, cmap="gray")
ax[0].set_title("Original Image")

ax[1].imshow(filtered_image, cmap="gray")
ax[1].set_title("Filtered Image (Low-pass Gaussian)")
