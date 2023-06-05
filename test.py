import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from PIL import Image


def resize_image(image, new_width, new_height):
    # Convert image to grayscale
    image = image.convert("L")

    # Get original image dimensions
    width, height = image.size

    # Create a grid of coordinates for the original image
    x = np.linspace(0, width, width)
    y = np.linspace(0, height, height)

    # Create a grid of coordinates for the resized image
    new_x = np.linspace(0, width, new_width)
    new_y = np.linspace(0, height, new_height)

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Perform cubic spline interpolation along the rows
    row_interpolated = CubicSpline(y, image_array)(new_y)

    # Perform cubic spline interpolation along the columns
    resized_image = CubicSpline(x, row_interpolated.T)(new_x).T

    # Convert the interpolated image back to PIL image
    resized_image = Image.fromarray(resized_image.astype(np.uint8))

    return resized_image


# Load the input image
input_image = Image.open("download.jpg")

# Specify the new dimensions for the resized image
new_width = 800
new_height = 600

# Resize the image using cubic spline interpolation
resized_image = resize_image(input_image, new_width, new_height)

# Display the original and resized images
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(input_image, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Resized Image")
plt.imshow(resized_image, cmap="gray")

plt.show()
