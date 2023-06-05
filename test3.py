"""
from scipy.interpolate import CubicSpline

x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

cs = CubicSpline(x, y)

# Evaluate the interpolated values
x_interp = [1.5, 2.5, 3.5]
y_interp = cs(x_interp)

print("Interpolated values:", y_interp)
"""

"""
import numpy as np


def cubic_spline_interpolation(x, y, x_interp):
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y) / h

    # Step 4: Initialize arrays
    alpha = np.zeros(n)
    l = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)

    # Step 5: Solve the tridiagonal system of equations
    for i in range(1, n - 1):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (delta[i] - h[i - 1] * z[i - 1]) / l[i]

    # Step 6: Set the last values of the arrays
    l[n - 1] = 1
    z[n - 1] = 0

    # Step 7: Compute the coefficients of the cubic polynomials
    b = np.zeros(n - 1)
    c = np.zeros(n)
    d = np.zeros(n - 1)

    # Step 8: Backsubstitution
    for i in range(n - 2, -1, -1):
        c[i] = z[i] - mu[i] * c[i + 1]
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    # Step 9: Evaluate interpolated values
    y_interp = np.zeros(len(x_interp))
    for i, x_i in enumerate(x_interp):
        j = np.searchsorted(x, x_i) - 1
        dx = x_i - x[j]
        y_interp[i] = y[j] + b[j] * dx + c[j] * dx ** 2 + d[j] * dx ** 3

    return y_interp


# Example usage:
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 4, 9, 16, 25])
x_interp = np.array([1.5, 2.5, 3.5])

y_interp = cubic_spline_interpolation(x, y, x_interp)
print("Interpolated values:", y_interp)

def CubicSpline(x, y, x_interp):
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y) / h

    # Step 4: Initialize arrays
    alpha = np.zeros(n)
    l = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)

    # Step 5: Solve the tridiagonal system of equations
    for i in range(1, n - 1):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (delta[i] - h[i - 1] * z[i - 1]) / l[i]

    # Step 6: Set the last values of the arrays
    l[n - 1] = 1
    z[n - 1] = 0

    # Step 7: Compute the coefficients of the cubic polynomials
    b = np.zeros(n - 1)
    c = np.zeros(n)
    d = np.zeros(n - 1)

    # Step 8: Backsubstitution
    for i in range(n - 2, -1, -1):
        c[i] = z[i] - mu[i] * c[i + 1]
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    # Step 9: Evaluate interpolated values
    y_interp = np.zeros(len(x_interp))
    for i, x_i in enumerate(x_interp):
        j = np.searchsorted(x, x_i) - 1
        dx = x_i - x[j]
        y_interp[i] = y[j] + b[j] * dx + c[j] * dx ** 2 + d[j] * dx ** 3

    return y_interp


# Example usage:
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 4, 9, 16, 25])
x_interp = np.array([1.5, 2.5, 3.5])

y_interp = cubic_spline_interpolation(x, y, x_interp)
print("Interpolated values:", y_interp)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from PIL import Image


def resize_image(image, new_width, new_height):
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

    # Perform cubic spline interpolation along the rows for each channel
    row_interpolated = np.zeros((new_height, width, 3))
    for channel in range(3):
        row_interpolated[:, :, channel] = CubicSpline(y, image_array[:, :, channel])(new_y)

    # Perform cubic spline interpolation along the columns for each channel
    resized_image = np.zeros((new_height, new_width, 3))
    for channel in range(3):
        resized_image[:, :, channel] = CubicSpline(x, row_interpolated[:, :, channel].T)(new_x).T

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
plt.imshow(input_image)

plt.subplot(1, 2, 2)
plt.title("Resized Image")
plt.imshow(resized_image)

plt.show()
