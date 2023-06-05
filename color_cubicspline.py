
#a 400x500 image tells us (no of pixels in width)x(no of pixels in height)

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from PIL import Image
#importing the image module from PIL/Python Imaging Library


def resize_image(image, new_width, new_height):
    # Get original image dimensions
    width, height = image.size
    #image.size returns the width(x-axis) and height(y-axis) of the image

    # Create a grid of coordinates for the original image
    x = np.linspace(0, width, width)
    y = np.linspace(0, height, height)

    # Create a grid of coordinates for the resized image
    #linspace(start,stop,number of elements) generates an array of evenly
    #spaced numbers over start,stop
    new_x = np.linspace(0, width, new_width)
    new_y = np.linspace(0, height, new_height)

    # new_y and new_x are the new heights and weight of the new resized image (array)

    # Convert the image to a numpy array
    image_array = np.array(image)
    #will have dimensions corresponding to the width and height of the image,
    # and each element of the array will represent the pixel values.
    #each element [i,j] represents the pixel values at the corresponding location in the image.

    #since the image is in RGB, each element of the array will typically
    #hold three values representing the intensity of red,green and blue channels

    # Perform cubic spline interpolation along the rows for each channel
    # np.zeros returns a new array filled with zeros
    row_interpolated = np.zeros((new_height, width, 3))
    # range(3) is the same as range(0,3) : 0,1,2,
    # where channel take the values 0,1,2 over 3 iterations
    for channel in range(3):
        row_interpolated[:, :, channel] = CubicSpline(y, image_array[:, :, channel])(new_y)
    #when accessing elements of a numpy array colon,(:) is used to select all
    #elements along the specific axis

    #output of the CubicSpline is a callable function
    #new_y calls the interpolated function with new_y as the argument


    # Perform cubic spline interpolation along the columns for each channel
    resized_image = np.zeros((new_height, new_width, 3))
    for channel in range(3):
        resized_image[:, :, channel] = CubicSpline(x, row_interpolated[:, :, channel].T)(new_x).T

    # Convert the interpolated image back to PIL image
    resized_image = Image.fromarray(resized_image.astype(np.uint8))
    #Image.fromarray() creates a PIL image
    #converting numpy array stored in resized_image back to and image,
    # astype() converts pixel values to fall within 0-255 range,(rgb value range (0-255) and uint8 value range(0-255))
    #we use this just to make sure values outside this range are converted back to the nearest value

    return resized_image


# Load the input image
input_image = Image.open("download.jpg")

# Specify the new dimensions for the resized image
new_width = 400
new_height = 400

# Resize the image using cubic spline interpolation
resized_image = resize_image(input_image, new_width, new_height)

# Display the original and resized images

#subplot() function to draw multiple plots in a single winodow
plt.subplot(1, 2, 1) #represents 1 row,2 column and 1 represents the index(1st plot)
plt.title("Original Image")
plt.imshow(input_image)

plt.subplot(1, 2, 2) #represents 1 row,2 column and 2 represents the index(2nd plot)
plt.title("Resized Image")
plt.imshow(resized_image)

plt.show()
