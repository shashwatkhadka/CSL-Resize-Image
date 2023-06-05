
"""
#how linspace works?
import numpy as np

x=np.linspace(0,10,6);

print(x)

"""


"""
#how each array element stores rgb values
import numpy as np
from PIL import Image

input_image = Image.open("download.jpg")
image_array = np.array(input_image)

print(image_array.dtype)#uint8=unsigned integer
print(image_array[0,0])

#to check:
# https://www.rapidtables.com/convert/color/rgb-to-hex.html

"""
