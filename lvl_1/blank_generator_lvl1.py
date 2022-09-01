
from PIL import Image
import os
import cv2 as cv
import numpy as np
main_directory = os.getcwd()
directory_to_save = os.path.join(main_directory, 'generated_images', 'white')
if not os.path.exists(directory_to_save):
    os.makedirs(directory_to_save)

for i in range(5000):
    filename = 'white_{}.jpg'.format(i)
    image = Image.new('RGBA', (300, 300), (255, 255, 255))
    image_data = np.asarray(image)
    cv.imwrite(os.path.join(directory_to_save, filename), image_data)



