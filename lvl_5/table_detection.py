import os
import cv2 as cv

main_path = os.path.split(os.getcwd())[0]
file_path = os.path.join(main_path, "7.png")

import matplotlib.pyplot as plt
import keras_ocr


pipeline = keras_ocr.pipeline.Pipeline()


images = keras_ocr.tools.read(file_path)

prediction = pipeline.recognize([images])[0]

boxes = [value[1] for value in prediction]

canvas = keras_ocr.tools.drawBoxes(image=images, boxes=boxes, color=(255, 0, 0), thickness=1)
plt.imshow(canvas)
plt.show()