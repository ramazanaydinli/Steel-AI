


"""

Created by Ramazan AYDINLI for educational purposes
Aim is generating artificial x-section images that may be used for
training neural network models

"""
import numpy
import numpy as np
from PIL import Image, ImageDraw
import random
import cv2 as cv


# Set how many image you want
n = 500

# Set min and max image height
min_y = 1200
max_y = 1500

# Set min and max shape_ratio (Ratio of the x-section / image size)
min_shape_ratio = 0.4
max_shape_ratio = 0.7

# Set the corner size (blank area that cannot have any drawing or text)
corner_size = 0.1


def generate_image(image_x, image_y):
    """ Generates image
    Args:
        image_x : width of image
        image_y : height of image
    Returns:
        img: generated image
        """
    img = Image.new('RGBA', (image_x, image_y), (255, 255, 255))
    return img

def create_i_beam():
    x_1_coordinate = 0
    y_1_coordinate = 0
    x_2_coordinate = random.randint(8, 15)
    y_2_coordinate = 1
    h_length = random.randint(5, 7)
    coordinates = [x_1_coordinate, x_2_coordinate, y_1_coordinate, y_2_coordinate, h_length]
    return coordinates

def centered_drawing_coordinates(real_drawing_coordinates, img_x, img_y):
    real_drawing_coordinates_x1 = real_drawing_coordinates[0]
    real_drawing_coordinates_x2 = real_drawing_coordinates[1]
    real_drawing_coordinates_y1 = real_drawing_coordinates[2]
    real_drawing_coordinates_y2 = real_drawing_coordinates[3]
    real_drawing_coordinates_h = real_drawing_coordinates[4]
    halved_x = (img_x / 2) - ((real_drawing_coordinates_x2 - real_drawing_coordinates_x1)/4)
    halved_y = img_y / 2 - ((real_drawing_coordinates_y2 - real_drawing_coordinates_y1 + real_drawing_coordinates_h)/2)
    x1_exact_location = int(real_drawing_coordinates_x1/2 + halved_x)
    x2_exact_location = int(real_drawing_coordinates_x2/2 + halved_x)
    y1_exact_location = int(real_drawing_coordinates_y1/2 + halved_y)
    y2_exact_location = int(real_drawing_coordinates_y2/2 + halved_y)
    coordinates = [x1_exact_location, x2_exact_location, y1_exact_location, y2_exact_location, real_drawing_coordinates_h]
    return coordinates


def specify_points(final_coordinates):
    line_coordinates = []
    line_coordinates.append((final_coordinates[0], final_coordinates[2]))
    line_coordinates.append((final_coordinates[0], final_coordinates[3]))
    line_coordinates.append((final_coordinates[1], final_coordinates[2]))
    line_coordinates.append((final_coordinates[1], final_coordinates[3]))
    h_top_left = ((final_coordinates[0] + final_coordinates[1])/2) - ((final_coordinates[3] - final_coordinates[2])/2)
    h_top_right = (h_top_left + final_coordinates[3] - final_coordinates[2])
    top_to_bottom_y_point_difference = final_coordinates[4] + final_coordinates[3] - final_coordinates[2]
    line_coordinates.append((h_top_left, final_coordinates[3]))
    line_coordinates.append((h_top_left, final_coordinates[3]+final_coordinates[4]))
    line_coordinates.append((h_top_right, final_coordinates[3]))
    line_coordinates.append((h_top_right, final_coordinates[3]+final_coordinates[4]))
    line_coordinates.append((final_coordinates[0], final_coordinates[2]+top_to_bottom_y_point_difference))
    line_coordinates.append((final_coordinates[0], final_coordinates[3]+top_to_bottom_y_point_difference))
    line_coordinates.append((final_coordinates[1], final_coordinates[2]+top_to_bottom_y_point_difference))
    line_coordinates.append((final_coordinates[1], final_coordinates[3]+top_to_bottom_y_point_difference))
    

    return line_coordinates


for i in range(n):

    img_y = random.randint(min_y, max_y)
    img_x = int(img_y/(2**0.5))

    image = generate_image(img_x, img_y)

    # shape_ratio = random.random()*(max_shape_ratio - min_shape_ratio) + min_shape_ratio
    # skew_ratio_x = (0.5 - random.random()) * (1 - 0.1 - shape_ratio)
    # skew_ratio_y = (0.5 - random.random()) * (1 - 0.1 - shape_ratio)
    # coordinates=[]
    # coordinates = create_i_beam()
    # multiplication_ratio_of_coordinates = (shape_ratio * img_x) / np.max(coordinates)
    # real_drawing_coordinates = numpy.multiply(int(multiplication_ratio_of_coordinates), coordinates)
    # final_coordinates = []
    # final_coordinates = centered_drawing_coordinates(real_drawing_coordinates, img_x, img_y)
    # line_coordinates = specify_points(final_coordinates)
    # img = ImageDraw.Draw(image)
    # img.line((line_coordinates[0], line_coordinates[1]), fill='black', width=2)
    # img.line((line_coordinates[1], line_coordinates[3]), fill='black', width=2)
    # img.line((line_coordinates[2], line_coordinates[0]), fill='black', width=2)
    # img.line((line_coordinates[3], line_coordinates[2]), fill='black', width=2)
    # img.line((line_coordinates[4], line_coordinates[5]), fill='black', width=2)
    # img.line((line_coordinates[6], line_coordinates[7]), fill='black', width=2)
    # img.line((line_coordinates[8], line_coordinates[9]), fill='black', width=2)
    # img.line((line_coordinates[9], line_coordinates[11]), fill='black', width=2)
    # img.line((line_coordinates[10], line_coordinates[8]), fill='black', width=2)
    # img.line((line_coordinates[11], line_coordinates[10]), fill='black', width=2)
    # image.show()
    image_data = np.asarray(image)
    cv.imwrite('C:\\Users\\METE\Desktop\\LevelUp\\generated_images\\White\\{}.jpg'.format(i), image_data)





