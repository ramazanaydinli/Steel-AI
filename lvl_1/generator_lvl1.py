"""

Created by Ramazan AYDINLI for educational purposes
Aim is generating artificial x-section images that may be used for
training neural network models

"""


import numpy as np
from PIL import Image, ImageDraw
import random
import cv2 as cv
import os

number_of_images = 5000

min_image_height = 300
max_image_height = 300
a4_size = False
image_width = 300

min_x_section_ratio_to_image_area = 0.7
max_x_section_ratio_to_image_area = 0.9
blank_corner_ratio_to_image_area = 0.05

tabular_x_section_probability = 0.25
built_up_consisting_2_shapes_probability = 0.25
built_up_consisting_3_shapes_probability = 0.25
built_up_consisting_4_shapes_probability = 0.25

tabular_shape_list = ['Plate', 'IPN', 'UPN', 'Legged']

current_working_directory = os.getcwd()
main_directory_to_save_images = os.path.join(current_working_directory, 'generated_images')
directory_to_save = os.path.join(main_directory_to_save_images, 'section_images')
if not os.path.exists(directory_to_save):
    os.makedirs(directory_to_save)

def generate_image(width, height):
    """ Generates image
    :arg
        width : width of image
        height : height of image
    :return
        generated_image : generated image
        """
    generated_image = Image.new('RGBA', (width, height), (255, 255, 255))
    return generated_image


def choose_shape():
    """ Randomly choosing general shape
    :return
        choice : chosen_x_section
    """
    general_shape_list = ['tabular_x_section', 'built_up_consisting_2_shapes',
                          'built_up_consisting_3_shapes', 'built_up_consisting_4_shapes',
                          ]
    choice = random.choices(general_shape_list, weights=(
        tabular_x_section_probability,
        built_up_consisting_2_shapes_probability,
        built_up_consisting_3_shapes_probability,
        built_up_consisting_4_shapes_probability
    ), k=1)
    return choice


def rotate(p, origin, degrees):
    angle = np.deg2rad(degrees)
    r = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((r @ (p.T-o.T) + o.T).T)


def create_geometry(selection_list):
    """
    Creates simple unscaled geometry with selected elements
    :param selection_list: list of selected tabular x-sections
    :return: x_coordinates: x coordinates of unscaled section
    y_coordinates: y coordinates of unscaled section
    """
    number_of_selected_elements = len(selection_list)
    element_of_origin = selection_list[0]
    t = 2 * random.randint(1, 2)
    l = 2 * random.randint(14, 24)
    x_coordinates = []
    y_coordinates = []
    is_horizontal = bool(random.getrandbits(1))
    if number_of_selected_elements == 1:
        section_type = selection_list[0]
    else:
        section_type = 'Built-up'
    if element_of_origin == 'Plate':
        x_coordinates.append((-l/2, l/2))
        y_coordinates.append((-t/2, t/2))
        if number_of_selected_elements == 2:
            x_coordinates = []
            y_coordinates = []
            if selection_list[1] == 'IPN':
                random_generator = random.randint(0, 4)
                if random_generator == 0:
                    x_coordinates.append((-l / 2, l / 2))
                    x_coordinates.append((-l / 2, l / 2))
                    x_coordinates.append((-t / 2, t / 2))
                    x_coordinates.append((-l / 2, l / 2))
                    x_coordinates.append((-l / 2, l / 2))
                    y_coordinates.append((l / 2 + 2 * t, l / 2 + t))
                    y_coordinates.append((l / 2 + t, l / 2))
                    y_coordinates.append((-l / 2, l / 2))
                    y_coordinates.append((-l / 2 - t, -l / 2))
                    y_coordinates.append((-l / 2 - 2 * t, -l / 2 - t))
                elif random_generator == 1:
                    x_coordinates.append((-l / 4, l / 4))
                    x_coordinates.append((-l / 2, l / 2))
                    x_coordinates.append((-t / 2, t / 2))
                    x_coordinates.append((-l / 2, l / 2))
                    x_coordinates.append((-l / 4, l / 4))
                    y_coordinates.append((l / 2 + 2 * t, l / 2 + t))
                    y_coordinates.append((l / 2 + t, l / 2))
                    y_coordinates.append((-l / 2, l / 2))
                    y_coordinates.append((-l / 2 - t, -l / 2))
                    y_coordinates.append((-l / 2 - 2 * t, -l / 2 - t))
                elif random_generator == 2:
                    x_coordinates.append((-5 * l / 2, 5 * l / 2))
                    x_coordinates.append((-3 * l / 2, -l / 2))
                    x_coordinates.append((l / 2, 3 * l / 2))
                    x_coordinates.append((-l - t / 2, -l + t / 2))
                    x_coordinates.append((l - t / 2, l + t / 2))
                    x_coordinates.append((-3 * l / 2, -l / 2))
                    x_coordinates.append((l / 2, 3 * l / 2))
                    x_coordinates.append((-5 * l / 2, 5 * l / 2))
                    y_coordinates.append((l / 2 + 2 * t, l / 2 + t))
                    y_coordinates.append((l / 2 + t, l / 2))
                    y_coordinates.append((l / 2 + t, l / 2))
                    y_coordinates.append((-l / 2, l / 2))
                    y_coordinates.append((-l / 2, l / 2))
                    y_coordinates.append((-l / 2, -l / 2 - t))
                    y_coordinates.append((-l / 2, -l / 2 - t))
                    y_coordinates.append((-l / 2 - t, -l / 2 - 2 * t))
                    x_coordinates, y_coordinates = np.divide(x_coordinates,2), np.divide(y_coordinates,2)
                elif random_generator == 3:
                    x_coordinates.append((-3 * l / 2, 3 * l / 2))
                    x_coordinates.append((-l / 2, l / 2))
                    x_coordinates.append((-t / 2, t / 2))
                    x_coordinates.append((-l / 2, l / 2))
                    x_coordinates.append((-3 * l / 2, 3 * l / 2))
                    y_coordinates.append((l / 2 + 2 * t, l / 2 + t))
                    y_coordinates.append((l / 2 + t, l / 2))
                    y_coordinates.append((-l / 2, l / 2))
                    y_coordinates.append((-l / 2, -l / 2 - t))
                    y_coordinates.append((-l / 2 - 2 * t, -l / 2 - t))
                elif random_generator == 4:
                    x_coordinates.append((-l / 2 - t, -l / 2))
                    x_coordinates.append((-l / 2, l / 2))
                    x_coordinates.append((l / 2, l / 2 + t))
                    x_coordinates.append((-t / 2, t / 2))
                    x_coordinates.append((-l / 2, l / 2))
                    y_coordinates.append((-l / 2 - t, l / 2 + t))
                    y_coordinates.append((l / 2, l / 2 + t))
                    y_coordinates.append((-l / 2 - t, l / 2 + t))
                    y_coordinates.append((-l / 2, l / 2))
                    y_coordinates.append((-l / 2 - t, -l / 2))
            elif selection_list[1] == 'UPN':
                random_generator = random.randint(0, 2)
                if random_generator == 0:
                    x_coordinates.append((-l / 2 - t, l / 2 + t))
                    x_coordinates.append((-l / 2 - t, -l / 2))
                    x_coordinates.append((-l / 2, 0))
                    x_coordinates.append((0, l / 2))
                    x_coordinates.append((l / 2, l / 2 + t))
                    x_coordinates.append((-l / 2, 0))
                    x_coordinates.append((0, l / 2))
                    x_coordinates.append((-l / 2 - t, l / 2 + t))
                    y_coordinates.append((l / 2, l / 2 + t))
                    y_coordinates.append((-l / 2, l / 2))
                    y_coordinates.append((l / 2 - t, l / 2))
                    y_coordinates.append((l / 2 - t, l / 2))
                    y_coordinates.append((-l / 2, l / 2))
                    y_coordinates.append((-l / 2 - t, -l / 2))
                    y_coordinates.append((-l / 2 - t, -l / 2))
                    y_coordinates.append((-l / 2 - t, -l / 2))
                if random_generator == 1:
                    x_coordinates.append((-l - t, l + t))
                    x_coordinates.append((-l - t, -l))
                    x_coordinates.append((-l, -l / 2))
                    x_coordinates.append((l / 2, l))
                    x_coordinates.append((l, l + t))
                    x_coordinates.append((-l, -l / 2))
                    x_coordinates.append((l / 2, l))
                    x_coordinates.append((-l - t, l + t))
                    y_coordinates.append((l / 2, l / 2 + t))
                    y_coordinates.append((-l / 2, l / 2))
                    y_coordinates.append((l / 2 - t, l / 2))
                    y_coordinates.append((l / 2 - t, l / 2))
                    y_coordinates.append((-l / 2, l / 2))
                    y_coordinates.append((-l / 2, -l / 2 + t))
                    y_coordinates.append((-l / 2, -l / 2 + t))
                    y_coordinates.append((-l / 2 - t, -l / 2))
                if random_generator == 2:
                    x_coordinates.append((-l, l))
                    x_coordinates.append((-l, -l / 2 - t))
                    x_coordinates.append((-l / 2 - t, -l / 2))
                    x_coordinates.append((l / 2, l / 2 + t))
                    x_coordinates.append((l / 2 + t, l))
                    x_coordinates.append((-l, -l / 2 - t))
                    x_coordinates.append((l / 2 + t, l))
                    x_coordinates.append((-l, l))
                    y_coordinates.append((l / 2, l / 2 + t))
                    y_coordinates.append((l / 2 - t, l / 2))
                    y_coordinates.append((-l / 2, l / 2))
                    y_coordinates.append((-l / 2, l / 2))
                    y_coordinates.append((l / 2 - t, l / 2))
                    y_coordinates.append((-l / 2, -l / 2 + t))
                    y_coordinates.append((-l / 2, -l / 2 + t))
                    y_coordinates.append((-l / 2 - t, -l / 2))
            elif selection_list[1] == 'Legged':
                random_generator = random.randint(0, 1)
                if random_generator == 0:
                    x_coordinates.append((-3 * t / 2, -t / 2))
                    x_coordinates.append((-t / 2, t / 2))
                    x_coordinates.append((t / 2, 3 * t / 2))
                    x_coordinates.append((-l - t / 2, -3 * t / 2))
                    x_coordinates.append((3 * t / 2, l + t / 2))
                    y_coordinates.append((-l / 2, l / 2))
                    y_coordinates.append((-l / 2, l / 2))
                    y_coordinates.append((-l / 2, l / 2))
                    y_coordinates.append((-l / 2, -l / 2 + t))
                    y_coordinates.append((-l / 2, -l / 2 + t))
                if random_generator == 1:
                    x_coordinates.append((-3 * l / 2 - t, -3 * l / 2))
                    x_coordinates.append((-3 * l / 2, 3 * l / 2))
                    x_coordinates.append((3 * l / 2, 3 * l / 2 + t))
                    x_coordinates.append((-3 * l / 2, -3 * l / 2 + t))
                    x_coordinates.append((-3 * l / 2 + t, -l / 2))
                    x_coordinates.append((l / 2, 3 * l / 2 - t))
                    x_coordinates.append((3 * l / 2 - t, 3 * l / 2))
                    x_coordinates.append((-3 * l / 2, -3 * l / 2 + t))
                    x_coordinates.append((-3 * l / 2 + t, -l / 2))
                    x_coordinates.append((l / 2, 3 * l / 2 - t))
                    x_coordinates.append((3 * l / 2 - t, 3 * l / 2))
                    x_coordinates.append((-3 * l / 2, 3 * l / 2))
                    y_coordinates.append((-3 * l / 2 - t, 3 * l / 2 + t))
                    y_coordinates.append((3 * l / 2 + t, 3 * l / 2))
                    y_coordinates.append((-3 * l / 2 - t, 3 * l / 2 + t))
                    y_coordinates.append((3 * l / 2, l / 2))
                    y_coordinates.append((3 * l / 2, 3 * l / 2 - t))
                    y_coordinates.append((3 * l / 2, 3 * l / 2 - t))
                    y_coordinates.append((3 * l / 2, l / 2))
                    y_coordinates.append((-3 * l / 2, -l / 2))
                    y_coordinates.append((-3 * l / 2, -3 * l / 2 + t))
                    y_coordinates.append((-3 * l / 2, -3 * l / 2 + t))
                    y_coordinates.append((-3 * l / 2, -l / 2))
                    y_coordinates.append((-3 * l / 2 - t, -3 * l / 2))
                    x_coordinates, y_coordinates = np.divide(x_coordinates, 2), np.divide(y_coordinates, 2)
            else:
                x_coordinates.append((-l / 2, l / 2))
                y_coordinates.append((-t / 2, t / 2))
    elif element_of_origin == 'IPN':
        x_coordinates.append((-l / 2, l / 2))
        x_coordinates.append((-t / 2, t / 2))
        x_coordinates.append((-l / 2, l / 2))
        y_coordinates.append((l / 2, l / 2 - t))
        y_coordinates.append((-l / 2 + t, l / 2 - t))
        y_coordinates.append((-l / 2, -l / 2 + t))
        if number_of_selected_elements == 2:
            x_coordinates = []
            y_coordinates = []
            if selection_list[1] == 'Plate':
                random_generator = random.randint(0, 4)
                if random_generator == 0:
                    x_coordinates.append((-l / 2, l / 2))
                    x_coordinates.append((-l / 2, l / 2))
                    x_coordinates.append((-t / 2, t / 2))
                    x_coordinates.append((-l / 2, l / 2))
                    x_coordinates.append((-l / 2, l / 2))
                    y_coordinates.append((l / 2 + 2 * t, l / 2 + t))
                    y_coordinates.append((l / 2 + t, l / 2))
                    y_coordinates.append((-l / 2, l / 2))
                    y_coordinates.append((-l / 2 - t, -l / 2))
                    y_coordinates.append((-l / 2 - 2 * t, -l / 2 - t))
                elif random_generator == 1:
                    x_coordinates.append((-l / 4, l / 4))
                    x_coordinates.append((-l / 2, l / 2))
                    x_coordinates.append((-t / 2, t / 2))
                    x_coordinates.append((-l / 2, l / 2))
                    x_coordinates.append((-l / 4, l / 4))
                    y_coordinates.append((l / 2 + 2 * t, l / 2 + t))
                    y_coordinates.append((l / 2 + t, l / 2))
                    y_coordinates.append((-l / 2, l / 2))
                    y_coordinates.append((-l / 2 - t, -l / 2))
                    y_coordinates.append((-l / 2 - 2 * t, -l / 2 - t))
                elif random_generator == 2:
                    x_coordinates.append((-5 * l / 2, 5 * l / 2))
                    x_coordinates.append((-3 * l / 2, -l / 2))
                    x_coordinates.append((l / 2, 3 * l / 2))
                    x_coordinates.append((-l - t / 2, -l + t / 2))
                    x_coordinates.append((l - t / 2, l + t / 2))
                    x_coordinates.append((-3 * l / 2, -l / 2))
                    x_coordinates.append((l / 2, 3 * l / 2))
                    x_coordinates.append((-5 * l / 2, 5 * l / 2))
                    y_coordinates.append((l / 2 + 2 * t, l / 2 + t))
                    y_coordinates.append((l / 2 + t, l / 2))
                    y_coordinates.append((l / 2 + t, l / 2))
                    y_coordinates.append((-l / 2, l / 2))
                    y_coordinates.append((-l / 2, l / 2))
                    y_coordinates.append((-l / 2, -l / 2 - t))
                    y_coordinates.append((-l / 2, -l / 2 - t))
                    y_coordinates.append((-l / 2 - t, -l / 2 - 2 * t))
                elif random_generator == 3:
                    x_coordinates.append((-3 * l / 2, 3 * l / 2))
                    x_coordinates.append((-l / 2, l / 2))
                    x_coordinates.append((-t / 2, t / 2))
                    x_coordinates.append((-l / 2, l / 2))
                    x_coordinates.append((-3 * l / 2, 3 * l / 2))
                    y_coordinates.append((l / 2 + 2 * t, l / 2 + t))
                    y_coordinates.append((l / 2 + t, l / 2))
                    y_coordinates.append((-l / 2, l / 2))
                    y_coordinates.append((-l / 2, -l / 2 - t))
                    y_coordinates.append((-l / 2 - 2 * t, -l / 2 - t))
                elif random_generator == 4:
                    x_coordinates.append((-l / 2 - t, -l / 2))
                    x_coordinates.append((-l / 2, l / 2))
                    x_coordinates.append((l / 2, l / 2 + t))
                    x_coordinates.append((-t / 2, t / 2))
                    x_coordinates.append((-l / 2, l / 2))
                    y_coordinates.append((-l / 2 - t, l / 2 + t))
                    y_coordinates.append((l / 2, l / 2 + t))
                    y_coordinates.append((-l / 2 - t, l / 2 + t))
                    y_coordinates.append((-l / 2, l / 2))
                    y_coordinates.append((-l / 2 - t, -l / 2))
            else:
                x_coordinates.append((-l / 2, l / 2))
                x_coordinates.append((-t / 2, t / 2))
                x_coordinates.append((-l / 2, l / 2))
                y_coordinates.append((l / 2, l / 2 - t))
                y_coordinates.append((-l / 2 + t, l / 2 - t))
                y_coordinates.append((-l / 2, -l / 2 + t))
    elif element_of_origin == 'UPN':
        x_coordinates.append((-l / 2, -l / 2 + t))
        x_coordinates.append((l / 2 - t, l / 2))
        x_coordinates.append((-l / 2 + t, l / 2 - t))
        y_coordinates.append((-l / 4, l / 4))
        y_coordinates.append((-l / 4, l / 4))
        y_coordinates.append((-l / 4, - l / 4 + t))
    elif element_of_origin == 'Legged':
        x_coordinates.append((-l / 2, -l / 2 + t))
        x_coordinates.append((-l / 2 + t, l / 2))
        y_coordinates.append((-l / 2, l / 2))
        y_coordinates.append((-l / 2, - l / 2 + t))
    if is_horizontal:
        x_coordinates, y_coordinates = y_coordinates, x_coordinates

    return x_coordinates, y_coordinates, section_type


def generate_from_tabular(k):
    """
    Selects shape from tabular shape list
    :param
    k : number of selection
    :return:
    selected_elements : selected tabular shapes
    """
    selection = random.sample(tabular_shape_list, k)
    return selection


for i in range(number_of_images):
    image_height = random.randint(min_image_height, max_image_height)
    if a4_size:
        image_width = int(image_height / (2**0.5))
    image = generate_image(image_width, image_height)
    chosen_shape = choose_shape()

    if chosen_shape == ['tabular_x_section']:
        number_of_x_section = 1
    elif chosen_shape == ['built_up_consisting_2_shapes']:
        number_of_x_section = 2
    elif chosen_shape == ['built_up_consisting_3_shapes']:
        number_of_x_section = 3
    else:
        number_of_x_section = 4

    selected_elements = generate_from_tabular(number_of_x_section)
    unscaled_x_coordinates, unscaled_y_coordinates, section_type = create_geometry(selected_elements)
    available_x_pixel_space = (int(blank_corner_ratio_to_image_area * image_width),
                               int((1 - blank_corner_ratio_to_image_area) * image_width))
    available_y_pixel_space = (int(blank_corner_ratio_to_image_area * image_height),
                               int((1 - blank_corner_ratio_to_image_area) * image_height))
    max_x_length_to_scale = np.max(unscaled_x_coordinates)
    max_y_length_to_scale = np.max(unscaled_y_coordinates)
    x_center = int((available_x_pixel_space[0] + available_x_pixel_space[1]) / 2)
    y_center = int((available_y_pixel_space[0] + available_y_pixel_space[1]) / 2)
    ratio_of_max_scale_x = (available_x_pixel_space[1] - x_center) / max_x_length_to_scale
    ratio_of_max_scale_y = (available_y_pixel_space[1] - y_center) / max_y_length_to_scale
    applicable_x_scale_factor = int(ratio_of_max_scale_x * random.uniform(min_x_section_ratio_to_image_area,
                                                                          max_x_section_ratio_to_image_area))
    applicable_y_scale_factor = int(ratio_of_max_scale_y * random.uniform(min_x_section_ratio_to_image_area,
                                                                          max_x_section_ratio_to_image_area))
    applicable_scale_factor = min(applicable_x_scale_factor, applicable_y_scale_factor)

    scaled_x_coordinates = np.multiply(unscaled_x_coordinates, applicable_scale_factor)
    scaled_x_coordinates = [x + x_center for x in scaled_x_coordinates]
    scaled_y_coordinates = np.multiply(unscaled_y_coordinates, applicable_scale_factor)
    scaled_y_coordinates = [y + y_center for y in scaled_y_coordinates]
    nominal_corner_coordinates = np.hstack((scaled_x_coordinates, scaled_y_coordinates))
    line_thickness = random.randint(1, 2)
    nominal_line_coordinates = []
    if len(nominal_corner_coordinates) > 0:
        for j in range(len(nominal_corner_coordinates)):
            for t in range(2):
                for k in range(2):
                    nominal_line_coordinates.append((nominal_corner_coordinates[j][t],
                                                     nominal_corner_coordinates[j][k+2]))

    img = ImageDraw.Draw(image)
    if len(nominal_corner_coordinates) > 0:
        for p in range(len(nominal_corner_coordinates)):
            img.line((nominal_line_coordinates[4 * p + 0], nominal_line_coordinates[4 * p + 2]), fill='black', width=2)
            img.line((nominal_line_coordinates[4 * p + 1], nominal_line_coordinates[4 * p + 3]), fill='black', width=2)
            img.line((nominal_line_coordinates[4 * p + 2], nominal_line_coordinates[4 * p + 3]), fill='black', width=2)
            img.line((nominal_line_coordinates[4 * p + 0], nominal_line_coordinates[4 * p + 1]), fill='black', width=2)
        image_data = np.asarray(image)
        filename = '{}.jpg'.format(i)
        cv.imwrite(os.path.join(directory_to_save, filename), image_data)
