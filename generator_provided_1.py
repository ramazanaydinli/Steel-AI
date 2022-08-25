"""

Created by Ramazan AYDINLI for educational purposes
Aim is generating artificial x-section images that may be used for
training neural network models

"""


import numpy as np
from PIL import Image, ImageDraw
import random
import cv2 as cv

number_of_images = 500

min_image_height = 1200
max_image_height = 1500

min_x_section_ratio_to_image_area = 0.2
max_x_section_ratio_to_image_area = 0.4
blank_corner_ratio_to_image_area = 0.05

tabular_x_section_probability = 0.25
built_up_consisting_2_shapes_probability = 0.25
built_up_consisting_3_shapes_probability = 0.25
built_up_consisting_4_shapes_probability = 0.25


tabular_shape_list = ['Plate', 'IPN', 'UPN', 'Legged']


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
    :return: nominal_corner_coordinates: coordinates of unscaled section
    """
    number_of_selected_elements = len(selection_list)
    element_of_origin = selection_list[0]
    nominal_central_element_thickness = 2
    nominal_central_element_length = 20
    central_coordinate = (0, 0)
    if element_of_origin == 'Plate':
        is_horizontal = bool(random.getrandbits(1))
        if is_horizontal:
            x1 = (-nominal_central_element_length/2, nominal_central_element_length/2)
            y1 = (-nominal_central_element_thickness/2, nominal_central_element_thickness/2)
        else:
            x1 = (-nominal_central_element_thickness/2, nominal_central_element_thickness/2)
            y1 = (-nominal_central_element_length/2, nominal_central_element_length/2)
        if number_of_selected_elements > 1:
            for j in range(number_of_selected_elements):
                print(j)
    elif element_of_origin == 'IPN':
        is_horizontal = bool(random.getrandbits(1))
        None
    elif element_of_origin == 'UPN':
        is_horizontal = bool(random.getrandbits(1))
        None
    elif element_of_origin == 'Legged':
        is_horizontal = bool(random.getrandbits(1))
        None
    return None


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
    image_width = int(image_height / (2**0.5))
    image = generate_image(image_width, image_height)
    chosen_shape = choose_shape()

    if chosen_shape == ['tabular_x_section']:
        number_of_x_section = 1
    elif chosen_shape == ['built_up_consisting_2_shapes']:
        number_of_x_section = 2
    elif chosen_shape == ['built_up_consisting_3_shapes']:
        number_of_x_section = 3
    elif chosen_shape == ['built_up_consisting_4_shapes']:
        number_of_x_section = 4

    selected_elements = generate_from_tabular(number_of_x_section)
    nominal_corner_coordinates = create_geometry(selected_elements)
