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
built_up_consisting_5_shapes_probability = 0.25

tabular_shape_list = ['Plate', 'IPN', 'IPE', 'HEA', 'HEB', 'HEM',
                      'UPN', 'HD', 'Equal_Leg', 'Unequal_Leg']

def generate_image(width, height):
    """ Generates image
    Args:
        width : width of image
        height : height of image
    Returns:
        generated_image : generated image
        """
    generated_image = Image.new('RGBA', (width, height), (255, 255, 255))
    return generated_image


def choose_shape():
    """ Randomly choosing shape
    Returns:
        choice : chosen_x_section
    """
    general_shape_list = ['tabular_x_section', 'built_up_consisting_2_shapes',
                          'built_up_consisting_3_shapes', 'built_up_consisting_5_shapes']
    choice = random.choices(general_shape_list, weights=(
        tabular_x_section_probability,
        built_up_consisting_2_shapes_probability,
        built_up_consisting_3_shapes_probability,
        built_up_consisting_5_shapes_probability
    ), k=1)
    return choice


def generate_tabular(shape_list):
    """ Generates tabular x_section
    Returns:
        line_coordinates : nominal line coordinates
    """
    choice = random.choice(shape_list)
    return choice


def generate_built_2(shape_list):
    """ Generates 2 shaped built-up x_section
    Returns:
        line_coordinates : nominal line coordinates
    """
    choices = random.sample(shape_list, k=2)
    print(choices)
    return choices


def generate_built_3(shape_list):
    """ Generates 3 shaped built-up x_section
    Returns:
        line_coordinates : nominal line coordinates
    """
    return None


def generate_built_5(shape_list):
    """ Generates 5 shaped built-up x_section
    Returns:
        line_coordinates : nominal line coordinates
    """
    return None


for i in range(number_of_images):
    image_height = random.randint(min_image_height, max_image_height)
    image_width = int(image_height / (2**0.5))
    image = generate_image(image_width, image_height)
    chosen_shape = choose_shape()

    if chosen_shape == ['tabular_x_section']:
        nominal_line_coordinates = generate_tabular(tabular_shape_list)
    elif chosen_shape == ['built_up_consisting_2_shapes']:
        nominal_line_coordinates = generate_built_2(tabular_shape_list)
    elif chosen_shape == ['built_up_consisting_3_shapes']:
        nominal_line_coordinates = generate_built_3(tabular_shape_list)
    elif chosen_shape == ['built_up_consisting_5_shapes']:
        nominal_line_coordinates = generate_built_5(tabular_shape_list)


