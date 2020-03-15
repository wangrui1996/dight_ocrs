import cv2
import numpy

def wrapper_image(image):
    image = image.astype(numpy.float32) / 255.0
    return image