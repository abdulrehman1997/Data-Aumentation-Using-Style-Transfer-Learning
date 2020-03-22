import numpy as np
from PIL import Image
import requests
import os
# from VGGFACE import VGGFACE

from keras import backend
from keras.models import Model, load_model
from keras.applications.vgg16 import VGG16
from scipy.optimize import fmin_l_bfgs_b
import time
import tensorflow as tf


CHANNELS = 3
IMAGE_SIZE = 224
IMAGE_WIDTH = IMAGE_SIZE
IMAGE_HEIGHT = IMAGE_SIZE
IMAGENET_MEAN_RGB_VALUES = [123.68, 116.779, 103.939]
TOTAL_VARIATION_LOSS_FACTOR = 1.25

def preprocess(image):
	image_array = np.asarray(image, dtype="float32")
	image_array = np.expand_dims(image_array, axis=0)
	image_array[:, :, :, 0] -= IMAGENET_MEAN_RGB_VALUES[2]
	image_array[:, :, :, 1] -= IMAGENET_MEAN_RGB_VALUES[1]
	image_array[:, :, :, 2] -= IMAGENET_MEAN_RGB_VALUES[0]
	image_array = image_array[:, :, :, ::-1]
	# print(image_array.shape)

	image = backend.variable(image_array)
	return(image)

def content_loss(content, combination):
	return backend.sum(backend.square(combination - content))

def gram_matrix(x):
	features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
	gram = backend.dot(features, backend.transpose(features))
	return gram

def compute_style_loss(style, combination):
	style = gram_matrix(style)
	combination = gram_matrix(combination)
	size = IMAGE_HEIGHT * IMAGE_WIDTH
	return backend.sum(backend.square(style - combination)) / (4. * (CHANNELS ** 2) * (size ** 2))

def total_variation_loss(x):
	a = backend.square(x[:, :IMAGE_HEIGHT-1, :IMAGE_WIDTH-1, :] - x[:, 1:, :IMAGE_WIDTH-1, :])
	b = backend.square(x[:, :IMAGE_HEIGHT-1, :IMAGE_WIDTH-1, :] - x[:, :IMAGE_HEIGHT-1, 1:, :])
	return backend.sum(backend.pow(a + b, TOTAL_VARIATION_LOSS_FACTOR))