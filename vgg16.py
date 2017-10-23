from __future__ import print_function, division

import os, json, math, cv2
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras.backend as K
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image
from keras.utils.data_utils import get_file


vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1,1,3))
def vgg_preprocess(x):
	x = x - vgg_mean
	return x[:, ::-1] # reverse axis rgb->bgr


class Vgg16():

	def __init__(self):
		self.FILE_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/'
		self.create()
		self.get_classes()


	def get_classes(self):
		fname = 'imagenet_class_index.json'
		fpath = get_file(fname , 'https://s3.amazonaws.com/deep-learning-models/image-models/'+fname, cache_subdir='models')

		with open(fpath) as f:
			class_dict = json.load(f)
		self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]


	def ConvBlock(self, layers, filters):
		model = self.model
		
		# Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
		for i in range(layers):
			model.add(Conv2D(filters, (3, 3), padding='same', activation='relu'))
		model.add(MaxPooling2D((2,2)))


	def FCBlock(self):
		model = self.model
		model.add(Dense(4096, activation='relu'))
		model.add(Dropout(0.5))


	def create(self):
		model = self.model = Sequential()
		model.add(Lambda(vgg_preprocess, input_shape=(224, 224, 3)))

		self.ConvBlock(2, 64)
		self.ConvBlock(2, 128)
		self.ConvBlock(3, 256)
		self.ConvBlock(3, 512)
		self.ConvBlock(3, 512)

		model.add(Flatten())
		self.FCBlock()
		self.FCBlock()
		model.add(Dense(1000, activation='softmax'))

		fname = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
		model.load_weights(get_file(fname, self.FILE_PATH+fname, cache_subdir='models'))


	def compile(self, lr=0.001):
		self.model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

	
	def finetune(self, batches):
		model = self.model
		model.pop()

		for layer in model.layers: layer.trainable = False
		model.add(Dense(batches.num_class, activation='softmax'))

		self.compile()

	
	def fit(self, batches, val_batches, epochs=1):
		self.model.fit_generator(batches, steps_per_epoch=batches.samples, epochs=epochs, validation_data=val_batches, validation_steps=val_batches.samples)

	
	def get_batches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical'):
		return gen.flow_from_directory(path, target_size=(224,224), class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)













