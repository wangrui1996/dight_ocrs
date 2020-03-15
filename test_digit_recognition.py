#-*- coding:utf-8 -*-
import os
import json
import threading
import numpy as np
from scipy import ndimage
from PIL import Image, ImageFont, ImageDraw

import tensorflow as tf
from tensorflow.python.keras import losses
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.layers import Input, Dense, Flatten
from tensorflow.python.keras.layers.core import Reshape, Masking, Lambda, Permute
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard

from imp import reload
from src.models import densenet
#import densenet
import numpy
alphabet = u'0123456789 '

np.random.seed(55)


from tensorflow.python import keras
import random

# Reverse translation of numerical classes back to characters
def labels_to_text(labels):
    ret = []
    print(labels)
    for c in labels:
        print(c)
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:-2, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
def get_model(img_h, nclass):
    input = Input(shape=(img_h, None, 1), name='the_input')
    y_pred = densenet.dense_cnn(input, nclass)

    basemodel = Model(inputs=input, outputs=y_pred)
    basemodel.summary()

    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])
    model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)
    # clipnorm seems to speeds up convergence
    #sgd = SGD(learning_rate=0.02,
    #          decay=1e-6,
    #          momentum=0.9,
    #          nesterov=True)
    #model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=['accuracy'])

    return basemodel, model

def decode(pred):
    char_list = []
    pred_text = pred.argmax(axis=2)[0]
    print(labels_to_text(pred_text))
    for i in range(len(pred_text)):
        char_list.append(labels_to_text(pred_text))
        if pred_text[i] !=  - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
            char_list.append(characters[pred_text[i]])
    return u''.join(char_list)

def predict(img, test_func):
    height, width, _ = img.shape
    scale = height * 1.0 / 32
    width = int(width / scale)
    img = cv2.resize(img, (width, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    from src.utils import wrapper_image
    input_data = np.ones((1, 32, width, 1), dtype=np.float)
    img = wrapper_image(img)
    input_data[0,:,:,0] = img

    '''
    img_array = np.array(img.convert('1'))
    boundary_array = np.concatenate((img_array[0, :], img_array[:, width - 1], img_array[31, :], img_array[:, 0]), axis=0)
    if np.median(boundary_array) == 0:  # 将黑底白字转换为白底黑字
        img = ImageOps.invert(img)
    '''

    out = test_func([input_data])[0]
    y_pred = out[:, 2:-2, :]

    out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0])[:, :]
    print("o:",out[0])
    out = labels_to_text(out[0])
    out = decode(y_pred)

    return out
import cv2
if __name__ == '__main__':
    img_w = 256
    img_h = 32
    minibatch_size = 32
    model_save_path = "./save_models"
    basemodel, model = get_model(img_h, len(alphabet))
    modelPath = './models/keras.h5'
    if os.path.exists(modelPath):
        print("Loading model weights...")
        basemodel.load_weights(modelPath)
        print('done!')
    test_func = K.function([basemodel.input], [basemodel.output])
    img = cv2.imread("test.jpg")
    text = predict(img, test_func)
    print(text)

