# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 09:57:50 2022

Please refer to https://github.com/liguge/1D-Grad-CAM-for-interpretable-intelligent-fault-diagnosis

@article{ZHANG2022110242,  
title = {Fault diagnosis for small samples based on attention mechanism},  
journal = {Measurement},  
volume = {187},  
pages = {110242},  
year = {2022},  
issn = {0263-2241},  
doi = {https://doi.org/10.1016/j.measurement.2021.110242 },  
url = {https://www.sciencedirect.com/science/article/pii/S0263224121011507},  
author = {Xin Zhang and Chao He and Yanping Lu and Biao Chen and Le Zhu and Li Zhang}  
} 


"""

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers.core import Lambda
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))


def target_category_loss_output_shape(input_shape):
    return input_shape


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def resize_1d(array, shape):
    res = np.zeros(shape)
    if array.shape[0] >= shape:
        ratio = array.shape[0]/shape
        for i in range(array.shape[0]):
            res[int(i/ratio)] += array[i]*(1-(i/ratio-int(i/ratio)))
            if int(i/ratio) != shape-1:
                res[int(i/ratio)+1] += array[i]*(i/ratio-int(i/ratio))
            else:
                res[int(i/ratio)] += array[i]*(i/ratio-int(i/ratio))
        res = res[::-1]
        array = array[::-1]
        for i in range(array.shape[0]):
            res[int(i/ratio)] += array[i]*(1-(i/ratio-int(i/ratio)))
            if int(i/ratio) != shape-1:
                res[int(i/ratio)+1] += array[i]*(i/ratio-int(i/ratio))
            else:
                res[int(i/ratio)] += array[i]*(i/ratio-int(i/ratio))
        res = res[::-1]/(2*ratio)
        array = array[::-1]
    else:
        ratio = shape/array.shape[0]
        left = 0
        right = 1
        for i in range(shape):
            if left < int(i/ratio):
                left += 1
                right += 1
            if right > array.shape[0]-1:
                res[i] += array[left]
            else:
                res[i] += array[right] * \
                    (i - left * ratio)/ratio+array[left]*(right*ratio-i)/ratio
        res = res[::-1]
        array = array[::-1]
        left = 0
        right = 1
        for i in range(shape):
            if left < int(i/ratio):
                left += 1
                right += 1
            if right > array.shape[0]-1:
                res[i] += array[left]
            else:
                res[i] += array[right] * \
                    (i - left * ratio)/ratio+array[left]*(right*ratio-i)/ratio
        res = res[::-1]/2
        array = array[::-1]
    return res


def grad_cam(input_model, data, category_index, layer_name, nb_classes):
    def target_layer(x):
        return target_category_loss(x, category_index, nb_classes)
    x = input_model.layers[-1].output
    x = Lambda(target_layer, output_shape=target_category_loss_output_shape)(x)
    model = Model(input_model.layers[0].input, x)
    loss = K.sum(model.layers[-1].output)
    conv_output = [l for l in model.layers if l.name is layer_name][0].output


    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function(
        [model.layers[0].input], [conv_output, grads])
    output, grads_val = gradient_function([data])
    output, grads_val = output[0, :], grads_val[0, :, :]
    weights = np.mean(grads_val, axis=(0)) 
    # cam = np.ones(output.shape[0: 1], dtype=np.float32)
    # for i, w in enumerate(weights):
    #     cam += w * output[:, i]
    cam = output.dot(weights)
    print(cam.shape,'cam')
    # print(cam)
    cam = resize_1d(cam, (data.shape[1]))
    cam = np.maximum(cam, 0)
    # heatmap = cam / np.max(cam)
    print(cam.shape,'cam')
    heatmap = (cam - np.min(cam))/(np.max(cam) - np.min(cam)+1e-10)
    return heatmap


    
pred = MIXCNN.predict(x_test[0][np.newaxis, :, :])

# print(x_test.shape)

# print(pred, 'pred')

category_index = np.argmax(pred)

# print(category_index,'category_index')

for layer in MIXCNN.layers:
  if 'conv1d' in layer.name: 
    conv_name = layer.name

a1 = x_test[0][np.newaxis, :, :]
heatmap1 = grad_cam(wdcnn_multi1, a1, category_index, conv_name,10)

import scipy.io as scio
dataNew = "D:\\apcnn_datanew_8.mat"
d = np.arange(1,1024,1)
scio.savemat(dataNew, mdict={'cam': heatmap1, 'data': a1, 'suzu': d})


pred = MIXCNN.predict(x_test[2023][np.newaxis, :, :])

# print(x_test.shape)

# print(pred, 'pred')

category_index = np.argmax(pred)



for layer in MIXCNN.layers:
  if 'conv1d' in layer.name: 
    conv_name = layer.name

# print(category_index,'category_index')

a = x_test[2023][np.newaxis, :, :]
heatmap = grad_cam(MIXCNN, a, category_index, conv_name,10)

dataNew = "D:\\f2_datanew_4-1.mat"
d = np.arange(1,1024,1)
scio.savemat(dataNew, mdict={'cam2': heatmap, 'data': a, 'suzu': d})









