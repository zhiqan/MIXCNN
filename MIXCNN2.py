# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 08:24:14 2022

@author: Zhiqian
"""


import os
import pandas as pd
from scipy.io import loadmat,savemat
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.layers import  Dense, Flatten, Dropout, Conv1D, Activation, AveragePooling1D
from tensorflow.keras.layers import Input, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn import metrics
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import LeakyReLU,add,Reshape
from  tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import DepthwiseConv1D





def mixconv(x,channal=64,kersize=64,m=1,c=1,AM=True):
     depth_conv_1 =DepthwiseConv1D(kernel_size=kersize, dilation_rate=m,depth_multiplier=c,padding="same",use_bias=False)(x)
     #depth_conv_1=Conv1D(64,kersize ,padding='same')(x)
     act_2 = tf.nn.relu(depth_conv_1)
     bn_2 = BatchNormalization()(act_2)
     if AM==True:  #自注意力，但效果不好 
         #add_1 = ECA(bn_2,64,3)
         #add_1 = CHSP1(bn_2,64,4)
         #add_1=A_CAM(bn_2,128)
         add_1=CHSP3(bn_2,s_num=256,reduction_ratio=4)
         add_1 = add([add_1, bn_2])
     else:
         add_1 =add([x, bn_2])
     conv_1 = Conv1D(channal, kernel_size=1,strides=1, padding="same")(add_1)
     act_3 = tf.nn.relu(conv_1)
     bn_3 = BatchNormalization()(act_3) 
     return bn_3




class Data_read:
    def __init__(self, snr='None'):

        mat = loadmat('D:\\论文mix-cnn\\模型\\西储大学轴承故障\\数据准\\训练集\\10类训_400.204_8db_dataset1024.mat')
        mat1 = loadmat('D:\\论文mix-cnn\\模型\\西储大学轴承故障\\数据准\\测试集\\10类测_120.204_8db_dataset1024.mat')
        
        #mat = loadmat('D:\\论文mix-cnn\\模型\\西储大学轴承故障\\数据\\10类_-10db_dataset1024.mat')
        #mat = loadmat('D:\\国际会议论文\会议论文资料\\PYshenduxuexi\\Fault_Diagnosis_CNN-master\\Datasets\\data7\\None\\dataset1024.mat')
        self.X_train = mat['X_train']
        self.X_test = mat1['X_train']
        self.y_train = self.onehot(np.array(mat['y_train'][:,0],dtype=int))
        self.y_test = self.onehot(np.array(mat1['y_train'][:,0],dtype=int))
        scaler = MinMaxScaler()
        self.X_train_minmax = scaler.fit_transform(self.X_train.T).T
        self.X_test_minmax = scaler.fit_transform(self.X_test.T).T


    def onehot(self,labels):
        '''one-hot 编码'''
        n_sample = len(labels)
        n_class = max(labels) + 1
        onehot_labels = np.zeros((n_sample, n_class))
        onehot_labels[np.arange(n_sample), labels] = 1
        return onehot_labels

    #def add_noise(self,snr):


'MIXCNN训练前准备'
data = Data_read(snr='None')
'''
# 选择一组训练与测试集
X_train = data.X_train_minmax # 临时代替
y_train = data.y_train

# 各组测试集
X_test = data.X_test_minmax
y_test = data.y_test

X_test = np.vstack((data.X_test_minmax,data.X_train_minmax))
y_test = np.vstack((data.y_test,data.y_train))
'''

X_train = data.X_train_minmax
y_train1 = data.y_train
X_test = data.X_test_minmax # 临时代替
y_test = data.y_test

X_va, X_test, y_va, y_test1= train_test_split(X_test, y_test, test_size=0.85)

x_train1=X_train[:,:,np.newaxis]#转换成cnn的输入格式
x_va=X_va[:,:,np.newaxis]
x_test1=X_test[:,:,np.newaxis]


earlystop = EarlyStopping(monitor= 'val_loss', min_delta=0 , patience=160, verbose=0, mode='min')
seq_len1 = x_train1.shape[1]
sens1 = x_train1.shape[2]
input_shape1 = (seq_len1, sens1)
input_signal = Input(input_shape1)
x = Conv1D(128, kernel_size=32, strides=4, padding='same')(input_signal)
x = BatchNormalization()(x)
x = Activation('relu')(x)
for i in range(3):
    x1=mixconv(x,channal=128,kersize=64,m=1,c=1,AM=False)
    x=x1
#x7=mixconv(x7,channal=256,kersize=9,m=1,c=1,AM=False)
x7 = BatchNormalization()(x)
x7= GlobalAveragePooling1D()(x7)
#x7 = Dropout(0.5)(x7)
out = Dense(10,  activation='sigmoid')(x7)
MIXCNN = Model(inputs=[input_signal],outputs=out)

tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=156324)
sdg=tf.keras.optimizers.SGD(lr=0.001)
ADAM=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.99, beta_2=0.999, epsilon=1e-8)
MIXCNN.compile(optimizer=ADAM,
            loss='mean_squared_error',
            metrics=['accuracy'])  
#print(wdcnn_multi.summary()) 'mean_squared_error' 
#'binary_crossentropy'损失函数效果要比mean_squared_error；差0.005左右


history=MIXCNN.fit(x_train1, y_train1,validation_data = (x_test1,y_test1), epochs =100, batch_size =1600, verbose=1, 
         callbacks =[earlystop], shuffle = True)

dic={"Train_loss":history.history['loss'],
     "Test_loss":history.history['val_loss'],
     "Train_acc":history.history['accuracy'],
     'val_acc':history.history['val_accuracy']}
df=pd.DataFrame(dic)
c="D:\\论文mix-cnn\\模型\\西储大学轴承故障\\数据准\\模型训练曲线\\8db_MIXCNN2_C=128_K=64.csv"
df.to_csv(c)

#from tensorflow.keras.models import load_model

MIXCNN.save('D:\\论文mix-cnn\\模型\\西储大学轴承故障\\模型\\模型准\\c=128_k=64_8dbmixcnn2.h5')