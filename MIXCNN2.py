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
from sklearn.preprocessing import  MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.layers import  Dense, Conv1D, Activation
from tensorflow.keras.layers import Input, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn import metrics
from tensorflow.keras.models import Model
#from tensorflow.keras.layers import concatenate
#from tensorflow.keras.layers import LeakyReLU,add,Reshape
from  tensorflow.keras.layers import GlobalAveragePooling1D,add
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
        mat1 = loadmat('D:\\论文mix-cnn\\模型\\西储大学轴承故障\\数据准\\测试集\\10类测_240.204_8db_dataset1024.mat')
        
        
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

X_va, X_test, y_va, y_test1= train_test_split(X_test, y_test, test_size=0.5)

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
#sdg=tf.keras.optimizers.SGD(lr=0.001)
ADAM=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.99, beta_2=0.999, epsilon=1e-8)
MIXCNN.compile(optimizer=ADAM,
            loss='mean_squared_error',
            metrics=['accuracy'])  
#print(MIXCNN.summary()) 'mean_squared_error' 



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

#MIXCNN.save('D:\\论文mix-cnn\\模型\\西储大学轴承故障\\模型\\模型准\\c=128_k=64_8dbmixcnn2.h5')


oos_test_y = []
oos_test_prob = []
aggregated_score = 0
aggregated_test_score = 0
runtime = 0
oos_test_activations = []
oos_y = []
oos_pred = []
oos_test_pred = []
oos_test_y = []
oos_test_prob = []

# Predictions on the validation set
predictions = MIXCNN.predict([X_va])

# Raw probabilities to chosen class (highest probability)
predictions = np.argmax(predictions,axis=1)
# Append predictions of the validation set to empty list
oos_pred.append(predictions)  

# Measure this fold's accuracy on validation set compared to actual labels
y_compare = np.argmax(y_va,axis=1) 
score = metrics.accuracy_score(y_compare, predictions)
print(f"Validation fold score(accuracy): {score}")






'''
flops
'''

import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
print('TensorFlow:', tf.__version__)


forward_pass = tf.function(
    MIXCNN.call,
    input_signature=[tf.TensorSpec(shape=(1,) + MIXCNN.input_shape[1:])])

graph_info = profile(forward_pass.get_concrete_function().graph,
                        options=ProfileOptionBuilder.float_operation())

flops = graph_info.total_float_ops // 2
print('Flops: {:,}'.format(flops))




##########
 #######
 
 ##########
 #ROC曲线------SEU dataset
 ########
from sklearn.metrics import roc_curve, auc
from scipy import interp
fpr = dict()
tpr = dict()
roc_auc = dict()
#y_t = onehot(predictions)  
classes=[0,1,2,3,4,5,6,7,8]
for i in range(len(classes)):
    fpr[i], tpr[i], thresholds = roc_curve(y_test1[:, i],predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], thresholds = roc_curve(y_test1.ravel(),predictions.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# macro-average ROC curve 方法二）

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(len(classes)):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# 求平均计算ROC包围的面积AUC
mean_tpr /= len(classes)
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

from itertools import cycle
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(9), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))



plt.figure()
plt.plot(fpr["micro"], tpr["micro"],'k-',color='y',
         label='XXXX ROC curve micro-average(AUC = {0:0.4f})'
               ''.format(roc_auc["micro"]),
          linestyle='-.', linewidth=3)

plt.plot(fpr["macro"], tpr["macro"],'k-',color='k',
         label='XXXX ROC curve macro-average(AUC = {0:0.4f})'
               ''.format(roc_auc["macro"]),
          linestyle='-.', linewidth=3)
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc="lower right")
plt.grid(linestyle='-.')  
plt.grid(True)
plt.show()

















