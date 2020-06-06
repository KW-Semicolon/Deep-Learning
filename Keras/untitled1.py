# basic model with IPD9(5*6) + 3 feature
import numpy as np
import pandas as pd
import math
import keras
import random
import statistics

from keras import layers
from keras import models
from keras.models import load_model

# set variation
        
sub_num = 40 # amount of subject
data_amt = 1000 # data in 1 phase
data_phase = 15 # state during 1 minute
num_feature = 9 # amount of feature
result_info = 2 # number of output(SBP, DBP)

# call data
dataset_output = pd.read_csv('data_output.csv')
dataset_output2 = pd.read_csv('output_2.csv')
dataset_output3 = pd.read_csv('sub_data.csv')
dataset_output_feat = pd.read_csv('data_info2.csv')

dataset_ipd9_amt40 = pd.read_csv('ipd9_amt40.csv')
dataset_ipd = pd.read_csv('data_ipd.csv')
dataset_ipd9 = pd.read_csv('ipd_data_amt9.csv')
dataset_ip_bcg1 = pd.read_csv('ip_data_bcg1.csv')
dataset_ip_bcg2 = pd.read_csv('ip_data_bcg2.csv')
dataset_ipd_5x6 = pd.read_csv('ipd_data_200430.csv')

dataset_norm_bcg1 = pd.read_csv('norm_bcg1.csv')
dataset_norm_bcg2 = pd.read_csv('norm_bcg2.csv')
dataset_bcg1 = pd.read_csv('data_BCG1.csv')
dataset_bcg2 = pd.read_csv('data_BCG2.csv')

output = dataset_output.values
output2 = dataset_output2.values
output3 = dataset_output3.values
feat = dataset_output_feat.values

ipd9_amt40 = dataset_ipd9_amt40.values
ipd = dataset_ipd.values
ipd9 = dataset_ipd9.values
ip1 = dataset_ip_bcg1.values
ip2 = dataset_ip_bcg2.values
ipd_5x6 = dataset_ipd_5x6.values

norm_bcg1 = dataset_norm_bcg1.values
norm_bcg2 = dataset_norm_bcg2.values
bcg1 = dataset_bcg1.values
bcg2 = dataset_bcg2.values

# make new array

all_data = np.zeros((sub_num * data_phase , num_feature))
all_result = np.zeros((sub_num * data_phase, result_info))
idx = 0

for i in range(sub_num): # number of subject
    for j in range(data_phase): 
        for k in range(num_feature):
            all_data[data_phase*i + j][k] = statistics.median(ipd9[idx])
            idx+=1
    for j in range(data_phase): # all phase
        all_result[i*data_phase + j] = output[j][2*i:2*i + 2]
'''
idx = 0    
for i in range(5):
    for j in range(90):
        all_data[i*90 + j][num_feature:] = feat[i]
'''

num_epochs = 30 # time of repeat

k = 60 # fold data
num_val_sample = len(all_data) // k

def build_model(): # build model func for k-fold validation
    model = models.Sequential()
    model.add(layers.Dense(25, activation='relu', input_shape=(9, )))
    model.add(layers.Dense(2))
    
    # model.add(layers.Dropout(0.5))
    
    model.compile(optimizer='Adam', loss='mse', metrics=['mae'])
    return model

def ME(a, b):
    me = 0
    for i in range(len(a)):
        for j in range(len(a[0])):
            me += abs(a[i][j] - b[i][j])
            
    return me/(len(a)*len(a[0]))

def STD(a, b):
    std = 0
    for i in range(len(a)):
        for j in range(len(a[0])):
            std += (a[i][j] - b[i][j]) * (a[i][j] - b[i][j])
    
    return math.sqrt(std/(len(a)*len(a[0])))

all_mae_history = []
ME_data = []
STD_data = []

for i in range(k):
    print('processing fold #', i + 1)
    # for validation
    val_data = all_data[i * num_val_sample : (i + 1) * num_val_sample]
    val_target = all_result[i * num_val_sample : (i + 1) * num_val_sample]
    
    # for training
    partial_train_data = np.concatenate(
        [all_data[:i * num_val_sample],
         all_data[(i + 1) * num_val_sample:]], axis=0)
    
    partial_train_target = np.concatenate(
        [all_result[:i * num_val_sample], 
         all_result[(i + 1) * num_val_sample:]], axis=0)
    
    model = build_model()
    
    # train model
    history = model.fit(partial_train_data, partial_train_target,
                         epochs=num_epochs, batch_size=1, verbose=2)
    
    ME_data.append(ME(val_target, model.predict(val_data)))
    STD_data.append(STD(val_target, model.predict(val_data)))
    
    mae_history = history.history['mae']
    all_mae_history.append(mae_history)
    

model.summary()
average_mae_history = [np.mean([x[i] for x in all_mae_history]) for i in range(num_epochs)]

# print data in flot     
import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)

plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

print(sum(ME_data) / len(ME_data))
print(sum(STD_data) / len(STD_data))
model.save('model01.h5')