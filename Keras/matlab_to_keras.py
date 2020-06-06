# write data in result file
import os.path    
import time
import numpy as np
import pandas as pd
import statistics
import csv

from keras.models import load_model

def dir_empty(dir_path):
    return not any([True for _ in os.scandir(dir_path)])

def clock(num):
    data = []
    
    data.append(num % 60)
    num = int(num / 60)
        
    data.append(num % 60)
    num = int(num / 60)
        
    data.append(num % 24)
    num  = int(num / 24)
        
    result = ""
    
    for i in range(3):
        if data[2 - i] < 10:
            result +="0"
        
        result +=str(data[2 - i])
        if i < 2:
            result+=":"
    
    print(result)
        
# variation

model_name = 'model01.h5'
head_file = 'C:\\Users\\user1617\\data'
file = [head_file + '\\data0', head_file + '\\data1', head_file + '\\data2']
file_result = head_file + '\\result'
sec = 0


while True:
    if dir_empty(file[2]) is False:
        print("Insert data. Wait please...")
        time.sleep(16)
        model = load_model('C:\\Users\\user1617\\data\\model\\' + model_name)
        insert_addr = file[2] + '\\ipd_data.csv'
        insert = pd.read_csv(insert_addr).values
        # print(np.shape(insert))
        
        inp = np.zeros((1, 9))
        for i in range(9):
            inp[0][i] = statistics.median(insert[i])
        
        res = model.predict(inp)

        os.remove(insert_addr)
        f = open(file_result + '\\bp.txt', 'w')
        f.write(str(int(res[0][0])) + ', ' + str(int(res[0][1])))
        f.close()
        print("Success!")
    
    time.sleep(1)
    clock(sec)
    #print(sec)
    sec += 1        