from pandas import read_csv
import numpy as np
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
import os

os.chdir('.//data//')


file = read_csv('ebird_BR-AM_jabiru_1900_2017_1_12_linegraphs.csv')

array = file.values
x = array[9,3:50]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(x)

print(rescaledX*10)





