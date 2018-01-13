# Sikandar 13 - 01 - 2018 Saturday
# I have a EHD.txt file in C:\Users\musi0010\Desktop\jobs\zedge_task\mirflickr\meta\features
# In this program I will input querry image and then find 10 most similar images.
# the distance measures will be
# 1) Man manhattan (L1) distance
# 2) Euc Euclidean (L2) distance
# 3) Cos Cos distance
# 4) Sed Structure Entropic Distance
# 5) Ham Hamming distance over bit map

#libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from EdgeHistogramComputer import EdgeHistogramComputer
from Manhattandistance import Manhattandistance
from quantization import quantization
from displayclosestimages import displayclosestimages
# query image
qImg = "im100.jpg";

# initial setup
computer = EdgeHistogramComputer(4,4)
numofImg = 25000
imageDirectory = "C:/Users/musi0010/Desktop/jobs/zedge_task/mirflickr/"
text_file_EHD = open("C:/Users/musi0010/Desktop/jobs/zedge_task/mirflickr/meta/features/EHD.txt", "r")


################### Start: Find features of query image ####################################################
qImgname = imageDirectory + qImg
img = cv2.imread(qImgname)

# calculating 4 x 4 x 5 EHD descriptor using EdgeHistogramComputer.py
descriptor = computer.compute(img)  # so descriptor is a vector of 4 x 4 x 5 = 80 dimensions
descriptor_reshaped = descriptor.reshape(16, 5)

# calculate vertical descriptor group
vert = np.zeros([4, 5])
for i in range(0, 4):
    vert[i, :] = sum(descriptor_reshaped[range(i, 16, 4), :]) / 4

# calculate horizontal descriptor group
hori = np.zeros([4, 5])
for i in range(0, 4):
    hori[i, :] = sum(descriptor_reshaped[i * 4 + 0:i * 4 + 4, :]) / 4

# calculate neighboring descriptor group
neighbor1 = (descriptor_reshaped[0, :] + descriptor_reshaped[1, :] + descriptor_reshaped[4, :] + descriptor_reshaped[5,:]) / 4  # 1 2 5 6 (in python 0 1 4 5)
neighbor2 = (descriptor_reshaped[2, :] + descriptor_reshaped[3, :] + descriptor_reshaped[6, :] + descriptor_reshaped[7,:]) / 4  # 3 4 7 8 (in python 2 3 6 7)
neighbor3 = (descriptor_reshaped[8, :] + descriptor_reshaped[9, :] + descriptor_reshaped[12,:] + descriptor_reshaped[13,:]) / 4  # 9 10 13 14 (in python 8 9 12 13)
neighbor4 = (descriptor_reshaped[10, :] + descriptor_reshaped[11, :]+ descriptor_reshaped[14,:]+ descriptor_reshaped[15,:]) / 4  # 11 12 15 16 (in python 10 11 14 15)
neighbor5 = (descriptor_reshaped[5, :] + descriptor_reshaped[6, :] + descriptor_reshaped[9, :] + descriptor_reshaped[10,:]) / 4  # 6 7 10 11 (in python 5 6 9 10)

# calculate global descriptor
global_des = sum(descriptor_reshaped[:, :]) / 16

# group all the descriptors 16 local, 4 vertical, 4 horizontal, 5 neighbor, 1 global = 30 x 5 ( gradients) = 150 features per image
EHD_vector1 = np.append(descriptor_reshaped, vert)
EHD_vector2 = np.append(hori, neighbor1);
EHD_vector3 = np.append(neighbor2, neighbor3)
EHD_vector4 = np.append(neighbor4, neighbor5)
EHD_vector5 = np.append(global_des, [])
EHD_vector = np.concatenate((EHD_vector1, EHD_vector2, EHD_vector3, EHD_vector4,EHD_vector5))  # ,neighbor1,neighbor2,neighbor3,neighbor4,global_des
EHD_vector_list = list(EHD_vector)
EHD_test_vector = EHD_vector_list
EHD_test_vector_quant = quantization(EHD_test_vector, 6)

################### End: Find features of query image ####################################################
print EHD_vector_list

#################################### Start: Manhatan distance##############################
Man =[]

for line in text_file_EHD:
     currentline = line.split(",")
     EHD_train_vector = map(float,currentline)
     EHD_train_vector = EHD_train_vector[1:]
     dist = Manhattandistance(EHD_test_vector_quant, EHD_train_vector)
     Man.append(dist)


print EHD_train_vector
print Man
sorted_Man = sorted(Man)
print sorted_Man
IndecesofSortedMan = np.argsort(Man)#[i[0] for i in sorted(enumerate(Man), key=lambda x:x[1])]

#################################### End: Manhatan distance##############################


#################### Start: Display 10 most similar images#########################
numimg = 10
IndecesofSortedMan = [x+1 for x in IndecesofSortedMan] # increment the indices by one
print IndecesofSortedMan
displayclosestimages(qImg,imageDirectory,sorted_Man,IndecesofSortedMan,numimg)
#################### End: Display 10 most similar images#########################

##################### Start: ANMARR (Average Normalized Modified Reterival Rank)######################

##################### End: ANMARR (Average Normalized Modified Reterival Rank)######################

text_file_EHD.close()
