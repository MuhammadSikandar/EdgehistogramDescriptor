import cv2
import numpy as np
import matplotlib.pyplot as plt

def displayclosestimages(qImg,imageDirectory,sorted_Man,IndecesofSortedMan,numimg):
    testimgread = imageDirectory + qImg
    imgtest = cv2.imread(testimgread)

    cv2.imshow('image', imgtest)
    plt.figure(1)
    for ii in range(1,numimg+1):
        similarimgindex = "im%u.jpg" % IndecesofSortedMan[ii]
        similarimgpath = imageDirectory + similarimgindex
        similarimg = cv2.imread(similarimgpath)
        plt.subplot(2,5,ii)
        plt.imshow(similarimg)
    plt.show()
    cv2.waitKey(0)
